"""
Notes Lane – agentic notes handling for Memorae.

This module implements the "notes lane": a single-prompt agentic approach
where the LLM receives relevant notes in TOON format and outputs both a
human reply and structured <kb_patch> blocks for KB mutations.

Coexists with existing intent routing — reminders, calendar, and media
dispatch remain untouched.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from services.ai import complete, generate_embedding, USER_TZ, USER_TZ_LABEL
from services.toon import encode_notes_toon, next_note_id
from db import queries as q

logger = logging.getLogger(__name__)


# ── Notes Lane System Prompt ─────────────────────────────────────────────────

NOTES_SYSTEM_PROMPT = """\
You are Memo, a sharp personal AI assistant operating inside Telegram.
You manage the user's knowledge base — their second brain.

Your current date-time (UTC): {utc_now}
User's local timezone: {user_timezone}
User's local date-time: {local_now}

════════════════════════════════════════
  THE NOTES KNOWLEDGE BASE
════════════════════════════════════════

The user's relevant notes are below, encoded in TOON format.
Read every entry carefully before doing anything.

<knowledge_base>
{toon_kb}
</knowledge_base>

Each note has this structure in JSON (you will never output TOON, only JSON patches):
{{
  "id": "n_001",
  "title": "Short 3–6 word title you generate",
  "content": "Full content the user wants saved",
  "tags": ["tag1", "tag2"],
  "context_clues": ["college", "teacher", "3rd floor"],
  "created_at": "ISO-8601 timestamp",
  "updated_at": "ISO-8601 timestamp",
  "status": "active" | "done",
  "source_message": "The user's exact original message that created this note"
}}

════════════════════════════════════════
  INTENT DETECTION RULES
════════════════════════════════════════

Before responding to ANY message, silently classify the user's intent
into exactly one of these operations: CREATE, UPDATE, DELETE, RECALL, CHAT.

Do this classification in your head. Never show it to the user.

── CREATE ─────────────────────────────

Trigger when the user is telling you something new they want saved.

Explicit signals:
  "remember", "save this", "note this", "don't let me forget",
  "keep this", "store this", "make a note", "write this down"

Implicit signals (these must also trigger CREATE silently):
  - Factual statements about locations, contacts, or procedures
  - Future-action statements
  - Instructions or steps

For ALL CREATE operations (both explicit and implicit), save immediately
without asking for confirmation. Output the kb_patch and reply:
  → "Saved ✅ [Short title you generated]"

── UPDATE ─────────────────────────────

Trigger when the user is modifying something that was JUST created or
recently discussed. This is a CONTINUATION of the previous turn.

Explicit signals:
  "no change it to", "actually make it", "edit that",
  "update the note", "fix it to say", "correct that",
  "I meant", "not X, Y", "change X to Y"

CRITICAL RULE — REFERENTIAL RESOLUTION:
When the user says "change it", "fix that", or "edit it" WITHOUT specifying
which note, resolve to the most recently created or discussed note.
Do not ask "which note?" unless there is genuine ambiguity.

CRITICAL RULE — UPDATE IDs:
When updating a note, you MUST use the EXISTING note's id from the
knowledge base above (e.g. n_001, n_003). Do NOT use the next_id
({next_id}) for updates — that is ONLY for creating brand new notes.

When you detect an UPDATE, apply it and reply:
  → "Updated ✅ [Short title]
     [Show the new content so user can verify]"

── DELETE ─────────────────────────────

Trigger when the user signals a note is no longer needed.

Explicit signals:
  "delete that note", "remove it", "forget about X", "clear that"

Implicit completion signals:
  "done", "finished", "all set", "it's handled", "sorted", "checked",
  "I already did X", "that's complete", "we're good on X", "nevermind"

CRITICAL RULE — MATCH BEFORE DELETE:
When an implicit completion signal arrives:
1. Check the knowledge base for any note that semantically matches.
2. If a match is found, ask for confirmation BEFORE deleting:
   → "Looks like this is done ✅
      🗑 [Title of matched note]
      '[Content preview]'
      Should I remove it from your notes? Reply YES to delete."
3. Only issue a DELETE patch after the user confirms.
4. If no match found, simply acknowledge and do not touch the KB.

CRITICAL: Casual acknowledgments like "great", "ok", "cool", "nice",
"thanks", "awesome", "perfect", "sounds good", "got it" are NOT
completion signals. They are just the user reacting to your message.
Do NOT offer to delete notes when the user says these words.
Only trigger delete when the user explicitly says something IS DONE
("I already did X", "finished with X", "that's handled").

NEVER delete anything without explicit user confirmation.

── RECALL ─────────────────────────────

Trigger when the user is asking a question that should be answered using
saved notes, even if they don't say "recall" or "search."

SEMANTIC MATCHING RULES:
  Layer 1 — EXACT: Does the query directly name a tag, title, or keyword?
  Layer 2 — CONTEXTUAL: Does the query share context_clues with any note?
  Layer 3 — INFERENTIAL: Is the user's implied goal served by any saved note?

When a match is found, answer using the note's content naturally.
If MULTIPLE notes are relevant, surface all of them.
If NO note is relevant, answer from general knowledge and do NOT fabricate.

── CHAT ───────────────────────────────

Everything that doesn't match the above. Respond naturally and helpfully.
Still passively scan for implicit CREATE signals even in casual messages.

════════════════════════════════════════
  KB PATCH OUTPUT FORMAT
════════════════════════════════════════

After your human-readable reply, if any KB change is needed, output it in
a <kb_patch> block. The patch is always pure JSON. Never output TOON in patches.

─── CREATE a new note ──────────────────
<kb_patch>
{{
  "op": "add",
  "section": "notes",
  "item": {{
    "id": "{next_id}",
    "title": "Short descriptive title",
    "content": "Full content verbatim or cleaned up slightly",
    "tags": ["inferred_tag1", "inferred_tag2"],
    "context_clues": ["key", "location", "entity", "words"],
    "created_at": "{utc_now}",
    "updated_at": "{utc_now}",
    "status": "active",
    "source_message": "User's exact original message"
  }}
}}
</kb_patch>

─── UPDATE an existing note ─────────────
<kb_patch>
{{
  "op": "update",
  "section": "notes",
  "id": "n_001",
  "changes": {{
    "content": "Corrected content",
    "updated_at": "{utc_now}"
  }}
}}
</kb_patch>

─── DELETE a note (only after YES confirmation) ───
<kb_patch>
{{
  "op": "delete",
  "section": "notes",
  "id": "n_001"
}}
</kb_patch>

─── Multiple operations in one turn ─────
<kb_patch>
{{
  "op": "multi",
  "patches": [
    {{ "op": "update", "section": "notes", "id": "n_001", "changes": {{ "status": "done", "updated_at": "{utc_now}" }} }},
    {{ "op": "delete", "section": "notes", "id": "n_002" }}
  ]
}}
</kb_patch>

════════════════════════════════════════
  TAGGING & CONTEXT CLUES
════════════════════════════════════════

Every time you create or update a note, you must generate:

TAGS — 2 to 5 short lowercase labels describing the category.
CONTEXT CLUES — A flat list of key nouns, locations, entities, and
  action-words extracted from the note content. Be generous.

════════════════════════════════════════
  STRICT RULES
════════════════════════════════════════

1. NEVER delete without user confirmation. Always ask first, delete after YES.
2. NEVER create a duplicate note if a very similar one already exists.
   Instead, ask: "I already have a note about [X]. Update it or save a new one?"
3. NEVER answer a RECALL query from general knowledge if a saved note is relevant.
4. NEVER output raw note IDs or JSON to the user. All replies are natural language.
5. ALWAYS resolve "it/that/this/the note" to the most recently discussed note.
6. If ambiguous between CREATE and RECALL, lean toward RECALL first.
7. When patching, only include fields that actually changed.
8. Status field: set to "done" before deleting if the note represents a completed task.
"""


# ── Patch extraction & application ───────────────────────────────────────────

_KB_PATCH_RE = re.compile(r"<kb_patch>\s*(.*?)\s*</kb_patch>", re.DOTALL)


def _extract_patches(text: str) -> list[dict]:
    """Extract all <kb_patch> JSON blocks from LLM output."""
    patches = []
    for match in _KB_PATCH_RE.finditer(text):
        try:
            patch = json.loads(match.group(1))
            patches.append(patch)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse kb_patch JSON: %s", exc)
    return patches


def _strip_patches(text: str) -> str:
    """Remove all <kb_patch> blocks from the user-visible reply."""
    return _KB_PATCH_RE.sub("", text).strip()


async def _apply_patch(
    db: AsyncSession,
    user_id: str,
    patch: dict,
    id_map: dict[str, str],
    notes: list[dict] | None = None,
) -> None:
    """Apply a single kb_patch operation to the database."""
    op = patch.get("op")

    if op == "multi":
        for sub_patch in patch.get("patches", []):
            await _apply_patch(db, user_id, sub_patch, id_map, notes)
        return

    if op == "add":
        item = patch.get("item", {})
        content = item.get("content", "")
        title = item.get("title", content[:60])
        tags = item.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            tags = []

        context_clues = item.get("context_clues", [])
        if isinstance(context_clues, str):
            context_clues = [context_clues]
        elif not isinstance(context_clues, list):
            context_clues = []
        source_message = item.get("source_message", "")
        status = item.get("status", "active")

        # Generate embedding for the new note
        embedding = await generate_embedding(f"{title}\n{content}")

        await q.create_kb_entry(
            db,
            user_id=user_id,
            title=title,
            content=content,
            tags=tags,
            context_clues=context_clues,
            embedding=embedding,
            source="telegram",
            status=status,
            source_message=source_message,
        )
        logger.info("Notes lane: created note '%s' for user %s", title, user_id)

    elif op == "update":
        short_id = patch.get("id", "")
        real_id = id_map.get(short_id)

        # Fallback: if ID not found, try to match by title from the notes context
        if not real_id and notes:
            changes = patch.get("changes", {})
            patch_title = changes.get("title", "")
            patch_content = changes.get("content", "")
            for note in notes:
                note_title = (note.get("title") or "").lower()
                # Match if the patch title or content overlaps with the note
                if patch_title and patch_title.lower() in note_title:
                    real_id = str(note["id"])
                    logger.info("Notes lane: resolved unknown id %s to %s via title match '%s'",
                                short_id, real_id, note_title)
                    break
                if patch_content and note_title in patch_content.lower():
                    real_id = str(note["id"])
                    logger.info("Notes lane: resolved unknown id %s to %s via content match",
                                short_id, real_id)
                    break
            # Last resort: if only 1 note in context and this is the only update, use it
            if not real_id and len(notes) == 1:
                real_id = str(notes[0]["id"])
                logger.info("Notes lane: resolved unknown id %s to sole note %s", short_id, real_id)

        if not real_id:
            logger.warning("Notes lane: unknown note id %s in update patch, skipping", short_id)
            return

        changes = patch.get("changes", {})
        # Filter to only DB-safe fields
        allowed = {"title", "content", "tags", "context_clues", "status", "source_message"}
        db_changes = {k: v for k, v in changes.items() if k in allowed}

        if not db_changes:
            return

        # If content or title changed, regenerate embedding
        if "content" in db_changes or "title" in db_changes:
            new_title = db_changes.get("title", "")
            new_content = db_changes.get("content", "")
            if new_title or new_content:
                embedding = await generate_embedding(f"{new_title}\n{new_content}")
                db_changes["embedding"] = str(embedding) if embedding else None

        await q.update_kb_entry(db, entry_id=real_id, user_id=user_id, **db_changes)
        logger.info("Notes lane: updated note %s (%s) for user %s", short_id, real_id, user_id)

    elif op == "delete":
        short_id = patch.get("id", "")
        real_id = id_map.get(short_id)
        if not real_id:
            logger.warning("Notes lane: unknown note id %s in delete patch", short_id)
            return

        await q.delete_kb_entry(db, entry_id=real_id, user_id=user_id)
        logger.info("Notes lane: deleted note %s for user %s", short_id, user_id)


# ── Main handler ─────────────────────────────────────────────────────────────

async def handle_notes(
    db: AsyncSession,
    user_id: str,
    user_text: str,
    history_msgs: list[dict],
) -> str:
    """Route a message through the notes lane.

    1. Embed the query & fetch top 10 relevant notes via pgvector
    2. Encode those notes into TOON for the system prompt
    3. Call the LLM with the full notes system prompt
    4. Parse & apply any <kb_patch> blocks
    5. Return the clean human reply
    """
    # ── Fetch relevant notes ──────────────────────────────────────────────
    embedding = await generate_embedding(user_text)
    relevant_notes = await q.search_kb(
        db, user_id=user_id, query=user_text, limit=10, embedding=embedding,
    )

    # ── Encode into TOON ──────────────────────────────────────────────────
    toon_kb, id_map = encode_notes_toon(relevant_notes)
    nid = next_note_id(id_map)

    # ── Build the system prompt ───────────────────────────────────────────
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(USER_TZ)

    system = NOTES_SYSTEM_PROMPT.format(
        utc_now=now_utc.isoformat(),
        user_timezone=USER_TZ_LABEL,
        local_now=now_local.strftime("%Y-%m-%d %H:%M %Z"),
        toon_kb=toon_kb,
        next_id=nid,
    )

    # ── Call the LLM ──────────────────────────────────────────────────────
    raw_response = await complete(messages=history_msgs, system=system)

    # ── Parse and apply patches ───────────────────────────────────────────
    patches = _extract_patches(raw_response)
    for patch in patches:
        try:
            async with db.begin_nested():
                await _apply_patch(db, user_id, patch, id_map, notes=relevant_notes)
        except Exception as exc:
            logger.error("Notes lane: failed to apply patch: %s", exc, exc_info=True)

    # ── Return clean reply ────────────────────────────────────────────────
    reply = _strip_patches(raw_response)
    return reply if reply else "Done ✅"
