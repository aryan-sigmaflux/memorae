"""
Low-level database query helpers.
All functions accept an AsyncSession and return plain dicts / lists of dicts.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(r: Any) -> dict:
    return dict(r._mapping) if r else {}


def _rows(rs: Any) -> list[dict]:
    return [dict(r._mapping) for r in rs]


# ── Users ─────────────────────────────────────────────────────────────────────

async def get_or_create_user(db: AsyncSession, telegram_id: str, display_name: str | None = None) -> dict:
    row = await db.execute(
        text("SELECT * FROM users WHERE telegram_id = :telegram_id"),
        {"telegram_id": telegram_id},
    )
    user = _row(row.fetchone())
    if user:
        return user

    row = await db.execute(
        text(
            "INSERT INTO users (telegram_id, display_name) VALUES (:telegram_id, :name) RETURNING *"
        ),
        {"telegram_id": telegram_id, "name": display_name},
    )
    return _row(row.fetchone())


async def update_user_google_tokens(db: AsyncSession, user_id: str, tokens: dict) -> None:
    import json
    await db.execute(
        text("UPDATE users SET google_tokens = :tokens WHERE id = :id"),
        {"tokens": json.dumps(tokens), "id": user_id},
    )


async def get_user_by_id(db: AsyncSession, user_id: str) -> dict | None:
    row = await db.execute(text("SELECT * FROM users WHERE id = :id"), {"id": user_id})
    return _row(row.fetchone()) or None


# ── Conversations ─────────────────────────────────────────────────────────────

async def get_or_create_conversation(db: AsyncSession, user_id: str) -> dict:
    """Return the most recent open conversation, or start a new one."""
    row = await db.execute(
        text(
            "SELECT * FROM conversations WHERE user_id = :uid "
            "ORDER BY last_message_at DESC LIMIT 1"
        ),
        {"uid": user_id},
    )
    conv = _row(row.fetchone())
    if conv:
        return conv

    row = await db.execute(
        text("INSERT INTO conversations (user_id) VALUES (:uid) RETURNING *"),
        {"uid": user_id},
    )
    return _row(row.fetchone())


async def touch_conversation(db: AsyncSession, conversation_id: str) -> None:
    await db.execute(
        text("UPDATE conversations SET last_message_at = NOW() WHERE id = :id"),
        {"id": conversation_id},
    )


# ── Messages ──────────────────────────────────────────────────────────────────

async def save_message(
    db: AsyncSession,
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    telegram_message_id: str | None = None,
    media_url: str | None = None,
    media_type: str | None = None,
) -> dict:
    row = await db.execute(
        text(
            "INSERT INTO messages "
            "(conversation_id, user_id, role, content, telegram_message_id, media_url, media_type) "
            "VALUES (:cid, :uid, :role, :content, :tid, :murl, :mtype) "
            "ON CONFLICT (telegram_message_id) DO NOTHING RETURNING *"
        ),
        {
            "cid": conversation_id,
            "uid": user_id,
            "role": role,
            "content": content,
            "tid": telegram_message_id,
            "murl": media_url,
            "mtype": media_type,
        },
    )
    return _row(row.fetchone())


async def get_recent_messages(db: AsyncSession, conversation_id: str, limit: int = 20) -> list[dict]:
    rows = await db.execute(
        text(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = :cid "
            "ORDER BY created_at DESC LIMIT :lim"
        ),
        {"cid": conversation_id, "lim": limit},
    )
    msgs = _rows(rows.fetchall())
    return list(reversed(msgs))  # chronological order


# ── Knowledge Base ────────────────────────────────────────────────────────────

async def create_kb_entry(
    db: AsyncSession,
    user_id: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    context_clues: list[str] | None = None,
    embedding: list[float] | None = None,
    source: str = "manual",
    status: str = "active",
    source_message: str | None = None,
    media_url: str | None = None,
    media_type: str | None = None,
    metadata: dict | None = None,
) -> dict:
    import json
    emb_str = str(embedding) if embedding else None

    row = await db.execute(
        text(
            "INSERT INTO kb_entries (user_id, title, content, tags, context_clues, metadata, embedding, source, status, source_message, media_url, media_type) "
            "VALUES (:uid, :title, :content, :tags, :ctx, CAST(:meta AS jsonb), CAST(:emb AS vector), :src, :status, :smsg, :url, :type) RETURNING *"
        ),
        {
            "uid": user_id,
            "title": title,
            "content": content,
            "tags": tags or [],
            "ctx": context_clues or [],
            "meta": json.dumps(metadata or {}),
            "emb": emb_str,
            "src": source,
            "status": status,
            "smsg": source_message,
            "url": media_url,
            "type": media_type,
        },
    )
    return _row(row.fetchone())


async def search_kb(
    db: AsyncSession,
    user_id: str,
    query: str,
    limit: int = 5,
    embedding: list[float] | None = None,
    category: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[dict]:
    """
    Search KB entries using semantic search via pgvector.
    Results are ranked by cosine distance (embedding <=> :emb).

    Optional metadata filters (Memorae v2 §3): `category` matches
    metadata->>'category'; `date_from`/`date_to` bound created_at.
    """
    # Build shared filter clause + params for both the vector and fallback paths.
    filters = ""
    params: dict[str, Any] = {"uid": user_id}
    if category:
        filters += " AND metadata->>'category' = :category"
        params["category"] = category
    if date_from:
        filters += " AND created_at >= :date_from"
        params["date_from"] = date_from
    if date_to:
        filters += " AND created_at <= :date_to"
        params["date_to"] = date_to

    if embedding:
        params["emb"] = str(embedding)
        rows = await db.execute(
            text(
                "SELECT *, 1 - (embedding <=> CAST(:emb AS vector)) AS similarity_score "
                "FROM kb_entries "
                "WHERE user_id = :uid AND embedding IS NOT NULL" + filters + " "
                "ORDER BY embedding <=> CAST(:emb AS vector) "
                "LIMIT 15"
            ),
            params,
        )
        results = _rows(rows.fetchall())
        if results:
            import re
            query_words = set(re.findall(r'\w+', query.lower()))
            query_numbers = set(re.findall(r'\d+', query))
            
            def score(r: dict) -> float:
                s = r.get("similarity_score", 0.0)
                title = (r.get("title") or "").lower()
                title_numbers = set(re.findall(r'\d+', title))
                
                # Big boost for matching numbers (crucial for differentiating like Sem 1 vs Sem 2)
                for num in query_numbers:
                    if num in title_numbers:
                        s += 0.3
                        
                # Small boost for overlapping words
                title_words = set(re.findall(r'\w+', title))
                overlap = len(query_words.intersection(title_words))
                s += 0.02 * overlap
                return s
                
            results.sort(key=score, reverse=True)
            return results[:limit]

    # Fallback: if no embeddings exist or semantic search yields nothing, return the most recent entries
    # to act as a "working memory" context for the AI, so it can deduce answers fluidly.
    rows = await db.execute(
        text(
            "SELECT * FROM kb_entries "
            "WHERE user_id = :uid" + filters + " "
            "ORDER BY updated_at DESC LIMIT :lim"
        ),
        {**params, "lim": limit},
    )
    results = _rows(rows.fetchall())

    return results


async def list_user_notes(
    db: AsyncSession, user_id: str, media_only: bool = False, limit: int = 20,
) -> list[dict]:
    """List a user's notes, most recent first. Optionally only those with media."""
    where = "user_id = :uid"
    if media_only:
        where += " AND media_url IS NOT NULL"
    rows = await db.execute(
        text(f"SELECT * FROM kb_entries WHERE {where} ORDER BY updated_at DESC LIMIT :lim"),
        {"uid": user_id, "lim": limit},
    )
    return _rows(rows.fetchall())


async def get_kb_entry(db: AsyncSession, entry_id: str, user_id: str) -> dict | None:
    row = await db.execute(
        text("SELECT * FROM kb_entries WHERE id = :id AND user_id = :uid"),
        {"id": entry_id, "uid": user_id},
    )
    return _row(row.fetchone()) or None


async def list_kb_entries(db: AsyncSession, user_id: str, tag: str | None = None) -> list[dict]:
    if tag:
        rows = await db.execute(
            text("SELECT * FROM kb_entries WHERE user_id = :uid AND :tag = ANY(tags) ORDER BY updated_at DESC"),
            {"uid": user_id, "tag": tag},
        )
    else:
        rows = await db.execute(
            text("SELECT * FROM kb_entries WHERE user_id = :uid ORDER BY updated_at DESC"),
            {"uid": user_id},
        )
    return _rows(rows.fetchall())


async def update_kb_entry(db: AsyncSession, entry_id: str, user_id: str, **fields) -> dict:
    clauses = []
    for k in fields:
        if k == "embedding" and fields[k] is not None:
            clauses.append(f"{k} = CAST(:{k} AS vector)")
        else:
            clauses.append(f"{k} = :{k}")
    set_clauses = ", ".join(clauses)
    row = await db.execute(
        text(f"UPDATE kb_entries SET {set_clauses} WHERE id = :id AND user_id = :uid RETURNING *"),
        {"id": entry_id, "uid": user_id, **fields},
    )
    return _row(row.fetchone())


async def delete_kb_entry(db: AsyncSession, entry_id: str, user_id: str) -> bool:
    result = await db.execute(
        text("DELETE FROM kb_entries WHERE id = :id AND user_id = :uid"),
        {"id": entry_id, "uid": user_id},
    )
    return result.rowcount > 0


# ── Reminders ────────────────────────────────────────────────────────────────

async def create_reminder(
    db: AsyncSession,
    user_id: str,
    title: str,
    remind_at: datetime,
    body: str | None = None,
    recurrence: str | None = None,
    google_event_id: str | None = None,
) -> dict:
    row = await db.execute(
        text(
            "INSERT INTO reminders (user_id, title, body, remind_at, recurrence, google_event_id) "
            "VALUES (:uid, :title, :body, :at, :rec, :gid) RETURNING *"
        ),
        {"uid": user_id, "title": title, "body": body, "at": remind_at, "rec": recurrence, "gid": google_event_id},
    )
    return _row(row.fetchone())


async def get_due_reminders(db: AsyncSession) -> list[dict]:
    rows = await db.execute(
        text("SELECT r.*, u.telegram_id FROM reminders r JOIN users u ON u.id = r.user_id "
             "WHERE r.remind_at <= NOW() AND r.sent = FALSE ORDER BY r.remind_at"),
    )
    return _rows(rows.fetchall())


async def mark_reminder_sent(db: AsyncSession, reminder_id: str) -> None:
    await db.execute(
        text("UPDATE reminders SET sent = TRUE WHERE id = :id"),
        {"id": reminder_id},
    )


async def update_reminder_at(db: AsyncSession, reminder_id: str, next_at: datetime) -> None:
    await db.execute(
        text("UPDATE reminders SET remind_at = :at WHERE id = :id"),
        {"at": next_at, "id": reminder_id},
    )


async def update_most_recent_reminder(db: AsyncSession, user_id: str, remind_at: datetime) -> dict | None:
    """Update the most recently created reminder for this user."""
    row = await db.execute(
        text(
            "UPDATE reminders SET remind_at = :at "
            "WHERE id = (SELECT id FROM reminders WHERE user_id = :uid ORDER BY created_at DESC LIMIT 1) "
            "RETURNING *"
        ),
        {"at": remind_at, "uid": user_id},
    )
    return _row(row.fetchone()) or None


async def list_reminders(db: AsyncSession, user_id: str) -> list[dict]:
    rows = await db.execute(
        text("SELECT * FROM reminders WHERE user_id = :uid AND sent = FALSE ORDER BY remind_at"),
        {"uid": user_id},
    )
    return _rows(rows.fetchall())


async def delete_all_reminders(db: AsyncSession, user_id: str) -> int:
    """DELETE FROM reminders WHERE user_id = $1 AND sent = FALSE. Returns count."""
    result = await db.execute(
        text("DELETE FROM reminders WHERE user_id = :uid AND sent = FALSE"),
        {"uid": user_id},
    )
    return result.rowcount


async def delete_reminder_by_query(db: AsyncSession, user_id: str, query: str) -> int:
    """DELETE FROM reminders WHERE user_id = $1 AND sent = FALSE AND title ILIKE '%{query}%'. Returns count."""
    result = await db.execute(
        text("DELETE FROM reminders WHERE user_id = :uid AND sent = FALSE AND title ILIKE :q"),
        {"uid": user_id, "q": f"%{query}%"},
    )
    return result.rowcount


# ── Pending actions (server-side conversation state) ─────────────────────────

async def set_pending_action(
    db: AsyncSession, user_id: str, kind: str, payload: dict, ttl_seconds: int = 300,
) -> None:
    """Upsert the user's single in-flight pending action with a TTL."""
    import json
    from datetime import datetime, timedelta, timezone

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    await db.execute(
        text(
            "INSERT INTO pending_actions (user_id, kind, payload, expires_at) "
            "VALUES (:uid, :kind, CAST(:payload AS jsonb), :exp) "
            "ON CONFLICT (user_id) DO UPDATE SET "
            "kind = :kind, payload = CAST(:payload AS jsonb), expires_at = :exp, created_at = NOW()"
        ),
        {"uid": user_id, "kind": kind, "payload": json.dumps(payload), "exp": expires_at},
    )


async def get_pending_action(db: AsyncSession, user_id: str, kind: str | None = None) -> dict | None:
    """Return the user's pending action if present, unexpired, and matching `kind`.

    Expired rows are cleared lazily. `payload` is always returned as a dict.
    """
    from datetime import datetime, timezone

    row = await db.execute(
        text("SELECT * FROM pending_actions WHERE user_id = :uid"), {"uid": user_id}
    )
    rec = _row(row.fetchone())
    if not rec:
        return None
    if rec["expires_at"] < datetime.now(timezone.utc):
        await clear_pending_action(db, user_id)
        return None
    if kind and rec["kind"] != kind:
        return None

    payload = rec.get("payload")
    if isinstance(payload, str):
        import json
        try:
            rec["payload"] = json.loads(payload)
        except json.JSONDecodeError:
            rec["payload"] = {}
    elif not isinstance(payload, dict):
        rec["payload"] = {}
    return rec


async def clear_pending_action(db: AsyncSession, user_id: str) -> None:
    await db.execute(
        text("DELETE FROM pending_actions WHERE user_id = :uid"), {"uid": user_id}
    )


# ── Patches ───────────────────────────────────────────────────────────────────

async def create_patch(db: AsyncSession, user_id: str, diff: dict, kb_entry_id: str | None = None) -> dict:
    import json
    row = await db.execute(
        text(
            "INSERT INTO patches (user_id, kb_entry_id, diff) VALUES (:uid, :kid, :diff) RETURNING *"
        ),
        {"uid": user_id, "kid": kb_entry_id, "diff": json.dumps(diff)},
    )
    return _row(row.fetchone())


async def resolve_patch(db: AsyncSession, patch_id: str, user_id: str, status: str) -> dict:
    row = await db.execute(
        text("UPDATE patches SET status = :status WHERE id = :id AND user_id = :uid RETURNING *"),
        {"status": status, "id": patch_id, "uid": user_id},
    )
    return _row(row.fetchone())