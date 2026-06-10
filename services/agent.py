"""
Memorae v2 agent loop (ReAct-style native tool calling).

Replaces the v1 TOON / `<kb_patch>` / multi-stage intent-parsing pipeline with
a single inspectable loop over the provider SDK's `tools` array.

Loop guards (Memorae v2 §2.2):
  * max iterations per turn (runaway-cost protection),
  * argument validation before execution (in services.tools),
  * tool errors returned as data so the model can self-correct.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

from config import get_settings
from services.ai import complete_with_tools, count_tokens
from services.tools import TOOL_SCHEMAS, ToolContext, execute_tool, user_tz, user_tz_label

logger = logging.getLogger(__name__)
settings = get_settings()


SYSTEM_TEMPLATE = """\
You are Memo, a warm, witty, and reliable personal assistant living inside Telegram.
You are the user's second brain: you save notes AND files (photos, PDFs, documents),
recall them, set reminders, and manage their Google Calendar. Saved photos/PDFs/files
ARE notes that have attached media — you absolutely DO have access to them. Never tell
the user you can't access their files; look them up instead.

Current time (user's local): {local_now}
User's timezone: {user_tz}
Google Calendar connected: {google_connected}

How you work:
- Decide what the user wants and use the available tools to do it. Do not describe
  tool calls to the user; just call them and then reply naturally.
- SAVING: When the user tells you something worth remembering (a fact, plan, contact,
  instruction), call create_note with the full content. Confirm briefly afterwards.
- RECALL: When the user asks a question that their notes might answer, call
  search_notes. Pass the category and/or time_range filters when the user scopes
  the request (e.g. "my work notes", "notes from last week"). Answer ONLY from the
  returned notes. If search_notes returns no notes, say you couldn't find anything —
  never invent an answer.
- LISTING: For "what do you have", "list my notes", or "show my files/images/PDFs",
  call list_notes (set media_only=true for files/images/PDFs) and list what comes back.
  Do NOT answer these from memory and do NOT claim you have no files — always check first.
- If a recalled note contains a tag like (MEDIA_REF: media/...) and the user
  wants the file, include that exact tag verbatim in your reply so it can be sent.
- EDITING: To change a note, find it with search_notes, then call edit_note with the
  full rewritten content.
- DELETING (two steps): First find the note with search_notes, then call delete_note —
  this only STAGES the deletion and returns confirmation_required. Show the title and
  ask "Delete '[title]'?", then STOP (no more tools this turn). When the user confirms
  in their next message, call confirm_delete. If they decline, just move on.
- REMINDERS: Resolve relative times ("tomorrow at 5pm") into an absolute ISO-8601
  timestamp WITH the user's timezone offset, using the current time above. The backend
  re-validates and rejects past times.
- CALENDAR: Use schedule_meeting / get_calendar_events. If a calendar tool returns
  {{"error": "google_not_connected"}}, tell the user to connect Google via /start.

Keep replies concise and plain-text (no markdown). Be honest when you can't do something.
"""


def _build_system_prompt(user: dict) -> str:
    tz = user_tz(user)
    now_local = datetime.now(timezone.utc).astimezone(tz)
    return SYSTEM_TEMPLATE.format(
        local_now=now_local.strftime("%Y-%m-%d %H:%M (%A)"),
        user_tz=user_tz_label(user),
        google_connected="yes" if user.get("google_refresh_token") else "no",
    )


def _assistant_dict(msg) -> dict:
    out: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return out


async def run_agent(ctx: ToolContext, history_msgs: list[dict]) -> str:
    """Run the agent loop for one user turn and return the final reply text.

    `history_msgs` is the recent conversation in chronological order, already
    ending with the user's current message.
    """
    messages: list[dict] = [{"role": "system", "content": _build_system_prompt(ctx.user)}]
    messages.extend(history_msgs)

    final_text = ""
    reported_tokens = 0           # cumulative tokens the provider reported (may be 0)
    tokens_used = 0               # best estimate used for the ceiling check
    tool_failures: dict[str, int] = {}  # per-tool consecutive-error counter (retry cap)
    aborted = False

    for iteration in range(settings.agent_max_iterations):
        msg, turn_tokens = await complete_with_tools(messages, TOOL_SCHEMAS)
        reported_tokens += turn_tokens

        # Some providers (OpenRouter) omit usage; estimate from the transcript
        # so the per-turn ceiling still works.
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if not tc.id:
                    tc.id = f"call_{uuid.uuid4().hex}"
        messages.append(_assistant_dict(msg))
        tokens_used = max(reported_tokens, count_tokens(json.dumps(messages, default=str)))

        if not msg.tool_calls:
            final_text = msg.content or ""
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                raw_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                result = {"error": "invalid_arguments", "detail": "arguments were not valid JSON"}
            else:
                started = time.monotonic()
                result = await execute_tool(name, raw_args, ctx)
                logger.info(
                    "🛠 tool=%s args=%s ms=%d result_keys=%s",
                    name, raw_args, int((time.monotonic() - started) * 1000), list(result.keys()),
                )

            # Retry cap: if the same tool keeps failing, stop letting the model
            # retry it and hand back a terminal error so it answers the user.
            if "error" in result:
                tool_failures[name] = tool_failures.get(name, 0) + 1
                if tool_failures[name] > settings.agent_max_tool_retries:
                    result = {
                        "error": "tool_unavailable",
                        "detail": f"{name} failed {tool_failures[name]} times; stop retrying and "
                                  "tell the user you couldn't complete that part.",
                    }
            else:
                tool_failures.pop(name, None)

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)}
            )

        # Per-turn token ceiling (§2.2): abort before another costly round-trip.
        if tokens_used >= settings.agent_max_tokens_per_turn:
            logger.warning(
                "Agent hit token ceiling (%d/%d) for user %s — aborting loop.",
                tokens_used, settings.agent_max_tokens_per_turn, ctx.user_id,
            )
            aborted = True
            break
    else:
        # Loop exceeded max iterations — force a final text-only answer.
        logger.warning("Agent hit max iterations (%d) for user %s",
                       settings.agent_max_iterations, ctx.user_id)
        aborted = True

    if aborted and not final_text:
        final_text = await _force_final_reply(messages)

    billable = reported_tokens or tokens_used
    cost = billable / 1000 * settings.ai_price_per_1k_tokens
    logger.info(
        "📊 turn complete: user=%s tokens=%d (reported=%d) est_cost=$%.4f",
        ctx.user_id, billable, reported_tokens, cost,
    )
    return final_text or "Done ✅"


async def _force_final_reply(messages: list[dict]) -> str:
    """Ask the model for a plain-text wrap-up after the iteration cap."""
    from services.ai import _client

    try:
        response = await _client().chat.completions.create(
            model=settings.ai_model,
            max_tokens=settings.ai_max_tokens,
            messages=messages
            + [{"role": "system", "content": "Stop using tools. Reply to the user now in plain text."}],
            tool_choice="none",
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Final reply generation failed: %s", exc)
        return "Sorry, that took too many steps — could you rephrase?"
