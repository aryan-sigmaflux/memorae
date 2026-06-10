"""
Tools for the Memorae v2 agent (native function calling).

Each tool has:
  * an OpenAI-compatible JSON schema (advertised to the model), and
  * an async implementation that takes a `ToolContext` plus validated args.

Security invariant (Memorae v2 §3): `user_id` is ALWAYS taken from the
authenticated session (`ToolContext`), never from an LLM-supplied argument, so
the model cannot read or mutate another user's data.

Tool implementations return plain dicts. Execution errors are returned as
`{"error": ...}` data — never raised — so the agent can recover.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
import db.queries as q

from services.ai import (
    extract_metadata,
    generate_embedding,
    generate_note_title,
    rerank_notes,
    rewrite_query,
)

logger = logging.getLogger(__name__)

# Recurrence keywords understood by jobs/reminders.py (the cron rescheduler).
_RECURRENCE_VALUES = ("daily", "weekly", "hourly", "monthly")

_DEFAULT_TZ = "Asia/Kolkata"


def user_tz(user: dict) -> ZoneInfo:
    """Resolve the user's IANA timezone from their row, defaulting to IST."""
    name = (user or {}).get("timezone") or _DEFAULT_TZ
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo(_DEFAULT_TZ)


def user_tz_label(user: dict) -> str:
    """Human-readable tz label for the system prompt, e.g. 'Asia/Kolkata (IST UTC+05:30)'."""
    tz = user_tz(user)
    now = datetime.now(tz)
    off = now.strftime("%z")  # e.g. +0530
    return f"{tz.key} ({now.strftime('%Z')} UTC{off[:3]}:{off[3:]})"


@dataclass
class ToolContext:
    """Per-turn execution context. Never exposed to the model."""
    db: AsyncSession
    user_id: str
    user: dict
    # Stamp of when this turn began. confirm_delete only honours a staged
    # deletion created in an EARLIER turn, so staging + confirming cannot both
    # happen in one model turn (a real two-phase gate, not just a prompt rule).
    turn_started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Argument schemas (validated before execution) ────────────────────────────

class CreateNoteArgs(BaseModel):
    content: str = Field(..., description="The full note content to save.")


class SearchNotesArgs(BaseModel):
    query: str = Field(..., description="What to search the user's notes for.")
    category: str | None = Field(
        default=None,
        description="Optional category filter, e.g. work, health, finance, travel, personal.",
    )
    time_range: str | None = Field(
        default=None,
        description="Optional recency filter: one of today, yesterday, this_week, last_week, "
                    "this_month, last_month. Use only when the user scopes by time.",
    )


class EditNoteArgs(BaseModel):
    note_id: str = Field(..., description="The id of the note to edit.")
    content: str = Field(..., description="The full new content for the note.")


class DeleteNoteArgs(BaseModel):
    note_id: str = Field(..., description="The id of the note to delete.")


class CreateReminderArgs(BaseModel):
    title: str
    trigger_datetime: str = Field(
        ..., description="ISO-8601 datetime with timezone offset, e.g. 2026-06-11T17:00:00+05:30."
    )
    recurrence: str | None = Field(
        default=None, description="One of daily, weekly, hourly, monthly, or null for one-shot."
    )


class DeleteReminderArgs(BaseModel):
    query: str | None = Field(
        default=None, description="Substring of the reminder title to delete; null deletes all."
    )


class ScheduleMeetingArgs(BaseModel):
    title: str
    start_time: str = Field(..., description="ISO-8601 datetime with timezone offset.")
    duration_minutes: int = 60
    attendee_email: str | None = None


# ── Datetime helper ──────────────────────────────────────────────────────────

def _resolve_to_utc(value: str, tz: ZoneInfo) -> datetime:
    """Parse a model-supplied datetime string into a TZ-aware UTC datetime.

    Tries ISO-8601 first; falls back to dateparser for looser expressions.
    Naive results are assumed to be in the user's local timezone `tz`. Raises
    ValueError on unparseable input or a time in the past.
    """
    dt: datetime | None = None
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        import dateparser

        dt = dateparser.parse(
            value,
            settings={
                "RELATIVE_BASE": datetime.now(tz).replace(tzinfo=None),
                "PREFER_DATES_FROM": "future",
                "TIMEZONE": tz.key,
                "RETURN_AS_TIMEZONE_AWARE": True,
            },
        )
    if dt is None:
        raise ValueError(f"Could not parse datetime: {value!r}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    dt_utc = dt.astimezone(timezone.utc)

    if dt_utc < datetime.now(timezone.utc):
        raise ValueError("That time is in the past.")
    return dt_utc


# ── Tool implementations ─────────────────────────────────────────────────────

def _coerce_metadata(value: Any) -> dict:
    """JSONB may arrive as a dict or a raw JSON string depending on the driver."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


async def _create_note(ctx: ToolContext, args: CreateNoteArgs) -> dict:
    title = await generate_note_title(args.content)
    metadata = await extract_metadata(args.content)
    embedding = await generate_embedding(f"{title}\n{args.content}")
    entry = await q.create_kb_entry(
        ctx.db,
        user_id=ctx.user_id,
        title=title,
        content=args.content,
        embedding=embedding,
        metadata=metadata,
        source="telegram",
    )
    return {"note_id": str(entry["id"]), "title": title, "category": metadata.get("category")}


def _resolve_time_range(time_range: str | None, tz: ZoneInfo) -> tuple[datetime | None, datetime | None]:
    """Translate a coarse time_range label into UTC (date_from, date_to) bounds."""
    if not time_range:
        return None, None
    now_local = datetime.now(tz)
    today = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    tr = time_range.lower().strip()

    if tr == "today":
        start, end = today, today + timedelta(days=1)
    elif tr == "yesterday":
        start, end = today - timedelta(days=1), today
    elif tr == "this_week":
        start, end = today - timedelta(days=today.weekday()), now_local
    elif tr == "last_week":
        this_week_start = today - timedelta(days=today.weekday())
        start, end = this_week_start - timedelta(days=7), this_week_start
    elif tr == "this_month":
        start, end = today.replace(day=1), now_local
    elif tr == "last_month":
        this_month_start = today.replace(day=1)
        prev_month_end = this_month_start - timedelta(days=1)
        start, end = prev_month_end.replace(day=1), this_month_start
    else:
        return None, None

    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


async def _search_notes(ctx: ToolContext, args: SearchNotesArgs) -> dict:
    rewritten = await rewrite_query(args.query)
    embedding = await generate_embedding(rewritten)
    date_from, date_to = _resolve_time_range(args.time_range, user_tz(ctx.user))
    candidates = await q.search_kb(
        ctx.db,
        user_id=ctx.user_id,
        query=rewritten,
        limit=10,
        embedding=embedding,
        category=args.category,
        date_from=date_from,
        date_to=date_to,
    )

    # Relevance gate: drop notes below the similarity floor so the model gets an
    # empty result (and says "not found") rather than answering from noise. Rows
    # from the no-embedding fallback have no score, so they bypass the gate.
    settings = get_settings()
    candidates = [
        c for c in candidates
        if c.get("similarity_score") is None
        or c["similarity_score"] >= settings.search_min_similarity
    ]
    if not candidates:
        return {"notes": []}

    top = await rerank_notes(args.query, candidates, top_n=3)
    notes = [
        {
            "id": str(n["id"]),
            "title": n.get("title"),
            "content": n.get("content"),
            "category": _coerce_metadata(n.get("metadata")).get("category"),
            "media_url": n.get("media_url"),
            "created_at": n["created_at"].isoformat() if n.get("created_at") else None,
        }
        for n in top
    ]
    return {"notes": notes}


async def _edit_note(ctx: ToolContext, args: EditNoteArgs) -> dict:
    # Read the current title up front so we can write content + the matching
    # embedding in a SINGLE update — no window where content and vector disagree.
    note = await q.get_kb_entry(ctx.db, args.note_id, ctx.user_id)
    if not note:
        return {"error": f"No note found with id {args.note_id}."}

    embedding = await generate_embedding(f"{note.get('title', '')}\n{args.content}")
    fields: dict[str, Any] = {"content": args.content}
    if embedding:
        fields["embedding"] = str(embedding)
    updated = await q.update_kb_entry(
        ctx.db, entry_id=args.note_id, user_id=ctx.user_id, **fields,
    )
    if not updated:
        return {"error": f"No note found with id {args.note_id}."}
    return {"ok": True, "note_id": args.note_id}


async def _delete_note(ctx: ToolContext, args: DeleteNoteArgs) -> dict:
    """Stage a deletion — does NOT delete yet. The model must ask the user, then
    call confirm_delete on a later turn."""
    note = await q.get_kb_entry(ctx.db, args.note_id, ctx.user_id)
    if not note:
        return {"error": f"No note found with id {args.note_id}."}

    await q.set_pending_action(
        ctx.db, ctx.user_id, "delete_note",
        {"note_id": args.note_id, "title": note.get("title")},
        ttl_seconds=300,
    )
    return {
        "status": "confirmation_required",
        "title": note.get("title"),
        "instruction": "Ask the user to confirm deletion of this note. Do not call any "
                       "more tools this turn — wait for their reply, then call confirm_delete.",
    }


async def _confirm_delete(ctx: ToolContext, args: BaseModel) -> dict:
    """Execute a deletion staged by delete_note on a PREVIOUS turn."""
    pending = await q.get_pending_action(ctx.db, ctx.user_id, "delete_note")
    if not pending:
        return {"error": "nothing_to_confirm", "detail": "No staged deletion (it may have expired)."}

    # Reject if it was staged this same turn — forces a real user round-trip.
    if pending["created_at"] >= ctx.turn_started_at:
        return {
            "error": "awaiting_user_confirmation",
            "detail": "Ask the user to confirm first; only confirm after they reply.",
        }

    note_id = pending["payload"].get("note_id")
    title = pending["payload"].get("title")
    deleted = await q.delete_kb_entry(ctx.db, entry_id=note_id, user_id=ctx.user_id)
    await q.clear_pending_action(ctx.db, ctx.user_id)
    if not deleted:
        return {"error": f"Note {note_id} no longer exists."}
    return {"ok": True, "title": title}


async def _create_reminder(ctx: ToolContext, args: CreateReminderArgs) -> dict:
    tz = user_tz(ctx.user)
    try:
        remind_at = _resolve_to_utc(args.trigger_datetime, tz)
    except ValueError as exc:
        return {"error": str(exc)}

    recurrence = args.recurrence
    if recurrence and recurrence not in _RECURRENCE_VALUES:
        recurrence = None  # ignore unsupported recurrence rather than failing

    reminder = await q.create_reminder(
        ctx.db,
        user_id=ctx.user_id,
        title=args.title,
        remind_at=remind_at,
        recurrence=recurrence,
    )
    local = remind_at.astimezone(tz)
    return {
        "reminder_id": str(reminder["id"]),
        "title": reminder["title"],
        "local_time": local.strftime("%b %d, %Y %I:%M %p"),
        "recurrence": recurrence,
    }


async def _list_reminders(ctx: ToolContext, args: BaseModel) -> dict:
    tz = user_tz(ctx.user)
    reminders = await q.list_reminders(ctx.db, ctx.user_id)
    return {
        "reminders": [
            {
                "id": str(r["id"]),
                "title": r["title"],
                "remind_at": r["remind_at"].astimezone(tz).strftime("%b %d, %Y %I:%M %p"),
                "recurrence": r.get("recurrence"),
            }
            for r in reminders
        ]
    }


async def _delete_reminder(ctx: ToolContext, args: DeleteReminderArgs) -> dict:
    if args.query:
        count = await q.delete_reminder_by_query(ctx.db, ctx.user_id, args.query)
    else:
        count = await q.delete_all_reminders(ctx.db, ctx.user_id)
    return {"deleted": count}


def _google_token_dict(user: dict) -> dict | None:
    if not user.get("google_refresh_token"):
        return None
    expiry = user.get("google_token_expiry")
    return {
        "token": user.get("google_access_token"),
        "refresh_token": user.get("google_refresh_token"),
        "expiry": str(expiry).replace(" ", "T") if expiry else None,
    }


async def _schedule_meeting(ctx: ToolContext, args: ScheduleMeetingArgs) -> dict:
    tokens = _google_token_dict(ctx.user)
    if tokens is None:
        return {"error": "google_not_connected"}
    try:
        start_utc = _resolve_to_utc(args.start_time, user_tz(ctx.user))
    except ValueError as exc:
        return {"error": str(exc)}

    from services.google_cal import create_event

    try:
        event, new_tokens = await create_event(
            token_dict=tokens,
            title=args.title,
            start_iso=start_utc.isoformat(),
            duration_minutes=args.duration_minutes,
            timezone="UTC",
            attendee_email=args.attendee_email,
        )
    except Exception as exc:
        logger.error("schedule_meeting failed: %s", exc)
        return {"error": "calendar_error", "detail": str(exc)}

    await q.update_user_google_tokens(ctx.db, ctx.user_id, new_tokens)
    return {
        "title": args.title,
        "meet_link": event.get("hangoutLink"),
        "event_id": event.get("id"),
        "attendee_email": args.attendee_email,
    }


async def _get_calendar_events(ctx: ToolContext, args: BaseModel) -> dict:
    tokens = _google_token_dict(ctx.user)
    if tokens is None:
        return {"error": "google_not_connected"}

    from services.google_cal import list_events

    try:
        events, new_tokens = await list_events(token_dict=tokens, max_results=7)
    except Exception as exc:
        logger.error("get_calendar_events failed: %s", exc)
        return {"error": "calendar_error", "detail": str(exc)}

    await q.update_user_google_tokens(ctx.db, ctx.user_id, new_tokens)
    return {
        "events": [
            {
                "summary": e.get("summary"),
                "start": e.get("start", {}).get("dateTime") or e.get("start", {}).get("date"),
            }
            for e in events
        ]
    }


# ── Registry ─────────────────────────────────────────────────────────────────

class _NoArgs(BaseModel):
    pass


# name -> (arg model, implementation)
_REGISTRY: dict[str, tuple[type[BaseModel], Callable[[ToolContext, Any], Awaitable[dict]]]] = {
    "create_note": (CreateNoteArgs, _create_note),
    "search_notes": (SearchNotesArgs, _search_notes),
    "edit_note": (EditNoteArgs, _edit_note),
    "delete_note": (DeleteNoteArgs, _delete_note),
    "confirm_delete": (_NoArgs, _confirm_delete),
    "create_reminder": (CreateReminderArgs, _create_reminder),
    "list_reminders": (_NoArgs, _list_reminders),
    "delete_reminder": (DeleteReminderArgs, _delete_reminder),
    "schedule_meeting": (ScheduleMeetingArgs, _schedule_meeting),
    "get_calendar_events": (_NoArgs, _get_calendar_events),
}


def _schema(name: str, description: str, model: type[BaseModel]) -> dict:
    params = model.model_json_schema()
    params.pop("title", None)
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": params},
    }


TOOL_SCHEMAS: list[dict] = [
    _schema("create_note", "Save a new note to the user's knowledge base. The backend "
            "generates the title and embedding automatically.", CreateNoteArgs),
    _schema("search_notes", "Search the user's saved notes semantically. Returns the most "
            "relevant notes (with their ids) or an empty list.", SearchNotesArgs),
    _schema("edit_note", "Overwrite the content of an existing note. Find the note id with "
            "search_notes first.", EditNoteArgs),
    _schema("delete_note", "Stage a note for deletion (does NOT delete yet). Returns "
            "confirmation_required; then ask the user to confirm and wait for their reply.", DeleteNoteArgs),
    _schema("confirm_delete", "Execute a deletion previously staged by delete_note, after the "
            "user has confirmed it in their reply.", _NoArgs),
    _schema("create_reminder", "Schedule a reminder. trigger_datetime must be an absolute "
            "ISO-8601 timestamp resolved from the user's local time.", CreateReminderArgs),
    _schema("list_reminders", "List the user's upcoming (unsent) reminders.", _NoArgs),
    _schema("delete_reminder", "Delete reminders by title substring, or all of them.", DeleteReminderArgs),
    _schema("schedule_meeting", "Create a Google Calendar event with a Google Meet link.", ScheduleMeetingArgs),
    _schema("get_calendar_events", "List the user's upcoming Google Calendar events.", _NoArgs),
]


async def execute_tool(name: str, raw_args: dict, ctx: ToolContext) -> dict:
    """Validate args against the tool schema and execute. Errors come back as data."""
    entry = _REGISTRY.get(name)
    if entry is None:
        return {"error": f"Unknown tool: {name}"}

    model, impl = entry
    try:
        args = model.model_validate(raw_args or {})
    except ValidationError as exc:
        return {"error": "invalid_arguments", "detail": exc.errors()}

    try:
        return await impl(ctx, args)
    except Exception as exc:  # never let a tool crash the turn
        logger.error("Tool %s raised: %s", name, exc, exc_info=True)
        return {"error": "tool_execution_failed", "detail": str(exc)}
