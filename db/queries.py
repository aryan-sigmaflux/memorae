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

async def get_or_create_user(db: AsyncSession, wa_phone: str, display_name: str | None = None) -> dict:
    row = await db.execute(
        text("SELECT * FROM users WHERE wa_phone = :phone"),
        {"phone": wa_phone},
    )
    user = _row(row.fetchone())
    if user:
        return user

    row = await db.execute(
        text(
            "INSERT INTO users (wa_phone, display_name) VALUES (:phone, :name) RETURNING *"
        ),
        {"phone": wa_phone, "name": display_name},
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
    wa_message_id: str | None = None,
    media_url: str | None = None,
    media_type: str | None = None,
) -> dict:
    row = await db.execute(
        text(
            "INSERT INTO messages "
            "(conversation_id, user_id, role, content, wa_message_id, media_url, media_type) "
            "VALUES (:cid, :uid, :role, :content, :wid, :murl, :mtype) "
            "ON CONFLICT (wa_message_id) DO NOTHING RETURNING *"
        ),
        {
            "cid": conversation_id,
            "uid": user_id,
            "role": role,
            "content": content,
            "wid": wa_message_id,
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
    source: str = "manual",
) -> dict:
    row = await db.execute(
        text(
            "INSERT INTO kb_entries (user_id, title, content, tags, source) "
            "VALUES (:uid, :title, :content, :tags, :source) RETURNING *"
        ),
        {"uid": user_id, "title": title, "content": content, "tags": tags or [], "source": source},
    )
    return _row(row.fetchone())


async def search_kb(db: AsyncSession, user_id: str, query: str, limit: int = 5) -> list[dict]:
    rows = await db.execute(
        text(
            "SELECT *, similarity(content, :q) AS score FROM kb_entries "
            "WHERE user_id = :uid AND content % :q "
            "ORDER BY score DESC LIMIT :lim"
        ),
        {"uid": user_id, "q": query, "lim": limit},
    )
    return _rows(rows.fetchall())


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
    set_clauses = ", ".join(f"{k} = :{k}" for k in fields)
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
        text("SELECT r.*, u.wa_phone FROM reminders r JOIN users u ON u.id = r.user_id "
             "WHERE r.remind_at <= NOW() AND r.sent = FALSE ORDER BY r.remind_at"),
    )
    return _rows(rows.fetchall())


async def mark_reminder_sent(db: AsyncSession, reminder_id: str) -> None:
    await db.execute(
        text("UPDATE reminders SET sent = TRUE WHERE id = :id"),
        {"id": reminder_id},
    )


async def list_reminders(db: AsyncSession, user_id: str) -> list[dict]:
    rows = await db.execute(
        text("SELECT * FROM reminders WHERE user_id = :uid AND sent = FALSE ORDER BY remind_at"),
        {"uid": user_id},
    )
    return _rows(rows.fetchall())


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