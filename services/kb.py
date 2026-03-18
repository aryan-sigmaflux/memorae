"""
Knowledge Base service.
Orchestrates DB queries + AI extraction for KB CRUD operations.
"""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

import db.queries as q
from services.ai import extract_kb_fields

logger = logging.getLogger(__name__)


async def remember(db: AsyncSession, user_id: str, raw_text: str) -> dict:
    """Extract structured fields from raw text and save to KB."""
    fields = await extract_kb_fields(raw_text)
    entry = await q.create_kb_entry(
        db,
        user_id=user_id,
        title=fields.get("title", raw_text[:60]),
        content=fields.get("content", raw_text),
        tags=fields.get("tags", []),
        source="whatsapp",
    )
    logger.info("KB entry created: %s for user %s", entry["id"], user_id)
    return entry


async def recall(db: AsyncSession, user_id: str, query: str) -> list[dict]:
    """Search KB entries for the given query."""
    results = await q.search_kb(db, user_id=user_id, query=query)
    if not results:
        # Fallback: list all and let AI summarise (handled by caller)
        results = await q.list_kb_entries(db, user_id=user_id)
    return results


async def forget(db: AsyncSession, user_id: str, query: str) -> bool:
    """Find the best matching KB entry and delete it."""
    results = await q.search_kb(db, user_id=user_id, query=query, limit=1)
    if not results:
        return False
    deleted = await q.delete_kb_entry(db, entry_id=str(results[0]["id"]), user_id=user_id)
    return deleted


async def list_entries(db: AsyncSession, user_id: str, tag: str | None = None) -> list[dict]:
    return await q.list_kb_entries(db, user_id=user_id, tag=tag)


async def update_entry(db: AsyncSession, entry_id: str, user_id: str, **fields) -> dict:
    return await q.update_kb_entry(db, entry_id=entry_id, user_id=user_id, **fields)