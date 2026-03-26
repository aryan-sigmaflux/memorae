"""
Knowledge Base service.
Orchestrates DB queries + AI extraction for KB CRUD operations.
"""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

import db.queries as q
from services.ai import extract_kb_fields, generate_embedding

logger = logging.getLogger(__name__)


async def remember(db: AsyncSession, user_id: str, raw_text: str, media_url: str | None = None, media_type: str | None = None) -> dict:
    """Extract structured fields from raw text and save to KB."""
    fields = await extract_kb_fields(raw_text)
    
    title = fields.get("title", raw_text[:60])
    content = fields.get("content", raw_text)
    
    embedding = await generate_embedding(f"{title}\n{content}")
    
    entry = await q.create_kb_entry(
        db,
        user_id=user_id,
        title=title,
        content=content,
        tags=fields.get("tags", []),
        embedding=embedding,
        source="telegram",
        media_url=media_url,
        media_type=media_type,
    )
    logger.info("KB entry created: %s for user %s", entry["id"], user_id)
    return entry


async def recall(db: AsyncSession, user_id: str, query: str) -> list[dict]:
    """Search KB entries for the given query."""
    embedding = await generate_embedding(query)
    return await q.search_kb(db, user_id=user_id, query=query, embedding=embedding)


async def forget(db: AsyncSession, user_id: str, query: str) -> bool:
    """Find the best matching KB entry and delete it."""
    embedding = await generate_embedding(query)
    results = await q.search_kb(db, user_id=user_id, query=query, limit=1, embedding=embedding)
    if not results:
        return False
    deleted = await q.delete_kb_entry(db, entry_id=str(results[0]["id"]), user_id=user_id)
    return deleted


async def list_entries(db: AsyncSession, user_id: str, tag: str | None = None) -> list[dict]:
    return await q.list_kb_entries(db, user_id=user_id, tag=tag)


async def update_entry(db: AsyncSession, entry_id: str, user_id: str, **fields) -> dict:
    return await q.update_kb_entry(db, entry_id=entry_id, user_id=user_id, **fields)