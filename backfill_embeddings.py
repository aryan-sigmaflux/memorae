"""Backfill embeddings for kb_entries that have none.

After switching the embedding model (nomic 768-dim -> nvidia nemotron 1024-dim),
the embedding column is rebuilt and existing rows end up with embedding = NULL.
This script re-embeds those rows with the current model so they become
searchable again.

Usage:
    python backfill_embeddings.py            # only rows missing an embedding
    python backfill_embeddings.py --all      # re-embed every row (force refresh)

Run it from the host (or `docker compose exec app python backfill_embeddings.py`)
once the new schema is applied.
"""
import argparse
import asyncio
import logging

from sqlalchemy import text

from db.connection import AsyncSessionLocal
from services.ai import generate_embedding

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("backfill")


async def backfill(force_all: bool) -> None:
    where = "" if force_all else "WHERE embedding IS NULL"
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                text(f"SELECT id, title, content FROM kb_entries {where} ORDER BY created_at")
            )
        ).all()

    total = len(rows)
    logger.info("Found %d entr%s to embed.", total, "y" if total == 1 else "ies")
    if not total:
        return

    done = 0
    failed = 0
    for row in rows:
        entry_id, title, content = row
        text_to_embed = f"{title or ''}\n{content or ''}".strip()
        embedding = await generate_embedding(text_to_embed)
        if not embedding:
            failed += 1
            logger.warning("  ✗ %s — embedding failed, skipped", entry_id)
            continue

        # Commit per row so a mid-run failure still preserves prior progress.
        async with AsyncSessionLocal() as session:
            await session.execute(
                text("UPDATE kb_entries SET embedding = CAST(:emb AS vector) WHERE id = :id"),
                {"emb": str(embedding), "id": entry_id},
            )
            await session.commit()
        done += 1
        if done % 10 == 0 or done == total:
            logger.info("  %d/%d embedded...", done, total)

    logger.info("✅ Backfill complete: %d embedded, %d failed.", done, failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill kb_entries embeddings.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-embed every row, not just those missing an embedding.",
    )
    args = parser.parse_args()
    asyncio.run(backfill(args.all))
