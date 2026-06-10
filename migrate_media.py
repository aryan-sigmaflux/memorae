"""
One-off media migration: local `media_bucket/` files -> MinIO, and rewrite the
DB references from the legacy form to the MinIO form.

  legacy media_url:  /media/<name>            ->  media/<name>
  legacy tag:        (LOCAL_PATH: media_bucket/<name>) -> (MEDIA_REF: media/<name>)

Idempotent: re-running re-uploads files (overwrite) and the DB updates simply
match 0 rows the second time.

Run it inside the app image with the host media_bucket mounted, so it can reach
both MinIO (minio:9000) and Postgres (db:5432) on the compose network:

  docker compose run --rm -v "$(pwd)/media_bucket:/app/media_bucket" app python migrate_media.py
"""
from __future__ import annotations

import asyncio
import mimetypes
import os

from sqlalchemy import text

from db.connection import AsyncSessionLocal
from services import storage

MEDIA_DIR = "media_bucket"


async def _upload_local_files() -> int:
    if not os.path.isdir(MEDIA_DIR):
        print(f"⚠️  No '{MEDIA_DIR}/' directory found — skipping file upload "
              "(mount it with -v if you have legacy files).")
        return 0

    await storage.ensure_bucket()
    uploaded = 0
    for name in os.listdir(MEDIA_DIR):
        path = os.path.join(MEDIA_DIR, name)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            data = f.read()
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        await storage.upload_bytes(f"media/{name}", data, content_type)
        uploaded += 1
        print(f"  ↑ media/{name}")
    return uploaded


async def _rewrite_db_references() -> None:
    async with AsyncSessionLocal() as db:
        # media_url: "/media/<name>" -> "media/<name>"
        r1 = await db.execute(text(
            "UPDATE messages SET media_url = 'media/' || substring(media_url from '^/media/(.*)$') "
            "WHERE media_url LIKE '/media/%'"
        ))
        r2 = await db.execute(text(
            "UPDATE kb_entries SET media_url = 'media/' || substring(media_url from '^/media/(.*)$') "
            "WHERE media_url LIKE '/media/%'"
        ))
        # content tag: "LOCAL_PATH: media_bucket/" -> "MEDIA_REF: media/"
        r3 = await db.execute(text(
            "UPDATE kb_entries SET content = replace(content, 'LOCAL_PATH: media_bucket/', 'MEDIA_REF: media/') "
            "WHERE content LIKE '%LOCAL_PATH: media_bucket/%'"
        ))
        r4 = await db.execute(text(
            "UPDATE messages SET content = replace(content, 'LOCAL_PATH: media_bucket/', 'MEDIA_REF: media/') "
            "WHERE content LIKE '%LOCAL_PATH: media_bucket/%'"
        ))
        await db.commit()
        print(f"  messages.media_url updated: {r1.rowcount}")
        print(f"  kb_entries.media_url updated: {r2.rowcount}")
        print(f"  kb_entries.content tags updated: {r3.rowcount}")
        print(f"  messages.content tags updated: {r4.rowcount}")


async def main() -> None:
    if not storage.is_configured():
        raise SystemExit("MinIO is not configured — run this inside the app container.")

    print("→ Uploading local media to MinIO …")
    uploaded = await _upload_local_files()

    print("→ Rewriting database references …")
    await _rewrite_db_references()

    print(f"\n✅ Done. Uploaded {uploaded} file(s).")


if __name__ == "__main__":
    asyncio.run(main())
