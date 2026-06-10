"""
Object storage backed by MinIO (S3-compatible).

Replaces the local `media_bucket` folder. Media bytes (Telegram photos, voice
notes, documents) are stored under keys like `media/<file_id>.<ext>` and the key
is persisted in the `media_url` column.

The MinIO SDK is synchronous, so every call is wrapped in `asyncio.to_thread`
to avoid blocking the FastAPI event loop.
"""
from __future__ import annotations

import asyncio
import io
import logging
from datetime import timedelta
from functools import lru_cache

from config import get_settings

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    return bool(get_settings().minio_endpoint)


@lru_cache
def _client():
    from minio import Minio

    s = get_settings()
    if not s.minio_endpoint:
        raise RuntimeError("MinIO is not configured (set MINIO_ENDPOINT / keys in .env).")
    return Minio(
        s.minio_endpoint,
        access_key=s.minio_access_key,
        secret_key=s.minio_secret_key,
        secure=s.minio_secure,
        region=s.minio_region or None,
    )


def _bucket() -> str:
    return get_settings().minio_bucket


# ── Sync primitives (run via to_thread) ──────────────────────────────────────

def _ensure_bucket_sync() -> None:
    client = _client()
    bucket = _bucket()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info("Created MinIO bucket: %s", bucket)


def _upload_sync(key: str, data: bytes, content_type: str) -> str:
    _client().put_object(
        _bucket(), key, io.BytesIO(data), length=len(data), content_type=content_type,
    )
    return key


def _download_sync(key: str) -> bytes:
    resp = _client().get_object(_bucket(), key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def _delete_sync(key: str) -> None:
    _client().remove_object(_bucket(), key)


def _presigned_sync(key: str, expires_seconds: int) -> str:
    return _client().presigned_get_object(
        _bucket(), key, expires=timedelta(seconds=expires_seconds),
    )


# ── Async API ────────────────────────────────────────────────────────────────

async def ensure_bucket() -> None:
    await asyncio.to_thread(_ensure_bucket_sync)


async def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Store bytes under `key` and return the key."""
    return await asyncio.to_thread(_upload_sync, key, data, content_type)


async def download_bytes(key: str) -> bytes:
    return await asyncio.to_thread(_download_sync, key)


async def delete(key: str) -> None:
    await asyncio.to_thread(_delete_sync, key)


async def presigned_url(key: str, expires_seconds: int = 3600) -> str:
    """Return a time-limited download URL (useful for a web dashboard)."""
    return await asyncio.to_thread(_presigned_sync, key, expires_seconds)
