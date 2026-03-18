"""
Auth router.
Handles Google OAuth2 callback and simple API key validation.
"""
from __future__ import annotations

import logging
import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import get_settings
from db.connection import get_db
from db.queries import get_user_by_id, update_user_google_tokens
from services.google_cal import exchange_code

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/auth", tags=["auth"])
bearer_scheme = HTTPBearer(auto_error=False)


# ── Simple bearer token guard ─────────────────────────────────────────────────

async def require_api_key(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    if not creds or creds.credentials != settings.secret_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return creds.credentials


# ── Google OAuth callback ─────────────────────────────────────────────────────

@router.get("/google/callback", response_class=HTMLResponse)
async def google_callback(
    code: str = Query(...),
    state: str = Query(...),
    error: str | None = Query(default=None),
) -> HTMLResponse:
    if error:
        return HTMLResponse(
            content=f"<h2>Authorization failed: {error}</h2>",
            status_code=400,
        )

    try:
        user_id = urllib.parse.unquote(state)
        tokens = exchange_code(code)
    except Exception as exc:
        logger.error("Google OAuth exchange failed: %s", exc)
        return HTMLResponse(content="<h2>Failed to connect Google Calendar.</h2>", status_code=500)

    async with get_db() as db:
        user = await get_user_by_id(db, user_id)
        if not user:
            return HTMLResponse(content="<h2>User not found.</h2>", status_code=404)
        await update_user_google_tokens(db, user_id=user_id, tokens=tokens)

    return HTMLResponse(
        content=(
            "<html><body style='font-family:sans-serif;text-align:center;margin-top:80px'>"
            "<h2>✅ Google Calendar connected!</h2>"
            "<p>You can close this tab and return to WhatsApp.</p>"
            "</body></html>"
        )
    )


# ── Health / status ───────────────────────────────────────────────────────────

@router.get("/status")
async def auth_status() -> dict:
    return {"status": "ok", "env": settings.app_env}