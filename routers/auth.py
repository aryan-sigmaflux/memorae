"""
Auth router.
Handles Google OAuth2 callback and login.
"""
from __future__ import annotations

import logging
import urllib.parse
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google_auth_oauthlib.flow import Flow
from sqlalchemy import text

from config import get_settings
from db.connection import get_db
from db.queries import get_user_by_id
from services.telegram import get_telegram_client

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/auth", tags=["auth"])
bearer_scheme = HTTPBearer(auto_error=False)

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/meetings.space.created",
    "https://www.googleapis.com/auth/contacts.readonly"
]

# ── Simple bearer token guard ─────────────────────────────────────────────────

async def require_api_key(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    if not creds or creds.credentials != settings.secret_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return creds.credentials

# ── Google OAuth ─────────────────────────────────────────────────────────────

@router.get("/google/login", response_class=RedirectResponse)
async def google_login(user_id: str = Query(...)):
    flow = Flow.from_client_secrets_file(
        settings.google_client_secrets_file,
        scopes=SCOPES
    )
    flow.redirect_uri = settings.google_redirect_uri
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        state=user_id,
        include_granted_scopes="true"
    )
    return RedirectResponse(auth_url)

@router.get("/google/callback", response_class=HTMLResponse)
async def google_callback(
    request: Request,
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
        flow = Flow.from_client_secrets_file(
            settings.google_client_secrets_file,
            scopes=SCOPES
        )
        flow.redirect_uri = settings.google_redirect_uri
        flow.fetch_token(code=code)
        creds = flow.credentials

        google_email = None
        if creds.id_token:
            import jwt
            try:
                decoded = jwt.decode(creds.id_token, options={"verify_signature": False})
                google_email = decoded.get("email")
            except BaseException:
                pass
                
        if not google_email:
            try:
                from googleapiclient.discovery import build
                service = build("oauth2", "v2", credentials=creds)
                user_info = service.userinfo().get().execute()
                google_email = user_info.get("email")
            except Exception:
                pass

        async with get_db() as db:
            user = await get_user_by_id(db, user_id)
            if not user:
                return HTMLResponse(content="<h2>User not found.</h2>", status_code=404)
            
            await db.execute(
                text("""
                    UPDATE users 
                    SET google_access_token = :access,
                        google_refresh_token = :refresh,
                        google_token_expiry = :expiry,
                        google_email = :email
                    WHERE id = :uid
                """),
                {
                    "access": creds.token,
                    "refresh": creds.refresh_token,
                    "expiry": creds.expiry,
                    "email": google_email,
                    "uid": user_id
                }
            )
            await db.commit()
            
            # Send Telegram message via tg client
            telegram_id = user.get("telegram_id")
            if telegram_id:
                try:
                    tg = get_telegram_client()
                    await tg.send_text(
                        to=telegram_id,
                        text="✅ Google account connected! You can now use Calendar and Meet."
                    )
                except Exception as e:
                    logger.error(f"Failed to send success message to telegram user {telegram_id}: {e}")

    except Exception as exc:
        logger.error("Google OAuth exchange failed: %s", exc)
        return HTMLResponse(content="<h2>Failed to connect Google Calendar.</h2>", status_code=500)

    return HTMLResponse(
        content=(
            "<html><body style='font-family:sans-serif;text-align:center;margin-top:80px'>"
            "<h2>✅ Google account connected successfully!</h2>"
            "<p>You can close this tab and return to Telegram.</p>"
            "</body></html>"
        )
    )

# ── Health / status ───────────────────────────────────────────────────────────

@router.get("/status")
async def auth_status() -> dict:
    return {"status": "ok", "env": settings.app_env}