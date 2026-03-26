"""
Google Auth service.
Provides Credentials and handles token refresh.
"""
from __future__ import annotations

import logging
from datetime import datetime
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from sqlalchemy import text

from config import get_settings
from db.connection import get_db

logger = logging.getLogger(__name__)
settings = get_settings()

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/meetings.space.created",
    "https://www.googleapis.com/auth/contacts.readonly"
]

async def get_google_credentials(user: dict) -> Credentials | None:
    access_token = user.get("google_access_token")
    refresh_token = user.get("google_refresh_token")
    if not refresh_token:
        return None

    expiry = None
    if user.get("google_token_expiry"):
        import dateutil.parser
        try:
            # If it's already a datetime object vs string
            val = user["google_token_expiry"]
            if isinstance(val, datetime):
                expiry = val
            else:
                expiry = dateutil.parser.parse(val)
        except Exception:
            pass

    # Credentials expects naive UTC expiry in some versions/scenarios for comparison
    if expiry and expiry.tzinfo:
        from datetime import timezone
        expiry = expiry.astimezone(timezone.utc).replace(tzinfo=None)

    creds = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        scopes=SCOPES,
        expiry=expiry,
    )

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            # Save new tokens
            async with get_db() as db:
                await db.execute(
                    text("""
                        UPDATE users 
                        SET google_access_token = :access,
                            google_token_expiry = :expiry
                        WHERE id = :uid
                    """),
                    {
                        "access": creds.token,
                        "expiry": creds.expiry,
                        "uid": user["id"]
                    }
                )
                await db.commit()
        except Exception as exc:
            logger.error("Failed to refresh Google token: %s", exc)
            return None

    return creds
