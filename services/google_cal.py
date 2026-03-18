"""
Google Calendar service.
Handles OAuth token refresh and Calendar API operations.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
]

CLIENT_CONFIG = {
    "web": {
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "redirect_uris": [settings.google_redirect_uri],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}


# ── OAuth helpers ─────────────────────────────────────────────────────────────

def get_auth_url(state: str) -> str:
    flow = Flow.from_client_config(CLIENT_CONFIG, scopes=SCOPES)
    flow.redirect_uri = settings.google_redirect_uri
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        state=state,
        prompt="consent",
    )
    return auth_url


def exchange_code(code: str) -> dict:
    flow = Flow.from_client_config(CLIENT_CONFIG, scopes=SCOPES)
    flow.redirect_uri = settings.google_redirect_uri
    flow.fetch_token(code=code)
    creds = flow.credentials
    return _creds_to_dict(creds)


def _creds_to_dict(creds: Credentials) -> dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }


def _build_service(token_dict: dict):
    expiry = None
    if token_dict.get("expiry"):
        expiry = datetime.fromisoformat(token_dict["expiry"])

    creds = Credentials(
        token=token_dict["token"],
        refresh_token=token_dict.get("refresh_token"),
        token_uri=token_dict.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=token_dict.get("client_id", settings.google_client_id),
        client_secret=token_dict.get("client_secret", settings.google_client_secret),
        scopes=token_dict.get("scopes", SCOPES),
        expiry=expiry,
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("calendar", "v3", credentials=creds, cache_discovery=False), creds


# ── Calendar operations ───────────────────────────────────────────────────────

async def create_event(
    token_dict: dict,
    title: str,
    start_iso: str,
    duration_minutes: int = 60,
    description: str = "",
    timezone: str = "UTC",
) -> tuple[dict, dict]:
    """
    Create a Calendar event.
    Returns (event_dict, refreshed_token_dict).
    """
    service, creds = _build_service(token_dict)

    start_dt = datetime.fromisoformat(start_iso)
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    event_body = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": timezone},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": timezone},
    }
    event = service.events().insert(calendarId="primary", body=event_body).execute()
    logger.info("Calendar event created: %s", event.get("id"))
    return event, _creds_to_dict(creds)


async def list_events(
    token_dict: dict,
    time_min: str | None = None,
    time_max: str | None = None,
    max_results: int = 10,
) -> tuple[list[dict], dict]:
    service, creds = _build_service(token_dict)

    if time_min is None:
        time_min = datetime.now(timezone.utc).isoformat()

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    return events_result.get("items", []), _creds_to_dict(creds)


async def delete_event(token_dict: dict, event_id: str) -> dict:
    service, creds = _build_service(token_dict)
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return _creds_to_dict(creds)