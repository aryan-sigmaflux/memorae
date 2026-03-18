"""
WhatsApp Webhook router.
Handles GET verification and POST message events from Meta.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from config import get_settings
from db.connection import get_db
from db.queries import (
    get_or_create_conversation,
    get_or_create_user,
    get_recent_messages,
    save_message,
    touch_conversation,
    create_reminder,
    list_reminders,
)
from services.ai import complete, parse_intent, parse_datetime
from services.kb import forget, recall, remember
from services.media import process_media
from services.toon import (
    Intent,
    format_calendar_events,
    format_kb_results,
    format_reminders,
    get_system_prompt,
)
from services.whatsapp import get_whatsapp_client

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/webhook", tags=["webhook"])


# ── Signature verification ────────────────────────────────────────────────────

def _verify_signature(body: bytes, sig_header: str | None) -> bool:
    if not sig_header or not sig_header.startswith("sha256="):
        return False
    expected = hmac.new(
        settings.wa_access_token.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", sig_header)


# ── GET – hub verification ─────────────────────────────────────────────────────

@router.get("")
async def verify_webhook(request: Request) -> PlainTextResponse:
    params = request.query_params
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == settings.wa_verify_token
    ):
        return PlainTextResponse(params.get("hub.challenge", ""))
    raise HTTPException(status_code=403, detail="Verification failed")


# ── POST – incoming messages ──────────────────────────────────────────────────

@router.post("")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> Response:
    body_bytes = await request.body()

    # Optionally verify signature in production
    if settings.is_production:
        sig = request.headers.get("x-hub-signature-256")
        if not _verify_signature(body_bytes, sig):
            raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    background_tasks.add_task(_process_payload, payload)
    return Response(status_code=200)


# ── Message processing (background) ──────────────────────────────────────────

async def _process_payload(payload: dict) -> None:
    try:
        entry = payload.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})

        messages = value.get("messages", [])
        contacts = value.get("contacts", [])
        display_name = contacts[0]["profile"]["name"] if contacts else None

        for msg in messages:
            await _handle_message(msg, display_name)
    except Exception as exc:
        logger.error("Webhook processing error: %s", exc, exc_info=True)


async def _handle_message(msg: dict, display_name: str | None) -> None:
    wa = get_whatsapp_client()
    phone = msg["from"]
    wa_message_id = msg["id"]
    msg_type = msg.get("type", "text")

    # Mark as read
    try:
        await wa.mark_as_read(wa_message_id)
    except Exception:
        pass

    async with get_db() as db:
        user = await get_or_create_user(db, wa_phone=phone, display_name=display_name)
        user_id = str(user["id"])

        conv = await get_or_create_conversation(db, user_id=user_id)
        conv_id = str(conv["id"])

        # ── Extract text content ──────────────────────────────────────────────
        user_text = ""
        media_url = None
        media_mime = None

        if msg_type == "text":
            user_text = msg.get("text", {}).get("body", "")

        elif msg_type in ("image", "audio", "document", "video"):
            media_obj = msg.get(msg_type, {})
            media_id = media_obj.get("id")
            media_mime = media_obj.get("mime_type", "application/octet-stream")
            caption = media_obj.get("caption", "")

            try:
                media_bytes = await wa.download_media(media_id)
                extracted = await process_media(media_bytes, media_mime)
                user_text = f"{caption}\n[{msg_type.upper()}]: {extracted}".strip()
            except Exception as exc:
                logger.error("Media processing failed: %s", exc)
                user_text = caption or f"[{msg_type} message]"

        else:
            logger.debug("Unhandled message type: %s", msg_type)
            return

        if not user_text:
            return

        # ── Save user message ─────────────────────────────────────────────────
        await save_message(
            db, conv_id, user_id, "user", user_text,
            wa_message_id=wa_message_id,
            media_url=media_url, media_type=media_mime,
        )
        await touch_conversation(db, conv_id)

        # ── Parse intent ──────────────────────────────────────────────────────
        parsed = await parse_intent(user_text)
        reply = await _route_intent(parsed, db, user, conv_id)

        # ── Save assistant message & send ─────────────────────────────────────
        await save_message(db, conv_id, user_id, "assistant", reply)
        await wa.send_text(to=phone, text=reply)


async def _route_intent(parsed, db, user, conv_id: str) -> str:
    user_id = str(user["id"])
    intent = parsed.intent
    payload = parsed.payload

    # ── Remember ──────────────────────────────────────────────────────────────
    if intent == Intent.REMEMBER:
        entry = await remember(db, user_id, parsed.raw)
        return f"Got it! I've saved: \"{entry['title']}\" to your notes."

    # ── Recall ────────────────────────────────────────────────────────────────
    if intent == Intent.RECALL:
        query = payload.get("query", parsed.raw)
        results = await recall(db, user_id, query)
        return format_kb_results(results)

    # ── Forget ────────────────────────────────────────────────────────────────
    if intent == Intent.FORGET:
        deleted = await forget(db, user_id, payload.get("query", parsed.raw))
        return "Done, I've forgotten that." if deleted else "I couldn't find that in your notes."

    # ── Remind ────────────────────────────────────────────────────────────────
    if intent == Intent.REMIND:
        dt_str = payload.get("datetime_str", "")
        iso = await parse_datetime(dt_str, user.get("timezone", "UTC")) if dt_str else None
        if not iso:
            return "I couldn't understand that date/time. Try: 'Remind me tomorrow at 9am to call John'."
        from datetime import datetime
        remind_at = datetime.fromisoformat(iso)
        reminder = await create_reminder(
            db,
            user_id=user_id,
            title=payload.get("title", parsed.raw[:80]),
            remind_at=remind_at,
            body=payload.get("body"),
        )
        return f"Reminder set: \"{reminder['title']}\" at {remind_at.strftime('%b %d, %Y %I:%M %p')}."

    # ── List reminders ────────────────────────────────────────────────────────
    if intent == Intent.LIST_REMINDERS:
        reminders = await list_reminders(db, user_id)
        return format_reminders(reminders)

    # ── Calendar ──────────────────────────────────────────────────────────────
    if intent in (Intent.ADD_CALENDAR, Intent.SHOW_CALENDAR):
        tokens = user.get("google_tokens")
        if not tokens:
            from config import get_settings
            s = get_settings()
            import urllib.parse
            state = urllib.parse.quote(user_id)
            from services.google_cal import get_auth_url
            url = get_auth_url(state=state)
            return f"Please connect your Google Calendar first: {url}"

        if intent == Intent.ADD_CALENDAR:
            dt_str = payload.get("datetime_str", "")
            iso = await parse_datetime(dt_str, user.get("timezone", "UTC")) if dt_str else None
            if not iso:
                return "I couldn't understand the date/time for the event."
            from services.google_cal import create_event
            import json
            event, new_tokens = await create_event(
                token_dict=tokens if isinstance(tokens, dict) else json.loads(tokens),
                title=payload.get("title", "New Event"),
                start_iso=iso,
                duration_minutes=payload.get("duration_minutes", 60),
                timezone=user.get("timezone", "UTC"),
            )
            from db.queries import update_user_google_tokens
            await update_user_google_tokens(db, user_id, new_tokens)
            return f"Event added: \"{event['summary']}\" – {event.get('htmlLink', '')}"

        if intent == Intent.SHOW_CALENDAR:
            from services.google_cal import list_events
            import json
            events, new_tokens = await list_events(
                token_dict=tokens if isinstance(tokens, dict) else json.loads(tokens),
                max_results=7,
            )
            from db.queries import update_user_google_tokens
            await update_user_google_tokens(db, user_id, new_tokens)
            return format_calendar_events(events)

    # ── Fallback: general chat with conversation history ──────────────────────
    history = await get_recent_messages(db, conv_id, limit=10)
    history_msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    # Replace the last user message (already saved) with current
    if not history_msgs or history_msgs[-1]["role"] != "user":
        history_msgs.append({"role": "user", "content": parsed.raw})

    return await complete(messages=history_msgs, system=get_system_prompt())