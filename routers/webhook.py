"""
Telegram Webhook router.
Handles POST message events from Telegram.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Request, Response

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
    delete_all_reminders,
    delete_reminder_by_query,
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
from services.telegram import get_telegram_client
from telegram import Update

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/webhook", tags=["webhook"])

# In-memory dict for storing unconfirmed media uploads temporarily
pending_saves: dict[str, dict] = {}

# ── POST – incoming messages ──────────────────────────────────────────────────

@router.post("")
async def webhook(
    req: Request,
    background_tasks: BackgroundTasks,
) -> Response:
    data = await req.json()
    background_tasks.add_task(_process_payload, data)
    return Response(content='{"ok": true}', media_type="application/json", status_code=200)


# ── Message processing (background) ──────────────────────────────────────────

async def _process_payload(payload: dict) -> None:
    try:
        if "update_id" not in payload:
            logger.warning("Received non-Telegram payload (missing 'update_id'). Ignoring.")
            return

        tg = get_telegram_client()
        update = Update.de_json(payload, tg.bot)
        logger.debug("Received update: %s", payload.get("update_id"))

        if not update.message:
            return  # we only process new messages

        chat_id = update.message.chat.id
        if hasattr(settings, "allowed_chat_ids") and settings.allowed_chat_ids:
            if chat_id not in settings.allowed_chat_ids and str(chat_id) not in [str(c) for c in settings.allowed_chat_ids]:
                logger.info(f"Ignoring message from unauthorized chat_id: {chat_id}")
                return

        display_name = update.message.from_user.first_name if update.message.from_user else None
        
        await _handle_message(update, display_name, tg)
    except Exception as exc:
        logger.error("Webhook processing error: %s", exc, exc_info=True)


async def _handle_message(update: Update, display_name: str | None, tg: Any) -> None:
    chat_id = update.message.chat.id
    message_id = str(update.message.message_id)

    async with get_db() as db:
        user = await get_or_create_user(db, telegram_id=str(chat_id), display_name=display_name)
        user_id = str(user["id"])

        conv = await get_or_create_conversation(db, user_id=user_id)
        conv_id = str(conv["id"])

        # ── Extract text content ──────────────────────────────────────────────
        user_text = ""
        media_url = None
        media_mime = None
        
        caption = update.message.caption or ""
        is_media = False

        if update.message.text:
            user_text = update.message.text

        elif update.message.photo or (hasattr(update.message, 'document') and update.message.document) or (hasattr(update.message, 'audio') and update.message.audio) or (hasattr(update.message, 'voice') and update.message.voice):
            is_media = True
            if update.message.photo:
                photo = update.message.photo[-1]
                media_id = photo.file_id
                media_mime = "image/jpeg"
                prefix = "[IMAGE]"
            elif hasattr(update.message, 'document') and update.message.document:
                doc = update.message.document
                media_id = doc.file_id
                media_mime = doc.mime_type or "application/octet-stream"
                prefix = "[DOCUMENT]"
            else:
                audio = update.message.audio or update.message.voice
                media_id = audio.file_id
                media_mime = audio.mime_type or "audio/ogg"
                prefix = "[AUDIO]"
            
            try:
                media_bytes = await tg.download_media(media_id)
                
                import os
                import mimetypes
                os.makedirs("media_bucket", exist_ok=True)
                ext = mimetypes.guess_extension(media_mime) or ".bin"
                # Some audio extensions fix
                if media_mime == "audio/ogg":
                    ext = ".ogg"
                local_path = f"media_bucket/{media_id}{ext}"
                with open(local_path, "wb") as f:
                    f.write(media_bytes)
                media_url = f"/media/{media_id}{ext}"
                
                # Immediate status update for the user
                media_label = prefix.strip("[]").lower()
                if media_label == "document": media_label = "pdf"
                await tg.send_text(to=chat_id, text=f"Reading the {media_label}... this might take a while")

                try:
                    extracted = await process_media(media_bytes, media_mime)
                except Exception as exc:
                    logger.error("Media processing OCR failed, falling back to basic extraction: %s", exc)
                    extracted = f"{prefix} image captured."

                user_text = f"{caption}\n{prefix}(LOCAL_PATH: {local_path}): {extracted}".strip()
                if user_text.startswith("\n"): user_text = user_text.lstrip("\n")
                
                from services.kb import remember
                from services.toon import quick_parse, Intent as TIntent
                
                q_intent = quick_parse(caption) if caption else None
                
                # If they explicitly told the bot to save it, do it silently and return
                if q_intent == TIntent.REMEMBER:
                    logger.info("👤 USER (%s) (SILENT SAVE): %s", display_name or chat_id, user_text)
                    await remember(db, user_id, user_text, media_url=media_url, media_type=media_mime)
                    await save_message(db, conv_id, user_id, "user", user_text, telegram_message_id=message_id, media_url=media_url, media_type=media_mime)
                    await touch_conversation(db, conv_id)
                    return
                # Otherwise, it lacks explicit save instruction. Prompt the user!
                else:
                    pending_saves[user_id] = {
                        "media_url": media_url,
                        "media_type": media_mime,
                        "user_text": user_text,
                        "description": extracted
                    }
                    
                    logger.info("👤 USER (%s): %s", display_name or chat_id, user_text)
                    await save_message(db, conv_id, user_id, "user", user_text, telegram_message_id=message_id, media_url=media_url, media_type=media_mime)
                    await touch_conversation(db, conv_id)
                    
                    reply = f"{extracted}\n\nWant me to save this?"
                    logger.info("🤖 ASSISTANT: %s", reply)
                    await save_message(db, conv_id, user_id, "assistant", reply)
                    await tg.send_text(to=chat_id, text=reply)
                    return
            except Exception as exc:
                logger.error("Complete media dispatching fail: %s", exc)
                return
                
        else:
            logger.debug("Unhandled message type")
            return

        if not user_text:
            return

        if user_text.strip() == "/start":
            is_connected = bool(user.get("google_refresh_token"))

            if is_connected:
                reply = "👋 Welcome back! What can I help you with today?"
            else:
                oauth_url = f"{settings.api_base_url}/auth/google/login?user_id={user_id}"
                reply = (
                    "👋 Welcome to Memo!\n\n"
                    "To get started, connect your Google account so I can manage your Calendar and Meet.\n\n"
                    f"🔗 Sign in with Google: {oauth_url}\n\n"
                    "Once connected, just chat with me naturally — I'll handle the rest!"
                )
            
            logger.info("👤 USER (%s): %s", display_name or chat_id, user_text)
            await save_message(db, conv_id, user_id, "user", user_text, telegram_message_id=message_id)
            await touch_conversation(db, conv_id)
            logger.info("🤖 ASSISTANT: %s", reply)
            await save_message(db, conv_id, user_id, "assistant", reply)
            await tg.send_text(to=chat_id, text=reply)
            return

        # ── Save user message ─────────────────────────────────────────────────
        logger.info("👤 USER (%s): %s", display_name or chat_id, user_text)
        await save_message(
            db, conv_id, user_id, "user", user_text,
            telegram_message_id=message_id,
            media_url=media_url, media_type=media_mime,
        )
        await touch_conversation(db, conv_id)

        # Indicate that the bot is "typing..."
        await tg.send_typing_action(chat_id)

        history = await get_recent_messages(db, conv_id, limit=6)
        history_msgs = [{"role": m["role"], "content": m["content"]} for m in history]

        # ── Parse intent ──────────────────────────────────────────────────────
        parsed = await parse_intent(user_text, history_msgs)
        logger.info("🎯 INTENT: %s", parsed.intent)
        
        # If the user denies or moves on, clear their pending state (unless intent is CONFIRM_SAVE or REMEMBER)
        if parsed.intent not in (Intent.CONFIRM_SAVE, Intent.REMEMBER) and user_id in pending_saves:
            # We only clear it if they replied to something else, ignoring the media prompt
            if not is_media:
                pending_saves.pop(user_id, None)

        reply = await _route_intent(parsed, db, user, conv_id, history_msgs)
        logger.info("🤖 ASSISTANT: %s", reply)

        # ── Save assistant message & send ─────────────────────────────────────
        await save_message(db, conv_id, user_id, "assistant", reply)
        
        import re
        path_match = re.search(r'LOCAL_PATH:\s*(media_bucket/[^\s)]+)', reply)
        if path_match:
            local_path = path_match.group(1)
            clean_reply = re.sub(r'\(?LOCAL_PATH:\s*media_bucket/[^\s)]+\)?', '', reply).strip()
            try:
                with open(local_path, "rb") as f:
                    await tg.bot.send_photo(chat_id=chat_id, photo=f, caption=clean_reply)
            except Exception as e:
                logger.error("Failed to send photo: %s", e)
                await tg.send_text(to=chat_id, text=clean_reply)
        else:
            await tg.send_text(to=chat_id, text=reply)

async def _route_intent(parsed, db, user, conv_id: str, history_msgs: list[dict]) -> str:
    user_id = str(user["id"])
    intent = parsed.intent
    payload = parsed.payload

    # ── Connect Google ────────────────────────────────────────────────────────
    if intent == Intent.CONNECT_GOOGLE:
        oauth_url = f"{settings.api_base_url}/auth/google/login?user_id={user_id}"
        return f"Tap here to connect your Google account: {oauth_url}"

    # ── Remember ──────────────────────────────────────────────────────────────
    if intent == Intent.REMEMBER:
        text_to_save = payload.get("content_to_save") or parsed.raw
        
        if user_id in pending_saves:
            p = pending_saves.pop(user_id)
            combined_text = f"User Instruction: {text_to_save}\n\nDocument Content:\n{p['user_text']}"
            entry = await remember(db, user_id, combined_text, media_url=p["media_url"], media_type=p["media_type"])
        else:
            entry = await remember(db, user_id, text_to_save)
            
        return f"Got it! I've saved this to your notes: \"{entry['title']}\""

    # ── Confirm Save ──────────────────────────────────────────────────────────
    if intent == Intent.CONFIRM_SAVE:
        if user_id in pending_saves:
            p = pending_saves.pop(user_id)
            entry = await remember(db, user_id, p["user_text"], media_url=p["media_url"], media_type=p["media_type"])
            return "Image saved."
        else:
            # They said "yes" but there's nothing pending
            pass # Fall through to letting AI chat about whatever they said yes to

    # ── Recall ────────────────────────────────────────────────────────────────
    if intent == Intent.RECALL:
        query = payload.get("query", parsed.raw)
        logger.info(f"Recall query: {query}")
        results = await recall(db, user_id, query)
        
        # If no results and query implies media, fallback to recent media
        media_keywords = {"image", "photo", "picture", "video", "file", "pdf", "document", "img"}
        query_words = set(query.lower().split())
        
        # Fuzzy check: do any query words contain or start with a media keyword? (catches "imaage", "photos", etc)
        is_media_query = any(any(kw in qw or qw in kw for kw in media_keywords) for qw in query_words)
        
        if not results and is_media_query:
            from sqlalchemy import text
            rows = await db.execute(
                text("SELECT * FROM kb_entries WHERE user_id = :uid AND media_url IS NOT NULL ORDER BY updated_at DESC LIMIT 5"),
                {"uid": user_id}
            )
            results = [dict(r._mapping) for r in rows.fetchall()]
            
        # If still no results, but they have an active unsaved pending save, logically map that into the query loop!
        if not results and user_id in pending_saves and is_media_query:
            pending = pending_saves[user_id]
            results = [{
                "title": "Unsaved Pending Image",
                "content": pending["description"],
                "media_url": pending["media_url"],
                "media_type": pending["media_type"]
            }]
            
        if not results:
            return "I couldn't find anything relevant in your notes."
            
        # Check if any entry has a natively saved file path
        # If yes, resolve the file path and send the media natively with the content as caption.
        from services.telegram import get_telegram_client
        tg = get_telegram_client()
        media_dispatched = False
        
        # Only dispatch files natively if the user explicitly asked to "send/show/give/fetch/see" it.
        fetch_keywords = {"send", "show", "give", "display", "get", "fetch", "see", "retrieve", "image", "photo", "file", "pdf"}
        is_fetch_request = any(kw in parsed.raw.lower() for kw in fetch_keywords)
        is_question = "?" in parsed.raw or any(kw in parsed.raw.lower() for kw in ["what", "how", "who", "where", "why", "tell"])
        
        # Native dispatch only if it's a retrieval request and NOT just a general question about content
        if is_fetch_request or not is_question:
            for r in results:
                if r.get("media_url"):
                    local_path = r["media_url"].lstrip("/").replace("media/", "media_bucket/")
                    try:
                        with open(local_path, "rb") as f:
                            chat_id = user["telegram_id"]
                            m_type = r.get("media_type", "")
                            if m_type.startswith("image/"):
                                await tg.bot.send_photo(chat_id=chat_id, photo=f)
                            elif m_type.startswith("video/"):
                                await tg.bot.send_video(chat_id=chat_id, video=f)
                            else:
                                await tg.bot.send_document(chat_id=chat_id, document=f)
                        media_dispatched = True
                        break # Usually we only dispatch the single best match 
                    except Exception as e:
                        logger.error("Failed to native-send recalled media file: %s", e)
        
        # Feed the KB results as context to the AI so it answers the question naturally
        kb_context = "\n".join([f"- {r['title']}: {r['content']}" for r in results[:5]])
        context_prompt = (
            f"The user asked: \"{parsed.raw}\"\n\n"
            f"Here are relevant fresh entries from their personal notes database:\n{kb_context}\n\n"
            "CRITICAL: When answering a specific question:\n"
            "- If the saved notes contain an entry that explicitly matches the type of information the user asked for (same label or clear context), answer directly and confidently. Example: user asks 'what is my roll number' and notes contain 'Student Roll Number: 77' → respond 'Your student roll number is 77.'\n"
            "- If the user asks to 'send' or 'show' a saved document, return its FULL content verbatim, not a summary. Do not describe it — show it.\n"
            "- Only use the hedging response ('I don't have that saved. I do have X — is that what you meant?') when the notes contain a NUMBER or VALUE but its context does NOT match what was asked. Example: user asks 'roll number' but only vehicle numbers are saved.\n"
            "- Never hedge when you have an exact or near-exact context match.\n\n"
            "CRITICAL RULE: ALWAYS use these fresh database entries as the sole source of truth. Ignore any older values that might appear in the conversation history or your memory.\n"
            "Answer the user's question using ONLY the fresh information from their notes above. "
            "Be concise and direct."
        )
        
        system = get_system_prompt()
        if media_dispatched:
             system += "\nCRITICAL RULE: We already sent the user their requested file natively! If they ONLY asked an imperative like 'send me the image', 'fetch the file', 'show the photo', return exactly '[SILENT]' and NOTHING else. But if they asked a QUESTION about the file content (like 'tell me about the image'), you MUST answer using the OCR context text!"
        
        raw_final = await complete(
            messages=[{"role": "user", "content": context_prompt}],
            system=system,
        )
        
        if "[SILENT]" in raw_final:
            return ""
            
        import re
        return re.sub(r'\[IMAGE\]\(?LOCAL_PATH:\s*media_bucket/[^\s)]+\)?:\s*', '', raw_final).strip()

    # ── Forget ────────────────────────────────────────────────────────────────
    if intent == Intent.FORGET:
        deleted = await forget(db, user_id, payload.get("query", parsed.raw))
        return "Done, I've forgotten that." if deleted else "I couldn't find that in your notes."

    # ── Remind ────────────────────────────────────────────────────────────────
    if intent == Intent.REMIND:
        dt_str = payload.get("datetime_str", "")
        if not dt_str:
            return "I couldn't understand that date/time. Try: 'Remind me tomorrow at 9am to call John'."
        try:
            iso = await parse_datetime(dt_str)
        except ValueError:
            return "I couldn't understand that date/time. Try: 'Remind me tomorrow at 9am to call John'."
        from datetime import datetime, timezone
        from services.ai import USER_TZ
        remind_at = datetime.fromisoformat(iso)
        
        if remind_at.tzinfo is None:
            remind_at = remind_at.replace(tzinfo=USER_TZ).astimezone(timezone.utc)
        
        # Correction detection
        correction_keywords = {"no", "wrong", "not that", "i said", "i meant", "change it to"}
        is_correction = any(kw in parsed.raw.lower() for kw in correction_keywords)
        
        if is_correction:
            from db.queries import update_most_recent_reminder
            updated = await update_most_recent_reminder(db, user_id=user_id, remind_at=remind_at)
            if updated:
                remind_at_local = remind_at.astimezone(USER_TZ)
                return f"Got it, correction made! I've updated \"{updated['title']}\" to {remind_at_local.strftime('%b %d, %Y %I:%M %p')}."

        title = payload.get("title") or payload.get("body") or parsed.raw[:80].strip() or "Reminder"
        reminder = await create_reminder(
            db,
            user_id=user_id,
            title=title,
            remind_at=remind_at,
            body=payload.get("body"),
            recurrence=payload.get("recurrence"),
        )
        remind_at_local = remind_at.astimezone(USER_TZ)
        return f"Reminder set: \"{reminder['title']}\" at {remind_at_local.strftime('%b %d, %Y %I:%M %p')}."

    # ── List reminders ────────────────────────────────────────────────────────
    if intent == Intent.LIST_REMINDERS:
        reminders = await list_reminders(db, user_id)
        return format_reminders(reminders)

    # ── Delete reminders ──────────────────────────────────────────────────────
    if intent == Intent.DELETE_REMINDERS:
        query = parsed.payload.get("query")
        if query:
            count = await delete_reminder_by_query(db, user_id, query)
            return f"Deleted {count} reminder(s) matching '{query}'."
        else:
            count = await delete_all_reminders(db, user_id)
            return f"Done. Deleted all {count} pending reminder(s)."

    # ── Calendar ──────────────────────────────────────────────────────────────
    if intent in (Intent.ADD_CALENDAR, Intent.SHOW_CALENDAR):
        if not user.get("google_refresh_token"):
            oauth_url = f"{settings.api_base_url}/auth/google/login?user_id={user_id}"
            return f"Please connect your Google Calendar first: {oauth_url}"
        
        tokens = {
            "token": user.get("google_access_token"),
            "refresh_token": user.get("google_refresh_token"),
            "expiry": str(user.get("google_token_expiry")).replace(" ", "T") if user.get("google_token_expiry") else None
        }

        if intent == Intent.ADD_CALENDAR:
            dt_str = payload.get("datetime_str", "")
            if not dt_str:
                return "I couldn't understand the date/time for the event. Try something like 'tomorrow at 5pm' or 'March 25 at 3pm'."
            try:
                iso = await parse_datetime(dt_str)
            except ValueError:
                return "I couldn't understand the date/time for the event. Try something like 'tomorrow at 5pm' or 'March 25 at 3pm'."
            
            from datetime import datetime, timezone
            from services.ai import USER_TZ
            cal_start = datetime.fromisoformat(iso)
            if cal_start.tzinfo is None:
                iso = cal_start.replace(tzinfo=USER_TZ).astimezone(timezone.utc).isoformat()
                
            from services.google_cal import create_event
            import json
            attendee_email = payload.get("attendee_email")
            title = payload.get("title", "New Event")
            event, new_tokens = await create_event(
                token_dict=tokens if isinstance(tokens, dict) else json.loads(tokens),
                title=title,
                start_iso=iso,
                duration_minutes=payload.get("duration_minutes", 60),
                timezone="Asia/Kolkata",
                attendee_email=attendee_email,
            )
            from db.queries import update_user_google_tokens
            await update_user_google_tokens(db, user_id, new_tokens)
            
            meet_link = event.get("hangoutLink")
            if meet_link:
                return (
                    f'✅ "{title}" scheduled!\n'
                    f'📹 Meet: {meet_link}\n'
                    f'✉️ Invite sent to {attendee_email}' if attendee_email else f'✅ "{title}" scheduled!\n📹 Meet: {meet_link}'
                )
            return f'✅ "{title}" added to calendar.'

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
    # Always check KB for relevant context before chatting
    from db.queries import search_kb
    kb_results = await search_kb(db, user_id=user_id, query=parsed.raw, limit=20)
    system = get_system_prompt()
    if kb_results:
        kb_context = "\n".join([f"- {r['title']}: {r['content']}" for r in kb_results])
        system += (
            f"\n\nYou have access to the following freshly fetched personal notes from the database. "
            "CRITICAL: When answering a specific question:\n"
            "- If the saved notes contain an entry that explicitly matches the type of information the user asked for (same label or clear context), answer directly and confidently. Example: user asks 'what is my roll number' and notes contain 'Student Roll Number: 77' → respond 'Your student roll number is 77.'\n"
            "- Only use the hedging response ('I don't have that saved. I do have X — is that what you meant?') when the notes contain a NUMBER or VALUE but its context does NOT match what was asked. Example: user asks 'roll number' but only vehicle numbers are saved.\n"
            "- Never hedge when you have an exact or near-exact context match.\n\n"
            "CRITICAL RULE: When answering questions, summarizing, or sorting stored data, ALWAYS use THESE fresh database entries as the sole source of truth. IGNORE older values that might appear in the conversation history! \n\n"
            f"DATABASE NOTES:\n{kb_context}"
        )

    # Replace the last user message (already saved) with current
    if not history_msgs or history_msgs[-1]["role"] != "user":
        history_msgs.append({"role": "user", "content": parsed.raw})

    return await complete(messages=history_msgs, system=system)
