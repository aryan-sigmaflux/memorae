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
    set_pending_action,
    get_pending_action,
    clear_pending_action,
)
from services.agent import run_agent
from services.kb import remember
from services.media import process_media
from services.tools import ToolContext
from services.telegram import get_telegram_client
from telegram import Update

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/webhook", tags=["webhook"])

# Media uploads awaiting a "save this?" confirmation are stored server-side in
# the pending_actions table (kind='media_save') so they survive across workers
# and restarts — see db.queries.set_pending_action.
MEDIA_SAVE_TTL = 1800  # seconds

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

                import mimetypes
                from services import storage
                ext = mimetypes.guess_extension(media_mime) or ".bin"
                # Some audio extensions fix
                if media_mime == "audio/ogg":
                    ext = ".ogg"
                # Object key in the MinIO bucket; also stored in media_url.
                media_key = f"media/{media_id}{ext}"
                await storage.upload_bytes(media_key, media_bytes, content_type=media_mime)
                media_url = media_key

                # Immediate status update for the user
                media_label = prefix.strip("[]").lower()
                if media_label == "document": media_label = "pdf"
                await tg.send_text(to=chat_id, text=f"Reading the {media_label}... this might take a while")

                try:
                    extracted = await process_media(media_bytes, media_mime)
                except Exception as exc:
                    logger.error("Media processing OCR failed, falling back to basic extraction: %s", exc)
                    extracted = f"{prefix} image captured."

                user_text = f"{caption}\n{prefix}(MEDIA_REF: {media_key}): {extracted}".strip()
                if user_text.startswith("\n"): user_text = user_text.lstrip("\n")
                
                from services.kb import remember
                from services.persona import quick_parse, Intent as TIntent
                
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
                    await set_pending_action(
                        db, user_id, "media_save",
                        {
                            "media_url": media_url,
                            "media_type": media_mime,
                            "user_text": user_text,
                            "description": extracted,
                        },
                        ttl_seconds=MEDIA_SAVE_TTL,
                    )

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

        history = await get_recent_messages(db, conv_id, limit=8)
        history_msgs = [{"role": m["role"], "content": m["content"]} for m in history]

        # ── Pending media-save confirmation (handled before the agent) ────────
        pending_media = await get_pending_action(db, user_id, "media_save")
        if pending_media:
            from services.persona import quick_parse, Intent as TIntent
            hint = quick_parse(user_text)
            if hint in (TIntent.CONFIRM_SAVE, TIntent.REMEMBER):
                p = pending_media["payload"]
                await clear_pending_action(db, user_id)
                # On an explicit "save this as <label>", merge the user's instruction
                # with the extracted document content so the label informs the title.
                content = p["user_text"]
                if hint == TIntent.REMEMBER:
                    content = f"User instruction: {user_text}\n\nDocument content:\n{p['user_text']}"
                entry = await remember(
                    db, user_id, content,
                    media_url=p["media_url"], media_type=p["media_type"],
                )
                reply = f"Saved ✅ \"{entry['title']}\""
                logger.info("🤖 ASSISTANT: %s", reply)
                await save_message(db, conv_id, user_id, "assistant", reply)
                await tg.send_text(to=chat_id, text=reply)
                return
            # User moved on — drop the stale pending save and fall through.
            await clear_pending_action(db, user_id)

        # ── Agent loop (native tool calling, Memorae v2) ──────────────────────
        ctx = ToolContext(db=db, user_id=user_id, user=user)
        reply = await run_agent(ctx, history_msgs)
        logger.info("🤖 ASSISTANT: %s", reply)

        # ── Save assistant message & send ─────────────────────────────────────
        await save_message(db, conv_id, user_id, "assistant", reply)
        
        import io
        import re
        import mimetypes
        # The model echoes a media tag when the user wants a file delivered:
        #   (MEDIA_REF: media/<id>.<ext>)  -> fetched from MinIO
        #   (LOCAL_PATH: media_bucket/...)  -> legacy local files (pre-MinIO)
        ref_match = re.search(r'MEDIA_REF:\s*([^\s)]+)', reply)
        legacy_match = re.search(r'LOCAL_PATH:\s*(media_bucket/[^\s)]+)', reply)
        clean_reply = re.sub(r'\(?\s*(?:MEDIA_REF|LOCAL_PATH):\s*[^\s)]+\s*\)?', '', reply).strip()

        media_key: str | None = None
        media_bytes_out: bytes | None = None
        if ref_match:
            media_key = ref_match.group(1)
            from services import storage
            try:
                media_bytes_out = await storage.download_bytes(media_key)
            except Exception as e:
                logger.error("Failed to fetch media %s from MinIO: %s", media_key, e)
        elif legacy_match:
            media_key = legacy_match.group(1)
            try:
                with open(media_key, "rb") as f:
                    media_bytes_out = f.read()
            except Exception as e:
                logger.error("Failed to read legacy local media %s: %s", media_key, e)

        if media_bytes_out is not None:
            filename = media_key.rsplit("/", 1)[-1]
            mime = mimetypes.guess_type(filename)[0] or ""
            buf = io.BytesIO(media_bytes_out)
            buf.name = filename
            try:
                # Choose the right Telegram method by file type — PDFs/docs sent as
                # photos are rejected, so route non-images to send_document/video.
                if mime.startswith("image/"):
                    await tg.bot.send_photo(chat_id=chat_id, photo=buf, caption=clean_reply or None)
                elif mime.startswith("video/"):
                    await tg.bot.send_video(chat_id=chat_id, video=buf, caption=clean_reply or None)
                else:
                    await tg.bot.send_document(
                        chat_id=chat_id, document=buf, filename=filename, caption=clean_reply or None,
                    )
            except Exception as e:
                logger.error("Failed to send media %s: %s", media_key, e)
                await tg.send_text(to=chat_id, text=clean_reply or reply)
        elif reply:
            await tg.send_text(to=chat_id, text=reply)
