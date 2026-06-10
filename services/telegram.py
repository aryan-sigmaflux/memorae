"""
Telegram client service.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from telegram import Bot
from telegram.request import HTTPXRequest

from config import get_settings

logger = logging.getLogger(__name__)


class TelegramClient:
    def __init__(self):
        settings = get_settings()
        # Slightly generous timeouts + a small connection pool so a brief network
        # blip to api.telegram.org doesn't fail a send. write_timeout is larger to
        # allow uploading media (PDFs/images) via send_document/send_photo.
        request = HTTPXRequest(
            connection_pool_size=8,
            connect_timeout=10.0,
            read_timeout=30.0,
            write_timeout=60.0,
            pool_timeout=10.0,
        )
        self.bot = Bot(token=settings.telegram_bot_token, request=request)

    async def send_text(self, to: int | str, text: str) -> None:
        """Send a text message to a user."""
        await self.bot.send_message(chat_id=to, text=text)

    async def send_typing_action(self, to: int | str) -> None:
        """Send the 'typing…' indicator. Non-essential — never let it raise."""
        from telegram.constants import ChatAction
        try:
            await self.bot.send_chat_action(chat_id=to, action=ChatAction.TYPING)
        except Exception as exc:
            logger.warning("typing action failed (ignored): %s", exc)

    async def download_media(self, file_id: str) -> bytes:
        """Download media content by ID."""
        file = await self.bot.get_file(file_id)
        b = await file.download_as_bytearray()
        return bytes(b)


@lru_cache
def get_telegram_client() -> TelegramClient:
    return TelegramClient()
