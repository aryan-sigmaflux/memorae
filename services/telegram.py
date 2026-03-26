"""
Telegram client service.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from telegram import Bot
from config import get_settings

logger = logging.getLogger(__name__)


class TelegramClient:
    def __init__(self):
        settings = get_settings()
        self.bot = Bot(token=settings.telegram_bot_token)

    async def send_text(self, to: int | str, text: str) -> None:
        """Send a text message to a user."""
        await self.bot.send_message(chat_id=to, text=text)

    async def send_typing_action(self, to: int | str) -> None:
        """Send typing action to a user."""
        from telegram.constants import ChatAction
        await self.bot.send_chat_action(chat_id=to, action=ChatAction.TYPING)

    async def download_media(self, file_id: str) -> bytes:
        """Download media content by ID."""
        file = await self.bot.get_file(file_id)
        b = await file.download_as_bytearray()
        return bytes(b)


@lru_cache
def get_telegram_client() -> TelegramClient:
    return TelegramClient()
