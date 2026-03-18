"""
WhatsApp client service using Meta Cloud API.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import httpx

from config import get_settings

logger = logging.getLogger(__name__)


class WhatsAppClient:
    def __init__(self):
        settings = get_settings()
        self.phone_number_id = settings.wa_phone_number_id
        self.access_token = settings.wa_access_token
        self.api_base = settings.wa_api_base
        self.base_url = f"{self.api_base}/{self.phone_number_id}"
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def send_text(self, to: str, text: str) -> dict:
        """Send a text message to a user."""
        url = f"{self.base_url}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"body": text},
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def mark_as_read(self, wa_message_id: str) -> dict:
        """Mark a message as read."""
        url = f"{self.base_url}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": wa_message_id,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def download_media(self, media_id: str) -> bytes:
        """Download media content by ID."""
        # 1. Get the media URL
        url = f"{self.api_base}/{media_id}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            media_data = response.json()
            download_url = media_data.get("url")

            if not download_url:
                raise ValueError(f"Could not find download URL for media {media_id}")

            # 2. Download the actual content
            # Media download requires the same access token but often behaves differently
            # for some reason Meta APIs are finicky about headers here.
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = await client.get(download_url, headers=headers)
            response.raise_for_status()
            return response.content


@lru_cache
def get_whatsapp_client() -> WhatsAppClient:
    return WhatsAppClient()