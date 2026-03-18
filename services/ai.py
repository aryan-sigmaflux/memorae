"""
AI service – all completions routed through OpenRouter.
OpenRouter exposes an OpenAI-compatible API so we use the openai SDK with
a custom base_url and extra headers.
Docs: https://openrouter.ai/docs
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from config import get_settings
from services.toon import INTENT_SYSTEM, Intent, ParsedIntent, get_system_prompt, quick_parse

logger = logging.getLogger(__name__)
settings = get_settings()


def _client() -> AsyncOpenAI:
    """Build an OpenAI-compatible client pointed at OpenRouter."""
    extra_headers: dict[str, str] = {}
    if settings.openrouter_site_url:
        extra_headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_site_name:
        extra_headers["X-Title"] = settings.openrouter_site_name

    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers=extra_headers,
    )


# ── Core completion ───────────────────────────────────────────────────────────

async def complete(
    messages: list[dict],
    system: str | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """
    Send a chat completion via OpenRouter.
    messages: [{"role": "user"|"assistant", "content": str}, ...]
    Returns the assistant text.
    """
    system = system or get_system_prompt()
    max_tokens = max_tokens or settings.ai_max_tokens

    full_messages = [{"role": "system", "content": system}] + messages

    kwargs: dict[str, Any] = {
        "model": settings.ai_model,
        "max_tokens": max_tokens,
        "messages": full_messages,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await _client().chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


# ── Intent parsing ────────────────────────────────────────────────────────────

async def parse_intent(user_message: str) -> ParsedIntent:
    """
    Try a cheap regex parse first; fall back to AI for ambiguous messages.
    """
    quick = quick_parse(user_message)

    try:
        raw_json = await complete(
            messages=[{"role": "user", "content": user_message}],
            system=INTENT_SYSTEM,
            max_tokens=300,
            json_mode=True,
        )
        data = json.loads(raw_json)
        intent = Intent(data.get("intent", "chat"))
        payload = data.get("payload", {})
    except Exception as exc:
        logger.warning("Intent parse failed: %s", exc)
        intent = quick or Intent.CHAT
        payload = {}

    return ParsedIntent(intent=intent, payload=payload, raw=user_message)


# ── Structured extraction helpers ─────────────────────────────────────────────

async def extract_kb_fields(text: str) -> dict:
    """Given free text, extract title, content, and tags for a KB entry."""
    prompt = (
        "Extract a knowledge base entry from the following text. "
        'Return JSON: {"title": str, "content": str, "tags": [str]}. '
        "Be concise. Text:\n\n" + text
    )
    raw = await complete(
        messages=[{"role": "user", "content": prompt}],
        system="You are a precise information extractor. Return only valid JSON.",
        max_tokens=400,
        json_mode=True,
    )
    try:
        return json.loads(raw)
    except Exception:
        return {"title": text[:60], "content": text, "tags": []}


async def parse_datetime(natural_text: str, user_timezone: str = "UTC") -> str | None:
    """Convert natural language date/time to ISO-8601 string."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    prompt = (
        f"Current UTC time: {now}. User timezone: {user_timezone}.\n"
        f"Convert this to an ISO-8601 datetime string (with timezone offset): '{natural_text}'\n"
        'Return JSON: {"iso": "2025-01-15T09:00:00+05:30"} or {"iso": null} if unparseable.'
    )
    raw = await complete(
        messages=[{"role": "user", "content": prompt}],
        system="You are a datetime parser. Return only valid JSON.",
        max_tokens=100,
        json_mode=True,
    )
    try:
        return json.loads(raw).get("iso")
    except Exception:
        return None


async def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
    """
    Transcribe audio using Whisper via OpenRouter.
    Falls back to a direct OpenAI call if OPENAI_API_KEY is set,
    since not all OpenRouter tiers expose audio transcription.
    """
    import io
    from openai import AsyncOpenAI as DirectOpenAI

    direct_key = getattr(settings, "openai_api_key", None)
    if direct_key:
        client = DirectOpenAI(api_key=direct_key)
        model = "whisper-1"
    else:
        client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        model = "openai/whisper-1"

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.ogg"

    transcript = await client.audio.transcriptions.create(
        model=model,
        file=audio_file,
        response_format="text",
    )
    return transcript


# ── Vision helper ─────────────────────────────────────────────────────────────

async def describe_image(image_b64: str, mime_type: str = "image/jpeg") -> str:
    """Describe an image using a vision-capable model via OpenRouter."""
    response = await _client().chat.completions.create(
        model=settings.ai_model,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                    {"type": "text", "text": "Describe this image briefly and extract any visible text."},
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""