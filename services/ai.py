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
from services.toon import (
    INTENT_SYSTEM,
    Intent,
    ParsedIntent,
    extract_recurrence,
    get_system_prompt,
    quick_parse,
)

from zoneinfo import ZoneInfo

USER_TZ_LABEL = "Asia/Kolkata (IST, UTC+5:30)"
USER_TZ = ZoneInfo("Asia/Kolkata")

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


# ── Embeddings ────────────────────────────────────────────────────────────────

async def generate_embedding(text: str) -> list[float]:
    """Generate a vector embedding for the given text via local Ollama."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    try:
        response = await client.embeddings.create(
            model="nomic-embed-text",
            input=text
        )
        return response.data[0].embedding
    except Exception as exc:
        logger.error("Ollama embedding failed: %s", exc)
        return []


# ── Intent parsing ────────────────────────────────────────────────────────────

async def parse_intent(user_message: str, history_msgs: list[dict] = None) -> ParsedIntent:
    """
    Try a cheap regex parse first; fall back to AI for ambiguous messages.
    """
    quick = quick_parse(user_message)
    # Only use quick_parse for intents that don't require AI extraction
    if quick in (Intent.LIST_REMINDERS, Intent.SHOW_CALENDAR, Intent.CONFIRM_SAVE):
        return ParsedIntent(intent=quick, payload={}, raw=user_message)

    if history_msgs:
        context = "Recent History:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in history_msgs[-3:]])
        prompt = f"{context}\n\nUser message: {user_message}"
    else:
        prompt = user_message

    try:
        from datetime import datetime
        dated_system = (
            f"Today is {datetime.now(USER_TZ).strftime('%Y-%m-%d, %A')}. "
            f"Timezone is {USER_TZ_LABEL}.\n\n" + INTENT_SYSTEM
        )
        raw_json = await complete(
            messages=[{"role": "user", "content": prompt}],
            system=dated_system,
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
        "CRITICAL FOR TITLING: Derive the 'title' from BOTH the user's label (e.g. 'sem 1 gazette') AND the document content. "
        "Format: '{user_label} - {institution} - {date}'. "
        "Example: 'Semester 1 Gazette - Thakur College - December 2023'. "
        "NEVER use a generic title when the user has explicitly named the document. "
        "IMPORTANT: If the text contains a file path tag like (LOCAL_PATH: media_bucket/...), you MUST include that exact tag in the 'content' field. Do not discard it. "
        "Be concise. Text:\n\n" + text
    )
    raw = await complete(
        messages=[{"role": "user", "content": prompt}],
        system="You are a precise information extractor. You name documents based on both user labels and provided content, ensuring specific context is preserved. Return only valid JSON.",
        max_tokens=500,
        json_mode=True,
    )
    try:
        return json.loads(raw)
    except Exception:
        return {"title": text[:60], "content": text, "tags": []}


async def parse_datetime(datetime_str: str) -> str | None:
    """Normalize a datetime string (from intent parser) and parse it with dateparser.

    Raises ValueError if no date/time could be parsed.
    """
    from datetime import datetime, timezone
    import dateparser

    if not datetime_str or not datetime_str.strip():
        raise ValueError("Empty datetime string")

    # Pre-clean recurring keywords
    _, cleaned_str = extract_recurrence(datetime_str)

    # Normalize: fix typos / expand abbreviations via a lightweight LLM call
    try:
        normalized_str = await complete(
            messages=[{
                "role": "user",
                "content": (
                    "Fix any spelling mistakes in this time expression and return "
                    f"only the corrected string, nothing else: '{cleaned_str}'"
                ),
            }],
            system="You are a spelling corrector. Return only the corrected time expression, no quotes, no explanation.",
            max_tokens=50,
        )
        normalized_str = normalized_str.strip().strip("'\"")
        if not normalized_str:
            normalized_str = cleaned_str
    except Exception:
        normalized_str = cleaned_str

    now_ist = datetime.now(USER_TZ)

    import re
    result = dateparser.parse(
        normalized_str,
        settings={
            "RELATIVE_BASE": now_ist,
            "PREFER_DATES_FROM": "future",
            "TIMEZONE": "Asia/Kolkata",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DAY_OF_MONTH": "first",
        }
    )
    
    if result is None:
        # Check if it looks like a bare time (HH:MM or 6pm or 14:30)
        # Patterns: 10:55, 10:55am, 6pm, 14:30
        if re.match(r'^(\d{1,2}:\d{2}(am|pm)?|\d{1,2}(am|pm))$', normalized_str.strip(), re.IGNORECASE):
            logger.debug("[DATEPARSER] Failed initial parse but looks like bare time, trying 'today at' fallback")
            result = dateparser.parse(
                f"today at {normalized_str}",
                settings={
                    "RELATIVE_BASE": now_ist,
                    "PREFER_DATES_FROM": "future",
                    "TIMEZONE": "Asia/Kolkata",
                    "RETURN_AS_TIMEZONE_AWARE": True,
                }
            )

    logger.debug("[DATEPARSER] input=%r normalized=%r result=%r", datetime_str, normalized_str, result)

    if result is None:
        logger.warning("[DATEPARSER] Failed to parse: %r (normalized: %r)", datetime_str, normalized_str)
        raise ValueError(f"Could not parse time from: {datetime_str!r}")

    return result.astimezone(timezone.utc).isoformat()


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