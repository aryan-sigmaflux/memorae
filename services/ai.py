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
from services.persona import get_system_prompt

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


# ── Tool-calling completion (Memorae v2 agent loop) ──────────────────────────

async def complete_with_tools(
    messages: list[dict],
    tools: list[dict],
    max_tokens: int | None = None,
) -> tuple[Any, int]:
    """One round-trip of a native tool-calling completion.

    `messages` must already include the system message and full running
    transcript (user turn, prior assistant tool calls, tool results).
    Returns `(assistant_message, total_tokens_used)` so the caller can inspect
    `.tool_calls` / `.content` and enforce a per-turn token ceiling.
    """
    response = await _client().chat.completions.create(
        model=settings.ai_model,
        max_tokens=max_tokens or settings.ai_max_tokens,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 0) or 0
    return response.choices[0].message, total_tokens


# Cached tiktoken encoder for client-side token estimation (fallback when the
# provider does not return usage, which OpenRouter often omits).
_ENCODER = None


def count_tokens(text: str) -> int:
    """Best-effort token count. Falls back to a ~4-chars/token heuristic."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = False  # mark as unavailable
    if _ENCODER:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


async def _fast_json(prompt: str, system: str, max_tokens: int = 300) -> dict:
    """Call the cheap/fast model and parse a JSON object response."""
    try:
        response = await _client().chat.completions.create(
            model=settings.ai_fast_model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        logger.warning("Fast JSON call failed: %s", exc)
        return {}


async def generate_note_title(content: str) -> str:
    """Generate a 10-15 word dense, keyword-rich title optimised for retrieval."""
    data = await _fast_json(
        prompt=f"Content:\n{content}",
        system=(
            "You write a single dense, keyword-rich title (10-15 words) for a personal "
            "note, optimised so the note can be found later by semantic search. Pack in "
            "the key entities, dates, numbers, and topic words. "
            'Return JSON: {"title": str}.'
        ),
        max_tokens=80,
    )
    title = (data.get("title") or "").strip()
    return title or content[:80].strip()


async def extract_metadata(content: str) -> dict:
    """Extract structured metadata (category, entities, dates) from a note.

    Returns a JSON-serialisable dict suitable for the `metadata` JSONB column.
    Always returns a dict; on failure returns an empty one.
    """
    data = await _fast_json(
        prompt=f"Note content:\n{content}",
        system=(
            "Extract structured metadata from a personal note. "
            'Return JSON: {"category": str, "entities": [str], "dates": [str]} where '
            "category is a single lowercase label (e.g. work, health, finance, travel, "
            "personal), entities are key people/places/orgs/things mentioned, and dates "
            "are any explicit calendar dates in ISO format (YYYY-MM-DD). Use empty lists "
            "when none apply."
        ),
        max_tokens=200,
    )
    if not isinstance(data, dict):
        return {}
    category = data.get("category")
    entities = data.get("entities")
    dates = data.get("dates")
    return {
        "category": category if isinstance(category, str) else None,
        "entities": entities if isinstance(entities, list) else [],
        "dates": dates if isinstance(dates, list) else [],
    }


async def rewrite_query(query: str) -> str:
    """Rewrite a casual user query into a dense retrieval query."""
    data = await _fast_json(
        prompt=f"User query: {query}",
        system=(
            "Rewrite the user's casual question into a dense, keyword-rich search query "
            "for semantic retrieval over their personal notes. Keep any specific labels, "
            'numbers, names, and dates. Return JSON: {"query": str}.'
        ),
        max_tokens=80,
    )
    rewritten = (data.get("query") or "").strip()
    return rewritten or query


async def rerank_notes(query: str, notes: list[dict], top_n: int = 3) -> list[dict]:
    """LLM-based reranker: pick the most relevant notes for the query.

    Dependency-free stand-in for a cross-encoder. Returns at most `top_n` notes
    in relevance order. On any failure, falls back to the original order.
    """
    if len(notes) <= top_n:
        return notes

    catalogue = "\n".join(
        f'[{i}] {n.get("title", "")} :: {(n.get("content") or "")[:200]}'
        for i, n in enumerate(notes)
    )
    data = await _fast_json(
        prompt=f"Query: {query}\n\nNotes:\n{catalogue}",
        system=(
            "You are a reranker. Given a query and a list of candidate notes, return the "
            "indices of the most relevant notes, best first, excluding irrelevant ones. "
            'Return JSON: {"indices": [int, ...]} with at most '
            f"{top_n} indices."
        ),
        max_tokens=100,
    )
    indices = data.get("indices")
    if not isinstance(indices, list):
        return notes[:top_n]
    picked: list[dict] = []
    for idx in indices:
        if isinstance(idx, int) and 0 <= idx < len(notes):
            picked.append(notes[idx])
        if len(picked) >= top_n:
            break
    return picked or notes[:top_n]


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
        "IMPORTANT: If the text contains a media tag like (MEDIA_REF: media/...), you MUST include that exact tag in the 'content' field. Do not discard it. "
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