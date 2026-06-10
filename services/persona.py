"""
Persona + lightweight command detection for Memorae.

This replaces the old `toon.py`. There is no TOON format anywhere in v2 — the
agent communicates with the model exclusively via native JSON tool calling.
What remains here is just the assistant persona and a couple of cheap regex
helpers used by the media-upload flow (no LLM call needed for obvious commands).
"""
from __future__ import annotations

import re
from enum import Enum

from config import get_settings

settings = get_settings()


class Intent(str, Enum):
    """Minimal command hints used by the media-upload flow only."""
    REMEMBER = "remember"          # "save this", "remember this"
    CONFIRM_SAVE = "confirm_save"  # "yes" / "save it" after a save prompt
    UNKNOWN = "unknown"


# ── Persona system prompts ────────────────────────────────────────────────────

PERSONAS: dict[str, str] = {
    "friendly_assistant": (
        f"You are {settings.toon_name}, a warm, witty, and reliable personal assistant living inside Telegram. "
        "You help users remember things, set reminders, manage their calendar, and have thoughtful conversations. "
        "Keep replies concise (≤3 sentences unless the user asks for detail). Use plain text – no markdown. "
        "When you save something, confirm it briefly. "
        "If the user asks to 'see', 'send', or 'show' a saved image/file, include the exact "
        "(MEDIA_REF: media/...) tag in your response so the system can deliver it. "
        "When you can't do something, say so honestly and suggest an alternative."
    ),
    "professional": (
        f"You are {settings.toon_name}, a professional executive assistant. "
        "Communicate formally, be precise, and prioritise efficiency. No filler words."
    ),
    "casual": (
        f"You are {settings.toon_name}, a chill friend who helps you stay organised. "
        "Talk casually, use occasional emojis, keep it fun."
    ),
}


def get_system_prompt() -> str:
    persona = settings.toon_persona
    return PERSONAS.get(persona, PERSONAS["friendly_assistant"])


# ── Quick regex command detection (cheap, no AI call) ─────────────────────────

_REMEMBER_RE = re.compile(r"^(remember|note|save|store|write down|capture)\b", re.I)
_CONFIRM_RE = re.compile(r"^(yes|yeah|sure|yep|y|save it|do it)\s*$", re.I)


def quick_parse(text: str) -> Intent | None:
    """Detect the two commands the media flow needs, else None."""
    t = (text or "").strip()
    if _REMEMBER_RE.match(t):
        return Intent.REMEMBER
    if _CONFIRM_RE.match(t):
        return Intent.CONFIRM_SAVE
    return None
