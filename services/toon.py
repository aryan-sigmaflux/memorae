"""
Toon – the AI persona layer for Memorae.
Wraps the raw AI service with Memo's personality and provides
intent parsing, response formatting, and command routing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from config import get_settings

settings = get_settings()


class Intent(str, Enum):
    CHAT = "chat"
    REMEMBER = "remember"          # Save something to KB
    RECALL = "recall"              # Search / retrieve from KB
    REMIND = "remind"              # Set a reminder
    LIST_REMINDERS = "list_reminders"
    ADD_CALENDAR = "add_calendar"
    SHOW_CALENDAR = "show_calendar"
    FORGET = "forget"              # Delete KB entry
    UNKNOWN = "unknown"


@dataclass
class ParsedIntent:
    intent: Intent
    payload: dict[str, Any]
    raw: str


# ── Persona system prompts ────────────────────────────────────────────────────

PERSONAS: dict[str, str] = {
    "friendly_assistant": (
        f"You are {settings.toon_name}, a warm, witty, and reliable personal assistant living inside WhatsApp. "
        "You help users remember things, set reminders, manage their calendar, and have thoughtful conversations. "
        "Keep replies concise (≤3 sentences unless the user asks for detail). Use plain text – no markdown. "
        "When you save something to the knowledge base, confirm it briefly. "
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

INTENT_SYSTEM = """
You are an intent parser. Given a user message, return a JSON object with:
{
  "intent": one of [chat, remember, recall, remind, list_reminders, add_calendar, show_calendar, forget],
  "payload": {
    // for remember:  { "title": str, "content": str, "tags": [str] }
    // for recall:    { "query": str }
    // for remind:    { "title": str, "datetime_str": str, "body": str|null }
    // for add_calendar: { "title": str, "datetime_str": str, "duration_minutes": int|null }
    // for forget:    { "query": str }
    // for chat / list_reminders / show_calendar / unknown: {}
  }
}
Return only valid JSON, nothing else.
"""


def get_system_prompt() -> str:
    persona = settings.toon_persona
    return PERSONAS.get(persona, PERSONAS["friendly_assistant"])


# ── Intent parsing helpers ────────────────────────────────────────────────────

# Quick regex pre-filters (cheap, no AI call needed for obvious commands)
_REMEMBER_RE = re.compile(r"^(remember|note|save|store|write down)\b", re.I)
_RECALL_RE = re.compile(r"^(recall|what did|do you remember|find|search|look up)\b", re.I)
_REMIND_RE = re.compile(r"^(remind|set a reminder|alert me)\b", re.I)
_LIST_RE = re.compile(r"^(list|show).*(reminder|alarm|task)", re.I)
_CAL_ADD_RE = re.compile(r"^(add to (my )?calendar|schedule|book)\b", re.I)
_CAL_SHOW_RE = re.compile(r"^(show|what'?s on|check).*(calendar|schedule|agenda)", re.I)
_FORGET_RE = re.compile(r"^(forget|delete|remove)\b", re.I)


def quick_parse(text: str) -> Intent | None:
    t = text.strip()
    if _REMEMBER_RE.match(t):
        return Intent.REMEMBER
    if _RECALL_RE.match(t):
        return Intent.RECALL
    if _REMIND_RE.match(t):
        return Intent.REMIND
    if _LIST_RE.search(t):
        return Intent.LIST_REMINDERS
    if _CAL_ADD_RE.match(t):
        return Intent.ADD_CALENDAR
    if _CAL_SHOW_RE.match(t):
        return Intent.SHOW_CALENDAR
    if _FORGET_RE.match(t):
        return Intent.FORGET
    return None


def format_kb_results(results: list[dict]) -> str:
    if not results:
        return "I couldn't find anything relevant in your notes."
    lines = []
    for i, r in enumerate(results[:5], 1):
        lines.append(f"{i}. {r['title']}: {r['content'][:120]}{'…' if len(r['content']) > 120 else ''}")
    return "\n".join(lines)


def format_reminders(reminders: list[dict]) -> str:
    if not reminders:
        return "You have no upcoming reminders."
    lines = ["Your upcoming reminders:"]
    for r in reminders:
        lines.append(f"• {r['title']} – {r['remind_at']}")
    return "\n".join(lines)


def format_calendar_events(events: list[dict]) -> str:
    if not events:
        return "Nothing on your calendar for that period."
    lines = ["Your calendar:"]
    for e in events:
        start = e.get("start", {}).get("dateTime") or e.get("start", {}).get("date", "?")
        lines.append(f"• {e['summary']} at {start}")
    return "\n".join(lines)