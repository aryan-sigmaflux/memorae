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
    CONFIRM_SAVE = "confirm_save"  # User confirming they want an image/media saved
    CONNECT_GOOGLE = "connect_google"
    DELETE_REMINDERS = "delete_reminders"
    UNKNOWN = "unknown"


@dataclass
class ParsedIntent:
    intent: Intent
    payload: dict[str, Any]
    raw: str


# ── Persona system prompts ────────────────────────────────────────────────────

PERSONAS: dict[str, str] = {
    "friendly_assistant": (
        f"You are {settings.toon_name}, a warm, witty, and reliable personal assistant living inside Telegram. "
        "You help users remember things, set reminders, manage their calendar, and have thoughtful conversations. "
        "Keep replies concise (≤3 sentences unless the user asks for detail). Use plain text – no markdown. "
        "When you save something to the knowledge base, confirm it briefly. "
        "IMPORTANT: You CAN save images and files directly because the system automatically extracts them to text and local paths. Never tell the user you cannot save images or files. "
        "If the user asks to 'see', 'send', or 'show' a saved image/file, retrieve it from your notes and MUST include the exact (LOCAL_PATH: media_bucket/...) in your response so the system can deliver it."
        "If you are requested to send an image back, make sure you include (LOCAL_PATH: media_bucket/...) in your reply exactly as it was provided to you. "
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
You are an intent parser. Given the conversational history and a new user message, return a JSON object with:
{
  "intent": one of [chat, remember, recall, remind, list_reminders, add_calendar, show_calendar, forget, connect_google, delete_reminders],
  "payload": {
    // for remember:  { "content_to_save": "Exact text/data from the conversation to save (include any LOCAL_PATH context if applicable)" }
    // for recall:    { "query": str }
    // for remind:    { "title": str, "datetime_str": str, "body": str|null, "recurrence": "daily"|"weekly"|"hourly"|"monthly"|null }
    // for add_calendar: { "title": str, "datetime_str": str, "duration_minutes": int|null, "attendee_email": str|null }
    // for forget:    { "query": str }
    // for delete_reminders: { "query": str|null }
    // for chat / list_reminders / show_calendar / confirm_save / unknown: {}
  }
}
Return only valid JSON, nothing else. 
CRITICAL RULE: 
- For 'remind': if the user mentions recurring patterns like 'every day', 'every morning', 'every night', 'daily', 'every hour', 'weekly', etc., set 'recurrence' to the appropriate keyword ('daily', 'hourly', 'weekly') AND KEEP only the time/base-date part in 'datetime_str' (e.g. '6:07pm').
- For 'remind': ALWAYS infer a short, descriptive string for 'title' from the message context. NEVER return null for 'title'. If the message is very short, use the entire message as the title.
- For 'remind' or 'add_calendar': if the user explicitly mentions a DATE with a YEAR (e.g., 'March 20 2026'), you MUST return 'datetime_str' as a full ISO 8601 string (e.g., '2026-03-20T18:25:00'). NEVER roll over to the next day if a year is stated.
- If the user explicitly uploads a photo AND says 'save this image' or 'store this photo', it is 'remember'.
- If the conversational bot just asked the user "Want me to save this?" and the user replied "yes", "yeah", "save it", "sure", etc., map it to "confirm_save".
- If the user says 'send me the image', 'show me that photo', 'retrieve the file', it is 'recall'. 
- For 'delete_reminders': if the user says "cancel all reminders", "delete all reminders", "clear my reminders", "remove all reminders" or similar, map to delete_reminders. Set 'query' to null for "all", or a search string for specific ones.
The word 'save' in a retrieval request (e.g., 'send the file I saved') MUST be interpreted as RECALL.
"""


def get_system_prompt() -> str:
    persona = settings.toon_persona
    return PERSONAS.get(persona, PERSONAS["friendly_assistant"])


# ── Intent parsing helpers ────────────────────────────────────────────────────

# Quick regex pre-filters (cheap, no AI call needed for obvious commands)
_REMEMBER_RE = re.compile(r"^(remember|note|save|store|write down|capture)\b", re.I)
_RECALL_RE = re.compile(r"^(recall|what did|do you remember|find|search|look up|send me|show me|get|fetch|give me)\b", re.I)
_REMIND_RE = re.compile(r"^(remind|set a reminder|alert me)\b", re.I)
_LIST_RE = re.compile(r"^(list|show).*(reminder|alarm|task)", re.I)
_CAL_ADD_RE = re.compile(r"^(add to (my )?calendar|schedule|book)\b", re.I)
_CAL_SHOW_RE = re.compile(r"^(show|what'?s on|check).*(calendar|schedule|agenda)", re.I)
_FORGET_RE = re.compile(r"^(forget|delete|remove)\b", re.I)
_CONFIRM_RE = re.compile(r"^(yes|yeah|sure|yep|y|save it|do it)\s*$", re.I)
_CONNECT_GOOGLE_RE = re.compile(r"^(connect google|link my google|login with google)", re.I)

# Recurring patterns
_RECURRENCE_MAP = {
    "daily": re.compile(r"\b(every\s*day|everyday|daily|each day)\b", re.I),
    "weekly": re.compile(r"\b(every\s*week|weekly|every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", re.I),
    "hourly": re.compile(r"\b(every\s*hour|hourly)\b", re.I),
    "monthly": re.compile(r"\b(every\s*month|monthly)\b", re.I),
    "morning": re.compile(r"\b(every\s*morning)\b", re.I),
    "night": re.compile(r"\b(every\s*night)\b", re.I),
}

def extract_recurrence(text: str) -> tuple[str | None, str]:
    """
    Detect recurring keywords and return (recurrence_type, cleaned_text).
    Example: "everyday at 6pm" -> ("daily", "at 6pm")
    """
    recurrence = None
    cleaned = text
    
    for r_type, regex in _RECURRENCE_MAP.items():
        if regex.search(cleaned):
            # Special cases for night/morning to daily mapping
            if r_type in ("morning", "night"):
                recurrence = "daily"
            else:
                recurrence = r_type
            cleaned = regex.sub("", cleaned).strip()
            # Remove leftovers like "at every day" -> "at"
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            
    return recurrence, cleaned


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
    if _CONFIRM_RE.match(t):
        return Intent.CONFIRM_SAVE
    if _CONNECT_GOOGLE_RE.match(t):
        return Intent.CONNECT_GOOGLE
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