"""
Reminders job.
Runs on a fixed interval and sends WhatsApp messages for due reminders.
"""
from __future__ import annotations

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import get_settings
from db.connection import AsyncSessionLocal
from db.queries import get_due_reminders, mark_reminder_sent
from services.whatsapp import get_whatsapp_client

logger = logging.getLogger(__name__)
settings = get_settings()

_scheduler: AsyncIOScheduler | None = None


async def _dispatch_reminders() -> None:
    """Check for due reminders and send WA messages."""
    async with AsyncSessionLocal() as db:
        try:
            due = await get_due_reminders(db)
            if not due:
                return

            wa = get_whatsapp_client()
            for reminder in due:
                phone = reminder["wa_phone"]
                title = reminder["title"]
                body = reminder.get("body") or ""
                text = f"⏰ Reminder: *{title}*" + (f"\n{body}" if body else "")

                try:
                    await wa.send_text(to=phone, text=text)
                    await mark_reminder_sent(db, reminder_id=str(reminder["id"]))
                    logger.info("Sent reminder %s to %s", reminder["id"], phone)
                except Exception as exc:
                    logger.error("Failed to send reminder %s: %s", reminder["id"], exc)

            await db.commit()
        except Exception as exc:
            logger.error("Reminder dispatch error: %s", exc)
            await db.rollback()


def start_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler

    _scheduler = AsyncIOScheduler(timezone="UTC")
    _scheduler.add_job(
        _dispatch_reminders,
        trigger=IntervalTrigger(minutes=settings.reminder_check_interval_minutes),
        id="reminder_dispatch",
        replace_existing=True,
        misfire_grace_time=60,
    )
    _scheduler.start()
    logger.info(
        "Reminder scheduler started (interval: %s min)",
        settings.reminder_check_interval_minutes,
    )
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Reminder scheduler stopped")