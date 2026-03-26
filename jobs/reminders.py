"""
Reminders job.
Runs on a fixed interval and sends Telegram messages for due reminders.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import get_settings
from db.connection import AsyncSessionLocal
from db.queries import get_due_reminders, mark_reminder_sent
from services.telegram import get_telegram_client

logger = logging.getLogger(__name__)
settings = get_settings()

_scheduler: AsyncIOScheduler | None = None


async def _dispatch_reminders() -> None:
    """Check for due reminders and send Telegram messages."""
    async with AsyncSessionLocal() as db:
        try:
            due = await get_due_reminders(db)
            if not due:
                return

            tg = get_telegram_client()
            for reminder in due:
                chat_id = reminder["telegram_id"]
                title = reminder["title"]
                body = reminder.get("body") or ""
                recurrence = reminder.get("recurrence")
                text = f"⏰ Reminder: *{title}*" + (f"\n{body}" if body else "")

                try:
                    await tg.send_text(to=chat_id, text=text)
                    
                    if recurrence:
                        from db.queries import update_reminder_at
                        
                        next_at = reminder["remind_at"]
                        delta = None
                        if recurrence == "hourly":
                            delta = timedelta(hours=1)
                        elif recurrence == "daily":
                            delta = timedelta(days=1)
                        elif recurrence == "weekly":
                            delta = timedelta(weeks=1)
                        elif recurrence == "monthly":
                            delta = timedelta(days=30)
                        
                        if delta:
                            # Advance until it's in the future
                            now_utc = datetime.now(timezone.utc)
                            while next_at <= now_utc:
                                next_at += delta
                                
                            await update_reminder_at(db, reminder_id=str(reminder["id"]), next_at=next_at)
                            logger.info("Rescheduled recurring reminder %s to %s", reminder["id"], next_at)
                        else:
                            await mark_reminder_sent(db, reminder_id=str(reminder["id"]))
                            continue
                    else:
                        await mark_reminder_sent(db, reminder_id=str(reminder["id"]))
                        logger.info("Sent reminder %s to %s", reminder["id"], chat_id)
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