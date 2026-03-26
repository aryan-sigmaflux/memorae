"""
Memorae – Telegram AI assistant backend.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from db.connection import close_db, init_db
from jobs.reminders import start_scheduler, stop_scheduler
from routers import auth, webhook

logging.basicConfig(
    level=logging.DEBUG if get_settings().app_env == "development" else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Show logs from libraries if needed
# for _lib in ["httpcore","httpx","telegram.Bot","sqlalchemy.engine","urllib3",
#              "googleapiclient.discovery","google.auth.transport.requests",
#              "openai._base_client","apscheduler.scheduler","apscheduler.executors.default"]:
#     logging.getLogger(_lib).setLevel(logging.WARNING)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info("Starting Memorae (env=%s)", settings.app_env)
    await init_db()
    start_scheduler()
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────────
    stop_scheduler()
    await close_db()
    logger.info("Memorae shut down cleanly")


app = FastAPI(
    title="Memorae",
    description="Telegram AI assistant with memory, reminders, and calendar integration.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url=None,
)

# CORS (narrow in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else [settings.api_base_url],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

import os
from fastapi.staticfiles import StaticFiles

# Mount local media bucket
os.makedirs("media_bucket", exist_ok=True)
app.mount("/media", StaticFiles(directory="media_bucket"), name="media")

# Routers
app.include_router(webhook.router)
app.include_router(auth.router)


@app.get("/", tags=["health"])
async def health() -> dict:
    return {"status": "ok", "service": "memorae"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)