from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class KBEntryCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    source: str = "manual"


class KBEntryUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    tags: list[str] | None = None


class KBEntryOut(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    content: str
    tags: list[str]
    source: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KBSearchResult(KBEntryOut):
    score: float | None = None


class KBListResponse(BaseModel):
    entries: list[KBEntryOut]
    total: int