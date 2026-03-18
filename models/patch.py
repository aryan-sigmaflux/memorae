from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel


class PatchDiff(BaseModel):
    original: str | None = None
    proposed: str
    fields_changed: list[str] = []
    reason: str | None = None


class PatchCreate(BaseModel):
    kb_entry_id: UUID | None = None
    diff: PatchDiff


class PatchOut(BaseModel):
    id: UUID
    user_id: UUID
    kb_entry_id: UUID | None
    diff: PatchDiff
    status: Literal["pending", "accepted", "rejected"]
    created_at: datetime

    model_config = {"from_attributes": True}


class PatchResolve(BaseModel):
    status: Literal["accepted", "rejected"]