from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

MemoryType = Literal["profile", "event", "knowledge", "behavior", "skill"]


class Resource(BaseModel):
    id: str
    url: str
    modality: str
    local_path: str
    caption: str | None = None
    embedding: list[float] | None = None


class MemoryItem(BaseModel):
    id: str
    resource_id: str
    created_at: datetime | None
    memory_type: MemoryType
    summary: str
    embedding: list[float]

    def get_content(self) -> str:
        return self.summary


class MemoryActivityItem(BaseModel):
    id: int
    content: str
    embedding: list[float]


class MemoryCategory(BaseModel):
    id: str
    name: str
    description: str
    embedding: list[float] | None = None
    summary: str | None = None


class CategoryItem(BaseModel):
    item_id: str
    category_id: str


class MemoryCluster(BaseModel):
    id: str
    user_id: str
    name: str
    summary: str
    embedding: list[float]


class MemoryActivityItem(BaseModel):
    id: int
    user_id: int
    conversation_id: str
    session_id: str | None = None
    content: str | None = None
    mentioned_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    search_content: str | None = None
    embedding: list[float] | None = None
    clustered: bool | None = None
    similarity_score: float | None = None

    def get_content(self) -> str:
        return self.content