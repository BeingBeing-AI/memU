from sqlalchemy import DateTime

from memu.models import MemoryItem


class ExtMemoryItem(MemoryItem):
    similarity_score: float
    created_at: str
    updated_at: str
