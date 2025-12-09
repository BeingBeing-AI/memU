from typing import List

from pydantic import BaseModel

from memu.models import MemoryItem


# Request models for API endpoints
class ConversationMessage(BaseModel):
    role: str
    content: str


class MemorizeRequest(BaseModel):
    user_id: str
    external_id: str
    conversation: List[ConversationMessage]


class RetrieveRequest(BaseModel):
    user_id: str
    query: str
    context_messages: List[ConversationMessage] | None = None
    retrieved_content: str | None = None
    retrieve_type: str = "rag"


class ExtMemoryItem(MemoryItem):
    similarity_score: float
    created_at: str
    updated_at: str
