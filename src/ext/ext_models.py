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
    summary_user_profile: bool = True


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


class WeightedQuery(BaseModel):
    query: str
    weight: float = 1.0


class MultiRetrieveRequest(BaseModel):
    user_id: str
    queries: List[WeightedQuery]
    retrieve_type: str = "light"
