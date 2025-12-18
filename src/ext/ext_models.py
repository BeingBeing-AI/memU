from typing import List

from pydantic import BaseModel, ConfigDict

from memu.models import MemoryItem


class NonStrictBaseModel(BaseModel):
    # Allow coercion instead of strict type enforcement for external-facing requests.
    model_config = ConfigDict(strict=False)


# Request models for API endpoints
class ConversationMessage(NonStrictBaseModel):
    role: str
    content: str


class MemorizeRequest(NonStrictBaseModel):
    user_id: str
    external_id: str
    conversation: List[ConversationMessage]
    summary_user_profile: bool | None = True


class RetrieveRequest(NonStrictBaseModel):
    user_id: str
    query: str
    context_messages: List[ConversationMessage] | None = None
    retrieved_content: str | None = None
    retrieve_type: str = "rag"


class ExtMemoryItem(MemoryItem):
    similarity_score: float
    created_at: str
    updated_at: str


class WeightedQuery(NonStrictBaseModel):
    query: str
    weight: float = 1.0


class MultiRetrieveRequest(NonStrictBaseModel):
    user_id: str
    queries: List[WeightedQuery] = None
    query: str = None
    top_k: int = 10
    min_similarity: float = 0.3
    retrieve_type: str = "light"
