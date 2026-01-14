from typing import List

from pydantic import BaseModel, ConfigDict
from typing import Literal

from memu.models import MemoryItem

RetrieveQuerySource = Literal["activity_item", "memory_item", "global_memory"]

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


class ExtMemoryItem(MemoryItem, NonStrictBaseModel):
    similarity_score: float
    mentioned_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class WeightedQuery(NonStrictBaseModel):
    query: str
    weight: float = 1.0


class MultiRetrieveRequest(NonStrictBaseModel):
    user_id: str
    queries: List[WeightedQuery] | None = None
    query: str | None = None
    top_k: int = 10
    min_similarity: float = 0.4
    query_sources: List[RetrieveQuerySource] = ["activity_item", "memory_item"]


class AddMemoryItemRequest(NonStrictBaseModel):
    user_id: str
    summaries: list[str]
    resource_id: str | None = None

