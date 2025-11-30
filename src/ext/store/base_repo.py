from memu.models import CategoryItem, MemoryCategory, MemoryItem, MemoryType, Resource
from abc import ABC, abstractmethod


class BaseMemoryStore(ABC):
    @abstractmethod
    def create_resource(self, *, url: str, modality: str, local_path: str) -> Resource:
        pass

    @abstractmethod
    def get_or_create_category(self, *, name: str, description: str, embedding: list[float]) -> MemoryCategory:
        pass

    @abstractmethod
    def create_item(
            self, *, resource_id: str, memory_type: MemoryType, summary: str, embedding: list[float]
    ) -> MemoryItem:
        pass

    @abstractmethod
    def link_item_category(self, item_id: str, cat_id: str) -> CategoryItem:
        pass
