from typing import List

import os
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.sql import func

from ext.store.pg_session import shared_engine, Base
from memu.models import MemoryActivityItem
from ext.ext_models import ExtMemoryItem

VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))

class MemoryActivityItemModel(Base):
    __tablename__ = "memory_activity_items"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    conversation_id = Column(String(255), nullable=False)
    session_id = Column(String(255), nullable=True)
    content = Column(Text, nullable=True)
    mentioned_at = Column(Date, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    search_content = Column(TSVECTOR, nullable=True)
    embedding = Column(VECTOR(VECTOR_DIMENSION), nullable=True)
    clustered = Column(Boolean, default=False, nullable=False)


def get_all_activity_items(user_id: int) -> List[MemoryActivityItem]:
    """获取当前用户的所有记忆活动项"""
    session = shared_engine.session()
    try:
        results = session.query(MemoryActivityItemModel).filter(
            MemoryActivityItemModel.user_id == user_id
        ).all()

        activity_items: List[MemoryActivityItem] = []
        for db_item in results:
            activity_items.append(
                MemoryActivityItem(
                    id=db_item.id,
                    user_id=db_item.user_id,
                    conversation_id=db_item.conversation_id,
                    session_id=db_item.session_id,
                    content=db_item.content,
                    mentioned_at=db_item.mentioned_at,
                    created_at=db_item.created_at,
                    updated_at=db_item.updated_at,
                    search_content=db_item.search_content,
                    embedding=db_item.embedding.tolist() if db_item.embedding is not None else None,
                    clustered=db_item.clustered,
                )
            )
        return activity_items
    finally:
        session.close()


def _model_to_item(item: MemoryActivityItemModel) -> MemoryActivityItem:
    return MemoryActivityItem(
        id=item.id,
        user_id=item.user_id,
        conversation_id=item.conversation_id,
        session_id=item.session_id,
        content=item.content,
        mentioned_at=str(item.mentioned_at),
        created_at=str(item.created_at),
        updated_at=str(item.updated_at),
        search_content=item.search_content,
    )

def retrieve_activity_items_to_memory(
        user_id: int,
        qvec: List[float],
        top_k: int = 10,
        min_similarity: float = 0.3,
        include_embedding: bool = False,
) -> List[ExtMemoryItem]:
    """基于向量相似度检索当前用户的记忆活动项"""
    session = shared_engine.session()
    try:
        similarity = 1 - MemoryActivityItemModel.embedding.cosine_distance(qvec)
        query = (
            session.query(
                MemoryActivityItemModel,
                similarity.label("similarity_score"),
            )
            .filter(
                MemoryActivityItemModel.user_id == user_id,
                similarity >= min_similarity,
            )
            .order_by(MemoryActivityItemModel.embedding.cosine_distance(qvec))
        )

        results = query.limit(top_k).all()

        activity_items: List[MemoryActivityItem] = []
        for db_item, similarity_score in results:
            activity_items.append(
                ExtMemoryItem(
                    id=str(db_item.id),
                    user_id=db_item.user_id,
                    resource_id=db_item.session_id,
                    memory_type="event",
                    summary=db_item.content,
                    mentioned_at=str(db_item.mentioned_at),
                    created_at=str(db_item.created_at),
                    updated_at=str(db_item.updated_at),
                    similarity_score=similarity_score,
                )
            )
        return activity_items
    finally:
        session.close()
