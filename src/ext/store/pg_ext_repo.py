from typing import List

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
    embedding = Column(VECTOR(1024), nullable=True)
    clustered = Column(Boolean, default=False, nullable=False)


class ExtStore():
    def __init__(self):
        self.engine = shared_engine.engine
        self.session_local = shared_engine.session

    def get_all_activity_items(self, user_id: int) -> List[MemoryActivityItem]:
        """获取当前用户的所有记忆活动项"""
        session = self.session_local()
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
