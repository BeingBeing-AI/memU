import json
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, String, Text, func

from ext.store.pg_session import Base, shared_engine


class TaskModel(Base):
    __tablename__ = "tasks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    type = Column(String(64), nullable=False)
    name = Column(String(255), nullable=True)
    key = Column(String(255), nullable=True)
    status = Column(String(64), nullable=False)
    content = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    elapsed_time = Column(BigInteger, nullable=True)
    failure_count = Column(Integer, default=0, nullable=False)

    __table_args__ = (
        Index("idx_tasks_key_type", "key", "type"),
        Index("idx_tasks_status_id", "status", "id"),
    )


def create_task(
    *,
    task_type: str,
    status: str,
    name: str | None = None,
    key: str | None = None,
    content: dict[str, Any] | None = None,
    started_at: datetime | None = None,
) -> TaskModel:
    session = shared_engine.session()
    try:
        task = TaskModel(
            type=task_type,
            name=name,
            key=key,
            status=status,
            content=json.dumps(content, ensure_ascii=True) if content is not None else None,
            started_at=started_at,
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        return task
    finally:
        session.close()


def get_task_by_id(task_id: int) -> TaskModel | None:
    session = shared_engine.session()
    try:
        return session.query(TaskModel).filter(TaskModel.id == task_id).first()
    finally:
        session.close()


def update_task(
    task_id: int,
    *,
    status: str | None = None,
    content: dict[str, Any] | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    elapsed_time: int | None = None,
    increment_failure_count: bool = False,
) -> bool:
    session = shared_engine.session()
    try:
        task = session.query(TaskModel).filter(TaskModel.id == task_id).first()
        if not task:
            return False
        if status is not None:
            task.status = status
        if content is not None:
            task.content = json.dumps(content, ensure_ascii=True)
        if started_at is not None:
            task.started_at = started_at
        if completed_at is not None:
            task.completed_at = completed_at
        if elapsed_time is not None:
            task.elapsed_time = elapsed_time
        if increment_failure_count:
            task.failure_count = (task.failure_count or 0) + 1
        task.updated_at = datetime.utcnow()
        session.commit()
        return True
    finally:
        session.close()
