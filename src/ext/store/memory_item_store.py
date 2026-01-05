import uuid
from typing import List

from ext.ext_models import ExtMemoryItem
from ext.store.pg_repo import MemoryItemModel
from ext.store.pg_session import shared_engine
from memu.models import MemoryItem


def get_all_memory_items(user_id: str, include_embedding: bool = False) -> List[MemoryItem]:
    """获取当前用户所有记忆项（未删除的）"""
    session = shared_engine.session()
    try:
        results = session.query(MemoryItemModel).filter(
            MemoryItemModel.user_id == user_id,
            MemoryItemModel.is_deleted == False
        ).all()

        items = []
        for db_item in results:
            item = MemoryItem(
                id=db_item.id,
                resource_id=db_item.resource_id,
                created_at=db_item.created_at,
                memory_type=db_item.memory_type,
                summary=db_item.summary,
                embedding=db_item.embedding.tolist() if include_embedding and db_item.embedding is not None else [],
            )
            items.append(item)
        return items
    finally:
        session.close()


def retrieve_memory_items(
        user_id: str, qvec: List[float], top_k: int = 10, min_similarity: float = 0.3, include_embedding: bool = False
) -> List[ExtMemoryItem]:
    """
    通过pgvector实现对memory_items表中embedding的向量检索（仅限当前用户）

    Args:
        qvec: 查询的embedding向量
        top_k: 返回最相似的top_k个结果
        min_similarity: 最小相似度阈值，低于此值的结果将被过滤掉

    Returns:
        List[ExtMemoryItem]: 最相似的记忆项列表
    """
    session = shared_engine.session()
    try:
        # 使用pgvector的"<=>"操作符计算余弦距离，并按距离升序排列（最相似的在前）
        # 余弦距离转为余弦相似度: similarity = 1 - distance
        query = session.query(
            MemoryItemModel,
            (1 - MemoryItemModel.embedding.cosine_distance(qvec)).label('similarity_score')
        ).filter(
            MemoryItemModel.user_id == user_id,
            MemoryItemModel.is_deleted.is_(False),
            (1 - MemoryItemModel.embedding.cosine_distance(qvec)) >= min_similarity
        ).order_by(
            MemoryItemModel.embedding.cosine_distance(qvec)
        )

        results = query.limit(top_k).all()

        memory_items = []
        for db_item, similarity_score in results:
            memory_item = ExtMemoryItem(
                id=db_item.id,
                resource_id=db_item.resource_id,
                memory_type=db_item.memory_type,
                summary=db_item.summary,
                embedding=db_item.embedding.tolist() if include_embedding and db_item.embedding is not None else [],
                similarity_score=similarity_score,
                created_at=db_item.created_at.isoformat() if db_item.created_at else None,
                updated_at=db_item.updated_at.isoformat() if db_item.updated_at else None
            )
            memory_items.append(memory_item)

        return memory_items
    finally:
        session.close()


def update_condensation_items(user_id: str, old_ids: list[str], new_items: List[MemoryItem]) -> tuple[int, int]:
    """
    在同一个事务中执行记忆项的删除和插入操作（用于condensation）

    Args:
        user_id: 用户ID
        old_ids: 需要删除的记忆项ID列表
        new_items: 需要插入的新记忆项列表

    Returns:
        tuple: (删除的记录数, 插入的记录数)

    Raises:
        Exception: 如果操作失败，会自动回滚事务
    """
    session = shared_engine.session()
    try:
        # 1. 删除原数据（软删除）
        delete_result = session.query(MemoryItemModel).filter(
            MemoryItemModel.user_id == user_id,
            MemoryItemModel.id.in_(old_ids),
            MemoryItemModel.is_deleted == False
        ).update({
            MemoryItemModel.is_deleted: True
        }, synchronize_session=False)

        # 2. 插入新数据
        db_items = []
        for item in new_items:
            db_item = MemoryItemModel(
                id=item.id,
                user_id=user_id,
                created_at=item.created_at,
                resource_id=item.resource_id,
                memory_type=item.memory_type,
                summary=item.summary,
                embedding=item.embedding if item.embedding else None,
                is_deleted=False
            )
            db_items.append(db_item)

        session.add_all(db_items)

        # 3. 提交事务（两个操作在同一个事务中）
        session.commit()

        return delete_result, len(db_items)

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
