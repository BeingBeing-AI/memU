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
            created_at = db_item.created_at.isoformat() if db_item.created_at else None
            memory_item = ExtMemoryItem(
                id=db_item.id,
                resource_id=db_item.resource_id,
                memory_type=db_item.memory_type,
                summary=db_item.summary,
                # 目前不需要返回 embedding
                embedding=None,
                similarity_score=similarity_score,
                # 先使用 created_at 作为 mentioned_at
                mentioned_at=created_at,
                created_at=created_at,
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


def add_memory_items(memory_items: List[MemoryItem], user_id: str) -> List[MemoryItem]:
    """批量添加记忆项到数据库

    Args:
        memory_items: MemoryItem 对象列表，包含要保存的记忆项信息
        user_id: 用户ID

    Returns:
        List[MemoryItem]: 保存成功后的 MemoryItem 对象列表

    Raises:
        Exception: 如果保存失败，会抛出异常
    """
    session = shared_engine.session()
    try:
        # 创建数据库模型列表
        db_items = []
        for memory_item in memory_items:
            # 生成新的ID（如果memory_item没有ID）
            item_id = memory_item.id or str(uuid.uuid4())

            db_item = MemoryItemModel(
                id=item_id,
                user_id=user_id,
                created_at=memory_item.created_at,
                resource_id=memory_item.resource_id,
                memory_type=str(memory_item.memory_type),
                summary=memory_item.summary,
                embedding=memory_item.embedding,
                is_deleted=False
            )
            db_items.append(db_item)

        # 批量插入
        session.add_all(db_items)
        session.commit()

        # 返回保存成功的 MemoryItem 对象列表
        saved_items = []
        for db_item in db_items:
            # 找到对应的原始 memory_item
            original_item = next((item for item in memory_items if item.id == db_item.id), None)
            memory_type = original_item.memory_type if original_item else db_item.memory_type

            saved_item = MemoryItem(
                id=db_item.id,
                resource_id=db_item.resource_id,
                created_at=db_item.created_at,
                memory_type=memory_type,
                summary=db_item.summary,
                embedding=db_item.embedding.tolist() if db_item.embedding is not None else []
            )
            saved_items.append(saved_item)

        return saved_items
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
