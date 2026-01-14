from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    String,
    Table,
    Text,
)
from sqlalchemy.sql import func

from ext.ext_models import ExtMemoryItem
from ext.store.base_repo import BaseMemoryStore
from ext.store.pg_session import shared_engine, Base
from memu.models import (
    CategoryItem,
    MemoryCategory,
    MemoryCluster,
    MemoryItem,
    MemoryType,
    Resource,
)

logger = logging.getLogger(__name__)

VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))


category_items_table = Table(
    "category_items",
    Base.metadata,
    Column("item_id", String(255), primary_key=True),
    Column("category_id", String(255), primary_key=True),
)


class MemoryResourceModel(Base):
    __tablename__ = "memory_resources"

    id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    url = Column(String(512), nullable=False)
    modality = Column(String(50), nullable=False)
    local_path = Column(Text, nullable=False)
    caption = Column(Text, nullable=True)
    embedding = Column(VECTOR(VECTOR_DIMENSION), nullable=True)
    process_status = Column(String(50), default="processing", nullable=False)


class MemoryCategoryModel(Base):
    __tablename__ = "memory_categories"

    id = Column(String(255), primary_key=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    user_id = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    embedding = Column(VECTOR(VECTOR_DIMENSION), nullable=True)
    summary = Column(Text, nullable=True)

    __table_args__ = (
        Index('idx_memory_categories_user_name', 'user_id', 'name'),
    )


class MemoryItemModel(Base):
    __tablename__ = "memory_items"

    id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    resource_id = Column(String(255), nullable=False)
    memory_type = Column(String(50), nullable=False)
    summary = Column(Text, nullable=False)
    embedding = Column(VECTOR(VECTOR_DIMENSION), nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)

    __table_args__ = (
        Index('idx_memory_items_user_id', 'user_id'),
        Index('idx_memory_items_user_deleted', 'user_id', 'is_deleted'),
    )


class MemoryClusterModel(Base):
    __tablename__ = "memory_clusters"

    id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    name = Column(String(255), nullable=False)
    summary = Column(Text, nullable=False)
    embedding = Column(VECTOR(VECTOR_DIMENSION), nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)

    __table_args__ = (
        Index('idx_memory_clusters_user_name', 'user_id', 'name', postgresql_where=(is_deleted == False)),
        Index('idx_memory_clusters_user_deleted', 'user_id', 'is_deleted', postgresql_where=(is_deleted == False)),
    )


class PgMemoryCategory(MemoryCategory):
    """PgStore 专用的 MemoryCategory 子类，支持自动更新数据库"""
    _store: Optional["PgStore"] = None

    def __setattr__(self, name: str, value):
        super().__setattr__(name, value)
        # 如果是 summary 属性被修改，且有 store 实例，则自动更新数据库
        if name == "summary" and self._store is not None:
            self._store.update_category_summary(self.id, value)


class PgStore(BaseMemoryStore):
    def __init__(self, user_id: str) -> None:
        """
        初始化PostgreSQL存储（使用pgvector）
        """
        self.user_id = user_id
        self.engine = shared_engine.engine
        self.session_local = shared_engine.session
        self.categories = CategoriesAccessor(self)

    def update_resource_status(self, resource_url: str, status: str) -> bool:
        """更新资源的处理状态（仅限当前用户）"""
        if status not in ["processing", "success"]:
            raise ValueError("Invalid status. Must be one of: processing, success")

        session = self.session_local()
        try:
            # 更新数据库中的状态
            updated = session.query(MemoryResourceModel).filter(
                MemoryResourceModel.url == resource_url,
                MemoryResourceModel.user_id == self.user_id
            ).update({"process_status": status})
            session.commit()
            return updated > 0
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def get_resource_by_url(self, url: str) -> Optional[MemoryResourceModel]:
        """根据 url 查找资源（仅限当前用户）"""
        session = self.session_local()
        try:
            return session.query(MemoryResourceModel).filter(
                MemoryResourceModel.url == url,
                MemoryResourceModel.user_id == self.user_id
            ).first()
        finally:
            session.close()

    def create_resource(
            self, *, url: str, modality: str, local_path: str, caption: str = None
    ) -> Resource:
        """创建资源"""
        session = self.session_local()
        try:
            resource_id = str(uuid.uuid4())
            db_resource = MemoryResourceModel(
                id=resource_id,
                user_id=self.user_id,
                url=url,
                modality=modality,
                local_path=local_path,
                caption=caption,
            )

            session.add(db_resource)
            session.commit()
            session.refresh(db_resource)

            return Resource(
                id=resource_id,
                url=url,
                modality=modality,
                local_path=local_path,
                caption=caption,
            )
        finally:
            session.close()

    def get_or_create_category(
            self, *, name: str, description: str, embedding: List[float]
    ) -> MemoryCategory:
        """获取或创建记忆类别（仅限当前用户）"""
        session = self.session_local()
        try:
            # 首先尝试查找现有类别（仅限当前用户）
            db_category = session.query(MemoryCategoryModel).filter(
                MemoryCategoryModel.name == name,
                MemoryCategoryModel.user_id == self.user_id
            ).first()

            if db_category:
                # 如果现有类别没有embedding，更新它
                if db_category.embedding is None and embedding is not None:
                    db_category.embedding = embedding
                    session.commit()
                    session.refresh(db_category)

                category = PgMemoryCategory(
                    id=db_category.id,
                    name=db_category.name,
                    description=db_category.description,
                    embedding=(
                        db_category.embedding.tolist()
                        if db_category.embedding is not None
                        else embedding
                    ),
                    summary=db_category.summary,
                )
                # 设置 store 实例，以便在修改 summary 时自动更新数据库
                category._store = self
                return category
            else:
                # 创建新类别
                category_id = str(uuid.uuid4())
                db_category = MemoryCategoryModel(
                    id=category_id,
                    user_id=self.user_id,
                    name=name,
                    description=description,
                    embedding=embedding,
                )

                session.add(db_category)
                session.commit()
                session.refresh(db_category)

                category = PgMemoryCategory(
                    id=category_id,
                    name=name,
                    description=description,
                    embedding=embedding,
                )
                # 设置 store 实例，以便在修改 summary 时自动更新数据库
                category._store = self
                return category
        finally:
            session.close()

    def create_item(
            self,
            *,
            resource_id: str,
            memory_type: MemoryType,
            summary: str,
            embedding: List[float],
            remove_similarity: float = 0.9,
    ) -> MemoryItem:
        if remove_similarity:
            self.remove_similar_items(summary, embedding, min_similarity=remove_similarity)

        """创建记忆项"""
        session = self.session_local()
        try:
            item_id = str(uuid.uuid4())
            db_item = MemoryItemModel(
                id=item_id,
                user_id=self.user_id,
                resource_id=resource_id,
                memory_type=memory_type,
                summary=summary,
                embedding=embedding,
                is_deleted=False,
            )

            session.add(db_item)
            session.commit()
            session.refresh(db_item)

            return MemoryItem(
                id=item_id,
                resource_id=resource_id,
                created_at=datetime.now(),
                memory_type=memory_type,
                summary=summary,
                embedding=embedding,
            )
        finally:
            session.close()

    def remove_similar_items(self, summary, embedding, min_similarity):
        related_items = self.retrieve_memory_items(embedding, min_similarity=min_similarity)
        if related_items:
            removed_items_str = "\n".join([f"{item.memory_type} - {item.summary}" for item in related_items])
            logger.info(
                f"Removing {len(related_items)} similar items, new summary: {summary}\nRemoved items: {removed_items_str}")
        for related_item in related_items:
            self.remove_memory_item(related_item.id)

    def remove_memory_item(self, item_id: str) -> bool:
        """Soft delete a memory item belonging to the current user."""
        session = self.session_local()
        try:
            updated = session.query(MemoryItemModel).filter(
                MemoryItemModel.id == item_id,
                MemoryItemModel.user_id == self.user_id,
                MemoryItemModel.is_deleted.is_(False),
            ).update({"is_deleted": True})
            session.commit()
            return updated > 0
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def link_item_category(self, item_id: str, cat_id: str) -> CategoryItem:
        """链接记忆项和类别（验证item属于当前用户）"""
        session = self.session_local()
        try:
            # 首先验证item是否属于当前用户
            item_exists = session.query(MemoryItemModel).filter(
                MemoryItemModel.id == item_id,
                MemoryItemModel.user_id == self.user_id,
                MemoryItemModel.is_deleted.is_(False),
            ).first()

            if not item_exists:
                raise ValueError(f"Item {item_id} not found or does not belong to user {self.user_id}")

            # 验证category是否属于当前用户
            category_exists = session.query(MemoryCategoryModel).filter(
                MemoryCategoryModel.id == cat_id,
                MemoryCategoryModel.user_id == self.user_id
            ).first()

            if not category_exists:
                raise ValueError(f"Category {cat_id} not found or does not belong to user {self.user_id}")

            # 检查关系是否已存在
            existing = session.query(category_items_table).filter(
                category_items_table.c.item_id == item_id,
                category_items_table.c.category_id == cat_id
            ).first()

            if not existing:
                # 添加关联
                session.execute(
                    category_items_table.insert().values(
                        item_id=item_id, category_id=cat_id
                    )
                )
                session.commit()

            return CategoryItem(item_id=item_id, category_id=cat_id)
        finally:
            session.close()

    def create_clusters(
            self,
            *,
            clusters: List[dict]
    ) -> List[MemoryCluster]:
        """批量创建记忆集群（仅限当前用户）"""
        session = self.session_local()
        try:
            # 生成所有集群的ID
            cluster_data = []
            for cluster in clusters:
                cluster_id = str(uuid.uuid4())
                cluster_data.append({
                    "id": cluster_id,
                    "user_id": self.user_id,
                    "name": cluster["name"],
                    "summary": cluster["summary"],
                    "embedding": cluster["embedding"],
                    "is_deleted": False,
                })

            # 批量插入
            session.add_all([
                MemoryClusterModel(
                    id=data["id"],
                    user_id=data["user_id"],
                    name=data["name"],
                    summary=data["summary"],
                    embedding=data["embedding"],
                    is_deleted=data["is_deleted"],
                ) for data in cluster_data
            ])
            session.commit()

            # 返回创建的MemoryCluster对象列表
            created_clusters = []
            for data in cluster_data:
                created_cluster = MemoryCluster(
                    id=data["id"],
                    user_id=data["user_id"],
                    name=data["name"],
                    summary=data["summary"],
                    embedding=data["embedding"],
                    is_deleted=data["is_deleted"],
                )
                created_clusters.append(created_cluster)

            return created_clusters
        finally:
            session.close()

    def get_all_items(self) -> List[MemoryItem]:
        """获取当前用户所有记忆项（未删除的）"""
        session = self.session_local()
        try:
            results = session.query(MemoryItemModel).filter(
                MemoryItemModel.user_id == self.user_id,
                MemoryItemModel.is_deleted == False
            ).all()

            items = []
            for db_item in results:
                item = MemoryItem(
                    id=db_item.id,
                    resource_id=db_item.resource_id,
                    memory_type=db_item.memory_type,
                    summary=db_item.summary,
                    embedding=db_item.embedding.tolist() if db_item.embedding is not None else [],
                )
                items.append(item)
            return items
        finally:
            session.close()

    def get_all_clusters(self) -> List[MemoryCluster]:
        """获取当前用户的所有记忆集群（未删除的）"""
        session = self.session_local()
        try:
            results = session.query(MemoryClusterModel).filter(
                MemoryClusterModel.user_id == self.user_id,
                MemoryClusterModel.is_deleted == False
            ).all()

            clusters = []
            for db_cluster in results:
                cluster = MemoryCluster(
                    id=db_cluster.id,
                    user_id=db_cluster.user_id,
                    name=db_cluster.name,
                    summary=db_cluster.summary,
                    embedding=db_cluster.embedding.tolist() if db_cluster.embedding is not None else [],
                )
                clusters.append(cluster)

            return clusters
        finally:
            session.close()

    def remove_all_clusters(self) -> bool:
        pass

    def update_clusters_and_mark_items(
            self,
            new_clusters: List[dict],
            item_ids: List[str]
    ) -> bool:
        """
        事务性地更新聚类信息：
        1. 将原来的所有cluster设置为is_deleted=true
        2. 插入新的cluster值
        3. 将传入的MemoryItem的is_clustered=true

        Args:
            new_clusters: 新聚类的列表，每个元素包含name, summary, embedding
            item_ids: 需要标记为已聚类的记忆项ID列表

        Returns:
            bool: 操作是否成功
        """
        session = self.session_local()
        try:
            # 开始事务
            session.begin()

            # 1. 将当前用户的所有cluster标记为已删除
            session.query(MemoryClusterModel).filter(
                MemoryClusterModel.user_id == self.user_id
            ).update({"is_deleted": True})

            # 2. 插入新的cluster
            cluster_data = []
            for cluster in new_clusters:
                cluster_id = str(uuid.uuid4())
                cluster_data.append({
                    "id": cluster_id,
                    "user_id": self.user_id,
                    "name": cluster["name"],
                    "summary": cluster["summary"],
                    "embedding": cluster["embedding"],
                    "is_deleted": False,
                })

            if cluster_data:
                session.add_all([
                    MemoryClusterModel(
                        id=data["id"],
                        user_id=data["user_id"],
                        name=data["name"],
                        summary=data["summary"],
                        embedding=data["embedding"],
                        is_deleted=data["is_deleted"],
                    ) for data in cluster_data
                ])

            # 3. 将指定的记忆项标记为已聚类
            if item_ids:
                session.query(MemoryItemModel).filter(
                    MemoryItemModel.id.in_(item_ids),
                    MemoryItemModel.user_id == self.user_id,
                    MemoryItemModel.is_deleted.is_(False)
                ).update({"is_clustered": True})

            # 提交事务
            session.commit()
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"更新聚类失败: {e}")
            return False
        finally:
            session.close()

    def retrieve_memory_clusters(
            self, qvec: List[float], top_k: int = 5
    ) -> List[MemoryCluster]:
        session = self.session_local()
        try:
            # 使用pgvector的"<=>"操作符计算余弦距离，并按距离升序排列（最相似的在前）
            results = session.query(MemoryClusterModel).filter(
                MemoryClusterModel.user_id == self.user_id,
                MemoryClusterModel.is_deleted == False
            ).order_by(
                MemoryClusterModel.embedding.cosine_distance(qvec)
            ).limit(top_k).all()

            # 将数据库模型转换为MemoryCluster对象
            clusters = []
            for db_cluster in results:
                cluster = MemoryCluster(
                    id=db_cluster.id,
                    user_id=db_cluster.user_id,
                    name=db_cluster.name,
                    summary=db_cluster.summary,
                    embedding=db_cluster.embedding.tolist() if db_cluster.embedding is not None else [],
                    is_deleted=db_cluster.is_deleted,
                )
                clusters.append(cluster)

            return clusters
        finally:
            session.close()

    def close(self) -> None:
        """关闭数据库连接"""
        self.engine.dispose()

    def items(self):
        """返回当前用户的所有memory_items的迭代器"""
        session = self.session_local()
        try:
            db_items = session.query(MemoryItemModel).filter(
                MemoryItemModel.user_id == self.user_id,
                MemoryItemModel.is_deleted.is_(False),
            ).all()
            for db_item in db_items:
                item = MemoryItem(
                    id=str(db_item.id),
                    resource_id=str(db_item.resource_id),
                    memory_type=str(db_item.memory_type),
                    summary=str(db_item.summary),
                    embedding=db_item.embedding.tolist() if db_item.embedding is not None else [],
                )
                yield item
        finally:
            session.close()

    def update_category_summary(self, category_id: str, summary: str) -> bool:
        """更新类别的 summary 字段（仅限当前用户）"""
        session = self.session_local()
        try:
            # 更新数据库中的 summary
            updated = session.query(MemoryCategoryModel).filter(
                MemoryCategoryModel.id == category_id,
                MemoryCategoryModel.user_id == self.user_id
            ).update({"summary": summary.strip()})
            session.commit()
            return updated > 0
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def retrieve_memory_items(
            self, qvec: List[float], top_k: int = 10, min_similarity: float = 0.3
    ) -> List[ExtMemoryItem]:
        """
        通过pgvector实现对memory_items表中embedding的向量检索（仅限当前用户）

        Args:
            qvec: 查询的embedding向量
            top_k: 返回最相似的top_k个结果
            min_similarity: 最小相似度阈值，低于此值的结果将被过滤掉

        Returns:
            List[MemoryItem]: 最相似的记忆项列表
        """
        session = self.session_local()
        try:
            # 使用pgvector的"<=>"操作符计算余弦距离，并按距离升序排列（最相似的在前）
            # 余弦距离转为余弦相似度: similarity = 1 - distance
            query = session.query(
                MemoryItemModel,
                (1 - MemoryItemModel.embedding.cosine_distance(qvec)).label('similarity_score')
            ).filter(
                MemoryItemModel.user_id == self.user_id,
                MemoryItemModel.is_deleted.is_(False),
                (1 - MemoryItemModel.embedding.cosine_distance(qvec)) >= min_similarity
            ).order_by(
                MemoryItemModel.embedding.cosine_distance(qvec)
            )

            results = query.limit(top_k).all()

            # 将数据库模型转换为MemoryItem对象
            memory_items = []
            for db_item, similarity_score in results:
                memory_item = ExtMemoryItem(
                    id=db_item.id,
                    resource_id=db_item.resource_id,
                    memory_type=db_item.memory_type,
                    summary=db_item.summary,
                    embedding=db_item.embedding.tolist() if db_item.embedding is not None else [],
                    similarity_score=similarity_score,
                    created_at=db_item.created_at.isoformat() if db_item.created_at else None,
                    updated_at=db_item.updated_at.isoformat() if db_item.updated_at else None
                )
                memory_items.append(memory_item)

            return memory_items
        finally:
            session.close()

    def retrieve_memory_categories(
            self, qvec: List[float], top_k: int = 5
    ) -> List[MemoryCategory]:
        """
        通过pgvector实现对memorycategory表中embedding的向量检索（仅限当前用户）

        Args:
            qvec: 查询的embedding向量
            top_k: 返回最相似的top_k个结果

        Returns:
            List[MemoryCategory]: 最相似的记忆类别列表
        """
        session = self.session_local()
        try:
            # 使用pgvector的"<=>"操作符计算余弦距离，并按距离升序排列（最相似的在前）
            results = session.query(MemoryCategoryModel).filter(
                MemoryCategoryModel.user_id == self.user_id
            ).order_by(
                MemoryCategoryModel.embedding.cosine_distance(qvec)
            ).limit(top_k).all()

            # 将数据库模型转换为MemoryCategory对象
            memory_categories = []
            for db_category in results:
                memory_category = MemoryCategory(
                    id=str(db_category.id),
                    name=str(db_category.name),
                    description=str(db_category.description),
                    embedding=db_category.embedding.tolist() if db_category.embedding is not None else [],
                    summary=str(db_category.summary) if db_category.summary is not None else None,
                )
                memory_categories.append(memory_category)

            return memory_categories
        finally:
            session.close()

    def get_all_categories(self) -> List[MemoryCategory]:
        """
        获取当前用户的所有memory category

        Returns:
            List[MemoryCategory]: 所有记忆类别列表
        """
        session = self.session_local()
        try:
            # 查询当前用户的所有类别
            results = session.query(MemoryCategoryModel).filter(
                MemoryCategoryModel.user_id == self.user_id
            ).all()

            # 将数据库模型转换为MemoryCategory对象
            memory_categories = []
            for db_category in results:
                memory_category = MemoryCategory(
                    id=str(db_category.id),
                    name=str(db_category.name),
                    description=str(db_category.description),
                    embedding=db_category.embedding.tolist() if db_category.embedding is not None else [],
                    summary=str(db_category.summary) if db_category.summary is not None else None,
                )
                memory_categories.append(memory_category)

            return memory_categories
        finally:
            session.close()

    def get_category_by_name(self, category_name: str) -> Optional[MemoryCategory]:
        session = self.session_local()
        try:
            # 查询当前用户的所有类别
            result = session.query(MemoryCategoryModel).filter(
                MemoryCategoryModel.user_id == self.user_id,
                MemoryCategoryModel.name == category_name
            ).first()

            if not result:
                return None

            return MemoryCategory(
                id=str(result.id),
                name=str(result.name),
                description=str(result.description),
                embedding=result.embedding.tolist() if result.embedding is not None else [],
                summary=str(result.summary) if result.summary is not None else None,
            )
        finally:
            session.close()

    def get_unclustered_items(self) -> List[MemoryItem]:
        """
        获取当前用户所有未聚类的记忆项（is_clustered为False的项）

        Returns:
            List[MemoryItem]: 未聚类的记忆项列表
        """
        session = self.session_local()
        try:
            # 查询当前用户所有未聚类且未删除的记忆项
            results = session.query(MemoryItemModel).filter(
                MemoryItemModel.user_id == self.user_id,
                MemoryItemModel.is_deleted.is_(False),
                MemoryItemModel.is_clustered.is_(False)
            ).all()

            # 将数据库模型转换为MemoryItem对象
            memory_items = []
            for db_item in results:
                item = MemoryItem(
                    id=db_item.id,
                    resource_id=db_item.resource_id,
                    memory_type=db_item.memory_type,
                    summary=db_item.summary,
                    embedding=db_item.embedding.tolist() if db_item.embedding is not None else [],
                )
                memory_items.append(item)

            return memory_items
        finally:
            session.close()


class CategoriesAccessor:
    """Categories 访问器，提供类似字典的接口来访问数据库中的 MemoryCategory 对象"""

    def __init__(self, store: PgStore):
        self.store = store

    def get(self, category_id: str) -> Optional[MemoryCategory]:
        """根据 ID 获取 MemoryCategory 对象，如果不存在则返回 None"""
        session = self.store.session_local()
        try:
            db_category = session.query(MemoryCategoryModel).filter(
                MemoryCategoryModel.id == category_id,
                MemoryCategoryModel.user_id == self.store.user_id
            ).first()

            if db_category:
                category = PgMemoryCategory(
                    id=db_category.id,
                    name=db_category.name,
                    description=db_category.description,
                    embedding=(
                        db_category.embedding.tolist()
                        if db_category.embedding is not None
                        else []
                    ),
                    summary=db_category.summary,
                )
                # 设置 store 实例，以便在修改 summary 时自动更新数据库
                category._store = self.store
                return category
            return None
        finally:
            session.close()

    def __getitem__(self, category_id: str) -> MemoryCategory:
        """支持字典式的访问方式"""
        return self.get(category_id)

    def __contains__(self, category_id: str) -> bool:
        """支持 'in' 操作符"""
        return self.get(category_id) is not None

    def items(self):
        """支持字典的items()方法，返回(category_id, MemoryCategory)元组的迭代器"""
        session = self.store.session_local()
        try:
            db_categories = session.query(MemoryCategoryModel).filter(
                MemoryCategoryModel.user_id == self.store.user_id
            ).all()
            for db_category in db_categories:
                embedding_data = db_category.embedding.tolist() if db_category.embedding is not None else []
                category = PgMemoryCategory(
                    id=str(db_category.id),
                    name=str(db_category.name),
                    description=str(db_category.description),
                    embedding=embedding_data,
                    summary=str(db_category.summary) if db_category.summary is not None else None,
                )
                # 设置 store 实例，以便在修改 summary 时自动更新数据库
                category._store = self.store
                yield str(db_category.id), category
        finally:
            session.close()
