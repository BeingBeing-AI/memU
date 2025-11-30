from __future__ import annotations

import uuid
from typing import List, Optional

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    Table,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ext.store.base_repo import BaseMemoryStore
from memu.models import (
    CategoryItem,
    MemoryCategory,
    MemoryItem,
    MemoryType,
    Resource,
)


class PgMemoryCategory(MemoryCategory):
    """PgStore 专用的 MemoryCategory 子类，支持自动更新数据库"""
    _store: Optional["PgStore"] = None

    def __setattr__(self, name: str, value):
        super().__setattr__(name, value)
        # 如果是 summary 属性被修改，且有 store 实例，则自动更新数据库
        if name == "summary" and self._store is not None:
            self._store.update_category_summary(self.id, value)

VECTOR_DIMENSION = 2560

Base = declarative_base()

category_items_table = Table(
    "category_items",
    Base.metadata,
    Column("item_id", String(255), primary_key=True),
    Column("category_id", String(255), primary_key=True),
)


class MemoryResourceModel(Base):
    __tablename__ = "memory_resources"

    id = Column(String(255), primary_key=True)
    url = Column(String(512), nullable=False)
    modality = Column(String(50), nullable=False)
    local_path = Column(Text, nullable=False)
    caption = Column(Text, nullable=True)
    embedding = Column(VECTOR(VECTOR_DIMENSION), nullable=True)


class MemoryCategoryModel(Base):
    __tablename__ = "memory_categories"

    id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=False)
    embedding = Column(VECTOR(VECTOR_DIMENSION), nullable=True)
    summary = Column(Text, nullable=True)


class MemoryItemModel(Base):
    __tablename__ = "memory_items"

    id = Column(String(255), primary_key=True)
    resource_id = Column(String(255), nullable=False)
    memory_type = Column(String(50), nullable=False)
    summary = Column(Text, nullable=False)
    embedding = Column(VECTOR(VECTOR_DIMENSION), nullable=False)


class PgStore(BaseMemoryStore):
    def __init__(self, connection_string: str) -> None:
        """
        初始化PostgreSQL存储（使用pgvector）

        Args:
            connection_string: PostgreSQL连接字符串，格式如：
                "postgresql://user:password@host:port/database"
        """
        self.engine = create_engine(
            connection_string,
            echo=False,  # 设置为True可以看到SQL语句
            pool_pre_ping=True,
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self.categories = CategoriesAccessor(self)
        self._create_tables()

    def _create_tables(self) -> None:
        """创建数据库表（需要确保已安装 pgvector 扩展）"""
        Base.metadata.create_all(bind=self.engine)

    def create_resource(
            self, *, url: str, modality: str, local_path: str, caption: str = None
    ) -> Resource:
        """创建资源"""
        session = self.SessionLocal()
        try:
            resource_id = str(uuid.uuid4())
            db_resource = MemoryResourceModel(
                id=resource_id,
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
        """获取或创建记忆类别"""
        session = self.SessionLocal()
        try:
            # 首先尝试查找现有类别
            db_category = session.query(MemoryCategoryModel).filter(
                MemoryCategoryModel.name == name
            ).first()

            if db_category:
                # 如果现有类别没有embedding，更新它
                if db_category.embedding is None:
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
    ) -> MemoryItem:
        """创建记忆项"""
        session = self.SessionLocal()
        try:
            item_id = str(uuid.uuid4())
            db_item = MemoryItemModel(
                id=item_id,
                resource_id=resource_id,
                memory_type=memory_type,
                summary=summary,
                embedding=embedding,
            )

            session.add(db_item)
            session.commit()
            session.refresh(db_item)

            return MemoryItem(
                id=item_id,
                resource_id=resource_id,
                memory_type=memory_type,
                summary=summary,
                embedding=embedding,
            )
        finally:
            session.close()

    def link_item_category(self, item_id: str, cat_id: str) -> CategoryItem:
        """链接记忆项和类别"""
        session = self.SessionLocal()
        try:
            # 检查关系是否已存在
            existing = session.query(category_items_table).filter(
                category_items_table.c.item_id == item_id
            ).filter(
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

    def close(self) -> None:
        """关闭数据库连接"""
        self.engine.dispose()

    def update_category_summary(self, category_id: str, summary: str) -> bool:
        """更新类别的 summary 字段"""
        session = self.SessionLocal()
        try:
            # 更新数据库中的 summary
            updated = session.query(MemoryCategoryModel).filter(
                MemoryCategoryModel.id == category_id
            ).update({"summary": summary.strip()})
            session.commit()
            return updated > 0
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()


class CategoriesAccessor:
    """Categories 访问器，提供类似字典的接口来访问数据库中的 MemoryCategory 对象"""

    def __init__(self, store: PgStore):
        self.store = store

    def get(self, category_id: str) -> Optional[MemoryCategory]:
        """根据 ID 获取 MemoryCategory 对象，如果不存在则返回 None"""
        session = self.store.SessionLocal()
        try:
            db_category = session.query(MemoryCategoryModel).filter(
                MemoryCategoryModel.id == category_id
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
        category = self.get(category_id)
        if category is None:
            raise KeyError(f"Category with id '{category_id}' not found")
        return category

    def __contains__(self, category_id: str) -> bool:
        """支持 'in' 操作符"""
        return self.get(category_id) is not None
