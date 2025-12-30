import os
from typing import List, Optional
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Engine,
    Index,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def _get_connection_string() -> str:
    """Read connection string from PG_URL or fall back to local default."""
    return os.getenv("PG_URL", "postgresql://root:dev123@localhost:5432/starfy")


class SharedEngine:
    """全局共享的 engine 封装类"""

    def __init__(self, connection_string: str):
        self.engine, self.session = self.init_pg_engine(connection_string)

    @staticmethod
    def init_pg_engine(connection_string: str, echo: bool = False) -> tuple[Engine, sessionmaker[Session]]:
        """
        初始化全局共享的 PostgreSQL engine

        Args:
            connection_string: PostgreSQL连接字符串，格式如：
                "postgresql://user:password@host:port/database"
            echo: 是否打印SQL语句，默认为False

        Returns:
            Engine: SQLAlchemy engine 实例
        """

        engine = create_engine(
            connection_string,
            echo=echo,
            pool_pre_ping=True,
        )
        session_local = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )
        # 创建数据库表
        Base.metadata.create_all(bind=engine)

        return engine, session_local

    def dispose(self):
        """关闭并清理 engine"""
        self.engine.dispose()


# 全局共享的 engine 实例
shared_engine: Optional[SharedEngine] = SharedEngine(
    connection_string=_get_connection_string())
