import os

from sqlalchemy import (
    BigInteger,
    Column,
    String,
    Text,
    UniqueConstraint,
    DateTime,
    func,
)

from ext.store.pg_session import shared_engine, Base

env = os.getenv('ENV', 'prod')


class AppConfigModel(Base):
    __tablename__ = "app_config"

    __table_args__ = (
        UniqueConstraint('env', 'cfg_key', name='uk_env_key'),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    env = Column(String(16), nullable=False)  # test | prod
    key = Column('cfg_key', String(128), nullable=False)
    value = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    value_format = Column('value_format', String(32), nullable=True)  # e.g. string, number, json, markdown
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


def get_config_by_key(key: str) -> AppConfigModel:
    """根据环境和键获取配置"""
    session = shared_engine.session()
    try:
        return session.query(AppConfigModel).filter(
            AppConfigModel.env == env,
            AppConfigModel.key == key
        ).first()
    finally:
        session.close()


def get_value_by_key(key: str) -> str:
    model = get_config_by_key(key)
    return model.value if model else None
