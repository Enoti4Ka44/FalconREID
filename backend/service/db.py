"""Асинхронное подключение к PostgreSQL (SQLAlchemy + asyncpg)."""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings

_engine = None
_session_factory = None


class Base(DeclarativeBase):
    pass


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db():
    """Dependency для FastAPI: yields AsyncSession."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def init_db():
    """Создание таблиц (вызывается при старте приложения)."""
    engine = get_engine()
    async with engine.begin() as conn:
        from . import models_db  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)


async def seed_db(settings):
    """Создаёт тестового пользователя если его нет."""
    from sqlalchemy import select

    from .auth import hash_password
    from .models_db import User

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(User).where(User.username == settings.seed_username)
        )
        if result.scalar_one_or_none() is None:
            user = User(
                username=settings.seed_username,
                hashed_password=hash_password(settings.seed_password),
            )
            session.add(user)
            await session.commit()
            print(f"[seed] создан пользователь: {settings.seed_username}")


async def close_db():
    """Закрытие соединения (вызывается при остановке)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
