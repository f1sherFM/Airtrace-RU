"""Database runtime singleton for Stage 2 history storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from config import DatabaseConfig

from .session import async_session_factory, create_async_engine_from_url, resolve_database_url


@dataclass
class DatabaseRuntime:
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    config: DatabaseConfig


database_runtime: Optional[DatabaseRuntime] = None


def initialize_database_runtime(db_config: DatabaseConfig) -> DatabaseRuntime:
    global database_runtime
    database_url = resolve_database_url(db_config.url)
    engine = create_async_engine_from_url(
        database_url,
        echo=db_config.echo,
        pool_size=db_config.pool_size,
        max_overflow=db_config.max_overflow,
    )
    database_runtime = DatabaseRuntime(
        engine=engine,
        session_factory=async_session_factory(engine),
        config=db_config,
    )
    return database_runtime


async def close_database_runtime() -> None:
    global database_runtime
    if database_runtime is None:
        return
    await database_runtime.engine.dispose()
    database_runtime = None
