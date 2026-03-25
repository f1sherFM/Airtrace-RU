import pytest

from config import DatabaseConfig
from infrastructure.db.runtime import close_database_runtime, initialize_database_runtime
from infrastructure.db.session import normalize_sync_database_url


def test_stage2_database_config_defaults_to_memory_backend():
    config = DatabaseConfig()

    assert config.history_backend in {"memory", "database"}
    assert config.enabled is False


def test_stage2_database_config_enables_database_backend_with_url():
    config = DatabaseConfig(url="sqlite+aiosqlite:///./stage2.db", history_backend="database")

    assert config.enabled is True
    assert config.alembic_url == "sqlite+aiosqlite:///./stage2.db"


def test_stage2_normalize_sync_database_url_for_alembic():
    assert normalize_sync_database_url("postgresql+asyncpg://user:pass@db/app") == "postgresql://user:pass@db/app"
    assert normalize_sync_database_url("sqlite+aiosqlite:///./test.db") == "sqlite:///./test.db"


@pytest.mark.asyncio
async def test_stage2_database_runtime_initializes_and_closes():
    runtime = initialize_database_runtime(
        DatabaseConfig(
            url="sqlite+aiosqlite:///./stage2_runtime_test.db",
            history_backend="database",
        )
    )

    assert runtime.config.enabled is True
    assert runtime.engine is not None

    await close_database_runtime()
