from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config


def _alembic_config(database_url: str) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_stage2_alembic_upgrade_creates_history_storage_tables(tmp_path: Path):
    db_path = tmp_path / "stage2_migrations.db"
    database_url = f"sqlite+aiosqlite:///{db_path.as_posix()}"

    previous = os.environ.get("ALEMBIC_DATABASE_URL")
    os.environ["ALEMBIC_DATABASE_URL"] = database_url
    try:
        command.upgrade(_alembic_config(database_url), "head")
    finally:
        if previous is None:
            os.environ.pop("ALEMBIC_DATABASE_URL", None)
        else:
            os.environ["ALEMBIC_DATABASE_URL"] = previous

    connection = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "locations" in tables
        assert "air_quality_snapshots" in tables
        assert "data_provenance" in tables

        snapshot_indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list('air_quality_snapshots')").fetchall()
        }
        assert "ix_air_quality_snapshots_location_hour" in snapshot_indexes
    finally:
        connection.close()
