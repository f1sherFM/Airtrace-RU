"""Programmatic Alembic helpers for Stage 2 runtime."""

from __future__ import annotations

import os

from alembic import command
from alembic.config import Config


def run_database_migrations(database_url: str) -> None:
    previous = os.environ.get("ALEMBIC_DATABASE_URL")
    os.environ["ALEMBIC_DATABASE_URL"] = database_url
    try:
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(config, "head")
    finally:
        if previous is None:
            os.environ.pop("ALEMBIC_DATABASE_URL", None)
        else:
            os.environ["ALEMBIC_DATABASE_URL"] = previous
