"""Stage 2 backfill entrypoint."""

from __future__ import annotations

import asyncio
import json

from application.services.backfill import HistoryBackfillService, UnifiedWeatherHistoricalProvider
from application.services.history_storage import HistoryPersistenceService
from config import config
from core.settings import get_cities_mapping
from history_ingestion import JsonlDeadLetterSink
from infrastructure.db import close_database_runtime, initialize_database_runtime
from infrastructure.repositories import SQLAlchemyHistoryRepository, SQLAlchemyLocationRepository


async def _run() -> None:
    if not config.database.enabled:
        raise RuntimeError("DATABASE_URL and HISTORY_STORAGE_BACKEND=database are required for backfill")

    runtime = initialize_database_runtime(config.database)
    location_repository = SQLAlchemyLocationRepository(runtime.session_factory)
    history_repository = SQLAlchemyHistoryRepository(runtime.session_factory)
    persistence_service = HistoryPersistenceService(
        location_repository=location_repository,
        history_repository=history_repository,
    )
    cities_mapping = get_cities_mapping()
    await persistence_service.bootstrap_configured_locations(cities_mapping)

    service = HistoryBackfillService(
        persistence_service=persistence_service,
        provider=UnifiedWeatherHistoricalProvider(),
        dead_letter_sink=JsonlDeadLetterSink("logs/history_backfill_dead_letter.jsonl"),
    )
    result = await service.run(cities_mapping=cities_mapping, days=30)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    await close_database_runtime()


if __name__ == "__main__":
    asyncio.run(_run())
