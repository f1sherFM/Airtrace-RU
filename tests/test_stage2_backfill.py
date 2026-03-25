from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from application.services.backfill import HistoryBackfillService
from application.services.history_storage import HistoryPersistenceService
from infrastructure.db.base import Base
from infrastructure.db.session import async_session_factory, create_async_engine_from_url
from infrastructure.repositories.sqlalchemy_history import SQLAlchemyHistoryRepository, SQLAlchemyLocationRepository
from history_ingestion import InMemoryDeadLetterSink
from schemas import AQIInfo, AirQualityData, LocationInfo, PollutantData, ResponseMetadata


class UnsupportedProvider:
    async def supports_history(self, *, lat: float, lon: float) -> bool:
        return False

    async def fetch_hourly_history(self, *, lat: float, lon: float, start_utc: datetime, end_utc: datetime):
        return []


class FakeHistoricalProvider:
    def __init__(self, items: list[AirQualityData]):
        self._items = items

    async def supports_history(self, *, lat: float, lon: float) -> bool:
        return True

    async def fetch_hourly_history(self, *, lat: float, lon: float, start_utc: datetime, end_utc: datetime):
        return list(self._items)


@pytest.fixture
async def backfill_runtime(tmp_path: Path):
    db_path = tmp_path / "stage2_backfill.db"
    engine = create_async_engine_from_url(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_session_factory(engine)
    location_repository = SQLAlchemyLocationRepository(session_factory)
    history_repository = SQLAlchemyHistoryRepository(session_factory)
    persistence_service = HistoryPersistenceService(
        location_repository=location_repository,
        history_repository=history_repository,
    )
    try:
        yield history_repository, persistence_service
    finally:
        await engine.dispose()


def _historical_item(hour_offset: int) -> AirQualityData:
    timestamp = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=hour_offset)
    return AirQualityData(
        timestamp=timestamp,
        location=LocationInfo(latitude=55.7558, longitude=37.6176),
        aqi=AQIInfo(value=80 + hour_offset, category="Умеренное", color="#FFFF00", description="Качество воздуха приемлемо"),
        pollutants=PollutantData(pm2_5=20.0, pm10=30.0, no2=15.0, so2=5.0, o3=70.0),
        recommendations="Ограничить активность на улице",
        nmu_risk="low",
        health_warnings=[],
        metadata=ResponseMetadata(
            data_source="historical",
            freshness="fresh",
            confidence=0.86,
            confidence_explanation="confidence derived from source=historical",
            fallback_used=False,
            cache_age_seconds=0,
        ),
    )


@pytest.mark.asyncio
async def test_stage2_backfill_records_unsupported_provider_without_fake_data(backfill_runtime):
    history_repository, persistence_service = backfill_runtime
    dead_letter = InMemoryDeadLetterSink()
    service = HistoryBackfillService(
        persistence_service=persistence_service,
        provider=UnsupportedProvider(),
        dead_letter_sink=dead_letter,
    )

    result = await service.run(
        cities_mapping={"moscow": {"name": "Москва", "lat": 55.7558, "lon": 37.6176}},
        days=30,
    )

    assert result.total_locations == 1
    assert result.unsupported_locations == 1
    assert result.inserted_snapshots == 0
    assert len(dead_letter.events) == 1
    assert await history_repository.count_snapshots() == 0


@pytest.mark.asyncio
async def test_stage2_backfill_is_idempotent_when_provider_has_history(backfill_runtime):
    history_repository, persistence_service = backfill_runtime
    items = [_historical_item(1), _historical_item(2)]
    service = HistoryBackfillService(
        persistence_service=persistence_service,
        provider=FakeHistoricalProvider(items),
        dead_letter_sink=InMemoryDeadLetterSink(),
    )
    cities_mapping = {"moscow": {"name": "Москва", "lat": 55.7558, "lon": 37.6176}}

    first = await service.run(cities_mapping=cities_mapping, days=30)
    second = await service.run(cities_mapping=cities_mapping, days=30)

    assert first.inserted_snapshots == 2
    assert second.inserted_snapshots == 0
    assert second.duplicate_snapshots == 2
    assert await history_repository.count_snapshots() == 2
