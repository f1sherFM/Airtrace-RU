from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from infrastructure.db.base import Base
from infrastructure.db.session import async_session_factory, create_async_engine_from_url
from infrastructure.repositories.sqlalchemy_history import (
    SQLAlchemyAggregationRepository,
    SQLAlchemyHistoryRepository,
    SQLAlchemyLocationRepository,
)
from schemas import DataSource, HistoricalSnapshotRecord, HistoryFreshness, PollutantData, ResponseMetadata


@pytest.fixture
async def session_factory(tmp_path: Path):
    db_path = tmp_path / "stage2_repositories.db"
    engine = create_async_engine_from_url(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield async_session_factory(engine)
    finally:
        await engine.dispose()


def _snapshot(*, snapshot_hour_utc: datetime, aqi: int, city_code: str = "moscow") -> HistoricalSnapshotRecord:
    return HistoricalSnapshotRecord(
        snapshot_hour_utc=snapshot_hour_utc,
        city_code=city_code,
        latitude=55.7558,
        longitude=37.6176,
        aqi=aqi,
        pollutants=PollutantData(pm2_5=25.4, pm10=45.2, no2=35.1, so2=12.3, o3=85.7),
        data_source=DataSource.LIVE,
        freshness=HistoryFreshness.FRESH,
        confidence=0.92,
        metadata=ResponseMetadata(
            data_source="live",
            freshness="fresh",
            confidence=0.92,
            confidence_explanation="confidence derived from source=live",
            fallback_used=False,
            cache_age_seconds=120,
        ),
    )


@pytest.mark.asyncio
async def test_stage2_location_repository_reuses_city_and_coordinate_locations(session_factory):
    repository = SQLAlchemyLocationRepository(session_factory)

    moscow = await repository.upsert_city_location(
        city_code="moscow",
        name="Москва",
        latitude=55.7558,
        longitude=37.6176,
    )
    lookup = await repository.get_by_city_code("moscow")
    custom = await repository.get_or_create_coordinates(latitude=60.0001, longitude=30.0001)
    custom_again = await repository.get_or_create_coordinates(latitude=60.0001, longitude=30.0001)

    assert lookup == moscow
    assert custom.id == custom_again.id
    assert len(await repository.list_active_locations()) == 2


@pytest.mark.asyncio
async def test_stage2_history_repository_inserts_idempotently_and_queries_ranges(session_factory):
    location_repository = SQLAlchemyLocationRepository(session_factory)
    history_repository = SQLAlchemyHistoryRepository(session_factory)
    location = await location_repository.upsert_city_location(
        city_code="moscow",
        name="Москва",
        latitude=55.7558,
        longitude=37.6176,
    )
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    recent = _snapshot(snapshot_hour_utc=now - timedelta(hours=1), aqi=85)
    old = _snapshot(snapshot_hour_utc=now - timedelta(days=8), aqi=66)

    inserted_recent = await history_repository.insert_snapshot(location=location, record=recent, dedupe_key="recent")
    inserted_duplicate = await history_repository.insert_snapshot(location=location, record=recent, dedupe_key="recent")
    inserted_old = await history_repository.insert_snapshot(location=location, record=old, dedupe_key="old")

    day_result = await history_repository.query_snapshots(
        start_utc=now - timedelta(hours=24),
        end_utc=now,
        city_code="moscow",
        limit=10,
        offset=0,
    )
    month_result = await history_repository.query_snapshots(
        start_utc=now - timedelta(days=30),
        end_utc=now,
        city_code="moscow",
        limit=10,
        offset=0,
    )

    assert inserted_recent is True
    assert inserted_duplicate is False
    assert inserted_old is True
    assert day_result["total"] == 1
    assert month_result["total"] == 2
    assert month_result["items"][0].metadata.confidence_explanation == "confidence derived from source=live"


@pytest.mark.asyncio
async def test_stage2_aggregation_repository_returns_daily_aggregates(session_factory):
    location_repository = SQLAlchemyLocationRepository(session_factory)
    history_repository = SQLAlchemyHistoryRepository(session_factory)
    aggregation_repository = SQLAlchemyAggregationRepository(session_factory)
    location = await location_repository.upsert_city_location(
        city_code="moscow",
        name="Москва",
        latitude=55.7558,
        longitude=37.6176,
    )
    day = datetime(2026, 3, 24, 0, 0, tzinfo=timezone.utc)
    await history_repository.insert_snapshot(location=location, record=_snapshot(snapshot_hour_utc=day + timedelta(hours=1), aqi=80), dedupe_key="a")
    await history_repository.insert_snapshot(location=location, record=_snapshot(snapshot_hour_utc=day + timedelta(hours=2), aqi=100), dedupe_key="b")

    aggregates = await aggregation_repository.query_daily_aggregates(
        start_utc=day,
        end_utc=day + timedelta(days=1),
        city_code="moscow",
    )

    assert len(aggregates) == 1
    assert aggregates[0].aqi_min == 80
    assert aggregates[0].aqi_max == 100
    assert aggregates[0].sample_count == 2
