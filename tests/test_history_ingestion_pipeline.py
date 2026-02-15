"""
Unit tests for historical ingestion pipeline (Issue 1.2).
"""

import pytest
from datetime import datetime, timezone

from history_ingestion import (
    HistoryIngestionPipeline,
    InMemoryDeadLetterSink,
    InMemoryHistoricalSnapshotStore,
)
from schemas import AQIInfo, AirQualityData, LocationInfo, PollutantData


def _sample_air_quality() -> AirQualityData:
    return AirQualityData(
        timestamp=datetime(2026, 2, 15, 18, 15, tzinfo=timezone.utc),
        location=LocationInfo(latitude=55.7558, longitude=37.6176),
        aqi=AQIInfo(
            value=85,
            category="Умеренное",
            color="#FFFF00",
            description="Качество воздуха приемлемо",
        ),
        pollutants=PollutantData(pm2_5=25.4, pm10=45.2, no2=35.1, so2=12.3, o3=85.7),
        recommendations="Ограничить активность на улице",
        nmu_risk="medium",
        health_warnings=["Чувствительные люди должны быть осторожны"],
    )


@pytest.mark.asyncio
async def test_ingestion_idempotent_duplicate_snapshot():
    calls = {"count": 0}

    async def fake_fetch(lat: float, lon: float) -> AirQualityData:
        calls["count"] += 1
        return _sample_air_quality()

    store = InMemoryHistoricalSnapshotStore()
    pipeline = HistoryIngestionPipeline(
        fetch_current_data=fake_fetch,
        snapshot_store=store,
        dead_letter_sink=InMemoryDeadLetterSink(),
        max_retries=2,
        retry_delay_seconds=0.0,
        canonical_locations=[],
    )

    first = await pipeline.ingest_location(55.7558, 37.6176, city_code="moscow")
    second = await pipeline.ingest_location(55.7558, 37.6176, city_code="moscow")

    assert first is True
    assert second is True
    assert calls["count"] == 2
    assert store.count() == 1


@pytest.mark.asyncio
async def test_ingestion_retry_succeeds_before_limit():
    calls = {"count": 0}

    async def flaky_fetch(lat: float, lon: float) -> AirQualityData:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary provider failure")
        return _sample_air_quality()

    dead_letter = InMemoryDeadLetterSink()
    store = InMemoryHistoricalSnapshotStore()
    pipeline = HistoryIngestionPipeline(
        fetch_current_data=flaky_fetch,
        snapshot_store=store,
        dead_letter_sink=dead_letter,
        max_retries=3,
        retry_delay_seconds=0.0,
        canonical_locations=[],
    )

    result = await pipeline.ingest_location(59.9311, 30.3609, city_code="spb")

    assert result is True
    assert calls["count"] == 3
    assert store.count() == 1
    assert dead_letter.events == []


@pytest.mark.asyncio
async def test_ingestion_retry_exhausted_writes_dead_letter():
    calls = {"count": 0}

    async def always_fail(lat: float, lon: float) -> AirQualityData:
        calls["count"] += 1
        raise RuntimeError("permanent provider failure")

    dead_letter = InMemoryDeadLetterSink()
    store = InMemoryHistoricalSnapshotStore()
    pipeline = HistoryIngestionPipeline(
        fetch_current_data=always_fail,
        snapshot_store=store,
        dead_letter_sink=dead_letter,
        max_retries=2,
        retry_delay_seconds=0.0,
        canonical_locations=[],
    )

    result = await pipeline.ingest_location(55.0084, 82.9357, city_code="novosibirsk")

    assert result is False
    assert calls["count"] == 3  # initial attempt + 2 retries
    assert store.count() == 0
    assert len(dead_letter.events) == 1
    assert dead_letter.events[0]["city_code"] == "novosibirsk"
