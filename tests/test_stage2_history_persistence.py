from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

import main
from application.services.history_storage import HistoryPersistenceService, RepositoryBackedHistoricalSnapshotStore
from infrastructure.db.base import Base
from infrastructure.db.session import async_session_factory, create_async_engine_from_url
from infrastructure.repositories.sqlalchemy_history import SQLAlchemyHistoryRepository, SQLAlchemyLocationRepository
from history_ingestion import HistoryIngestionPipeline, InMemoryDeadLetterSink
from schemas import AQIInfo, AirQualityData, DataSource, LocationInfo, PollutantData, ResponseMetadata


@pytest.fixture
async def history_runtime(tmp_path: Path):
    db_path = tmp_path / "stage2_history_runtime.db"
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
    await persistence_service.bootstrap_configured_locations(
        {
            "moscow": {"name": "Москва", "lat": 55.7558, "lon": 37.6176},
        }
    )
    store = RepositoryBackedHistoricalSnapshotStore(
        history_repository=history_repository,
        persistence_service=persistence_service,
    )
    try:
        yield history_repository, persistence_service, store
    finally:
        await engine.dispose()


def _sample_air_quality() -> AirQualityData:
    return AirQualityData(
        timestamp=datetime.now(timezone.utc).replace(minute=15, second=0, microsecond=0),
        location=LocationInfo(latitude=55.7558, longitude=37.6176),
        aqi=AQIInfo(value=85, category="Умеренное", color="#FFFF00", description="Качество воздуха приемлемо"),
        pollutants=PollutantData(pm2_5=25.4, pm10=45.2, no2=35.1, so2=12.3, o3=85.7),
        recommendations="Ограничить активность на улице",
        nmu_risk="medium",
        health_warnings=["Чувствительные люди должны быть осторожны"],
        metadata=ResponseMetadata(
            data_source="live",
            freshness="fresh",
            confidence=0.92,
            confidence_explanation="confidence derived from source=live",
            fallback_used=False,
            cache_age_seconds=60,
        ),
    )


@pytest.mark.asyncio
async def test_stage2_pipeline_persists_snapshots_idempotently_with_db_store(history_runtime):
    history_repository, persistence_service, store = history_runtime

    async def fake_fetch(lat: float, lon: float) -> AirQualityData:
        return _sample_air_quality()

    pipeline = HistoryIngestionPipeline(
        fetch_current_data=fake_fetch,
        snapshot_store=store,
        persistence_service=persistence_service,
        dead_letter_sink=InMemoryDeadLetterSink(),
        canonical_locations=[],
        max_retries=1,
        retry_delay_seconds=0.0,
    )

    first = await pipeline.ingest_location(55.7558, 37.6176, city_code="moscow", data_source=DataSource.LIVE)
    second = await pipeline.ingest_location(55.7558, 37.6176, city_code="moscow", data_source=DataSource.LIVE)

    assert first is True
    assert second is True
    assert await history_repository.count_snapshots() == 1


@pytest.mark.asyncio
async def test_stage2_history_api_reads_from_repository_backed_store(history_runtime):
    _, persistence_service, store = history_runtime
    data = _sample_air_quality()
    await persistence_service.persist_current_observation(
        lat=55.7558,
        lon=37.6176,
        data=data,
        city_code="moscow",
    )
    main.history_snapshot_store = store

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history?range=24h&city=moscow&page=1&page_size=10")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        item = payload["items"][0]
        assert item["city_code"] == "moscow"
        assert item["metadata"]["confidence"] == item["confidence"]


@pytest.mark.asyncio
async def test_stage2_repository_backed_store_export_shape_remains_compatible(history_runtime):
    _, persistence_service, store = history_runtime
    data = _sample_air_quality()
    await persistence_service.persist_current_observation(
        lat=55.7558,
        lon=37.6176,
        data=data,
        city_code="moscow",
    )
    main.history_snapshot_store = store

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history/export/json?hours=24&city=moscow")
        assert response.status_code == 200
        payload = response.json()
        assert isinstance(payload, list)
        assert payload[0]["data_source"] == "live"
