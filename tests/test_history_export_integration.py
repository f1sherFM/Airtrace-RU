"""
Integration tests for historical export endpoints (Issue 1.4).
"""

from datetime import datetime, timedelta, timezone

import httpx
import pytest

import main
from history_ingestion import InMemoryHistoricalSnapshotStore
from schemas import DataSource, HistoricalSnapshotRecord, HistoryFreshness, PollutantData


def _seed_store() -> InMemoryHistoricalSnapshotStore:
    store = InMemoryHistoricalSnapshotStore()
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    record = HistoricalSnapshotRecord(
        snapshot_hour_utc=now - timedelta(hours=1),
        city_code="moscow",
        latitude=55.7558,
        longitude=37.6176,
        aqi=85,
        pollutants=PollutantData(pm2_5=25.4, pm10=45.2, no2=35.1, so2=12.3, o3=85.7),
        data_source=DataSource.LIVE,
        freshness=HistoryFreshness.FRESH,
        confidence=0.92,
    )
    store._records["seed1"] = record
    return store


@pytest.mark.asyncio
async def test_history_export_json_response_headers_and_payload():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history/export/json?hours=24&city=moscow")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert "attachment; filename=" in response.headers.get("content-disposition", "").lower()
        assert response.headers.get("x-airtrace-export-type") == "historical-json"
        payload = response.json()
        assert isinstance(payload, list)
        assert payload[0]["data_source"] == "live"
        assert "freshness" in payload[0]
        assert "confidence" in payload[0]


@pytest.mark.asyncio
async def test_history_export_csv_response_headers_and_payload():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history/export/csv?hours=24&city=moscow")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment; filename=" in response.headers.get("content-disposition", "").lower()
        assert response.headers.get("x-airtrace-export-type") == "historical-csv"
        body = response.text
        assert "snapshot_hour_utc,city_code,latitude,longitude,aqi" in body
        assert "live" in body


@pytest.mark.asyncio
async def test_history_export_hours_limit_validation_error():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history/export/json?hours=721&city=moscow")
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_history_export_partial_coordinates_validation_error():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history/export/csv?hours=24&lat=55.75")
        assert response.status_code == 400
