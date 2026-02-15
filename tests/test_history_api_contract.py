"""
Contract tests for /history API endpoint (Issue 1.3).
"""

from datetime import datetime, timedelta, timezone
import pytest
import httpx

import main
from history_ingestion import InMemoryHistoricalSnapshotStore
from schemas import DataSource, HistoricalSnapshotRecord, HistoryFreshness, PollutantData


def _seed_store() -> InMemoryHistoricalSnapshotStore:
    store = InMemoryHistoricalSnapshotStore()
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    r1 = HistoricalSnapshotRecord(
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
    r2 = HistoricalSnapshotRecord(
        snapshot_hour_utc=now - timedelta(hours=3),
        city_code="moscow",
        latitude=55.7558,
        longitude=37.6176,
        aqi=77,
        pollutants=PollutantData(pm2_5=19.0, pm10=33.2, no2=21.7, so2=9.1, o3=73.4),
        data_source=DataSource.HISTORICAL,
        freshness=HistoryFreshness.STALE,
        confidence=0.89,
    )
    r3 = HistoricalSnapshotRecord(
        snapshot_hour_utc=now - timedelta(days=10),
        city_code="moscow",
        latitude=55.7558,
        longitude=37.6176,
        aqi=62,
        pollutants=PollutantData(pm2_5=14.2, pm10=25.1, no2=18.4, so2=8.8, o3=56.3),
        data_source=DataSource.FALLBACK,
        freshness=HistoryFreshness.EXPIRED,
        confidence=0.50,
    )

    store._records["k1"] = r1
    store._records["k2"] = r2
    store._records["k3"] = r3
    return store


@pytest.mark.asyncio
async def test_history_contract_ranges_and_provenance_fields():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for preset in ("24h", "7d", "30d"):
            response = await client.get(f"/history?range={preset}&city=moscow&page=1&page_size=10")
            assert response.status_code == 200
            payload = response.json()
            assert payload["range"] == preset
            assert "items" in payload
            if payload["items"]:
                item = payload["items"][0]
                assert "data_source" in item
                assert "freshness" in item
                assert "confidence" in item


@pytest.mark.asyncio
async def test_history_contract_pagination():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history?range=24h&city=moscow&page=1&page_size=1")
        assert response.status_code == 200
        payload = response.json()
        assert payload["page"] == 1
        assert payload["page_size"] == 1
        assert payload["total"] == 2
        assert len(payload["items"]) == 1


@pytest.mark.asyncio
async def test_history_contract_invalid_partial_coordinates():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history?range=24h&lat=55.7")
        assert response.status_code == 400
