"""
Tests for daily digest summaries (Issue 5.4).
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import main
from history_ingestion import InMemoryHistoricalSnapshotStore
from schemas import DataSource, HistoricalSnapshotRecord, HistoryFreshness, PollutantData


def _seed_history_for_digest() -> InMemoryHistoricalSnapshotStore:
    store = InMemoryHistoricalSnapshotStore()
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    values = [70, 80, 95, 120, 145]
    for idx, aqi in enumerate(values):
        ts = now - timedelta(hours=len(values) - idx)
        store._records[f"k{idx}"] = HistoricalSnapshotRecord(
            snapshot_hour_utc=ts,
            city_code="moscow",
            latitude=55.7558,
            longitude=37.6176,
            aqi=aqi,
            pollutants=PollutantData(pm2_5=20.0, pm10=30.0, no2=10.0, so2=5.0, o3=40.0),
            data_source=DataSource.LIVE,
            freshness=HistoryFreshness.FRESH,
            confidence=0.9,
        )
    return store


@pytest.mark.asyncio
async def test_daily_digest_contains_trend_warnings_actions():
    main.history_snapshot_store = _seed_history_for_digest()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/alerts/digest/daily?city=moscow")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["trend"] in {"improving", "stable", "worsening"}
        assert isinstance(payload["top_warnings"], list) and len(payload["top_warnings"]) >= 1
        assert isinstance(payload["recommended_actions"], list) and len(payload["recommended_actions"]) >= 1


@pytest.mark.asyncio
async def test_daily_digest_and_deliver():
    main.history_snapshot_store = _seed_history_for_digest()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with patch.object(main.telegram_delivery_service, "send_message", AsyncMock(return_value={
            "channel": "telegram",
            "status": "sent",
            "attempts": 1,
            "event_id": "digest:moscow",
        })):
            resp = await client.get("/alerts/digest/daily-and-deliver?city=moscow&chat_id=123")
            assert resp.status_code == 200
            assert resp.json()["status"] == "sent"
