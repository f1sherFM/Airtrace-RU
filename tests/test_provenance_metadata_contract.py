"""
Contract tests for unified provenance metadata (Issue 2.1).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import main
from history_ingestion import InMemoryHistoricalSnapshotStore
from schemas import AQIInfo, AirQualityData, LocationInfo, PollutantData
from schemas import DataSource, HistoricalSnapshotRecord, HistoryFreshness


def _sample_air_quality() -> AirQualityData:
    return AirQualityData(
        timestamp=datetime(2026, 2, 15, 18, 0, tzinfo=timezone.utc),
        location=LocationInfo(latitude=55.7558, longitude=37.6176),
        aqi=AQIInfo(value=85, category="Умеренное", color="#FFFF00", description="Качество воздуха приемлемо"),
        pollutants=PollutantData(pm2_5=25.4, pm10=45.2, no2=35.1, so2=12.3, o3=85.7),
        recommendations="Ограничить активность на улице",
        nmu_risk="medium",
        health_warnings=["Чувствительные люди должны быть осторожны"],
    )


@pytest.mark.asyncio
async def test_current_and_forecast_include_unified_metadata():
    with patch.object(main.unified_weather_service, "get_current_combined_data", AsyncMock(return_value=_sample_air_quality())), patch.object(
        main.unified_weather_service, "get_forecast_combined_data", AsyncMock(return_value=[_sample_air_quality()])
    ):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            current = await client.get("/weather/current?lat=55.7558&lon=37.6176")
            forecast = await client.get("/weather/forecast?lat=55.7558&lon=37.6176")

            assert current.status_code == 200
            assert forecast.status_code == 200

            current_payload = current.json()
            forecast_payload = forecast.json()[0]

            for payload in (current_payload, forecast_payload):
                assert "data_source" in payload
                assert "freshness" in payload
                assert "confidence" in payload
                assert "metadata" in payload
                assert "data_source" in payload["metadata"]
                assert "freshness" in payload["metadata"]
                assert "confidence" in payload["metadata"]
                assert payload["data_source"] == payload["metadata"]["data_source"]
                assert payload["freshness"] == payload["metadata"]["freshness"]
                assert payload["confidence"] == payload["metadata"]["confidence"]


@pytest.mark.asyncio
async def test_history_include_unified_metadata():
    store = InMemoryHistoricalSnapshotStore()
    store._records["k1"] = HistoricalSnapshotRecord(
        snapshot_hour_utc=datetime.now(timezone.utc),
        city_code="moscow",
        latitude=55.7558,
        longitude=37.6176,
        aqi=85,
        pollutants=PollutantData(pm2_5=25.4, pm10=45.2, no2=35.1, so2=12.3, o3=85.7),
        data_source=DataSource.LIVE,
        freshness=HistoryFreshness.FRESH,
        confidence=0.92,
    )
    main.history_snapshot_store = store

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history?range=24h&page=1&page_size=1")
        assert response.status_code == 200
        payload = response.json()
        assert "items" in payload and len(payload["items"]) == 1
        item = payload["items"][0]
        assert "data_source" in item
        assert "freshness" in item
        assert "confidence" in item
        assert "metadata" in item
        assert "data_source" in item["metadata"]
        assert "freshness" in item["metadata"]
        assert "confidence" in item["metadata"]
        assert item["data_source"] == item["metadata"]["data_source"]
        assert item["freshness"] == item["metadata"]["freshness"]
        assert item["confidence"] == item["metadata"]["confidence"]
