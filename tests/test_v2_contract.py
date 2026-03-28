"""Stage 3 contract tests for the stable readonly public API v2."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import main
from history_ingestion import InMemoryHistoricalSnapshotStore
from rate_limit_middleware import RateLimitMiddleware
from schemas import (
    AQIInfo,
    AirQualityData,
    DataSource,
    HistoricalSnapshotRecord,
    HistoryFreshness,
    LocationInfo,
    PollutantData,
)


def _sample_air_quality() -> AirQualityData:
    return AirQualityData(
        timestamp=datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc),
        location=LocationInfo(latitude=55.7558, longitude=37.6176),
        aqi=AQIInfo(value=85, category="Умеренное", color="#FFFF00", description="Качество воздуха приемлемо"),
        pollutants=PollutantData(pm2_5=25.4, pm10=45.2, no2=35.1, so2=12.3, o3=85.7),
        recommendations="Ограничить активность на улице",
        nmu_risk="medium",
        health_warnings=["Чувствительным группам стоит быть осторожнее"],
    )


def _seed_store() -> InMemoryHistoricalSnapshotStore:
    store = InMemoryHistoricalSnapshotStore()
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    records = [
        HistoricalSnapshotRecord(
            snapshot_hour_utc=now - timedelta(hours=1),
            city_code="moscow",
            latitude=55.7558,
            longitude=37.6176,
            aqi=82,
            pollutants=PollutantData(pm2_5=21.0, pm10=38.0, no2=24.0, so2=9.0, o3=70.0),
            data_source=DataSource.LIVE,
            freshness=HistoryFreshness.FRESH,
            confidence=0.93,
        ),
        HistoricalSnapshotRecord(
            snapshot_hour_utc=now - timedelta(days=1, hours=2),
            city_code="moscow",
            latitude=55.7558,
            longitude=37.6176,
            aqi=95,
            pollutants=PollutantData(pm2_5=28.0, pm10=44.0, no2=30.0, so2=11.0, o3=79.0),
            data_source=DataSource.HISTORICAL,
            freshness=HistoryFreshness.STALE,
            confidence=0.87,
        ),
        HistoricalSnapshotRecord(
            snapshot_hour_utc=now - timedelta(days=2, hours=4),
            city_code="moscow",
            latitude=55.7558,
            longitude=37.6176,
            aqi=62,
            pollutants=PollutantData(pm2_5=14.2, pm10=25.1, no2=18.4, so2=8.8, o3=56.3),
            data_source=DataSource.FALLBACK,
            freshness=HistoryFreshness.EXPIRED,
            confidence=0.50,
        ),
    ]
    for idx, record in enumerate(records, start=1):
        store._records[f"k{idx}"] = record
    return store


@pytest.mark.asyncio
async def test_v2_current_forecast_and_health_include_version_headers_and_metadata():
    sample = _sample_air_quality()
    with patch.object(main.unified_weather_service, "get_current_combined_data", AsyncMock(return_value=sample)), patch.object(
        main.unified_weather_service, "get_forecast_combined_data", AsyncMock(return_value=[sample])
    ), patch.object(main.AirQualityService, "check_external_api_health", AsyncMock(return_value="healthy")), patch.object(
        main.unified_weather_service, "check_weather_api_health", AsyncMock(return_value={"status": "healthy"})
    ), patch.object(main, "get_connection_pool_manager") as pool_manager_mock:
        pool_manager_mock.return_value.health_check_all = AsyncMock(return_value={"open_meteo": True, "weather_api": True})
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            current = await client.get("/v2/current?lat=55.7558&lon=37.6176")
            forecast = await client.get("/v2/forecast?lat=55.7558&lon=37.6176&hours=24")
            health = await client.get("/v2/health")

    for response in (current, forecast, health):
        assert response.headers["x-airtrace-api-version"] == "2"
        assert response.headers["x-airtrace-api-contract"] == "readonly"

    current_payload = current.json()
    assert current.status_code == 200
    assert "metadata" in current_payload
    assert current_payload["metadata"]["confidence"] >= 0.0

    forecast_payload = forecast.json()
    assert forecast.status_code == 200
    assert isinstance(forecast_payload, list)
    assert forecast_payload[0]["metadata"]["data_source"]

    health_payload = health.json()
    assert health.status_code == 200
    assert health_payload["status"] in {"healthy", "degraded", "unhealthy"}
    assert health_payload["public_status"] in {"healthy", "degraded", "unhealthy"}


@pytest.mark.asyncio
async def test_v2_history_supports_sort_and_preserves_pagination():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        desc_response = await client.get("/v2/history?range=30d&page=1&page_size=10&sort=desc&city=moscow")
        asc_response = await client.get("/v2/history?range=30d&page=1&page_size=10&sort=asc&city=moscow")

    assert desc_response.status_code == 200
    assert asc_response.status_code == 200

    desc_payload = desc_response.json()
    asc_payload = asc_response.json()
    assert desc_payload["page"] == 1
    assert desc_payload["page_size"] == 10
    assert desc_payload["total"] == 3
    assert desc_payload["items"][0]["snapshot_hour_utc"] > desc_payload["items"][-1]["snapshot_hour_utc"]
    assert asc_payload["items"][0]["snapshot_hour_utc"] < asc_payload["items"][-1]["snapshot_hour_utc"]
    assert "metadata" in asc_payload["items"][0]


@pytest.mark.asyncio
async def test_v2_trends_contract_returns_daily_points_and_summary():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v2/trends?range=7d&city=moscow")

    assert response.status_code == 200
    payload = response.json()
    assert payload["range"] == "7d"
    assert payload["aggregation"] == "day"
    assert payload["trend"] in {"improving", "stable", "worsening", "insufficient_data"}
    assert "summary" in payload
    assert "location" in payload
    assert "points" in payload
    if payload["points"]:
        point = payload["points"][0]
        assert "timestamp" in point
        assert "aqi_min" in point
        assert "aqi_max" in point
        assert "aqi_avg" in point
        assert "sample_count" in point
        assert "avg_confidence" in point
        assert "dominant_source" in point


@pytest.mark.asyncio
async def test_v2_trends_empty_state_returns_insufficient_data():
    main.history_snapshot_store = InMemoryHistoricalSnapshotStore()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v2/trends?range=7d&city=moscow")

    assert response.status_code == 200
    payload = response.json()
    assert payload["trend"] == "insufficient_data"
    assert payload["points"] == []


@pytest.mark.asyncio
async def test_v2_error_contract_for_validation_and_service_unavailable():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        validation = await client.get("/v2/current?lat=999&lon=37.6176")
        missing = await client.get("/v2/current?lat=55.7558")

    assert validation.status_code == 422
    validation_payload = validation.json()
    assert validation_payload["code"] == "VALIDATION_ERROR"
    assert "message" in validation_payload
    assert "details" in validation_payload
    assert "timestamp" in validation_payload

    assert missing.status_code == 422
    missing_payload = missing.json()
    assert missing_payload["code"] == "VALIDATION_ERROR"
    assert "details" in missing_payload

    with patch.object(main.unified_weather_service, "get_current_combined_data", AsyncMock(side_effect=ConnectionError("boom"))), patch(
        "application.queries.readonly.get_graceful_degradation_manager"
    ) as degradation_mock:
        degradation_mock.return_value.get_stale_data = AsyncMock(return_value=None)
        degradation_mock.return_value.should_prioritize_core_functionality = AsyncMock(return_value=False)
        degradation_mock.return_value.get_cached_response_for_rate_limiting = AsyncMock(return_value=None)
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            unavailable = await client.get("/v2/current?lat=55.7558&lon=37.6176")

    assert unavailable.status_code == 503
    unavailable_payload = unavailable.json()
    assert unavailable_payload["code"] == "SERVICE_UNAVAILABLE"
    assert "message" in unavailable_payload
    assert "timestamp" in unavailable_payload


@pytest.mark.asyncio
async def test_v2_forecast_service_unavailable_uses_flat_error_contract():
    with patch.object(main.unified_weather_service, "get_forecast_combined_data", AsyncMock(side_effect=ConnectionError("boom"))), patch(
        "application.queries.readonly.get_graceful_degradation_manager"
    ) as degradation_mock:
        degradation_mock.return_value.get_stale_data = AsyncMock(return_value=None)
        degradation_mock.return_value.should_prioritize_core_functionality = AsyncMock(return_value=False)
        degradation_mock.return_value.get_cached_response_for_rate_limiting = AsyncMock(return_value=None)
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            unavailable = await client.get("/v2/forecast?lat=55.7558&lon=37.6176")

    assert unavailable.status_code == 503
    payload = unavailable.json()
    assert payload["code"] == "SERVICE_UNAVAILABLE"
    assert "message" in payload
    assert "timestamp" in payload


def test_v2_rate_limit_response_uses_flat_error_contract():
    middleware = RateLimitMiddleware(app=main.app, enabled=True)
    result = type("FakeRateLimitResult", (), {"retry_after": 5, "current_usage": 100, "limit": 100, "to_headers": lambda self: {}})()
    response = middleware._create_rate_limit_response(result, "/v2/current")
    payload = response.body.decode("utf-8")
    assert "RATE_LIMIT_EXCEEDED" in payload
    assert "X-AirTrace-API-Version".lower() in {key.lower() for key in response.headers.keys()}
