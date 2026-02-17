"""
Contract tests for API v2 namespace and normalized health schema (Issue #3).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import main
from history_ingestion import InMemoryHistoricalSnapshotStore
from schemas import AQIInfo, AirQualityData, DataSource, HistoricalSnapshotRecord, HistoryFreshness, LocationInfo, PollutantData


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
async def test_v2_current_and_forecast_operational():
    with patch.object(main.unified_weather_service, "get_current_combined_data", AsyncMock(return_value=_sample_air_quality())), patch.object(
        main.unified_weather_service, "get_forecast_combined_data", AsyncMock(return_value=[_sample_air_quality()])
    ):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            current = await client.get("/v2/current?lat=55.7558&lon=37.6176")
            forecast = await client.get("/v2/forecast?lat=55.7558&lon=37.6176")
            assert current.status_code == 200
            assert forecast.status_code == 200
            assert "metadata" in current.json()
            assert isinstance(forecast.json(), list)


@pytest.mark.asyncio
async def test_v2_forecast_supports_custom_hours():
    sample = _sample_air_quality()
    with patch.object(main.unified_weather_service, "get_forecast_combined_data", AsyncMock(return_value=[sample] * 72)) as forecast_mock:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v2/forecast?lat=55.7558&lon=37.6176&hours=72")
            assert response.status_code == 200
            payload = response.json()
            assert isinstance(payload, list)
            assert len(payload) == 72
            forecast_mock.assert_awaited_with(55.7558, 37.6176, hours=72)


@pytest.mark.asyncio
async def test_v2_history_operational():
    store = InMemoryHistoricalSnapshotStore()
    store._records["k1"] = HistoricalSnapshotRecord(
        snapshot_hour_utc=datetime.now(timezone.utc),
        city_code="moscow",
        latitude=55.7558,
        longitude=37.6176,
        aqi=90,
        pollutants=PollutantData(pm2_5=30.0, pm10=55.0, no2=22.0, so2=10.0, o3=70.0),
        data_source=DataSource.LIVE,
        freshness=HistoryFreshness.FRESH,
        confidence=0.91,
    )
    main.history_snapshot_store = store
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v2/history?range=24h&page=1&page_size=10&city=moscow")
        assert response.status_code == 200
        payload = response.json()
        assert "items" in payload
        assert payload["items"][0]["metadata"]["confidence"] >= 0.0


@pytest.mark.asyncio
async def test_health_normalized_contract_v1_and_v2():
    with patch.object(main.AirQualityService, "check_external_api_health", AsyncMock(return_value="healthy")), patch.object(
        main.unified_weather_service, "check_weather_api_health", AsyncMock(return_value={"status": "healthy"})
    ), patch.object(main, "get_connection_pool_manager") as pool_manager_mock:
        pool_manager_mock.return_value.health_check_all = AsyncMock(return_value={"open_meteo": True, "weather_api": True})
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for endpoint in ("/health", "/v2/health"):
                response = await client.get(endpoint)
                assert response.status_code == 200
                payload = response.json()
                assert payload["status"] in {"healthy", "degraded", "unhealthy"}
                assert isinstance(payload["services"], dict)
                for comp in payload["services"].values():
                    assert isinstance(comp, dict)
                    assert comp.get("status") in {"healthy", "degraded", "unhealthy"}


@pytest.mark.asyncio
async def test_v1_v2_namespace_response_compatibility():
    with patch.object(main.unified_weather_service, "get_current_combined_data", AsyncMock(return_value=_sample_air_quality())), patch.object(
        main.unified_weather_service, "get_forecast_combined_data", AsyncMock(return_value=[_sample_air_quality()])
    ):
        store = InMemoryHistoricalSnapshotStore()
        store._records["k1"] = HistoricalSnapshotRecord(
            snapshot_hour_utc=datetime.now(timezone.utc),
            city_code="moscow",
            latitude=55.7558,
            longitude=37.6176,
            aqi=90,
            pollutants=PollutantData(pm2_5=30.0, pm10=55.0, no2=22.0, so2=10.0, o3=70.0),
            data_source=DataSource.LIVE,
            freshness=HistoryFreshness.FRESH,
            confidence=0.91,
        )
        main.history_snapshot_store = store

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pairs = [
                ("/weather/current?lat=55.7558&lon=37.6176", "/v2/current?lat=55.7558&lon=37.6176"),
                ("/weather/forecast?lat=55.7558&lon=37.6176", "/v2/forecast?lat=55.7558&lon=37.6176"),
                ("/history?range=24h&page=1&page_size=10&city=moscow", "/v2/history?range=24h&page=1&page_size=10&city=moscow"),
            ]
            for legacy, v2 in pairs:
                legacy_resp = await client.get(legacy)
                v2_resp = await client.get(v2)
                assert legacy_resp.status_code == 200
                assert v2_resp.status_code == 200
                assert legacy_resp.json() == v2_resp.json()


def test_health_component_normalization_for_dict_without_status():
    component = main._normalize_health_component(
        {
            "stale_data_serving": "enabled",
            "cached_response_serving": "enabled",
            "minimal_response_generation": "enabled",
        }
    )
    assert component["status"] == "healthy"
    assert isinstance(component["details"], dict)


@pytest.mark.asyncio
async def test_health_services_have_strict_component_shape():
    with patch.object(main.AirQualityService, "check_external_api_health", AsyncMock(return_value="healthy")), patch.object(
        main.unified_weather_service, "check_weather_api_health", AsyncMock(return_value={"status": "healthy"})
    ), patch.object(main, "get_connection_pool_manager") as pool_manager_mock:
        pool_manager_mock.return_value.health_check_all = AsyncMock(return_value={"open_meteo": True, "weather_api": True})
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            payload = response.json()
            for comp in payload["services"].values():
                assert isinstance(comp, dict)
                assert set(comp.keys()) == {"status", "details"}
                assert comp["status"] in {"healthy", "degraded", "unhealthy"}
                assert isinstance(comp["details"], dict)
