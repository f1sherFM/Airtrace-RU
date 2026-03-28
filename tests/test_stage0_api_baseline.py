from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import main
from history_ingestion import InMemoryHistoricalSnapshotStore
from schemas import AQIInfo, AirQualityData, DataSource, HistoricalSnapshotRecord, HistoryFreshness, LocationInfo, PollutantData


def _sample_air_quality() -> AirQualityData:
    return AirQualityData(
        timestamp=datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc),
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
            snapshot_hour_utc=now - timedelta(hours=5),
            city_code="moscow",
            latitude=55.7558,
            longitude=37.6176,
            aqi=77,
            pollutants=PollutantData(pm2_5=19.0, pm10=33.2, no2=21.7, so2=9.1, o3=73.4),
            data_source=DataSource.HISTORICAL,
            freshness=HistoryFreshness.STALE,
            confidence=0.89,
        ),
        HistoricalSnapshotRecord(
            snapshot_hour_utc=now - timedelta(days=10),
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
async def test_stage0_current_endpoint_preserves_core_shape():
    sample = _sample_air_quality()
    with patch.object(main.unified_weather_service, "get_current_combined_data", AsyncMock(return_value=sample)):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/weather/current?lat=55.7558&lon=37.6176")

    assert response.status_code == 200
    payload = response.json()
    assert "aqi" in payload
    assert "pollutants" in payload
    assert "metadata" in payload
    assert payload["aqi"]["category"] == "Умеренное"
    assert payload["recommendations"] == "Ограничить активность на улице"


@pytest.mark.asyncio
async def test_stage0_forecast_endpoint_preserves_list_semantics():
    sample = _sample_air_quality()
    with patch.object(main.unified_weather_service, "get_forecast_combined_data", AsyncMock(return_value=[sample] * 3)):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/weather/forecast?lat=55.7558&lon=37.6176")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 3
    assert "aqi" in payload[0]
    assert "metadata" in payload[0]


@pytest.mark.asyncio
async def test_stage0_forecast_endpoint_keeps_coordinate_validation():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/weather/forecast?lat=95&lon=37.6176")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_stage0_history_endpoint_preserves_paging_range_and_provenance_fields():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history?range=24h&city=moscow&page=1&page_size=10")

    assert response.status_code == 200
    payload = response.json()
    assert payload["range"] == "24h"
    assert payload["page"] == 1
    assert payload["page_size"] == 10
    assert "items" in payload
    assert payload["items"]
    item = payload["items"][0]
    assert "data_source" in item
    assert "freshness" in item
    assert "confidence" in item
    assert "metadata" in item


@pytest.mark.asyncio
async def test_stage0_health_and_v2_health_preserve_basic_shape():
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
                assert payload["public_status"] in {"healthy", "degraded", "unhealthy"}
                assert isinstance(payload["services"], dict)


@pytest.mark.asyncio
async def test_stage0_history_export_json_preserves_media_type_and_download_headers():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history/export/json?hours=24&city=moscow")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "attachment; filename=" in response.headers.get("content-disposition", "").lower()
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_stage0_history_export_csv_preserves_media_type_and_download_headers():
    main.history_snapshot_store = _seed_store()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/history/export/csv?hours=24&city=moscow")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=" in response.headers.get("content-disposition", "").lower()
    assert "snapshot_hour_utc" in response.text
