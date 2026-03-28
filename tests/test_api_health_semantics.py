from unittest.mock import AsyncMock, patch

import httpx
import pytest

import main
from tests.test_stage5_ssr_rendering import web_app


@pytest.mark.asyncio
async def test_health_exposes_engineering_and_public_status_separately():
    with patch.object(main.AirQualityService, "check_external_api_health", AsyncMock(return_value="healthy")), patch.object(
        main.unified_weather_service, "check_weather_api_health", AsyncMock(return_value={"status": "unhealthy"})
    ), patch.object(main, "get_connection_pool_manager") as pool_manager_mock:
        pool_manager_mock.return_value.health_check_all = AsyncMock(return_value={"open_meteo": True, "weather_api": True})
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v2/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["public_status"] == "healthy"


@pytest.mark.asyncio
async def test_ssr_api_health_endpoint_prefers_public_status_for_user_facing_status():
    original = web_app.air_service.check_health

    async def _fake_health():
        return {"status": "degraded", "public_status": "healthy", "reachable": True}

    web_app.air_service.check_health = _fake_health
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")
    finally:
        web_app.air_service.check_health = original

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["public_status"] == "healthy"
    assert payload["backend_api"] == "degraded"
