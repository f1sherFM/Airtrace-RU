import httpx
import pytest

from tests.test_stage5_ssr_rendering import (
    _current_payload,
    _history_payload,
    _trends_payload,
    web_app,
)


@pytest.mark.asyncio
async def test_stage5_city_forecast_toggle_requests_selected_horizon():
    original_current = web_app.air_service.get_current_data
    original_forecast = web_app.air_service.get_forecast_data
    original_history = web_app.air_service.get_history_data
    original_trends = web_app.air_service.get_trends_data
    original_health = web_app.air_service.check_health
    requested_hours: list[int] = []

    async def _fake_current(lat: float, lon: float):
        return _current_payload(lat, lon)

    async def _fake_forecast(lat: float, lon: float, hours: int = 24):
        requested_hours.append(hours)
        return [_current_payload(lat, lon, source="forecast")] * hours

    async def _fake_history(**kwargs):
        return _history_payload()

    async def _fake_trends(**kwargs):
        return _trends_payload()

    async def _fake_health():
        return {"status": "healthy", "public_status": "healthy", "reachable": True}

    web_app.air_service.get_current_data = _fake_current
    web_app.air_service.get_forecast_data = _fake_forecast
    web_app.air_service.get_history_data = _fake_history
    web_app.air_service.get_trends_data = _fake_trends
    web_app.air_service.check_health = _fake_health
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            city_response = await client.get("/city/moscow?forecast_hours=48")
            custom_response = await client.get("/custom?lat=55.7558&lon=37.6176&city_name=Moscow&forecast_hours=48")
    finally:
        web_app.air_service.get_current_data = original_current
        web_app.air_service.get_forecast_data = original_forecast
        web_app.air_service.get_history_data = original_history
        web_app.air_service.get_trends_data = original_trends
        web_app.air_service.check_health = original_health

    assert city_response.status_code == 200
    assert custom_response.status_code == 200
    assert requested_hours == [48, 48]
    assert 'forecast_hours=24#forecast-section' in city_response.text
    assert 'forecast_hours=48#forecast-section' in city_response.text
