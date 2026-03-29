from datetime import datetime, timedelta, timezone

import httpx
import pytest

from tests.test_stage5_ssr_rendering import (
    _current_payload,
    _history_payload,
    _trends_payload,
    web_app,
)


def _forecast_payload(lat: float, lon: float, hours: int) -> list[dict]:
    start = datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc)
    payload: list[dict] = []
    for offset in range(hours):
        item = _current_payload(lat, lon, source="forecast")
        item["timestamp"] = (start + timedelta(hours=offset)).isoformat()
        item["aqi"]["value"] = 70 + (offset % 24)
        payload.append(item)
    return payload


@pytest.mark.asyncio
async def test_stage5_forecast_renders_grouped_by_day_for_48h():
    original_current = web_app.air_service.get_current_data
    original_forecast = web_app.air_service.get_forecast_data
    original_history = web_app.air_service.get_history_data
    original_trends = web_app.air_service.get_trends_data

    async def _fake_current(lat: float, lon: float):
        return _current_payload(lat, lon)

    async def _fake_forecast(lat: float, lon: float, hours: int = 24):
        return _forecast_payload(lat, lon, hours)

    async def _fake_history(**kwargs):
        return _history_payload()

    async def _fake_trends(**kwargs):
        return _trends_payload()

    web_app.air_service.get_current_data = _fake_current
    web_app.air_service.get_forecast_data = _fake_forecast
    web_app.air_service.get_history_data = _fake_history
    web_app.air_service.get_trends_data = _fake_trends
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/city/moscow?forecast_hours=48")
    finally:
        web_app.air_service.get_current_data = original_current
        web_app.air_service.get_forecast_data = original_forecast
        web_app.air_service.get_history_data = original_history
        web_app.air_service.get_trends_data = original_trends

    assert response.status_code == 200
    assert response.text.count('data-testid="forecast-day-group"') >= 2
    assert "29 марта" in response.text
    assert "30 марта" in response.text
