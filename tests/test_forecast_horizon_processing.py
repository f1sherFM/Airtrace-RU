from datetime import datetime, timedelta, timezone

import pytest

from services import AirQualityService


@pytest.mark.asyncio
async def test_process_forecast_data_respects_requested_max_hours():
    service = AirQualityService()
    async def _no_weather(lat, lon):
        return None
    service._get_weather_data = _no_weather  # type: ignore[assignment]

    start = datetime(2026, 2, 17, 0, 0, tzinfo=timezone.utc)
    times = [(start + timedelta(hours=i)).isoformat() for i in range(72)]
    hourly = {
        "time": times,
        "pm2_5": [10.0] * 72,
        "pm10": [20.0] * 72,
        "nitrogen_dioxide": [30.0] * 72,
        "sulphur_dioxide": [5.0] * 72,
        "ozone": [40.0] * 72,
    }
    payload = {"latitude": 55.7558, "longitude": 37.6176, "hourly": hourly}

    data_48 = await service._process_forecast_data(payload, 55.7558, 37.6176, max_hours=48)
    data_72 = await service._process_forecast_data(payload, 55.7558, 37.6176, max_hours=72)

    assert len(data_48) == 48
    assert len(data_72) == 72
