"""
Integration tests for alert rule API (Issue 5.1).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import main
from schemas import AQIInfo, AirQualityData, LocationInfo, PollutantData


def _sample_air_quality(aqi_value: int = 160, nmu_risk: str = "high") -> AirQualityData:
    return AirQualityData(
        timestamp=datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc),
        location=LocationInfo(latitude=55.7558, longitude=37.6176),
        aqi=AQIInfo(value=aqi_value, category="Вредно", color="#EF4444", description="Высокое загрязнение"),
        pollutants=PollutantData(pm2_5=50.0, pm10=80.0, no2=45.0, so2=16.0, o3=92.0),
        recommendations="Ограничьте активность на улице",
        nmu_risk=nmu_risk,
        health_warnings=["Высокий риск для чувствительных групп"],
    )


@pytest.mark.asyncio
async def test_alert_rule_crud_and_check_current():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        create_resp = await client.post(
            "/alerts/rules",
            json={
                "name": "AQI >= 140",
                "aqi_threshold": 140,
                "nmu_levels": ["high", "critical"],
                "cooldown_minutes": 30,
            },
        )
        assert create_resp.status_code == 200
        rule_id = create_resp.json()["id"]

        list_resp = await client.get("/alerts/rules")
        assert list_resp.status_code == 200
        assert any(item["id"] == rule_id for item in list_resp.json())

        update_resp = await client.put(
            f"/alerts/rules/{rule_id}",
            json={"name": "AQI >= 145", "cooldown_minutes": 45, "chat_id": "123456"},
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["name"] == "AQI >= 145"
        assert update_resp.json()["cooldown_minutes"] == 45
        assert update_resp.json()["chat_id"] == "123456"

        with patch.object(main.unified_weather_service, "get_current_combined_data", AsyncMock(return_value=_sample_air_quality())):
            check_resp = await client.get("/alerts/check-current?lat=55.7558&lon=37.6176")
            assert check_resp.status_code == 200
            events = check_resp.json()
            assert any((not evt["suppressed"]) for evt in events)

        delete_resp = await client.delete(f"/alerts/rules/{rule_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["deleted"] is True
