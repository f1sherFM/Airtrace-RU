"""
Integration tests for Telegram alert delivery endpoints (Issue 5.2).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import main
from schemas import AQIInfo, AirQualityData, AlertRuleCreate, LocationInfo, PollutantData


def _sample_air_quality(aqi_value: int = 170, nmu_risk: str = "high") -> AirQualityData:
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
async def test_telegram_send_endpoint_returns_delivery_result():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with patch.object(main.telegram_delivery_service, "send_message", AsyncMock(return_value={
            "channel": "telegram",
            "status": "sent",
            "attempts": 1,
            "event_id": None,
        })):
            resp = await client.post("/alerts/telegram/send", json={"chat_id": "123", "message": "hello"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "sent"


@pytest.mark.asyncio
async def test_check_current_and_deliver_sends_unsuppressed_alerts():
    rule = main.alert_rule_engine.create_rule(
        AlertRuleCreate(name="AQI>=150", aqi_threshold=150, cooldown_minutes=30, nmu_levels=[])
    )
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with patch.object(main.unified_weather_service, "get_current_combined_data", AsyncMock(return_value=_sample_air_quality())), patch.object(
            main.telegram_delivery_service,
            "send_message",
            AsyncMock(return_value={"channel": "telegram", "status": "sent", "attempts": 1, "event_id": "evt"}),
        ):
            resp = await client.get("/alerts/check-current-and-deliver?lat=55.7558&lon=37.6176&chat_id=123")
            assert resp.status_code == 200
            items = resp.json()
            assert len(items) >= 1
            assert any(item["status"] == "sent" for item in items)

    main.alert_rule_engine.delete_rule(rule.id)


@pytest.mark.asyncio
async def test_check_current_and_deliver_uses_rule_chat_subscription():
    rule = main.alert_rule_engine.create_rule(
        AlertRuleCreate(name="AQI>=150 subscribed", aqi_threshold=150, cooldown_minutes=30, nmu_levels=[], chat_id="777")
    )
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with patch.object(main.unified_weather_service, "get_current_combined_data", AsyncMock(return_value=_sample_air_quality())), patch.object(
            main.telegram_delivery_service,
            "send_message",
            AsyncMock(return_value={"channel": "telegram", "status": "sent", "attempts": 1, "event_id": "evt"}),
        ) as send_mock:
            resp = await client.get("/alerts/check-current-and-deliver?lat=55.7558&lon=37.6176")
            assert resp.status_code == 200
            assert any(item["status"] == "sent" for item in resp.json())
            assert send_mock.await_args.kwargs["chat_id"] == "777"

    main.alert_rule_engine.delete_rule(rule.id)
