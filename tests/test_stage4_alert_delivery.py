from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import logging

import main
from application.services.alerts import AlertSubscriptionService
from infrastructure.repositories import (
    InMemoryAlertAuditRepository,
    InMemoryAlertDeliveryAttemptRepository,
    InMemoryAlertIdempotencyRepository,
    InMemoryAlertSubscriptionRepository,
)
from schemas import AQIInfo, AirQualityData, AlertRuleCreate, LocationInfo, PollutantData


class FakeTelegramDeliveryService:
    def __init__(self, *, status: str = "sent"):
        self.status = status
        self.sent: list[dict[str, str | None]] = []

    async def send_message(self, chat_id: str, text: str, event_id: str | None = None):
        self.sent.append({"chat_id": chat_id, "text": text, "event_id": event_id})
        return {
            "channel": "telegram",
            "status": self.status,
            "attempts": 1,
            "event_id": event_id,
            "error": None if self.status == "sent" else "delivery_failed",
        }


def _sample_air_quality(aqi_value: int = 170, nmu_risk: str = "high") -> AirQualityData:
    return AirQualityData(
        timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        location=LocationInfo(latitude=55.7558, longitude=37.6176),
        aqi=AQIInfo(value=aqi_value, category="Р’СЂРµРґРЅРѕ", color="#EF4444", description="Р’С‹СЃРѕРєРѕРµ Р·Р°РіСЂСЏР·РЅРµРЅРёРµ"),
        pollutants=PollutantData(pm2_5=50.0, pm10=80.0, no2=45.0, so2=16.0, o3=92.0),
        recommendations="РћРіСЂР°РЅРёС‡РёС‚СЊ Р°РєС‚РёРІРЅРѕСЃС‚СЊ РЅР° СѓР»РёС†Рµ",
        nmu_risk=nmu_risk,
        health_warnings=["Р’С‹СЃРѕРєРёР№ СЂРёСЃРє РґР»СЏ С‡СѓРІСЃС‚РІРёС‚РµР»СЊРЅС‹С… РіСЂСѓРїРї"],
    )


@contextmanager
def _patched_alert_runtime(*, delivery_status: str = "sent"):
    original = getattr(main, "alert_subscription_service", None)
    delivery_repository = InMemoryAlertDeliveryAttemptRepository()
    telegram_service = FakeTelegramDeliveryService(status=delivery_status)
    service = AlertSubscriptionService(
        subscription_repository=InMemoryAlertSubscriptionRepository(),
        delivery_attempt_repository=delivery_repository,
        audit_repository=InMemoryAlertAuditRepository(),
        idempotency_repository=InMemoryAlertIdempotencyRepository(),
        telegram_delivery_service=telegram_service,
    )
    main.alert_subscription_service = service
    try:
        yield service, delivery_repository, telegram_service
    finally:
        main.alert_subscription_service = original


@pytest.mark.asyncio
async def test_stage4_delivery_service_persists_attempts_and_cooldown_state():
    with _patched_alert_runtime() as (service, delivery_repository, telegram_service):
        rule = await service.create_legacy_rule(
            AlertRuleCreate(name="AQI>=150", aqi_threshold=150, cooldown_minutes=30, chat_id="777")
        )
        events = await service.evaluate_conditions(
            aqi=170,
            nmu_risk="high",
            lat=55.7558,
            lon=37.6176,
            now=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        delivered = await service.deliver_events(events=events, aqi=170, nmu_risk="high")
        suppressed = await service.evaluate_conditions(
            aqi=170,
            nmu_risk="high",
            lat=55.7558,
            lon=37.6176,
            now=datetime(2026, 3, 26, 12, 5, tzinfo=timezone.utc),
        )

        assert rule.id == events[0].rule_id
        assert delivered[0].status == "sent"
        assert delivery_repository._attempts[0].subscription_id == rule.id
        assert telegram_service.sent[0]["chat_id"] == "777"
        assert suppressed[0].suppressed is True
        assert "cooldown" in suppressed[0].reasons


@pytest.mark.asyncio
async def test_stage4_delivery_service_emits_runtime_logs(caplog):
    caplog.set_level(logging.INFO)
    with _patched_alert_runtime() as (service, _delivery_repository, _telegram_service):
        await service.create_legacy_rule(
            AlertRuleCreate(name="AQI>=150", aqi_threshold=150, cooldown_minutes=30, chat_id="777")
        )
        events = await service.evaluate_conditions(
            aqi=170,
            nmu_risk="high",
            lat=55.7558,
            lon=37.6176,
            now=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        )
        await service.deliver_events(events=events, aqi=170, nmu_risk="high")

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Alert triggered:" in messages
    assert "Alert delivery result:" in messages
    assert "status=sent" in messages


@pytest.mark.asyncio
async def test_stage4_legacy_alert_routes_still_work_over_new_service():
    with _patched_alert_runtime() as _runtime, patch.object(
        main.unified_weather_service,
        "get_current_combined_data",
        AsyncMock(return_value=_sample_air_quality()),
    ), patch.dict("os.environ", {"ALERTS_API_KEY": "test-alert-key"}, clear=False):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            create = await client.post(
                "/alerts/rules",
                json={"name": "AQI >= 140", "aqi_threshold": 140, "cooldown_minutes": 30, "chat_id": "123"},
            )
            assert create.status_code == 200
            rule_id = create.json()["id"]

            deliver = await client.get(
                "/alerts/check-current-and-deliver?lat=55.7558&lon=37.6176",
                headers={"X-API-Key": "test-alert-key"},
            )
            assert deliver.status_code == 200
            assert any(item["status"] == "sent" for item in deliver.json())

            check = await client.get("/alerts/check-current?lat=55.7558&lon=37.6176")
            assert check.status_code == 200
            assert any(event["rule_id"] == rule_id for event in check.json())
