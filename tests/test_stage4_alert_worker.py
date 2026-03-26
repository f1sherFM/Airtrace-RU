from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from application.services.alert_worker import AlertEvaluationWorker
from application.services.alerts import AlertSubscriptionService
from infrastructure.repositories import (
    InMemoryAlertAuditRepository,
    InMemoryAlertDeliveryAttemptRepository,
    InMemoryAlertIdempotencyRepository,
    InMemoryAlertSubscriptionRepository,
)
from schemas import AQIInfo, AirQualityData, AlertSubscriptionCreate, LocationInfo, PollutantData


class FakeTelegramDeliveryService:
    def __init__(self, failed_chat_ids: set[str] | None = None):
        self.failed_chat_ids = failed_chat_ids or set()
        self.sent: list[dict[str, str | None]] = []

    async def send_message(self, chat_id: str, text: str, event_id: str | None = None):
        self.sent.append({"chat_id": chat_id, "text": text, "event_id": event_id})
        failed = chat_id in self.failed_chat_ids
        return {
            "channel": "telegram",
            "status": "failed" if failed else "sent",
            "attempts": 1,
            "event_id": event_id,
            "error": "delivery_failed" if failed else None,
        }


def _sample_air_quality(
    *,
    lat: float,
    lon: float,
    aqi_value: int = 170,
    nmu_risk: str = "high",
) -> AirQualityData:
    return AirQualityData(
        timestamp=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        location=LocationInfo(latitude=lat, longitude=lon),
        aqi=AQIInfo(value=aqi_value, category="Опасное", color="#EF4444", description="Высокий риск"),
        pollutants=PollutantData(pm2_5=50.0, pm10=80.0, no2=45.0, so2=16.0, o3=92.0),
        recommendations="Сократите активность на улице",
        nmu_risk=nmu_risk,
        health_warnings=["Высокий риск для чувствительных групп"],
    )


def _build_service(*, failed_chat_ids: set[str] | None = None):
    subscription_repository = InMemoryAlertSubscriptionRepository()
    delivery_repository = InMemoryAlertDeliveryAttemptRepository()
    audit_repository = InMemoryAlertAuditRepository()
    idempotency_repository = InMemoryAlertIdempotencyRepository()
    telegram_service = FakeTelegramDeliveryService(failed_chat_ids=failed_chat_ids)
    service = AlertSubscriptionService(
        subscription_repository=subscription_repository,
        delivery_attempt_repository=delivery_repository,
        audit_repository=audit_repository,
        idempotency_repository=idempotency_repository,
        telegram_delivery_service=telegram_service,
    )
    return service, subscription_repository, delivery_repository, audit_repository, telegram_service


@pytest.mark.asyncio
async def test_stage4_alert_worker_groups_fetches_and_triggers_aqi_and_nmu():
    service, _subscriptions, delivery_repository, _audit_repository, telegram = _build_service()
    await service.create_subscription(
        AlertSubscriptionCreate(
            name="Moscow AQI",
            city="moscow",
            aqi_threshold=150,
            cooldown_minutes=30,
            channel="telegram",
            chat_id="111",
        )
    )
    await service.create_subscription(
        AlertSubscriptionCreate(
            name="Moscow NMU",
            city="moscow",
            nmu_levels=["high"],
            cooldown_minutes=30,
            channel="telegram",
            chat_id="222",
        )
    )
    await service.create_subscription(
        AlertSubscriptionCreate(
            name="SPB AQI",
            lat=59.9343,
            lon=30.3351,
            aqi_threshold=150,
            cooldown_minutes=30,
            channel="telegram",
            chat_id="333",
        )
    )
    await service.create_subscription(
        AlertSubscriptionCreate(
            name="Disabled",
            city="moscow",
            aqi_threshold=150,
            enabled=False,
            cooldown_minutes=30,
            channel="telegram",
            chat_id="444",
        )
    )

    calls: list[tuple[float, float]] = []

    async def fetch_current_data(lat: float, lon: float):
        calls.append((round(lat, 4), round(lon, 4)))
        return _sample_air_quality(lat=lat, lon=lon, aqi_value=170, nmu_risk="high")

    worker = AlertEvaluationWorker(alert_service=service, fetch_current_data=fetch_current_data)
    result = await worker.run_cycle()

    assert result.location_groups == 2
    assert result.triggered_alerts == 3
    assert len(calls) == 2
    assert len(delivery_repository._attempts) == 3
    assert {item["chat_id"] for item in telegram.sent} == {"111", "222", "333"}


@pytest.mark.asyncio
async def test_stage4_alert_worker_respects_quiet_hours_and_cooldown():
    service, _subscriptions, delivery_repository, _audit_repository, telegram = _build_service()
    await service.create_subscription(
        AlertSubscriptionCreate(
            name="Quiet Hours",
            city="moscow",
            aqi_threshold=150,
            cooldown_minutes=30,
            quiet_hours_start=12,
            quiet_hours_end=13,
            channel="telegram",
            chat_id="111",
        )
    )
    await service.create_subscription(
        AlertSubscriptionCreate(
            name="Cooldown",
            lat=59.9343,
            lon=30.3351,
            aqi_threshold=150,
            cooldown_minutes=30,
            channel="telegram",
            chat_id="222",
        )
    )

    current_time = datetime(2026, 3, 26, 12, 5, tzinfo=timezone.utc)

    async def fetch_current_data(lat: float, lon: float):
        return _sample_air_quality(lat=lat, lon=lon, aqi_value=170, nmu_risk="high")

    worker = AlertEvaluationWorker(
        alert_service=service,
        fetch_current_data=fetch_current_data,
        now_provider=lambda: current_time,
    )
    first_result = await worker.run_cycle()
    current_time = datetime(2026, 3, 26, 12, 10, tzinfo=timezone.utc)
    second_result = await worker.run_cycle()

    assert first_result.triggered_alerts == 1
    assert second_result.triggered_alerts == 0
    assert len(delivery_repository._attempts) == 1
    assert telegram.sent[0]["chat_id"] == "222"


@pytest.mark.asyncio
async def test_stage4_alert_worker_continues_after_fetch_and_delivery_failures():
    service, _subscriptions, delivery_repository, _audit_repository, telegram = _build_service(failed_chat_ids={"222"})
    await service.create_subscription(
        AlertSubscriptionCreate(
            name="Broken fetch",
            city="moscow",
            aqi_threshold=150,
            cooldown_minutes=30,
            channel="telegram",
            chat_id="111",
        )
    )
    await service.create_subscription(
        AlertSubscriptionCreate(
            name="Delivery fail",
            lat=59.9343,
            lon=30.3351,
            aqi_threshold=150,
            cooldown_minutes=30,
            channel="telegram",
            chat_id="222",
        )
    )
    await service.create_subscription(
        AlertSubscriptionCreate(
            name="Delivery ok",
            lat=59.9343,
            lon=30.3351,
            nmu_levels=["high"],
            cooldown_minutes=30,
            channel="telegram",
            chat_id="333",
        )
    )

    async def fetch_current_data(lat: float, lon: float):
        if round(lat, 4) == 55.7558:
            raise RuntimeError("fetch failed")
        return _sample_air_quality(lat=lat, lon=lon, aqi_value=170, nmu_risk="high")

    worker = AlertEvaluationWorker(alert_service=service, fetch_current_data=fetch_current_data)
    result = await worker.run_cycle()

    assert result.location_groups == 2
    assert result.failed_fetches == 1
    assert result.failed_deliveries == 1
    assert len(delivery_repository._attempts) == 2
    assert {attempt.status for attempt in delivery_repository._attempts} == {"failed", "sent"}
    assert {item["chat_id"] for item in telegram.sent} == {"222", "333"}


@pytest.mark.asyncio
async def test_stage4_alert_worker_run_forever_shuts_down_gracefully():
    service, *_ = _build_service()

    async def fetch_current_data(lat: float, lon: float):
        return _sample_air_quality(lat=lat, lon=lon)

    worker = AlertEvaluationWorker(alert_service=service, fetch_current_data=fetch_current_data)
    task = asyncio.create_task(worker.run_forever(interval_seconds=60))
    await asyncio.sleep(0)
    task.cancel()
    await task

    assert task.done() is True
    assert task.cancelled() is False
