from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import httpx
import pytest

import main
from application.services.alerts import AlertSubscriptionService
from infrastructure.repositories import (
    InMemoryAlertAuditRepository,
    InMemoryAlertDeliveryAttemptRepository,
    InMemoryAlertIdempotencyRepository,
    InMemoryAlertSubscriptionRepository,
)


class FakeTelegramDeliveryService:
    def __init__(self):
        self.sent: list[dict[str, str | None]] = []

    async def send_message(self, chat_id: str, text: str, event_id: str | None = None):
        payload = {
            "channel": "telegram",
            "status": "sent",
            "attempts": 1,
            "event_id": event_id,
            "error": None,
        }
        self.sent.append({"chat_id": chat_id, "text": text, "event_id": event_id})
        return payload


@contextmanager
def _patched_alert_service():
    original = getattr(main, "alert_subscription_service", None)
    service = AlertSubscriptionService(
        subscription_repository=InMemoryAlertSubscriptionRepository(),
        delivery_attempt_repository=InMemoryAlertDeliveryAttemptRepository(),
        audit_repository=InMemoryAlertAuditRepository(),
        idempotency_repository=InMemoryAlertIdempotencyRepository(),
        telegram_delivery_service=FakeTelegramDeliveryService(),
    )
    main.alert_subscription_service = service
    try:
        yield service
    finally:
        main.alert_subscription_service = original


@pytest.mark.asyncio
async def test_stage4_v2_alerts_crud_and_idempotency():
    with _patched_alert_service(), patch.dict("os.environ", {"ALERTS_API_KEY": "test-alert-key"}, clear=False):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            create = await client.post(
                "/v2/alerts",
                headers={"X-API-Key": "test-alert-key", "Idempotency-Key": "create-1"},
                json={
                    "name": "Moscow high AQI",
                    "city": "moscow",
                    "aqi_threshold": 140,
                    "nmu_levels": ["high"],
                    "cooldown_minutes": 30,
                    "channel": "telegram",
                    "chat_id": "123",
                },
            )
            create_again = await client.post(
                "/v2/alerts",
                headers={"X-API-Key": "test-alert-key", "Idempotency-Key": "create-1"},
                json={
                    "name": "Moscow high AQI",
                    "city": "moscow",
                    "aqi_threshold": 140,
                    "nmu_levels": ["high"],
                    "cooldown_minutes": 30,
                    "channel": "telegram",
                    "chat_id": "123",
                },
            )
            assert create.status_code == 201
            assert create_again.status_code == 201
            created = create.json()
            assert create_again.json()["id"] == created["id"]

            list_response = await client.get("/v2/alerts", headers={"X-API-Key": "test-alert-key"})
            assert list_response.status_code == 200
            assert list_response.json()[0]["id"] == created["id"]

            get_response = await client.get(f"/v2/alerts/{created['id']}", headers={"X-API-Key": "test-alert-key"})
            assert get_response.status_code == 200
            assert get_response.json()["city"] == "moscow"

            update = await client.patch(
                f"/v2/alerts/{created['id']}",
                headers={"X-API-Key": "test-alert-key", "Idempotency-Key": "update-1"},
                json={"cooldown_minutes": 45, "chat_id": "999"},
            )
            assert update.status_code == 200
            assert update.json()["cooldown_minutes"] == 45
            assert update.json()["chat_id"] == "999"

            delete = await client.delete(f"/v2/alerts/{created['id']}", headers={"X-API-Key": "test-alert-key"})
            assert delete.status_code == 200
            assert delete.json() == {"deleted": True, "id": created["id"]}

            missing = await client.get(f"/v2/alerts/{created['id']}", headers={"X-API-Key": "test-alert-key"})
            assert missing.status_code == 404


@pytest.mark.asyncio
async def test_stage4_v2_alerts_auth_and_validation_errors_use_flat_contract():
    with _patched_alert_service(), patch.dict("os.environ", {"ALERTS_API_KEY": "test-alert-key"}, clear=False):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            unauthorized = await client.get("/v2/alerts")
            assert unauthorized.status_code == 401
            unauthorized_payload = unauthorized.json()
            assert unauthorized_payload["code"] == "UNAUTHORIZED"
            assert "message" in unauthorized_payload
            assert "timestamp" in unauthorized_payload

            invalid = await client.post(
                "/v2/alerts",
                headers={"X-API-Key": "test-alert-key"},
                json={
                    "name": "Invalid",
                    "lat": 55.7558,
                    "aqi_threshold": 100,
                    "channel": "telegram",
                    "chat_id": "123",
                },
            )
            assert invalid.status_code == 422
            invalid_payload = invalid.json()
            assert invalid_payload["code"] == "VALIDATION_ERROR"
            assert "details" in invalid_payload

            create = await client.post(
                "/v2/alerts",
                headers={"X-API-Key": "test-alert-key", "Idempotency-Key": "same-key"},
                json={
                    "name": "Once",
                    "city": "moscow",
                    "aqi_threshold": 150,
                    "channel": "telegram",
                    "chat_id": "123",
                },
            )
            conflict = await client.post(
                "/v2/alerts",
                headers={"X-API-Key": "test-alert-key", "Idempotency-Key": "same-key"},
                json={
                    "name": "Changed",
                    "city": "moscow",
                    "aqi_threshold": 151,
                    "channel": "telegram",
                    "chat_id": "123",
                },
            )
            assert create.status_code == 201
            assert conflict.status_code == 409
            assert conflict.json()["code"] == "CONFLICT"


@pytest.mark.asyncio
async def test_stage4_v2_alerts_invalid_key_and_missing_auth_config_use_flat_contract():
    with _patched_alert_service(), patch.dict("os.environ", {"ALERTS_API_KEY": "test-alert-key", "ALERTS_API_KEYS": ""}, clear=False):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            invalid_key = await client.get("/v2/alerts", headers={"X-API-Key": "wrong-key"})

        assert invalid_key.status_code == 401
        payload = invalid_key.json()
        assert payload["code"] == "UNAUTHORIZED"
        assert "message" in payload
        assert "timestamp" in payload

    with _patched_alert_service(), patch.dict("os.environ", {"ALERTS_API_KEY": "", "ALERTS_API_KEYS": ""}, clear=False):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            unconfigured = await client.get("/v2/alerts", headers={"X-API-Key": "anything"})

        assert unconfigured.status_code == 503
        payload = unconfigured.json()
        assert payload["code"] == "SERVICE_UNAVAILABLE"
        assert "message" in payload
        assert "timestamp" in payload
