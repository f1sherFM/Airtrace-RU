import json

import httpx
import pytest

from application.web.service import WebAppService


@pytest.mark.asyncio
async def test_web_alerts_service_uses_backend_api_for_crud():
    requests: list[tuple[str, str, dict | None]] = []
    subscription = {
        "id": "sub-1",
        "name": "Surgut test",
        "enabled": True,
        "city": "surgut",
        "lat": None,
        "lon": None,
        "aqi_threshold": 0,
        "nmu_levels": [],
        "cooldown_minutes": 30,
        "quiet_hours_start": None,
        "quiet_hours_end": None,
        "channel": "telegram",
        "chat_id": "5110137438",
        "last_triggered_at": None,
        "last_delivery_status": None,
        "created_at": "2026-03-29T18:48:39.632834+00:00",
        "updated_at": "2026-03-29T18:48:39.632834+00:00",
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        payload = None
        if request.content:
            payload = json.loads(request.content.decode("utf-8"))
        requests.append((request.method, request.url.path, payload))
        assert request.headers["X-API-Key"] == "test-alert-key"

        if request.method == "GET" and request.url.path == "/v2/alerts":
            return httpx.Response(200, json=[subscription])
        if request.method == "POST" and request.url.path == "/v2/alerts":
            created = dict(subscription)
            created.update(payload or {})
            return httpx.Response(201, json=created)
        if request.method == "PATCH" and request.url.path == "/v2/alerts/sub-1":
            updated = dict(subscription)
            updated.update(payload or {})
            return httpx.Response(200, json=updated)
        if request.method == "DELETE" and request.url.path == "/v2/alerts/sub-1":
            return httpx.Response(200, json={"deleted": True, "id": "sub-1"})
        raise AssertionError(f"Unexpected request {request.method} {request.url}")

    create_payload = {
        "name": "Surgut test",
        "enabled": True,
        "city": "surgut",
        "aqi_threshold": 0,
        "nmu_levels": [],
        "cooldown_minutes": 30,
        "channel": "telegram",
        "chat_id": "5110137438",
    }

    service = WebAppService(
        alerts_api_base_url="http://testserver",
        alerts_api_key="test-alert-key",
        alerts_transport=httpx.MockTransport(_handler),
    )

    rules = await service.list_alert_rules()
    created = await service.create_alert_rule(create_payload)
    updated = await service.update_alert_rule("sub-1", {"cooldown_minutes": 45})
    deleted = await service.delete_alert_rule("sub-1")

    assert rules == [subscription]
    assert created["city"] == "surgut"
    assert updated["cooldown_minutes"] == 45
    assert deleted == {"deleted": True, "id": "sub-1"}
    assert requests == [
        ("GET", "/v2/alerts", None),
        ("POST", "/v2/alerts", create_payload),
        ("PATCH", "/v2/alerts/sub-1", {"cooldown_minutes": 45}),
        ("DELETE", "/v2/alerts/sub-1", None),
    ]
