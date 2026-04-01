import json
from types import SimpleNamespace
from unittest.mock import patch

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


@pytest.mark.asyncio
async def test_web_alerts_service_reads_api_key_lazily_from_environment(monkeypatch: pytest.MonkeyPatch):
    seen_headers: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers["X-API-Key"])
        return httpx.Response(200, json=[])

    monkeypatch.delenv("ALERTS_API_KEY", raising=False)
    service = WebAppService(
        alerts_api_base_url="http://testserver",
        alerts_transport=httpx.MockTransport(_handler),
    )
    assert service._use_backend_alerts_api() is False

    monkeypatch.setenv("ALERTS_API_KEY", "runtime-key")
    assert service._use_backend_alerts_api() is True

    rules = await service.list_alert_rules()

    assert rules == []
    assert seen_headers == ["runtime-key"]


@pytest.mark.asyncio
async def test_web_alerts_service_reads_api_base_url_lazily_from_environment(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, path, headers=None, json=None):
            captured["method"] = method
            captured["path"] = path
            return SimpleNamespace(is_success=True, status_code=200, content=b"[]", json=lambda: [])

    def _factory(**kwargs):
        captured["factory_kwargs"] = kwargs
        return _Client()

    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("WEB_API_BASE_URL", raising=False)
    service = WebAppService(alerts_api_key="test-alert-key")
    assert service._use_backend_alerts_api() is False

    monkeypatch.setenv("API_BASE_URL", "https://api.example.com")
    assert service._use_backend_alerts_api() is True

    with patch("application.web.service.create_internal_async_client", side_effect=_factory):
        rules = await service.list_alert_rules()

    assert rules == []
    assert captured["factory_kwargs"]["base_url"] == "https://api.example.com"
    assert captured["path"] == "/v2/alerts"


@pytest.mark.asyncio
async def test_web_alerts_service_builds_internal_http_client():
    captured: dict[str, object] = {}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, path, headers=None, json=None):
            captured["method"] = method
            captured["path"] = path
            captured["headers"] = headers
            captured["json"] = json
            return SimpleNamespace(is_success=True, status_code=200, content=b"[]", json=lambda: [])

    def _factory(**kwargs):
        captured["factory_kwargs"] = kwargs
        return _Client()

    service = WebAppService(
        alerts_api_base_url="http://testserver",
        alerts_api_key="test-alert-key",
        alerts_api_timeout_seconds=8.0,
        alerts_api_trust_env=False,
    )

    with patch("application.web.service.create_internal_async_client", side_effect=_factory):
        rules = await service.list_alert_rules()

    assert rules == []
    assert captured["factory_kwargs"]["base_url"] == "http://testserver"
    assert captured["factory_kwargs"]["timeout_seconds"] == 8.0
    assert captured["factory_kwargs"]["trust_env"] is False
    assert captured["method"] == "GET"
    assert captured["path"] == "/v2/alerts"
