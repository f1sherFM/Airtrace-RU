"""Smoke tests for generated Stage 3 SDK assets."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import httpx

SDK_SRC = Path("sdk/python/src").resolve()
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

from airtrace_sdk import AirTraceClient


def test_python_sdk_generated_client_calls_v2_routes_via_mock_transport():
    seen: list[tuple[str, str, str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.method,
                request.url.path,
                str(request.url),
                request.headers.get("Idempotency-Key"),
            )
        )
        return httpx.Response(200, json={"ok": True, "path": request.url.path, "method": request.method})

    transport = httpx.MockTransport(handler)
    with AirTraceClient(base_url="http://testserver", api_key="test-key", transport=transport) as client:
        assert client.get_health()["path"] == "/v2/health"
        assert client.get_current(lat=55.7558, lon=37.6176)["path"] == "/v2/current"
        assert client.get_history_by_city(city="moscow", sort="asc")["path"] == "/v2/history"
        assert client.get_trends_by_city(city="moscow", range="7d")["path"] == "/v2/trends"
        assert client.list_alerts()["path"] == "/v2/alerts"
        assert client.get_alert(subscription_id="sub-1")["path"] == "/v2/alerts/sub-1"
        assert client.create_alert(payload={"name": "x"}, idempotency_key="idem-create")["method"] == "POST"
        assert client.update_alert(subscription_id="sub-1", payload={"enabled": False}, idempotency_key="idem-update")["method"] == "PATCH"
        assert client.delete_alert(subscription_id="sub-1")["method"] == "DELETE"

    assert [(item[0], item[1]) for item in seen] == [
        ("GET", "/v2/health"),
        ("GET", "/v2/current"),
        ("GET", "/v2/history"),
        ("GET", "/v2/trends"),
        ("GET", "/v2/alerts"),
        ("GET", "/v2/alerts/sub-1"),
        ("POST", "/v2/alerts"),
        ("PATCH", "/v2/alerts/sub-1"),
        ("DELETE", "/v2/alerts/sub-1"),
    ]
    assert "sort=asc" in seen[2][2]
    assert seen[6][3] == "idem-create"
    assert seen[7][3] == "idem-update"


def test_generated_sdk_assets_reference_public_openapi_and_trends_methods():
    py_readme = Path("sdk/python/README.md").read_text(encoding="utf-8")
    py_client = Path("sdk/python/src/airtrace_sdk/client.py").read_text(encoding="utf-8")
    js_readme = Path("sdk/js/README.md").read_text(encoding="utf-8")
    js_client = Path("sdk/js/src/index.ts").read_text(encoding="utf-8")

    assert "openapi/airtrace-v2.openapi.json" in py_readme
    assert "Generated from openapi/airtrace-v2.openapi.json" in py_client
    assert "get_trends_by_city" in py_client
    assert "sort: str = \"desc\"" in py_client
    assert "create_alert" in py_client
    assert "api_key: Optional[str] = None" in py_client

    assert "openapi/airtrace-v2.openapi.json" in js_readme
    assert "getTrendsByCity" in js_client
    assert 'sort?: "asc" | "desc"' in js_client
    assert "createAlert" in js_client
    assert "apiKey?: string;" in js_client


def test_generated_public_openapi_mentions_trends_and_history_sort():
    payload = json.loads(Path("openapi/airtrace-v2.openapi.json").read_text(encoding="utf-8"))
    history_parameters = payload["paths"]["/v2/history"]["get"]["parameters"]
    assert any(param["name"] == "sort" for param in history_parameters)
    assert "/v2/trends" in payload["paths"]
    assert "/v2/alerts" in payload["paths"]
    assert "/v2/alerts/{subscription_id}" in payload["paths"]
