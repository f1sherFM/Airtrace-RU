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
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, str(request.url)))
        return httpx.Response(200, json={"ok": True, "path": request.url.path})

    transport = httpx.MockTransport(handler)
    with AirTraceClient(base_url="http://testserver", transport=transport) as client:
        assert client.get_health()["path"] == "/v2/health"
        assert client.get_current(lat=55.7558, lon=37.6176)["path"] == "/v2/current"
        assert client.get_history_by_city(city="moscow", sort="asc")["path"] == "/v2/history"
        assert client.get_trends_by_city(city="moscow", range="7d")["path"] == "/v2/trends"

    assert [item[0] for item in seen] == ["/v2/health", "/v2/current", "/v2/history", "/v2/trends"]
    assert "sort=asc" in seen[2][1]


def test_generated_sdk_assets_reference_public_openapi_and_trends_methods():
    py_readme = Path("sdk/python/README.md").read_text(encoding="utf-8")
    py_client = Path("sdk/python/src/airtrace_sdk/client.py").read_text(encoding="utf-8")
    js_readme = Path("sdk/js/README.md").read_text(encoding="utf-8")
    js_client = Path("sdk/js/src/index.ts").read_text(encoding="utf-8")

    assert "openapi/airtrace-v2.openapi.json" in py_readme
    assert "Generated from openapi/airtrace-v2.openapi.json" in py_client
    assert "get_trends_by_city" in py_client
    assert "sort: str = \"desc\"" in py_client

    assert "openapi/airtrace-v2.openapi.json" in js_readme
    assert "getTrendsByCity" in js_client
    assert 'sort?: "asc" | "desc"' in js_client


def test_generated_public_openapi_mentions_trends_and_history_sort():
    payload = json.loads(Path("openapi/airtrace-v2.openapi.json").read_text(encoding="utf-8"))
    history_parameters = payload["paths"]["/v2/history"]["get"]["parameters"]
    assert any(param["name"] == "sort" for param in history_parameters)
    assert "/v2/trends" in payload["paths"]
