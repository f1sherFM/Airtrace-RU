from pathlib import Path
import importlib.util
import os
import sys

import httpx
import pytest
from fastapi.templating import Jinja2Templates


def _load_web_app_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "web" / "web_app.py"
    previous_cwd = os.getcwd()
    added_path = str(repo_root / "web")
    try:
        os.chdir(repo_root / "web")
        if added_path not in sys.path:
            sys.path.insert(0, added_path)
        spec = importlib.util.spec_from_file_location("web_app", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.templates = Jinja2Templates(directory=str(repo_root / "web" / "templates"))
        module.templates.env.globals["encoding"] = "utf-8"
        module.templates.env.globals["format_time"] = module.format_time
        module.templates.env.globals["translate_api_status"] = module.translate_api_status
        module.templates.env.globals["translate_source"] = module.translate_source
        module.templates.env.globals["translate_freshness"] = module.translate_freshness
        module.templates.env.globals["translate_trend"] = module.translate_trend
        return module
    finally:
        os.chdir(previous_cwd)


web_app = _load_web_app_module()


@pytest.mark.asyncio
async def test_stage0_ssr_index_renders_html_with_city_marker():
    original = web_app.air_service.check_health

    async def _fake_health():
        return {"status": "healthy", "reachable": True}

    web_app.air_service.check_health = _fake_health
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")
    finally:
        web_app.air_service.check_health = original

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    html = response.text
    assert "AirTrace RU" in html
    assert "Москва" in html


@pytest.mark.asyncio
async def test_stage0_ssr_city_page_renders_with_stubbed_backend():
    original_current = web_app.air_service.get_current_data
    original_forecast = web_app.air_service.get_forecast_data
    original_history = web_app.air_service.get_history_data

    async def _fake_current(lat: float, lon: float):
        return {
            "aqi": {"value": 85, "category": "Умеренное", "color": "#FFFF00"},
            "nmu_risk": "medium",
            "pollutants": {"pm2_5": 25.4, "pm10": 45.2, "no2": 35.1, "so2": 12.3, "o3": 85.7},
            "metadata": {"confidence": 0.91},
        }

    async def _fake_forecast(lat: float, lon: float, hours: int = 24):
        return [{"aqi": {"value": 80}, "timestamp": "2026-03-25T13:00:00+00:00"}]

    async def _fake_history(**kwargs):
        return {"items": [{"aqi": 82, "metadata": {"confidence": 0.88}}], "total": 1, "range": "24h"}

    web_app.air_service.get_current_data = _fake_current
    web_app.air_service.get_forecast_data = _fake_forecast
    web_app.air_service.get_history_data = _fake_history
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/city/moscow")
    finally:
        web_app.air_service.get_current_data = original_current
        web_app.air_service.get_forecast_data = original_forecast
        web_app.air_service.get_history_data = original_history

    assert response.status_code == 200
    html = response.text
    assert "AirTrace RU" in html
    assert "Москва" in html


@pytest.mark.asyncio
async def test_stage0_ssr_custom_form_renders():
    transport = httpx.ASGITransport(app=web_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/custom")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "AirTrace RU" in response.text


@pytest.mark.asyncio
async def test_stage0_ssr_custom_submit_renders_with_stubbed_backend():
    original_current = web_app.air_service.get_current_data
    original_forecast = web_app.air_service.get_forecast_data
    original_history = web_app.air_service.get_history_data

    async def _fake_current(lat: float, lon: float):
        return {
            "aqi": {"value": 72, "category": "Хорошее", "color": "#00E400"},
            "nmu_risk": "low",
            "pollutants": {"pm2_5": 12.1, "pm10": 18.0, "no2": 12.0, "so2": 3.0, "o3": 49.0},
            "metadata": {"confidence": 0.95},
        }

    async def _fake_forecast(lat: float, lon: float, hours: int = 24):
        return [{"aqi": {"value": 70}, "timestamp": "2026-03-25T13:00:00+00:00"}]

    async def _fake_history(**kwargs):
        return {"items": [{"aqi": 68, "metadata": {"confidence": 0.84}}], "total": 1, "range": "24h"}

    web_app.air_service.get_current_data = _fake_current
    web_app.air_service.get_forecast_data = _fake_forecast
    web_app.air_service.get_history_data = _fake_history
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/custom", data={"lat": "55.7558", "lon": "37.6176", "city_name": "Москва"})
    finally:
        web_app.air_service.get_current_data = original_current
        web_app.air_service.get_forecast_data = original_forecast
        web_app.air_service.get_history_data = original_history

    assert response.status_code == 200
    assert "AirTrace RU" in response.text


@pytest.mark.asyncio
async def test_stage0_ssr_custom_invalid_coordinates_keep_validation_behavior():
    transport = httpx.ASGITransport(app=web_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/custom", data={"lat": "91", "lon": "37.6176", "city_name": "X"})

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_stage0_ssr_alert_settings_renders_smoke_markers():
    original = web_app.air_service.list_alert_rules

    async def _fake_list():
        return []

    web_app.air_service.list_alert_rules = _fake_list
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/alerts/settings")
    finally:
        web_app.air_service.list_alert_rules = original

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    html = response.text
    assert "AirTrace RU" in html
    assert "Telegram" in html


@pytest.mark.asyncio
async def test_stage0_ssr_error_flow_renders_error_template():
    original_current = web_app.air_service.get_current_data

    async def _fake_current(lat: float, lon: float):
        raise RuntimeError("stage0 smoke failure")

    web_app.air_service.get_current_data = _fake_current
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/city/moscow")
    finally:
        web_app.air_service.get_current_data = original_current

    assert response.status_code == 200
    html = response.text
    assert "stage0 smoke failure" in html


@pytest.mark.asyncio
async def test_stage0_ssr_api_health_reports_backend_reachability_keys():
    original = web_app.air_service.check_health

    async def _fake_health():
        return {"status": "degraded", "reachable": False}

    web_app.air_service.check_health = _fake_health
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")
    finally:
        web_app.air_service.check_health = original

    assert response.status_code == 200
    payload = response.json()
    assert "backend_reachable" in payload
    assert "backend_api" in payload


@pytest.mark.asyncio
async def test_stage0_ssr_export_route_preserves_download_behavior():
    original_ts = web_app.air_service.get_time_series_data

    async def _fake_ts(lat: float, lon: float, hours: int = 24):
        return [
            {
                "timestamp": "2026-03-25T12:00:00+00:00",
                "aqi": {"value": 80, "category": "Умеренное"},
                "pollutants": {"pm2_5": 21.0, "pm10": 30.0, "no2": 20.0, "so2": 8.0, "o3": 55.0},
                "nmu_risk": "low",
                "location": {"latitude": lat, "longitude": lon},
            }
        ]

    web_app.air_service.get_time_series_data = _fake_ts
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/export/moscow?format=json&hours=24")
    finally:
        web_app.air_service.get_time_series_data = original_ts

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "attachment; filename=" in response.headers.get("content-disposition", "").lower()
