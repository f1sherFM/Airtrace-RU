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


def _current_payload(lat: float, lon: float, *, source: str = "live") -> dict:
    return {
        "timestamp": "2026-03-26T12:00:00+00:00",
        "location": {"latitude": lat, "longitude": lon},
        "aqi": {
            "value": 82,
            "category": "Moderate",
            "color": "#F59E0B",
            "description": "Air quality is acceptable for most people.",
        },
        "pollutants": {"pm2_5": 18.5, "pm10": 26.0, "no2": 11.2, "so2": 3.4, "o3": 42.8},
        "recommendations": "Reduce intense outdoor workouts if you are sensitive.",
        "nmu_risk": "medium",
        "metadata": {
            "data_source": source,
            "freshness": "fresh",
            "confidence": 0.92,
            "confidence_explanation": "High confidence from primary live source",
            "fallback_used": False,
            "cache_age_seconds": 0,
        },
        "health_warnings": [],
    }


def _history_payload() -> dict:
    return {
        "items": [
            {
                "timestamp": "2026-03-26T11:00:00+00:00",
                "aqi": 80,
                "metadata": {"data_source": "historical", "freshness": "fresh", "confidence": 0.89},
            },
            {
                "timestamp": "2026-03-26T10:00:00+00:00",
                "aqi": 76,
                "metadata": {"data_source": "historical", "freshness": "fresh", "confidence": 0.88},
            },
        ],
        "total": 2,
        "range": "24h",
    }


def _trends_payload() -> dict:
    return {
        "range": "7d",
        "location": {"city_code": "moscow", "latitude": 55.7558, "longitude": 37.6176},
        "aggregation": "day",
        "trend": "stable",
        "summary": "Moscow stayed broadly stable over the last 7 days.",
        "points": [
            {
                "timestamp": "2026-03-20T00:00:00+00:00",
                "aqi_min": 55,
                "aqi_max": 92,
                "aqi_avg": 73.4,
                "sample_count": 24,
                "avg_confidence": 0.86,
                "dominant_source": "historical",
            },
            {
                "timestamp": "2026-03-21T00:00:00+00:00",
                "aqi_min": 58,
                "aqi_max": 89,
                "aqi_avg": 71.1,
                "sample_count": 24,
                "avg_confidence": 0.88,
                "dominant_source": "historical",
            },
        ],
    }


@pytest.mark.asyncio
async def test_stage5_city_page_renders_explainability_block():
    original_current = web_app.air_service.get_current_data
    original_forecast = web_app.air_service.get_forecast_data
    original_history = web_app.air_service.get_history_data
    original_trends = web_app.air_service.get_trends_data
    original_health = web_app.air_service.check_health

    async def _fake_current(lat: float, lon: float):
        return _current_payload(lat, lon)

    async def _fake_forecast(lat: float, lon: float, hours: int = 24):
        return [_current_payload(lat, lon, source="forecast")]

    async def _fake_history(**kwargs):
        return _history_payload()

    async def _fake_trends(**kwargs):
        return _trends_payload()

    async def _fake_health():
        return {"status": "healthy", "public_status": "healthy", "reachable": True}

    web_app.air_service.get_current_data = _fake_current
    web_app.air_service.get_forecast_data = _fake_forecast
    web_app.air_service.get_history_data = _fake_history
    web_app.air_service.get_trends_data = _fake_trends
    web_app.air_service.check_health = _fake_health
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/city/moscow")
    finally:
        web_app.air_service.get_current_data = original_current
        web_app.air_service.get_forecast_data = original_forecast
        web_app.air_service.get_history_data = original_history
        web_app.air_service.get_trends_data = original_trends
        web_app.air_service.check_health = original_health

    assert response.status_code == 200
    html = response.text
    assert "Пояснение к данным" in html
    assert "Источник" in html
    assert "Свежесть" in html
    assert "Уверенность" in html


@pytest.mark.asyncio
async def test_stage5_history_page_renders_with_mock_data():
    original_current = web_app.air_service.get_current_data
    original_history = web_app.air_service.get_history_data

    async def _fake_current(lat: float, lon: float):
        return _current_payload(lat, lon)

    async def _fake_history(**kwargs):
        return _history_payload()

    web_app.air_service.get_current_data = _fake_current
    web_app.air_service.get_history_data = _fake_history
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/history/moscow?range=7d")
    finally:
        web_app.air_service.get_current_data = original_current
        web_app.air_service.get_history_data = original_history

    assert response.status_code == 200
    html = response.text
    assert "История" in html
    assert "AQI 80" in html
    assert "Источник:" in html


@pytest.mark.asyncio
async def test_stage5_trends_page_renders_with_mock_data():
    original_current = web_app.air_service.get_current_data
    original_trends = web_app.air_service.get_trends_data

    async def _fake_current(lat: float, lon: float):
        return _current_payload(lat, lon)

    async def _fake_trends(**kwargs):
        return _trends_payload()

    web_app.air_service.get_current_data = _fake_current
    web_app.air_service.get_trends_data = _fake_trends
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/trends/moscow?range=7d")
    finally:
        web_app.air_service.get_current_data = original_current
        web_app.air_service.get_trends_data = original_trends

    assert response.status_code == 200
    html = response.text
    assert "Тренды" in html
    assert "стабильно" in html
    assert "aqi avg" in html


@pytest.mark.asyncio
async def test_stage5_compare_page_renders_two_and_three_cities():
    original_current = web_app.air_service.get_current_data
    original_trends = web_app.air_service.get_trends_data

    async def _fake_current(lat: float, lon: float):
        return _current_payload(lat, lon)

    async def _fake_trends(**kwargs):
        payload = _trends_payload()
        city_key = kwargs.get("city_key") or "custom"
        payload["summary"] = f"{city_key} trend summary"
        return payload

    web_app.air_service.get_current_data = _fake_current
    web_app.air_service.get_trends_data = _fake_trends
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response_two = await client.get("/compare?cities=moscow,spb")
            response_three = await client.get("/compare?cities=moscow,spb,surgut")
    finally:
        web_app.air_service.get_current_data = original_current
        web_app.air_service.get_trends_data = original_trends

    assert response_two.status_code == 200
    assert response_three.status_code == 200
    assert response_two.text.count("AQI сейчас") == 2
    assert response_three.text.count("AQI сейчас") == 3


@pytest.mark.asyncio
async def test_stage5_new_routes_smoke_pack():
    original_current = web_app.air_service.get_current_data
    original_forecast = web_app.air_service.get_forecast_data
    original_history = web_app.air_service.get_history_data
    original_trends = web_app.air_service.get_trends_data
    original_alerts = web_app.air_service.list_alert_rules
    original_health = web_app.air_service.check_health

    async def _fake_current(lat: float, lon: float):
        return _current_payload(lat, lon)

    async def _fake_forecast(lat: float, lon: float, hours: int = 24):
        return [_current_payload(lat, lon, source="forecast")]

    async def _fake_history(**kwargs):
        return _history_payload()

    async def _fake_trends(**kwargs):
        return _trends_payload()

    async def _fake_alerts():
        return []

    async def _fake_health():
        return {"status": "healthy", "reachable": True}

    web_app.air_service.get_current_data = _fake_current
    web_app.air_service.get_forecast_data = _fake_forecast
    web_app.air_service.get_history_data = _fake_history
    web_app.air_service.get_trends_data = _fake_trends
    web_app.air_service.list_alert_rules = _fake_alerts
    web_app.air_service.check_health = _fake_health
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for route in [
                "/",
                "/city/moscow",
                "/custom",
                "/alerts/settings",
                "/history/moscow",
                "/trends/moscow",
                "/compare?cities=moscow,spb",
            ]:
                response = await client.get(route)
                assert response.status_code == 200, route
    finally:
        web_app.air_service.get_current_data = original_current
        web_app.air_service.get_forecast_data = original_forecast
        web_app.air_service.get_history_data = original_history
        web_app.air_service.get_trends_data = original_trends
        web_app.air_service.list_alert_rules = original_alerts
        web_app.air_service.check_health = original_health


@pytest.mark.asyncio
async def test_stage5_api_status_badge_is_consistent_across_pages():
    original_current = web_app.air_service.get_current_data
    original_forecast = web_app.air_service.get_forecast_data
    original_history = web_app.air_service.get_history_data
    original_trends = web_app.air_service.get_trends_data
    original_alerts = web_app.air_service.list_alert_rules
    original_health = web_app.air_service.check_health

    async def _fake_current(lat: float, lon: float):
        return _current_payload(lat, lon)

    async def _fake_forecast(lat: float, lon: float, hours: int = 24):
        return [_current_payload(lat, lon, source="forecast")]

    async def _fake_history(**kwargs):
        return _history_payload()

    async def _fake_trends(**kwargs):
        return _trends_payload()

    async def _fake_alerts():
        return []

    async def _fake_health():
        return {"status": "degraded", "reachable": True}

    web_app.air_service.get_current_data = _fake_current
    web_app.air_service.get_forecast_data = _fake_forecast
    web_app.air_service.get_history_data = _fake_history
    web_app.air_service.get_trends_data = _fake_trends
    web_app.air_service.list_alert_rules = _fake_alerts
    web_app.air_service.check_health = _fake_health
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            responses = [
                await client.get("/"),
                await client.get("/city/moscow"),
                await client.get("/history/moscow"),
                await client.get("/trends/moscow"),
                await client.get("/compare?cities=moscow,spb"),
                await client.get("/alerts/settings"),
                await client.get("/custom"),
            ]
    finally:
        web_app.air_service.get_current_data = original_current
        web_app.air_service.get_forecast_data = original_forecast
        web_app.air_service.get_history_data = original_history
        web_app.air_service.get_trends_data = original_trends
        web_app.air_service.list_alert_rules = original_alerts
        web_app.air_service.check_health = original_health

    for response in responses:
        assert response.status_code == 200
        assert "API с ограничениями" in response.text


@pytest.mark.asyncio
async def test_stage5_api_status_badge_prefers_public_status_over_engineering_status():
    original_current = web_app.air_service.get_current_data
    original_forecast = web_app.air_service.get_forecast_data
    original_history = web_app.air_service.get_history_data
    original_trends = web_app.air_service.get_trends_data
    original_alerts = web_app.air_service.list_alert_rules
    original_health = web_app.air_service.check_health

    async def _fake_current(lat: float, lon: float):
        return _current_payload(lat, lon)

    async def _fake_forecast(lat: float, lon: float, hours: int = 24):
        return [_current_payload(lat, lon, source="forecast")]

    async def _fake_history(**kwargs):
        return _history_payload()

    async def _fake_trends(**kwargs):
        return _trends_payload()

    async def _fake_alerts():
        return []

    async def _fake_health():
        return {"status": "degraded", "public_status": "healthy", "reachable": True}

    web_app.air_service.get_current_data = _fake_current
    web_app.air_service.get_forecast_data = _fake_forecast
    web_app.air_service.get_history_data = _fake_history
    web_app.air_service.get_trends_data = _fake_trends
    web_app.air_service.list_alert_rules = _fake_alerts
    web_app.air_service.check_health = _fake_health
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            responses = [
                await client.get("/"),
                await client.get("/city/moscow"),
                await client.get("/history/moscow"),
                await client.get("/trends/moscow"),
                await client.get("/compare?cities=moscow,spb"),
                await client.get("/alerts/settings"),
                await client.get("/custom"),
            ]
    finally:
        web_app.air_service.get_current_data = original_current
        web_app.air_service.get_forecast_data = original_forecast
        web_app.air_service.get_history_data = original_history
        web_app.air_service.get_trends_data = original_trends
        web_app.air_service.list_alert_rules = original_alerts
        web_app.air_service.check_health = original_health

    for response in responses:
        assert response.status_code == 200
        assert "API работает" in response.text
