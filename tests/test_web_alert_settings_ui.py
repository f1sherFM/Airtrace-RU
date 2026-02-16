"""
Smoke tests for alert settings UI flow (Issue #20).
"""

from pathlib import Path
import importlib.util
import os

import httpx
import pytest
from fastapi.templating import Jinja2Templates


def _load_web_app_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "web" / "web_app.py"
    previous_cwd = os.getcwd()
    try:
        os.chdir(repo_root / "web")
        spec = importlib.util.spec_from_file_location("web_app", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.templates = Jinja2Templates(directory=str(repo_root / "web" / "templates"))
        return module
    finally:
        os.chdir(previous_cwd)


web_app = _load_web_app_module()


@pytest.mark.asyncio
async def test_alert_settings_page_renders_russian_ui():
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
    html = response.text
    assert "Создать правило" in html
    assert "Текущие правила" in html
    assert "Telegram chat_id" in html


@pytest.mark.asyncio
async def test_alert_settings_crud_redirect_flow():
    calls = {"create": 0, "update": 0, "delete": 0}
    original_create = web_app.air_service.create_alert_rule
    original_update = web_app.air_service.update_alert_rule
    original_delete = web_app.air_service.delete_alert_rule

    async def _fake_create(payload):
        calls["create"] += 1
        return {"id": "r1", **payload}

    async def _fake_update(rule_id, payload):
        calls["update"] += 1
        return {"id": rule_id, **payload}

    async def _fake_delete(rule_id):
        calls["delete"] += 1
        return {"deleted": True, "rule_id": rule_id}

    web_app.air_service.create_alert_rule = _fake_create
    web_app.air_service.update_alert_rule = _fake_update
    web_app.air_service.delete_alert_rule = _fake_delete
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as client:
            create_resp = await client.post(
                "/alerts/settings/create",
                data={
                    "name": "AQI >= 140",
                    "aqi_threshold": "140",
                    "nmu_levels": "high,critical",
                    "cooldown_minutes": "30",
                    "channel": "telegram",
                    "chat_id": "777",
                    "enabled": "on",
                },
            )
            update_resp = await client.post(
                "/alerts/settings/update/r1",
                data={
                    "name": "AQI >= 150",
                    "aqi_threshold": "150",
                    "nmu_levels": "critical",
                    "cooldown_minutes": "45",
                    "channel": "telegram",
                    "chat_id": "777",
                    "enabled": "on",
                },
            )
            delete_resp = await client.post("/alerts/settings/delete/r1")
    finally:
        web_app.air_service.create_alert_rule = original_create
        web_app.air_service.update_alert_rule = original_update
        web_app.air_service.delete_alert_rule = original_delete

    assert create_resp.status_code == 303
    assert update_resp.status_code == 303
    assert delete_resp.status_code == 303
    assert calls == {"create": 1, "update": 1, "delete": 1}
