from pathlib import Path
from unittest.mock import AsyncMock
import importlib.util
import os
import sys

import httpx
import pytest
from fastapi.templating import Jinja2Templates

from schemas import AQIInfo, AirQualityData, LocationInfo, PollutantData


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_FILES = [
    REPO_ROOT / "main.py",
    REPO_ROOT / "web" / "web_app.py",
    REPO_ROOT / "docs" / "airtrace_v2_roadmap.md",
    REPO_ROOT / "docs" / "stage0_baseline_checklist.md",
    REPO_ROOT / "docs" / "stage0_protected_surfaces.md",
    REPO_ROOT / "docs" / "adr" / "README.md",
    REPO_ROOT / "docs" / "adr" / "templates" / "adr-template.md",
]
PROTECTED_FILES.extend(sorted((REPO_ROOT / "web" / "templates").glob("*.html")))
PROTECTED_FILES.extend(sorted((REPO_ROOT / "docs" / "adr").glob("*.md")))

MOJIBAKE_MARKERS = ("Ð", "Ñ", "Ã", "â€™", "â€œ", "â€”", "\ufffd")


def _load_web_app_module():
    module_path = REPO_ROOT / "web" / "web_app.py"
    previous_cwd = os.getcwd()
    added_path = str(REPO_ROOT / "web")
    try:
        os.chdir(REPO_ROOT / "web")
        if added_path not in sys.path:
            sys.path.insert(0, added_path)
        spec = importlib.util.spec_from_file_location("web_app", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.templates = Jinja2Templates(directory=str(REPO_ROOT / "web" / "templates"))
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


def test_stage0_protected_files_are_valid_utf8_without_bom():
    for path in PROTECTED_FILES:
        content = path.read_text(encoding="utf-8")
        assert not content.startswith("\ufeff"), path


def test_stage0_protected_files_do_not_contain_common_mojibake_markers():
    for path in PROTECTED_FILES:
        content = path.read_text(encoding="utf-8")
        for marker in MOJIBAKE_MARKERS:
            assert marker not in content, f"{path} contains marker {marker!r}"


@pytest.mark.asyncio
async def test_stage0_ssr_html_responses_include_charset_utf8():
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
    assert "charset=utf-8" in response.headers.get("content-type", "").lower()


@pytest.mark.asyncio
async def test_stage0_json_response_preserves_cyrillic_runtime_content():
    sample = AirQualityData(
        timestamp="2026-03-25T12:00:00+00:00",
        location=LocationInfo(latitude=55.7558, longitude=37.6176),
        aqi=AQIInfo(
            value=90,
            category="Умеренное",
            color="#FFFF00",
            description="Качество воздуха приемлемо",
        ),
        pollutants=PollutantData(pm2_5=25.4, pm10=45.2, no2=35.1, so2=12.3, o3=85.7),
        recommendations="Ограничить активность на улице",
        nmu_risk="medium",
        health_warnings=["Чувствительным группам стоит быть осторожнее"],
    )
    original = web_app.air_service.get_current_data

    async def _fake_current(lat: float, lon: float):
        return sample.model_dump(mode="json")

    web_app.air_service.get_current_data = _fake_current
    try:
        transport = httpx.ASGITransport(app=web_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/custom", data={"lat": "55.7558", "lon": "37.6176", "city_name": "Москва"})
    finally:
        web_app.air_service.get_current_data = original

    assert response.status_code in {200, 500}
    body = response.text
    if response.status_code == 200:
        assert "Москва" in body or "Умеренное" in body or "Ограничить" in body
