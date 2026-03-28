import httpx
import pytest

import main


@pytest.mark.asyncio
async def test_v2_history_requires_exactly_one_locator():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get("/v2/history?range=24h")
        partial = await client.get("/v2/history?range=24h&lat=55.7558")
        both = await client.get("/v2/history?range=24h&city=moscow&lat=55.7558&lon=37.6176")
        unknown = await client.get("/v2/history?range=24h&city=unknown-city")

    assert missing.status_code == 400
    assert missing.json()["code"] == "VALIDATION_ERROR"
    assert partial.status_code == 400
    assert partial.json()["code"] == "VALIDATION_ERROR"
    assert both.status_code == 400
    assert both.json()["code"] == "VALIDATION_ERROR"
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_v2_trends_requires_exactly_one_locator():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get("/v2/trends?range=7d")
        partial = await client.get("/v2/trends?range=7d&lon=37.6176")
        both = await client.get("/v2/trends?range=7d&city=moscow&lat=55.7558&lon=37.6176")
        unknown = await client.get("/v2/trends?range=7d&city=unknown-city")

    assert missing.status_code == 400
    assert missing.json()["code"] == "VALIDATION_ERROR"
    assert partial.status_code == 400
    assert partial.json()["code"] == "VALIDATION_ERROR"
    assert both.status_code == 400
    assert both.json()["code"] == "VALIDATION_ERROR"
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "NOT_FOUND"
