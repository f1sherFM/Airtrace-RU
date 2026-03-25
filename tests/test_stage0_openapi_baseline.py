from fastapi.testclient import TestClient

from main import app


def test_stage0_openapi_baseline_exposes_protected_routes_and_schemas():
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200

    payload = response.json()
    assert "paths" in payload
    assert "components" in payload
    assert "schemas" in payload["components"]

    for route in (
        "/weather/current",
        "/weather/forecast",
        "/history",
        "/health",
        "/history/export/json",
        "/history/export/csv",
        "/v2/current",
        "/v2/forecast",
        "/v2/history",
        "/v2/health",
    ):
        assert route in payload["paths"], route

    assert "AirQualityData" in payload["components"]["schemas"]
    assert "HealthCheckResponse" in payload["components"]["schemas"]
    assert "get" in payload["paths"]["/weather/current"]
    assert "get" in payload["paths"]["/v2/history"]
