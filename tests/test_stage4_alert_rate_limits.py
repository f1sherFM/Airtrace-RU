from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rate_limit_middleware import RateLimitMiddleware
from rate_limiter import RateLimiter
from rate_limit_types import EndpointCategory, RateLimitConfig


def _build_app() -> TestClient:
    app = FastAPI()

    @app.get("/v2/alerts")
    async def list_alerts():
        return {"ok": True}

    @app.post("/v2/alerts")
    async def create_alert():
        return {"ok": True}

    limiter = RateLimiter()
    limiter._redis_enabled = False
    limiter.configure_limits(
        EndpointCategory.ALERTS_READ,
        RateLimitConfig(requests_per_minute=1, burst_multiplier=1.0, window_size_seconds=60),
    )
    limiter.configure_limits(
        EndpointCategory.ALERTS_WRITE,
        RateLimitConfig(requests_per_minute=1, burst_multiplier=1.0, window_size_seconds=60),
    )

    app.add_middleware(
        RateLimitMiddleware,
        rate_limiter=limiter,
        enabled=True,
        skip_paths=[],
    )
    return TestClient(app)


def test_stage4_alert_rate_limits_use_separate_read_and_write_policies():
    client = _build_app()

    read_one = client.get("/v2/alerts")
    read_two = client.get("/v2/alerts")
    write_one = client.post("/v2/alerts")
    write_two = client.post("/v2/alerts")

    assert read_one.status_code == 200
    assert read_one.headers["X-RateLimit-Policy"] == "alerts-read"
    assert read_two.status_code == 429
    assert read_two.headers["X-RateLimit-Policy"] == "alerts-read"

    assert write_one.status_code == 200
    assert write_one.headers["X-RateLimit-Policy"] == "alerts-write"
    assert write_two.status_code == 429
    assert write_two.headers["X-RateLimit-Policy"] == "alerts-write"


def test_stage4_alert_rate_limits_keep_flat_v2_error_contract():
    client = _build_app()

    client.post("/v2/alerts")
    blocked = client.post("/v2/alerts")

    assert blocked.status_code == 429
    payload = blocked.json()
    assert payload["code"] == "RATE_LIMIT_EXCEEDED"
    assert "message" in payload
    assert "timestamp" in payload
    assert "details" in payload
    assert blocked.headers["X-AirTrace-API-Version"] == "2"
