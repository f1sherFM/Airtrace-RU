from __future__ import annotations

from types import SimpleNamespace

from core import sentry as sentry_module


def test_sanitize_sentry_event_filters_sensitive_request_fields():
    event = {
        "request": {
            "headers": {"Authorization": "Bearer secret", "X-API-Key": "top-secret"},
            "data": {"lat": 55.7, "lon": 37.6, "chat_id": "5110137438", "safe": "ok"},
            "query_string": "lat=55.7&lon=37.6",
        },
        "extra": {"telegram_bot_token": "token", "safe": "ok"},
        "user": {"chat_id": "5110137438"},
    }

    sanitized = sentry_module.sanitize_sentry_event(event)

    assert sanitized["request"]["headers"]["Authorization"] == "[FILTERED_FOR_PRIVACY]"
    assert sanitized["request"]["headers"]["X-API-Key"] == "[FILTERED_FOR_PRIVACY]"
    assert sanitized["request"]["data"]["lat"] == "[FILTERED_FOR_PRIVACY]"
    assert sanitized["request"]["data"]["lon"] == "[FILTERED_FOR_PRIVACY]"
    assert sanitized["request"]["data"]["chat_id"] == "[FILTERED_FOR_PRIVACY]"
    assert sanitized["request"]["data"]["safe"] == "ok"
    assert sanitized["request"]["query_string"] == "[FILTERED_FOR_PRIVACY]"
    assert sanitized["extra"]["telegram_bot_token"] == "[FILTERED_FOR_PRIVACY]"
    assert sanitized["user"]["chat_id"] == "[FILTERED_FOR_PRIVACY]"


def test_init_sentry_no_dsn_is_noop(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    sentry_module._SENTRY_INITIALIZED = False

    assert sentry_module.init_sentry(app_role="api") is False


def test_init_sentry_uses_sdk_when_configured(monkeypatch):
    calls: list[dict[str, object]] = []

    class StubFastApiIntegration:
        pass

    def stub_init(**kwargs):
        calls.append(kwargs)

    def stub_import_module(name: str):
        if name == "sentry_sdk":
            return SimpleNamespace(init=stub_init)
        if name == "sentry_sdk.integrations.fastapi":
            return SimpleNamespace(FastApiIntegration=StubFastApiIntegration)
        raise ImportError(name)

    monkeypatch.setenv("SENTRY_DSN", "https://example@sentry.invalid/1")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "production")
    monkeypatch.setenv("SENTRY_RELEASE", "airtrace-v2-test")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.25")
    monkeypatch.setattr(sentry_module.importlib, "import_module", stub_import_module)
    sentry_module._SENTRY_INITIALIZED = False

    assert sentry_module.init_sentry(app_role="api") is True
    assert len(calls) == 1
    assert calls[0]["dsn"] == "https://example@sentry.invalid/1"
    assert calls[0]["environment"] == "production"
    assert calls[0]["release"] == "airtrace-v2-test"
    assert calls[0]["traces_sample_rate"] == 0.25
    assert calls[0]["send_default_pii"] is False
    assert calls[0]["before_send"] is sentry_module.sanitize_sentry_event
    assert isinstance(calls[0]["integrations"][0], StubFastApiIntegration)
