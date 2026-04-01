"""Optional Sentry bootstrap with privacy-aware event sanitization."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_SENTRY_INITIALIZED = False
_REDACTED = "[FILTERED_FOR_PRIVACY]"

_SENSITIVE_KEYWORDS = {
    "alerts_api_key",
    "api_key",
    "authorization",
    "chat_id",
    "coordinates",
    "cookie",
    "dsn",
    "lat",
    "latitude",
    "lon",
    "longitude",
    "set-cookie",
    "telegram_bot_token",
    "token",
    "x-api-key",
}


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(keyword.replace("-", "_") in normalized for keyword in _SENSITIVE_KEYWORDS)


def _sanitize_value(key: str | None, value: Any) -> Any:
    if key and _is_sensitive_key(key):
        return _REDACTED
    if isinstance(value, dict):
        return {nested_key: _sanitize_value(str(nested_key), nested_value) for nested_key, nested_value in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(key, item) for item in value)
    return value


def sanitize_sentry_event(event: dict[str, Any], hint: dict[str, Any] | None = None) -> dict[str, Any]:
    sanitized = dict(event)

    request = dict(sanitized.get("request") or {})
    if request:
        request["headers"] = _sanitize_value("headers", request.get("headers"))
        request["data"] = _sanitize_value("data", request.get("data"))
        request["cookies"] = _sanitize_value("cookies", request.get("cookies"))
        request["query_string"] = _REDACTED if request.get("query_string") else request.get("query_string")
        sanitized["request"] = request

    if "extra" in sanitized:
        sanitized["extra"] = _sanitize_value("extra", sanitized["extra"])
    if "contexts" in sanitized:
        sanitized["contexts"] = _sanitize_value("contexts", sanitized["contexts"])
    if "user" in sanitized:
        sanitized["user"] = _sanitize_value("user", sanitized["user"])
    if "tags" in sanitized:
        sanitized["tags"] = _sanitize_value("tags", sanitized["tags"])

    return sanitized


def _get_traces_sample_rate() -> float:
    raw = os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0")
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid SENTRY_TRACES_SAMPLE_RATE=%r, falling back to 0", raw)
        return 0.0


def init_sentry(*, app_role: str) -> bool:
    global _SENTRY_INITIALIZED

    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return False

    if _SENTRY_INITIALIZED:
        return True

    try:
        sentry_sdk = importlib.import_module("sentry_sdk")
        fastapi_module = importlib.import_module("sentry_sdk.integrations.fastapi")
    except ImportError:
        logger.warning("Sentry requested but sentry-sdk is not installed")
        return False

    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        release=os.getenv("SENTRY_RELEASE", "airtrace-v2"),
        traces_sample_rate=_get_traces_sample_rate(),
        send_default_pii=False,
        before_send=sanitize_sentry_event,
        integrations=[fastapi_module.FastApiIntegration()],
    )
    _SENTRY_INITIALIZED = True
    logger.info("Sentry initialized for %s", app_role)
    return True
