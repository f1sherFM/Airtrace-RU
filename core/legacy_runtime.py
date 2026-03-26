"""Compatibility accessors for legacy main-module runtime state."""

from __future__ import annotations

import sys
from typing import Any, Callable

from connection_pool import get_connection_pool_manager as default_get_connection_pool_manager


_RUNTIME_DEFAULTS: dict[str, Any] = {
    "air_quality_service": None,
    "history_ingestion_pipeline": None,
    "history_snapshot_store": None,
    "alert_rule_engine": None,
    "alert_subscription_service": None,
    "telegram_delivery_service": None,
}


def _main_module():
    return sys.modules.get("main")


def get_runtime_value(name: str, default: Any = None) -> Any:
    module = _main_module()
    if module is not None and hasattr(module, name):
        return getattr(module, name)
    return _RUNTIME_DEFAULTS.get(name, default)


def set_runtime_value(name: str, value: Any) -> None:
    _RUNTIME_DEFAULTS[name] = value
    module = _main_module()
    if module is not None:
        setattr(module, name, value)


def get_air_quality_service() -> Any:
    return get_runtime_value("air_quality_service")


def set_air_quality_service(service: Any) -> None:
    set_runtime_value("air_quality_service", service)


def get_history_ingestion_pipeline() -> Any:
    return get_runtime_value("history_ingestion_pipeline")


def set_history_ingestion_pipeline(pipeline: Any) -> None:
    set_runtime_value("history_ingestion_pipeline", pipeline)


def get_history_snapshot_store() -> Any:
    return get_runtime_value("history_snapshot_store")


def set_history_snapshot_store(store: Any) -> None:
    set_runtime_value("history_snapshot_store", store)


def get_alert_rule_engine() -> Any:
    return get_runtime_value("alert_rule_engine")


def get_alert_subscription_service() -> Any:
    return get_runtime_value("alert_subscription_service")


def set_alert_subscription_service(service: Any) -> None:
    set_runtime_value("alert_subscription_service", service)


def get_telegram_delivery_service() -> Any:
    return get_runtime_value("telegram_delivery_service")


def get_connection_pool_manager_callable() -> Callable[..., Any]:
    module = _main_module()
    if module is not None and hasattr(module, "get_connection_pool_manager"):
        return getattr(module, "get_connection_pool_manager")
    return default_get_connection_pool_manager
