"""Shared health query logic and normalization helpers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from config import config
from core.legacy_runtime import get_connection_pool_manager_callable
from graceful_degradation import get_graceful_degradation_manager
from schemas import HealthCheckResponse
from unified_weather_service import unified_weather_service
from domain.aqi.calculator import AQICalculator
from domain.nmu.detector import check_nmu_risk

from .readonly import get_or_create_air_quality_service

logger = logging.getLogger(__name__)


def _normalize_health_status(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"healthy", "ok", "up", "enabled", "active"}:
        return "healthy"
    if normalized in {"disabled"}:
        return "degraded"
    if normalized in {"unhealthy", "down", "error", "failed"}:
        return "unhealthy"
    if normalized in {"degraded", "warning", "unknown"}:
        return "degraded"
    return "degraded"


def _normalize_health_component(value: Any) -> dict[str, Any]:
    def _merge_statuses(statuses: list[str]) -> str:
        normalized = [_normalize_health_status(s) for s in statuses]
        if not normalized:
            return "degraded"
        if any(s == "unhealthy" for s in normalized):
            return "unhealthy"
        if all(s == "healthy" for s in normalized):
            return "healthy"
        return "degraded"

    def _infer_status_from_mapping(mapping: dict[str, Any]) -> str:
        derived_statuses: list[str] = []
        for item in mapping.values():
            if isinstance(item, dict):
                nested_status = item.get("status")
                if nested_status is not None:
                    derived_statuses.append(str(nested_status))
                else:
                    derived_statuses.append(_infer_status_from_mapping(item))
            elif isinstance(item, str):
                derived_statuses.append(item)
            elif isinstance(item, bool):
                derived_statuses.append("healthy" if item else "unhealthy")
            else:
                derived_statuses.append("degraded")
        return _merge_statuses(derived_statuses)

    if isinstance(value, dict):
        raw_status = value.get("status")
        if raw_status is None:
            raw_status = _infer_status_from_mapping(value)
        normalized_status = _normalize_health_status(str(raw_status))
        details = {k: v for k, v in value.items() if k != "status"}
        return {"status": normalized_status, "details": details}
    if isinstance(value, str):
        return {"status": _normalize_health_status(value), "details": {}}
    return {"status": "degraded", "details": {"raw": value}}


def _is_optional_health_component(name: str) -> bool:
    if name.startswith("pool_"):
        return True
    return name in {
        "external_api",
        "weather_api",
        "cache",
        "connection_pools",
        "rate_limiting",
        "fallback_capabilities",
    }


def _derive_overall_health_status(normalized_services: dict[str, dict[str, Any]]) -> str:
    critical_statuses: list[str] = []
    optional_statuses: list[str] = []

    for name, component in normalized_services.items():
        status = _normalize_health_status(component.get("status", "degraded"))
        if _is_optional_health_component(name):
            optional_statuses.append(status)
        else:
            critical_statuses.append(status)

    if any(status == "unhealthy" for status in critical_statuses):
        return "unhealthy"
    if any(status == "degraded" for status in critical_statuses):
        return "degraded"
    if any(status in {"unhealthy", "degraded"} for status in optional_statuses):
        return "degraded"
    return "healthy"


def _derive_public_health_status(normalized_services: dict[str, dict[str, Any]]) -> str:
    core_components = (
        "api",
        "external_api",
        "aqi_calculator",
        "nmu_detector",
    )
    statuses = [
        _normalize_health_status(normalized_services.get(name, {}).get("status", "degraded"))
        for name in core_components
    ]

    if any(status == "unhealthy" for status in statuses):
        return "unhealthy"
    if any(status == "degraded" for status in statuses):
        return "degraded"
    return "healthy"


async def query_health() -> HealthCheckResponse:
    services_status: dict[str, Any] = {}
    overall_status = "healthy"
    aqi_calculator = AQICalculator()

    try:
        services_status["api"] = "healthy"

        degradation_manager = get_graceful_degradation_manager()
        degradation_status = await degradation_manager.get_comprehensive_health_status()
        services_status["graceful_degradation"] = {
            "status": degradation_status["overall_status"],
            "system_under_stress": degradation_status["system_under_stress"],
            "prioritize_core_functionality": degradation_status["prioritize_core_functionality"],
            "stale_data_entries": degradation_status["stale_data_entries"],
            "fallback_statistics": degradation_status["fallback_statistics"],
        }
        if degradation_status["overall_status"] == "unhealthy":
            overall_status = "unhealthy"
        elif degradation_status["overall_status"] == "degraded" and overall_status == "healthy":
            overall_status = "degraded"

        service = get_or_create_air_quality_service()

        try:
            external_api_status = await service.check_external_api_health()
            services_status["external_api"] = external_api_status
            if external_api_status not in ["healthy"] and overall_status == "healthy":
                overall_status = "degraded"
        except Exception as exc:
            logger.error("External API health check failed: %s", exc)
            services_status["external_api"] = "unhealthy"
            overall_status = "unhealthy"

        try:
            cache_status = service.cache_manager.get_status()
            services_status["cache"] = cache_status
            if "unhealthy" in cache_status and overall_status == "healthy":
                overall_status = "degraded"
        except Exception as exc:
            logger.error("Cache health check failed: %s", exc)
            services_status["cache"] = "unhealthy"
            if overall_status == "healthy":
                overall_status = "degraded"

        try:
            aqi_value, category, color = aqi_calculator.calculate_aqi({"pm2_5": 25.0, "pm10": 50.0})
            services_status["aqi_calculator"] = "healthy" if aqi_value > 0 and category and color else "unhealthy"
            if services_status["aqi_calculator"] == "unhealthy" and overall_status == "healthy":
                overall_status = "degraded"
        except Exception as exc:
            logger.error("AQI calculator health check failed: %s", exc)
            services_status["aqi_calculator"] = "unhealthy"
            if overall_status == "healthy":
                overall_status = "degraded"

        try:
            from middleware import get_privacy_middleware

            services_status["privacy_middleware"] = "healthy" if get_privacy_middleware() else "degraded"
        except Exception as exc:
            logger.error("Privacy middleware health check failed: %s", exc)
            services_status["privacy_middleware"] = "unhealthy"

        try:
            from rate_limit_middleware import get_rate_limit_manager

            if config.performance.rate_limiting_enabled:
                rate_limit_manager = get_rate_limit_manager()
                if rate_limit_manager and rate_limit_manager.is_enabled():
                    services_status["rate_limiting"] = "healthy"
                else:
                    services_status["rate_limiting"] = "unhealthy"
                    if overall_status == "healthy":
                        overall_status = "degraded"
            else:
                services_status["rate_limiting"] = "disabled"
        except Exception as exc:
            logger.error("Rate limiting health check failed: %s", exc)
            services_status["rate_limiting"] = "unhealthy"

        try:
            nmu_risk = check_nmu_risk({"pm2_5": 30.0, "pm10": 60.0})
            services_status["nmu_detector"] = "healthy" if nmu_risk in {"low", "medium", "high", "critical"} else "degraded"
        except Exception as exc:
            logger.error("NMU detector health check failed: %s", exc)
            services_status["nmu_detector"] = "unhealthy"

        try:
            if config.performance.connection_pooling_enabled:
                pool_manager = get_connection_pool_manager_callable()()
                pool_health = await pool_manager.health_check_all()
                all_pools_healthy = all(pool_health.values())
                services_status["connection_pools"] = "healthy" if all_pools_healthy else "degraded"
                for service_name, healthy in pool_health.items():
                    services_status[f"pool_{service_name}"] = "healthy" if healthy else "unhealthy"
                    if not healthy and overall_status == "healthy":
                        overall_status = "degraded"
            else:
                services_status["connection_pools"] = "disabled"
        except Exception as exc:
            logger.error("Connection pool health check failed: %s", exc)
            services_status["connection_pools"] = "unhealthy"

        try:
            weather_api_health = await unified_weather_service.check_weather_api_health()
            services_status["weather_api"] = weather_api_health["status"]
            if weather_api_health["status"] == "unhealthy" and overall_status == "healthy":
                overall_status = "degraded"
        except Exception as exc:
            logger.error("WeatherAPI health check failed: %s", exc)
            services_status["weather_api"] = "unhealthy"

        services_status["fallback_capabilities"] = {
            "stale_data_serving": "enabled",
            "cached_response_serving": "enabled",
            "minimal_response_generation": "enabled",
            "core_functionality_prioritization": "enabled",
        }

        normalized_services = {name: _normalize_health_component(value) for name, value in services_status.items()}
        overall_from_components = _derive_overall_health_status(normalized_services)
        normalized_overall = _normalize_health_status(overall_status)
        if overall_from_components == "unhealthy":
            normalized_overall = "unhealthy"
        elif overall_from_components == "degraded" and normalized_overall == "healthy":
            normalized_overall = "degraded"

        public_status = _derive_public_health_status(normalized_services)
        return HealthCheckResponse(
            status=normalized_overall,
            public_status=public_status,
            services=normalized_services,
        )
    except Exception as exc:
        logger.error("Health check failed with unexpected error: %s", exc)
        return HealthCheckResponse(
            status="unhealthy",
            public_status="unhealthy",
            services={
                "api": {"status": "healthy", "details": {}},
                "external_api": {"status": "degraded", "details": {"reason": "unknown"}},
                "cache": {"status": "degraded", "details": {"reason": "unknown"}},
                "aqi_calculator": {"status": "degraded", "details": {"reason": "unknown"}},
                "privacy_middleware": {"status": "degraded", "details": {"reason": "unknown"}},
                "nmu_detector": {"status": "degraded", "details": {"reason": "unknown"}},
                "graceful_degradation": {"status": "degraded", "details": {"reason": "unknown"}},
                "fallback_capabilities": {"status": "degraded", "details": {"reason": "unknown"}},
            },
        )


async def query_liveness() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "airtrace-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def query_readiness() -> dict[str, Any]:
    health_payload = await query_health()
    overall = _normalize_health_status(health_payload.status)
    ready = overall != "unhealthy"
    reasons: list[str] = []
    for name, component in health_payload.services.items():
        status = _normalize_health_status(component.get("status", "degraded"))
        if _is_optional_health_component(name):
            continue
        if status == "unhealthy":
            reasons.append(f"{name}:unhealthy")

    return {
        "status": "ready" if ready else "not_ready",
        "overall": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reasons": reasons,
    }
