"""Operational and observability routes preserved during Stage 1."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from config import config
from core.privacy_validation import privacy_validator
from rate_limit_middleware import get_rate_limit_manager
from rate_limit_monitoring import get_rate_limit_monitor
from unified_weather_service import unified_weather_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/metrics", include_in_schema=False)
async def get_metrics():
    try:
        unified_stats = await unified_weather_service.get_service_statistics()
        cache_stats = await unified_weather_service.cache_manager.get_stats()
        metrics = {
            "cache_entries": cache_stats.key_count,
            "cache_hit_rate": cache_stats.hit_rate,
            "cache_miss_rate": cache_stats.miss_rate,
            "service_status": "running",
            "unified_weather_service": unified_stats,
            "components": {
                "aqi_calculator": "active",
                "nmu_detector": "active",
                "privacy_middleware": "active",
                "multi_level_cache": "active",
                "weather_api_integration": "active" if config.weather_api.enabled else "disabled",
            },
        }
        if config.performance.rate_limiting_enabled:
            try:
                metrics["rate_limiting"] = await get_rate_limit_manager().get_comprehensive_stats()
                metrics["components"]["rate_limiting"] = "active"
            except Exception as exc:
                logger.warning("Failed to get rate limiting metrics: %s", exc)
                metrics["components"]["rate_limiting"] = "error"
        else:
            metrics["components"]["rate_limiting"] = "disabled"

        if config.performance.connection_pooling_enabled:
            try:
                from connection_pool import get_connection_pool_manager

                metrics["connection_pools"] = await get_connection_pool_manager().get_all_stats()
                metrics["components"]["connection_pooling"] = "active"
            except Exception as exc:
                logger.warning("Failed to get connection pool metrics: %s", exc)
                metrics["components"]["connection_pooling"] = "error"
        else:
            metrics["components"]["connection_pooling"] = "disabled"
        return metrics
    except Exception as exc:
        logger.error("Error getting metrics: %s", exc)
        return {"error": "Metrics unavailable"}


@router.get("/metrics/prometheus", include_in_schema=False, response_class=PlainTextResponse)
async def get_prometheus_metrics():
    try:
        from prometheus_exporter import get_prometheus_exporter

        return await get_prometheus_exporter().export_metrics()
    except Exception as exc:
        logger.error("Error exporting Prometheus metrics: %s", exc)
        return "# Error exporting metrics\n"


@router.get("/metrics/validate", include_in_schema=False)
async def validate_prometheus_metrics():
    try:
        from prometheus_exporter import get_prometheus_exporter

        return await get_prometheus_exporter().validate_metrics_completeness()
    except Exception as exc:
        logger.error("Error validating Prometheus metrics: %s", exc)
        return {"error": "Metrics validation failed", "details": str(exc)}


@router.get("/metrics/comprehensive", include_in_schema=False)
async def get_comprehensive_metrics():
    try:
        comprehensive_metrics = {}

        try:
            from performance_monitor import get_performance_monitor

            monitor = get_performance_monitor()
            comprehensive_metrics["performance"] = {
                "stats": monitor.get_performance_stats().__dict__,
                "endpoint_stats": monitor.get_endpoint_stats(),
                "summary": monitor.get_metrics_summary(),
            }
        except Exception as exc:
            comprehensive_metrics["performance"] = {"error": str(exc)}

        try:
            if hasattr(unified_weather_service, "cache_manager"):
                cache_stats = await unified_weather_service.cache_manager.get_stats()
                comprehensive_metrics["cache"] = cache_stats.__dict__
        except Exception as exc:
            comprehensive_metrics["cache"] = {"error": str(exc)}

        try:
            from rate_limiter import get_rate_limiter

            comprehensive_metrics["rate_limiting"] = (await get_rate_limiter().get_metrics()).__dict__
        except Exception as exc:
            comprehensive_metrics["rate_limiting"] = {"error": str(exc)}

        try:
            from connection_pool import get_connection_pool_manager

            comprehensive_metrics["connection_pools"] = await get_connection_pool_manager().get_all_stats()
        except Exception as exc:
            comprehensive_metrics["connection_pools"] = {"error": str(exc)}

        try:
            from resource_manager import get_resource_manager

            comprehensive_metrics["resource_management"] = (await get_resource_manager().get_resource_usage()).__dict__
        except Exception as exc:
            comprehensive_metrics["resource_management"] = {"error": str(exc)}

        try:
            from system_monitor import get_system_monitor

            system_monitor = get_system_monitor()
            if system_monitor.metrics_history:
                comprehensive_metrics["system"] = system_monitor.metrics_history[-1].__dict__
        except Exception as exc:
            comprehensive_metrics["system"] = {"error": str(exc)}

        try:
            from weather_api_manager import get_weather_api_manager

            comprehensive_metrics["weatherapi"] = await get_weather_api_manager().get_api_status()
        except Exception as exc:
            comprehensive_metrics["weatherapi"] = {"error": str(exc)}

        try:
            from request_optimizer import get_request_optimizer

            comprehensive_metrics["request_optimization"] = await get_request_optimizer().get_optimization_stats()
        except Exception as exc:
            comprehensive_metrics["request_optimization"] = {"error": str(exc)}

        try:
            from prometheus_exporter import get_prometheus_exporter

            exporter = get_prometheus_exporter()
            comprehensive_metrics["alerts"] = {
                "active_alerts": [alert.__dict__ for alert in exporter.get_active_alerts()],
                "recent_history": [alert.__dict__ for alert in exporter.get_alert_history(limit=10)],
            }
        except Exception as exc:
            comprehensive_metrics["alerts"] = {"error": str(exc)}

        return comprehensive_metrics
    except Exception as exc:
        logger.error("Error getting comprehensive metrics: %s", exc)
        return {"error": "Comprehensive metrics unavailable", "details": str(exc)}


@router.get("/rate-limit-status", include_in_schema=False)
async def get_rate_limit_status():
    if not config.performance.rate_limiting_enabled:
        return {"status": "disabled", "message": "Rate limiting is not enabled"}
    try:
        return {
            "status": "enabled",
            "comprehensive_stats": await get_rate_limit_manager().get_comprehensive_stats(),
            "recent_violations": get_rate_limit_monitor().get_violation_summary(),
            "endpoint_statistics": get_rate_limit_monitor().get_endpoint_statistics(),
        }
    except Exception as exc:
        logger.error("Error getting rate limit status: %s", exc)
        return {"status": "error", "error": str(exc)}


@router.get("/privacy-compliance", include_in_schema=False)
async def get_privacy_compliance_status():
    try:
        privacy_validator.reset()
        test_cache_key = unified_weather_service.cache_manager._generate_key(55.7558, 37.6176)
        privacy_validator.validate_cache_key_privacy(test_cache_key, "privacy_compliance_check")
        test_metrics = await unified_weather_service.get_service_statistics()
        privacy_validator.validate_metrics_anonymization(test_metrics, "privacy_compliance_check")

        if config.performance.monitoring_enabled:
            try:
                from performance_monitor import PerformanceMonitor

                monitor = PerformanceMonitor()
                privacy_validator.validate_performance_monitoring_privacy(
                    {
                        "endpoint": "/weather/current",
                        "method": "GET",
                        "duration": 0.5,
                        "status_code": 200,
                    },
                    "privacy_compliance_check",
                )
                del monitor
            except Exception as exc:
                logger.warning("Performance monitoring privacy check failed: %s", exc)

        report = privacy_validator.generate_compliance_report()
        return {
            "privacy_compliance": {
                "is_compliant": report.is_compliant,
                "compliance_score": report.compliance_score,
                "total_checks": report.total_checks,
                "passed_checks": report.passed_checks,
                "violations_count": len(report.violations),
                "warnings_count": len(report.warnings),
                "validation_timestamp": report.validation_timestamp.isoformat(),
            },
            "violations": [
                {
                    "type": violation.violation_type.value,
                    "description": violation.description,
                    "location": violation.location,
                    "severity": violation.severity,
                    "timestamp": violation.timestamp.isoformat(),
                }
                for violation in report.violations
            ],
            "warnings": report.warnings,
            "recommendations": report.recommendations,
            "privacy_settings": {
                "coordinate_hashing_enabled": config.cache.hash_coordinates,
                "coordinate_precision": config.cache.coordinate_precision,
                "cache_key_prefix": config.cache.l2_key_prefix,
                "privacy_middleware_enabled": True,
            },
        }
    except Exception as exc:
        logger.error("Privacy compliance check failed: %s", exc)
        return {"privacy_compliance": {"is_compliant": False, "compliance_score": 0.0, "error": str(exc)}}


@router.get("/system-status", include_in_schema=False)
async def get_system_status():
    try:
        from graceful_degradation import get_graceful_degradation_manager

        comprehensive_status = await get_graceful_degradation_manager().get_comprehensive_health_status()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "degradation_status": comprehensive_status,
            "configuration": {
                "rate_limiting_enabled": config.performance.rate_limiting_enabled,
                "connection_pooling_enabled": config.performance.connection_pooling_enabled,
                "weather_api_enabled": config.weather_api.enabled,
                "redis_enabled": config.performance.redis_enabled,
                "monitoring_enabled": getattr(config.performance, "monitoring_enabled", False),
            },
        }
    except Exception as exc:
        logger.error("System status check failed: %s", exc)
        return {"timestamp": datetime.now(timezone.utc).isoformat(), "error": str(exc), "status": "error"}


@router.get("/config-audit", include_in_schema=False)
async def get_configuration_audit():
    try:
        audit_manager = config.get_audit_manager()
        if not audit_manager:
            return {"status": "disabled", "message": "Configuration audit is not enabled"}
        audit_trail = audit_manager.get_audit_trail(limit=100)
        impact_summary = audit_manager.get_performance_impact_summary(time_range=timedelta(hours=24))
        formatted_trail = []
        for entry in audit_trail:
            entry_data = {
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp.isoformat(),
                "change": {
                    "change_id": entry.change.change_id,
                    "component": entry.change.component,
                    "setting_path": entry.change.setting_path,
                    "old_value": entry.change.old_value,
                    "new_value": entry.change.new_value,
                    "change_type": entry.change.change_type,
                    "source": entry.change.source,
                    "validation_status": entry.change.validation_status,
                    "validation_errors": entry.change.validation_errors,
                },
                "rollback_available": entry.rollback_available,
            }
            if entry.performance_impact:
                entry_data["performance_impact"] = {
                    "impact_severity": entry.performance_impact.impact_severity,
                    "impact_metrics": entry.performance_impact.impact_metrics,
                    "recommendations": entry.performance_impact.recommendations,
                }
            formatted_trail.append(entry_data)
        return {
            "status": "enabled",
            "audit_trail": formatted_trail,
            "performance_impact_summary": impact_summary,
            "total_entries": len(formatted_trail),
        }
    except Exception as exc:
        logger.error("Error getting configuration audit: %s", exc)
        return {"status": "error", "error": str(exc)}


@router.get("/config-audit/component/{component}", include_in_schema=False)
async def get_component_configuration_audit(component: str):
    try:
        audit_manager = config.get_audit_manager()
        if not audit_manager:
            return {"status": "disabled", "message": "Configuration audit is not enabled"}
        audit_trail = audit_manager.get_audit_trail(component=component, time_range=timedelta(days=7), limit=50)
        formatted_trail = []
        for entry in audit_trail:
            entry_data = {
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp.isoformat(),
                "setting_path": entry.change.setting_path,
                "old_value": entry.change.old_value,
                "new_value": entry.change.new_value,
                "change_type": entry.change.change_type,
                "source": entry.change.source,
                "validation_status": entry.change.validation_status,
            }
            if entry.performance_impact:
                entry_data["impact_severity"] = entry.performance_impact.impact_severity
                entry_data["impact_metrics"] = entry.performance_impact.impact_metrics
            formatted_trail.append(entry_data)
        return {"component": component, "audit_trail": formatted_trail, "total_entries": len(formatted_trail)}
    except Exception as exc:
        logger.error("Error getting component configuration audit: %s", exc)
        return {"status": "error", "error": str(exc)}


@router.post("/config-audit/snapshot", include_in_schema=False)
async def create_configuration_snapshot():
    try:
        audit_manager = config.get_audit_manager()
        if not audit_manager:
            return {"status": "disabled", "message": "Configuration audit is not enabled"}
        snapshot_id = audit_manager.create_configuration_snapshot()
        if snapshot_id:
            return {"status": "success", "snapshot_id": snapshot_id, "timestamp": datetime.now(timezone.utc).isoformat()}
        return {"status": "error", "message": "Failed to create configuration snapshot"}
    except Exception as exc:
        logger.error("Error creating configuration snapshot: %s", exc)
        return {"status": "error", "error": str(exc)}


@router.get("/config-audit/performance-impact", include_in_schema=False)
async def get_configuration_performance_impact():
    try:
        audit_manager = config.get_audit_manager()
        if not audit_manager:
            return {"status": "disabled", "message": "Configuration audit is not enabled"}
        return {
            "status": "enabled",
            "performance_impact": {
                "last_24_hours": audit_manager.get_performance_impact_summary(timedelta(hours=24)),
                "last_7_days": audit_manager.get_performance_impact_summary(timedelta(days=7)),
                "last_30_days": audit_manager.get_performance_impact_summary(timedelta(days=30)),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.error("Error getting configuration performance impact: %s", exc)
        return {"status": "error", "error": str(exc)}


@router.get("/version", include_in_schema=False)
async def get_version():
    return {
        "service": "AirTrace RU Backend",
        "version": "0.3.1",
        "api_version": "v1",
        "features": [
            "Russian AQI calculation",
            "NMU risk detection",
            "Privacy protection",
            "Data caching",
            "Async processing",
            "Configuration audit trail",
        ],
    }
