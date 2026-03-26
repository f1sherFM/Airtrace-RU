"""View-model builders for the Python SSR layer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from core.settings import get_cities_mapping

from .service import WebAppService


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_pollutants(payload: dict[str, Any]) -> dict[str, Any]:
    pollutants = dict(payload.get("pollutants") or {})
    for key in ("pm2_5", "pm10", "no2", "so2", "o3"):
        pollutants.setdefault(key, None)
    return pollutants


def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("data_source", payload.get("data_source", "live"))
    metadata.setdefault("freshness", payload.get("freshness", "unknown"))
    metadata.setdefault("confidence", payload.get("confidence", 0.0))
    metadata.setdefault("confidence_explanation", payload.get("confidence_explanation"))
    metadata.setdefault("fallback_used", payload.get("fallback_used", False))
    metadata.setdefault("cache_age_seconds", payload.get("cache_age_seconds"))
    return metadata


def build_explainability(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata_from_payload(payload)
    return {
        "source": metadata.get("data_source", "unknown"),
        "freshness": metadata.get("freshness", "unknown"),
        "confidence": metadata.get("confidence", 0.0),
        "confidence_explanation": metadata.get("confidence_explanation"),
        "fallback_used": metadata.get("fallback_used", False),
        "cache_age_seconds": metadata.get("cache_age_seconds"),
    }


def normalize_current_payload(payload: dict[str, Any], *, lat: float, lon: float) -> dict[str, Any]:
    normalized = dict(payload or {})
    aqi = dict(normalized.get("aqi") or {})
    aqi.setdefault("value", 0)
    aqi.setdefault("category", "Нет данных")
    aqi.setdefault("color", "#FFFFFF")
    aqi.setdefault("description", "Описание AQI пока недоступно")
    normalized["aqi"] = aqi
    normalized["location"] = normalized.get("location") or {"latitude": lat, "longitude": lon}
    normalized["pollutants"] = _normalize_pollutants(normalized)
    normalized.setdefault("timestamp", _utc_now_iso())
    normalized.setdefault("recommendations", "Рекомендации пока недоступны")
    normalized.setdefault("nmu_risk", "low")
    normalized.setdefault("health_warnings", [])
    normalized["metadata"] = _metadata_from_payload(normalized)
    return normalized


def normalize_forecast_payload(items: list[dict[str, Any]], *, lat: float, lon: float) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items or []:
        normalized.append(normalize_current_payload(item, lat=lat, lon=lon))
    return normalized


def normalize_history_payload(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items or []:
        record = dict(item)
        record.setdefault("timestamp", record.get("snapshot_hour_utc", _utc_now_iso()))
        record.setdefault("aqi", 0)
        record.setdefault("anomaly_detected", False)
        record.setdefault("anomaly_type", None)
        record["metadata"] = _metadata_from_payload(record)
        normalized.append(record)
    return normalized


async def build_index_context(*, request: Any, service: WebAppService) -> dict[str, Any]:
    health = await service.check_health()
    return {
        "request": request,
        "cities": get_cities_mapping(),
        "api_status": health.get("status", "degraded"),
        "api_reachable": bool(health.get("reachable", False)),
        "title": "AirTrace RU - Мониторинг качества воздуха",
    }


async def build_city_page_context(
    *,
    request: Any,
    service: WebAppService,
    city_key: str,
    city: dict[str, Any],
    is_custom: bool = False,
) -> dict[str, Any]:
    lat = float(city["lat"])
    lon = float(city["lon"])
    current_raw, forecast_raw, history_raw, trends_raw = await asyncio.gather(
        service.get_current_data(lat, lon),
        service.get_forecast_data(lat, lon),
        service.get_history_data(
            city_key="" if is_custom else city_key,
            lat=lat,
            lon=lon,
            range_preset="24h",
            page_size=48,
        ),
        service.get_trends_data(
            city_key="" if is_custom else city_key,
            lat=lat if is_custom else None,
            lon=lon if is_custom else None,
            range_preset="7d",
        ),
    )
    current = normalize_current_payload(current_raw, lat=lat, lon=lon)
    forecast = normalize_forecast_payload(forecast_raw[:8], lat=lat, lon=lon)
    history_items = normalize_history_payload((history_raw or {}).get("items", [])[:12])
    explainability = build_explainability(current)
    return {
        "request": request,
        "cities": get_cities_mapping(),
        "current_city": city,
        "city_key": city_key,
        "data": current,
        "forecast": forecast,
        "history": history_items,
        "trends": trends_raw,
        "trend_summary": (trends_raw or {}).get("summary"),
        "explainability": explainability,
        "api_status": "healthy",
        "title": f"AirTrace RU - {city['name']}",
        "is_custom": is_custom,
    }


async def build_history_page_context(
    *,
    request: Any,
    service: WebAppService,
    city_key: str,
    city: dict[str, Any],
    range_preset: str,
    is_custom: bool = False,
) -> dict[str, Any]:
    lat = float(city["lat"])
    lon = float(city["lon"])
    history_raw = await service.get_history_data(
        city_key="" if is_custom else city_key,
        lat=lat,
        lon=lon,
        range_preset=range_preset,
        page_size=200,
    )
    current = normalize_current_payload(await service.get_current_data(lat, lon), lat=lat, lon=lon)
    return {
        "request": request,
        "cities": get_cities_mapping(),
        "current_city": city,
        "city_key": city_key,
        "selected_range": range_preset,
        "history_records": normalize_history_payload((history_raw or {}).get("items", [])),
        "explainability": build_explainability(current),
        "title": f"История - {city['name']}",
        "api_status": "healthy",
        "is_custom": is_custom,
    }


async def build_trends_page_context(
    *,
    request: Any,
    service: WebAppService,
    city_key: str,
    city: dict[str, Any],
    range_preset: str,
    is_custom: bool = False,
) -> dict[str, Any]:
    lat = float(city["lat"])
    lon = float(city["lon"])
    trends = await service.get_trends_data(
        city_key="" if is_custom else city_key,
        lat=lat if is_custom else None,
        lon=lon if is_custom else None,
        range_preset=range_preset,
    )
    current = normalize_current_payload(await service.get_current_data(lat, lon), lat=lat, lon=lon)
    return {
        "request": request,
        "cities": get_cities_mapping(),
        "current_city": city,
        "city_key": city_key,
        "selected_range": range_preset,
        "trend_payload": trends,
        "explainability": build_explainability(current),
        "title": f"Тренды - {city['name']}",
        "api_status": "healthy",
        "is_custom": is_custom,
    }


async def build_compare_page_context(
    *,
    request: Any,
    service: WebAppService,
    city_keys: list[str],
) -> dict[str, Any]:
    cities = get_cities_mapping()
    selected = [key for key in city_keys if key in cities][:3]
    cards: list[dict[str, Any]] = []
    for key in selected:
        city = cities[key]
        current_raw, trends_raw = await asyncio.gather(
            service.get_current_data(city["lat"], city["lon"]),
            service.get_trends_data(city_key=key, range_preset="7d"),
        )
        current = normalize_current_payload(current_raw, lat=city["lat"], lon=city["lon"])
        cards.append(
            {
                "city_key": key,
                "city": city,
                "current": current,
                "trend_payload": trends_raw,
                "explainability": build_explainability(current),
            }
        )
    return {
        "request": request,
        "cities": cities,
        "compare_cards": cards,
        "selected_cities": selected,
        "title": "Сравнение городов",
        "api_status": "healthy",
    }


async def build_alerts_page_context(*, request: Any, service: WebAppService) -> dict[str, Any]:
    subscriptions = await service.list_alert_rules()
    return {
        "request": request,
        "cities": get_cities_mapping(),
        "rules": subscriptions,
        "api_status": "healthy",
        "title": "AirTrace RU - Подписки на уведомления",
    }
