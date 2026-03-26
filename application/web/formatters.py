"""Presentation helpers for the Python SSR layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


def format_time(timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%H:%M")
    except (TypeError, ValueError, AttributeError):
        return "--:--"


def normalize_api_status(status: Optional[str]) -> str:
    normalized = (status or "").strip().lower()
    if normalized in {"healthy", "ok", "up", "enabled", "active"}:
        return "healthy"
    if normalized in {"unhealthy", "down", "failed", "error"}:
        return "unhealthy"
    return "degraded"


def translate_api_status(status: Optional[str]) -> str:
    mapping = {
        "healthy": "работает",
        "degraded": "с ограничениями",
        "unhealthy": "недоступен",
    }
    return mapping.get(normalize_api_status(status), "с ограничениями")


def get_aqi_class(aqi: int) -> str:
    if aqi <= 50:
        return "aqi-good"
    if aqi <= 100:
        return "aqi-moderate"
    if aqi <= 150:
        return "aqi-unhealthy-sensitive"
    if aqi <= 200:
        return "aqi-unhealthy"
    if aqi <= 300:
        return "aqi-very-unhealthy"
    return "aqi-hazardous"


def get_nmu_config(risk: str) -> dict[str, str]:
    configs = {
        "low": {
            "border": "border-l-green-400",
            "icon": "shield-check",
            "color": "text-green-400",
            "level": "Низкий риск",
            "description": "Неблагоприятные условия рассеивания не ожидаются.",
        },
        "medium": {
            "border": "border-l-yellow-400",
            "icon": "shield",
            "color": "text-yellow-400",
            "level": "Умеренный риск",
            "description": "Следите за изменениями качества воздуха.",
        },
        "high": {
            "border": "border-l-orange-400",
            "icon": "shield-alert",
            "color": "text-orange-400",
            "level": "Высокий риск",
            "description": "Возможны неблагоприятные условия рассеивания.",
        },
        "critical": {
            "border": "border-l-red-500",
            "icon": "shield-x",
            "color": "text-red-500",
            "level": "Критический риск",
            "description": "Действуют условия, близкие к режиму НМУ.",
        },
    }
    return dict(configs.get((risk or "low").lower(), configs["low"]))


def get_action_plan(aqi_value: int, nmu_risk: str) -> dict[str, Any]:
    risk = (nmu_risk or "low").lower()
    if aqi_value >= 200 or risk == "critical":
        risk = "critical"
    elif aqi_value >= 150 or risk == "high":
        risk = "high"
    elif aqi_value >= 100 or risk == "medium":
        risk = "medium"
    else:
        risk = "low"

    plans = {
        "low": {
            "title": "Что делать сейчас: низкий риск",
            "color": "green",
            "risk_label": "низкий",
            "general": [
                "Обычная активность на улице допустима.",
                "Проветривание можно оставить в обычном режиме.",
            ],
            "sensitive": [
                "При появлении симптомов сократите время прогулки.",
                "Если есть хронические заболевания, держите базовые лекарства под рукой.",
            ],
        },
        "medium": {
            "title": "Что делать сейчас: умеренный риск",
            "color": "yellow",
            "risk_label": "умеренный",
            "general": [
                "Сократите интенсивные тренировки на улице.",
                "Планируйте прогулки на часы с более чистым воздухом.",
            ],
            "sensitive": [
                "Сократите длительное пребывание на улице.",
                "Для долгого выхода используйте маску или респиратор.",
            ],
        },
        "high": {
            "title": "Что делать сейчас: высокий риск",
            "color": "orange",
            "risk_label": "высокий",
            "general": [
                "Избегайте длительной и интенсивной активности на улице.",
                "Держите окна закрытыми в часы пикового загрязнения.",
            ],
            "sensitive": [
                "По возможности оставайтесь в помещении.",
                "Используйте очиститель воздуха и следите за симптомами.",
            ],
        },
        "critical": {
            "title": "Что делать сейчас: критический риск",
            "color": "red",
            "risk_label": "критический",
            "general": [
                "Отложите прогулки и активность на улице.",
                "Сведите к минимуму попадание наружного воздуха в помещение.",
            ],
            "sensitive": [
                "Оставайтесь дома и выходите только при необходимости.",
                "При ухудшении самочувствия обратитесь за медицинской помощью.",
            ],
        },
    }
    plan = {
        **plans[risk],
        "general": list(plans[risk]["general"]),
        "sensitive": list(plans[risk]["sensitive"]),
    }
    plan["immediate"] = [plan["general"][0], plan["sensitive"][0]]
    return plan


def translate_source(source: Optional[str]) -> str:
    mapping = {
        "live": "онлайн",
        "forecast": "прогноз",
        "historical": "история",
        "database": "база данных",
        "cache": "кэш",
        "unknown": "неизвестно",
    }
    return mapping.get((source or "unknown").strip().lower(), source or "неизвестно")


def translate_freshness(freshness: Optional[str]) -> str:
    mapping = {
        "fresh": "свежие",
        "stale": "устаревшие",
        "unknown": "неизвестно",
    }
    return mapping.get((freshness or "unknown").strip().lower(), freshness or "неизвестно")


def translate_trend(trend: Optional[str]) -> str:
    mapping = {
        "improving": "улучшается",
        "stable": "стабильно",
        "worsening": "ухудшается",
        "insufficient_data": "недостаточно данных",
    }
    return mapping.get((trend or "insufficient_data").strip().lower(), trend or "недостаточно данных")
