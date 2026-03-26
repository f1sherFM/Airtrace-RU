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
            "level": "Low Risk",
            "description": "No black-sky conditions expected.",
        },
        "medium": {
            "border": "border-l-yellow-400",
            "icon": "shield",
            "color": "text-yellow-400",
            "level": "Moderate Risk",
            "description": "Watch for changes in air quality.",
        },
        "high": {
            "border": "border-l-orange-400",
            "icon": "shield-alert",
            "color": "text-orange-400",
            "level": "High Risk",
            "description": "Unfavorable dispersion conditions are possible.",
        },
        "critical": {
            "border": "border-l-red-500",
            "icon": "shield-x",
            "color": "text-red-500",
            "level": "Critical",
            "description": "Black-sky conditions are active.",
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
            "title": "What to do now: low risk",
            "color": "green",
            "risk_label": "low",
            "general": [
                "Normal outdoor activity is acceptable.",
                "Ventilation can remain in a standard mode.",
            ],
            "sensitive": [
                "If symptoms appear, reduce walking time outdoors.",
                "Keep baseline medication nearby if you have chronic conditions.",
            ],
        },
        "medium": {
            "title": "What to do now: moderate risk",
            "color": "yellow",
            "risk_label": "medium",
            "general": [
                "Reduce intense outdoor workouts.",
                "Plan walks for hours with cleaner air.",
            ],
            "sensitive": [
                "Reduce long time outdoors.",
                "Use a mask/respirator for longer outdoor exposure.",
            ],
        },
        "high": {
            "title": "What to do now: high risk",
            "color": "orange",
            "risk_label": "high",
            "general": [
                "Avoid long or intense outdoor activity.",
                "Keep windows closed during peak pollution periods.",
            ],
            "sensitive": [
                "Stay indoors when possible.",
                "Use an air purifier and monitor symptoms.",
            ],
        },
        "critical": {
            "title": "What to do now: critical risk",
            "color": "red",
            "risk_label": "critical",
            "general": [
                "Postpone walks and outdoor activity.",
                "Minimize intake of outdoor air indoors.",
            ],
            "sensitive": [
                "Stay indoors and go outside only if necessary.",
                "Seek medical help if your condition worsens.",
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

