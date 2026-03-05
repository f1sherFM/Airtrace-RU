#!/usr/bin/env python3
"""
AirTrace RU - Р’РµР±-РїСЂРёР»РѕР¶РµРЅРёРµ РЅР° Python
Р—Р°РјРµРЅР° JavaScript РёРЅС‚РµСЂС„РµР№СЃР° РЅР° СЃРµСЂРІРµСЂРЅС‹Р№ СЂРµРЅРґРµСЂРёРЅРі
"""

import httpx
import csv
import io
import sys
import logging
from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from typing import Dict, Any, List, Optional
import uvicorn
from datetime import datetime
import json
import os
from pathlib import Path

# Ensure repository root is importable when launched from ./web.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from http_transport import create_async_client

app = FastAPI(title="AirTrace RU Web Interface")
logger = logging.getLogger(__name__)

# РќР°СЃС‚СЂРѕР№РєР° С€Р°Р±Р»РѕРЅРѕРІ
templates = Jinja2Templates(directory="templates")

# РЎС‚Р°С‚РёС‡РµСЃРєРёРµ С„Р°Р№Р»С‹ (CSS, РёР·РѕР±СЂР°Р¶РµРЅРёСЏ)
app.mount("/static", StaticFiles(directory="static"), name="static")

# РљРѕРЅС„РёРіСѓСЂР°С†РёСЏ
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Р“РѕСЂРѕРґР°
CITIES = {
    "moscow": {"name": "РњРѕСЃРєРІР°", "lat": 55.7558, "lon": 37.6176},
    "spb": {"name": "РЎР°РЅРєС‚-РџРµС‚РµСЂР±СѓСЂРі", "lat": 59.9311, "lon": 30.3609},
    "magnitogorsk": {"name": "РњР°РіРЅРёС‚РѕРіРѕСЂСЃРє", "lat": 53.4069, "lon": 58.9794},
    "ekaterinburg": {"name": "Р•РєР°С‚РµСЂРёРЅР±СѓСЂРі", "lat": 56.8431, "lon": 60.6454},
    "novosibirsk": {"name": "РќРѕРІРѕСЃРёР±РёСЂСЃРє", "lat": 55.0084, "lon": 82.9357},
    "chelyabinsk": {"name": "Р§РµР»СЏР±РёРЅСЃРє", "lat": 55.1644, "lon": 61.4368},
    "nizhny": {"name": "РќРёР¶РЅРёР№ РќРѕРІРіРѕСЂРѕРґ", "lat": 56.3269, "lon": 44.0075},
    "samara": {"name": "РЎР°РјР°СЂР°", "lat": 53.2001, "lon": 50.15},
}

class AirQualityService:
    """РЎРµСЂРІРёСЃ РґР»СЏ СЂР°Р±РѕС‚С‹ СЃ API РєР°С‡РµСЃС‚РІР° РІРѕР·РґСѓС…Р°"""
    
    def __init__(self):
        # Unified transport policy with explicit trust_env handling.
        self.client = create_async_client(
            max_connections=10,
            max_keepalive_connections=5,
            read_timeout=30.0,
        )
        self.alerts_api_key = os.getenv("ALERTS_API_KEY", "").strip()

    def _alert_auth_headers(self) -> Dict[str, str]:
        if not self.alerts_api_key:
            return {}
        return {"X-API-Key": self.alerts_api_key}
    
    async def get_current_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """РџРѕР»СѓС‡РµРЅРёРµ С‚РµРєСѓС‰РёС… РґР°РЅРЅС‹С… Рѕ РєР°С‡РµСЃС‚РІРµ РІРѕР·РґСѓС…Р°"""
        try:
            response = await self.client.get(
                f"{API_BASE_URL}/weather/current",
                params={"lat": lat, "lon": lon}
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"API РЅРµРґРѕСЃС‚СѓРїРµРЅ: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"РћС€РёР±РєР° API: {e}")
    
    async def get_forecast_data(self, lat: float, lon: float, hours: int = 24) -> list:
        """РџРѕР»СѓС‡РµРЅРёРµ РїСЂРѕРіРЅРѕР·Р° РєР°С‡РµСЃС‚РІР° РІРѕР·РґСѓС…Р°"""
        try:
            response = await self.client.get(
                f"{API_BASE_URL}/weather/forecast",
                params={"lat": lat, "lon": lon, "hours": hours}
            )
            response.raise_for_status()
            return response.json()
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as e:
            logger.warning("Forecast request failed, returning empty forecast: %s", e)
            return []  # РџСЂРѕРіРЅРѕР· РЅРµ РєСЂРёС‚РёС‡РµРЅ

    async def get_history_data(
        self,
        *,
        city_key: str = "",
        lat: float = 0.0,
        lon: float = 0.0,
        range_preset: str = "24h",
        page_size: int = 48,
    ) -> Dict[str, Any]:
        """РџРѕР»СѓС‡РµРЅРёРµ РёСЃС‚РѕСЂРёС‡РµСЃРєРёС… РґР°РЅРЅС‹С… РєР°С‡РµСЃС‚РІР° РІРѕР·РґСѓС…Р° РёР· backend history API."""
        try:
            params: Dict[str, Any] = {"range": range_preset, "page": 1, "page_size": page_size}
            if city_key:
                params["city"] = city_key
            else:
                params["lat"] = lat
                params["lon"] = lon

            response = await self.client.get(f"{API_BASE_URL}/history", params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as e:
            logger.warning("History request failed, returning empty history: %s", e)
            return {"items": [], "total": 0, "range": range_preset}
    
    async def get_time_series_data(self, lat: float, lon: float, hours: int = 24) -> List[Dict[str, Any]]:
        """РџРѕР»СѓС‡РµРЅРёРµ СЂРµР°Р»СЊРЅС‹С… РїРѕС‡Р°СЃРѕРІС‹С… РґР°РЅРЅС‹С… РЅР° РѕСЃРЅРѕРІРµ РїСЂРѕРіРЅРѕР·Р° Р±РµР· СЃРёРјСѓР»СЏС†РёРё"""
        try:
            forecast_data = await self.get_forecast_data(lat, lon, hours=hours)

            if not forecast_data:
                return []

            # Р’РѕР·РІСЂР°С‰Р°РµРј С‚РѕР»СЊРєРѕ С„Р°РєС‚РёС‡РµСЃРєРёРµ С‚РѕС‡РєРё РїСЂРѕРіРЅРѕР·Р° Р±РµР· РёСЃРєСѓСЃСЃС‚РІРµРЅРЅРѕР№ РіРµРЅРµСЂР°С†РёРё.
            return forecast_data[:hours]
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as e:
            logger.warning("Time-series build failed, returning empty data: %s", e)
            return []
    
    async def check_health(self) -> Dict[str, Any]:
        """РџСЂРѕРІРµСЂРєР° Р·РґРѕСЂРѕРІСЊСЏ API"""
        try:
            response = await self.client.get(f"{API_BASE_URL}/health")
            response.raise_for_status()
            payload = response.json()
            payload["reachable"] = True
            return payload
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as e:
            logger.warning("Primary health probe failed, trying fallback probe: %s", e)
            # Fallback probe: backend may be reachable even if /health is degraded/unavailable.
            try:
                probe = await self.client.get(f"{API_BASE_URL}/version")
                probe.raise_for_status()
                return {"status": "degraded", "reachable": True}
            except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as fallback_error:
                logger.warning("Fallback health probe failed: %s", fallback_error)
                return {"status": "unhealthy", "reachable": False}

    async def list_alert_rules(self) -> List[Dict[str, Any]]:
        try:
            response = await self.client.get(f"{API_BASE_URL}/alerts/rules")
            response.raise_for_status()
            return response.json()
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as e:
            logger.warning("Failed to fetch alert rules, returning empty list: %s", e)
            return []

    async def create_alert_rule(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.client.post(f"{API_BASE_URL}/alerts/rules", json=payload)
        response.raise_for_status()
        return response.json()

    async def update_alert_rule(self, rule_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.client.put(f"{API_BASE_URL}/alerts/rules/{rule_id}", json=payload)
        response.raise_for_status()
        return response.json()

    async def delete_alert_rule(self, rule_id: str) -> Dict[str, Any]:
        response = await self.client.delete(f"{API_BASE_URL}/alerts/rules/{rule_id}")
        response.raise_for_status()
        return response.json()

    async def get_daily_digest(
        self,
        *,
        city: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if city:
            params["city"] = city
        elif lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        response = await self.client.get(f"{API_BASE_URL}/alerts/digest/daily", params=params)
        response.raise_for_status()
        return response.json()

    async def deliver_daily_digest(
        self,
        *,
        chat_id: str,
        city: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"chat_id": chat_id}
        if city:
            params["city"] = city
        elif lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        response = await self.client.get(
            f"{API_BASE_URL}/alerts/digest/daily-and-deliver",
            params=params,
            headers=self._alert_auth_headers(),
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Р—Р°РєСЂС‹С‚РёРµ HTTP РєР»РёРµРЅС‚Р°"""
        await self.client.aclose()

# Р“Р»РѕР±Р°Р»СЊРЅС‹Р№ СЃРµСЂРІРёСЃ
air_service = AirQualityService()

def format_time(timestamp: str) -> str:
    """Р¤РѕСЂРјР°С‚РёСЂРѕРІР°РЅРёРµ РІСЂРµРјРµРЅРё"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%H:%M')
    except (TypeError, ValueError):
        return "--:--"


def normalize_api_status(status: Optional[str]) -> str:
    normalized = (status or "").strip().lower()
    if normalized in {"healthy", "ok", "up", "enabled", "active"}:
        return "healthy"
    if normalized in {"degraded", "warning", "unknown", "disabled"}:
        return "degraded"
    if normalized in {"unhealthy", "down", "failed", "error"}:
        return "unhealthy"
    return "degraded"

def get_aqi_class(aqi: int) -> str:
    """РџРѕР»СѓС‡РµРЅРёРµ CSS РєР»Р°СЃСЃР° РґР»СЏ AQI"""
    if aqi <= 50:
        return "aqi-good"
    elif aqi <= 100:
        return "aqi-moderate"
    elif aqi <= 150:
        return "aqi-unhealthy-sensitive"
    elif aqi <= 200:
        return "aqi-unhealthy"
    elif aqi <= 300:
        return "aqi-very-unhealthy"
    else:
        return "aqi-hazardous"

def get_nmu_config(risk: str) -> Dict[str, str]:
    """РљРѕРЅС„РёРіСѓСЂР°С†РёСЏ РґР»СЏ РѕС‚РѕР±СЂР°Р¶РµРЅРёСЏ РќРњРЈ СЂРёСЃРєР°"""
    configs = {
        "low": {
            "border": "border-l-green-400",
            "icon": "shield-check",
            "color": "text-green-400",
            "level": "РќРёР·РєРёР№ СЂРёСЃРє",
            "description": "Р РµР¶РёРј В«Р§РµСЂРЅРѕРіРѕ РЅРµР±Р°В» РЅРµ РѕР¶РёРґР°РµС‚СЃСЏ"
        },
        "medium": {
            "border": "border-l-yellow-400",
            "icon": "shield",
            "color": "text-yellow-400",
            "level": "РЈРјРµСЂРµРЅРЅС‹Р№ СЂРёСЃРє",
            "description": "РЎР»РµРґРёС‚Рµ Р·Р° РёР·РјРµРЅРµРЅРёСЏРјРё РєР°С‡РµСЃС‚РІР° РІРѕР·РґСѓС…Р°"
        },
        "high": {
            "border": "border-l-orange-400",
            "icon": "shield-alert",
            "color": "text-orange-400",
            "level": "Р’С‹СЃРѕРєРёР№ СЂРёСЃРє",
            "description": "Р’РѕР·РјРѕР¶РЅС‹ РЅРµР±Р»Р°РіРѕРїСЂРёСЏС‚РЅС‹Рµ СѓСЃР»РѕРІРёСЏ"
        },
        "critical": {
            "border": "border-l-red-500",
            "icon": "shield-x",
            "color": "text-red-500",
            "level": "РљР РРўРР§Р•РЎРљРР™",
            "description": "Р РµР¶РёРј В«Р§РµСЂРЅРѕРіРѕ РЅРµР±Р°В» Р°РєС‚РёРІРµРЅ!"
        }
    }
    return configs.get(risk, configs["low"])


def get_action_plan(aqi_value: int, nmu_risk: str) -> Dict[str, Any]:
    """Action-oriented recommendations by risk level and sensitivity profile."""
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
            "title": "Р§С‚Рѕ РґРµР»Р°С‚СЊ СЃРµР№С‡Р°СЃ: РќРёР·РєРёР№ СЂРёСЃРє",
            "color": "green",
            "risk_label": "low",
            "general": [
                "РћР±С‹С‡РЅР°СЏ Р°РєС‚РёРІРЅРѕСЃС‚СЊ РЅР° СѓР»РёС†Рµ РґРѕРїСѓСЃС‚РёРјР°.",
                "РџСЂРѕРІРµС‚СЂРёРІР°РЅРёРµ РїРѕРјРµС‰РµРЅРёР№ РјРѕР¶РЅРѕ РґРµР»Р°С‚СЊ РІ СЃС‚Р°РЅРґР°СЂС‚РЅРѕРј СЂРµР¶РёРјРµ.",
            ],
            "sensitive": [
                "РџСЂРё СЃРёРјРїС‚РѕРјР°С… (РєР°С€РµР»СЊ, СЂР°Р·РґСЂР°Р¶РµРЅРёРµ) СЃРѕРєСЂР°С‚РёС‚Рµ РїСЂРѕРіСѓР»РєРё.",
                "Р”РµСЂР¶РёС‚Рµ Р±Р°Р·РѕРІС‹Рµ Р»РµРєР°СЂСЃС‚РІР° РїСЂРё СЃРµР±Рµ, РµСЃР»Рё РµСЃС‚СЊ С…СЂРѕРЅРёС‡РµСЃРєРёРµ Р·Р°Р±РѕР»РµРІР°РЅРёСЏ.",
            ],
        },
        "medium": {
            "title": "Р§С‚Рѕ РґРµР»Р°С‚СЊ СЃРµР№С‡Р°СЃ: РЈРјРµСЂРµРЅРЅС‹Р№ СЂРёСЃРє",
            "color": "yellow",
            "risk_label": "medium",
            "general": [
                "РЈРјРµРЅСЊС€РёС‚Рµ РёРЅС‚РµРЅСЃРёРІРЅС‹Рµ С‚СЂРµРЅРёСЂРѕРІРєРё РЅР° РѕС‚РєСЂС‹С‚РѕРј РІРѕР·РґСѓС…Рµ.",
                "РџР»Р°РЅРёСЂСѓР№С‚Рµ РїСЂРѕРіСѓР»РєРё РІ С‡Р°СЃС‹ СЃ Р±РѕР»РµРµ С‡РёСЃС‚С‹Рј РІРѕР·РґСѓС…РѕРј.",
            ],
            "sensitive": [
                "РЎРѕРєСЂР°С‚РёС‚Рµ РґР»РёС‚РµР»СЊРЅРѕРµ РїСЂРµР±С‹РІР°РЅРёРµ РЅР° СѓР»РёС†Рµ.",
                "РСЃРїРѕР»СЊР·СѓР№С‚Рµ РјР°СЃРєСѓ/СЂРµСЃРїРёСЂР°С‚РѕСЂ РїСЂРё РґР»РёС‚РµР»СЊРЅС‹С… РІС‹С…РѕРґР°С….",
            ],
        },
        "high": {
            "title": "Р§С‚Рѕ РґРµР»Р°С‚СЊ СЃРµР№С‡Р°СЃ: Р’С‹СЃРѕРєРёР№ СЂРёСЃРє",
            "color": "orange",
            "risk_label": "high",
            "general": [
                "РР·Р±РµРіР°Р№С‚Рµ РґР»РёС‚РµР»СЊРЅС‹С… РЅР°РіСЂСѓР·РѕРє РЅР° СѓР»РёС†Рµ.",
                "Р—Р°РєСЂС‹РІР°Р№С‚Рµ РѕРєРЅР° РЅР° РїРµСЂРёРѕРґ РїРёРєРѕРІРѕРіРѕ Р·Р°РіСЂСЏР·РЅРµРЅРёСЏ.",
            ],
            "sensitive": [
                "РџРѕ РІРѕР·РјРѕР¶РЅРѕСЃС‚Рё РѕСЃС‚Р°РІР°Р№С‚РµСЃСЊ РІ РїРѕРјРµС‰РµРЅРёРё.",
                "РСЃРїРѕР»СЊР·СѓР№С‚Рµ РѕС‡РёСЃС‚РёС‚РµР»СЊ РІРѕР·РґСѓС…Р° Рё РєРѕРЅС‚СЂРѕР»РёСЂСѓР№С‚Рµ СЃРёРјРїС‚РѕРјС‹.",
            ],
        },
        "critical": {
            "title": "Р§С‚Рѕ РґРµР»Р°С‚СЊ СЃРµР№С‡Р°СЃ: РљСЂРёС‚РёС‡РµСЃРєРёР№ СЂРёСЃРє",
            "color": "red",
            "risk_label": "critical",
            "general": [
                "РћС‚Р»РѕР¶РёС‚Рµ РїСЂРѕРіСѓР»РєРё Рё С„РёР·РёС‡РµСЃРєСѓСЋ Р°РєС‚РёРІРЅРѕСЃС‚СЊ РЅР° СѓР»РёС†Рµ.",
                "РњР°РєСЃРёРјР°Р»СЊРЅРѕ РѕРіСЂР°РЅРёС‡СЊС‚Рµ РїСЂРёС‚РѕРє РЅР°СЂСѓР¶РЅРѕРіРѕ РІРѕР·РґСѓС…Р°.",
            ],
            "sensitive": [
                "РћСЃС‚Р°РІР°Р№С‚РµСЃСЊ РІ РїРѕРјРµС‰РµРЅРёРё, РІС‹С…РѕРґ С‚РѕР»СЊРєРѕ РїСЂРё РЅРµРѕР±С…РѕРґРёРјРѕСЃС‚Рё.",
                "РџСЂРё СѓС…СѓРґС€РµРЅРёРё СЃРѕСЃС‚РѕСЏРЅРёСЏ РѕР±СЂР°С‰Р°Р№С‚РµСЃСЊ Р·Р° РјРµРґРёС†РёРЅСЃРєРѕР№ РїРѕРјРѕС‰СЊСЋ.",
            ],
        },
    }
    plan = plans[risk]
    plan["immediate"] = [plan["general"][0], plan["sensitive"][0]]
    return plan

def prepare_export_data(time_series_data: List[Dict[str, Any]], city_name: str) -> List[Dict[str, Any]]:
    """РџРѕРґРіРѕС‚РѕРІРєР° РґР°РЅРЅС‹С… РґР»СЏ СЌРєСЃРїРѕСЂС‚Р°"""
    export_data = []
    
    for point in time_series_data:
        export_point = {
            "timestamp": point["timestamp"],
            "city": city_name,
            "latitude": point["location"]["latitude"],
            "longitude": point["location"]["longitude"],
            "aqi_value": point["aqi"]["value"],
            "aqi_category": point["aqi"]["category"],
            "pm2_5": point["pollutants"]["pm2_5"],
            "pm10": point["pollutants"]["pm10"],
            "no2": point["pollutants"]["no2"],
            "so2": point["pollutants"]["so2"],
            "o3": point["pollutants"]["o3"],
            "nmu_risk": point["nmu_risk"]
        }
        export_data.append(export_point)
    
    return export_data

def create_csv_export(data: List[Dict[str, Any]]) -> str:
    """РЎРѕР·РґР°РЅРёРµ CSV С„Р°Р№Р»Р° РёР· РґР°РЅРЅС‹С…"""
    if not data:
        return ""
    
    output = io.StringIO()
    fieldnames = data[0].keys()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(data)
    
    return output.getvalue()

def create_json_export(data: List[Dict[str, Any]]) -> str:
    """РЎРѕР·РґР°РЅРёРµ JSON С„Р°Р№Р»Р° РёР· РґР°РЅРЅС‹С…"""
    return json.dumps(data, ensure_ascii=False, indent=2)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Р“Р»Р°РІРЅР°СЏ СЃС‚СЂР°РЅРёС†Р°"""
    health = await air_service.check_health()
    
    api_reachable = bool(health.get("reachable", False))
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cities": CITIES,
        "api_status": normalize_api_status(health.get("status")),
        "api_reachable": api_reachable,
        "title": "AirTrace RU вЂ” РњРѕРЅРёС‚РѕСЂРёРЅРі РєР°С‡РµСЃС‚РІР° РІРѕР·РґСѓС…Р°"
    })

@app.get("/city/{city_key}", response_class=HTMLResponse)
async def city_data(request: Request, city_key: str):
    """РЎС‚СЂР°РЅРёС†Р° СЃ РґР°РЅРЅС‹РјРё РґР»СЏ РєРѕРЅРєСЂРµС‚РЅРѕРіРѕ РіРѕСЂРѕРґР°"""
    
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="Р“РѕСЂРѕРґ РЅРµ РЅР°Р№РґРµРЅ")
    
    city = CITIES[city_key]
    
    try:
        # РџРѕР»СѓС‡Р°РµРј С‚РµРєСѓС‰РёРµ РґР°РЅРЅС‹Рµ
        current_data = await air_service.get_current_data(city["lat"], city["lon"])
        
        # РџРѕР»СѓС‡Р°РµРј РїСЂРѕРіРЅРѕР·
        forecast_data = await air_service.get_forecast_data(city["lat"], city["lon"])
        history_data = await air_service.get_history_data(
            city_key=city_key,
            range_preset="24h",
            page_size=48
        )
        
        # РћР±СЂР°Р±Р°С‚С‹РІР°РµРј РїСЂРѕРіРЅРѕР· (РїРµСЂРІС‹Рµ 8 С‡Р°СЃРѕРІ)
        forecast_hours = forecast_data[:8] if forecast_data else []
        
        # Р¤РѕСЂРјР°С‚РёСЂСѓРµРј РґР°РЅРЅС‹Рµ РґР»СЏ С€Р°Р±Р»РѕРЅР°
        context = {
            "request": request,
            "cities": CITIES,
            "current_city": city,
            "city_key": city_key,
            "data": current_data,
            "forecast": forecast_hours,
            "history": history_data.get("items", [])[:12],
            "aqi_class": get_aqi_class(current_data["aqi"]["value"]),
            "nmu_config": get_nmu_config(current_data["nmu_risk"]),
            "action_plan": get_action_plan(current_data["aqi"]["value"], current_data.get("nmu_risk", "low")),
            "format_time": format_time,
            "api_status": "healthy",
            "title": f"AirTrace RU вЂ” {city['name']}"
        }
        
        return templates.TemplateResponse("city.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        # Р’ СЃР»СѓС‡Р°Рµ РѕС€РёР±РєРё РїРѕРєР°Р·С‹РІР°РµРј СЃС‚СЂР°РЅРёС†Сѓ СЃ РѕС€РёР±РєРѕР№
        return templates.TemplateResponse("error.html", {
            "request": request,
            "cities": CITIES,
            "error_message": str(e),
            "city": city,
            "title": f"РћС€РёР±РєР° вЂ” {city['name']}"
        })

@app.get("/custom", response_class=HTMLResponse)
async def custom_city_form(request: Request):
    """Р¤РѕСЂРјР° РґР»СЏ РІРІРѕРґР° РїСЂРѕРёР·РІРѕР»СЊРЅРѕРіРѕ РіРѕСЂРѕРґР°"""
    return templates.TemplateResponse("custom_city.html", {
        "request": request,
        "cities": CITIES,
        "api_status": "healthy",
        "title": "AirTrace RU вЂ” РџСЂРѕРёР·РІРѕР»СЊРЅС‹Р№ РіРѕСЂРѕРґ"
    })

@app.post("/custom", response_class=HTMLResponse)
async def custom_city_data(request: Request, lat: float = Form(...), lon: float = Form(...), city_name: str = Form("")):
    """РћР±СЂР°Р±РѕС‚РєР° РґР°РЅРЅС‹С… РґР»СЏ РїСЂРѕРёР·РІРѕР»СЊРЅРѕРіРѕ РіРѕСЂРѕРґР°"""
    
    # Р’Р°Р»РёРґР°С†РёСЏ РєРѕРѕСЂРґРёРЅР°С‚
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=400, detail="РЁРёСЂРѕС‚Р° РґРѕР»Р¶РЅР° Р±С‹С‚СЊ РѕС‚ -90 РґРѕ 90")
    if not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Р”РѕР»РіРѕС‚Р° РґРѕР»Р¶РЅР° Р±С‹С‚СЊ РѕС‚ -180 РґРѕ 180")
    
    # РЎРѕР·РґР°РµРј РІСЂРµРјРµРЅРЅС‹Р№ РѕР±СЉРµРєС‚ РіРѕСЂРѕРґР°
    custom_city = {
        "name": city_name if city_name else f"РљРѕРѕСЂРґРёРЅР°С‚С‹ {lat:.2f}, {lon:.2f}",
        "lat": lat,
        "lon": lon
    }
    
    try:
        # РџРѕР»СѓС‡Р°РµРј С‚РµРєСѓС‰РёРµ РґР°РЅРЅС‹Рµ
        current_data = await air_service.get_current_data(lat, lon)
        
        # РџРѕР»СѓС‡Р°РµРј РїСЂРѕРіРЅРѕР·
        forecast_data = await air_service.get_forecast_data(lat, lon)
        history_data = await air_service.get_history_data(
            lat=lat,
            lon=lon,
            range_preset="24h",
            page_size=48
        )
        
        # РћР±СЂР°Р±Р°С‚С‹РІР°РµРј РїСЂРѕРіРЅРѕР· (РїРµСЂРІС‹Рµ 8 С‡Р°СЃРѕРІ)
        forecast_hours = forecast_data[:8] if forecast_data else []
        
        # Р¤РѕСЂРјР°С‚РёСЂСѓРµРј РґР°РЅРЅС‹Рµ РґР»СЏ С€Р°Р±Р»РѕРЅР°
        context = {
            "request": request,
            "cities": CITIES,
            "current_city": custom_city,
            "city_key": "custom",
            "data": current_data,
            "forecast": forecast_hours,
            "history": history_data.get("items", [])[:12],
            "aqi_class": get_aqi_class(current_data["aqi"]["value"]),
            "nmu_config": get_nmu_config(current_data["nmu_risk"]),
            "action_plan": get_action_plan(current_data["aqi"]["value"], current_data.get("nmu_risk", "low")),
            "format_time": format_time,
            "api_status": "healthy",
            "title": f"AirTrace RU вЂ” {custom_city['name']}",
            "is_custom": True
        }
        
        return templates.TemplateResponse("city.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        # Р’ СЃР»СѓС‡Р°Рµ РѕС€РёР±РєРё РїРѕРєР°Р·С‹РІР°РµРј СЃС‚СЂР°РЅРёС†Сѓ СЃ РѕС€РёР±РєРѕР№
        return templates.TemplateResponse("error.html", {
            "request": request,
            "cities": CITIES,
            "error_message": str(e),
            "city": custom_city,
            "title": f"РћС€РёР±РєР° вЂ” {custom_city['name']}"
        })

@app.post("/refresh/{city_key}")
async def refresh_city_data(city_key: str):
    """API РґР»СЏ РѕР±РЅРѕРІР»РµРЅРёСЏ РґР°РЅРЅС‹С… РіРѕСЂРѕРґР°"""
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="Р“РѕСЂРѕРґ РЅРµ РЅР°Р№РґРµРЅ")
    
    # РџРµСЂРµРЅР°РїСЂР°РІР»СЏРµРј РЅР° СЃС‚СЂР°РЅРёС†Сѓ РіРѕСЂРѕРґР° РґР»СЏ РѕР±РЅРѕРІР»РµРЅРёСЏ
    return RedirectResponse(url=f"/city/{city_key}", status_code=303)


@app.get("/alerts/settings", response_class=HTMLResponse)
async def alert_settings_page(request: Request):
    """Alert settings UI with full CRUD for rules."""
    rules = await air_service.list_alert_rules()
    return templates.TemplateResponse(
        "alerts.html",
        {
            "request": request,
            "cities": CITIES,
            "rules": rules,
            "api_status": "healthy",
            "title": "AirTrace RU вЂ” РќР°СЃС‚СЂРѕР№РєРё Р°Р»РµСЂС‚РѕРІ",
        },
    )


@app.get("/api/alerts/digest-preview")
async def alert_digest_preview_api(
    city_key: Optional[str] = Query(None),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lon: Optional[float] = Query(None, ge=-180, le=180),
):
    city = city_key if city_key else None
    return await air_service.get_daily_digest(city=city, lat=lat, lon=lon)


@app.post("/api/alerts/digest-deliver")
async def alert_digest_deliver_api(
    chat_id: str = Form(...),
    city_key: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    city = city_key if city_key else None
    return await air_service.deliver_daily_digest(chat_id=chat_id, city=city, lat=lat, lon=lon)


@app.post("/alerts/settings/create")
async def alert_settings_create(
    name: str = Form(...),
    enabled: Optional[str] = Form(None),
    aqi_threshold: Optional[int] = Form(None),
    nmu_levels: Optional[str] = Form(None),
    cooldown_minutes: int = Form(60),
    quiet_hours_start: Optional[int] = Form(None),
    quiet_hours_end: Optional[int] = Form(None),
    channel: str = Form("telegram"),
    chat_id: Optional[str] = Form(None),
):
    payload = {
        "name": name,
        "enabled": enabled == "on",
        "aqi_threshold": aqi_threshold,
        "nmu_levels": [x.strip() for x in (nmu_levels or "").split(",") if x.strip()],
        "cooldown_minutes": cooldown_minutes,
        "quiet_hours_start": quiet_hours_start,
        "quiet_hours_end": quiet_hours_end,
        "channel": channel,
        "chat_id": chat_id or None,
    }
    await air_service.create_alert_rule(payload)
    return RedirectResponse(url="/alerts/settings", status_code=303)


@app.post("/alerts/settings/update/{rule_id}")
async def alert_settings_update(
    rule_id: str,
    name: str = Form(...),
    enabled: Optional[str] = Form(None),
    aqi_threshold: Optional[int] = Form(None),
    nmu_levels: Optional[str] = Form(None),
    cooldown_minutes: int = Form(60),
    quiet_hours_start: Optional[int] = Form(None),
    quiet_hours_end: Optional[int] = Form(None),
    channel: str = Form("telegram"),
    chat_id: Optional[str] = Form(None),
):
    payload = {
        "name": name,
        "enabled": enabled == "on",
        "aqi_threshold": aqi_threshold,
        "nmu_levels": [x.strip() for x in (nmu_levels or "").split(",") if x.strip()],
        "cooldown_minutes": cooldown_minutes,
        "quiet_hours_start": quiet_hours_start,
        "quiet_hours_end": quiet_hours_end,
        "channel": channel,
        "chat_id": chat_id or None,
    }
    await air_service.update_alert_rule(rule_id, payload)
    return RedirectResponse(url="/alerts/settings", status_code=303)


@app.post("/alerts/settings/delete/{rule_id}")
async def alert_settings_delete(rule_id: str):
    await air_service.delete_alert_rule(rule_id)
    return RedirectResponse(url="/alerts/settings", status_code=303)

@app.get("/api/health")
async def api_health():
    """РџСЂРѕРІРµСЂРєР° Р·РґРѕСЂРѕРІСЊСЏ РІРµР±-РїСЂРёР»РѕР¶РµРЅРёСЏ"""
    backend_health = await air_service.check_health()
    backend_status = normalize_api_status(backend_health.get("status"))
    reachable = bool(backend_health.get("reachable", False))
    overall_status = backend_status if reachable else "unhealthy"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "backend_api": backend_status,
        "backend_reachable": reachable,
        "cities_available": len(CITIES)
    }

@app.get("/api/historical/{city_key}")
@app.get("/api/timeseries/{city_key}")
async def get_timeseries_data_api(
    city_key: str,
    hours: int = Query(24, ge=1, le=168)  # Р”Рѕ РЅРµРґРµР»Рё РїРѕС‡Р°СЃРѕРІРѕРіРѕ РїСЂРѕРіРЅРѕР·Р°
):
    """Р­РЅРґРїРѕРёРЅС‚ С‚Р°Р№РјСЃРµСЂРёР№: РІРѕР·РІСЂР°С‰Р°РµС‚ СЂРµР°Р»СЊРЅС‹Р№ РїРѕС‡Р°СЃРѕРІРѕР№ РїСЂРѕРіРЅРѕР· (Р±РµР· СЃРёРјСѓР»СЏС†РёРё)"""
    
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="Р“РѕСЂРѕРґ РЅРµ РЅР°Р№РґРµРЅ")
    
    city = CITIES[city_key]
    try:
        # РџРѕР»СѓС‡Р°РµРј С„Р°РєС‚РёС‡РµСЃРєРёРµ РґР°РЅРЅС‹Рµ РїСЂРѕРіРЅРѕР·Р°
        time_series_data = await air_service.get_time_series_data(
            city["lat"], city["lon"], hours
        )
        
        if not time_series_data:
            raise HTTPException(status_code=503, detail="РќРµ СѓРґР°Р»РѕСЃСЊ РїРѕР»СѓС‡РёС‚СЊ РґР°РЅРЅС‹Рµ РїСЂРѕРіРЅРѕР·Р°")
        
        return {
            "city": city["name"],
            "source": "forecast",
            "period_hours_requested": hours,
            "period_hours_available": len(time_series_data),
            "data_points": len(time_series_data),
            "data": time_series_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"РћС€РёР±РєР° РїСЂРё РїРѕР»СѓС‡РµРЅРёРё РґР°РЅРЅС‹С…: {str(e)}")


@app.get("/api/history/{city_key}")
async def get_history_data_api(
    city_key: str,
    period: str = Query("24h", pattern="^(24h|7d|30d)$")
):
    """History API РґР»СЏ UI, СЃС‚СЂРѕРіРѕ РѕС‚РґРµР»РµРЅРЅС‹Р№ РѕС‚ forecast."""
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="Р“РѕСЂРѕРґ РЅРµ РЅР°Р№РґРµРЅ")

    city = CITIES[city_key]
    try:
        data = await air_service.get_history_data(
            city_key=city_key,
            lat=city["lat"],
            lon=city["lon"],
            range_preset=period,
            page_size=200,
        )
        return {
            "city": city["name"],
            "source": "history",
            "range": period,
            "data_points": len(data.get("items", [])),
            "data": data.get("items", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"РћС€РёР±РєР° РїСЂРё РїРѕР»СѓС‡РµРЅРёРё РёСЃС‚РѕСЂРёРё: {str(e)}")

@app.get("/api/historical-custom")
@app.get("/api/timeseries-custom")
async def get_timeseries_custom_data_api(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city_name: str = Query("Custom Location"),
    hours: int = Query(24, ge=1, le=168)
):
    """Р­РЅРґРїРѕРёРЅС‚ С‚Р°Р№РјСЃРµСЂРёР№ РґР»СЏ РїСЂРѕРёР·РІРѕР»СЊРЅС‹С… РєРѕРѕСЂРґРёРЅР°С‚ (Р±РµР· СЃРёРјСѓР»СЏС†РёРё)"""
    
    try:
        # РџРѕР»СѓС‡Р°РµРј С„Р°РєС‚РёС‡РµСЃРєРёРµ РґР°РЅРЅС‹Рµ РїСЂРѕРіРЅРѕР·Р°
        time_series_data = await air_service.get_time_series_data(lat, lon, hours)
        
        if not time_series_data:
            raise HTTPException(status_code=503, detail="РќРµ СѓРґР°Р»РѕСЃСЊ РїРѕР»СѓС‡РёС‚СЊ РґР°РЅРЅС‹Рµ РїСЂРѕРіРЅРѕР·Р°")
        
        return {
            "city": city_name,
            "coordinates": {"lat": lat, "lon": lon},
            "source": "forecast",
            "period_hours_requested": hours,
            "period_hours_available": len(time_series_data),
            "data_points": len(time_series_data),
            "data": time_series_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"РћС€РёР±РєР° РїСЂРё РїРѕР»СѓС‡РµРЅРёРё РґР°РЅРЅС‹С…: {str(e)}")


@app.get("/api/history-custom")
async def get_history_custom_data_api(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city_name: str = Query("Custom Location"),
    period: str = Query("24h", pattern="^(24h|7d|30d)$")
):
    """History API РґР»СЏ РїСЂРѕРёР·РІРѕР»СЊРЅС‹С… РєРѕРѕСЂРґРёРЅР°С‚."""
    try:
        data = await air_service.get_history_data(
            lat=lat,
            lon=lon,
            range_preset=period,
            page_size=200,
        )
        return {
            "city": city_name,
            "coordinates": {"lat": lat, "lon": lon},
            "source": "history",
            "range": period,
            "data_points": len(data.get("items", [])),
            "data": data.get("items", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"РћС€РёР±РєР° РїСЂРё РїРѕР»СѓС‡РµРЅРёРё РёСЃС‚РѕСЂРёРё: {str(e)}")

@app.get("/export/{city_key}")
async def export_city_data(
    city_key: str,
    format: str = Query(..., pattern="^(csv|json)$"),
    hours: int = Query(24, ge=1, le=168)  # РћС‚ 1 РґРѕ 168 С‡Р°СЃРѕРІ (РЅРµРґРµР»СЏ)
):
    """Р­РєСЃРїРѕСЂС‚ РґР°РЅРЅС‹С… РіРѕСЂРѕРґР° РІ CSV РёР»Рё JSON С„РѕСЂРјР°С‚Рµ"""
    
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="Р“РѕСЂРѕРґ РЅРµ РЅР°Р№РґРµРЅ")
    
    city = CITIES[city_key]
    
    try:
        # РџРѕР»СѓС‡Р°РµРј СЂРµР°Р»СЊРЅС‹Рµ РїРѕС‡Р°СЃРѕРІС‹Рµ РґР°РЅРЅС‹Рµ РїСЂРѕРіРЅРѕР·Р°
        time_series_data = await air_service.get_time_series_data(
            city["lat"], city["lon"], hours
        )
        
        if not time_series_data:
            raise HTTPException(status_code=503, detail="РќРµ СѓРґР°Р»РѕСЃСЊ РїРѕР»СѓС‡РёС‚СЊ РґР°РЅРЅС‹Рµ РґР»СЏ СЌРєСЃРїРѕСЂС‚Р°")
        
        # РџРѕРґРіРѕС‚Р°РІР»РёРІР°РµРј РґР°РЅРЅС‹Рµ РґР»СЏ СЌРєСЃРїРѕСЂС‚Р°
        export_data = prepare_export_data(time_series_data, city["name"])
        
        # РЎРѕР·РґР°РµРј С„Р°Р№Р» РІ РЅСѓР¶РЅРѕРј С„РѕСЂРјР°С‚Рµ
        if format == "csv":
            content = create_csv_export(export_data)
            media_type = "text/csv"
            filename = f"airtrace_ru_{city_key}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        else:  # json
            content = create_json_export(export_data)
            media_type = "application/json"
            filename = f"airtrace_ru_{city_key}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        # Р’РѕР·РІСЂР°С‰Р°РµРј С„Р°Р№Р» РґР»СЏ СЃРєР°С‡РёРІР°РЅРёСЏ
        return StreamingResponse(
            io.StringIO(content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"РћС€РёР±РєР° РїСЂРё СЌРєСЃРїРѕСЂС‚Рµ РґР°РЅРЅС‹С…: {str(e)}")

@app.get("/export-custom")
async def export_custom_data(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city_name: str = Query("Custom Location"),
    format: str = Query(..., pattern="^(csv|json)$"),
    hours: int = Query(24, ge=1, le=168)
):
    """Р­РєСЃРїРѕСЂС‚ РґР°РЅРЅС‹С… РґР»СЏ РїСЂРѕРёР·РІРѕР»СЊРЅС‹С… РєРѕРѕСЂРґРёРЅР°С‚"""
    
    try:
        # РџРѕР»СѓС‡Р°РµРј СЂРµР°Р»СЊРЅС‹Рµ РїРѕС‡Р°СЃРѕРІС‹Рµ РґР°РЅРЅС‹Рµ РїСЂРѕРіРЅРѕР·Р°
        time_series_data = await air_service.get_time_series_data(lat, lon, hours)
        
        if not time_series_data:
            raise HTTPException(status_code=503, detail="РќРµ СѓРґР°Р»РѕСЃСЊ РїРѕР»СѓС‡РёС‚СЊ РґР°РЅРЅС‹Рµ РґР»СЏ СЌРєСЃРїРѕСЂС‚Р°")
        
        # РџРѕРґРіРѕС‚Р°РІР»РёРІР°РµРј РґР°РЅРЅС‹Рµ РґР»СЏ СЌРєСЃРїРѕСЂС‚Р°
        export_data = prepare_export_data(time_series_data, city_name)
        
        # РЎРѕР·РґР°РµРј С„Р°Р№Р» РІ РЅСѓР¶РЅРѕРј С„РѕСЂРјР°С‚Рµ
        if format == "csv":
            content = create_csv_export(export_data)
            media_type = "text/csv"
            filename = f"airtrace_ru_custom_{lat}_{lon}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        else:  # json
            content = create_json_export(export_data)
            media_type = "application/json"
            filename = f"airtrace_ru_custom_{lat}_{lon}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        # Р’РѕР·РІСЂР°С‰Р°РµРј С„Р°Р№Р» РґР»СЏ СЃРєР°С‡РёРІР°РЅРёСЏ
        return StreamingResponse(
            io.StringIO(content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"РћС€РёР±РєР° РїСЂРё СЌРєСЃРїРѕСЂС‚Рµ РґР°РЅРЅС‹С…: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Р—Р°РєСЂС‹С‚РёРµ СЂРµСЃСѓСЂСЃРѕРІ РїСЂРё РѕСЃС‚Р°РЅРѕРІРєРµ"""
    await air_service.close()

if __name__ == "__main__":
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
        log_level="info"
    )
