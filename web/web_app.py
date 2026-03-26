#!/usr/bin/env python3
"""AirTrace RU Python SSR app powered directly by the application layer."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


class UTF8HTMLResponse(HTMLResponse):
    """HTML response with explicit UTF-8 charset."""

    def __init__(self, content: Any = None, **kwargs: Any):
        super().__init__(content=content, **kwargs)
        self.headers["content-type"] = "text/html; charset=utf-8"


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from application.web import (  # noqa: E402
    WebAppService,
    build_alerts_page_context,
    build_city_page_context,
    build_compare_page_context,
    build_history_page_context,
    build_index_context,
    build_trends_page_context,
)
from cities_data import CITIES as YAML_CITIES  # noqa: E402


CITIES = YAML_CITIES
air_service = WebAppService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        await air_service.close()


app = FastAPI(title="AirTrace RU Web Interface", lifespan=lifespan)
logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="templates")
templates.env.globals["encoding"] = "utf-8"
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def add_charset_to_html(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type and "charset" not in content_type:
        response.headers["content-type"] = "text/html; charset=utf-8"
    return response


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


def get_nmu_config(risk: str) -> Dict[str, str]:
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
    return configs.get((risk or "low").lower(), configs["low"])


def get_action_plan(aqi_value: int, nmu_risk: str) -> Dict[str, Any]:
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
    plan = plans[risk]
    plan["immediate"] = [plan["general"][0], plan["sensitive"][0]]
    return plan


def prepare_export_data(time_series_data: List[Dict[str, Any]], city_name: str) -> List[Dict[str, Any]]:
    export_data = []
    for point in time_series_data:
        export_data.append(
            {
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
                "nmu_risk": point["nmu_risk"],
            }
        )
    return export_data


def create_csv_export(data: List[Dict[str, Any]]) -> str:
    if not data:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()


def create_json_export(data: List[Dict[str, Any]]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _render_city_error(request: Request, city: dict[str, Any], error_message: str):
    return templates.TemplateResponse(
        request,
        "error.html",
        {
            "request": request,
            "cities": CITIES,
            "error_message": error_message,
            "city": city,
            "title": f"Error - {city['name']}",
        },
    )


def _build_alert_payload(
    *,
    name: str,
    enabled: Optional[str],
    aqi_threshold: Optional[int],
    nmu_levels: Optional[str],
    cooldown_minutes: int,
    quiet_hours_start: Optional[int],
    quiet_hours_end: Optional[int],
    channel: str,
    chat_id: Optional[str],
    city: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "enabled": enabled == "on",
        "aqi_threshold": aqi_threshold,
        "nmu_levels": [value.strip() for value in (nmu_levels or "").split(",") if value.strip()],
        "cooldown_minutes": cooldown_minutes,
        "quiet_hours_start": quiet_hours_start,
        "quiet_hours_end": quiet_hours_end,
        "channel": channel,
        "chat_id": chat_id or None,
    }
    if city:
        payload["city"] = city
    elif lat is not None or lon is not None:
        payload["lat"] = lat
        payload["lon"] = lon
    return payload


@app.get("/", response_class=UTF8HTMLResponse)
async def index(request: Request):
    context = await build_index_context(request=request, service=air_service)
    context["api_status"] = normalize_api_status(context.get("api_status"))
    return templates.TemplateResponse(request, "index.html", context)


@app.get("/city/{city_key}", response_class=UTF8HTMLResponse)
async def city_data(request: Request, city_key: str):
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")
    city = CITIES[city_key]
    try:
        context = await build_city_page_context(
            request=request,
            service=air_service,
            city_key=city_key,
            city=city,
        )
        context["aqi_class"] = get_aqi_class(context["data"]["aqi"]["value"])
        context["nmu_config"] = get_nmu_config(context["data"].get("nmu_risk", "low"))
        context["action_plan"] = get_action_plan(context["data"]["aqi"]["value"], context["data"].get("nmu_risk", "low"))
        context["format_time"] = format_time
        return templates.TemplateResponse(request, "city.html", context)
    except HTTPException:
        raise
    except Exception as exc:
        return _render_city_error(request, city, str(exc))


@app.get("/custom", response_class=UTF8HTMLResponse)
async def custom_city_form(request: Request):
    return templates.TemplateResponse(
        request,
        "custom_city.html",
        {
            "request": request,
            "cities": CITIES,
            "api_status": "healthy",
            "title": "AirTrace RU - Custom City",
        },
    )


@app.post("/custom", response_class=UTF8HTMLResponse)
async def custom_city_data(
    request: Request,
    lat: float = Form(...),
    lon: float = Form(...),
    city_name: str = Form(""),
):
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90")
    if not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180")

    custom_city = {
        "name": city_name if city_name else f"Coordinates {lat:.2f}, {lon:.2f}",
        "lat": lat,
        "lon": lon,
    }
    try:
        context = await build_city_page_context(
            request=request,
            service=air_service,
            city_key="custom",
            city=custom_city,
            is_custom=True,
        )
        context["aqi_class"] = get_aqi_class(context["data"]["aqi"]["value"])
        context["nmu_config"] = get_nmu_config(context["data"].get("nmu_risk", "low"))
        context["action_plan"] = get_action_plan(context["data"]["aqi"]["value"], context["data"].get("nmu_risk", "low"))
        context["format_time"] = format_time
        return templates.TemplateResponse(request, "city.html", context)
    except HTTPException:
        raise
    except Exception as exc:
        return _render_city_error(request, custom_city, str(exc))


@app.post("/refresh/{city_key}")
async def refresh_city_data(city_key: str):
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")
    return RedirectResponse(url=f"/city/{city_key}", status_code=303)


@app.get("/history/{city_key}", response_class=UTF8HTMLResponse)
async def history_page(request: Request, city_key: str, range: str = Query("24h", pattern="^(24h|7d|30d)$")):
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")
    context = await build_history_page_context(
        request=request,
        service=air_service,
        city_key=city_key,
        city=CITIES[city_key],
        range_preset=range,
    )
    context["format_time"] = format_time
    return templates.TemplateResponse(request, "history.html", context)


@app.get("/history/custom", response_class=UTF8HTMLResponse)
async def history_custom_page(
    request: Request,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city_name: str = Query("Custom Location"),
    range: str = Query("24h", pattern="^(24h|7d|30d)$"),
):
    city = {"name": city_name, "lat": lat, "lon": lon}
    context = await build_history_page_context(
        request=request,
        service=air_service,
        city_key="custom",
        city=city,
        range_preset=range,
        is_custom=True,
    )
    context["format_time"] = format_time
    return templates.TemplateResponse(request, "history.html", context)


@app.get("/trends/{city_key}", response_class=UTF8HTMLResponse)
async def trends_page(request: Request, city_key: str, range: str = Query("7d", pattern="^(7d|30d)$")):
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")
    context = await build_trends_page_context(
        request=request,
        service=air_service,
        city_key=city_key,
        city=CITIES[city_key],
        range_preset=range,
    )
    return templates.TemplateResponse(request, "trends.html", context)


@app.get("/trends/custom", response_class=UTF8HTMLResponse)
async def trends_custom_page(
    request: Request,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city_name: str = Query("Custom Location"),
    range: str = Query("7d", pattern="^(7d|30d)$"),
):
    city = {"name": city_name, "lat": lat, "lon": lon}
    context = await build_trends_page_context(
        request=request,
        service=air_service,
        city_key="custom",
        city=city,
        range_preset=range,
        is_custom=True,
    )
    return templates.TemplateResponse(request, "trends.html", context)


@app.get("/compare", response_class=UTF8HTMLResponse)
async def compare_cities_page(request: Request, cities: str = Query("moscow,spb")):
    city_keys = [value.strip().lower() for value in cities.split(",") if value.strip()]
    context = await build_compare_page_context(request=request, service=air_service, city_keys=city_keys)
    return templates.TemplateResponse(request, "compare.html", context)


@app.get("/alerts/settings", response_class=UTF8HTMLResponse)
async def alert_settings_page(request: Request):
    context = await build_alerts_page_context(request=request, service=air_service)
    return templates.TemplateResponse(request, "alerts.html", context)


@app.get("/api/alerts/digest-preview")
async def alert_digest_preview_api(
    city_key: Optional[str] = Query(None),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lon: Optional[float] = Query(None, ge=-180, le=180),
):
    return await air_service.get_daily_digest(city=city_key or None, lat=lat, lon=lon)


@app.post("/api/alerts/digest-deliver")
async def alert_digest_deliver_api(
    chat_id: str = Form(...),
    city_key: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    return await air_service.deliver_daily_digest(chat_id=chat_id, city=city_key or None, lat=lat, lon=lon)


@app.post("/alerts/settings/create")
async def alert_settings_create(
    name: str = Form(...),
    enabled: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    aqi_threshold: Optional[int] = Form(None),
    nmu_levels: Optional[str] = Form(None),
    cooldown_minutes: int = Form(60),
    quiet_hours_start: Optional[int] = Form(None),
    quiet_hours_end: Optional[int] = Form(None),
    channel: str = Form("telegram"),
    chat_id: Optional[str] = Form(None),
):
    payload = _build_alert_payload(
        name=name,
        enabled=enabled,
        aqi_threshold=aqi_threshold,
        nmu_levels=nmu_levels,
        cooldown_minutes=cooldown_minutes,
        quiet_hours_start=quiet_hours_start,
        quiet_hours_end=quiet_hours_end,
        channel=channel,
        chat_id=chat_id,
        city=city,
        lat=lat,
        lon=lon,
    )
    await air_service.create_alert_rule(payload)
    return RedirectResponse(url="/alerts/settings", status_code=303)


@app.post("/alerts/settings/update/{rule_id}")
async def alert_settings_update(
    rule_id: str,
    name: str = Form(...),
    enabled: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    aqi_threshold: Optional[int] = Form(None),
    nmu_levels: Optional[str] = Form(None),
    cooldown_minutes: int = Form(60),
    quiet_hours_start: Optional[int] = Form(None),
    quiet_hours_end: Optional[int] = Form(None),
    channel: str = Form("telegram"),
    chat_id: Optional[str] = Form(None),
):
    payload = _build_alert_payload(
        name=name,
        enabled=enabled,
        aqi_threshold=aqi_threshold,
        nmu_levels=nmu_levels,
        cooldown_minutes=cooldown_minutes,
        quiet_hours_start=quiet_hours_start,
        quiet_hours_end=quiet_hours_end,
        channel=channel,
        chat_id=chat_id,
        city=city,
        lat=lat,
        lon=lon,
    )
    await air_service.update_alert_rule(rule_id, payload)
    return RedirectResponse(url="/alerts/settings", status_code=303)


@app.post("/alerts/settings/delete/{rule_id}")
async def alert_settings_delete(rule_id: str):
    await air_service.delete_alert_rule(rule_id)
    return RedirectResponse(url="/alerts/settings", status_code=303)


@app.get("/api/health")
async def api_health():
    backend_health = await air_service.check_health()
    backend_status = normalize_api_status(backend_health.get("status"))
    reachable = bool(backend_health.get("reachable", False))
    return {
        "status": backend_status if reachable else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "backend_api": backend_status,
        "backend_reachable": reachable,
        "cities_available": len(CITIES),
    }


@app.get("/api/historical/{city_key}")
@app.get("/api/timeseries/{city_key}")
async def get_timeseries_data_api(city_key: str, hours: int = Query(24, ge=1, le=168)):
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")
    city = CITIES[city_key]
    time_series_data = await air_service.get_time_series_data(city["lat"], city["lon"], hours)
    if not time_series_data:
        raise HTTPException(status_code=503, detail="No forecast data available")
    return {
        "city": city["name"],
        "source": "forecast",
        "period_hours_requested": hours,
        "period_hours_available": len(time_series_data),
        "data_points": len(time_series_data),
        "data": time_series_data,
    }


@app.get("/api/history/{city_key}")
async def get_history_data_api(city_key: str, period: str = Query("24h", pattern="^(24h|7d|30d)$")):
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")
    city = CITIES[city_key]
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


@app.get("/api/historical-custom")
@app.get("/api/timeseries-custom")
async def get_timeseries_custom_data_api(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city_name: str = Query("Custom Location"),
    hours: int = Query(24, ge=1, le=168),
):
    time_series_data = await air_service.get_time_series_data(lat, lon, hours)
    if not time_series_data:
        raise HTTPException(status_code=503, detail="No forecast data available")
    return {
        "city": city_name,
        "coordinates": {"lat": lat, "lon": lon},
        "source": "forecast",
        "period_hours_requested": hours,
        "period_hours_available": len(time_series_data),
        "data_points": len(time_series_data),
        "data": time_series_data,
    }


@app.get("/api/history-custom")
async def get_history_custom_data_api(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city_name: str = Query("Custom Location"),
    period: str = Query("24h", pattern="^(24h|7d|30d)$"),
):
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


@app.get("/export/{city_key}")
async def export_city_data(
    city_key: str,
    format: str = Query(..., pattern="^(csv|json)$"),
    hours: int = Query(24, ge=1, le=168),
):
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")
    city = CITIES[city_key]
    time_series_data = await air_service.get_time_series_data(city["lat"], city["lon"], hours)
    if not time_series_data:
        raise HTTPException(status_code=503, detail="No export data available")
    export_data = prepare_export_data(time_series_data, city["name"])
    if format == "csv":
        content = create_csv_export(export_data)
        media_type = "text/csv"
        filename = f"airtrace_ru_{city_key}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    else:
        content = create_json_export(export_data)
        media_type = "application/json"
        filename = f"airtrace_ru_{city_key}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    return StreamingResponse(io.StringIO(content), media_type=media_type, headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/export-custom")
async def export_custom_data(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city_name: str = Query("Custom Location"),
    format: str = Query(..., pattern="^(csv|json)$"),
    hours: int = Query(24, ge=1, le=168),
):
    time_series_data = await air_service.get_time_series_data(lat, lon, hours)
    if not time_series_data:
        raise HTTPException(status_code=503, detail="No export data available")
    export_data = prepare_export_data(time_series_data, city_name)
    if format == "csv":
        content = create_csv_export(export_data)
        media_type = "text/csv"
        filename = f"airtrace_ru_custom_{lat}_{lon}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    else:
        content = create_json_export(export_data)
        media_type = "application/json"
        filename = f"airtrace_ru_custom_{lat}_{lon}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    return StreamingResponse(io.StringIO(content), media_type=media_type, headers={"Content-Disposition": f"attachment; filename={filename}"})

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    uvicorn.run("web_app:app", host="0.0.0.0", port=3000, reload=True, log_level="info")
