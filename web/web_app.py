#!/usr/bin/env python3
"""
AirTrace RU - Веб-приложение на Python
Замена JavaScript интерфейса на серверный рендеринг
"""

import httpx
import csv
import io
from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from typing import Dict, Any, List, Optional
import uvicorn
from datetime import datetime
import json

app = FastAPI(title="AirTrace RU Web Interface")

# Настройка шаблонов
templates = Jinja2Templates(directory="templates")

# Статические файлы (CSS, изображения)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Конфигурация
API_BASE_URL = "http://127.0.0.1:8000"

# Города
CITIES = {
    "moscow": {"name": "Москва", "lat": 55.7558, "lon": 37.6176},
    "spb": {"name": "Санкт-Петербург", "lat": 59.9311, "lon": 30.3609},
    "magnitogorsk": {"name": "Магнитогорск", "lat": 53.4069, "lon": 58.9794},
    "ekaterinburg": {"name": "Екатеринбург", "lat": 56.8431, "lon": 60.6454},
    "novosibirsk": {"name": "Новосибирск", "lat": 55.0084, "lon": 82.9357},
    "chelyabinsk": {"name": "Челябинск", "lat": 55.1644, "lon": 61.4368},
    "nizhny": {"name": "Нижний Новгород", "lat": 56.3269, "lon": 44.0075},
    "samara": {"name": "Самара", "lat": 53.2001, "lon": 50.15},
}

class AirQualityService:
    """Сервис для работы с API качества воздуха"""
    
    def __init__(self):
        # Avoid proxy env influence for local backend calls (127.0.0.1:8000).
        self.client = httpx.AsyncClient(timeout=30.0, trust_env=False)
    
    async def get_current_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Получение текущих данных о качестве воздуха"""
        try:
            response = await self.client.get(
                f"{API_BASE_URL}/weather/current",
                params={"lat": lat, "lon": lon}
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"API недоступен: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Ошибка API: {e}")
    
    async def get_forecast_data(self, lat: float, lon: float) -> list:
        """Получение прогноза качества воздуха"""
        try:
            response = await self.client.get(
                f"{API_BASE_URL}/weather/forecast",
                params={"lat": lat, "lon": lon}
            )
            response.raise_for_status()
            return response.json()
        except:
            return []  # Прогноз не критичен

    async def get_history_data(
        self,
        *,
        city_key: str = "",
        lat: float = 0.0,
        lon: float = 0.0,
        range_preset: str = "24h",
        page_size: int = 48,
    ) -> Dict[str, Any]:
        """Получение исторических данных качества воздуха из backend history API."""
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
        except Exception:
            return {"items": [], "total": 0, "range": range_preset}
    
    async def get_time_series_data(self, lat: float, lon: float, hours: int = 24) -> List[Dict[str, Any]]:
        """Получение реальных почасовых данных на основе прогноза без симуляции"""
        try:
            forecast_data = await self.get_forecast_data(lat, lon)

            if not forecast_data:
                return []

            # Возвращаем только фактические точки прогноза без искусственной генерации.
            return forecast_data[:hours]
        except Exception:
            return []
    
    async def check_health(self) -> Dict[str, Any]:
        """Проверка здоровья API"""
        try:
            response = await self.client.get(f"{API_BASE_URL}/health")
            response.raise_for_status()
            payload = response.json()
            payload["reachable"] = True
            return payload
        except Exception:
            # Fallback probe: backend may be reachable even if /health is degraded/unavailable.
            try:
                probe = await self.client.get(f"{API_BASE_URL}/version")
                probe.raise_for_status()
                return {"status": "degraded", "reachable": True}
            except Exception:
                return {"status": "unhealthy", "reachable": False}

    async def list_alert_rules(self) -> List[Dict[str, Any]]:
        try:
            response = await self.client.get(f"{API_BASE_URL}/alerts/rules")
            response.raise_for_status()
            return response.json()
        except Exception:
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
        response = await self.client.get(f"{API_BASE_URL}/alerts/digest/daily-and-deliver", params=params)
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Закрытие HTTP клиента"""
        await self.client.aclose()

# Глобальный сервис
air_service = AirQualityService()

def format_time(timestamp: str) -> str:
    """Форматирование времени"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%H:%M')
    except:
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
    """Получение CSS класса для AQI"""
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
    """Конфигурация для отображения НМУ риска"""
    configs = {
        "low": {
            "border": "border-l-green-400",
            "icon": "shield-check",
            "color": "text-green-400",
            "level": "Низкий риск",
            "description": "Режим «Черного неба» не ожидается"
        },
        "medium": {
            "border": "border-l-yellow-400",
            "icon": "shield",
            "color": "text-yellow-400",
            "level": "Умеренный риск",
            "description": "Следите за изменениями качества воздуха"
        },
        "high": {
            "border": "border-l-orange-400",
            "icon": "shield-alert",
            "color": "text-orange-400",
            "level": "Высокий риск",
            "description": "Возможны неблагоприятные условия"
        },
        "critical": {
            "border": "border-l-red-500",
            "icon": "shield-x",
            "color": "text-red-500",
            "level": "КРИТИЧЕСКИЙ",
            "description": "Режим «Черного неба» активен!"
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
            "title": "Что делать сейчас: Низкий риск",
            "color": "green",
            "risk_label": "low",
            "general": [
                "Обычная активность на улице допустима.",
                "Проветривание помещений можно делать в стандартном режиме.",
            ],
            "sensitive": [
                "При симптомах (кашель, раздражение) сократите прогулки.",
                "Держите базовые лекарства при себе, если есть хронические заболевания.",
            ],
        },
        "medium": {
            "title": "Что делать сейчас: Умеренный риск",
            "color": "yellow",
            "risk_label": "medium",
            "general": [
                "Уменьшите интенсивные тренировки на открытом воздухе.",
                "Планируйте прогулки в часы с более чистым воздухом.",
            ],
            "sensitive": [
                "Сократите длительное пребывание на улице.",
                "Используйте маску/респиратор при длительных выходах.",
            ],
        },
        "high": {
            "title": "Что делать сейчас: Высокий риск",
            "color": "orange",
            "risk_label": "high",
            "general": [
                "Избегайте длительных нагрузок на улице.",
                "Закрывайте окна на период пикового загрязнения.",
            ],
            "sensitive": [
                "По возможности оставайтесь в помещении.",
                "Используйте очиститель воздуха и контролируйте симптомы.",
            ],
        },
        "critical": {
            "title": "Что делать сейчас: Критический риск",
            "color": "red",
            "risk_label": "critical",
            "general": [
                "Отложите прогулки и физическую активность на улице.",
                "Максимально ограничьте приток наружного воздуха.",
            ],
            "sensitive": [
                "Оставайтесь в помещении, выход только при необходимости.",
                "При ухудшении состояния обращайтесь за медицинской помощью.",
            ],
        },
    }
    plan = plans[risk]
    plan["immediate"] = [plan["general"][0], plan["sensitive"][0]]
    return plan

def prepare_export_data(time_series_data: List[Dict[str, Any]], city_name: str) -> List[Dict[str, Any]]:
    """Подготовка данных для экспорта"""
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
    """Создание CSV файла из данных"""
    if not data:
        return ""
    
    output = io.StringIO()
    fieldnames = data[0].keys()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(data)
    
    return output.getvalue()

def create_json_export(data: List[Dict[str, Any]]) -> str:
    """Создание JSON файла из данных"""
    return json.dumps(data, ensure_ascii=False, indent=2)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница"""
    health = await air_service.check_health()
    
    api_reachable = bool(health.get("reachable", False))
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cities": CITIES,
        "api_status": normalize_api_status(health.get("status")),
        "api_reachable": api_reachable,
        "title": "AirTrace RU — Мониторинг качества воздуха"
    })

@app.get("/city/{city_key}", response_class=HTMLResponse)
async def city_data(request: Request, city_key: str):
    """Страница с данными для конкретного города"""
    
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="Город не найден")
    
    city = CITIES[city_key]
    
    try:
        # Получаем текущие данные
        current_data = await air_service.get_current_data(city["lat"], city["lon"])
        
        # Получаем прогноз
        forecast_data = await air_service.get_forecast_data(city["lat"], city["lon"])
        history_data = await air_service.get_history_data(
            city_key=city_key,
            range_preset="24h",
            page_size=48
        )
        
        # Обрабатываем прогноз (первые 8 часов)
        forecast_hours = forecast_data[:8] if forecast_data else []
        
        # Форматируем данные для шаблона
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
            "title": f"AirTrace RU — {city['name']}"
        }
        
        return templates.TemplateResponse("city.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        # В случае ошибки показываем страницу с ошибкой
        return templates.TemplateResponse("error.html", {
            "request": request,
            "cities": CITIES,
            "error_message": str(e),
            "city": city,
            "title": f"Ошибка — {city['name']}"
        })

@app.get("/custom", response_class=HTMLResponse)
async def custom_city_form(request: Request):
    """Форма для ввода произвольного города"""
    return templates.TemplateResponse("custom_city.html", {
        "request": request,
        "cities": CITIES,
        "api_status": "healthy",
        "title": "AirTrace RU — Произвольный город"
    })

@app.post("/custom", response_class=HTMLResponse)
async def custom_city_data(request: Request, lat: float = Form(...), lon: float = Form(...), city_name: str = Form("")):
    """Обработка данных для произвольного города"""
    
    # Валидация координат
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=400, detail="Широта должна быть от -90 до 90")
    if not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Долгота должна быть от -180 до 180")
    
    # Создаем временный объект города
    custom_city = {
        "name": city_name if city_name else f"Координаты {lat:.2f}, {lon:.2f}",
        "lat": lat,
        "lon": lon
    }
    
    try:
        # Получаем текущие данные
        current_data = await air_service.get_current_data(lat, lon)
        
        # Получаем прогноз
        forecast_data = await air_service.get_forecast_data(lat, lon)
        history_data = await air_service.get_history_data(
            lat=lat,
            lon=lon,
            range_preset="24h",
            page_size=48
        )
        
        # Обрабатываем прогноз (первые 8 часов)
        forecast_hours = forecast_data[:8] if forecast_data else []
        
        # Форматируем данные для шаблона
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
            "title": f"AirTrace RU — {custom_city['name']}",
            "is_custom": True
        }
        
        return templates.TemplateResponse("city.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        # В случае ошибки показываем страницу с ошибкой
        return templates.TemplateResponse("error.html", {
            "request": request,
            "cities": CITIES,
            "error_message": str(e),
            "city": custom_city,
            "title": f"Ошибка — {custom_city['name']}"
        })

@app.post("/refresh/{city_key}")
async def refresh_city_data(city_key: str):
    """API для обновления данных города"""
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="Город не найден")
    
    # Перенаправляем на страницу города для обновления
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
            "title": "AirTrace RU — Настройки алертов",
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
    """Проверка здоровья веб-приложения"""
    backend_health = await air_service.check_health()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "backend_api": normalize_api_status(backend_health.get("status")),
        "backend_reachable": bool(backend_health.get("reachable", False)),
        "cities_available": len(CITIES)
    }

@app.get("/api/historical/{city_key}")
@app.get("/api/timeseries/{city_key}")
async def get_timeseries_data_api(
    city_key: str,
    hours: int = Query(24, ge=1, le=168)  # До недели почасового прогноза
):
    """Эндпоинт таймсерий: возвращает реальный почасовой прогноз (без симуляции)"""
    
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="Город не найден")
    
    city = CITIES[city_key]
    try:
        # Получаем фактические данные прогноза
        time_series_data = await air_service.get_time_series_data(
            city["lat"], city["lon"], hours
        )
        
        if not time_series_data:
            raise HTTPException(status_code=503, detail="Не удалось получить данные прогноза")
        
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
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {str(e)}")


@app.get("/api/history/{city_key}")
async def get_history_data_api(
    city_key: str,
    period: str = Query("24h", pattern="^(24h|7d|30d)$")
):
    """History API для UI, строго отделенный от forecast."""
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="Город не найден")

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
        raise HTTPException(status_code=500, detail=f"Ошибка при получении истории: {str(e)}")

@app.get("/api/historical-custom")
@app.get("/api/timeseries-custom")
async def get_timeseries_custom_data_api(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city_name: str = Query("Custom Location"),
    hours: int = Query(24, ge=1, le=168)
):
    """Эндпоинт таймсерий для произвольных координат (без симуляции)"""
    
    try:
        # Получаем фактические данные прогноза
        time_series_data = await air_service.get_time_series_data(lat, lon, hours)
        
        if not time_series_data:
            raise HTTPException(status_code=503, detail="Не удалось получить данные прогноза")
        
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
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {str(e)}")


@app.get("/api/history-custom")
async def get_history_custom_data_api(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city_name: str = Query("Custom Location"),
    period: str = Query("24h", pattern="^(24h|7d|30d)$")
):
    """History API для произвольных координат."""
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
        raise HTTPException(status_code=500, detail=f"Ошибка при получении истории: {str(e)}")

@app.get("/export/{city_key}")
async def export_city_data(
    city_key: str,
    format: str = Query(..., pattern="^(csv|json)$"),
    hours: int = Query(24, ge=1, le=168)  # От 1 до 168 часов (неделя)
):
    """Экспорт данных города в CSV или JSON формате"""
    
    if city_key not in CITIES:
        raise HTTPException(status_code=404, detail="Город не найден")
    
    city = CITIES[city_key]
    
    try:
        # Получаем реальные почасовые данные прогноза
        time_series_data = await air_service.get_time_series_data(
            city["lat"], city["lon"], hours
        )
        
        if not time_series_data:
            raise HTTPException(status_code=503, detail="Не удалось получить данные для экспорта")
        
        # Подготавливаем данные для экспорта
        export_data = prepare_export_data(time_series_data, city["name"])
        
        # Создаем файл в нужном формате
        if format == "csv":
            content = create_csv_export(export_data)
            media_type = "text/csv"
            filename = f"airtrace_ru_{city_key}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        else:  # json
            content = create_json_export(export_data)
            media_type = "application/json"
            filename = f"airtrace_ru_{city_key}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        # Возвращаем файл для скачивания
        return StreamingResponse(
            io.StringIO(content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при экспорте данных: {str(e)}")

@app.get("/export-custom")
async def export_custom_data(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city_name: str = Query("Custom Location"),
    format: str = Query(..., pattern="^(csv|json)$"),
    hours: int = Query(24, ge=1, le=168)
):
    """Экспорт данных для произвольных координат"""
    
    try:
        # Получаем реальные почасовые данные прогноза
        time_series_data = await air_service.get_time_series_data(lat, lon, hours)
        
        if not time_series_data:
            raise HTTPException(status_code=503, detail="Не удалось получить данные для экспорта")
        
        # Подготавливаем данные для экспорта
        export_data = prepare_export_data(time_series_data, city_name)
        
        # Создаем файл в нужном формате
        if format == "csv":
            content = create_csv_export(export_data)
            media_type = "text/csv"
            filename = f"airtrace_ru_custom_{lat}_{lon}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        else:  # json
            content = create_json_export(export_data)
            media_type = "application/json"
            filename = f"airtrace_ru_custom_{lat}_{lon}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        # Возвращаем файл для скачивания
        return StreamingResponse(
            io.StringIO(content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при экспорте данных: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Закрытие ресурсов при остановке"""
    await air_service.close()

if __name__ == "__main__":
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
        log_level="info"
    )
