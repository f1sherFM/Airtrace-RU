#!/usr/bin/env python3
"""
AirTrace RU - Веб-приложение на Python
Замена JavaScript интерфейса на серверный рендеринг
"""

import asyncio
import httpx
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from typing import Optional, Dict, Any
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
        self.client = httpx.AsyncClient(timeout=30.0)
    
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
    
    async def check_health(self) -> Dict[str, Any]:
        """Проверка здоровья API"""
        try:
            response = await self.client.get(f"{API_BASE_URL}/health")
            response.raise_for_status()
            return response.json()
        except:
            return {"status": "unhealthy"}
    
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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница"""
    health = await air_service.check_health()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cities": CITIES,
        "api_status": health.get("status", "unknown"),
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
            "aqi_class": get_aqi_class(current_data["aqi"]["value"]),
            "nmu_config": get_nmu_config(current_data["nmu_risk"]),
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
            "aqi_class": get_aqi_class(current_data["aqi"]["value"]),
            "nmu_config": get_nmu_config(current_data["nmu_risk"]),
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

@app.get("/api/health")
async def api_health():
    """Проверка здоровья веб-приложения"""
    backend_health = await air_service.check_health()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "backend_api": backend_health.get("status", "unknown"),
        "cities_available": len(CITIES)
    }

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