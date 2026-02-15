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
from typing import Dict, Any, List
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
