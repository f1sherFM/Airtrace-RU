"""
AirTrace RU Backend - Main FastAPI Application

Privacy-first асинхронный REST API сервис для мониторинга качества воздуха
в российских городах с использованием российских стандартов ПДК.

Интегрирует все компоненты системы:
- FastAPI приложение с асинхронными эндпоинтами
- Privacy middleware для защиты приватности пользователей
- Сервисы интеграции с Open-Meteo API
- AQI калькулятор с российскими стандартами ПДК
- Система кэширования данных
- НМУ детектор для определения неблагоприятных условий
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Optional
import asyncio
import logging
import uvicorn
import httpx

from schemas import (
    AirQualityData,
    HealthCheckResponse,
    ErrorResponse,
    CoordinatesRequest
)
from services import AirQualityService
from utils import AQICalculator, check_nmu_risk, validate_coordinates
from middleware import PrivacyMiddleware, setup_privacy_logging, set_privacy_middleware, get_privacy_middleware

# Настройка privacy-aware логирования
setup_privacy_logging()
logger = logging.getLogger(__name__)

# Глобальный экземпляр сервиса для управления жизненным циклом
air_quality_service: Optional[AirQualityService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения.
    
    Инициализирует и очищает ресурсы при запуске и остановке приложения.
    """
    global air_quality_service
    
    # Startup
    logger.info("Starting AirTrace RU Backend...")
    
    # Инициализация сервисов
    air_quality_service = AirQualityService()
    logger.info("Air quality service initialized")
    
    # Запуск фоновых задач
    cleanup_task = asyncio.create_task(periodic_cleanup())
    logger.info("Background cleanup task started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AirTrace RU Backend...")
    
    # Остановка фоновых задач
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    # Очистка ресурсов сервисов
    if air_quality_service:
        await air_quality_service.cleanup()
        logger.info("Air quality service cleaned up")
    
    logger.info("Shutdown complete")


async def periodic_cleanup():
    """Периодическая очистка кэша и других ресурсов"""
    while True:
        try:
            await asyncio.sleep(300)  # Каждые 5 минут
            if air_quality_service:
                await air_quality_service.cache_manager.clear_expired()
                logger.debug("Periodic cache cleanup completed")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}")


# Создание FastAPI приложения с управлением жизненным циклом
app = FastAPI(
    title="AirTrace RU API",
    description="Air Quality Monitoring API for Russian cities with privacy-first approach",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware для поддержки веб-клиентов
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://localhost:8080",  # Vue development server
        "https://airtrace.ru",    # Production domain (example)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Privacy middleware для защиты приватности пользователей
privacy_middleware = PrivacyMiddleware(app, enable_request_logging=True)
app.add_middleware(PrivacyMiddleware, enable_request_logging=True)
set_privacy_middleware(privacy_middleware)

# Инициализация компонентов
aqi_calculator = AQICalculator()


def get_air_quality_service() -> AirQualityService:
    """Dependency для получения экземпляра сервиса качества воздуха"""
    global air_quality_service
    if air_quality_service is None:
        # Инициализируем сервис если он не был инициализирован
        air_quality_service = AirQualityService()
        logger.info("Air quality service initialized on demand")
    return air_quality_service


@app.get("/", include_in_schema=False)
async def root():
    """Корневой эндпоинт с информацией об API"""
    return {
        "service": "AirTrace RU Backend",
        "version": "1.0.0",
        "description": "Air Quality Monitoring API for Russian cities",
        "endpoints": {
            "current": "/weather/current",
            "forecast": "/weather/forecast", 
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/weather/current", response_model=AirQualityData)
async def get_current_air_quality(
    lat: float = Query(..., ge=-90, le=90, description="Широта"),
    lon: float = Query(..., ge=-180, le=180, description="Долгота"),
    service: AirQualityService = Depends(get_air_quality_service)
):
    """
    Получение текущих данных о качестве воздуха для указанных координат.
    
    Возвращает AQI индекс, рассчитанный по российским стандартам ПДК,
    с рекомендациями на русском языке и определением риска НМУ.
    
    Requirements: 1.1, 1.2, 1.4, 1.5
    """
    try:
        # Дополнительная валидация координат для российской территории
        if not validate_coordinates(lat, lon):
            logger.warning(f"Coordinates outside Russian territory requested")
            # Не блокируем запрос, но логируем предупреждение
        
        # Получение данных через сервис
        data = await service.get_current_air_quality(lat, lon)
        
        logger.info(f"Current air quality data provided - AQI: {data.aqi.value}, Category: {data.aqi.category}")
        return data
        
    except ValueError as e:
        logger.error(f"Validation error in current air quality request: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка валидации данных: {str(e)}"
        )
    except ConnectionError as e:
        logger.error(f"Connection error getting current air quality: {e}")
        raise HTTPException(
            status_code=503,
            detail="Внешний сервис временно недоступен. Попробуйте позже."
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error getting current air quality: {e}")
        raise HTTPException(
            status_code=503,
            detail="Внешний сервис временно недоступен. Попробуйте позже."
        )
    except httpx.RequestError as e:
        logger.error(f"Network error getting current air quality: {e}")
        raise HTTPException(
            status_code=503,
            detail="Внешний сервис временно недоступен. Попробуйте позже."
        )
    except Exception as e:
        logger.error(f"Unexpected error getting current air quality: {e}")
        raise HTTPException(
            status_code=500,
            detail="Временно недоступен сервис получения данных о качестве воздуха"
        )


@app.get("/weather/forecast", response_model=list[AirQualityData])
async def get_forecast_air_quality(
    lat: float = Query(..., ge=-90, le=90, description="Широта"),
    lon: float = Query(..., ge=-180, le=180, description="Долгота"),
    service: AirQualityService = Depends(get_air_quality_service)
):
    """
    Получение прогноза качества воздуха на 24 часа для указанных координат.
    
    Возвращает массив данных с почасовым прогнозом AQI индексов,
    рассчитанных по российским стандартам ПДК.
    
    Requirements: 1.1, 1.2, 1.4, 1.5
    """
    try:
        # Дополнительная валидация координат для российской территории
        if not validate_coordinates(lat, lon):
            logger.warning(f"Coordinates outside Russian territory requested for forecast")
        
        # Получение прогноза через сервис
        data = await service.get_forecast_air_quality(lat, lon)
        
        logger.info(f"Forecast air quality data provided - {len(data)} hours of data")
        return data
        
    except ValueError as e:
        logger.error(f"Validation error in forecast air quality request: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка валидации данных: {str(e)}"
        )
    except ConnectionError as e:
        logger.error(f"Connection error getting forecast air quality: {e}")
        raise HTTPException(
            status_code=503,
            detail="Внешний сервис прогноза временно недоступен. Попробуйте позже."
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error getting forecast air quality: {e}")
        raise HTTPException(
            status_code=503,
            detail="Внешний сервис прогноза временно недоступен. Попробуйте позже."
        )
    except httpx.RequestError as e:
        logger.error(f"Network error getting forecast air quality: {e}")
        raise HTTPException(
            status_code=503,
            detail="Внешний сервис прогноза временно недоступен. Попробуйте позже."
        )
    except Exception as e:
        logger.error(f"Unexpected error getting forecast air quality: {e}")
        raise HTTPException(
            status_code=500,
            detail="Временно недоступен сервис получения прогноза качества воздуха"
        )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Проверка работоспособности сервиса и его компонентов.
    
    Возвращает статус API, подключения к внешним сервисам и кэша.
    Проверяет связность с Open-Meteo API и статус системы кэширования.
    
    Requirements: 9.1, 9.2, 9.3, 9.4
    """
    services_status = {}
    overall_status = "healthy"
    
    try:
        # Проверка статуса API (всегда healthy если мы можем ответить)
        services_status["api"] = "healthy"
        
        # Получаем или создаем сервис для health check
        global air_quality_service
        if air_quality_service is None:
            air_quality_service = AirQualityService()
        
        service = air_quality_service
        
        # Проверка подключения к Open-Meteo API
        try:
            external_api_status = await service.check_external_api_health()
            services_status["external_api"] = external_api_status
            
            if external_api_status not in ["healthy"]:
                overall_status = "unhealthy"
                
        except Exception as e:
            logger.error(f"External API health check failed: {e}")
            services_status["external_api"] = "unhealthy"
            overall_status = "unhealthy"
        
        # Проверка статуса кэша
        try:
            cache_status = service.cache_manager.get_status()
            services_status["cache"] = cache_status
            
            if "unhealthy" in cache_status:
                overall_status = "unhealthy"
                
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            services_status["cache"] = "unhealthy"
            overall_status = "unhealthy"
        
        # Проверка AQI калькулятора
        try:
            test_pollutants = {"pm2_5": 25.0, "pm10": 50.0}
            aqi_value, category, color = aqi_calculator.calculate_aqi(test_pollutants)
            if aqi_value > 0 and category and color:
                services_status["aqi_calculator"] = "healthy"
            else:
                services_status["aqi_calculator"] = "unhealthy"
                overall_status = "unhealthy"
                
        except Exception as e:
            logger.error(f"AQI calculator health check failed: {e}")
            services_status["aqi_calculator"] = "unhealthy"
            overall_status = "unhealthy"
        
        # Проверка privacy middleware
        try:
            privacy_middleware = get_privacy_middleware()
            if privacy_middleware:
                services_status["privacy_middleware"] = "healthy"
            else:
                services_status["privacy_middleware"] = "degraded"
                
        except Exception as e:
            logger.error(f"Privacy middleware health check failed: {e}")
            services_status["privacy_middleware"] = "unhealthy"
        
        # Проверка НМУ детектора
        try:
            nmu_risk = check_nmu_risk({"pm2_5": 30.0, "pm10": 60.0})
            if nmu_risk in ["low", "medium", "high", "critical"]:
                services_status["nmu_detector"] = "healthy"
            else:
                services_status["nmu_detector"] = "degraded"
                
        except Exception as e:
            logger.error(f"NMU detector health check failed: {e}")
            services_status["nmu_detector"] = "unhealthy"
        
        logger.info(f"Health check completed - Overall status: {overall_status}")
        
        return HealthCheckResponse(
            status=overall_status,
            services=services_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed with unexpected error: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            services={
                "api": "healthy",
                "external_api": "unknown",
                "cache": "unknown",
                "aqi_calculator": "unknown",
                "privacy_middleware": "unknown",
                "nmu_detector": "unknown"
            }
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Обработчик HTTP исключений с форматированием ошибок"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    error_response = ErrorResponse(
        code=f"HTTP_{exc.status_code}",
        message=exc.detail
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode='json')
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Обработчик общих исключений с privacy-safe логированием"""
    logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
    
    error_response = ErrorResponse(
        code="INTERNAL_ERROR",
        message="Внутренняя ошибка сервера"
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode='json')
    )


# Дополнительные эндпоинты для мониторинга и отладки

@app.get("/metrics", include_in_schema=False)
async def get_metrics(service: AirQualityService = Depends(get_air_quality_service)):
    """
    Получение метрик системы для мониторинга.
    
    Возвращает базовые метрики работы системы без чувствительных данных.
    """
    try:
        cache_entries = len(service.cache_manager._cache)
        
        return {
            "cache_entries": cache_entries,
            "service_status": "running",
            "components": {
                "aqi_calculator": "active",
                "nmu_detector": "active",
                "privacy_middleware": "active"
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"error": "Metrics unavailable"}


@app.get("/version", include_in_schema=False)
async def get_version():
    """Получение информации о версии API"""
    return {
        "service": "AirTrace RU Backend",
        "version": "1.0.0",
        "api_version": "v1",
        "features": [
            "Russian AQI calculation",
            "NMU risk detection", 
            "Privacy protection",
            "Data caching",
            "Async processing"
        ]
    }


if __name__ == "__main__":
    # Конфигурация для запуска сервера
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,  # Отключено в продакшене для стабильности
        "log_level": "info",
        "access_log": True,
        "server_header": False,  # Скрытие информации о сервере для безопасности
        "date_header": False
    }
    
    logger.info("Starting AirTrace RU Backend server...")
    logger.info(f"Server configuration: {config}")
    
    uvicorn.run(
        "main:app",
        **config
    )