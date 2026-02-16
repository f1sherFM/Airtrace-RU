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
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime, timezone, timedelta
import asyncio
import logging
import uvicorn
import httpx
import os
import csv
import io
import json

from schemas import (
    AirQualityData,
    HealthCheckResponse,
    ErrorResponse,
    CoordinatesRequest,
    HistoryQueryResponse,
    HistoryRange,
    AlertRuleCreate,
    AlertRuleUpdate,
    AlertRule,
    AlertEvent,
    TelegramSendRequest,
    DeliveryResult,
    DailyDigestResponse,
)
from services import AirQualityService
from unified_weather_service import unified_weather_service
from privacy_compliance_validator import privacy_validator, get_privacy_compliance_report
from utils import AQICalculator, check_nmu_risk, validate_coordinates
from middleware import PrivacyMiddleware, setup_privacy_logging, set_privacy_middleware, get_privacy_middleware
from rate_limit_middleware import setup_rate_limiting, get_rate_limit_manager
from rate_limit_monitoring import get_rate_limit_monitor, setup_rate_limit_logging
from connection_pool import get_connection_pool_manager
from graceful_degradation import get_graceful_degradation_manager
from config import config
from history_ingestion import (
    HistoryIngestionPipeline,
    InMemoryHistoricalSnapshotStore,
    JsonlDeadLetterSink,
)
from alert_rule_engine import AlertRuleEngine
from telegram_delivery import TelegramDeliveryService, JsonlDeadLetterSink as TelegramDeadLetterSink

# Настройка privacy-aware логирования
setup_privacy_logging()
logger = logging.getLogger(__name__)

# Глобальный экземпляр сервиса для управления жизненным циклом
air_quality_service: Optional[AirQualityService] = None
history_ingestion_pipeline: Optional[HistoryIngestionPipeline] = None
history_snapshot_store: Optional[InMemoryHistoricalSnapshotStore] = None
alert_rule_engine = AlertRuleEngine()
telegram_delivery_service = TelegramDeliveryService(
    bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    max_retries=int(os.getenv("TELEGRAM_MAX_RETRIES", "3")),
    retry_delay_seconds=float(os.getenv("TELEGRAM_RETRY_DELAY_SECONDS", "0.7")),
    dead_letter_sink=TelegramDeadLetterSink("logs/telegram_dead_letter.jsonl"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения.
    
    Инициализирует и очищает ресурсы при запуске и остановке приложения.
    """
    global air_quality_service, history_ingestion_pipeline, history_snapshot_store
    
    # Startup
    logger.info("Starting AirTrace RU Backend...")
    
    # Инициализация сервисов
    air_quality_service = AirQualityService()
    logger.info("Air quality service initialized")
    
    # Initialize graceful degradation manager
    degradation_manager = get_graceful_degradation_manager()
    
    # Register components for health monitoring
    await degradation_manager.register_component(
        "external_api",
        lambda: air_quality_service.check_external_api_health()
    )
    
    await degradation_manager.register_component(
        "cache",
        lambda: "healthy" in air_quality_service.cache_manager.get_status()
    )
    
    if config.performance.rate_limiting_enabled:
        await degradation_manager.register_component(
            "rate_limiting",
            lambda: get_rate_limit_manager().is_enabled()
        )
    
    if config.weather_api.enabled:
        await degradation_manager.register_component(
            "weather_api",
            lambda: unified_weather_service.check_weather_api_health()
        )
    
    logger.info("Graceful degradation manager initialized")
    
    # Initialize Prometheus exporter if monitoring is enabled
    if config.performance.monitoring_enabled:
        try:
            from prometheus_exporter import setup_prometheus_exporter, setup_default_alerts
            from performance_monitor import setup_performance_monitoring
            
            # Setup performance monitoring
            performance_monitor = setup_performance_monitoring()
            logger.info("Performance monitoring initialized")
            
            # Setup Prometheus exporter with default alerts
            prometheus_exporter = setup_prometheus_exporter()
            setup_default_alerts()
            
            # Start alerting system
            await prometheus_exporter.start_alerting()
            logger.info("Prometheus exporter and alerting initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus monitoring: {e}")
    
    # Запуск фоновых задач
    cleanup_task = asyncio.create_task(periodic_cleanup())
    logger.info("Background cleanup task started")

    # Historical ingestion pipeline (Issue 1.2)
    history_store = InMemoryHistoricalSnapshotStore()
    history_snapshot_store = history_store
    dead_letter_sink = JsonlDeadLetterSink("logs/history_dead_letter.jsonl")
    history_ingestion_pipeline = HistoryIngestionPipeline(
        fetch_current_data=unified_weather_service.get_current_combined_data,
        snapshot_store=history_store,
        dead_letter_sink=dead_letter_sink,
        max_retries=int(os.getenv("HISTORY_INGEST_MAX_RETRIES", "3")),
        retry_delay_seconds=float(os.getenv("HISTORY_INGEST_RETRY_DELAY_SECONDS", "0.5")),
    )
    history_interval = int(os.getenv("HISTORY_INGEST_INTERVAL_SECONDS", "3600"))
    history_task = asyncio.create_task(periodic_history_ingestion(history_interval))
    logger.info("Background historical ingestion task started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AirTrace RU Backend...")
    
    # Остановка фоновых задач
    cleanup_task.cancel()
    history_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    try:
        await history_task
    except asyncio.CancelledError:
        pass
    
    # Cleanup graceful degradation manager
    try:
        await degradation_manager.cleanup()
        logger.info("Graceful degradation manager cleaned up")
    except Exception as e:
        logger.warning(f"Graceful degradation manager cleanup failed: {e}")
    
    # Cleanup Prometheus exporter if monitoring was enabled
    if config.performance.monitoring_enabled:
        try:
            from prometheus_exporter import get_prometheus_exporter
            prometheus_exporter = get_prometheus_exporter()
            await prometheus_exporter.stop_alerting()
            logger.info("Prometheus exporter cleaned up")
        except Exception as e:
            logger.warning(f"Prometheus exporter cleanup failed: {e}")
    
    # Очистка ресурсов unified weather service
    try:
        await unified_weather_service.cleanup()
        logger.info("Unified weather service cleaned up")
    except Exception as e:
        logger.warning(f"Unified weather service cleanup failed: {e}")
    
    # Очистка ресурсов сервисов (fallback)
    if air_quality_service:
        await air_quality_service.cleanup()
        logger.info("Air quality service cleaned up")
    
    # Очистка rate limiting ресурсов
    if config.performance.rate_limiting_enabled:
        try:
            rate_limit_manager = get_rate_limit_manager()
            await rate_limit_manager.cleanup()
            logger.info("Rate limiting cleaned up")
        except Exception as e:
            logger.warning(f"Rate limiting cleanup failed: {e}")
    
    # Очистка connection pool ресурсов
    if config.performance.connection_pooling_enabled:
        try:
            await get_connection_pool_manager().cleanup()
            logger.info("Connection pools cleaned up")
        except Exception as e:
            logger.warning(f"Connection pool cleanup failed: {e}")
    
    logger.info("Shutdown complete")


async def periodic_cleanup():
    """Периодическая очистка кэша и других ресурсов"""
    while True:
        try:
            await asyncio.sleep(300)  # Каждые 5 минут
            # Use unified weather service cache manager
            await unified_weather_service.cache_manager.clear_expired()
            logger.debug("Periodic cache cleanup completed")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}")


async def periodic_history_ingestion(interval_seconds: int = 3600):
    """Периодическая загрузка исторических часовых срезов."""
    global history_ingestion_pipeline

    run_on_startup = os.getenv("HISTORY_INGEST_RUN_ON_STARTUP", "true").lower() == "true"

    # Run one ingestion cycle at startup to seed history.
    if run_on_startup and history_ingestion_pipeline is not None:
        try:
            result = await history_ingestion_pipeline.ingest_once()
            logger.info("Initial history ingestion completed: %s", result)
        except Exception as e:
            logger.error(f"Initial history ingestion failed: {e}")

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            if history_ingestion_pipeline is None:
                continue
            result = await history_ingestion_pipeline.ingest_once()
            logger.info("Periodic history ingestion completed: %s", result)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error during periodic history ingestion: {e}")


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

# Rate limiting middleware для защиты от злоупотреблений
if config.performance.rate_limiting_enabled:
    # Setup rate limiting logging
    setup_rate_limit_logging()
    
    rate_limit_manager = setup_rate_limiting(
        app=app,
        enabled=True,
        skip_paths=["/docs", "/redoc", "/openapi.json", "/", "/version"]
    )
    logger.info("Rate limiting middleware enabled")
else:
    logger.info("Rate limiting middleware disabled")

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


def _normalize_health_status(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"healthy", "ok", "up", "enabled", "active"}:
        return "healthy"
    if normalized in {"disabled"}:
        return "degraded"
    if normalized in {"unhealthy", "down", "error", "failed"}:
        return "unhealthy"
    if normalized in {"degraded", "warning", "unknown"}:
        return "degraded"
    return "degraded"


def _normalize_health_component(value):
    def _merge_statuses(statuses):
        normalized = [_normalize_health_status(s) for s in statuses]
        if not normalized:
            return "degraded"
        if any(s == "unhealthy" for s in normalized):
            return "unhealthy"
        if all(s == "healthy" for s in normalized):
            return "healthy"
        return "degraded"

    def _infer_status_from_mapping(mapping):
        derived_statuses = []
        for item in mapping.values():
            if isinstance(item, dict):
                nested_status = item.get("status")
                if nested_status is not None:
                    derived_statuses.append(str(nested_status))
                else:
                    derived_statuses.append(_infer_status_from_mapping(item))
            elif isinstance(item, str):
                derived_statuses.append(item)
            elif isinstance(item, bool):
                derived_statuses.append("healthy" if item else "unhealthy")
            else:
                derived_statuses.append("degraded")
        return _merge_statuses(derived_statuses)

    if isinstance(value, dict):
        raw_status = value.get("status")
        if raw_status is None:
            raw_status = _infer_status_from_mapping(value)
        normalized_status = _normalize_health_status(str(raw_status))
        details = {k: v for k, v in value.items() if k != "status"}
        return {"status": normalized_status, "details": details}
    if isinstance(value, str):
        return {"status": _normalize_health_status(value), "details": {}}
    return {"status": "degraded", "details": {"raw": value}}


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
    lon: float = Query(..., ge=-180, le=180, description="Долгота")
):
    """
    Получение текущих данных о качестве воздуха с дополнительной информацией о погоде.
    
    Возвращает AQI индекс, рассчитанный по российским стандартам ПДК,
    с рекомендациями на русском языке, определением риска НМУ и данными о погоде
    от WeatherAPI (температура, ветер, давление) при их доступности.
    
    Поддерживает graceful degradation с подачей устаревших данных при медленности API.
    
    Requirements: 1.1, 1.2, 1.4, 1.5, 9.2, 9.3, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
    """
    try:
        if history_ingestion_pipeline is not None:
            history_ingestion_pipeline.register_custom_coordinates(lat, lon)

        # Дополнительная валидация координат для российской территории
        if not validate_coordinates(lat, lon):
            logger.warning(f"Coordinates outside Russian territory requested")
            # Не блокируем запрос, но логируем предупреждение
        
        degradation_manager = get_graceful_degradation_manager()
        cache_key = f"current_{lat}_{lon}"
        
        # Check if we should serve stale data due to API slowness
        try:
            # Try to get fresh data with timeout
            data = await asyncio.wait_for(
                unified_weather_service.get_current_combined_data(lat, lon),
                timeout=config.api.response_timeout if hasattr(config, 'api') else 10.0
            )
            
            # Store data for potential stale serving
            await degradation_manager.store_stale_data(cache_key, data.model_dump())
            
            logger.info(f"Current air quality data with weather provided - AQI: {data.aqi.value}, Category: {data.aqi.category}")
            return data
            
        except asyncio.TimeoutError:
            logger.warning("External API timeout - attempting to serve stale data")
            
            # Try to serve stale data
            stale_data = await degradation_manager.get_stale_data(cache_key)
            if stale_data:
                logger.info("Serving stale air quality data due to API timeout")
                return AirQualityData(**stale_data)
            
            # If no stale data available, try minimal response
            if await degradation_manager.should_prioritize_core_functionality():
                logger.info("Serving minimal response due to system degradation")
                minimal_data = await degradation_manager.get_minimal_response("current")
                return AirQualityData(**minimal_data)
            
            # Re-raise timeout if no fallback available
            raise HTTPException(
                status_code=503,
                detail="Внешний сервис медленно отвечает. Попробуйте позже."
            )
        
        except Exception as e:
            logger.error(f"Error getting current air quality: {e}")
            
            # Try to serve stale data on any error
            stale_data = await degradation_manager.get_stale_data(cache_key)
            if stale_data:
                logger.info("Serving stale air quality data due to API error")
                return AirQualityData(**stale_data)
            
            # Try cached response during rate limiting
            if "rate limit" in str(e).lower() or "429" in str(e):
                cached_response = await degradation_manager.get_cached_response_for_rate_limiting(cache_key)
                if cached_response:
                    logger.info("Serving cached response due to rate limiting")
                    return AirQualityData(**cached_response)
            
            # Fallback to minimal response if system is degraded
            if await degradation_manager.should_prioritize_core_functionality():
                logger.info("Serving minimal response due to system degradation")
                minimal_data = await degradation_manager.get_minimal_response("current")
                return AirQualityData(**minimal_data)
            
            # Standard error handling
            if isinstance(e, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Ошибка валидации данных: {str(e)}"
                )
            elif isinstance(e, (ConnectionError, httpx.RequestError)):
                raise HTTPException(
                    status_code=503,
                    detail="Внешний сервис временно недоступен. Попробуйте позже."
                )
            elif isinstance(e, httpx.HTTPStatusError):
                raise HTTPException(
                    status_code=503,
                    detail="Внешний сервис временно недоступен. Попробуйте позже."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Временно недоступен сервис получения данных о качестве воздуха"
                )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in current air quality endpoint: {e}")
        
        # Final fallback - try to serve any available data
        degradation_manager = get_graceful_degradation_manager()
        cache_key = f"current_{lat}_{lon}"
        stale_data = await degradation_manager.get_stale_data(cache_key)
        if stale_data:
            logger.info("Serving stale data as final fallback")
            return AirQualityData(**stale_data)
        
        # Absolute final fallback
        minimal_data = await degradation_manager.get_minimal_response("current")
        return AirQualityData(**minimal_data)


@app.get("/v2/current", response_model=AirQualityData)
async def get_current_air_quality_v2(
    lat: float = Query(..., ge=-90, le=90, description="Широта"),
    lon: float = Query(..., ge=-180, le=180, description="Долгота"),
):
    """v2 version of current endpoint."""
    return await get_current_air_quality(lat=lat, lon=lon)


@app.get("/weather/forecast", response_model=list[AirQualityData])
async def get_forecast_air_quality(
    lat: float = Query(..., ge=-90, le=90, description="Широта"),
    lon: float = Query(..., ge=-180, le=180, description="Долгота")
):
    """
    Получение прогноза качества воздуха на 24 часа с дополнительной информацией о погоде.
    
    Возвращает массив данных с почасовым прогнозом AQI индексов,
    рассчитанных по российским стандартам ПДК, с данными о погоде при их доступности.
    
    Поддерживает graceful degradation с подачей устаревших данных при медленности API.
    
    Requirements: 1.1, 1.2, 1.4, 1.5, 9.2, 9.3, 11.4, 11.5, 11.6
    """
    try:
        if history_ingestion_pipeline is not None:
            history_ingestion_pipeline.register_custom_coordinates(lat, lon)

        # Дополнительная валидация координат для российской территории
        if not validate_coordinates(lat, lon):
            logger.warning(f"Coordinates outside Russian territory requested for forecast")
        
        degradation_manager = get_graceful_degradation_manager()
        cache_key = f"forecast_{lat}_{lon}"
        
        # Check if we should serve stale data due to API slowness
        try:
            # Try to get fresh data with timeout
            data = await asyncio.wait_for(
                unified_weather_service.get_forecast_combined_data(lat, lon),
                timeout=config.api.response_timeout if hasattr(config, 'api') else 15.0
            )
            
            # Store data for potential stale serving
            forecast_data = [item.model_dump() for item in data]
            await degradation_manager.store_stale_data(cache_key, forecast_data)
            
            logger.info(f"Forecast air quality data with weather provided - {len(data)} hours of data")
            return data
            
        except asyncio.TimeoutError:
            logger.warning("External API timeout for forecast - attempting to serve stale data")
            
            # Try to serve stale data
            stale_data = await degradation_manager.get_stale_data(cache_key)
            if stale_data:
                logger.info("Serving stale forecast data due to API timeout")
                return [AirQualityData(**item) for item in stale_data]
            
            # If no stale data available, try minimal response
            if await degradation_manager.should_prioritize_core_functionality():
                logger.info("Serving minimal forecast response due to system degradation")
                minimal_data = await degradation_manager.get_minimal_response("forecast")
                return minimal_data
            
            # Re-raise timeout if no fallback available
            raise HTTPException(
                status_code=503,
                detail="Внешний сервис прогноза медленно отвечает. Попробуйте позже."
            )
        
        except Exception as e:
            logger.error(f"Error getting forecast air quality: {e}")
            
            # Try to serve stale data on any error
            stale_data = await degradation_manager.get_stale_data(cache_key)
            if stale_data:
                logger.info("Serving stale forecast data due to API error")
                return [AirQualityData(**item) for item in stale_data]
            
            # Try cached response during rate limiting
            if "rate limit" in str(e).lower() or "429" in str(e):
                cached_response = await degradation_manager.get_cached_response_for_rate_limiting(cache_key)
                if cached_response:
                    logger.info("Serving cached forecast response due to rate limiting")
                    return [AirQualityData(**item) for item in cached_response]
            
            # Fallback to minimal response if system is degraded
            if await degradation_manager.should_prioritize_core_functionality():
                logger.info("Serving minimal forecast response due to system degradation")
                minimal_data = await degradation_manager.get_minimal_response("forecast")
                return minimal_data
            
            # Standard error handling
            if isinstance(e, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Ошибка валидации данных: {str(e)}"
                )
            elif isinstance(e, (ConnectionError, httpx.RequestError)):
                raise HTTPException(
                    status_code=503,
                    detail="Внешний сервис прогноза временно недоступен. Попробуйте позже."
                )
            elif isinstance(e, httpx.HTTPStatusError):
                raise HTTPException(
                    status_code=503,
                    detail="Внешний сервис прогноза временно недоступен. Попробуйте позже."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Ошибка обработки прогноза качества воздуха"
                )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in forecast air quality endpoint: {e}")
        
        # Final fallback - try to serve any available data
        degradation_manager = get_graceful_degradation_manager()
        cache_key = f"forecast_{lat}_{lon}"
        stale_data = await degradation_manager.get_stale_data(cache_key)
        if stale_data:
            logger.info("Serving stale forecast data as final fallback")
            return [AirQualityData(**item) for item in stale_data]
        
        # Absolute final fallback
        minimal_data = await degradation_manager.get_minimal_response("forecast")
        return minimal_data


@app.get("/v2/forecast", response_model=list[AirQualityData])
async def get_forecast_air_quality_v2(
    lat: float = Query(..., ge=-90, le=90, description="Широта"),
    lon: float = Query(..., ge=-180, le=180, description="Долгота"),
):
    """v2 version of forecast endpoint."""
    return await get_forecast_air_quality(lat=lat, lon=lon)


@app.get("/history", response_model=HistoryQueryResponse)
async def get_history(
    range: HistoryRange = Query(HistoryRange.LAST_24H, description="Диапазон: 24h, 7d или 30d"),
    page: int = Query(1, ge=1, description="Номер страницы (с 1)"),
    page_size: int = Query(50, ge=1, le=500, description="Размер страницы"),
    city: Optional[str] = Query(None, min_length=2, max_length=64, description="Код города (например, moscow)"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Широта для custom history"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Долгота для custom history"),
):
    """
    Исторические данные качества воздуха с фильтрами диапазона и пагинацией.
    """
    global history_snapshot_store
    if history_snapshot_store is None:
        return HistoryQueryResponse(range=range, page=page, page_size=page_size, total=0, items=[])

    if (lat is None) != (lon is None):
        raise HTTPException(status_code=400, detail="Параметры lat и lon должны передаваться вместе")

    now = datetime.now(timezone.utc)
    if range == HistoryRange.LAST_24H:
        delta = timedelta(hours=24)
    elif range == HistoryRange.LAST_7D:
        delta = timedelta(days=7)
    else:
        delta = timedelta(days=30)

    start_utc = now - delta
    end_utc = now
    offset = (page - 1) * page_size

    query_result = await history_snapshot_store.query_snapshots(
        start_utc=start_utc,
        end_utc=end_utc,
        city_code=city,
        lat=lat,
        lon=lon,
        limit=page_size,
        offset=offset,
    )
    return HistoryQueryResponse(
        range=range,
        page=page,
        page_size=page_size,
        total=query_result["total"],
        items=query_result["items"],
    )


@app.get("/v2/history", response_model=HistoryQueryResponse)
async def get_history_v2(
    range: HistoryRange = Query(HistoryRange.LAST_24H, description="Диапазон: 24h, 7d или 30d"),
    page: int = Query(1, ge=1, description="Номер страницы (с 1)"),
    page_size: int = Query(50, ge=1, le=500, description="Размер страницы"),
    city: Optional[str] = Query(None, min_length=2, max_length=64, description="Код города (например, moscow)"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Широта для custom history"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Долгота для custom history"),
):
    """v2 version of history endpoint."""
    return await get_history(range=range, page=page, page_size=page_size, city=city, lat=lat, lon=lon)


@app.post("/alerts/rules", response_model=AlertRule)
async def create_alert_rule(payload: AlertRuleCreate):
    """Create user-defined alert rule (AQI/NMU, cooldown, quiet hours)."""
    return alert_rule_engine.create_rule(payload)


@app.get("/alerts/rules", response_model=list[AlertRule])
async def list_alert_rules():
    """List alert rules."""
    return alert_rule_engine.list_rules()


@app.delete("/alerts/rules/{rule_id}")
async def delete_alert_rule(rule_id: str):
    """Delete alert rule by id."""
    deleted = alert_rule_engine.delete_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Правило алерта не найдено")
    return {"deleted": True, "rule_id": rule_id}


@app.put("/alerts/rules/{rule_id}", response_model=AlertRule)
async def update_alert_rule(rule_id: str, payload: AlertRuleUpdate):
    """Update alert rule by id."""
    try:
        updated = alert_rule_engine.update_rule(rule_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if updated is None:
        raise HTTPException(status_code=404, detail="Правило алерта не найдено")
    return updated


@app.get("/alerts/check-current", response_model=list[AlertEvent])
async def check_current_alerts(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Evaluate alert rules against current data for location."""
    data = await unified_weather_service.get_current_combined_data(lat, lon)
    return alert_rule_engine.evaluate(
        aqi=data.aqi.value,
        nmu_risk=data.nmu_risk,
    )


@app.post("/alerts/telegram/send", response_model=DeliveryResult)
async def send_telegram_message(payload: TelegramSendRequest):
    """Send message via Telegram channel with retry/dead-letter."""
    result = await telegram_delivery_service.send_message(
        chat_id=payload.chat_id,
        text=payload.message,
    )
    return DeliveryResult(**result)


@app.get("/alerts/check-current-and-deliver", response_model=list[DeliveryResult])
async def check_current_alerts_and_deliver(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    chat_id: Optional[str] = Query(None, min_length=1, max_length=128),
):
    """Evaluate current alerts and deliver unsuppressed alerts to Telegram."""
    data = await unified_weather_service.get_current_combined_data(lat, lon)
    events = alert_rule_engine.evaluate(aqi=data.aqi.value, nmu_risk=data.nmu_risk)
    delivered: list[DeliveryResult] = []
    for idx, event in enumerate(events, start=1):
        if event.suppressed:
            continue
        rule = alert_rule_engine.get_rule(event.rule_id)
        destination_chat_id = chat_id or (rule.chat_id if rule else None)
        if not destination_chat_id:
            continue
        text = (
            f"AirTrace Alert #{idx}\n"
            f"Rule: {event.rule_name}\n"
            f"AQI: {data.aqi.value}\n"
            f"NMU: {data.nmu_risk}\n"
            f"Severity: {event.severity}\n"
            f"Reasons: {', '.join(event.reasons)}"
        )
        event_id = f"{event.rule_id}:{int(event.triggered_at.timestamp())}"
        result = await telegram_delivery_service.send_message(chat_id=destination_chat_id, text=text, event_id=event_id)
        delivered.append(DeliveryResult(**result))
    return delivered


@app.get("/alerts/delivery-status")
async def get_alert_delivery_status(limit: int = Query(20, ge=1, le=200)):
    """Get recent delivery status entries (in-memory tracking)."""
    return {"items": telegram_delivery_service.status_store.list_recent(limit=limit), "count": limit}


async def _build_daily_digest(
    *,
    city: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> DailyDigestResponse:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=24)
    items = []
    if history_snapshot_store is not None:
        result = await history_snapshot_store.query_snapshots(
            start_utc=start,
            end_utc=now,
            city_code=city,
            lat=lat,
            lon=lon,
            limit=500,
            offset=0,
        )
        items = list(reversed(result["items"]))  # chronological

    if not items and lat is not None and lon is not None:
        current = await unified_weather_service.get_current_combined_data(lat, lon)
        items = [current]

    if not items:
        return DailyDigestResponse(
            location_label=city or f"{lat},{lon}",
            trend="stable",
            top_warnings=["Недостаточно данных для полноценного дайджеста"],
            recommended_actions=["Проверьте доступность history данных и повторите позже"],
            summary_text="За последние 24 часа данных недостаточно для тренда.",
        )

    def _aqi_value(item):
        return item.aqi if hasattr(item, "aqi") and isinstance(item.aqi, int) else item.aqi.value

    first_aqi = _aqi_value(items[0])
    last_aqi = _aqi_value(items[-1])
    delta = last_aqi - first_aqi
    if delta >= 15:
        trend = "worsening"
    elif delta <= -15:
        trend = "improving"
    else:
        trend = "stable"

    max_aqi = max(_aqi_value(x) for x in items)
    warnings: list[str] = []
    if max_aqi >= 200:
        warnings.append("Были периоды очень высокого загрязнения (AQI >= 200)")
    elif max_aqi >= 150:
        warnings.append("Были периоды высокого загрязнения (AQI >= 150)")
    anomaly_count = sum(1 for x in items if getattr(x, "anomaly_detected", False))
    if anomaly_count > 0:
        warnings.append(f"Зафиксированы аномалии: {anomaly_count}")
    if not warnings:
        warnings.append("Критических эпизодов не зафиксировано")

    if trend == "worsening":
        actions = [
            "Сократите длительную активность на улице в ближайшие часы",
            "Проветривание переносите на периоды более низкого AQI",
        ]
    elif trend == "improving":
        actions = [
            "Можно планировать короткие прогулки в часы минимального AQI",
            "Сохраните базовую осторожность для чувствительных групп",
        ]
    else:
        actions = [
            "Поддерживайте стандартные меры предосторожности",
            "Отслеживайте обновления при изменении погодных условий",
        ]

    label = city or f"{lat},{lon}"
    summary = f"Дайджест за 24ч для {label}: тренд {trend}, максимум AQI {max_aqi}."
    return DailyDigestResponse(
        location_label=label,
        trend=trend,
        top_warnings=warnings,
        recommended_actions=actions,
        summary_text=summary,
    )


@app.get("/alerts/digest/daily", response_model=DailyDigestResponse)
async def get_daily_digest(
    city: Optional[str] = Query(None, min_length=2, max_length=64),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lon: Optional[float] = Query(None, ge=-180, le=180),
):
    """Build optional daily digest summary for selected location."""
    if city is None and ((lat is None) != (lon is None)):
        raise HTTPException(status_code=400, detail="Для custom локации передайте lat и lon вместе")
    return await _build_daily_digest(city=city, lat=lat, lon=lon)


@app.get("/alerts/digest/daily-and-deliver", response_model=DeliveryResult)
async def deliver_daily_digest(
    chat_id: str = Query(..., min_length=1, max_length=128),
    city: Optional[str] = Query(None, min_length=2, max_length=64),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lon: Optional[float] = Query(None, ge=-180, le=180),
):
    """Build daily digest and deliver to Telegram."""
    digest = await _build_daily_digest(city=city, lat=lat, lon=lon)
    message = (
        f"AirTrace Daily Digest\n"
        f"Локация: {digest.location_label}\n"
        f"Период: {digest.period}\n"
        f"Тренд: {digest.trend}\n"
        f"Предупреждения: {'; '.join(digest.top_warnings)}\n"
        f"Рекомендации: {'; '.join(digest.recommended_actions)}"
    )
    result = await telegram_delivery_service.send_message(chat_id=chat_id, text=message, event_id=f"digest:{digest.location_label}")
    return DeliveryResult(**result)


async def _fetch_history_records_for_export(
    *,
    hours: int,
    city: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
):
    global history_snapshot_store
    if history_snapshot_store is None:
        return []

    now = datetime.now(timezone.utc)
    start_utc = now - timedelta(hours=hours)
    query_result = await history_snapshot_store.query_snapshots(
        start_utc=start_utc,
        end_utc=now,
        city_code=city,
        lat=lat,
        lon=lon,
        limit=50000,
        offset=0,
    )
    return query_result["items"]


@app.get("/history/export/json")
async def export_history_json(
    hours: int = Query(24, ge=1, le=720, description="Период экспорта в часах (максимум 720 = 30 дней)"),
    city: Optional[str] = Query(None, min_length=2, max_length=64, description="Код города (например, moscow)"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Широта для custom history"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Долгота для custom history"),
):
    """
    Экспорт исторических наблюдений в JSON (реальные historical snapshots, не forecast).
    """
    if (lat is None) != (lon is None):
        raise HTTPException(status_code=400, detail="Параметры lat и lon должны передаваться вместе")

    records = await _fetch_history_records_for_export(hours=hours, city=city, lat=lat, lon=lon)
    payload = [item.model_dump(mode="json") for item in records]

    export_target = city if city else f"{lat}_{lon}" if lat is not None and lon is not None else "all"
    filename = f"airtrace_history_{export_target}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-AirTrace-Export-Type": "historical-json",
        },
    )


@app.get("/history/export/csv")
async def export_history_csv(
    hours: int = Query(24, ge=1, le=720, description="Период экспорта в часах (максимум 720 = 30 дней)"),
    city: Optional[str] = Query(None, min_length=2, max_length=64, description="Код города (например, moscow)"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Широта для custom history"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Долгота для custom history"),
):
    """
    Экспорт исторических наблюдений в CSV (реальные historical snapshots, не forecast).
    """
    if (lat is None) != (lon is None):
        raise HTTPException(status_code=400, detail="Параметры lat и lon должны передаваться вместе")

    records = await _fetch_history_records_for_export(hours=hours, city=city, lat=lat, lon=lon)
    output = io.StringIO()
    fieldnames = [
        "snapshot_hour_utc",
        "city_code",
        "latitude",
        "longitude",
        "aqi",
        "pm2_5",
        "pm10",
        "no2",
        "so2",
        "o3",
        "data_source",
        "freshness",
        "confidence",
        "confidence_explanation",
        "fallback_used",
        "cache_age_seconds",
        "ingested_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for item in records:
        pollutants = item.pollutants.model_dump()
        writer.writerow(
            {
                "snapshot_hour_utc": item.snapshot_hour_utc.isoformat(),
                "city_code": item.city_code,
                "latitude": item.latitude,
                "longitude": item.longitude,
                "aqi": item.aqi,
                "pm2_5": pollutants.get("pm2_5"),
                "pm10": pollutants.get("pm10"),
                "no2": pollutants.get("no2"),
                "so2": pollutants.get("so2"),
                "o3": pollutants.get("o3"),
                "data_source": item.data_source.value,
                "freshness": item.freshness.value,
                "confidence": item.confidence,
                "confidence_explanation": item.metadata.confidence_explanation,
                "fallback_used": item.metadata.fallback_used,
                "cache_age_seconds": item.metadata.cache_age_seconds,
                "ingested_at": item.ingested_at.isoformat(),
            }
        )

    export_target = city if city else f"{lat}_{lon}" if lat is not None and lon is not None else "all"
    filename = f"airtrace_history_{export_target}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-AirTrace-Export-Type": "historical-csv",
        },
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Проверка работоспособности сервиса и его компонентов.
    
    Возвращает статус API, подключения к внешним сервисам и кэша.
    Проверяет связность с Open-Meteo API и статус системы кэширования.
    Включает информацию о graceful degradation и fallback механизмах.
    
    Requirements: 9.1, 9.2, 9.3, 9.4, 9.6
    """
    services_status = {}
    overall_status = "healthy"
    
    try:
        # Проверка статуса API (всегда healthy если мы можем ответить)
        services_status["api"] = "healthy"
        
        # Get graceful degradation manager status
        degradation_manager = get_graceful_degradation_manager()
        degradation_status = await degradation_manager.get_comprehensive_health_status()
        
        # Include degradation manager status
        services_status["graceful_degradation"] = {
            "status": degradation_status["overall_status"],
            "system_under_stress": degradation_status["system_under_stress"],
            "prioritize_core_functionality": degradation_status["prioritize_core_functionality"],
            "stale_data_entries": degradation_status["stale_data_entries"],
            "fallback_statistics": degradation_status["fallback_statistics"]
        }
        
        # Update overall status based on degradation manager
        if degradation_status["overall_status"] == "unhealthy":
            overall_status = "unhealthy"
        elif degradation_status["overall_status"] == "degraded" and overall_status == "healthy":
            overall_status = "degraded"
        
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
                if overall_status == "healthy":
                    overall_status = "degraded"
                
        except Exception as e:
            logger.error(f"External API health check failed: {e}")
            services_status["external_api"] = "unhealthy"
            overall_status = "unhealthy"
        
        # Проверка статуса кэша
        try:
            cache_status = service.cache_manager.get_status()
            services_status["cache"] = cache_status
            
            if "unhealthy" in cache_status:
                if overall_status == "healthy":
                    overall_status = "degraded"
                
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            services_status["cache"] = "unhealthy"
            if overall_status == "healthy":
                overall_status = "degraded"
        
        # Проверка AQI калькулятора
        try:
            test_pollutants = {"pm2_5": 25.0, "pm10": 50.0}
            aqi_value, category, color = aqi_calculator.calculate_aqi(test_pollutants)
            if aqi_value > 0 and category and color:
                services_status["aqi_calculator"] = "healthy"
            else:
                services_status["aqi_calculator"] = "unhealthy"
                if overall_status == "healthy":
                    overall_status = "degraded"
                
        except Exception as e:
            logger.error(f"AQI calculator health check failed: {e}")
            services_status["aqi_calculator"] = "unhealthy"
            if overall_status == "healthy":
                overall_status = "degraded"
        
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
        
        # Проверка rate limiting middleware
        try:
            if config.performance.rate_limiting_enabled:
                rate_limit_manager = get_rate_limit_manager()
                if rate_limit_manager and rate_limit_manager.is_enabled():
                    services_status["rate_limiting"] = "healthy"
                else:
                    services_status["rate_limiting"] = "unhealthy"
                    if overall_status == "healthy":
                        overall_status = "degraded"
            else:
                services_status["rate_limiting"] = "disabled"
                
        except Exception as e:
            logger.error(f"Rate limiting health check failed: {e}")
            services_status["rate_limiting"] = "unhealthy"
        
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
        
        # Проверка connection pools
        try:
            if config.performance.connection_pooling_enabled:
                pool_health = await get_connection_pool_manager().health_check_all()
                all_pools_healthy = all(pool_health.values())
                services_status["connection_pools"] = "healthy" if all_pools_healthy else "degraded"
                
                # Add individual pool status
                for service_name, healthy in pool_health.items():
                    services_status[f"pool_{service_name}"] = "healthy" if healthy else "unhealthy"
                    if not healthy and overall_status == "healthy":
                        overall_status = "degraded"
            else:
                services_status["connection_pools"] = "disabled"
                
        except Exception as e:
            logger.error(f"Connection pool health check failed: {e}")
            services_status["connection_pools"] = "unhealthy"
        
        # Проверка WeatherAPI статуса
        try:
            weather_api_health = await unified_weather_service.check_weather_api_health()
            services_status["weather_api"] = weather_api_health["status"]
            
            if weather_api_health["status"] == "unhealthy":
                if overall_status == "healthy":
                    overall_status = "degraded"  # WeatherAPI is optional, so degraded not unhealthy
                
        except Exception as e:
            logger.error(f"WeatherAPI health check failed: {e}")
            services_status["weather_api"] = "unhealthy"
        
        # Add fallback capabilities status
        services_status["fallback_capabilities"] = {
            "stale_data_serving": "enabled",
            "cached_response_serving": "enabled",
            "minimal_response_generation": "enabled",
            "core_functionality_prioritization": "enabled"
        }
        
        logger.info(f"Health check completed - Overall status: {overall_status}")
        
        normalized_services = {name: _normalize_health_component(value) for name, value in services_status.items()}
        overall_from_components = "healthy"
        for comp in normalized_services.values():
            st = comp["status"]
            if st == "unhealthy":
                overall_from_components = "unhealthy"
                break
            if st == "degraded":
                overall_from_components = "degraded"
        normalized_overall = _normalize_health_status(overall_status)
        if overall_from_components == "unhealthy":
            normalized_overall = "unhealthy"
        elif overall_from_components == "degraded" and normalized_overall == "healthy":
            normalized_overall = "degraded"

        return HealthCheckResponse(
            status=normalized_overall,
            services=normalized_services
        )
        
    except Exception as e:
        logger.error(f"Health check failed with unexpected error: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            services={
                "api": {"status": "healthy", "details": {}},
                "external_api": {"status": "degraded", "details": {"reason": "unknown"}},
                "cache": {"status": "degraded", "details": {"reason": "unknown"}},
                "aqi_calculator": {"status": "degraded", "details": {"reason": "unknown"}},
                "privacy_middleware": {"status": "degraded", "details": {"reason": "unknown"}},
                "nmu_detector": {"status": "degraded", "details": {"reason": "unknown"}},
                "graceful_degradation": {"status": "degraded", "details": {"reason": "unknown"}},
                "fallback_capabilities": {"status": "degraded", "details": {"reason": "unknown"}}
            }
        )


@app.get("/v2/health", response_model=HealthCheckResponse)
async def health_check_v2():
    """v2 normalized health endpoint."""
    return await health_check()


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
async def get_metrics():
    """
    Получение метрик системы для мониторинга.
    
    Возвращает базовые метрики работы системы без чувствительных данных,
    включая статистику WeatherAPI интеграции.
    """
    try:
        # Get unified service statistics
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
                "weather_api_integration": "active" if config.weather_api.enabled else "disabled"
            }
        }
        
        # Add rate limiting metrics if enabled
        if config.performance.rate_limiting_enabled:
            try:
                rate_limit_manager = get_rate_limit_manager()
                rate_limit_stats = await rate_limit_manager.get_comprehensive_stats()
                metrics["rate_limiting"] = rate_limit_stats
                metrics["components"]["rate_limiting"] = "active"
            except Exception as e:
                logger.warning(f"Failed to get rate limiting metrics: {e}")
                metrics["components"]["rate_limiting"] = "error"
        else:
            metrics["components"]["rate_limiting"] = "disabled"
        
        # Add connection pool metrics if enabled
        if config.performance.connection_pooling_enabled:
            try:
                pool_stats = await get_connection_pool_manager().get_all_stats()
                metrics["connection_pools"] = pool_stats
                metrics["components"]["connection_pooling"] = "active"
            except Exception as e:
                logger.warning(f"Failed to get connection pool metrics: {e}")
                metrics["components"]["connection_pooling"] = "error"
        else:
            metrics["components"]["connection_pooling"] = "disabled"
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"error": "Metrics unavailable"}


@app.get("/metrics/prometheus", include_in_schema=False, response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus exposition format for monitoring
    and alerting systems. Includes comprehensive metrics from all
    performance optimization components.
    """
    try:
        from prometheus_exporter import get_prometheus_exporter
        exporter = get_prometheus_exporter()
        return await exporter.export_metrics()
    except Exception as e:
        logger.error(f"Error exporting Prometheus metrics: {e}")
        return "# Error exporting metrics\n"


@app.get("/metrics/validate", include_in_schema=False)
async def validate_prometheus_metrics():
    """
    Validate Prometheus metrics completeness and format compliance.
    
    Returns validation results including missing metrics and format issues.
    """
    try:
        from prometheus_exporter import get_prometheus_exporter
        exporter = get_prometheus_exporter()
        validation_result = await exporter.validate_metrics_completeness()
        return validation_result
    except Exception as e:
        logger.error(f"Error validating Prometheus metrics: {e}")
        return {"error": "Metrics validation failed", "details": str(e)}


@app.get("/metrics/comprehensive", include_in_schema=False)
async def get_comprehensive_metrics():
    """
    Get comprehensive metrics from all performance optimization components.
    
    Returns detailed metrics in JSON format including performance monitoring,
    caching, rate limiting, connection pooling, and resource management.
    """
    try:
        comprehensive_metrics = {}
        
        # Performance monitoring metrics
        try:
            from performance_monitor import get_performance_monitor
            monitor = get_performance_monitor()
            comprehensive_metrics["performance"] = {
                "stats": monitor.get_performance_stats().__dict__,
                "endpoint_stats": monitor.get_endpoint_stats(),
                "summary": monitor.get_metrics_summary()
            }
        except Exception as e:
            comprehensive_metrics["performance"] = {"error": str(e)}
        
        # Cache metrics
        try:
            from unified_weather_service import unified_weather_service
            if hasattr(unified_weather_service, 'cache_manager'):
                cache_stats = await unified_weather_service.cache_manager.get_stats()
                comprehensive_metrics["cache"] = cache_stats.__dict__
        except Exception as e:
            comprehensive_metrics["cache"] = {"error": str(e)}
        
        # Rate limiting metrics
        try:
            from rate_limiter import get_rate_limiter
            rate_limiter = get_rate_limiter()
            rate_metrics = await rate_limiter.get_metrics()
            comprehensive_metrics["rate_limiting"] = rate_metrics.__dict__
        except Exception as e:
            comprehensive_metrics["rate_limiting"] = {"error": str(e)}
        
        # Connection pool metrics
        try:
            from connection_pool import get_connection_pool_manager
            pool_manager = get_connection_pool_manager()
            pool_stats = await pool_manager.get_all_stats()
            comprehensive_metrics["connection_pools"] = pool_stats
        except Exception as e:
            comprehensive_metrics["connection_pools"] = {"error": str(e)}
        
        # Resource management metrics
        try:
            from resource_manager import get_resource_manager
            resource_manager = get_resource_manager()
            resource_usage = await resource_manager.get_resource_usage()
            comprehensive_metrics["resource_management"] = resource_usage.__dict__
        except Exception as e:
            comprehensive_metrics["resource_management"] = {"error": str(e)}
        
        # System metrics
        try:
            from system_monitor import get_system_monitor
            system_monitor = get_system_monitor()
            if system_monitor.metrics_history:
                latest_metrics = system_monitor.metrics_history[-1]
                comprehensive_metrics["system"] = latest_metrics.__dict__
        except Exception as e:
            comprehensive_metrics["system"] = {"error": str(e)}
        
        # WeatherAPI metrics
        try:
            from weather_api_manager import get_weather_api_manager
            weather_api_manager = get_weather_api_manager()
            api_stats = await weather_api_manager.get_api_status()
            comprehensive_metrics["weatherapi"] = api_stats
        except Exception as e:
            comprehensive_metrics["weatherapi"] = {"error": str(e)}
        
        # Request optimization metrics
        try:
            from request_optimizer import get_request_optimizer
            optimizer = get_request_optimizer()
            optimization_stats = await optimizer.get_optimization_stats()
            comprehensive_metrics["request_optimization"] = optimization_stats
        except Exception as e:
            comprehensive_metrics["request_optimization"] = {"error": str(e)}
        
        # Alert metrics
        try:
            from prometheus_exporter import get_prometheus_exporter
            exporter = get_prometheus_exporter()
            active_alerts = exporter.get_active_alerts()
            alert_history = exporter.get_alert_history(limit=10)
            comprehensive_metrics["alerts"] = {
                "active_alerts": [alert.__dict__ for alert in active_alerts],
                "recent_history": [alert.__dict__ for alert in alert_history]
            }
        except Exception as e:
            comprehensive_metrics["alerts"] = {"error": str(e)}
        
        return comprehensive_metrics
        
    except Exception as e:
        logger.error(f"Error getting comprehensive metrics: {e}")
        return {"error": "Comprehensive metrics unavailable", "details": str(e)}


@app.get("/rate-limit-status", include_in_schema=False)
async def get_rate_limit_status():
    """
    Получение статуса и метрик системы rate limiting.
    
    Возвращает подробную информацию о работе системы ограничения запросов.
    """
    if not config.performance.rate_limiting_enabled:
        return {"status": "disabled", "message": "Rate limiting is not enabled"}
    
    try:
        rate_limit_manager = get_rate_limit_manager()
        monitor = get_rate_limit_monitor()
        
        return {
            "status": "enabled",
            "comprehensive_stats": await rate_limit_manager.get_comprehensive_stats(),
            "recent_violations": monitor.get_violation_summary(),
            "endpoint_statistics": monitor.get_endpoint_statistics()
        }
    except Exception as e:
        logger.error(f"Error getting rate limit status: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/privacy-compliance", include_in_schema=False)
async def get_privacy_compliance_status():
    """
    Получение статуса соблюдения приватности системы.
    
    Возвращает отчет о соблюдении требований приватности,
    включая проверку кэш-ключей, метрик и логирования координат.
    """
    try:
        # Reset validator for fresh check
        privacy_validator.reset()
        
        # Test cache key privacy
        test_cache_key = unified_weather_service.cache_manager._generate_key(55.7558, 37.6176)
        privacy_validator.validate_cache_key_privacy(test_cache_key, "privacy_compliance_check")
        
        # Test metrics privacy
        test_metrics = await unified_weather_service.get_service_statistics()
        privacy_validator.validate_metrics_anonymization(test_metrics, "privacy_compliance_check")
        
        # Test performance monitoring privacy
        if config.performance.monitoring_enabled:
            try:
                from performance_monitor import PerformanceMonitor
                monitor = PerformanceMonitor()
                test_monitoring_data = {
                    "endpoint": "/weather/current",
                    "method": "GET",
                    "duration": 0.5,
                    "status_code": 200
                }
                privacy_validator.validate_performance_monitoring_privacy(
                    test_monitoring_data, 
                    "privacy_compliance_check"
                )
            except Exception as e:
                logger.warning(f"Performance monitoring privacy check failed: {e}")
        
        # Generate compliance report
        report = privacy_validator.generate_compliance_report()
        
        return {
            "privacy_compliance": {
                "is_compliant": report.is_compliant,
                "compliance_score": report.compliance_score,
                "total_checks": report.total_checks,
                "passed_checks": report.passed_checks,
                "violations_count": len(report.violations),
                "warnings_count": len(report.warnings),
                "validation_timestamp": report.validation_timestamp.isoformat()
            },
            "violations": [
                {
                    "type": v.violation_type.value,
                    "description": v.description,
                    "location": v.location,
                    "severity": v.severity,
                    "timestamp": v.timestamp.isoformat()
                }
                for v in report.violations
            ],
            "warnings": report.warnings,
            "recommendations": report.recommendations,
            "privacy_settings": {
                "coordinate_hashing_enabled": config.cache.hash_coordinates,
                "coordinate_precision": config.cache.coordinate_precision,
                "cache_key_prefix": config.cache.l2_key_prefix,
                "privacy_middleware_enabled": True
            }
        }
    except Exception as e:
        logger.error(f"Privacy compliance check failed: {e}")
        return {
            "privacy_compliance": {
                "is_compliant": False,
                "compliance_score": 0.0,
                "error": str(e)
            }
        }


@app.get("/system-status", include_in_schema=False)
async def get_system_status():
    """
    Получение детального статуса системы с информацией о graceful degradation.
    
    Возвращает подробную информацию о состоянии всех компонентов системы,
    статистику fallback механизмов и рекомендации по оптимизации.
    """
    try:
        degradation_manager = get_graceful_degradation_manager()
        comprehensive_status = await degradation_manager.get_comprehensive_health_status()
        
        # Add additional system information
        system_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "degradation_status": comprehensive_status,
            "configuration": {
                "rate_limiting_enabled": config.performance.rate_limiting_enabled,
                "connection_pooling_enabled": config.performance.connection_pooling_enabled,
                "weather_api_enabled": config.weather_api.enabled,
                "redis_enabled": config.performance.redis_enabled,
                "monitoring_enabled": getattr(config.performance, 'monitoring_enabled', False)
            }
        }
        
        return system_info
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "status": "error"
        }


@app.get("/config-audit", include_in_schema=False)
async def get_configuration_audit():
    """
    Получение аудита изменений конфигурации.
    
    Возвращает историю изменений конфигурации с информацией о влиянии на производительность.
    """
    try:
        audit_manager = config.get_audit_manager()
        if not audit_manager:
            return {"status": "disabled", "message": "Configuration audit is not enabled"}
        
        # Get recent audit trail
        audit_trail = audit_manager.get_audit_trail(limit=100)
        
        # Get performance impact summary
        impact_summary = audit_manager.get_performance_impact_summary(
            time_range=timedelta(hours=24)
        )
        
        # Format audit trail for response
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
                    "validation_errors": entry.change.validation_errors
                },
                "rollback_available": entry.rollback_available
            }
            
            if entry.performance_impact:
                entry_data["performance_impact"] = {
                    "impact_severity": entry.performance_impact.impact_severity,
                    "impact_metrics": entry.performance_impact.impact_metrics,
                    "recommendations": entry.performance_impact.recommendations
                }
            
            formatted_trail.append(entry_data)
        
        return {
            "status": "enabled",
            "audit_trail": formatted_trail,
            "performance_impact_summary": impact_summary,
            "total_entries": len(formatted_trail)
        }
        
    except Exception as e:
        logger.error(f"Error getting configuration audit: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/config-audit/component/{component}", include_in_schema=False)
async def get_component_configuration_audit(component: str):
    """
    Получение аудита изменений конфигурации для конкретного компонента.
    
    Args:
        component: Имя компонента (redis, cache, rate_limiting, etc.)
    """
    try:
        audit_manager = config.get_audit_manager()
        if not audit_manager:
            return {"status": "disabled", "message": "Configuration audit is not enabled"}
        
        # Get audit trail for specific component
        audit_trail = audit_manager.get_audit_trail(
            component=component,
            time_range=timedelta(days=7),
            limit=50
        )
        
        # Format response
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
                "validation_status": entry.change.validation_status
            }
            
            if entry.performance_impact:
                entry_data["impact_severity"] = entry.performance_impact.impact_severity
                entry_data["impact_metrics"] = entry.performance_impact.impact_metrics
            
            formatted_trail.append(entry_data)
        
        return {
            "component": component,
            "audit_trail": formatted_trail,
            "total_entries": len(formatted_trail)
        }
        
    except Exception as e:
        logger.error(f"Error getting component configuration audit: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/config-audit/snapshot", include_in_schema=False)
async def create_configuration_snapshot():
    """
    Создание снимка текущей конфигурации для возможности отката.
    """
    try:
        audit_manager = config.get_audit_manager()
        if not audit_manager:
            return {"status": "disabled", "message": "Configuration audit is not enabled"}
        
        snapshot_id = audit_manager.create_configuration_snapshot()
        
        if snapshot_id:
            return {
                "status": "success",
                "snapshot_id": snapshot_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {"status": "error", "message": "Failed to create configuration snapshot"}
        
    except Exception as e:
        logger.error(f"Error creating configuration snapshot: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/config-audit/performance-impact", include_in_schema=False)
async def get_configuration_performance_impact():
    """
    Получение сводки влияния изменений конфигурации на производительность.
    """
    try:
        audit_manager = config.get_audit_manager()
        if not audit_manager:
            return {"status": "disabled", "message": "Configuration audit is not enabled"}
        
        # Get performance impact summary for different time ranges
        impact_24h = audit_manager.get_performance_impact_summary(timedelta(hours=24))
        impact_7d = audit_manager.get_performance_impact_summary(timedelta(days=7))
        impact_30d = audit_manager.get_performance_impact_summary(timedelta(days=30))
        
        return {
            "status": "enabled",
            "performance_impact": {
                "last_24_hours": impact_24h,
                "last_7_days": impact_7d,
                "last_30_days": impact_30d
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting configuration performance impact: {e}")
        return {"status": "error", "error": str(e)}


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
            "Async processing",
            "Configuration audit trail"
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
