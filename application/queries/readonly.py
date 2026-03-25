"""Shared readonly query services used by both legacy and v2 routers."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import HTTPException

from config import config
from core.legacy_runtime import (
    get_air_quality_service,
    get_history_ingestion_pipeline,
    get_history_snapshot_store,
    set_air_quality_service,
)
from graceful_degradation import get_graceful_degradation_manager
from history_ingestion import InMemoryHistoricalSnapshotStore
from schemas import AirQualityData, HistoryQueryResponse, HistoryRange, HistorySortOrder
from services import AirQualityService
from unified_weather_service import unified_weather_service
from utils import validate_coordinates

logger = logging.getLogger(__name__)


def get_root_payload() -> dict[str, object]:
    return {
        "service": "AirTrace RU Backend",
        "version": "0.3.1",
        "description": "Air Quality Monitoring API for Russian cities",
        "endpoints": {
            "current": "/weather/current",
            "forecast": "/weather/forecast",
            "health": "/health",
            "docs": "/docs",
        },
    }


async def query_current_air_quality(lat: float, lon: float) -> AirQualityData:
    try:
        history_ingestion_pipeline = get_history_ingestion_pipeline()
        if history_ingestion_pipeline is not None:
            history_ingestion_pipeline.register_custom_coordinates(lat, lon)

        if not validate_coordinates(lat, lon):
            logger.warning("Coordinates outside Russian territory requested")

        degradation_manager = get_graceful_degradation_manager()
        cache_key = f"current_{lat}_{lon}"

        try:
            data = await asyncio.wait_for(
                unified_weather_service.get_current_combined_data(lat, lon),
                timeout=config.api.response_timeout if hasattr(config, "api") else 10.0,
            )
            await degradation_manager.store_stale_data(cache_key, data.model_dump())
            logger.info(
                "Current air quality data with weather provided - AQI: %s, Category: %s",
                data.aqi.value,
                data.aqi.category,
            )
            return data
        except asyncio.TimeoutError:
            logger.warning("External API timeout - attempting to serve stale data")
            stale_data = await degradation_manager.get_stale_data(cache_key)
            if stale_data:
                logger.info("Serving stale air quality data due to API timeout")
                return AirQualityData(**stale_data)

            if await degradation_manager.should_prioritize_core_functionality():
                logger.info("Serving minimal response due to system degradation")
                minimal_data = await degradation_manager.get_minimal_response("current")
                return AirQualityData(**minimal_data)

            raise HTTPException(
                status_code=503,
                detail="Внешний сервис медленно отвечает. Попробуйте позже.",
            )
        except Exception as exc:
            logger.error("Error getting current air quality: %s", exc)

            stale_data = await degradation_manager.get_stale_data(cache_key)
            if stale_data:
                logger.info("Serving stale air quality data due to API error")
                return AirQualityData(**stale_data)

            if "rate limit" in str(exc).lower() or "429" in str(exc):
                cached_response = await degradation_manager.get_cached_response_for_rate_limiting(cache_key)
                if cached_response:
                    logger.info("Serving cached response due to rate limiting")
                    return AirQualityData(**cached_response)

            if await degradation_manager.should_prioritize_core_functionality():
                logger.info("Serving minimal response due to system degradation")
                minimal_data = await degradation_manager.get_minimal_response("current")
                return AirQualityData(**minimal_data)

            if isinstance(exc, ValueError):
                raise HTTPException(status_code=400, detail=f"Ошибка валидации данных: {exc}")
            if isinstance(exc, (ConnectionError, httpx.RequestError, httpx.HTTPStatusError)):
                raise HTTPException(
                    status_code=503,
                    detail="Внешний сервис временно недоступен. Попробуйте позже.",
                )
            raise HTTPException(
                status_code=500,
                detail="Временно недоступен сервис получения данных о качестве воздуха",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error in current air quality endpoint: %s", exc)
        degradation_manager = get_graceful_degradation_manager()
        cache_key = f"current_{lat}_{lon}"
        stale_data = await degradation_manager.get_stale_data(cache_key)
        if stale_data:
            logger.info("Serving stale data as final fallback")
            return AirQualityData(**stale_data)

        minimal_data = await degradation_manager.get_minimal_response("current")
        return AirQualityData(**minimal_data)


async def query_forecast_air_quality(lat: float, lon: float, hours: int = 24) -> list[AirQualityData]:
    try:
        history_ingestion_pipeline = get_history_ingestion_pipeline()
        if history_ingestion_pipeline is not None:
            history_ingestion_pipeline.register_custom_coordinates(lat, lon)

        if not validate_coordinates(lat, lon):
            logger.warning("Coordinates outside Russian territory requested for forecast")

        degradation_manager = get_graceful_degradation_manager()
        cache_key = f"forecast_{lat}_{lon}_{hours}h"

        try:
            data = await asyncio.wait_for(
                unified_weather_service.get_forecast_combined_data(lat, lon, hours=hours),
                timeout=config.api.response_timeout if hasattr(config, "api") else 15.0,
            )
            forecast_data = [item.model_dump() for item in data]
            await degradation_manager.store_stale_data(cache_key, forecast_data)
            logger.info("Forecast air quality data with weather provided - %s hours of data", len(data))
            return data
        except asyncio.TimeoutError:
            logger.warning("External API timeout for forecast - attempting to serve stale data")
            stale_data = await degradation_manager.get_stale_data(cache_key)
            if stale_data:
                logger.info("Serving stale forecast data due to API timeout")
                return [AirQualityData(**item) for item in stale_data]

            if await degradation_manager.should_prioritize_core_functionality():
                logger.info("Serving minimal forecast response due to system degradation")
                return await degradation_manager.get_minimal_response("forecast")

            raise HTTPException(
                status_code=503,
                detail="Внешний сервис прогноза медленно отвечает. Попробуйте позже.",
            )
        except Exception as exc:
            logger.error("Error getting forecast air quality: %s", exc)
            stale_data = await degradation_manager.get_stale_data(cache_key)
            if stale_data:
                logger.info("Serving stale forecast data due to API error")
                return [AirQualityData(**item) for item in stale_data]

            if "rate limit" in str(exc).lower() or "429" in str(exc):
                cached_response = await degradation_manager.get_cached_response_for_rate_limiting(cache_key)
                if cached_response:
                    logger.info("Serving cached forecast response due to rate limiting")
                    return [AirQualityData(**item) for item in cached_response]

            if await degradation_manager.should_prioritize_core_functionality():
                logger.info("Serving minimal forecast response due to system degradation")
                return await degradation_manager.get_minimal_response("forecast")

            if isinstance(exc, ValueError):
                raise HTTPException(status_code=400, detail=f"Ошибка валидации данных: {exc}")
            if isinstance(exc, (ConnectionError, httpx.RequestError, httpx.HTTPStatusError)):
                raise HTTPException(
                    status_code=503,
                    detail="Внешний сервис прогноза временно недоступен. Попробуйте позже.",
                )
            raise HTTPException(status_code=500, detail="Ошибка обработки прогноза качества воздуха")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error in forecast air quality endpoint: %s", exc)
        degradation_manager = get_graceful_degradation_manager()
        cache_key = f"forecast_{lat}_{lon}_{hours}h"
        stale_data = await degradation_manager.get_stale_data(cache_key)
        if stale_data:
            logger.info("Serving stale forecast data as final fallback")
            return [AirQualityData(**item) for item in stale_data]
        return await degradation_manager.get_minimal_response("forecast")


def _resolve_history_delta(range_value: HistoryRange) -> timedelta:
    if range_value == HistoryRange.LAST_24H:
        return timedelta(hours=24)
    if range_value == HistoryRange.LAST_7D:
        return timedelta(days=7)
    return timedelta(days=30)


async def query_history(
    *,
    range_value: HistoryRange,
    page: int,
    page_size: int,
    sort: HistorySortOrder = HistorySortOrder.DESC,
    city: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> HistoryQueryResponse:
    history_snapshot_store: Optional[InMemoryHistoricalSnapshotStore] = get_history_snapshot_store()
    if history_snapshot_store is None:
        return HistoryQueryResponse(range=range_value, page=page, page_size=page_size, total=0, items=[])

    if (lat is None) != (lon is None):
        raise HTTPException(status_code=400, detail="Параметры lat и lon должны передаваться вместе")

    now = datetime.now(timezone.utc)
    start_utc = now - _resolve_history_delta(range_value)
    offset = (page - 1) * page_size
    query_result = await history_snapshot_store.query_snapshots(
        start_utc=start_utc,
        end_utc=now,
        city_code=city,
        lat=lat,
        lon=lon,
        limit=page_size,
        offset=offset,
        sort=sort,
    )
    return HistoryQueryResponse(
        range=range_value,
        page=page,
        page_size=page_size,
        total=query_result["total"],
        items=query_result["items"],
    )


def get_or_create_air_quality_service() -> AirQualityService:
    service = get_air_quality_service()
    if service is None:
        service = AirQualityService()
        set_air_quality_service(service)
        logger.info("Air quality service initialized on demand")
    return service
