"""Legacy readonly API routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from application.queries.health import query_health, query_liveness, query_readiness
from application.queries.readonly import (
    get_root_payload,
    query_current_air_quality,
    query_forecast_air_quality,
    query_history,
)
from schemas import AirQualityData, HealthCheckResponse, HistoryQueryResponse, HistoryRange

router = APIRouter()


@router.get("/", include_in_schema=False)
async def root():
    return get_root_payload()


@router.get("/weather/current", response_model=AirQualityData)
async def get_current_air_quality(
    lat: float = Query(..., ge=-90, le=90, description="Широта"),
    lon: float = Query(..., ge=-180, le=180, description="Долгота"),
):
    return await query_current_air_quality(lat=lat, lon=lon)


@router.get("/weather/forecast", response_model=list[AirQualityData])
async def get_forecast_air_quality(
    lat: float = Query(..., ge=-90, le=90, description="Широта"),
    lon: float = Query(..., ge=-180, le=180, description="Долгота"),
    hours: int = Query(24, ge=1, le=168, description="Горизонт прогноза в часах (1..168)"),
):
    return await query_forecast_air_quality(lat=lat, lon=lon, hours=hours)


@router.get("/history", response_model=HistoryQueryResponse)
async def get_history(
    range: HistoryRange = Query(HistoryRange.LAST_24H, description="Диапазон: 24h, 7d или 30d"),
    page: int = Query(1, ge=1, description="Номер страницы (с 1)"),
    page_size: int = Query(50, ge=1, le=500, description="Размер страницы"),
    city: Optional[str] = Query(None, min_length=2, max_length=64, description="Код города (например, moscow)"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Широта для custom history"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Долгота для custom history"),
):
    return await query_history(range_value=range, page=page, page_size=page_size, city=city, lat=lat, lon=lon)


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return await query_health()


@router.get("/health/liveness", include_in_schema=False)
async def health_liveness():
    return await query_liveness()


@router.get("/health/readiness", include_in_schema=False)
async def health_readiness():
    return await query_readiness()
