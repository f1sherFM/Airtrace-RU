"""Stable read-only v2 API routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Response

from application.queries.health import query_health, query_liveness, query_readiness
from application.queries.v2_readonly import (
    V2_RESPONSE_HEADERS,
    query_current_air_quality_v2,
    query_forecast_air_quality_v2,
    query_history_v2,
    query_trends_v2,
)
from schemas import (
    AirQualityData,
    ErrorResponse,
    HealthCheckResponse,
    HistoryQueryResponse,
    HistoryRange,
    HistorySortOrder,
    HistoryTrendResponse,
    TrendRange,
)

router = APIRouter()


def _apply_v2_headers(response: Response) -> None:
    for header, value in V2_RESPONSE_HEADERS.items():
        response.headers[header] = value


V2_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.get("/v2/current", response_model=AirQualityData, responses=V2_ERROR_RESPONSES)
async def get_current_air_quality_v2(
    response: Response,
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
):
    _apply_v2_headers(response)
    return await query_current_air_quality_v2(lat=lat, lon=lon)


@router.get("/v2/forecast", response_model=list[AirQualityData], responses=V2_ERROR_RESPONSES)
async def get_forecast_air_quality_v2(
    response: Response,
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    hours: int = Query(24, ge=1, le=168, description="Forecast horizon in hours"),
):
    _apply_v2_headers(response)
    return await query_forecast_air_quality_v2(lat=lat, lon=lon, hours=hours)


@router.get("/v2/history", response_model=HistoryQueryResponse, responses=V2_ERROR_RESPONSES)
async def get_history_v2(
    response: Response,
    range: HistoryRange = Query(HistoryRange.LAST_24H, description="History range preset"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Page size"),
    sort: HistorySortOrder = Query(
        HistorySortOrder.DESC,
        description="Sort order: desc = newest first, asc = oldest first",
    ),
    city: Optional[str] = Query(None, min_length=2, max_length=64, description="Configured city code"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Custom latitude"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Custom longitude"),
):
    _apply_v2_headers(response)
    return await query_history_v2(
        range_value=range,
        page=page,
        page_size=page_size,
        sort=sort,
        city=city,
        lat=lat,
        lon=lon,
    )


@router.get("/v2/trends", response_model=HistoryTrendResponse, responses=V2_ERROR_RESPONSES)
async def get_history_trends_v2(
    response: Response,
    range: TrendRange = Query(TrendRange.LAST_7D, description="Trend range preset"),
    city: Optional[str] = Query(None, min_length=2, max_length=64, description="Configured city code"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Custom latitude"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Custom longitude"),
):
    _apply_v2_headers(response)
    return await query_trends_v2(range_value=range, city=city, lat=lat, lon=lon)


@router.get("/v2/health", response_model=HealthCheckResponse, responses=V2_ERROR_RESPONSES)
async def health_check_v2(response: Response):
    _apply_v2_headers(response)
    return await query_health()


@router.get("/v2/liveness", include_in_schema=False)
async def health_liveness_v2(response: Response):
    _apply_v2_headers(response)
    return await query_liveness()


@router.get("/v2/readiness", include_in_schema=False)
async def health_readiness_v2(response: Response):
    _apply_v2_headers(response)
    return await query_readiness()
