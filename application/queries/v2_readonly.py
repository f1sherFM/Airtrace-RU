"""Public v2 readonly query services and presenters."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Optional

from fastapi import HTTPException

from application.queries.readonly import (
    query_current_air_quality as query_current_air_quality_shared,
    query_forecast_air_quality as query_forecast_air_quality_shared,
    query_history as query_history_shared,
)
from core.settings import get_cities_mapping
from core.legacy_runtime import get_history_snapshot_store
import infrastructure.db as db_runtime_state
from infrastructure.repositories import SQLAlchemyAggregationRepository
from schemas import (
    AirQualityData,
    DailyAggregateRecord,
    HistoryQueryResponse,
    HistoryRange,
    HistorySortOrder,
    HistoryTrendResponse,
    TrendDirection,
    TrendLocation,
    TrendPoint,
    TrendRange,
)

V2_RESPONSE_HEADERS = {
    "X-AirTrace-API-Version": "2",
    "X-AirTrace-API-Contract": "readonly",
}


def _resolve_v2_locator(
    *,
    city: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> tuple[Optional[str], Optional[float], Optional[float]]:
    normalized_city = (city or "").strip().lower() or None
    has_city = normalized_city is not None
    has_lat = lat is not None
    has_lon = lon is not None

    if has_lat != has_lon:
        raise HTTPException(status_code=400, detail="Parameters lat and lon must be provided together")
    if has_city and has_lat:
        raise HTTPException(status_code=400, detail="Provide either city or lat/lon, not both")
    if not has_city and not has_lat:
        raise HTTPException(status_code=400, detail="Provide city or lat/lon")
    if has_city and normalized_city not in get_cities_mapping():
        raise HTTPException(status_code=404, detail="Configured city not found")
    return normalized_city, lat, lon


async def query_current_air_quality_v2(*, lat: float, lon: float) -> AirQualityData:
    return await query_current_air_quality_shared(lat=lat, lon=lon)


async def query_forecast_air_quality_v2(*, lat: float, lon: float, hours: int = 24) -> list[AirQualityData]:
    return await query_forecast_air_quality_shared(lat=lat, lon=lon, hours=hours)


async def query_history_v2(
    *,
    range_value: HistoryRange,
    page: int,
    page_size: int,
    sort: HistorySortOrder,
    city: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> HistoryQueryResponse:
    city, lat, lon = _resolve_v2_locator(city=city, lat=lat, lon=lon)
    return await query_history_shared(
        range_value=range_value,
        page=page,
        page_size=page_size,
        sort=sort,
        city=city,
        lat=lat,
        lon=lon,
    )


def _resolve_trend_delta(range_value: TrendRange) -> timedelta:
    if range_value == TrendRange.LAST_7D:
        return timedelta(days=7)
    return timedelta(days=30)


def _derive_trend(points: list[TrendPoint]) -> TrendDirection:
    if len(points) < 2:
        return TrendDirection.INSUFFICIENT_DATA
    delta = points[-1].aqi_avg - points[0].aqi_avg
    if delta <= -10:
        return TrendDirection.IMPROVING
    if delta >= 10:
        return TrendDirection.WORSENING
    return TrendDirection.STABLE


def _build_trend_summary(*, trend: TrendDirection, points: list[TrendPoint], location: TrendLocation) -> str:
    if not points:
        return f"No daily trend data available for {location.city_code or f'{location.latitude},{location.longitude}'}."

    avg_aqi = round(mean(point.aqi_avg for point in points), 1)
    max_aqi = max(point.aqi_max for point in points)
    label = location.city_code or f"{location.latitude:.4f},{location.longitude:.4f}"
    return f"{label}: trend={trend.value}, avg_aqi={avg_aqi}, max_aqi={max_aqi}, samples={len(points)}."


def _point_from_daily(record: DailyAggregateRecord) -> TrendPoint:
    return TrendPoint(
        timestamp=record.day_utc,
        aqi_min=record.aqi_min,
        aqi_max=record.aqi_max,
        aqi_avg=record.aqi_avg,
        sample_count=record.sample_count,
        avg_confidence=record.avg_confidence,
        dominant_source=record.dominant_source.value,
    )


async def _query_daily_aggregates(
    *,
    start_utc: datetime,
    end_utc: datetime,
    city: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> list[DailyAggregateRecord]:
    if db_runtime_state.database_runtime is not None:
        repository = SQLAlchemyAggregationRepository(db_runtime_state.database_runtime.session_factory)
        return await repository.query_daily_aggregates(
            start_utc=start_utc,
            end_utc=end_utc,
            city_code=city,
            lat=lat,
            lon=lon,
        )

    history_snapshot_store = get_history_snapshot_store()
    if history_snapshot_store is None:
        return []

    query_result = await history_snapshot_store.query_snapshots(
        start_utc=start_utc,
        end_utc=end_utc,
        city_code=city,
        lat=lat,
        lon=lon,
        limit=10000,
        offset=0,
        sort=HistorySortOrder.ASC,
    )
    grouped: dict[tuple[datetime, Optional[str], float, float], list] = {}
    for item in query_result["items"]:
        day_utc = item.snapshot_hour_utc.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        key = (day_utc, item.city_code, item.latitude, item.longitude)
        grouped.setdefault(key, []).append(item)

    aggregates: list[DailyAggregateRecord] = []
    for (day_utc, city_code, latitude, longitude), records in sorted(grouped.items()):
        aqi_values = [record.aqi for record in records]
        dominant_source = records[-1].data_source
        aggregates.append(
            DailyAggregateRecord(
                day_utc=day_utc,
                city_code=city_code,
                latitude=latitude,
                longitude=longitude,
                aqi_min=min(aqi_values),
                aqi_max=max(aqi_values),
                aqi_avg=round(sum(aqi_values) / len(aqi_values), 3),
                sample_count=len(records),
                dominant_source=dominant_source,
                avg_confidence=round(sum(record.confidence for record in records) / len(records), 3),
            )
        )
    return aggregates


async def query_trends_v2(
    *,
    range_value: TrendRange,
    city: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> HistoryTrendResponse:
    city, lat, lon = _resolve_v2_locator(city=city, lat=lat, lon=lon)

    now = datetime.now(timezone.utc)
    start_utc = now - _resolve_trend_delta(range_value)
    aggregates = await _query_daily_aggregates(
        start_utc=start_utc,
        end_utc=now,
        city=city,
        lat=lat,
        lon=lon,
    )
    points = [_point_from_daily(item) for item in aggregates]
    if city is not None:
        if aggregates:
            location = TrendLocation(city_code=city, latitude=aggregates[0].latitude, longitude=aggregates[0].longitude)
        else:
            city_payload = get_cities_mapping().get(city.lower())
            latitude = city_payload["lat"] if city_payload else (lat or 0.0)
            longitude = city_payload["lon"] if city_payload else (lon or 0.0)
            location = TrendLocation(city_code=city, latitude=latitude, longitude=longitude)
    else:
        location = TrendLocation(city_code=None, latitude=lat or 0.0, longitude=lon or 0.0)

    trend = _derive_trend(points)
    summary = _build_trend_summary(trend=trend, points=points, location=location)
    return HistoryTrendResponse(
        range=range_value,
        location=location,
        trend=trend,
        summary=summary,
        points=points,
    )
