"""Legacy non-readonly routes preserved during Stage 1 refactor."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from api.security import require_alert_delivery_auth
from core.legacy_runtime import (
    get_alert_subscription_service,
    get_history_snapshot_store,
    get_telegram_delivery_service,
)
from schemas import (
    AlertEvent,
    AlertRule,
    AlertRuleCreate,
    AlertRuleUpdate,
    DailyDigestResponse,
    DeliveryResult,
    TelegramSendRequest,
)
from unified_weather_service import unified_weather_service

router = APIRouter()


@router.post("/alerts/rules", response_model=AlertRule)
async def create_alert_rule(payload: AlertRuleCreate):
    try:
        return await get_alert_subscription_service().create_legacy_rule(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/alerts/rules", response_model=list[AlertRule])
async def list_alert_rules():
    return await get_alert_subscription_service().list_legacy_rules()


@router.delete("/alerts/rules/{rule_id}")
async def delete_alert_rule(rule_id: str):
    deleted = await get_alert_subscription_service().delete_legacy_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Правило алерта не найдено")
    return {"deleted": True, "rule_id": rule_id}


@router.put("/alerts/rules/{rule_id}", response_model=AlertRule)
async def update_alert_rule(rule_id: str, payload: AlertRuleUpdate):
    try:
        updated = await get_alert_subscription_service().update_legacy_rule(rule_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if updated is None:
        raise HTTPException(status_code=404, detail="Правило алерта не найдено")
    return updated


@router.get("/alerts/check-current", response_model=list[AlertEvent])
async def check_current_alerts(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    data = await unified_weather_service.get_current_combined_data(lat, lon)
    return await get_alert_subscription_service().evaluate_conditions(
        aqi=data.aqi.value,
        nmu_risk=data.nmu_risk,
        lat=lat,
        lon=lon,
    )


@router.post("/alerts/telegram/send", response_model=DeliveryResult)
async def send_telegram_message(
    payload: TelegramSendRequest,
    _auth: None = Depends(require_alert_delivery_auth),
):
    result = await get_telegram_delivery_service().send_message(chat_id=payload.chat_id, text=payload.message)
    return DeliveryResult(**result)


@router.get("/alerts/check-current-and-deliver", response_model=list[DeliveryResult])
async def check_current_alerts_and_deliver(
    _auth: None = Depends(require_alert_delivery_auth),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    chat_id: Optional[str] = Query(None, min_length=1, max_length=128),
):
    data = await unified_weather_service.get_current_combined_data(lat, lon)
    alert_subscription_service = get_alert_subscription_service()
    events = await alert_subscription_service.evaluate_conditions(
        aqi=data.aqi.value,
        nmu_risk=data.nmu_risk,
        lat=lat,
        lon=lon,
    )
    return await alert_subscription_service.deliver_events(
        events=events,
        aqi=data.aqi.value,
        nmu_risk=data.nmu_risk,
        chat_id_override=chat_id,
    )


@router.get("/alerts/delivery-status")
async def get_alert_delivery_status(limit: int = Query(20, ge=1, le=200)):
    telegram_delivery_service = get_telegram_delivery_service()
    return {"items": telegram_delivery_service.status_store.list_recent(limit=limit), "count": limit}


async def _build_daily_digest(
    *,
    city: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> DailyDigestResponse:
    history_snapshot_store = get_history_snapshot_store()
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
        items = list(reversed(result["items"]))

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


@router.get("/alerts/digest/daily", response_model=DailyDigestResponse)
async def get_daily_digest(
    city: Optional[str] = Query(None, min_length=2, max_length=64),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lon: Optional[float] = Query(None, ge=-180, le=180),
):
    if city is None and ((lat is None) != (lon is None)):
        raise HTTPException(status_code=400, detail="Для custom локации передайте lat и lon вместе")
    return await _build_daily_digest(city=city, lat=lat, lon=lon)


@router.get("/alerts/digest/daily-and-deliver", response_model=DeliveryResult)
async def deliver_daily_digest(
    _auth: None = Depends(require_alert_delivery_auth),
    chat_id: str = Query(..., min_length=1, max_length=128),
    city: Optional[str] = Query(None, min_length=2, max_length=64),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lon: Optional[float] = Query(None, ge=-180, le=180),
):
    digest = await _build_daily_digest(city=city, lat=lat, lon=lon)
    message = (
        f"AirTrace Daily Digest\n"
        f"Локация: {digest.location_label}\n"
        f"Период: {digest.period}\n"
        f"Тренд: {digest.trend}\n"
        f"Предупреждения: {'; '.join(digest.top_warnings)}\n"
        f"Рекомендации: {'; '.join(digest.recommended_actions)}"
    )
    result = await get_telegram_delivery_service().send_message(
        chat_id=chat_id,
        text=message,
        event_id=f"digest:{digest.location_label}",
    )
    return DeliveryResult(**result)


async def _fetch_history_records_for_export(
    *,
    hours: int,
    city: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
):
    history_snapshot_store = get_history_snapshot_store()
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


@router.get("/history/export/json")
async def export_history_json(
    hours: int = Query(24, ge=1, le=720, description="Период экспорта в часах (максимум 720 = 30 дней)"),
    city: Optional[str] = Query(None, min_length=2, max_length=64, description="Код города (например, moscow)"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Широта для custom history"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Долгота для custom history"),
):
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


@router.get("/history/export/csv")
async def export_history_csv(
    hours: int = Query(24, ge=1, le=720, description="Период экспорта в часах (максимум 720 = 30 дней)"),
    city: Optional[str] = Query(None, min_length=2, max_length=64, description="Код города (например, moscow)"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Широта для custom history"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Долгота для custom history"),
):
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
