"""Direct application-layer service used by the Python SSR web app."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import HTTPException

from application.queries.health import query_health
from application.queries.readonly import (
    query_current_air_quality,
    query_forecast_air_quality,
    query_history,
)
from application.queries.v2_readonly import query_trends_v2
from application.services.alerts import AlertSubscriptionService
from core.legacy_runtime import (
    get_alert_subscription_service,
    get_history_snapshot_store,
    get_telegram_delivery_service,
)
from infrastructure.repositories import (
    InMemoryAlertAuditRepository,
    InMemoryAlertDeliveryAttemptRepository,
    InMemoryAlertIdempotencyRepository,
    InMemoryAlertSubscriptionRepository,
)
from schemas import (
    AlertRuleCreate,
    AlertRuleUpdate,
    AlertSubscriptionCreate,
    AlertSubscriptionUpdate,
    DailyDigestResponse,
    HistoryRange,
    HistorySortOrder,
    TrendRange,
)
from telegram_delivery import JsonlDeadLetterSink, TelegramDeliveryService


_fallback_alert_service: AlertSubscriptionService | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_history_delta(range_value: HistoryRange) -> timedelta:
    if range_value == HistoryRange.LAST_24H:
        return timedelta(hours=24)
    if range_value == HistoryRange.LAST_7D:
        return timedelta(days=7)
    return timedelta(days=30)


class WebAppService:
    """Application-layer adapter for SSR routes."""

    async def get_current_data(self, lat: float, lon: float) -> dict[str, Any]:
        payload = await query_current_air_quality(lat=lat, lon=lon)
        return payload.model_dump(mode="json")

    async def get_forecast_data(self, lat: float, lon: float, hours: int = 24) -> list[dict[str, Any]]:
        payload = await query_forecast_air_quality(lat=lat, lon=lon, hours=hours)
        return [item.model_dump(mode="json") for item in payload]

    async def get_history_data(
        self,
        *,
        city_key: str = "",
        lat: float = 0.0,
        lon: float = 0.0,
        range_preset: str = "24h",
        page_size: int = 48,
        page: int = 1,
        sort: str = "desc",
    ) -> dict[str, Any]:
        payload = await query_history(
            range_value=HistoryRange(range_preset),
            page=page,
            page_size=page_size,
            sort=HistorySortOrder(sort),
            city=city_key or None,
            lat=None if city_key else lat,
            lon=None if city_key else lon,
        )
        return payload.model_dump(mode="json")

    async def get_trends_data(
        self,
        *,
        city_key: str = "",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        range_preset: str = "7d",
    ) -> dict[str, Any]:
        payload = await query_trends_v2(
            range_value=TrendRange(range_preset),
            city=city_key or None,
            lat=None if city_key else lat,
            lon=None if city_key else lon,
        )
        return payload.model_dump(mode="json")

    async def get_time_series_data(self, lat: float, lon: float, hours: int = 24) -> list[dict[str, Any]]:
        return await self.get_forecast_data(lat=lat, lon=lon, hours=hours)

    async def close(self) -> None:
        return None

    async def check_health(self) -> dict[str, Any]:
        try:
            payload = await query_health()
            data = payload.model_dump(mode="json")
            data["reachable"] = True
            return data
        except Exception:
            return {"status": "unhealthy", "reachable": False, "services": {}}

    def _get_or_create_alert_service(self) -> AlertSubscriptionService:
        global _fallback_alert_service
        runtime_service = get_alert_subscription_service()
        if runtime_service is not None:
            return runtime_service
        if _fallback_alert_service is None:
            telegram_service = get_telegram_delivery_service() or TelegramDeliveryService(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                dead_letter_sink=JsonlDeadLetterSink("logs/telegram_dead_letter.jsonl"),
            )
            _fallback_alert_service = AlertSubscriptionService(
                subscription_repository=InMemoryAlertSubscriptionRepository(),
                delivery_attempt_repository=InMemoryAlertDeliveryAttemptRepository(),
                audit_repository=InMemoryAlertAuditRepository(),
                idempotency_repository=InMemoryAlertIdempotencyRepository(),
                telegram_delivery_service=telegram_service,
            )
        return _fallback_alert_service

    async def list_alert_rules(self) -> list[dict[str, Any]]:
        service = self._get_or_create_alert_service()
        payload = await service.list_subscriptions()
        return [item.model_dump(mode="json") for item in payload]

    async def create_alert_rule(self, payload: dict[str, Any]) -> dict[str, Any]:
        service = self._get_or_create_alert_service()
        if payload.get("city") or payload.get("lat") is not None or payload.get("lon") is not None:
            created = await service.create_subscription(AlertSubscriptionCreate(**payload))
            return created.model_dump(mode="json")
        created = await service.create_legacy_rule(AlertRuleCreate(**payload))
        return created.model_dump(mode="json")

    async def update_alert_rule(self, rule_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        service = self._get_or_create_alert_service()
        if payload.get("city") or payload.get("lat") is not None or payload.get("lon") is not None:
            updated = await service.update_subscription(rule_id, AlertSubscriptionUpdate(**payload))
        else:
            updated = await service.update_legacy_rule(rule_id, AlertRuleUpdate(**payload))
        if updated is None:
            raise HTTPException(status_code=404, detail="Alert subscription not found")
        return updated.model_dump(mode="json")

    async def delete_alert_rule(self, rule_id: str) -> dict[str, Any]:
        service = self._get_or_create_alert_service()
        deleted = await service.delete_subscription(rule_id)
        if not deleted:
            deleted = await service.delete_legacy_rule(rule_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Alert subscription not found")
        return {"deleted": True, "rule_id": rule_id}

    async def get_daily_digest(
        self,
        *,
        city: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> dict[str, Any]:
        digest = await self._build_daily_digest(city=city, lat=lat, lon=lon)
        return digest.model_dump(mode="json")

    async def deliver_daily_digest(
        self,
        *,
        chat_id: str,
        city: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> dict[str, Any]:
        digest = await self._build_daily_digest(city=city, lat=lat, lon=lon)
        message = (
            f"AirTrace Daily Digest\n"
            f"Location: {digest.location_label}\n"
            f"Period: {digest.period}\n"
            f"Trend: {digest.trend}\n"
            f"Warnings: {'; '.join(digest.top_warnings)}\n"
            f"Actions: {'; '.join(digest.recommended_actions)}"
        )
        telegram_service = self._get_or_create_alert_service()._telegram_delivery_service
        return await telegram_service.send_message(chat_id=chat_id, text=message, event_id=f"digest:{digest.location_label}")

    async def _build_daily_digest(
        self,
        *,
        city: Optional[str],
        lat: Optional[float],
        lon: Optional[float],
    ) -> DailyDigestResponse:
        history_snapshot_store = get_history_snapshot_store()
        items: list[Any] = []
        if history_snapshot_store is not None:
            now = _utc_now()
            result = await history_snapshot_store.query_snapshots(
                start_utc=now - timedelta(hours=24),
                end_utc=now,
                city_code=city,
                lat=lat,
                lon=lon,
                limit=500,
                offset=0,
            )
            items = list(reversed(result["items"]))

        if not items and lat is not None and lon is not None:
            current = await query_current_air_quality(lat=lat, lon=lon)
            items = [current]

        label = city or f"{lat},{lon}"
        if not items:
            return DailyDigestResponse(
                location_label=label,
                trend="stable",
                top_warnings=["Insufficient data for a full daily digest"],
                recommended_actions=["Check history availability and try again later"],
                summary_text=f"No history data available for {label}.",
            )

        def _aqi_value(item: Any) -> int:
            if hasattr(item, "aqi") and hasattr(item.aqi, "value"):
                return item.aqi.value
            if hasattr(item, "aqi") and isinstance(item.aqi, int):
                return item.aqi
            if isinstance(item, dict):
                aqi = item.get("aqi")
                if isinstance(aqi, dict):
                    return int(aqi.get("value", 0))
                return int(aqi or 0)
            return 0

        first_aqi = _aqi_value(items[0])
        last_aqi = _aqi_value(items[-1])
        delta = last_aqi - first_aqi
        if delta >= 15:
            trend = "worsening"
        elif delta <= -15:
            trend = "improving"
        else:
            trend = "stable"

        max_aqi = max(_aqi_value(item) for item in items)
        warnings: list[str] = []
        if max_aqi >= 200:
            warnings.append("There were periods of very high pollution (AQI >= 200)")
        elif max_aqi >= 150:
            warnings.append("There were periods of high pollution (AQI >= 150)")
        if not warnings:
            warnings.append("No critical air-quality episodes were detected")

        if trend == "worsening":
            actions = [
                "Reduce long outdoor activity in the coming hours",
                "Move ventilation to lower-AQI periods",
            ]
        elif trend == "improving":
            actions = [
                "Short outdoor activities are more feasible now",
                "Sensitive groups should still keep basic precautions",
            ]
        else:
            actions = [
                "Keep standard precautions for outdoor exposure",
                "Track updates as weather conditions change",
            ]

        return DailyDigestResponse(
            location_label=label,
            trend=trend,
            top_warnings=warnings,
            recommended_actions=actions,
            summary_text=f"24h digest for {label}: trend={trend}, max_aqi={max_aqi}.",
        )
