"""Direct application-layer service used by the Python SSR web app."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
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
from http_transport import create_internal_async_client
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

    def __init__(
        self,
        *,
        alerts_api_base_url: Optional[str] = None,
        alerts_api_key: Optional[str] = None,
        alerts_api_timeout_seconds: Optional[float] = None,
        alerts_api_trust_env: Optional[bool] = None,
        alerts_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._alerts_api_base_url_override = alerts_api_base_url
        self._alerts_api_key_override = alerts_api_key
        timeout_raw = alerts_api_timeout_seconds
        if timeout_raw is None:
            timeout_raw = float(os.getenv("WEB_ALERTS_API_TIMEOUT_SECONDS", "10"))
        self._alerts_api_timeout_seconds = timeout_raw
        trust_env_raw = alerts_api_trust_env
        if trust_env_raw is None:
            trust_env_raw = os.getenv("WEB_ALERTS_API_TRUST_ENV", "false").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self._alerts_api_trust_env = trust_env_raw
        self._alerts_transport = alerts_transport
        self._alerts_backend_enabled = os.getenv("WEB_ALERTS_USE_BACKEND_API", "true").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

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

    def _use_backend_alerts_api(self) -> bool:
        return self._alerts_backend_enabled and bool(self._alerts_api_base_url) and bool(self._alerts_api_key)

    @property
    def _alerts_api_base_url(self) -> str:
        return (
            self._alerts_api_base_url_override
            or os.getenv("API_BASE_URL", "").strip()
            or os.getenv("WEB_API_BASE_URL", "").strip()
        ).rstrip("/")

    @property
    def _alerts_api_key(self) -> str:
        return (self._alerts_api_key_override or os.getenv("ALERTS_API_KEY", "")).strip()

    def _alerts_api_headers(self) -> dict[str, str]:
        return {"X-API-Key": self._alerts_api_key}

    async def _request_alerts_api(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
    ) -> Any:
        async with create_internal_async_client(
            base_url=self._alerts_api_base_url,
            timeout_seconds=self._alerts_api_timeout_seconds,
            transport=self._alerts_transport,
            trust_env=self._alerts_api_trust_env,
            max_connections=10,
            max_keepalive_connections=5,
        ) as client:
            try:
                response = await client.request(
                    method,
                    path,
                    headers=self._alerts_api_headers(),
                    json=json_body,
                )
            except httpx.RequestError as exc:
                raise HTTPException(status_code=503, detail=str(exc) or "Alert backend is unreachable") from exc
        if response.is_success:
            if response.status_code == 204 or not response.content:
                return None
            return response.json()

        message = response.text
        try:
            payload = response.json()
            message = payload.get("message") or payload.get("detail") or message
        except ValueError:
            payload = None
        raise HTTPException(status_code=response.status_code, detail=message or "Alert backend request failed")

    async def list_alert_rules(self) -> list[dict[str, Any]]:
        if self._use_backend_alerts_api():
            payload = await self._request_alerts_api("GET", "/v2/alerts")
            return list(payload or [])
        service = self._get_or_create_alert_service()
        payload = await service.list_subscriptions()
        return [item.model_dump(mode="json") for item in payload]

    async def create_alert_rule(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._use_backend_alerts_api():
            created = await self._request_alerts_api("POST", "/v2/alerts", json_body=payload)
            return dict(created)
        service = self._get_or_create_alert_service()
        if payload.get("city") or payload.get("lat") is not None or payload.get("lon") is not None:
            created = await service.create_subscription(AlertSubscriptionCreate(**payload))
            return created.model_dump(mode="json")
        created = await service.create_legacy_rule(AlertRuleCreate(**payload))
        return created.model_dump(mode="json")

    async def update_alert_rule(self, rule_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._use_backend_alerts_api():
            updated = await self._request_alerts_api("PATCH", f"/v2/alerts/{rule_id}", json_body=payload)
            return dict(updated)
        service = self._get_or_create_alert_service()
        if payload.get("city") or payload.get("lat") is not None or payload.get("lon") is not None:
            updated = await service.update_subscription(rule_id, AlertSubscriptionUpdate(**payload))
        else:
            updated = await service.update_legacy_rule(rule_id, AlertRuleUpdate(**payload))
        if updated is None:
            raise HTTPException(status_code=404, detail="Alert subscription not found")
        return updated.model_dump(mode="json")

    async def delete_alert_rule(self, rule_id: str) -> dict[str, Any]:
        if self._use_backend_alerts_api():
            deleted = await self._request_alerts_api("DELETE", f"/v2/alerts/{rule_id}")
            return dict(deleted or {"deleted": True, "id": rule_id})
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
                top_warnings=["Недостаточно данных для полного дайджеста за сутки"],
                recommended_actions=["Проверьте доступность истории и повторите позже"],
                summary_text=f"Для {label} пока нет достаточной истории.",
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
            warnings.append("Были периоды очень высокого загрязнения (AQI >= 200)")
        elif max_aqi >= 150:
            warnings.append("Были периоды высокого загрязнения (AQI >= 150)")
        if not warnings:
            warnings.append("Критических эпизодов загрязнения не обнаружено")

        if trend == "worsening":
            actions = [
                "Сократите длительную активность на улице в ближайшие часы",
                "Проветривайте помещение в периоды с более низким AQI",
            ]
        elif trend == "improving":
            actions = [
                "Короткие выходы на улицу сейчас допустимее",
                "Чувствительным группам всё равно стоит соблюдать базовые меры предосторожности",
            ]
        else:
            actions = [
                "Соблюдайте стандартные меры предосторожности при выходе на улицу",
                "Следите за обновлениями по мере изменения погодных условий",
            ]

        return DailyDigestResponse(
            location_label=label,
            trend=trend,
            top_warnings=warnings,
            recommended_actions=actions,
            summary_text=f"Дайджест за 24 часа для {label}: тренд — {trend}, максимальный AQI — {max_aqi}.",
        )
