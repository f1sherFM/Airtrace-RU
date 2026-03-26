"""In-memory repositories for Stage 4 alert subscriptions."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Optional

from application.repositories.alerts import (
    AlertAuditEntryRecord,
    AlertAuditRepository,
    AlertDeliveryAttemptRecord,
    AlertDeliveryAttemptRepository,
    AlertIdempotencyRecord,
    AlertIdempotencyRepository,
    AlertSubscriptionRecord,
    AlertSubscriptionRepository,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coordinate_key(latitude: float, longitude: float) -> str:
    return f"{round(latitude, 4):.4f},{round(longitude, 4):.4f}"


class InMemoryAlertSubscriptionRepository(AlertSubscriptionRepository):
    def __init__(self):
        self._subscriptions: dict[str, AlertSubscriptionRecord] = {}

    async def create_subscription(
        self,
        *,
        subscription_id: str,
        name: str,
        enabled: bool,
        city_code: Optional[str],
        latitude: Optional[float],
        longitude: Optional[float],
        coordinate_key: Optional[str],
        aqi_threshold: Optional[int],
        nmu_levels: list[str],
        cooldown_minutes: int,
        quiet_hours_start: Optional[int],
        quiet_hours_end: Optional[int],
        channel: str,
        chat_id: Optional[str],
    ) -> AlertSubscriptionRecord:
        now = _utc_now()
        record = AlertSubscriptionRecord(
            id=subscription_id,
            name=name,
            enabled=enabled,
            city_code=city_code,
            latitude=latitude,
            longitude=longitude,
            coordinate_key=coordinate_key,
            aqi_threshold=aqi_threshold,
            nmu_levels=list(nmu_levels),
            cooldown_minutes=cooldown_minutes,
            quiet_hours_start=quiet_hours_start,
            quiet_hours_end=quiet_hours_end,
            channel=channel,
            chat_id=chat_id,
            last_triggered_at=None,
            last_delivery_status=None,
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )
        self._subscriptions[subscription_id] = record
        return record

    async def list_subscriptions(self, *, include_deleted: bool = False) -> list[AlertSubscriptionRecord]:
        items = sorted(self._subscriptions.values(), key=lambda record: record.created_at, reverse=True)
        if include_deleted:
            return items
        return [record for record in items if record.deleted_at is None]

    async def list_active_subscriptions(self) -> list[AlertSubscriptionRecord]:
        return [
            record
            for record in await self.list_subscriptions(include_deleted=False)
            if record.enabled
        ]

    async def get_subscription(self, subscription_id: str) -> Optional[AlertSubscriptionRecord]:
        return self._subscriptions.get(subscription_id)

    async def update_subscription(
        self,
        subscription_id: str,
        *,
        name: str,
        enabled: bool,
        city_code: Optional[str],
        latitude: Optional[float],
        longitude: Optional[float],
        coordinate_key: Optional[str],
        aqi_threshold: Optional[int],
        nmu_levels: list[str],
        cooldown_minutes: int,
        quiet_hours_start: Optional[int],
        quiet_hours_end: Optional[int],
        channel: str,
        chat_id: Optional[str],
        last_triggered_at: Optional[datetime],
        last_delivery_status: Optional[str],
    ) -> Optional[AlertSubscriptionRecord]:
        existing = self._subscriptions.get(subscription_id)
        if existing is None or existing.deleted_at is not None:
            return None
        updated = replace(
            existing,
            name=name,
            enabled=enabled,
            city_code=city_code,
            latitude=latitude,
            longitude=longitude,
            coordinate_key=coordinate_key,
            aqi_threshold=aqi_threshold,
            nmu_levels=list(nmu_levels),
            cooldown_minutes=cooldown_minutes,
            quiet_hours_start=quiet_hours_start,
            quiet_hours_end=quiet_hours_end,
            channel=channel,
            chat_id=chat_id,
            last_triggered_at=last_triggered_at,
            last_delivery_status=last_delivery_status,
            updated_at=_utc_now(),
        )
        self._subscriptions[subscription_id] = updated
        return updated

    async def soft_delete_subscription(self, subscription_id: str) -> bool:
        existing = self._subscriptions.get(subscription_id)
        if existing is None or existing.deleted_at is not None:
            return False
        self._subscriptions[subscription_id] = replace(
            existing,
            deleted_at=_utc_now(),
            updated_at=_utc_now(),
        )
        return True

    async def list_applicable_subscriptions(
        self,
        *,
        city_code: Optional[str],
        latitude: float,
        longitude: float,
    ) -> list[AlertSubscriptionRecord]:
        coordinate_key = _coordinate_key(latitude, longitude)
        matches: list[AlertSubscriptionRecord] = []
        for record in await self.list_subscriptions(include_deleted=False):
            if not record.enabled:
                continue
            if record.city_code is None and record.coordinate_key is None:
                matches.append(record)
                continue
            if record.coordinate_key == coordinate_key:
                matches.append(record)
                continue
            if city_code is not None and record.city_code == city_code.lower():
                matches.append(record)
        return matches

    async def set_delivery_state(
        self,
        subscription_id: str,
        *,
        last_triggered_at: Optional[datetime],
        last_delivery_status: Optional[str],
    ) -> Optional[AlertSubscriptionRecord]:
        existing = self._subscriptions.get(subscription_id)
        if existing is None or existing.deleted_at is not None:
            return None
        updated = replace(
            existing,
            last_triggered_at=last_triggered_at,
            last_delivery_status=last_delivery_status,
            updated_at=_utc_now(),
        )
        self._subscriptions[subscription_id] = updated
        return updated


class InMemoryAlertDeliveryAttemptRepository(AlertDeliveryAttemptRepository):
    def __init__(self):
        self._attempts: list[AlertDeliveryAttemptRecord] = []

    async def record_attempt(
        self,
        *,
        subscription_id: str,
        event_id: Optional[str],
        channel: str,
        status: str,
        attempts: int,
        retry_count: int,
        error: Optional[str],
        provider_response: Optional[dict[str, Any]],
        dead_lettered: bool,
    ) -> AlertDeliveryAttemptRecord:
        record = AlertDeliveryAttemptRecord(
            id=len(self._attempts) + 1,
            subscription_id=subscription_id,
            event_id=event_id,
            channel=channel,
            status=status,
            attempts=attempts,
            retry_count=retry_count,
            error=error,
            provider_response=provider_response,
            dead_lettered=dead_lettered,
            created_at=_utc_now(),
        )
        self._attempts.append(record)
        return record


class InMemoryAlertAuditRepository(AlertAuditRepository):
    def __init__(self):
        self._entries: list[AlertAuditEntryRecord] = []

    async def record_event(
        self,
        *,
        subscription_id: Optional[str],
        action: str,
        payload: dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> AlertAuditEntryRecord:
        record = AlertAuditEntryRecord(
            id=len(self._entries) + 1,
            subscription_id=subscription_id,
            action=action,
            payload=dict(payload),
            idempotency_key=idempotency_key,
            created_at=_utc_now(),
        )
        self._entries.append(record)
        return record


class InMemoryAlertIdempotencyRepository(AlertIdempotencyRepository):
    def __init__(self):
        self._records: dict[tuple[str, str], AlertIdempotencyRecord] = {}

    async def get_record(self, *, scope: str, idempotency_key: str) -> Optional[AlertIdempotencyRecord]:
        return self._records.get((scope, idempotency_key))

    async def store_record(
        self,
        *,
        scope: str,
        idempotency_key: str,
        request_fingerprint: str,
        response_payload: dict[str, Any],
    ) -> AlertIdempotencyRecord:
        record = AlertIdempotencyRecord(
            scope=scope,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            response_payload=dict(response_payload),
            created_at=_utc_now(),
        )
        self._records[(scope, idempotency_key)] = record
        return record
