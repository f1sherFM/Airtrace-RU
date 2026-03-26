"""Repository contracts for Stage 4 alert subscriptions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class AlertSubscriptionRecord:
    id: str
    name: str
    enabled: bool
    city_code: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    coordinate_key: Optional[str]
    aqi_threshold: Optional[int]
    nmu_levels: list[str]
    cooldown_minutes: int
    quiet_hours_start: Optional[int]
    quiet_hours_end: Optional[int]
    channel: str
    chat_id: Optional[str]
    last_triggered_at: Optional[datetime]
    last_delivery_status: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]


@dataclass(frozen=True)
class AlertDeliveryAttemptRecord:
    id: int
    subscription_id: str
    event_id: Optional[str]
    channel: str
    status: str
    attempts: int
    retry_count: int
    error: Optional[str]
    provider_response: Optional[dict[str, Any]]
    dead_lettered: bool
    created_at: datetime


@dataclass(frozen=True)
class AlertAuditEntryRecord:
    id: int
    subscription_id: Optional[str]
    action: str
    payload: dict[str, Any]
    idempotency_key: Optional[str]
    created_at: datetime


@dataclass(frozen=True)
class AlertIdempotencyRecord:
    scope: str
    idempotency_key: str
    request_fingerprint: str
    response_payload: dict[str, Any]
    created_at: datetime


class AlertSubscriptionRepository(Protocol):
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
    ) -> AlertSubscriptionRecord: ...

    async def list_subscriptions(self, *, include_deleted: bool = False) -> list[AlertSubscriptionRecord]: ...

    async def list_active_subscriptions(self) -> list[AlertSubscriptionRecord]: ...

    async def get_subscription(self, subscription_id: str) -> Optional[AlertSubscriptionRecord]: ...

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
    ) -> Optional[AlertSubscriptionRecord]: ...

    async def soft_delete_subscription(self, subscription_id: str) -> bool: ...

    async def list_applicable_subscriptions(
        self,
        *,
        city_code: Optional[str],
        latitude: float,
        longitude: float,
    ) -> list[AlertSubscriptionRecord]: ...

    async def set_delivery_state(
        self,
        subscription_id: str,
        *,
        last_triggered_at: Optional[datetime],
        last_delivery_status: Optional[str],
    ) -> Optional[AlertSubscriptionRecord]: ...


class AlertDeliveryAttemptRepository(Protocol):
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
    ) -> AlertDeliveryAttemptRecord: ...


class AlertAuditRepository(Protocol):
    async def record_event(
        self,
        *,
        subscription_id: Optional[str],
        action: str,
        payload: dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> AlertAuditEntryRecord: ...


class AlertIdempotencyRepository(Protocol):
    async def get_record(self, *, scope: str, idempotency_key: str) -> Optional[AlertIdempotencyRecord]: ...

    async def store_record(
        self,
        *,
        scope: str,
        idempotency_key: str,
        request_fingerprint: str,
        response_payload: dict[str, Any],
    ) -> AlertIdempotencyRecord: ...
