"""Stage 4 alert subscription service."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from application.services.alert_rule_engine import AlertRuleEngine
from application.repositories.alerts import (
    AlertAuditRepository,
    AlertDeliveryAttemptRepository,
    AlertIdempotencyRepository,
    AlertSubscriptionRecord,
    AlertSubscriptionRepository,
)
from core.settings import get_cities_mapping
from schemas import (
    AlertChannel,
    AlertEvent,
    AlertRule,
    AlertRuleCreate,
    AlertRuleUpdate,
    AlertSubscription,
    AlertSubscriptionCreate,
    AlertSubscriptionUpdate,
    DeliveryResult,
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coordinate_key(latitude: float, longitude: float) -> str:
    return f"{round(latitude, 4):.4f},{round(longitude, 4):.4f}"


def _request_fingerprint(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AlertSubscriptionService:
    def __init__(
        self,
        *,
        subscription_repository: AlertSubscriptionRepository,
        delivery_attempt_repository: AlertDeliveryAttemptRepository,
        audit_repository: AlertAuditRepository,
        idempotency_repository: AlertIdempotencyRepository,
        telegram_delivery_service: object,
    ):
        self._subscription_repository = subscription_repository
        self._delivery_attempt_repository = delivery_attempt_repository
        self._audit_repository = audit_repository
        self._idempotency_repository = idempotency_repository
        self._telegram_delivery_service = telegram_delivery_service

    @staticmethod
    def _normalize_city_code(city: Optional[str]) -> Optional[str]:
        if city is None:
            return None
        normalized = city.strip().lower()
        return normalized or None

    @staticmethod
    def _normalize_nmu_levels(levels: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for level in levels:
            value = (level or "").strip().lower()
            if value and value not in seen:
                seen.add(value)
                normalized.append(value)
        return normalized

    def _resolve_target(
        self,
        *,
        city: Optional[str],
        lat: Optional[float],
        lon: Optional[float],
        allow_global: bool,
    ) -> tuple[Optional[str], Optional[float], Optional[float], Optional[str]]:
        city_code = self._normalize_city_code(city)
        has_coordinates = lat is not None or lon is not None
        if city_code and has_coordinates:
            raise ValueError("Provide either city or lat/lon, not both")
        if lat is not None and lon is None or lat is None and lon is not None:
            raise ValueError("lat and lon must be provided together")
        if city_code is None and lat is None and lon is None:
            if allow_global:
                return None, None, None, None
            raise ValueError("Provide either city or lat/lon for alert subscription")
        if city_code is not None:
            return city_code, None, None, None
        assert lat is not None and lon is not None
        return None, lat, lon, _coordinate_key(lat, lon)

    def _city_code_for_coordinates(self, *, lat: float, lon: float) -> Optional[str]:
        coordinate_key = _coordinate_key(lat, lon)
        for city_code, payload in get_cities_mapping().items():
            if _coordinate_key(payload["lat"], payload["lon"]) == coordinate_key:
                return city_code
        return None

    def _record_to_public(self, record: AlertSubscriptionRecord) -> AlertSubscription:
        if record.chat_id is None:
            raise ValueError("Subscription chat_id is not configured")
        return AlertSubscription(
            id=record.id,
            name=record.name,
            enabled=record.enabled,
            city=record.city_code,
            lat=record.latitude,
            lon=record.longitude,
            aqi_threshold=record.aqi_threshold,
            nmu_levels=list(record.nmu_levels),
            cooldown_minutes=record.cooldown_minutes,
            quiet_hours_start=record.quiet_hours_start,
            quiet_hours_end=record.quiet_hours_end,
            channel=AlertChannel(record.channel),
            chat_id=record.chat_id,
            last_triggered_at=record.last_triggered_at,
            last_delivery_status=record.last_delivery_status,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def _record_to_legacy(self, record: AlertSubscriptionRecord) -> AlertRule:
        return AlertRule(
            id=record.id,
            name=record.name,
            enabled=record.enabled,
            aqi_threshold=record.aqi_threshold,
            nmu_levels=list(record.nmu_levels),
            cooldown_minutes=record.cooldown_minutes,
            quiet_hours_start=record.quiet_hours_start,
            quiet_hours_end=record.quiet_hours_end,
            channel=record.channel,
            chat_id=record.chat_id,
            created_at=record.created_at,
        )

    async def _store_idempotent_response(
        self,
        *,
        scope: str,
        idempotency_key: Optional[str],
        request_payload: dict[str, object],
        response_payload: dict[str, object],
    ) -> None:
        if not idempotency_key:
            return
        await self._idempotency_repository.store_record(
            scope=scope,
            idempotency_key=idempotency_key,
            request_fingerprint=_request_fingerprint(request_payload),
            response_payload=response_payload,
        )

    async def _reuse_idempotent_response(
        self,
        *,
        scope: str,
        idempotency_key: Optional[str],
        request_payload: dict[str, object],
    ) -> Optional[dict[str, object]]:
        if not idempotency_key:
            return None
        record = await self._idempotency_repository.get_record(scope=scope, idempotency_key=idempotency_key)
        if record is None:
            return None
        if record.request_fingerprint != _request_fingerprint(request_payload):
            raise ValueError("Idempotency-Key was already used with a different request payload")
        return record.response_payload

    async def create_subscription(
        self,
        payload: AlertSubscriptionCreate,
        *,
        idempotency_key: Optional[str] = None,
    ) -> AlertSubscription:
        request_payload = payload.model_dump(mode="json")
        reused = await self._reuse_idempotent_response(
            scope="alerts:create",
            idempotency_key=idempotency_key,
            request_payload=request_payload,
        )
        if reused is not None:
            return AlertSubscription(**reused)

        city_code, latitude, longitude, coordinate_key = self._resolve_target(
            city=payload.city,
            lat=payload.lat,
            lon=payload.lon,
            allow_global=False,
        )
        record = await self._subscription_repository.create_subscription(
            subscription_id=str(uuid4()),
            name=payload.name,
            enabled=payload.enabled,
            city_code=city_code,
            latitude=latitude,
            longitude=longitude,
            coordinate_key=coordinate_key,
            aqi_threshold=payload.aqi_threshold,
            nmu_levels=self._normalize_nmu_levels(payload.nmu_levels),
            cooldown_minutes=payload.cooldown_minutes,
            quiet_hours_start=payload.quiet_hours_start,
            quiet_hours_end=payload.quiet_hours_end,
            channel=payload.channel.value,
            chat_id=payload.chat_id,
        )
        response = self._record_to_public(record)
        response_payload = response.model_dump(mode="json")
        await self._audit_repository.record_event(
            subscription_id=record.id,
            action="subscription_created",
            payload=response_payload,
            idempotency_key=idempotency_key,
        )
        await self._store_idempotent_response(
            scope="alerts:create",
            idempotency_key=idempotency_key,
            request_payload=request_payload,
            response_payload=response_payload,
        )
        return response

    async def list_subscriptions(self) -> list[AlertSubscription]:
        records = await self._subscription_repository.list_subscriptions(include_deleted=False)
        return [self._record_to_public(record) for record in records if record.chat_id is not None]

    async def list_active_subscription_records(self) -> list[AlertSubscriptionRecord]:
        return await self._subscription_repository.list_active_subscriptions()

    async def get_subscription(self, subscription_id: str) -> Optional[AlertSubscription]:
        record = await self._subscription_repository.get_subscription(subscription_id)
        if record is None or record.deleted_at is not None or record.chat_id is None:
            return None
        return self._record_to_public(record)

    async def update_subscription(
        self,
        subscription_id: str,
        payload: AlertSubscriptionUpdate,
        *,
        idempotency_key: Optional[str] = None,
    ) -> Optional[AlertSubscription]:
        existing = await self._subscription_repository.get_subscription(subscription_id)
        if existing is None or existing.deleted_at is not None:
            return None

        request_payload = payload.model_dump(mode="json", exclude_unset=True)
        reused = await self._reuse_idempotent_response(
            scope=f"alerts:update:{subscription_id}",
            idempotency_key=idempotency_key,
            request_payload=request_payload,
        )
        if reused is not None:
            return AlertSubscription(**reused)

        city = existing.city_code
        latitude = existing.latitude
        longitude = existing.longitude
        if "city" in payload.__pydantic_fields_set__:
            city = payload.city
            latitude = None
            longitude = None
        elif "lat" in payload.__pydantic_fields_set__ or "lon" in payload.__pydantic_fields_set__:
            city = None
            latitude = payload.lat
            longitude = payload.lon
        city_code, normalized_lat, normalized_lon, coordinate_key = self._resolve_target(
            city=city,
            lat=latitude,
            lon=longitude,
            allow_global=False,
        )

        merged_aqi_threshold = payload.aqi_threshold if "aqi_threshold" in payload.__pydantic_fields_set__ else existing.aqi_threshold
        merged_nmu_levels = (
            self._normalize_nmu_levels(payload.nmu_levels or [])
            if "nmu_levels" in payload.__pydantic_fields_set__
            else list(existing.nmu_levels)
        )
        if merged_aqi_threshold is None and not merged_nmu_levels:
            raise ValueError("At least one trigger is required: aqi_threshold or nmu_levels")

        merged_chat_id = payload.chat_id if "chat_id" in payload.__pydantic_fields_set__ else existing.chat_id
        if not merged_chat_id:
            raise ValueError("Telegram subscriptions require chat_id")

        updated = await self._subscription_repository.update_subscription(
            subscription_id,
            name=payload.name if payload.name is not None else existing.name,
            enabled=payload.enabled if payload.enabled is not None else existing.enabled,
            city_code=city_code,
            latitude=normalized_lat,
            longitude=normalized_lon,
            coordinate_key=coordinate_key,
            aqi_threshold=merged_aqi_threshold,
            nmu_levels=merged_nmu_levels,
            cooldown_minutes=payload.cooldown_minutes if payload.cooldown_minutes is not None else existing.cooldown_minutes,
            quiet_hours_start=payload.quiet_hours_start if "quiet_hours_start" in payload.__pydantic_fields_set__ else existing.quiet_hours_start,
            quiet_hours_end=payload.quiet_hours_end if "quiet_hours_end" in payload.__pydantic_fields_set__ else existing.quiet_hours_end,
            channel=(payload.channel.value if payload.channel is not None else existing.channel),
            chat_id=merged_chat_id,
            last_triggered_at=existing.last_triggered_at,
            last_delivery_status=existing.last_delivery_status,
        )
        if updated is None:
            return None

        response = self._record_to_public(updated)
        response_payload = response.model_dump(mode="json")
        await self._audit_repository.record_event(
            subscription_id=subscription_id,
            action="subscription_updated",
            payload={
                "before": self._record_to_public(existing).model_dump(mode="json") if existing.chat_id else {},
                "after": response_payload,
            },
            idempotency_key=idempotency_key,
        )
        await self._store_idempotent_response(
            scope=f"alerts:update:{subscription_id}",
            idempotency_key=idempotency_key,
            request_payload=request_payload,
            response_payload=response_payload,
        )
        return response

    async def delete_subscription(self, subscription_id: str) -> bool:
        existing = await self._subscription_repository.get_subscription(subscription_id)
        deleted = await self._subscription_repository.soft_delete_subscription(subscription_id)
        if deleted:
            await self._audit_repository.record_event(
                subscription_id=subscription_id,
                action="subscription_deleted",
                payload=self._record_to_public(existing).model_dump(mode="json") if existing and existing.chat_id else {"id": subscription_id},
            )
        return deleted

    async def create_legacy_rule(self, payload: AlertRuleCreate) -> AlertRule:
        normalized_levels = self._normalize_nmu_levels(payload.nmu_levels)
        if payload.aqi_threshold is None and not normalized_levels:
            raise ValueError("At least one trigger is required: aqi_threshold or nmu_levels")
        record = await self._subscription_repository.create_subscription(
            subscription_id=str(uuid4()),
            name=payload.name,
            enabled=payload.enabled,
            city_code=None,
            latitude=None,
            longitude=None,
            coordinate_key=None,
            aqi_threshold=payload.aqi_threshold,
            nmu_levels=normalized_levels,
            cooldown_minutes=payload.cooldown_minutes,
            quiet_hours_start=payload.quiet_hours_start,
            quiet_hours_end=payload.quiet_hours_end,
            channel=payload.channel,
            chat_id=payload.chat_id,
        )
        await self._audit_repository.record_event(
            subscription_id=record.id,
            action="legacy_rule_created",
            payload=self._record_to_legacy(record).model_dump(mode="json"),
        )
        return self._record_to_legacy(record)

    async def list_legacy_rules(self) -> list[AlertRule]:
        records = await self._subscription_repository.list_subscriptions(include_deleted=False)
        return [
            self._record_to_legacy(record)
            for record in records
            if record.city_code is None and record.coordinate_key is None
        ]

    async def update_legacy_rule(self, rule_id: str, payload: AlertRuleUpdate) -> Optional[AlertRule]:
        existing = await self._subscription_repository.get_subscription(rule_id)
        if existing is None or existing.deleted_at is not None:
            return None
        if existing.city_code is not None or existing.coordinate_key is not None:
            return None

        merged_aqi_threshold = payload.aqi_threshold if "aqi_threshold" in payload.__pydantic_fields_set__ else existing.aqi_threshold
        merged_nmu_levels = (
            self._normalize_nmu_levels(payload.nmu_levels or [])
            if "nmu_levels" in payload.__pydantic_fields_set__
            else list(existing.nmu_levels)
        )
        if merged_aqi_threshold is None and not merged_nmu_levels:
            raise ValueError("At least one trigger is required: aqi_threshold or nmu_levels")

        updated = await self._subscription_repository.update_subscription(
            rule_id,
            name=payload.name if payload.name is not None else existing.name,
            enabled=payload.enabled if payload.enabled is not None else existing.enabled,
            city_code=None,
            latitude=None,
            longitude=None,
            coordinate_key=None,
            aqi_threshold=merged_aqi_threshold,
            nmu_levels=merged_nmu_levels,
            cooldown_minutes=payload.cooldown_minutes if payload.cooldown_minutes is not None else existing.cooldown_minutes,
            quiet_hours_start=payload.quiet_hours_start if "quiet_hours_start" in payload.__pydantic_fields_set__ else existing.quiet_hours_start,
            quiet_hours_end=payload.quiet_hours_end if "quiet_hours_end" in payload.__pydantic_fields_set__ else existing.quiet_hours_end,
            channel=payload.channel if payload.channel is not None else existing.channel,
            chat_id=payload.chat_id if "chat_id" in payload.__pydantic_fields_set__ else existing.chat_id,
            last_triggered_at=existing.last_triggered_at,
            last_delivery_status=existing.last_delivery_status,
        )
        if updated is None:
            return None
        await self._audit_repository.record_event(
            subscription_id=rule_id,
            action="legacy_rule_updated",
            payload=self._record_to_legacy(updated).model_dump(mode="json"),
        )
        return self._record_to_legacy(updated)

    async def delete_legacy_rule(self, rule_id: str) -> bool:
        existing = await self._subscription_repository.get_subscription(rule_id)
        if existing is None or existing.city_code is not None or existing.coordinate_key is not None:
            return False
        deleted = await self._subscription_repository.soft_delete_subscription(rule_id)
        if deleted:
            await self._audit_repository.record_event(
                subscription_id=rule_id,
                action="legacy_rule_deleted",
                payload=self._record_to_legacy(existing).model_dump(mode="json"),
            )
        return deleted

    async def get_legacy_rule(self, rule_id: str) -> Optional[AlertRule]:
        record = await self._subscription_repository.get_subscription(rule_id)
        if record is None or record.deleted_at is not None:
            return None
        if record.city_code is not None or record.coordinate_key is not None:
            return None
        return self._record_to_legacy(record)

    async def evaluate_conditions(
        self,
        *,
        aqi: int,
        nmu_risk: Optional[str],
        lat: float,
        lon: float,
        city_code: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> list[AlertEvent]:
        now_utc = (now or _utc_now()).astimezone(timezone.utc)
        resolved_city_code = self._normalize_city_code(city_code) or self._city_code_for_coordinates(lat=lat, lon=lon)
        subscriptions = await self._subscription_repository.list_applicable_subscriptions(
            city_code=resolved_city_code,
            latitude=lat,
            longitude=lon,
        )
        normalized_nmu = (nmu_risk or "").lower()
        events: list[AlertEvent] = []

        for subscription in subscriptions:
            reasons: list[str] = []
            if subscription.aqi_threshold is not None and aqi >= subscription.aqi_threshold:
                reasons.append(f"aqi>={subscription.aqi_threshold}")
            if subscription.nmu_levels and normalized_nmu in {value.lower() for value in subscription.nmu_levels}:
                reasons.append(f"nmu={normalized_nmu}")
            if not reasons:
                continue

            severity = AlertRuleEngine._severity(aqi, normalized_nmu)
            if AlertRuleEngine._is_in_quiet_hours(now_utc, subscription.quiet_hours_start, subscription.quiet_hours_end):
                logger.info(
                    "Alert suppressed by quiet hours: subscription_id=%s name=%s reasons=%s",
                    subscription.id,
                    subscription.name,
                    reasons,
                )
                events.append(
                    AlertEvent(
                        rule_id=subscription.id,
                        rule_name=subscription.name,
                        severity=severity,
                        reasons=reasons + ["quiet_hours"],
                        suppressed=True,
                    )
                )
                continue

            cooldown_until = (
                subscription.last_triggered_at + timedelta(minutes=subscription.cooldown_minutes)
                if subscription.last_triggered_at is not None
                else None
            )
            if cooldown_until is not None and now_utc < cooldown_until:
                logger.info(
                    "Alert suppressed by cooldown: subscription_id=%s name=%s cooldown_until=%s reasons=%s",
                    subscription.id,
                    subscription.name,
                    cooldown_until.isoformat(),
                    reasons,
                )
                events.append(
                    AlertEvent(
                        rule_id=subscription.id,
                        rule_name=subscription.name,
                        severity=severity,
                        reasons=reasons + ["cooldown"],
                        suppressed=True,
                    )
                )
                continue

            await self._subscription_repository.set_delivery_state(
                subscription.id,
                last_triggered_at=now_utc,
                last_delivery_status=subscription.last_delivery_status,
            )
            logger.info(
                "Alert triggered: subscription_id=%s name=%s aqi=%s nmu=%s severity=%s reasons=%s",
                subscription.id,
                subscription.name,
                aqi,
                normalized_nmu or "unknown",
                severity,
                reasons,
            )
            events.append(
                AlertEvent(
                    rule_id=subscription.id,
                    rule_name=subscription.name,
                    triggered_at=now_utc,
                    severity=severity,
                    reasons=reasons,
                    suppressed=False,
                )
            )

        return events

    async def deliver_events(
        self,
        *,
        events: list[AlertEvent],
        aqi: int,
        nmu_risk: Optional[str],
        chat_id_override: Optional[str] = None,
    ) -> list[DeliveryResult]:
        delivered: list[DeliveryResult] = []
        for index, event in enumerate(events, start=1):
            if event.suppressed:
                continue
            subscription = await self._subscription_repository.get_subscription(event.rule_id)
            if subscription is None or subscription.deleted_at is not None:
                continue
            chat_id = chat_id_override or subscription.chat_id
            if not chat_id:
                continue
            text = (
                f"AirTrace Alert #{index}\n"
                f"Rule: {event.rule_name}\n"
                f"AQI: {aqi}\n"
                f"NMU: {nmu_risk}\n"
                f"Severity: {event.severity}\n"
                f"Reasons: {', '.join(event.reasons)}"
            )
            event_id = f"{event.rule_id}:{int(event.triggered_at.timestamp())}"
            raw_result = await self._telegram_delivery_service.send_message(
                chat_id=chat_id,
                text=text,
                event_id=event_id,
            )
            delivery_result = DeliveryResult(**raw_result)
            logger.info(
                "Alert delivery result: subscription_id=%s chat_id=%s status=%s attempts=%s event_id=%s error=%s",
                event.rule_id,
                chat_id,
                delivery_result.status,
                delivery_result.attempts,
                delivery_result.event_id,
                delivery_result.error,
            )
            await self._delivery_attempt_repository.record_attempt(
                subscription_id=event.rule_id,
                event_id=delivery_result.event_id,
                channel=delivery_result.channel,
                status=delivery_result.status,
                attempts=delivery_result.attempts,
                retry_count=max(0, delivery_result.attempts - 1),
                error=delivery_result.error,
                provider_response=raw_result,
                dead_lettered=delivery_result.status == "failed",
            )
            await self._subscription_repository.set_delivery_state(
                event.rule_id,
                last_triggered_at=event.triggered_at,
                last_delivery_status=delivery_result.status,
            )
            await self._audit_repository.record_event(
                subscription_id=event.rule_id,
                action="delivery_attempted",
                payload=delivery_result.model_dump(mode="json"),
            )
            delivered.append(delivery_result)
        return delivered
