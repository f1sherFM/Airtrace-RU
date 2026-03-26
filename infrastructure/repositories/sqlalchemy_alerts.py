"""SQLAlchemy repositories for Stage 4 alert subscriptions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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
from infrastructure.db.models import (
    AlertAuditLogModel,
    AlertDeliveryAttemptModel,
    AlertIdempotencyKeyModel,
    AlertSubscriptionModel,
)


def _coordinate_key(latitude: float, longitude: float) -> str:
    return f"{round(latitude, 4):.4f},{round(longitude, 4):.4f}"


def _to_subscription_record(model: AlertSubscriptionModel) -> AlertSubscriptionRecord:
    return AlertSubscriptionRecord(
        id=model.id,
        name=model.name,
        enabled=model.enabled,
        city_code=model.city_code,
        latitude=model.latitude,
        longitude=model.longitude,
        coordinate_key=model.coordinate_key,
        aqi_threshold=model.aqi_threshold,
        nmu_levels=list(model.nmu_levels or []),
        cooldown_minutes=model.cooldown_minutes,
        quiet_hours_start=model.quiet_hours_start,
        quiet_hours_end=model.quiet_hours_end,
        channel=model.channel,
        chat_id=model.chat_id,
        last_triggered_at=model.last_triggered_at,
        last_delivery_status=model.last_delivery_status,
        created_at=model.created_at,
        updated_at=model.updated_at,
        deleted_at=model.deleted_at,
    )


def _to_delivery_record(model: AlertDeliveryAttemptModel) -> AlertDeliveryAttemptRecord:
    return AlertDeliveryAttemptRecord(
        id=model.id,
        subscription_id=model.subscription_id,
        event_id=model.event_id,
        channel=model.channel,
        status=model.status,
        attempts=model.attempts,
        retry_count=model.retry_count,
        error=model.error,
        provider_response=model.provider_response,
        dead_lettered=model.dead_lettered,
        created_at=model.created_at,
    )


def _to_audit_record(model: AlertAuditLogModel) -> AlertAuditEntryRecord:
    return AlertAuditEntryRecord(
        id=model.id,
        subscription_id=model.subscription_id,
        action=model.action,
        payload=dict(model.payload or {}),
        idempotency_key=model.idempotency_key,
        created_at=model.created_at,
    )


def _to_idempotency_record(model: AlertIdempotencyKeyModel) -> AlertIdempotencyRecord:
    return AlertIdempotencyRecord(
        scope=model.scope,
        idempotency_key=model.idempotency_key,
        request_fingerprint=model.request_fingerprint,
        response_payload=dict(model.response_payload or {}),
        created_at=model.created_at,
    )


class _SQLAlchemyAlertRepositoryBase:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    @asynccontextmanager
    async def _session(self):
        async with self._session_factory() as session:
            yield session


class SQLAlchemyAlertSubscriptionRepository(_SQLAlchemyAlertRepositoryBase, AlertSubscriptionRepository):
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
        async with self._session() as session:
            model = AlertSubscriptionModel(
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
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return _to_subscription_record(model)

    async def list_subscriptions(self, *, include_deleted: bool = False) -> list[AlertSubscriptionRecord]:
        async with self._session() as session:
            statement = select(AlertSubscriptionModel)
            if not include_deleted:
                statement = statement.where(AlertSubscriptionModel.deleted_at.is_(None))
            statement = statement.order_by(AlertSubscriptionModel.created_at.desc())
            result = await session.execute(statement)
            return [_to_subscription_record(model) for model in result.scalars().all()]

    async def list_active_subscriptions(self) -> list[AlertSubscriptionRecord]:
        async with self._session() as session:
            result = await session.execute(
                select(AlertSubscriptionModel).where(
                    AlertSubscriptionModel.deleted_at.is_(None),
                    AlertSubscriptionModel.enabled.is_(True),
                ).order_by(AlertSubscriptionModel.created_at.desc())
            )
            return [_to_subscription_record(model) for model in result.scalars().all()]

    async def get_subscription(self, subscription_id: str) -> Optional[AlertSubscriptionRecord]:
        async with self._session() as session:
            result = await session.execute(select(AlertSubscriptionModel).where(AlertSubscriptionModel.id == subscription_id))
            model = result.scalar_one_or_none()
            return _to_subscription_record(model) if model is not None else None

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
        async with self._session() as session:
            result = await session.execute(
                select(AlertSubscriptionModel).where(
                    AlertSubscriptionModel.id == subscription_id,
                    AlertSubscriptionModel.deleted_at.is_(None),
                )
            )
            model = result.scalar_one_or_none()
            if model is None:
                return None
            model.name = name
            model.enabled = enabled
            model.city_code = city_code
            model.latitude = latitude
            model.longitude = longitude
            model.coordinate_key = coordinate_key
            model.aqi_threshold = aqi_threshold
            model.nmu_levels = list(nmu_levels)
            model.cooldown_minutes = cooldown_minutes
            model.quiet_hours_start = quiet_hours_start
            model.quiet_hours_end = quiet_hours_end
            model.channel = channel
            model.chat_id = chat_id
            model.last_triggered_at = last_triggered_at
            model.last_delivery_status = last_delivery_status
            model.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(model)
            return _to_subscription_record(model)

    async def soft_delete_subscription(self, subscription_id: str) -> bool:
        async with self._session() as session:
            result = await session.execute(
                select(AlertSubscriptionModel).where(
                    AlertSubscriptionModel.id == subscription_id,
                    AlertSubscriptionModel.deleted_at.is_(None),
                )
            )
            model = result.scalar_one_or_none()
            if model is None:
                return False
            model.deleted_at = datetime.now(timezone.utc)
            model.updated_at = model.deleted_at
            await session.commit()
            return True

    async def list_applicable_subscriptions(
        self,
        *,
        city_code: Optional[str],
        latitude: float,
        longitude: float,
    ) -> list[AlertSubscriptionRecord]:
        coordinate_key = _coordinate_key(latitude, longitude)
        predicates = [
            and_(AlertSubscriptionModel.city_code.is_(None), AlertSubscriptionModel.coordinate_key.is_(None)),
            AlertSubscriptionModel.coordinate_key == coordinate_key,
        ]
        if city_code is not None:
            predicates.append(AlertSubscriptionModel.city_code == city_code.lower())
        async with self._session() as session:
            result = await session.execute(
                select(AlertSubscriptionModel).where(
                    AlertSubscriptionModel.deleted_at.is_(None),
                    AlertSubscriptionModel.enabled.is_(True),
                    or_(*predicates),
                ).order_by(AlertSubscriptionModel.created_at.desc())
            )
            return [_to_subscription_record(model) for model in result.scalars().all()]

    async def set_delivery_state(
        self,
        subscription_id: str,
        *,
        last_triggered_at: Optional[datetime],
        last_delivery_status: Optional[str],
    ) -> Optional[AlertSubscriptionRecord]:
        async with self._session() as session:
            result = await session.execute(
                select(AlertSubscriptionModel).where(
                    AlertSubscriptionModel.id == subscription_id,
                    AlertSubscriptionModel.deleted_at.is_(None),
                )
            )
            model = result.scalar_one_or_none()
            if model is None:
                return None
            model.last_triggered_at = last_triggered_at
            model.last_delivery_status = last_delivery_status
            model.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(model)
            return _to_subscription_record(model)


class SQLAlchemyAlertDeliveryAttemptRepository(_SQLAlchemyAlertRepositoryBase, AlertDeliveryAttemptRepository):
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
        async with self._session() as session:
            model = AlertDeliveryAttemptModel(
                subscription_id=subscription_id,
                event_id=event_id,
                channel=channel,
                status=status,
                attempts=attempts,
                retry_count=retry_count,
                error=error,
                provider_response=provider_response,
                dead_lettered=dead_lettered,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return _to_delivery_record(model)


class SQLAlchemyAlertAuditRepository(_SQLAlchemyAlertRepositoryBase, AlertAuditRepository):
    async def record_event(
        self,
        *,
        subscription_id: Optional[str],
        action: str,
        payload: dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> AlertAuditEntryRecord:
        async with self._session() as session:
            model = AlertAuditLogModel(
                subscription_id=subscription_id,
                action=action,
                payload=payload,
                idempotency_key=idempotency_key,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return _to_audit_record(model)


class SQLAlchemyAlertIdempotencyRepository(_SQLAlchemyAlertRepositoryBase, AlertIdempotencyRepository):
    async def get_record(self, *, scope: str, idempotency_key: str) -> Optional[AlertIdempotencyRecord]:
        async with self._session() as session:
            result = await session.execute(
                select(AlertIdempotencyKeyModel).where(
                    AlertIdempotencyKeyModel.scope == scope,
                    AlertIdempotencyKeyModel.idempotency_key == idempotency_key,
                )
            )
            model = result.scalar_one_or_none()
            return _to_idempotency_record(model) if model is not None else None

    async def store_record(
        self,
        *,
        scope: str,
        idempotency_key: str,
        request_fingerprint: str,
        response_payload: dict[str, Any],
    ) -> AlertIdempotencyRecord:
        async with self._session() as session:
            model = AlertIdempotencyKeyModel(
                scope=scope,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                response_payload=response_payload,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return _to_idempotency_record(model)
