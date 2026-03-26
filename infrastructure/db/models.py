"""Stage 2 ORM models for history storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LocationModel(Base):
    __tablename__ = "locations"
    __table_args__ = (
        UniqueConstraint("coordinate_key", name="uq_locations_coordinate_key"),
        CheckConstraint("latitude >= -90 AND latitude <= 90", name="latitude_range"),
        CheckConstraint("longitude >= -180 AND longitude <= 180", name="longitude_range"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, unique=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    coordinate_key: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
        server_default=func.now(),
    )

    snapshots: Mapped[list["AirQualitySnapshotModel"]] = relationship(back_populates="location", cascade="all, delete-orphan")


class AirQualitySnapshotModel(Base):
    __tablename__ = "air_quality_snapshots"
    __table_args__ = (
        CheckConstraint("aqi >= 0 AND aqi <= 500", name="aqi_range"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id", ondelete="CASCADE"), nullable=False, index=True)
    snapshot_hour_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source_timestamp_utc: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    aqi: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    pm2_5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pm10: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    no2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    so2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    o3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    data_source: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    freshness: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    dedupe_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, server_default=func.now())

    location: Mapped[LocationModel] = relationship(back_populates="snapshots")
    provenance: Mapped[Optional["DataProvenanceModel"]] = relationship(
        back_populates="snapshot",
        uselist=False,
        cascade="all, delete-orphan",
    )


class DataProvenanceModel(Base):
    __tablename__ = "data_provenance"
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="provenance_confidence_range"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("air_quality_snapshots.id", ondelete="CASCADE"), nullable=False, unique=True)
    data_source: Mapped[str] = mapped_column(String(16), nullable=False)
    freshness: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    confidence_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
    cache_age_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_chain: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, server_default=func.now())

    snapshot: Mapped[AirQualitySnapshotModel] = relationship(back_populates="provenance")


class AlertSubscriptionModel(Base):
    __tablename__ = "alert_subscriptions"
    __table_args__ = (
        CheckConstraint(
            "(latitude IS NULL AND longitude IS NULL) OR (latitude IS NOT NULL AND longitude IS NOT NULL)",
            name="alert_subscription_coordinates_pair",
        ),
        CheckConstraint("aqi_threshold IS NULL OR (aqi_threshold >= 0 AND aqi_threshold <= 500)", name="alert_subscription_aqi_range"),
        CheckConstraint("cooldown_minutes >= 1 AND cooldown_minutes <= 1440", name="alert_subscription_cooldown_range"),
        CheckConstraint(
            "(quiet_hours_start IS NULL AND quiet_hours_end IS NULL) OR (quiet_hours_start IS NOT NULL AND quiet_hours_end IS NOT NULL)",
            name="alert_subscription_quiet_hours_pair",
        ),
        CheckConstraint("quiet_hours_start IS NULL OR (quiet_hours_start >= 0 AND quiet_hours_start <= 23)", name="alert_subscription_quiet_hours_start_range"),
        CheckConstraint("quiet_hours_end IS NULL OR (quiet_hours_end >= 0 AND quiet_hours_end <= 23)", name="alert_subscription_quiet_hours_end_range"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="1")
    city_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    coordinate_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    aqi_threshold: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    nmu_levels: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, server_default="[]")
    cooldown_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=60, server_default="60")
    quiet_hours_start: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    quiet_hours_end: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    channel: Mapped[str] = mapped_column(String(32), nullable=False, default="telegram", server_default="telegram")
    chat_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    last_triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_delivery_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
        server_default=func.now(),
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    delivery_attempts: Mapped[list["AlertDeliveryAttemptModel"]] = relationship(
        back_populates="subscription",
        cascade="all, delete-orphan",
    )
    audit_entries: Mapped[list["AlertAuditLogModel"]] = relationship(
        back_populates="subscription",
        cascade="all, delete-orphan",
    )


class AlertDeliveryAttemptModel(Base):
    __tablename__ = "alert_delivery_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subscription_id: Mapped[str] = mapped_column(ForeignKey("alert_subscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    event_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    provider_response: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    dead_lettered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, server_default=func.now())

    subscription: Mapped[AlertSubscriptionModel] = relationship(back_populates="delivery_attempts")


class AlertAuditLogModel(Base):
    __tablename__ = "alert_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subscription_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("alert_subscriptions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, server_default=func.now())

    subscription: Mapped[Optional[AlertSubscriptionModel]] = relationship(back_populates="audit_entries")


class AlertIdempotencyKeyModel(Base):
    __tablename__ = "alert_idempotency_keys"
    __table_args__ = (
        UniqueConstraint("scope", "idempotency_key", name="uq_alert_idempotency_scope_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(160), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    response_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, server_default=func.now())


__all__ = [
    "AlertAuditLogModel",
    "AlertDeliveryAttemptModel",
    "AlertIdempotencyKeyModel",
    "AlertSubscriptionModel",
    "AirQualitySnapshotModel",
    "Base",
    "DataProvenanceModel",
    "LocationModel",
]
