"""Stage 2 ORM models for history storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
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

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
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

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
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

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
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


__all__ = [
    "AirQualitySnapshotModel",
    "Base",
    "DataProvenanceModel",
    "LocationModel",
]
