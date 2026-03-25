"""SQLAlchemy repositories for Stage 2 history storage."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import Select, and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import joinedload

from application.repositories.history import (
    AggregationRepository,
    HistoryRepository,
    LocationRecord,
    LocationRepository,
)
from infrastructure.db.models import AirQualitySnapshotModel, DataProvenanceModel, LocationModel
from schemas import (
    DailyAggregateRecord,
    DataSource,
    HistoricalSnapshotRecord,
    HistoryFreshness,
    HistorySortOrder,
    PollutantData,
    ResponseMetadata,
)


def _coordinate_key(latitude: float, longitude: float) -> str:
    return f"{round(latitude, 4):.4f},{round(longitude, 4):.4f}"


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _location_from_model(model: LocationModel) -> LocationRecord:
    return LocationRecord(
        id=model.id,
        city_code=model.city_code,
        name=model.name,
        latitude=model.latitude,
        longitude=model.longitude,
        is_active=model.is_active,
    )


def _record_from_models(snapshot: AirQualitySnapshotModel, provenance: Optional[DataProvenanceModel]) -> HistoricalSnapshotRecord:
    metadata = ResponseMetadata(
        data_source=(provenance.data_source if provenance else snapshot.data_source),
        freshness=(provenance.freshness if provenance else snapshot.freshness),
        confidence=_to_float(provenance.confidence if provenance else snapshot.confidence) or 0.0,
        confidence_explanation=provenance.confidence_explanation if provenance else None,
        fallback_used=bool(provenance.fallback_used) if provenance else False,
        cache_age_seconds=provenance.cache_age_seconds if provenance else None,
    )
    return HistoricalSnapshotRecord(
        snapshot_hour_utc=snapshot.snapshot_hour_utc,
        city_code=snapshot.location.city_code,
        latitude=snapshot.location.latitude,
        longitude=snapshot.location.longitude,
        aqi=snapshot.aqi,
        pollutants=PollutantData(
            pm2_5=snapshot.pm2_5,
            pm10=snapshot.pm10,
            no2=snapshot.no2,
            so2=snapshot.so2,
            o3=snapshot.o3,
        ),
        data_source=DataSource(snapshot.data_source),
        freshness=HistoryFreshness(snapshot.freshness),
        confidence=_to_float(snapshot.confidence) or 0.0,
        ingested_at=snapshot.ingested_at,
        metadata=metadata,
    )


class _SQLAlchemyRepositoryBase:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    @asynccontextmanager
    async def _session(self):
        async with self._session_factory() as session:
            yield session


class SQLAlchemyLocationRepository(_SQLAlchemyRepositoryBase, LocationRepository):
    async def get_by_city_code(self, city_code: str) -> Optional[LocationRecord]:
        async with self._session() as session:
            result = await session.execute(select(LocationModel).where(LocationModel.city_code == city_code))
            model = result.scalar_one_or_none()
            return _location_from_model(model) if model else None

    async def upsert_city_location(
        self,
        *,
        city_code: str,
        name: str,
        latitude: float,
        longitude: float,
        is_active: bool = True,
    ) -> LocationRecord:
        coordinate_key = _coordinate_key(latitude, longitude)
        async with self._session() as session:
            result = await session.execute(select(LocationModel).where(LocationModel.city_code == city_code))
            model = result.scalar_one_or_none()
            if model is None:
                model = LocationModel(
                    city_code=city_code,
                    name=name,
                    latitude=latitude,
                    longitude=longitude,
                    coordinate_key=coordinate_key,
                    is_active=is_active,
                )
                session.add(model)
            else:
                model.name = name
                model.latitude = latitude
                model.longitude = longitude
                model.coordinate_key = coordinate_key
                model.is_active = is_active
            await session.commit()
            await session.refresh(model)
            return _location_from_model(model)

    async def get_or_create_coordinates(
        self,
        *,
        latitude: float,
        longitude: float,
        name: Optional[str] = None,
    ) -> LocationRecord:
        coordinate_key = _coordinate_key(latitude, longitude)
        async with self._session() as session:
            result = await session.execute(select(LocationModel).where(LocationModel.coordinate_key == coordinate_key))
            model = result.scalar_one_or_none()
            if model is None:
                model = LocationModel(
                    city_code=None,
                    name=name or coordinate_key,
                    latitude=latitude,
                    longitude=longitude,
                    coordinate_key=coordinate_key,
                    is_active=True,
                )
                session.add(model)
                await session.commit()
                await session.refresh(model)
            return _location_from_model(model)

    async def list_active_locations(self) -> list[LocationRecord]:
        async with self._session() as session:
            result = await session.execute(select(LocationModel).where(LocationModel.is_active.is_(True)).order_by(LocationModel.name))
            return [_location_from_model(model) for model in result.scalars().all()]


class SQLAlchemyHistoryRepository(_SQLAlchemyRepositoryBase, HistoryRepository):
    async def insert_snapshot(
        self,
        *,
        location: LocationRecord,
        record: HistoricalSnapshotRecord,
        dedupe_key: str,
        source_timestamp_utc: Optional[datetime] = None,
        source_chain: Optional[dict[str, object]] = None,
    ) -> bool:
        async with self._session() as session:
            snapshot = AirQualitySnapshotModel(
                location_id=location.id,
                snapshot_hour_utc=record.snapshot_hour_utc,
                source_timestamp_utc=source_timestamp_utc or record.snapshot_hour_utc,
                aqi=record.aqi,
                pm2_5=record.pollutants.pm2_5,
                pm10=record.pollutants.pm10,
                no2=record.pollutants.no2,
                so2=record.pollutants.so2,
                o3=record.pollutants.o3,
                data_source=record.data_source.value,
                freshness=record.freshness.value,
                confidence=record.confidence,
                dedupe_key=dedupe_key,
                ingested_at=record.ingested_at,
            )
            snapshot.provenance = DataProvenanceModel(
                data_source=record.metadata.data_source,
                freshness=record.metadata.freshness,
                confidence=record.metadata.confidence,
                confidence_explanation=record.metadata.confidence_explanation,
                fallback_used=record.metadata.fallback_used,
                cache_age_seconds=record.metadata.cache_age_seconds,
                source_chain=source_chain,
                raw_metadata=record.metadata.model_dump(mode="json"),
            )
            session.add(snapshot)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                return False
            return True

    async def query_snapshots(
        self,
        *,
        start_utc: datetime,
        end_utc: datetime,
        city_code: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
        sort: HistorySortOrder = HistorySortOrder.DESC,
    ) -> dict[str, object]:
        async with self._session() as session:
            filters = [
                AirQualitySnapshotModel.snapshot_hour_utc >= start_utc,
                AirQualitySnapshotModel.snapshot_hour_utc <= end_utc,
            ]
            if city_code is not None:
                filters.append(LocationModel.city_code == city_code.lower())
            if lat is not None and lon is not None:
                filters.append(LocationModel.coordinate_key == _coordinate_key(lat, lon))

            statement: Select[tuple[AirQualitySnapshotModel]] = (
                select(AirQualitySnapshotModel)
                .join(AirQualitySnapshotModel.location)
                .options(
                    joinedload(AirQualitySnapshotModel.location),
                    joinedload(AirQualitySnapshotModel.provenance),
                )
                .where(and_(*filters))
            )

            total_result = await session.execute(
                select(func.count(AirQualitySnapshotModel.id))
                .join(LocationModel)
                .where(and_(*filters))
            )
            total = int(total_result.scalar_one() or 0)

            order_column = (
                AirQualitySnapshotModel.snapshot_hour_utc.asc()
                if sort == HistorySortOrder.ASC
                else AirQualitySnapshotModel.snapshot_hour_utc.desc()
            )
            rows = await session.execute(statement.order_by(order_column).offset(offset).limit(limit))
            items = [_record_from_models(model, model.provenance) for model in rows.scalars().all()]
            return {"total": total, "items": items}

    async def count_snapshots(self) -> int:
        async with self._session() as session:
            result = await session.execute(select(func.count(AirQualitySnapshotModel.id)))
            return int(result.scalar_one() or 0)


class SQLAlchemyAggregationRepository(_SQLAlchemyRepositoryBase, AggregationRepository):
    async def query_daily_aggregates(
        self,
        *,
        start_utc: datetime,
        end_utc: datetime,
        city_code: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> list[DailyAggregateRecord]:
        history_repository = SQLAlchemyHistoryRepository(self._session_factory)
        result = await history_repository.query_snapshots(
            start_utc=start_utc,
            end_utc=end_utc,
            city_code=city_code,
            lat=lat,
            lon=lon,
            limit=10000,
            offset=0,
        )
        items: list[HistoricalSnapshotRecord] = list(reversed(result["items"]))
        grouped: dict[tuple[datetime, Optional[str], float, float], list[HistoricalSnapshotRecord]] = defaultdict(list)
        for item in items:
            day_utc = item.snapshot_hour_utc.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            key = (day_utc, item.city_code, item.latitude, item.longitude)
            grouped[key].append(item)

        aggregates: list[DailyAggregateRecord] = []
        for (day_utc, record_city_code, latitude, longitude), records in sorted(grouped.items(), reverse=True):
            aqi_values = [record.aqi for record in records]
            source_counts = Counter(record.data_source for record in records)
            dominant_source = source_counts.most_common(1)[0][0]
            aggregates.append(
                DailyAggregateRecord(
                    day_utc=day_utc,
                    city_code=record_city_code,
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
