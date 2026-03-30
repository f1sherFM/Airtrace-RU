"""History persistence services and repository-backed store adapter."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from application.services.anomaly_detection import HourlyAnomalyDetector
from application.repositories.history import HistoryRepository, LocationRepository, LocationRecord
from config import config
from schemas import AirQualityData, DataSource, HistoricalSnapshotRecord, HistorySortOrder, PollutantData, ResponseMetadata


def build_snapshot_dedupe_key(
    *,
    city_code: Optional[str],
    lat: float,
    lon: float,
    snapshot_hour_utc: datetime,
    data: AirQualityData,
) -> str:
    location_id = city_code if city_code else f"{lat:.4f},{lon:.4f}"
    payload = {
        "hour": snapshot_hour_utc.isoformat(),
        "location": location_id,
        "aqi": data.aqi.value,
        "pollutants": data.pollutants.model_dump(exclude_none=True),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def truncate_to_hour(dt: datetime) -> datetime:
    normalized = dt.astimezone(timezone.utc)
    return normalized.replace(minute=0, second=0, microsecond=0)


def build_canonical_locations_from_mapping(cities_mapping: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for city_code, payload in sorted(cities_mapping.items()):
        locations.append(
            {
                "city_code": city_code,
                "name": payload["name"],
                "lat": payload["lat"],
                "lon": payload["lon"],
            }
        )
    return locations


def apply_anomaly_metadata(items: list[HistoricalSnapshotRecord]) -> list[HistoricalSnapshotRecord]:
    anomaly_detector = HourlyAnomalyDetector(
        baseline_window=config.history.anomaly_baseline_window,
        min_absolute_delta=config.history.anomaly_min_absolute_delta,
        min_relative_delta=config.history.anomaly_min_relative_delta,
    )
    chronological = sorted((item.model_copy(deep=True) for item in items), key=lambda record: record.snapshot_hour_utc)
    previous_aqi: list[float] = []
    for item in chronological:
        result = anomaly_detector.evaluate(current_value=float(item.aqi), previous_values=previous_aqi)
        item.anomaly_detected = result.detected
        item.anomaly_type = result.anomaly_type
        item.anomaly_score = round(result.score, 3)
        item.anomaly_baseline_aqi = round(result.baseline, 2) if result.baseline > 0 else None
        previous_aqi.append(float(item.aqi))
    chronological.sort(key=lambda record: record.snapshot_hour_utc, reverse=True)
    return chronological


class HistoryPersistenceService:
    """Repository-aware snapshot persistence."""

    def __init__(self, *, location_repository: LocationRepository, history_repository: HistoryRepository):
        self._location_repository = location_repository
        self._history_repository = history_repository

    async def bootstrap_configured_locations(self, cities_mapping: dict[str, dict[str, Any]]) -> list[LocationRecord]:
        locations: list[LocationRecord] = []
        for city_code, payload in sorted(cities_mapping.items()):
            locations.append(
                await self._location_repository.upsert_city_location(
                    city_code=city_code,
                    name=payload["name"],
                    latitude=payload["lat"],
                    longitude=payload["lon"],
                    is_active=True,
                )
            )
        return locations

    async def resolve_location(
        self,
        *,
        lat: float,
        lon: float,
        city_code: Optional[str] = None,
        location_name: Optional[str] = None,
    ) -> LocationRecord:
        if city_code:
            existing = await self._location_repository.get_by_city_code(city_code)
            if existing is not None:
                return existing
            return await self._location_repository.upsert_city_location(
                city_code=city_code,
                name=location_name or city_code,
                latitude=lat,
                longitude=lon,
                is_active=True,
            )
        return await self._location_repository.get_or_create_coordinates(
            latitude=lat,
            longitude=lon,
            name=location_name,
        )

    async def persist_snapshot_record(
        self,
        *,
        record: HistoricalSnapshotRecord,
        dedupe_key: str,
        source_timestamp_utc: Optional[datetime] = None,
        source_chain: Optional[dict[str, object]] = None,
    ) -> bool:
        location = await self.resolve_location(
            lat=record.latitude,
            lon=record.longitude,
            city_code=record.city_code,
        )
        return await self._history_repository.insert_snapshot(
            location=location,
            record=record,
            dedupe_key=dedupe_key,
            source_timestamp_utc=source_timestamp_utc or record.snapshot_hour_utc,
            source_chain=source_chain,
        )

    async def persist_current_observation(
        self,
        *,
        lat: float,
        lon: float,
        data: AirQualityData,
        city_code: Optional[str] = None,
        source_chain: Optional[dict[str, object]] = None,
    ) -> bool:
        snapshot_hour_utc = truncate_to_hour(data.timestamp)
        dedupe_key = build_snapshot_dedupe_key(
            city_code=city_code,
            lat=lat,
            lon=lon,
            snapshot_hour_utc=snapshot_hour_utc,
            data=data,
        )
        metadata = ResponseMetadata(**data.metadata.model_dump())
        record = HistoricalSnapshotRecord(
            snapshot_hour_utc=snapshot_hour_utc,
            city_code=city_code,
            latitude=lat,
            longitude=lon,
            aqi=data.aqi.value,
            pollutants=PollutantData(**data.pollutants.model_dump()),
            data_source=DataSource(metadata.data_source),
            freshness=data.metadata.freshness,
            confidence=metadata.confidence,
            metadata=metadata,
        )
        return await self.persist_snapshot_record(
            record=record,
            dedupe_key=dedupe_key,
            source_timestamp_utc=data.timestamp,
            source_chain=source_chain,
        )


class RepositoryBackedHistoricalSnapshotStore:
    """Store adapter keeping the Stage 0/1 query contract over repositories."""

    def __init__(
        self,
        *,
        history_repository: HistoryRepository,
        persistence_service: HistoryPersistenceService,
    ):
        self._history_repository = history_repository
        self._persistence_service = persistence_service

    async def write_snapshot(self, dedupe_key: str, record: HistoricalSnapshotRecord) -> bool:
        return await self._persistence_service.persist_snapshot_record(record=record, dedupe_key=dedupe_key)

    async def count(self) -> int:
        return await self._history_repository.count_snapshots()

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
    ) -> Dict[str, Any]:
        result = await self._history_repository.query_snapshots(
            start_utc=start_utc,
            end_utc=end_utc,
            city_code=city_code,
            lat=lat,
            lon=lon,
            limit=limit,
            offset=offset,
            sort=sort,
        )
        items = apply_anomaly_metadata(list(result["items"]))
        if sort == HistorySortOrder.ASC:
            items = list(reversed(items))
        return {"total": result["total"], "items": items}
