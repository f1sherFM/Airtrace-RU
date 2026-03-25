"""Repository contracts for history storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol

from schemas import DailyAggregateRecord, HistoricalSnapshotRecord, HistorySortOrder


@dataclass(frozen=True)
class LocationRecord:
    id: int
    city_code: Optional[str]
    name: str
    latitude: float
    longitude: float
    is_active: bool


class LocationRepository(Protocol):
    async def get_by_city_code(self, city_code: str) -> Optional[LocationRecord]: ...

    async def upsert_city_location(
        self,
        *,
        city_code: str,
        name: str,
        latitude: float,
        longitude: float,
        is_active: bool = True,
    ) -> LocationRecord: ...

    async def get_or_create_coordinates(
        self,
        *,
        latitude: float,
        longitude: float,
        name: Optional[str] = None,
    ) -> LocationRecord: ...

    async def list_active_locations(self) -> list[LocationRecord]: ...


class HistoryRepository(Protocol):
    async def insert_snapshot(
        self,
        *,
        location: LocationRecord,
        record: HistoricalSnapshotRecord,
        dedupe_key: str,
        source_timestamp_utc: Optional[datetime] = None,
        source_chain: Optional[dict[str, object]] = None,
    ) -> bool: ...

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
    ) -> dict[str, object]: ...

    async def count_snapshots(self) -> int: ...


class AggregationRepository(Protocol):
    async def query_daily_aggregates(
        self,
        *,
        start_utc: datetime,
        end_utc: datetime,
        city_code: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> list[DailyAggregateRecord]: ...
