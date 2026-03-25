"""Controlled history backfill service for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol

from application.services.history_storage import (
    HistoryPersistenceService,
    build_snapshot_dedupe_key,
    truncate_to_hour,
)
from application.services.quality import QualityScorer
from schemas import AirQualityData, DataSource, HistoricalSnapshotRecord, PollutantData, ResponseMetadata


class HistoricalBackfillProvider(Protocol):
    async def supports_history(self, *, lat: float, lon: float) -> bool: ...

    async def fetch_hourly_history(
        self,
        *,
        lat: float,
        lon: float,
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[AirQualityData]: ...


class DeadLetterWriter(Protocol):
    async def write(self, event: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class BackfillResult:
    total_locations: int
    attempted_locations: int
    inserted_snapshots: int
    duplicate_snapshots: int
    failed_locations: int
    unsupported_locations: int


class UnifiedWeatherHistoricalProvider:
    """Current provider stub: explicit unsupported history until upstream exists."""

    async def supports_history(self, *, lat: float, lon: float) -> bool:
        return False

    async def fetch_hourly_history(
        self,
        *,
        lat: float,
        lon: float,
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[AirQualityData]:
        return []


class HistoryBackfillService:
    """Enumerates configured cities and backfills when provider capability exists."""

    def __init__(
        self,
        *,
        persistence_service: HistoryPersistenceService,
        provider: HistoricalBackfillProvider,
        dead_letter_sink: Optional[DeadLetterWriter] = None,
        quality_scorer: Optional[QualityScorer] = None,
    ):
        self._persistence_service = persistence_service
        self._provider = provider
        self._dead_letter_sink = dead_letter_sink
        self._quality_scorer = quality_scorer or QualityScorer()

    async def run(
        self,
        *,
        cities_mapping: dict[str, dict[str, Any]],
        days: int = 30,
    ) -> BackfillResult:
        end_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_utc = end_utc - timedelta(days=days)
        attempted_locations = 0
        inserted_snapshots = 0
        duplicate_snapshots = 0
        failed_locations = 0
        unsupported_locations = 0

        for city_code, payload in sorted(cities_mapping.items()):
            lat = payload["lat"]
            lon = payload["lon"]
            supported = await self._provider.supports_history(lat=lat, lon=lon)
            if not supported:
                unsupported_locations += 1
                await self._write_dead_letter(
                    {
                        "type": "history_backfill_unsupported",
                        "city_code": city_code,
                        "latitude": lat,
                        "longitude": lon,
                        "start_utc": start_utc.isoformat(),
                        "end_utc": end_utc.isoformat(),
                        "reason": "historical_provider_not_available",
                    }
                )
                continue

            attempted_locations += 1
            try:
                records = await self._provider.fetch_hourly_history(
                    lat=lat,
                    lon=lon,
                    start_utc=start_utc,
                    end_utc=end_utc,
                )
                for item in records:
                    quality = self._quality_scorer.score_snapshot(
                        record_time=item.timestamp,
                        data_source=DataSource.HISTORICAL,
                        source_available=True,
                        fallback_used=False,
                    )
                    snapshot_hour_utc = truncate_to_hour(item.timestamp)
                    dedupe_key = build_snapshot_dedupe_key(
                        city_code=city_code,
                        lat=lat,
                        lon=lon,
                        snapshot_hour_utc=snapshot_hour_utc,
                        data=item,
                    )
                    record = HistoricalSnapshotRecord(
                        snapshot_hour_utc=snapshot_hour_utc,
                        city_code=city_code,
                        latitude=lat,
                        longitude=lon,
                        aqi=item.aqi.value,
                        pollutants=PollutantData(**item.pollutants.model_dump()),
                        data_source=DataSource.HISTORICAL,
                        freshness=quality.freshness,
                        confidence=quality.confidence,
                        metadata=ResponseMetadata(
                            data_source=DataSource.HISTORICAL.value,
                            freshness=quality.freshness.value,
                            confidence=quality.confidence,
                            confidence_explanation=quality.confidence_explanation,
                            fallback_used=False,
                            cache_age_seconds=quality.cache_age_seconds,
                        ),
                    )
                    inserted = await self._persistence_service.persist_snapshot_record(
                        record=record,
                        dedupe_key=dedupe_key,
                        source_timestamp_utc=item.timestamp,
                        source_chain={"provider": type(self._provider).__name__, "mode": "backfill"},
                    )
                    if inserted:
                        inserted_snapshots += 1
                    else:
                        duplicate_snapshots += 1
            except Exception as exc:
                failed_locations += 1
                await self._write_dead_letter(
                    {
                        "type": "history_backfill_failed",
                        "city_code": city_code,
                        "latitude": lat,
                        "longitude": lon,
                        "start_utc": start_utc.isoformat(),
                        "end_utc": end_utc.isoformat(),
                        "error": str(exc),
                    }
                )

        return BackfillResult(
            total_locations=len(cities_mapping),
            attempted_locations=attempted_locations,
            inserted_snapshots=inserted_snapshots,
            duplicate_snapshots=duplicate_snapshots,
            failed_locations=failed_locations,
            unsupported_locations=unsupported_locations,
        )

    async def _write_dead_letter(self, event: dict[str, Any]) -> None:
        if self._dead_letter_sink is not None:
            await self._dead_letter_sink.write(event)
