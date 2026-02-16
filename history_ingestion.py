"""
Historical ingestion pipeline for hourly air-quality snapshots.
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from schemas import (
    AirQualityData,
    DataSource,
    HistoricalSnapshotRecord,
    HistoryFreshness,
    PollutantData,
    ResponseMetadata,
)
from confidence_scoring import ConfidenceInputs, calculate_confidence
from anomaly_detection import HourlyAnomalyDetector
from config import config

logger = logging.getLogger(__name__)


DEFAULT_CANONICAL_LOCATIONS: List[Dict[str, Any]] = [
    {"city_code": "moscow", "lat": 55.7558, "lon": 37.6176},
    {"city_code": "spb", "lat": 59.9311, "lon": 30.3609},
    {"city_code": "yekaterinburg", "lat": 56.8389, "lon": 60.6057},
    {"city_code": "novosibirsk", "lat": 55.0084, "lon": 82.9357},
]


class InMemoryHistoricalSnapshotStore:
    """Simple in-memory snapshot store with dedupe support for v4 bootstrap."""

    def __init__(self):
        self._records: Dict[str, HistoricalSnapshotRecord] = {}
        self._anomaly_detector = HourlyAnomalyDetector(
            baseline_window=config.history.anomaly_baseline_window,
            min_absolute_delta=config.history.anomaly_min_absolute_delta,
            min_relative_delta=config.history.anomaly_min_relative_delta,
        )

    async def write_snapshot(self, dedupe_key: str, record: HistoricalSnapshotRecord) -> bool:
        """Write snapshot and return True if inserted, False if duplicate."""
        if dedupe_key in self._records:
            return False
        self._records[dedupe_key] = record
        return True

    def count(self) -> int:
        return len(self._records)

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
    ) -> Dict[str, Any]:
        items: List[HistoricalSnapshotRecord] = []
        normalized_city = city_code.lower() if city_code else None
        rounded_lat = round(lat, 4) if lat is not None else None
        rounded_lon = round(lon, 4) if lon is not None else None

        for record in self._records.values():
            ts = record.snapshot_hour_utc.astimezone(timezone.utc)
            if ts < start_utc or ts > end_utc:
                continue
            if normalized_city is not None and (record.city_code or "").lower() != normalized_city:
                continue
            if rounded_lat is not None and round(record.latitude, 4) != rounded_lat:
                continue
            if rounded_lon is not None and round(record.longitude, 4) != rounded_lon:
                continue
            items.append(record.model_copy(deep=True))

        # Anomaly detection should run chronologically against local baseline.
        items.sort(key=lambda r: r.snapshot_hour_utc)
        previous_aqi: List[float] = []
        for item in items:
            result = self._anomaly_detector.evaluate(current_value=float(item.aqi), previous_values=previous_aqi)
            item.anomaly_detected = result.detected
            item.anomaly_type = result.anomaly_type
            item.anomaly_score = round(result.score, 3)
            item.anomaly_baseline_aqi = round(result.baseline, 2) if result.baseline > 0 else None
            previous_aqi.append(float(item.aqi))

        items.sort(key=lambda r: r.snapshot_hour_utc, reverse=True)
        total = len(items)
        paged = items[offset : offset + limit]
        return {"total": total, "items": paged}


class InMemoryDeadLetterSink:
    """Dead-letter sink used by tests and local runs."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    async def write(self, event: Dict[str, Any]) -> None:
        self.events.append(event)


class JsonlDeadLetterSink:
    """Dead-letter sink writing JSONL events to disk."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    async def write(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False)
        await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class HistoryIngestionPipeline:
    """
    Ingestion pipeline for hourly historical snapshots.
    Implements idempotency, retries and dead-letter logging.
    """

    def __init__(
        self,
        fetch_current_data: Callable[[float, float], Awaitable[AirQualityData]],
        snapshot_store: InMemoryHistoricalSnapshotStore,
        dead_letter_sink: Optional[Any] = None,
        canonical_locations: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 0.5,
    ):
        self.fetch_current_data = fetch_current_data
        self.snapshot_store = snapshot_store
        self.dead_letter_sink = dead_letter_sink
        self.canonical_locations = canonical_locations or DEFAULT_CANONICAL_LOCATIONS
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._custom_coordinates: Set[Tuple[float, float]] = set()

    def register_custom_coordinates(self, lat: float, lon: float) -> None:
        """Register custom coordinates to be ingested on schedule."""
        self._custom_coordinates.add((round(lat, 4), round(lon, 4)))

    def _build_dedupe_key(
        self,
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

    @staticmethod
    def _truncate_to_hour(dt: datetime) -> datetime:
        normalized = dt.astimezone(timezone.utc)
        return normalized.replace(minute=0, second=0, microsecond=0)

    @staticmethod
    def _calculate_freshness(record_time: datetime) -> HistoryFreshness:
        age_seconds = (datetime.now(timezone.utc) - record_time.astimezone(timezone.utc)).total_seconds()
        if age_seconds <= 3600:
            return HistoryFreshness.FRESH
        if age_seconds <= 6 * 3600:
            return HistoryFreshness.STALE
        return HistoryFreshness.EXPIRED

    async def ingest_location(
        self, lat: float, lon: float, city_code: Optional[str] = None, data_source: DataSource = DataSource.LIVE
    ) -> bool:
        """Ingest a single location with retry and dead-letter behavior."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                current_data = await self.fetch_current_data(lat, lon)
                snapshot_hour_utc = self._truncate_to_hour(current_data.timestamp)
                dedupe_key = self._build_dedupe_key(city_code, lat, lon, snapshot_hour_utc, current_data)
                freshness = self._calculate_freshness(current_data.timestamp)
                cache_age_seconds = max(
                    0, int((datetime.now(timezone.utc) - current_data.timestamp.astimezone(timezone.utc)).total_seconds())
                )
                confidence_score, confidence_reason = calculate_confidence(
                    ConfidenceInputs(
                        data_source=data_source.value,
                        source_available=(data_source != DataSource.FALLBACK),
                        cache_age_seconds=cache_age_seconds,
                        fallback_used=(data_source == DataSource.FALLBACK),
                    )
                )

                snapshot = HistoricalSnapshotRecord(
                    snapshot_hour_utc=snapshot_hour_utc,
                    city_code=city_code,
                    latitude=lat,
                    longitude=lon,
                    aqi=current_data.aqi.value,
                    pollutants=PollutantData(**current_data.pollutants.model_dump()),
                    data_source=data_source,
                    freshness=freshness,
                    confidence=confidence_score,
                    metadata=ResponseMetadata(
                        data_source=data_source.value,
                        freshness=freshness.value,
                        confidence=confidence_score,
                        confidence_explanation=confidence_reason,
                        fallback_used=(data_source == DataSource.FALLBACK),
                        cache_age_seconds=cache_age_seconds,
                    ),
                )

                inserted = await self.snapshot_store.write_snapshot(dedupe_key, snapshot)
                if not inserted:
                    logger.debug("Historical snapshot duplicate skipped: %s", dedupe_key)
                return True
            except Exception as exc:
                last_error = exc
                if attempt <= self.max_retries:
                    await asyncio.sleep(self.retry_delay_seconds)
                    continue

        logger.error("Historical ingestion failed for location after retries: lat=%s lon=%s", lat, lon)
        if self.dead_letter_sink is not None and last_error is not None:
            event = {
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "city_code": city_code,
                "latitude": lat,
                "longitude": lon,
                "error": str(last_error),
                "max_retries": self.max_retries,
            }
            await self.dead_letter_sink.write(event)
        return False

    async def ingest_once(self) -> Dict[str, int]:
        """Ingest all canonical and registered custom coordinates once."""
        total = 0
        success = 0
        failed = 0

        for loc in self.canonical_locations:
            total += 1
            ok = await self.ingest_location(
                lat=loc["lat"],
                lon=loc["lon"],
                city_code=loc.get("city_code"),
                data_source=DataSource.LIVE,
            )
            if ok:
                success += 1
            else:
                failed += 1

        for lat, lon in sorted(self._custom_coordinates):
            total += 1
            ok = await self.ingest_location(lat=lat, lon=lon, city_code=None, data_source=DataSource.LIVE)
            if ok:
                success += 1
            else:
                failed += 1

        return {"total": total, "success": success, "failed": failed}
