"""Quality and confidence services for persisted snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from domain.confidence.calculator import ConfidenceCalculator, ConfidenceInputs
from schemas import DataSource, HistoryFreshness


@dataclass(frozen=True)
class SnapshotQuality:
    freshness: HistoryFreshness
    confidence: float
    confidence_explanation: str
    fallback_used: bool
    cache_age_seconds: int


class QualityScorer:
    """Derives freshness and confidence for persisted history records."""

    def __init__(self, confidence_calculator: ConfidenceCalculator | None = None):
        self._confidence_calculator = confidence_calculator or ConfidenceCalculator()

    @staticmethod
    def calculate_freshness(record_time: datetime) -> HistoryFreshness:
        age_seconds = (datetime.now(timezone.utc) - record_time.astimezone(timezone.utc)).total_seconds()
        if age_seconds <= 3600:
            return HistoryFreshness.FRESH
        if age_seconds <= 6 * 3600:
            return HistoryFreshness.STALE
        return HistoryFreshness.EXPIRED

    def score_snapshot(
        self,
        *,
        record_time: datetime,
        data_source: DataSource,
        source_available: bool,
        fallback_used: bool,
        cache_age_seconds: int | None = None,
    ) -> SnapshotQuality:
        effective_cache_age = (
            cache_age_seconds
            if cache_age_seconds is not None
            else max(0, int((datetime.now(timezone.utc) - record_time.astimezone(timezone.utc)).total_seconds()))
        )
        confidence, explanation = self._confidence_calculator.calculate(
            ConfidenceInputs(
                data_source=data_source.value,
                source_available=source_available,
                cache_age_seconds=effective_cache_age,
                fallback_used=fallback_used,
            )
        )
        return SnapshotQuality(
            freshness=self.calculate_freshness(record_time),
            confidence=confidence,
            confidence_explanation=explanation,
            fallback_used=fallback_used,
            cache_age_seconds=effective_cache_age,
        )
