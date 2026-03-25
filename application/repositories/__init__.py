"""Repository interfaces for Stage 2 persistence."""

from .history import (
    AggregationRepository,
    HistoryRepository,
    LocationRecord,
    LocationRepository,
)

__all__ = [
    "AggregationRepository",
    "HistoryRepository",
    "LocationRecord",
    "LocationRepository",
]
