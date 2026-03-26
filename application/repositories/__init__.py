"""Repository interfaces for Stage 2/4 persistence."""

from .alerts import (
    AlertAuditEntryRecord,
    AlertAuditRepository,
    AlertDeliveryAttemptRecord,
    AlertDeliveryAttemptRepository,
    AlertIdempotencyRecord,
    AlertIdempotencyRepository,
    AlertSubscriptionRecord,
    AlertSubscriptionRepository,
)
from .history import (
    AggregationRepository,
    HistoryRepository,
    LocationRecord,
    LocationRepository,
)

__all__ = [
    "AlertAuditEntryRecord",
    "AlertAuditRepository",
    "AlertDeliveryAttemptRecord",
    "AlertDeliveryAttemptRepository",
    "AlertIdempotencyRecord",
    "AlertIdempotencyRepository",
    "AlertSubscriptionRecord",
    "AlertSubscriptionRepository",
    "AggregationRepository",
    "HistoryRepository",
    "LocationRecord",
    "LocationRepository",
]
