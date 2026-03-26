"""Concrete repository implementations for Stage 2/4 persistence."""

from .inmemory_alerts import (
    InMemoryAlertAuditRepository,
    InMemoryAlertDeliveryAttemptRepository,
    InMemoryAlertIdempotencyRepository,
    InMemoryAlertSubscriptionRepository,
)
from .sqlalchemy_alerts import (
    SQLAlchemyAlertAuditRepository,
    SQLAlchemyAlertDeliveryAttemptRepository,
    SQLAlchemyAlertIdempotencyRepository,
    SQLAlchemyAlertSubscriptionRepository,
)
from .sqlalchemy_history import (
    SQLAlchemyAggregationRepository,
    SQLAlchemyHistoryRepository,
    SQLAlchemyLocationRepository,
)

__all__ = [
    "InMemoryAlertAuditRepository",
    "InMemoryAlertDeliveryAttemptRepository",
    "InMemoryAlertIdempotencyRepository",
    "InMemoryAlertSubscriptionRepository",
    "SQLAlchemyAlertAuditRepository",
    "SQLAlchemyAlertDeliveryAttemptRepository",
    "SQLAlchemyAlertIdempotencyRepository",
    "SQLAlchemyAlertSubscriptionRepository",
    "SQLAlchemyAggregationRepository",
    "SQLAlchemyHistoryRepository",
    "SQLAlchemyLocationRepository",
]
