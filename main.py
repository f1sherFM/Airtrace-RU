"""AirTrace RU backend bootstrap for the Stage 1 modular monolith."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import uvicorn

from core.env import load_repo_env

load_repo_env(root_dir=Path(__file__).resolve().parent)
from core.sentry import init_sentry

init_sentry(app_role="api")

from application.services.alert_rule_engine import AlertRuleEngine
from application.queries.health import (
    _derive_overall_health_status,
    _is_optional_health_component,
    _normalize_health_component,
    _normalize_health_status,
)
from application.services.alerts import AlertSubscriptionService
from config import config  # re-export for compatibility
from connection_pool import get_connection_pool_manager
from core.app_factory import create_api_app
from history_ingestion import HistoryIngestionPipeline, InMemoryHistoricalSnapshotStore
from infrastructure.repositories import (
    InMemoryAlertAuditRepository,
    InMemoryAlertDeliveryAttemptRepository,
    InMemoryAlertIdempotencyRepository,
    InMemoryAlertSubscriptionRepository,
)
from services import AirQualityService
from telegram_delivery import TelegramDeliveryService, JsonlDeadLetterSink as TelegramDeadLetterSink
from unified_weather_service import unified_weather_service

logger = logging.getLogger(__name__)

air_quality_service: Optional[AirQualityService] = None
history_ingestion_pipeline: Optional[HistoryIngestionPipeline] = None
history_snapshot_store: Optional[InMemoryHistoricalSnapshotStore] = None
alert_rule_engine = AlertRuleEngine()
telegram_delivery_service = TelegramDeliveryService(
    bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    max_retries=int(os.getenv("TELEGRAM_MAX_RETRIES", "3")),
    retry_delay_seconds=float(os.getenv("TELEGRAM_RETRY_DELAY_SECONDS", "0.7")),
    dead_letter_sink=TelegramDeadLetterSink("logs/telegram_dead_letter.jsonl"),
)
alert_subscription_service = AlertSubscriptionService(
    subscription_repository=InMemoryAlertSubscriptionRepository(),
    delivery_attempt_repository=InMemoryAlertDeliveryAttemptRepository(),
    audit_repository=InMemoryAlertAuditRepository(),
    idempotency_repository=InMemoryAlertIdempotencyRepository(),
    telegram_delivery_service=telegram_delivery_service,
)

app = create_api_app()


if __name__ == "__main__":
    server_config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,
        "log_level": "info",
        "access_log": True,
        "server_header": False,
        "date_header": False,
    }
    logger.info("Starting AirTrace RU Backend server...")
    logger.info("Server configuration: %s", server_config)
    uvicorn.run("main:app", **server_config)
