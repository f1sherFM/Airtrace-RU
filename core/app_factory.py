"""FastAPI app factory for the Stage 1 modular monolith."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.legacy import router as legacy_router
from api.ops import router as ops_router
from api.v2.alerts import router as v2_alerts_router
from api.v1.readonly import router as v1_readonly_router
from api.v2.readonly import router as v2_readonly_router
from application.services.alert_worker import AlertEvaluationWorker
from application.services.alerts import AlertSubscriptionService
from config import config
from core.settings import get_cities_mapping, load_cities_config
from core.legacy_runtime import (
    get_air_quality_service,
    get_telegram_delivery_service,
    set_air_quality_service,
    set_alert_subscription_service,
    set_history_ingestion_pipeline,
    set_history_snapshot_store,
)
from graceful_degradation import get_graceful_degradation_manager
from infrastructure.db import close_database_runtime, initialize_database_runtime, run_database_migrations
from infrastructure.repositories import (
    InMemoryAlertAuditRepository,
    InMemoryAlertDeliveryAttemptRepository,
    InMemoryAlertIdempotencyRepository,
    InMemoryAlertSubscriptionRepository,
    SQLAlchemyAlertAuditRepository,
    SQLAlchemyAlertDeliveryAttemptRepository,
    SQLAlchemyAlertIdempotencyRepository,
    SQLAlchemyAlertSubscriptionRepository,
    SQLAlchemyAggregationRepository,
    SQLAlchemyHistoryRepository,
    SQLAlchemyLocationRepository,
)
from history_ingestion import (
    HistoryIngestionPipeline,
    InMemoryHistoricalSnapshotStore,
    JsonlDeadLetterSink,
)
from application.services.history_storage import (
    HistoryPersistenceService,
    RepositoryBackedHistoricalSnapshotStore,
    build_canonical_locations_from_mapping,
)
from middleware import PrivacyMiddleware, set_privacy_middleware, setup_privacy_logging
from rate_limit_middleware import setup_rate_limiting
from rate_limit_monitoring import setup_rate_limit_logging
from services import AirQualityService
from unified_weather_service import unified_weather_service

setup_privacy_logging()
logger = logging.getLogger(__name__)

V2_RESPONSE_HEADERS = {
    "X-AirTrace-API-Version": "2",
    "X-AirTrace-API-Contract": "readonly",
}


class UnicodeJSONResponse(JSONResponse):
    """JSON response with explicit UTF-8 serialization."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


async def periodic_cleanup():
    while True:
        try:
            await asyncio.sleep(300)
            await unified_weather_service.cache_manager.clear_expired()
            logger.debug("Periodic cache cleanup completed")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Error during periodic cleanup: %s", exc)


async def periodic_history_ingestion(interval_seconds: int = 3600):
    from core.legacy_runtime import get_history_ingestion_pipeline

    run_on_startup = os.getenv("HISTORY_INGEST_RUN_ON_STARTUP", "true").lower() == "true"
    history_ingestion_pipeline = get_history_ingestion_pipeline()
    if run_on_startup and history_ingestion_pipeline is not None:
        try:
            result = await history_ingestion_pipeline.ingest_once()
            logger.info("Initial history ingestion completed: %s", result)
        except Exception as exc:
            logger.error("Initial history ingestion failed: %s", exc)

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            history_ingestion_pipeline = get_history_ingestion_pipeline()
            if history_ingestion_pipeline is None:
                continue
            result = await history_ingestion_pipeline.ingest_once()
            logger.info("Periodic history ingestion completed: %s", result)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Error during periodic history ingestion: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AirTrace RU Backend...")

    air_quality_service = AirQualityService()
    set_air_quality_service(air_quality_service)
    logger.info("Air quality service initialized")

    degradation_manager = get_graceful_degradation_manager()
    await degradation_manager.register_component("external_api", lambda: air_quality_service.check_external_api_health())
    await degradation_manager.register_component(
        "cache",
        lambda: "healthy" in air_quality_service.cache_manager.get_status(),
    )

    if config.performance.rate_limiting_enabled:
        from rate_limit_middleware import get_rate_limit_manager

        await degradation_manager.register_component("rate_limiting", lambda: get_rate_limit_manager().is_enabled())
    if config.weather_api.enabled:
        await degradation_manager.register_component("weather_api", lambda: unified_weather_service.check_weather_api_health())

    cities_mapping = get_cities_mapping()
    canonical_locations = build_canonical_locations_from_mapping(cities_mapping)
    history_store: Any
    history_ingestion_kwargs: dict[str, Any] = {}
    alert_subscription_service: AlertSubscriptionService
    if config.database.enabled:
        if config.database.run_migrations_on_startup:
            run_database_migrations(config.database.alembic_url or config.database.url)
        runtime = initialize_database_runtime(config.database)
        location_repository = SQLAlchemyLocationRepository(runtime.session_factory)
        history_repository = SQLAlchemyHistoryRepository(runtime.session_factory)
        aggregation_repository = SQLAlchemyAggregationRepository(runtime.session_factory)
        alert_subscription_repository = SQLAlchemyAlertSubscriptionRepository(runtime.session_factory)
        alert_delivery_attempt_repository = SQLAlchemyAlertDeliveryAttemptRepository(runtime.session_factory)
        alert_audit_repository = SQLAlchemyAlertAuditRepository(runtime.session_factory)
        alert_idempotency_repository = SQLAlchemyAlertIdempotencyRepository(runtime.session_factory)
        persistence_service = HistoryPersistenceService(
            location_repository=location_repository,
            history_repository=history_repository,
        )
        await persistence_service.bootstrap_configured_locations(cities_mapping)
        history_store = RepositoryBackedHistoricalSnapshotStore(
            history_repository=history_repository,
            persistence_service=persistence_service,
        )
        history_ingestion_kwargs["persistence_service"] = persistence_service

        async def _persist_current_observation(lat: float, lon: float, data: Any) -> None:
            await persistence_service.persist_current_observation(
                lat=lat,
                lon=lon,
                data=data,
            )

        unified_weather_service.set_current_persistence_callback(_persist_current_observation)
        alert_subscription_service = AlertSubscriptionService(
            subscription_repository=alert_subscription_repository,
            delivery_attempt_repository=alert_delivery_attempt_repository,
            audit_repository=alert_audit_repository,
            idempotency_repository=alert_idempotency_repository,
            telegram_delivery_service=get_telegram_delivery_service(),
        )
        logger.info(
            "DB-backed history storage enabled (timescaledb=%s)",
            config.database.timescaledb_enabled,
        )
    else:
        history_store = InMemoryHistoricalSnapshotStore()
        unified_weather_service.set_current_persistence_callback(None)
        alert_subscription_service = AlertSubscriptionService(
            subscription_repository=InMemoryAlertSubscriptionRepository(),
            delivery_attempt_repository=InMemoryAlertDeliveryAttemptRepository(),
            audit_repository=InMemoryAlertAuditRepository(),
            idempotency_repository=InMemoryAlertIdempotencyRepository(),
            telegram_delivery_service=get_telegram_delivery_service(),
        )

    set_history_snapshot_store(history_store)
    set_alert_subscription_service(alert_subscription_service)
    history_ingestion_pipeline = HistoryIngestionPipeline(
        fetch_current_data=unified_weather_service.get_current_combined_data,
        snapshot_store=history_store,
        dead_letter_sink=JsonlDeadLetterSink("logs/history_dead_letter.jsonl"),
        canonical_locations=canonical_locations,
        max_retries=int(os.getenv("HISTORY_INGEST_MAX_RETRIES", "3")),
        retry_delay_seconds=float(os.getenv("HISTORY_INGEST_RETRY_DELAY_SECONDS", "0.5")),
        **history_ingestion_kwargs,
    )
    set_history_ingestion_pipeline(history_ingestion_pipeline)

    cleanup_task = asyncio.create_task(periodic_cleanup())
    history_task = asyncio.create_task(
        periodic_history_ingestion(int(os.getenv("HISTORY_INGEST_INTERVAL_SECONDS", "3600")))
    )
    alert_worker_task: asyncio.Task[None] | None = None
    if config.alert_evaluation.enabled:
        alert_worker = AlertEvaluationWorker(
            alert_service=alert_subscription_service,
            fetch_current_data=unified_weather_service.get_current_combined_data,
        )
        alert_worker_task = asyncio.create_task(
            alert_worker.run_forever(interval_seconds=config.alert_evaluation.interval_seconds)
        )
    logger.info("Background tasks started")

    try:
        yield
    finally:
        logger.info("Shutting down AirTrace RU Backend...")
        cleanup_task.cancel()
        history_task.cancel()
        tasks_to_cancel = [cleanup_task, history_task]
        if alert_worker_task is not None:
            alert_worker_task.cancel()
            tasks_to_cancel.append(alert_worker_task)
        for task in tasks_to_cancel:
            try:
                await task
            except asyncio.CancelledError:
                pass

        try:
            await degradation_manager.cleanup()
        except Exception as exc:
            logger.warning("Graceful degradation manager cleanup failed: %s", exc)

        try:
            await unified_weather_service.cleanup()
        except Exception as exc:
            logger.warning("Unified weather service cleanup failed: %s", exc)
        finally:
            unified_weather_service.set_current_persistence_callback(None)

        air_quality_service = get_air_quality_service()
        if air_quality_service is not None:
            await air_quality_service.cleanup()

        if config.performance.rate_limiting_enabled:
            try:
                from rate_limit_middleware import get_rate_limit_manager

                await get_rate_limit_manager().cleanup()
            except Exception as exc:
                logger.warning("Rate limiting cleanup failed: %s", exc)

        if config.performance.connection_pooling_enabled:
            try:
                from connection_pool import get_connection_pool_manager

                await get_connection_pool_manager().cleanup()
            except Exception as exc:
                logger.warning("Connection pool cleanup failed: %s", exc)

        set_air_quality_service(None)
        set_alert_subscription_service(None)
        set_history_ingestion_pipeline(None)
        set_history_snapshot_store(None)
        await close_database_runtime()
        logger.info("Shutdown complete")


def _register_exception_handlers(app: FastAPI) -> None:
    from schemas import ErrorResponse

    def _is_v2_request(request: Request) -> bool:
        return request.url.path.startswith("/v2/")

    def _status_to_v2_code(status_code: int) -> str:
        if status_code == 400:
            return "VALIDATION_ERROR"
        if status_code == 401:
            return "UNAUTHORIZED"
        if status_code == 403:
            return "FORBIDDEN"
        if status_code == 404:
            return "NOT_FOUND"
        if status_code == 409:
            return "CONFLICT"
        if status_code == 429:
            return "RATE_LIMIT_EXCEEDED"
        if status_code == 503:
            return "SERVICE_UNAVAILABLE"
        if 400 <= status_code < 500:
            return "VALIDATION_ERROR"
        return "INTERNAL_ERROR"

    def _build_v2_error_response(*, status_code: int, code: str, message: str, details: Any = None) -> UnicodeJSONResponse:
        normalized_details = jsonable_encoder(
            details,
            custom_encoder={
                Exception: lambda value: str(value),
            },
        )
        payload = ErrorResponse(code=code, message=message, details=None).model_dump(mode="json")
        payload["details"] = normalized_details
        return UnicodeJSONResponse(
            status_code=status_code,
            content=payload,
            headers=dict(V2_RESPONSE_HEADERS),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning("HTTP exception: %s - %s", exc.status_code, exc.detail)
        if _is_v2_request(request):
            return _build_v2_error_response(
                status_code=exc.status_code,
                code=_status_to_v2_code(exc.status_code),
                message=exc.detail if isinstance(exc.detail, str) else "Request failed",
                details=None if isinstance(exc.detail, str) else exc.detail,
            )
        error_response = ErrorResponse(code=f"HTTP_{exc.status_code}", message=exc.detail)
        return UnicodeJSONResponse(status_code=exc.status_code, content=error_response.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("Validation exception: %s", exc.errors())
        normalized_details = jsonable_encoder(
            exc.errors(),
            custom_encoder={
                Exception: lambda value: str(value),
            },
        )
        if _is_v2_request(request):
            return _build_v2_error_response(
                status_code=422,
                code="VALIDATION_ERROR",
                message="Request validation failed",
                details=normalized_details,
            )
        return JSONResponse(status_code=422, content={"detail": normalized_details})

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s: %s", type(exc).__name__, exc)
        if _is_v2_request(request):
            return _build_v2_error_response(
                status_code=500,
                code="INTERNAL_ERROR",
                message="Internal server error",
            )
        error_response = ErrorResponse(code="INTERNAL_ERROR", message="Внутренняя ошибка сервера")
        return UnicodeJSONResponse(status_code=500, content=error_response.model_dump(mode="json"))


def create_api_app() -> FastAPI:
    load_cities_config()

    app = FastAPI(
        title="AirTrace RU API",
        description="Air Quality Monitoring API for Russian cities with privacy-first approach",
        version="0.3.1",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        default_response_class=UnicodeJSONResponse,
    )

    @app.middleware("http")
    async def add_charset_header(request: Request, call_next):
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type and "charset" not in content_type:
            response.headers["content-type"] = "application/json; charset=utf-8"
        if request.url.path.startswith("/v2/"):
            for header, value in V2_RESPONSE_HEADERS.items():
                response.headers.setdefault(header, value)
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8080", "https://airtrace.ru"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    privacy_middleware = PrivacyMiddleware(app, enable_request_logging=True)
    app.add_middleware(PrivacyMiddleware, enable_request_logging=True)
    set_privacy_middleware(privacy_middleware)

    if config.performance.rate_limiting_enabled:
        setup_rate_limit_logging()
        setup_rate_limiting(
            app=app,
            enabled=True,
            skip_paths=["/docs", "/redoc", "/openapi.json", "/version"],
            trust_forwarded_headers=config.performance.rate_limit_trust_forwarded_headers,
            trusted_proxy_ips=config.performance.rate_limit_trusted_proxy_ips,
        )
        logger.info("Rate limiting middleware enabled")
    else:
        logger.info("Rate limiting middleware disabled")

    _register_exception_handlers(app)
    app.include_router(v1_readonly_router)
    app.include_router(v2_readonly_router)
    app.include_router(v2_alerts_router)
    app.include_router(legacy_router)
    app.include_router(ops_router)
    return app
