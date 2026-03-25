"""FastAPI app factory for the Stage 1 modular monolith."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.legacy import router as legacy_router
from api.ops import router as ops_router
from api.v1.readonly import router as v1_readonly_router
from api.v2.readonly import router as v2_readonly_router
from config import config
from core.legacy_runtime import (
    get_air_quality_service,
    set_air_quality_service,
    set_history_ingestion_pipeline,
    set_history_snapshot_store,
)
from core.settings import load_cities_config
from graceful_degradation import get_graceful_degradation_manager
from history_ingestion import (
    HistoryIngestionPipeline,
    InMemoryHistoricalSnapshotStore,
    JsonlDeadLetterSink,
)
from middleware import PrivacyMiddleware, set_privacy_middleware, setup_privacy_logging
from rate_limit_middleware import setup_rate_limiting
from rate_limit_monitoring import setup_rate_limit_logging
from services import AirQualityService
from unified_weather_service import unified_weather_service

setup_privacy_logging()
logger = logging.getLogger(__name__)


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

    history_store = InMemoryHistoricalSnapshotStore()
    set_history_snapshot_store(history_store)
    history_ingestion_pipeline = HistoryIngestionPipeline(
        fetch_current_data=unified_weather_service.get_current_combined_data,
        snapshot_store=history_store,
        dead_letter_sink=JsonlDeadLetterSink("logs/history_dead_letter.jsonl"),
        max_retries=int(os.getenv("HISTORY_INGEST_MAX_RETRIES", "3")),
        retry_delay_seconds=float(os.getenv("HISTORY_INGEST_RETRY_DELAY_SECONDS", "0.5")),
    )
    set_history_ingestion_pipeline(history_ingestion_pipeline)

    cleanup_task = asyncio.create_task(periodic_cleanup())
    history_task = asyncio.create_task(
        periodic_history_ingestion(int(os.getenv("HISTORY_INGEST_INTERVAL_SECONDS", "3600")))
    )
    logger.info("Background tasks started")

    try:
        yield
    finally:
        logger.info("Shutting down AirTrace RU Backend...")
        cleanup_task.cancel()
        history_task.cancel()
        for task in (cleanup_task, history_task):
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
        set_history_ingestion_pipeline(None)
        set_history_snapshot_store(None)
        logger.info("Shutdown complete")


def _register_exception_handlers(app: FastAPI) -> None:
    from schemas import ErrorResponse

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning("HTTP exception: %s - %s", exc.status_code, exc.detail)
        error_response = ErrorResponse(code=f"HTTP_{exc.status_code}", message=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=error_response.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s: %s", type(exc).__name__, exc)
        error_response = ErrorResponse(code="INTERNAL_ERROR", message="Внутренняя ошибка сервера")
        return JSONResponse(status_code=500, content=error_response.model_dump(mode="json"))


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
    app.include_router(legacy_router)
    app.include_router(ops_router)
    return app
