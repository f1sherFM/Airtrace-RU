"""Background alert evaluation worker for Stage 4 hardening."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from application.repositories.alerts import AlertSubscriptionRecord
from application.services.alerts import AlertSubscriptionService
from core.settings import get_cities_mapping

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_coordinate_key(latitude: float, longitude: float) -> str:
    return f"{round(latitude, 4):.4f},{round(longitude, 4):.4f}"


@dataclass(frozen=True)
class AlertEvaluationGroup:
    key: str
    city_code: Optional[str]
    latitude: float
    longitude: float
    subscription_ids: tuple[str, ...]


@dataclass(frozen=True)
class AlertWorkerCycleResult:
    location_groups: int = 0
    triggered_alerts: int = 0
    failed_fetches: int = 0
    failed_deliveries: int = 0


class AlertEvaluationWorker:
    def __init__(
        self,
        *,
        alert_service: AlertSubscriptionService,
        fetch_current_data: Callable[[float, float], Awaitable[object]],
        now_provider: Callable[[], datetime] = _utc_now,
    ):
        self._alert_service = alert_service
        self._fetch_current_data = fetch_current_data
        self._now_provider = now_provider

    def _resolve_group(self, subscription: AlertSubscriptionRecord) -> Optional[AlertEvaluationGroup]:
        if subscription.city_code:
            payload = get_cities_mapping().get(subscription.city_code)
            if payload is None:
                logger.warning(
                    "Skipping alert subscription %s: unknown city_code=%s",
                    subscription.id,
                    subscription.city_code,
                )
                return None
            return AlertEvaluationGroup(
                key=f"city:{subscription.city_code}",
                city_code=subscription.city_code,
                latitude=float(payload["lat"]),
                longitude=float(payload["lon"]),
                subscription_ids=(subscription.id,),
            )

        if subscription.latitude is not None and subscription.longitude is not None:
            coordinate_key = _normalize_coordinate_key(subscription.latitude, subscription.longitude)
            return AlertEvaluationGroup(
                key=f"coords:{coordinate_key}",
                city_code=None,
                latitude=float(subscription.latitude),
                longitude=float(subscription.longitude),
                subscription_ids=(subscription.id,),
            )

        logger.debug(
            "Skipping subscription %s in background worker because it has no location target",
            subscription.id,
        )
        return None

    async def build_groups(self) -> list[AlertEvaluationGroup]:
        records = await self._alert_service.list_active_subscription_records()
        grouped: dict[str, AlertEvaluationGroup] = {}
        for record in records:
            group = self._resolve_group(record)
            if group is None:
                continue
            existing = grouped.get(group.key)
            if existing is None:
                grouped[group.key] = group
                continue
            grouped[group.key] = AlertEvaluationGroup(
                key=existing.key,
                city_code=existing.city_code,
                latitude=existing.latitude,
                longitude=existing.longitude,
                subscription_ids=existing.subscription_ids + (record.id,),
            )
        return list(grouped.values())

    async def run_cycle(self) -> AlertWorkerCycleResult:
        cycle_started_at = self._now_provider()
        groups = await self.build_groups()
        logger.info("Alert worker cycle started: location_groups=%s", len(groups))

        triggered_alerts = 0
        failed_fetches = 0
        failed_deliveries = 0

        for group in groups:
            logger.info(
                "Alert worker evaluating group: key=%s city_code=%s subscriptions=%s",
                group.key,
                group.city_code,
                len(group.subscription_ids),
            )
            try:
                current = await self._fetch_current_data(group.latitude, group.longitude)
            except Exception as exc:
                failed_fetches += 1
                logger.error(
                    "Alert worker fetch failed for %s (%s, %s): %s",
                    group.key,
                    group.latitude,
                    group.longitude,
                    exc,
                )
                continue

            try:
                events = await self._alert_service.evaluate_conditions(
                    aqi=current.aqi.value,
                    nmu_risk=getattr(current, "nmu_risk", None),
                    lat=group.latitude,
                    lon=group.longitude,
                    city_code=group.city_code,
                    now=cycle_started_at,
                )
                unsuppressed = [event for event in events if not event.suppressed]
                logger.info(
                    "Alert worker evaluated group: key=%s total_events=%s unsuppressed=%s",
                    group.key,
                    len(events),
                    len(unsuppressed),
                )
                if not unsuppressed:
                    continue
                triggered_alerts += len(unsuppressed)
                delivery_results = await self._alert_service.deliver_events(
                    events=unsuppressed,
                    aqi=current.aqi.value,
                    nmu_risk=getattr(current, "nmu_risk", None),
                )
                logger.info(
                    "Alert worker delivery summary: key=%s sent=%s failed=%s",
                    group.key,
                    sum(1 for result in delivery_results if result.status == "sent"),
                    sum(1 for result in delivery_results if result.status != "sent"),
                )
                failed_deliveries += sum(1 for result in delivery_results if result.status != "sent")
            except Exception as exc:
                logger.error("Alert worker evaluation failed for %s: %s", group.key, exc)
                failed_deliveries += 1

        logger.info(
            "Alert worker cycle finished: location_groups=%s triggered_alerts=%s failed_fetches=%s failed_deliveries=%s",
            len(groups),
            triggered_alerts,
            failed_fetches,
            failed_deliveries,
        )
        return AlertWorkerCycleResult(
            location_groups=len(groups),
            triggered_alerts=triggered_alerts,
            failed_fetches=failed_fetches,
            failed_deliveries=failed_deliveries,
        )

    async def run_forever(self, *, interval_seconds: int) -> None:
        logger.info("Alert evaluation worker started with interval=%ss", interval_seconds)
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                await self.run_cycle()
        except asyncio.CancelledError:
            logger.info("Alert evaluation worker shutdown requested")
            return
