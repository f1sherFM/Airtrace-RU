from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from infrastructure.db.base import Base
from infrastructure.db.session import async_session_factory, create_async_engine_from_url
from infrastructure.repositories.sqlalchemy_alerts import (
    SQLAlchemyAlertAuditRepository,
    SQLAlchemyAlertDeliveryAttemptRepository,
    SQLAlchemyAlertIdempotencyRepository,
    SQLAlchemyAlertSubscriptionRepository,
)


@pytest.fixture
async def stage4_session_factory(tmp_path: Path):
    db_path = tmp_path / "stage4_repositories.db"
    engine = create_async_engine_from_url(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield async_session_factory(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_stage4_subscription_repository_crud_and_matching(stage4_session_factory):
    repository = SQLAlchemyAlertSubscriptionRepository(stage4_session_factory)
    global_record = await repository.create_subscription(
        subscription_id="global-1",
        name="Global",
        enabled=True,
        city_code=None,
        latitude=None,
        longitude=None,
        coordinate_key=None,
        aqi_threshold=150,
        nmu_levels=[],
        cooldown_minutes=30,
        quiet_hours_start=None,
        quiet_hours_end=None,
        channel="telegram",
        chat_id="111",
    )
    city_record = await repository.create_subscription(
        subscription_id="city-1",
        name="City",
        enabled=True,
        city_code="moscow",
        latitude=None,
        longitude=None,
        coordinate_key=None,
        aqi_threshold=120,
        nmu_levels=["high"],
        cooldown_minutes=45,
        quiet_hours_start=None,
        quiet_hours_end=None,
        channel="telegram",
        chat_id="222",
    )
    coord_record = await repository.create_subscription(
        subscription_id="coord-1",
        name="Coord",
        enabled=True,
        city_code=None,
        latitude=55.7558,
        longitude=37.6176,
        coordinate_key="55.7558,37.6176",
        aqi_threshold=100,
        nmu_levels=[],
        cooldown_minutes=15,
        quiet_hours_start=None,
        quiet_hours_end=None,
        channel="telegram",
        chat_id="333",
    )

    applicable = await repository.list_applicable_subscriptions(
        city_code="moscow",
        latitude=55.7558,
        longitude=37.6176,
    )
    assert {item.id for item in applicable} == {global_record.id, city_record.id, coord_record.id}

    updated = await repository.update_subscription(
        city_record.id,
        name="City updated",
        enabled=False,
        city_code="moscow",
        latitude=None,
        longitude=None,
        coordinate_key=None,
        aqi_threshold=130,
        nmu_levels=["critical"],
        cooldown_minutes=50,
        quiet_hours_start=22,
        quiet_hours_end=7,
        channel="telegram",
        chat_id="444",
        last_triggered_at=None,
        last_delivery_status="sent",
    )
    assert updated is not None
    assert updated.name == "City updated"
    assert updated.enabled is False
    assert updated.chat_id == "444"

    state = await repository.set_delivery_state(
        global_record.id,
        last_triggered_at=datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc),
        last_delivery_status="sent",
    )
    assert state is not None
    assert state.last_delivery_status == "sent"

    assert await repository.soft_delete_subscription(coord_record.id) is True
    active_ids = {item.id for item in await repository.list_subscriptions(include_deleted=False)}
    assert coord_record.id not in active_ids


@pytest.mark.asyncio
async def test_stage4_delivery_audit_and_idempotency_repositories(stage4_session_factory):
    subscription_repository = SQLAlchemyAlertSubscriptionRepository(stage4_session_factory)
    delivery_repository = SQLAlchemyAlertDeliveryAttemptRepository(stage4_session_factory)
    audit_repository = SQLAlchemyAlertAuditRepository(stage4_session_factory)
    idempotency_repository = SQLAlchemyAlertIdempotencyRepository(stage4_session_factory)

    subscription = await subscription_repository.create_subscription(
        subscription_id="sub-1",
        name="Repo",
        enabled=True,
        city_code="moscow",
        latitude=None,
        longitude=None,
        coordinate_key=None,
        aqi_threshold=100,
        nmu_levels=[],
        cooldown_minutes=30,
        quiet_hours_start=None,
        quiet_hours_end=None,
        channel="telegram",
        chat_id="111",
    )
    attempt = await delivery_repository.record_attempt(
        subscription_id=subscription.id,
        event_id="evt-1",
        channel="telegram",
        status="sent",
        attempts=1,
        retry_count=0,
        error=None,
        provider_response={"status": "sent"},
        dead_lettered=False,
    )
    audit = await audit_repository.record_event(
        subscription_id=subscription.id,
        action="subscription_created",
        payload={"id": subscription.id},
        idempotency_key="idem-1",
    )
    await idempotency_repository.store_record(
        scope="alerts:create",
        idempotency_key="idem-1",
        request_fingerprint="hash-1",
        response_payload={"id": subscription.id},
    )
    stored = await idempotency_repository.get_record(scope="alerts:create", idempotency_key="idem-1")

    assert attempt.subscription_id == subscription.id
    assert audit.subscription_id == subscription.id
    assert stored is not None
    assert stored.request_fingerprint == "hash-1"
