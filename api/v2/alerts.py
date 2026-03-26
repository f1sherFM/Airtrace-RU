"""Stable v2 alert subscription routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status

from api.security import require_alert_delivery_auth
from application.queries.v2_readonly import V2_RESPONSE_HEADERS
from core.legacy_runtime import get_alert_subscription_service
from schemas import (
    AlertSubscription,
    AlertSubscriptionCreate,
    AlertSubscriptionDeleteResponse,
    AlertSubscriptionUpdate,
    ErrorResponse,
)

router = APIRouter()

V2_ALERT_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def _apply_v2_headers(response: Response) -> None:
    for header, value in V2_RESPONSE_HEADERS.items():
        response.headers[header] = value


def _map_service_error(exc: ValueError) -> HTTPException:
    message = str(exc)
    if "Idempotency-Key" in message:
        return HTTPException(status_code=409, detail=message)
    return HTTPException(status_code=422, detail=message)


@router.post(
    "/v2/alerts",
    response_model=AlertSubscription,
    responses=V2_ALERT_ERROR_RESPONSES,
    status_code=status.HTTP_201_CREATED,
)
async def create_alert_subscription_v2(
    response: Response,
    payload: AlertSubscriptionCreate,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    _auth: None = Depends(require_alert_delivery_auth),
):
    _apply_v2_headers(response)
    service = get_alert_subscription_service()
    try:
        return await service.create_subscription(payload, idempotency_key=idempotency_key)
    except ValueError as exc:
        raise _map_service_error(exc)


@router.get("/v2/alerts", response_model=list[AlertSubscription], responses=V2_ALERT_ERROR_RESPONSES)
async def list_alert_subscriptions_v2(
    response: Response,
    _auth: None = Depends(require_alert_delivery_auth),
):
    _apply_v2_headers(response)
    return await get_alert_subscription_service().list_subscriptions()


@router.get("/v2/alerts/{subscription_id}", response_model=AlertSubscription, responses=V2_ALERT_ERROR_RESPONSES)
async def get_alert_subscription_v2(
    subscription_id: str,
    response: Response,
    _auth: None = Depends(require_alert_delivery_auth),
):
    _apply_v2_headers(response)
    subscription = await get_alert_subscription_service().get_subscription(subscription_id)
    if subscription is None:
        raise HTTPException(status_code=404, detail="Alert subscription not found")
    return subscription


@router.patch("/v2/alerts/{subscription_id}", response_model=AlertSubscription, responses=V2_ALERT_ERROR_RESPONSES)
async def update_alert_subscription_v2(
    subscription_id: str,
    response: Response,
    payload: AlertSubscriptionUpdate,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    _auth: None = Depends(require_alert_delivery_auth),
):
    _apply_v2_headers(response)
    service = get_alert_subscription_service()
    try:
        updated = await service.update_subscription(subscription_id, payload, idempotency_key=idempotency_key)
    except ValueError as exc:
        raise _map_service_error(exc)
    if updated is None:
        raise HTTPException(status_code=404, detail="Alert subscription not found")
    return updated


@router.delete("/v2/alerts/{subscription_id}", response_model=AlertSubscriptionDeleteResponse, responses=V2_ALERT_ERROR_RESPONSES)
async def delete_alert_subscription_v2(
    subscription_id: str,
    response: Response,
    _auth: None = Depends(require_alert_delivery_auth),
):
    _apply_v2_headers(response)
    deleted = await get_alert_subscription_service().delete_subscription(subscription_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Alert subscription not found")
    return AlertSubscriptionDeleteResponse(deleted=True, id=subscription_id)
