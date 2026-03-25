"""API security helpers shared by legacy routes."""

from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)


def _get_alert_delivery_api_keys() -> list[str]:
    keys: list[str] = []
    legacy_key = os.getenv("ALERTS_API_KEY", "").strip()
    if legacy_key:
        keys.append(legacy_key)

    for raw in os.getenv("ALERTS_API_KEYS", "").split(","):
        candidate = raw.strip()
        if candidate and candidate not in keys:
            keys.append(candidate)
    return keys


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1].strip()
        return token or None
    return None


async def require_alert_delivery_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> None:
    keys = _get_alert_delivery_api_keys()
    if not keys:
        logger.error("Alert delivery auth is not configured: set ALERTS_API_KEY or ALERTS_API_KEYS")
        raise HTTPException(status_code=503, detail="Alert delivery authentication is not configured")

    provided = (x_api_key or "").strip() or (_extract_bearer_token(authorization) or "")
    if not provided:
        raise HTTPException(
            status_code=401,
            detail="Missing alert delivery API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not any(secrets.compare_digest(provided, expected) for expected in keys):
        raise HTTPException(
            status_code=401,
            detail="Invalid alert delivery API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
