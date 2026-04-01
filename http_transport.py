"""Unified HTTP transport policy helpers (Issue #22)."""

from dataclasses import dataclass
import os
from typing import Any

from httpx import AsyncBaseTransport, AsyncClient, Limits, Timeout

from config import config


@dataclass(frozen=True)
class TransportPolicy:
    trust_env: bool
    connect_timeout: float
    read_timeout: float
    write_timeout: float
    pool_timeout: float
    max_retries: int
    retry_delay: float
    backoff_factor: float


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def get_transport_policy() -> TransportPolicy:
    api = config.api
    return TransportPolicy(
        trust_env=api.trust_env,
        connect_timeout=api.connect_timeout,
        read_timeout=api.read_timeout,
        write_timeout=api.write_timeout,
        pool_timeout=api.pool_timeout,
        max_retries=api.max_retries,
        retry_delay=api.retry_delay,
        backoff_factor=api.backoff_factor,
    )


def get_internal_transport_policy() -> TransportPolicy:
    return TransportPolicy(
        trust_env=_env_bool("INTERNAL_HTTP_TRUST_ENV", False),
        connect_timeout=_env_float("INTERNAL_HTTP_CONNECT_TIMEOUT", 5.0),
        read_timeout=_env_float("INTERNAL_HTTP_READ_TIMEOUT", 10.0),
        write_timeout=_env_float("INTERNAL_HTTP_WRITE_TIMEOUT", 10.0),
        pool_timeout=_env_float("INTERNAL_HTTP_POOL_TIMEOUT", 5.0),
        max_retries=0,
        retry_delay=0.0,
        backoff_factor=1.0,
    )


def get_external_transport_policy() -> TransportPolicy:
    policy = get_transport_policy()
    return TransportPolicy(
        trust_env=_env_bool("EXTERNAL_HTTP_TRUST_ENV", policy.trust_env),
        connect_timeout=_env_float("EXTERNAL_HTTP_CONNECT_TIMEOUT", policy.connect_timeout),
        read_timeout=_env_float("EXTERNAL_HTTP_READ_TIMEOUT", policy.read_timeout),
        write_timeout=_env_float("EXTERNAL_HTTP_WRITE_TIMEOUT", policy.write_timeout),
        pool_timeout=_env_float("EXTERNAL_HTTP_POOL_TIMEOUT", policy.pool_timeout),
        max_retries=policy.max_retries,
        retry_delay=policy.retry_delay,
        backoff_factor=policy.backoff_factor,
    )


def build_timeout(
    *,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    write_timeout: float | None = None,
    pool_timeout: float | None = None,
) -> Timeout:
    policy = get_transport_policy()
    return Timeout(
        connect=connect_timeout if connect_timeout is not None else policy.connect_timeout,
        read=read_timeout if read_timeout is not None else policy.read_timeout,
        write=write_timeout if write_timeout is not None else policy.write_timeout,
        pool=pool_timeout if pool_timeout is not None else policy.pool_timeout,
    )


def build_limits(
    *,
    max_connections: int,
    max_keepalive_connections: int,
) -> Limits:
    return Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
    )


def create_async_client(
    *,
    max_connections: int,
    max_keepalive_connections: int,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    write_timeout: float | None = None,
    pool_timeout: float | None = None,
    trust_env: bool | None = None,
    ) -> AsyncClient:
    policy = get_transport_policy()
    return AsyncClient(
        limits=build_limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        ),
        timeout=build_timeout(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
            pool_timeout=pool_timeout,
        ),
        trust_env=policy.trust_env if trust_env is None else trust_env,
    )


def _create_client_for_policy(
    *,
    policy: TransportPolicy,
    max_connections: int,
    max_keepalive_connections: int,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    write_timeout: float | None = None,
    pool_timeout: float | None = None,
    timeout_seconds: float | None = None,
    trust_env: bool | None = None,
    base_url: str | None = None,
    transport: AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> AsyncClient:
    if timeout_seconds is not None:
        connect_timeout = read_timeout = write_timeout = pool_timeout = timeout_seconds

    return AsyncClient(
        limits=build_limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        ),
        timeout=Timeout(
            connect=connect_timeout if connect_timeout is not None else policy.connect_timeout,
            read=read_timeout if read_timeout is not None else policy.read_timeout,
            write=write_timeout if write_timeout is not None else policy.write_timeout,
            pool=pool_timeout if pool_timeout is not None else policy.pool_timeout,
        ),
        trust_env=policy.trust_env if trust_env is None else trust_env,
        base_url=base_url or "",
        transport=transport,
        **kwargs,
    )


def create_internal_async_client(
    *,
    max_connections: int = 10,
    max_keepalive_connections: int = 5,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    write_timeout: float | None = None,
    pool_timeout: float | None = None,
    timeout_seconds: float | None = None,
    trust_env: bool | None = None,
    base_url: str | None = None,
    transport: AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> AsyncClient:
    return _create_client_for_policy(
        policy=get_internal_transport_policy(),
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
        pool_timeout=pool_timeout,
        timeout_seconds=timeout_seconds,
        trust_env=trust_env,
        base_url=base_url,
        transport=transport,
        **kwargs,
    )


def create_external_async_client(
    *,
    max_connections: int = 20,
    max_keepalive_connections: int = 10,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    write_timeout: float | None = None,
    pool_timeout: float | None = None,
    timeout_seconds: float | None = None,
    trust_env: bool | None = None,
    base_url: str | None = None,
    transport: AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> AsyncClient:
    return _create_client_for_policy(
        policy=get_external_transport_policy(),
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
        pool_timeout=pool_timeout,
        timeout_seconds=timeout_seconds,
        trust_env=trust_env,
        base_url=base_url,
        transport=transport,
        **kwargs,
    )
