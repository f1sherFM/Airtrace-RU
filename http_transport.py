"""
Unified HTTP transport policy helpers (Issue #22).
"""

from dataclasses import dataclass

from httpx import AsyncClient, Limits, Timeout

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
