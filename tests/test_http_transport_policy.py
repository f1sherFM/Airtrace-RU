"""
Tests for unified HTTP transport policy (Issue #22).
"""

import asyncio
from unittest.mock import patch

from http_transport import (
    create_async_client,
    create_external_async_client,
    create_internal_async_client,
    get_transport_policy,
)
from config import config


def test_transport_policy_uses_single_config_source():
    policy = get_transport_policy()
    assert policy.connect_timeout == config.api.connect_timeout
    assert policy.read_timeout == config.api.read_timeout
    assert policy.write_timeout == config.api.write_timeout
    assert policy.pool_timeout == config.api.pool_timeout
    assert policy.max_retries == config.api.max_retries
    assert policy.retry_delay == config.api.retry_delay
    assert policy.backoff_factor == config.api.backoff_factor


def test_transport_client_defaults_to_policy_trust_env_false(monkeypatch):
    previous = config.api.trust_env
    config.api.trust_env = False
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:2080")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:2080")
        client = create_async_client(max_connections=5, max_keepalive_connections=2)
        try:
            # Proxy env must not be consumed when policy trust_env is false.
            assert getattr(client, "_trust_env", True) is False
        finally:
            asyncio.run(client.aclose())
    finally:
        config.api.trust_env = previous


def test_transport_client_can_opt_in_trust_env(monkeypatch):
    previous = config.api.trust_env
    config.api.trust_env = False
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:2080")
        client = create_async_client(max_connections=5, max_keepalive_connections=2, trust_env=True)
        try:
            assert getattr(client, "_trust_env", False) is True
        finally:
            asyncio.run(client.aclose())
    finally:
        config.api.trust_env = previous


def test_internal_transport_defaults_to_trust_env_false(monkeypatch):
    monkeypatch.setenv("INTERNAL_HTTP_TRUST_ENV", "false")
    client = create_internal_async_client()
    try:
        assert getattr(client, "_trust_env", True) is False
    finally:
        asyncio.run(client.aclose())


def test_external_transport_can_enable_trust_env(monkeypatch):
    monkeypatch.setenv("EXTERNAL_HTTP_TRUST_ENV", "true")
    client = create_external_async_client()
    try:
        assert getattr(client, "_trust_env", False) is True
    finally:
        asyncio.run(client.aclose())


def test_internal_transport_timeout_seconds_sets_all_timeouts():
    client = create_internal_async_client(timeout_seconds=7.5)
    try:
        timeout = client.timeout
        assert timeout.connect == 7.5
        assert timeout.read == 7.5
        assert timeout.write == 7.5
        assert timeout.pool == 7.5
    finally:
        asyncio.run(client.aclose())


def test_external_transport_timeout_seconds_sets_all_timeouts():
    client = create_external_async_client(timeout_seconds=12.0)
    try:
        timeout = client.timeout
        assert timeout.connect == 12.0
        assert timeout.read == 12.0
        assert timeout.write == 12.0
        assert timeout.pool == 12.0
    finally:
        asyncio.run(client.aclose())
