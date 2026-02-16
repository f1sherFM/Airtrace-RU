# Unified HTTP Transport Policy (Issue #22)

Single source of transport behavior is defined in `config.api` and applied via `http_transport.py`.

## Policy Scope

- `trust_env` (proxy env usage)
- connect/read/write/pool timeouts
- retry settings (`max_retries`, `retry_delay`, `backoff_factor`)

## Applied Components

- `services.py` direct HTTP client fallback
- `connection_pool.py` pool client creation and recycling
- `web/web_app.py` web-to-backend HTTP client

## Proxy Edge-Case

- Default policy uses `trust_env=false` to prevent broken local proxy env vars from taking down all outbound calls.
- Override is possible via `API_TRANSPORT_TRUST_ENV=true` when environment-managed proxy routing is required.
