# Public API v2 Guide

Updated public integration guide for AirTrace RU API v2 (Issue #30).

## Base URL

Local:

```bash
http://localhost:8000
```

## Quick check

```bash
curl -fsS "http://localhost:8000/v2/health"
```

## Current air quality

```bash
curl -fsS "http://localhost:8000/v2/current?lat=55.7558&lon=37.6176"
```

## Forecast

```bash
curl -fsS "http://localhost:8000/v2/forecast?lat=55.7558&lon=37.6176"
```

## History (city preset)

```bash
curl -fsS "http://localhost:8000/v2/history?range=24h&page=1&page_size=20&city=moscow"
```

## History (custom coordinates)

```bash
curl -fsS "http://localhost:8000/v2/history?range=24h&page=1&page_size=20&lat=55.7558&lon=37.6176"
```

## Verified curl snippets

The snippets above are intentionally aligned with executable API contract paths:

- `/v2/current`
- `/v2/forecast`
- `/v2/history`
- `/v2/health`

Contract references:

- `tests/test_v2_contract.py`
- `tests/test_contract_snapshot.py`

## Migration notes (v1 -> v2)

- Keep existing integrations on v1 routes while migrating incrementally.
- For new integrations, use `/v2/*` routes by default.
- Core query parameters are preserved for `current`, `forecast`, and `history`.
- Health contract is normalized to `{status, details}` in v2.
- v1 and v2 currently return compatible payloads for core routes.

### Route mapping

- `/weather/current` -> `/v2/current`
- `/weather/forecast` -> `/v2/forecast`
- `/history` -> `/v2/history`
- `/health` -> `/v2/health`
