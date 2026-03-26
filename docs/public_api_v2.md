# Public API v2 Guide

Stable public integration guide for AirTrace RU API v2.

Source of truth:
- `openapi/airtrace-v2.openapi.json`
- `tests/test_v2_contract.py`
- `tests/test_contract_snapshot.py`

## Base URL

Local:

```bash
http://localhost:8000
```

## Quick check

```bash
curl -fsS "http://localhost:8000/v2/health"
```

## Stable readonly endpoints

- `/v2/current`
- `/v2/forecast`
- `/v2/history`
- `/v2/trends`
- `/v2/health`

## Protected write/read alert endpoints

- `POST /v2/alerts`
- `GET /v2/alerts`
- `GET /v2/alerts/{id}`
- `PATCH /v2/alerts/{id}`
- `DELETE /v2/alerts/{id}`

All `/v2/alerts*` routes require the shared alerts API key via `X-API-Key` or `Authorization: Bearer ...`.

## Current air quality

```bash
curl -fsS "http://localhost:8000/v2/current?lat=55.7558&lon=37.6176"
```

## Forecast

```bash
curl -fsS "http://localhost:8000/v2/forecast?lat=55.7558&lon=37.6176&hours=24"
```

## History (city preset)

Newest first:

```bash
curl -fsS "http://localhost:8000/v2/history?range=24h&page=1&page_size=20&sort=desc&city=moscow"
```

Oldest first:

```bash
curl -fsS "http://localhost:8000/v2/history?range=7d&page=1&page_size=20&sort=asc&city=moscow"
```

## History (custom coordinates)

```bash
curl -fsS "http://localhost:8000/v2/history?range=24h&page=1&page_size=20&sort=desc&lat=55.7558&lon=37.6176"
```

## Trends

By city:

```bash
curl -fsS "http://localhost:8000/v2/trends?range=7d&city=moscow"
```

By custom coordinates:

```bash
curl -fsS "http://localhost:8000/v2/trends?range=30d&lat=55.7558&lon=37.6176"
```

## Alert subscriptions

Create a Telegram subscription:

```bash
curl -fsS -X POST "http://localhost:8000/v2/alerts" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${ALERTS_API_KEY}" \
  -H "Idempotency-Key: create-moscow-alert-1" \
  -d '{
    "name": "Moscow AQI >= 140",
    "city": "moscow",
    "aqi_threshold": 140,
    "nmu_levels": ["high", "critical"],
    "cooldown_minutes": 30,
    "channel": "telegram",
    "chat_id": "123456"
  }'
```

List subscriptions:

```bash
curl -fsS "http://localhost:8000/v2/alerts" \
  -H "X-API-Key: ${ALERTS_API_KEY}"
```

## Canonical provenance metadata

For `current`, `forecast`, and history items, `metadata` is the canonical provenance block:

- `data_source`
- `freshness`
- `confidence`
- `confidence_explanation`
- `fallback_used`
- `cache_age_seconds`

Flat mirrors may still appear for compatibility, but clients should read provenance from `metadata`.

## Error model

All `/v2/*` failures use one flat JSON schema:

```json
{
  "code": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": [],
  "timestamp": "2026-03-25T12:00:00+00:00"
}
```

Stable codes in Stage 3:

- `VALIDATION_ERROR`
- `SERVICE_UNAVAILABLE`
- `RATE_LIMIT_EXCEEDED`
- `NOT_FOUND`
- `INTERNAL_ERROR`

## Scope

Stage 4 keeps the air-quality surface readonly, but adds protected alert subscription write-paths under `/v2/alerts*`.

## Migration notes (v1 -> v2)

- Keep existing integrations on v1 routes while migrating incrementally.
- For new integrations, use `/v2/*` routes by default.
- `v1` remains a legacy adapter over the new core.
- `v2` no longer aims for byte-for-byte equality with `v1`; it is the stable public contract.
- `history` adds `sort=asc|desc` in `v2`.
- `trends` is new in `v2`.
- alerts move from legacy `/alerts/rules*` to `/v2/alerts*`.

## Route mapping

- `/weather/current` -> `/v2/current`
- `/weather/forecast` -> `/v2/forecast`
- `/history` -> `/v2/history`
- `/health` -> `/v2/health`
- `/alerts/rules` -> `/v2/alerts`

## Deferred items

- Sparse fieldsets via `fields=...`
- External SDK publication
- Additional alert channels and per-client ownership
