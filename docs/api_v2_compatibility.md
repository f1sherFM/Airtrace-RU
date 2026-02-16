# API v2 Compatibility Matrix (Issue #10)

This document describes compatibility between legacy routes and the versioned `/v2` namespace.

## Route Matrix

| Legacy route | v2 route | Method | Response model | Compatibility status |
|---|---|---|---|---|
| `/weather/current` | `/v2/current` | `GET` | `AirQualityData` | Compatible alias |
| `/weather/forecast` | `/v2/forecast` | `GET` | `list[AirQualityData]` | Compatible alias |
| `/history` | `/v2/history` | `GET` | `HistoryQueryResponse` | Compatible alias |
| `/health` | `/v2/health` | `GET` | `HealthCheckResponse` | Compatible alias |

## Compatibility Guarantees

- Parameter contracts are preserved between legacy and `/v2` aliases.
- Response schema is preserved for both namespaces.
- Provenance fields remain available in both namespaces.
- Health component contract is normalized to `{status, details}` with status in `healthy/degraded/unhealthy`.

## Legacy Deprecation Notes

- Legacy routes remain operational for backward compatibility in v4.x.
- New client integrations should prefer `/v2/*` routes.
- No removal date is announced in v4.0.0; any removal will require a separate deprecation announcement and migration window.
