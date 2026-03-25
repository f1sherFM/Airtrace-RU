# Stage 0 Protected Surfaces

This document defines the exact runtime surfaces that Stage 0 freezes before any v2 core refactor starts.

## Protected API Routes

The following API routes are considered protected in Stage 0 and must remain behavior-compatible unless a later ADR explicitly changes them:

- `/weather/current`
- `/weather/forecast`
- `/history`
- `/health`
- `/history/export/json`
- `/history/export/csv`
- `/v2/current`
- `/v2/forecast`
- `/v2/history`
- `/v2/health`

## Protected SSR Routes

The following Python SSR routes are considered protected in Stage 0:

- `/`
- `/city/{city_key}`
- `/custom`
- `/alerts/settings`
- `/api/health`
- `/api/history/{city_key}`
- `/export/{city_key}`

## Legacy v1 Protected Surface

For Stage 0 purposes, legacy `v1` means:

- all current non-`/v2` API behavior
- all current web routes served by the Python SSR layer

This definition remains in effect until the migration explicitly replaces legacy routes with new implementations.

## OpenAPI Baseline Policy

Stage 0 verifies that `/openapi.json` remains reachable and still exposes the protected API routes and representative schemas.

Stage 0 does not introduce a committed OpenAPI snapshot artifact yet. Full snapshot diffing is deferred to later stages once the schema lifecycle is tighter.

## Encoding Enforcement Scope

Stage 0 encoding checks are enforced for:

- `main.py`
- `web/web_app.py`
- `web/templates/*.html`
- `docs/airtrace_v2_roadmap.md`
- `docs/adr/**/*.md`

`README.md` is intentionally excluded from strict enforcement in Stage 0 because it contains known legacy mojibake and is treated as deferred cleanup.
