# AirTrace v2

`airtrace-v2` is the active integration branch for the new AirTrace core.

This branch already includes:
- stable readonly `v2` API
- historical storage foundation
- trends and explainability
- DB-backed alerts write-paths
- gradual Python SSR migration onto the new application layer

The goal of `v2` is to move AirTrace from a large legacy app to a cleaner modular core without a big bang rewrite.

## Status

- Branch: `airtrace-v2`
- API model: `v1` legacy adapter + stable public `v2`
- Web model: Python SSR on top of the application layer
- Storage: repository-backed history and alerts foundation
- Migration style: staged rollout with regression gates

Completed stages in this branch:
- Stage 0: baseline and safety net
- Stage 1: core refactor
- Stage 2: data layer and history foundation
- Stage 3: stable readonly `v2`
- Stage 4: alerts write-paths
- Stage 4 hardening: alert worker and alert-specific rate limiting
- Stage 5: SSR UX migration

## What Is In v2

### API

- `GET /v2/current`
- `GET /v2/forecast`
- `GET /v2/history`
- `GET /v2/trends`
- `GET /v2/health`
- `POST /v2/alerts`
- `GET /v2/alerts`
- `GET /v2/alerts/{id}`
- `PATCH /v2/alerts/{id}`
- `DELETE /v2/alerts/{id}`

### Web

- city pages rendered through `application/web`
- history and trends pages
- compare cities UI
- alerts settings and subscription flows
- explainability block with source, freshness, and confidence

### Platform

- centralized city config in `config/cities.yaml`
- shared application layer for API and SSR
- Alembic migration chain
- SQLAlchemy-backed repositories
- privacy-safe cache keying
- Redis-backed cache path
- alert evaluation worker inside app lifespan

## Quick Start

### Requirements

- Python 3.10+
- `pip`
- Redis on `localhost:6379` for full local cache behavior
- optional PostgreSQL for DB-backed flows

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Full App

```powershell
python start_app.py
```

After startup:
- API: [http://localhost:8000](http://localhost:8000)
- Web: [http://localhost:3000](http://localhost:3000)
- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

### Run Separately

API:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Web:

```powershell
.\.venv\Scripts\Activate.ps1
cd web
python web_app.py
```

### Redis

If Redis is not already running locally:

```powershell
docker run -d --name airtrace-redis -p 6379:6379 redis:7-alpine
```

## Key Docs

- Roadmap: [docs/airtrace_v2_roadmap.md](docs/airtrace_v2_roadmap.md)
- Public API guide: [docs/public_api_v2.md](docs/public_api_v2.md)
- API compatibility notes: [docs/api_v2_compatibility.md](docs/api_v2_compatibility.md)
- Stage 0 protected surfaces: [docs/stage0_protected_surfaces.md](docs/stage0_protected_surfaces.md)
- Health probes: [docs/health_probes.md](docs/health_probes.md)
- Runtime SLO: [docs/slo_runtime_control.md](docs/slo_runtime_control.md)

## Public OpenAPI

- Public artifact: [openapi/airtrace-v2.openapi.json](openapi/airtrace-v2.openapi.json)

The committed artifact is the source for:
- contract snapshot tests
- in-repo SDK generation
- public `v2` integration docs

## Project Structure

```text
core/              app factory, lifecycle, settings
api/v1/            legacy adapter
api/v2/            public v2 API
application/       query services, use cases, web layer
domain/            AQI, NMU, confidence, pollutants
infrastructure/    DB, repositories, cache, providers
config/            cities.yaml and runtime config
web/               Python SSR app, templates, static assets
docs/              roadmap, ADRs, API and ops docs
tests/             regression, contract, SSR, migration gates
```

## Testing

Focused examples:

```powershell
.\.venv\Scripts\python -m pytest tests\test_v2_contract.py -q
.\.venv\Scripts\python -m pytest tests\test_stage5_ssr_rendering.py -q
```

Full regression is intentionally broader because this branch protects staged migration behavior across API, data layer, alerts, and SSR.

## Branch Notes

- `main` remains the stable production branch
- `airtrace-v2` is the integration branch for the new architecture
- root `README.md` in this branch is intentionally `v2`-first
- legacy context should be taken from branch history and staged docs, not from the old root README

## Next Direction

Current work after Stage 5 is focused on API hardening:
- clearer health semantics
- stronger contract and edge-case tests
- OpenAPI/runtime parity
- continued cleanup of remaining legacy behavior
