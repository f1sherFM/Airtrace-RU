# Production Docker Profile

This guide documents the production compose profile introduced in Issue #26.

## What is included

- `api`: FastAPI backend on `:8000`
- `web`: server-rendered web UI on `:3000`
- `redis`: cache/state backend required by API
- `db`: PostgreSQL service available in optional profile `with-db`

All services use `restart: unless-stopped` and have healthchecks.

## Start (standard)

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

Set production env first. At minimum:

- `SENTRY_DSN`
- `SENTRY_ENVIRONMENT`
- `SENTRY_RELEASE`
- `ALERTS_API_KEY`
- `TELEGRAM_BOT_TOKEN`

## Start with DB profile

```bash
docker compose -f docker-compose.prod.yml --profile with-db up -d --build
```

## Verify health

```bash
docker compose -f docker-compose.prod.yml ps
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:3000/
```

## Stop

```bash
docker compose -f docker-compose.prod.yml down
```

To remove volumes too:

```bash
docker compose -f docker-compose.prod.yml down -v
```

## Related docs

- `docs/vps_deployment_runbook.md`
- `.env.production.example`
- `Dockerfile.api`
