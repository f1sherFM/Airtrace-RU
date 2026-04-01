# VPS Deployment Runbook

This runbook is the shortest safe path to deploy `airtrace-v2` on a Linux VPS.

## Stack

- `api`: FastAPI backend
- `web`: Python SSR frontend
- `redis`: cache and invalidation backend
- `db`: PostgreSQL for persistent alerts/history

## Prerequisites

- Ubuntu 24.04 or similar Linux VPS
- Docker Engine with Compose plugin
- Git
- open ports:
  - `80` / `443` for reverse proxy later
  - optional direct `8000` / `3000` during first smoke checks

## 1. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
docker --version
docker compose version
```

## 2. Clone the repo

```bash
git clone https://github.com/f1sherFM/Airtrace-RU.git
cd Airtrace-RU
git checkout airtrace-v2
```

## 3. Create production env

Copy the template:

```bash
cp .env.production.example .env.production
```

At minimum fill in:

- `SENTRY_DSN`
- `SENTRY_ENVIRONMENT=production`
- `SENTRY_RELEASE=airtrace-v2`
- `ALERTS_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `DATABASE_URL`
- `ALEMBIC_DATABASE_URL`

## 4. Start the stack

With PostgreSQL profile enabled:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml --profile with-db up -d --build
```

## 5. Verify health

```bash
docker compose -f docker-compose.prod.yml ps
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:3000/
```

Expected result:

- `api` container healthy
- `web` container healthy
- `redis` healthy
- `db` healthy

## 6. Check alerts and Sentry wiring

```bash
curl -fsS http://localhost:8000/v2/health
docker compose -f docker-compose.prod.yml logs api --tail=100
docker compose -f docker-compose.prod.yml logs web --tail=100
```

If `SENTRY_DSN` is configured, startup and runtime exceptions should appear in Sentry.

## 7. Update deploy

```bash
git pull origin airtrace-v2
docker compose --env-file .env.production -f docker-compose.prod.yml --profile with-db up -d --build
```

## 8. Stop or rollback

```bash
docker compose -f docker-compose.prod.yml down
```

To remove data volumes too:

```bash
docker compose -f docker-compose.prod.yml down -v
```

## Notes

- The current compose profile does not yet include a reverse proxy. For public internet exposure, put Nginx or Caddy in front of `api` and `web`.
- Do not commit `.env.production`.
- First deploy should be verified with manual health checks before DNS cutover.
