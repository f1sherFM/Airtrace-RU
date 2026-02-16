# Incident Runbooks and Rollback

This runbook covers Incident response requirements from Issue #28.

## Severity levels

- `SEV-1`: full API outage or data corruption risk
- `SEV-2`: partial degradation (provider/cache failures, latency spikes)
- `SEV-3`: minor degradation with workaround

## On-call first 10 minutes checklist

- Confirm impact (`/health`, `/metrics`, web UI availability).
- Create incident channel and assign Incident Commander.
- Declare severity (`SEV-1/2/3`) and set update cadence.
- Freeze non-essential deployments.
- Start timeline with exact UTC timestamps.

## Runbook: API outage

Trigger:
- `/health` unavailable or returning non-200 for > 5 minutes.

Actions:
- Check process/container state: `docker compose -f docker-compose.prod.yml ps`
- Review API logs: `docker compose -f docker-compose.prod.yml logs --tail=200 api`
- Restart API service only:
  - `docker compose -f docker-compose.prod.yml restart api`
- Validate:
  - `curl -fsS http://localhost:8000/health`
  - `curl -fsS "http://localhost:8000/weather/current?lat=55.7558&lon=37.6176"`

Escalation:
- If still failing after one restart cycle, escalate to `SEV-1` and execute rollback.

## Runbook: Provider failure (external weather APIs)

Trigger:
- External provider errors > 20% for 5+ minutes.

Actions:
- Check degradation status: `curl -fsS http://localhost:8000/health | jq .components`
- Confirm fallback behavior through `/weather/current` and `/weather/forecast`.
- Temporarily reduce pressure by increasing cache TTL or reducing request burst.
- Keep service in degraded mode if user-facing endpoints remain available.

Escalation:
- If fallback is also failing, escalate to `SEV-1` and rollback.

## Runbook: Cache degradation (Redis)

Trigger:
- Redis healthcheck fails, cache error spikes, or eviction storm.

Actions:
- Inspect Redis container: `docker compose -f docker-compose.prod.yml logs --tail=200 redis`
- Validate Redis directly: `docker compose -f docker-compose.prod.yml exec redis redis-cli ping`
- Restart Redis:
  - `docker compose -f docker-compose.prod.yml restart redis`
- Validate API remains available (L1 fallback):
  - `curl -fsS http://localhost:8000/health`

Escalation:
- If API remains unavailable or error rate remains elevated, execute rollback.

## Verified rollback path

Rollback target:
- Last known good commit or release tag (example: `v4.0.0-rc1`).

Procedure:
1. `git fetch --tags origin`
2. `git checkout <known-good-tag-or-commit>`
3. `docker compose -f docker-compose.prod.yml up -d --build`
4. `curl -fsS http://localhost:8000/health`
5. `curl -fsS http://localhost:3000/`
6. `git checkout main` and prepare hotfix branch after incident stabilization.

Rollback verification checklist:
- API health is 200.
- Web UI loads successfully.
- At least one current weather request succeeds.
- Incident channel updated with rollback timestamp and target version.

## Post-incident checklist

- Add timeline and root cause in incident report.
- Link related issue/PR/commit.
- Add preventive action item with owner and due date.
- Update this runbook if procedure changed.
