# Health Probes

Updated health model for production probes.

## Endpoints

- `/health` and `/v2/health`:
  - Full component health.
  - `status` reflects critical components first.
  - Optional dependency failures (external providers, Redis/cache, pool subcomponents) degrade service instead of marking whole API unhealthy.

- `/health/liveness` and `/v2/liveness`:
  - Process-level probe.
  - Returns `healthy` when API process is alive.

- `/health/readiness` and `/v2/readiness`:
  - Traffic-readiness probe.
  - `ready` when overall service is not `unhealthy`.
  - `degraded` is still considered ready with fallback mode.

## Status semantics

- `healthy`: critical components are healthy.
- `degraded`: critical path is available, but optional dependencies are degraded/unhealthy.
- `unhealthy`: critical components are unhealthy; instance should be considered not ready.
