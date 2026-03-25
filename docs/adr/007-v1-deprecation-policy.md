# ADR 007: v1 Deprecation Policy

- Status: Proposed
- Date: 2026-03-25
- Owners: AirTrace RU Team

## Context

`v1` must remain available during v2 rollout but should not continue as a parallel product line indefinitely.

## Decision

Treat `v1` as a legacy adapter over the new core and define its deprecation policy explicitly once readonly `v2` is stable.

## Alternatives Considered

### Keep `v1` and `v2` indefinitely equal

- Summary: support both versions as peer APIs.
- Why not chosen: doubles maintenance burden and slows the migration.

## Consequences

- Clients need a clear migration path.
- Internal code should stop treating `v1` as the primary interface.

## Deferred Items

- Concrete deprecation dates
