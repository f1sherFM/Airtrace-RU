# ADR 006: Feature Flags for Migration

- Status: Proposed
- Date: 2026-03-25
- Owners: AirTrace RU Team

## Context

Large migration steps must not require a single cutover event.

## Decision

Use feature flags for unfinished or partially migrated flows that need to coexist safely with current behavior.

## Alternatives Considered

### Big bang cutover

- Summary: switch all traffic to the new implementation at once.
- Why not chosen: too risky for SSR and API migration.

## Consequences

- More configuration complexity.
- Safer rollout and easier rollback.

## Deferred Items

- Remote flag management platform
