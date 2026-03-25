# ADR 004: Readonly API First

- Status: Proposed
- Date: 2026-03-25
- Owners: AirTrace RU Team

## Context

Write-paths create long-lived contracts quickly and are hard to revise once clients depend on them.

## Decision

Ship read-only `v2` endpoints first and defer write-paths until the data model and readonly contract are stable.

## Alternatives Considered

### Build alerts and write-paths early

- Summary: expose subscriptions and write APIs together with the first `v2` endpoints.
- Why not chosen: too likely to lock in an unstable data model.

## Consequences

- Slower path to alert creation features.
- Better contract stability and fewer breaking revisions.

## Deferred Items

- Alert write API
- External SDK publication
