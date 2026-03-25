# ADR 009: Alert Write-Paths After Readonly v2

- Status: Proposed
- Date: 2026-03-25
- Owners: AirTrace RU Team

## Context

Alerts depend on stable location, history, confidence, and delivery behavior.

## Decision

Do not implement alert write-paths until readonly `v2` is stable and backed by the new data model.

## Alternatives Considered

### Build alerts in parallel with readonly `v2`

- Summary: expose write APIs while current/history/trends are still evolving.
- Why not chosen: too likely to force API and schema churn.

## Consequences

- Alert features arrive later.
- The write API can be designed on a stable foundation.

## Deferred Items

- Final alert payload shape
- External notification contract
