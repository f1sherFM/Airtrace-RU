# ADR 005: Python SSR Migration

- Status: Proposed
- Date: 2026-03-25
- Owners: AirTrace RU Team

## Context

The current web layer is already Python SSR and must keep working during the v2 transition.

## Decision

Retain Python SSR and migrate the web layer gradually so that routes move to the new application layer over time.

## Alternatives Considered

### Rewrite immediately to a TypeScript frontend

- Summary: replace the SSR layer with a separate frontend.
- Why not chosen: too disruptive for the current migration scope.

### Keep the current SSR structure unchanged

- Summary: leave business logic in the current web layer.
- Why not chosen: preserves the coupling that v2 is meant to reduce.

## Consequences

- Lower migration risk.
- SSR remains a first-class transport over the new application core.

## Deferred Items

- Full frontend technology re-evaluation
