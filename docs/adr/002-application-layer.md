# ADR 002: Application Layer

- Status: Proposed
- Date: 2026-03-25
- Owners: AirTrace RU Team

## Context

HTTP handlers and SSR routes currently own too much orchestration logic.

## Decision

Introduce an `application/` layer for use cases, query services, and orchestration between transport and domain logic.

## Alternatives Considered

### Put orchestration in API modules

- Summary: keep orchestration close to HTTP routes.
- Why not chosen: transport-specific code would remain tightly coupled to business flows.

### Move orchestration directly into domain

- Summary: let domain modules coordinate I/O and external services.
- Why not chosen: domain should remain focused on business rules, not transport and persistence coordination.

## Consequences

- More files and indirection.
- Better testability and reuse across API and SSR.

## Deferred Items

- Full CQRS split
- Separate command bus
