# ADR 001: Modular Monolith

- Status: Proposed
- Date: 2026-03-25
- Owners: AirTrace RU Team

## Context

AirTrace currently concentrates significant bootstrapping, routing, and orchestration into a small set of large files.

## Decision

Adopt a modular monolith structure with explicit layers for `core`, `api`, `application`, `domain`, and `infrastructure`.

## Alternatives Considered

### Keep the current structure

- Summary: continue evolving the current monolithic layout.
- Why not chosen: coupling and file size are already too high for the v2 scope.

### Split immediately into microservices

- Summary: create separate deployable services for API, ingestion, and web.
- Why not chosen: too much operational complexity for the current migration stage.

## Consequences

- Refactoring cost increases in the short term.
- Architectural boundaries become explicit.
- Future extraction into services remains possible.

## Deferred Items

- Service extraction
- Multi-repository split
