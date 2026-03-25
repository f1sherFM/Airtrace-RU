# ADR 008: Provenance and Confidence Model

- Status: Proposed
- Date: 2026-03-25
- Owners: AirTrace RU Team

## Context

AirTrace v2 needs explainable data quality, not only pollutant values.

## Decision

Model provenance, freshness, fallback-chain usage, and confidence as first-class response and storage concepts.

## Alternatives Considered

### Keep confidence as an incidental response helper

- Summary: compute confidence inline in response assembly only.
- Why not chosen: not reusable enough for history, trends, and alerts.

## Consequences

- More explicit schema and domain logic.
- Better explainability in API and UI.

## Deferred Items

- Advanced probabilistic scoring
