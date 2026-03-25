# Architecture Decision Records

This directory stores Architecture Decision Records for AirTrace v2.

## Purpose

ADRs capture major technical decisions that affect structure, migration strategy, public contracts, or operational behavior.

They are intended to make large changes explicit and reviewable.

## When to Write an ADR

Write an ADR when a change affects one or more of the following:

- system architecture
- public API contracts
- storage model or schema strategy
- migration policy
- deployment/runtime behavior
- feature-flag or rollout strategy

## Required Sections

Each ADR should contain:

- Context
- Decision
- Alternatives considered
- Consequences
- Deferred items

## Status Values

Use one of these status values near the top of the ADR:

- Proposed
- Accepted
- Superseded
- Deprecated

## Naming

- ADR files use zero-padded numeric prefixes.
- Example: `001-modular-monolith.md`

## Workflow

1. Create a new file from [`templates/adr-template.md`](templates/adr-template.md).
2. Fill in the decision and alternatives before implementation starts.
3. Mark the ADR as `Accepted` when the decision is approved.
4. If a later ADR replaces it, link both records and mark the old ADR as `Superseded`.

## Initial v2 ADR Set

- `001-modular-monolith.md`
- `002-application-layer.md`
- `003-timescaledb-for-history.md`
- `004-readonly-api-first.md`
- `005-python-ssr-migration.md`
- `006-feature-flags-for-migration.md`
- `007-v1-deprecation-policy.md`
- `008-provenance-and-confidence-model.md`
- `009-alert-write-paths-after-readonly-v2.md`
