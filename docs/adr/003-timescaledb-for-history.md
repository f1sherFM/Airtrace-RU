# ADR 003: TimescaleDB for History

- Status: Proposed
- Date: 2026-03-25
- Owners: AirTrace RU Team

## Context

AirTrace v2 needs durable history, aggregation, and trends over time-series data.

## Decision

Use PostgreSQL with TimescaleDB, starting with a schema compatible with both regular tables and future hypertable conversion.

## Alternatives Considered

### Plain PostgreSQL only

- Summary: use regular Postgres tables and avoid Timescale-specific features.
- Why not chosen: weaker fit for long-term time-series aggregation goals.

### Dedicated time-series database outside PostgreSQL

- Summary: choose a separate TSDB product.
- Why not chosen: increases operational overhead and weakens compatibility with the existing stack.

## Consequences

- Better fit for trends and historical aggregation.
- Requires migration discipline and DB operational knowledge.

## Deferred Items

- Compression policies
- Retention automation
