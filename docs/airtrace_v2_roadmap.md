# AirTrace v2 Roadmap

## Status

- State: Draft accepted as working artifact
- Integration branch: `airtrace-v2`
- Migration model: new core with `v1` as legacy adapter
- Web strategy: gradual Python SSR migration
- API strategy: readonly `v2` first, write-paths later

## Vision

AirTrace v2 moves the project from a large feature-rich monolith to a modular core with:

- a stable public API
- a real historical data layer
- explainability and provenance
- gradual Python SSR migration without a big bang rewrite

## Architecture Direction

- `v2` is built around a new core.
- `v1` remains as a legacy adapter.
- Python SSR stays, but the web layer is migrated gradually.
- Read-only `v2` is stabilized before any write-paths.
- There must be one source of truth for cities, configuration, texts, and contracts.

## Guardrails

- Do not implement `v2` as a proxy to `v1`.
- Do not start alerts before readonly `v2` is stable.
- Do not rewrite the entire web layer in one commit.
- Do not keep multiple sources of truth for cities, configuration, or texts.
- Do not publish SDKs externally before the schema is stable.

## Success Metrics

- `main.py`: thin bootstrap/composition root, without domain logic and without large routing blocks
- History: at least 30 days of confirmed data with provenance and freshness
- API: OpenAPI 3.1 plus contract tests for `v2`
- Web: business logic moved into `application/query services`
- Encoding: zero regressions
- CI: fast PR pipeline under 10 minutes, full suite separate

## Definition of Done

- The new core is separated from HTTP and SSR.
- `v1` works as a legacy adapter over the new core.
- History and trends are served from the database layer.
- Readonly `v2` is stable and covered by contract tests.
- Alerts are added only after readonly stabilization.
- Python SSR runs on top of the new application layer.
- Encoding, migrations, SSR, and API are protected by automated tests.

## Target Structure

```text
core/              settings, DI, lifecycle, app factory
api/v1/            legacy adapter
api/v2/            new public API
application/       use cases, query services, orchestration
domain/            AQI, NMU, confidence, provenance, pollutants
infrastructure/    repositories, DB, cache, providers, telemetry
config/            cities.yaml, feature flags, runtime config
web/               SSR routing and view composition
docs/adr/          architecture decision records
```

## Branching Policy

- Main v2 integration branch: `airtrace-v2`.
- `main` remains the stable production branch.
- All v2 work is done in short-lived branches created from `airtrace-v2`.
- Branch naming: `v2-stageX-topic`, for example `v2-stage1-core`, `v2-stage2-db`, `v2-stage3-readonly-api`.
- One branch equals one logical deliverable.
- Do not mix large core refactors, DB migrations, and SSR rewrites in one branch unless required.
- All changes land in `airtrace-v2` first after tests pass.
- Only safe, complete, backward-compatible changes go into `main`.
- Incomplete work reaching `main` must be behind feature flags.
- `airtrace-v2` must be synchronized regularly with `main`.
- Each stage must define exit criteria before merge.
- Architecture, schema, and public contract changes must come with ADRs and tests.

## ADR Track

### Goal

Record major architectural decisions explicitly.

### Proposed Structure

```text
docs/adr/
|- README.md
|- 001-modular-monolith.md
|- 002-application-layer.md
|- 003-timescaledb-for-history.md
|- 004-readonly-api-first.md
|- 005-python-ssr-migration.md
|- 006-feature-flags-for-migration.md
|- 007-v1-deprecation-policy.md
|- 008-provenance-and-confidence-model.md
|- 009-alert-write-paths-after-readonly-v2.md
`- templates/
   `- adr-template.md
```

### ADR Format

- Context
- Decision
- Alternatives considered
- Consequences
- Explicitly deferred items

## Stage 0: Baseline and Safety Net

### Goal

Freeze current behavior before refactoring.

### Deliverables

- Baseline inventory of current API and SSR flows
- List of legacy `v1` endpoints that must remain behavior-compatible
- Snapshot of current OpenAPI
- Snapshot of key HTTP responses for critical flows
- Baseline integration tests for `current`, `forecast`, `history`, `health`, and `export`
- Encoding and localization checks
- SSR smoke test pack

### Tests

- API baseline integration tests
- SSR smoke scenarios:
  - main page renders
  - city page renders
  - custom city page renders
  - history block renders
  - export endpoints respond
  - alerts settings page renders
  - error page renders with valid UTF-8 text
- Encoding and localization checks:
  - UTF-8 decode for text assets
  - no BOM in `.py`, `.html`, `.md`, `.js`, `.css`
  - mojibake pattern scan
  - `charset=utf-8` checks in HTML responses

### Exit Criteria

- Regressions are clearly defined.
- Key user flows are protected by tests.
- Current public behavior is captured well enough to refactor safely.

### Risks

- Refactoring without a baseline and then arguing whether behavior changed.

## Stage 1: Core Refactor

### Goal

Create a modular monolith without changing behavior.

### Deliverables

- `core/app_factory.py`
- Centralized bootstrapping moved out of `main.py`
- Typed settings layer
- `config/cities.yaml` as the single source of truth for cities
- Configuration validation at startup
- `api/v1/` for legacy adapters
- `api/v2/` for new API surface
- `application/` layer between API and domain
- Domain extraction from `utils.py`:
  - `domain/aqi/calculator.py`
  - `domain/nmu/detector.py`
  - `domain/confidence/*`
- Initial extraction of SSR business logic into `application/query services`

### Tests

- `v1` parity tests
- startup configuration validation tests
- integration regression tests for baseline flows

### Exit Criteria

- `main.py` is a thin bootstrap/composition root.
- `v1` behaves as before.
- Cities and configuration are centralized.
- API no longer owns domain logic directly.

### Risks

- Moving files around without actually reducing coupling.

## Stage 2: Data Layer

### Goal

Build a historical storage foundation and data quality model.

### Deliverables

- PostgreSQL with TimescaleDB
- Hybrid-compatible schema designed for future hypertables
- Domain/data models:
  - `Location`
  - `AirQualitySnapshot`
  - `DataProvenance`
  - `AlertRule` or `AlertSubscription`
- Alembic migration chain
- Repositories:
  - `HistoryRepository`
  - `AggregationRepository`
  - `LocationRepository`
- Persistence integrated into ingestion path with idempotency
- `QualityScorer`
- `ConfidenceCalculator`
- Freshness, fallback-chain, and provenance rules
- Backfill design for 30 days of history after source validation

### Tests

- migration apply tests
- schema contract tests
- repository integration tests
- read-after-write tests
- backfill idempotency tests

### Exit Criteria

- History is stored and read from the database layer.
- Snapshots, provenance, and confidence are stable and queryable.
- The project is ready for trends and explainability on top of real stored data.

### Risks

- Over-designing alerts too early around an unstable data model.
- Assuming 30-day backfill is feasible before validating source constraints.

## Stage 3: Public API v2, Read-only First

### Goal

Deliver a stable read-only `v2`.

### Deliverables

- OpenAPI 3.1 as a tracked artifact
- `GET /v2/current`
- `GET /v2/history`
- `GET /v2/trends`
- Unified `v2` error model
- Response metadata:
  - provenance
  - confidence
  - freshness
  - fallback indicators
- Versioning middleware
- Reused rate limiting and auth policy for `v2`
- Generated Python and TypeScript SDKs in-repo

### Tests

- contract tests per endpoint
- OpenAPI snapshot diff
- compatibility tests from legacy `v1` to the new core

### Exit Criteria

- Readonly `v2` is stable, documented, and testable.
- `v2` depends on the new core, not legacy routing.
- `v1` continues to work as a legacy adapter.

### Risks

- Adding write-paths too early and freezing a bad model.

## Stage 4: Write-paths and Alerts

### Goal

Add write capabilities only after readonly `v2` is stable.

### Deliverables

- `POST /v2/alerts`
- `GET /v2/alerts`
- `PATCH /v2/alerts/{id}`
- `DELETE /v2/alerts/{id}`
- API key support for protected endpoints
- Audit trail
- Retry and dead-letter strategy
- Idempotency keys for write operations

### Tests

- write-path integration tests
- auth tests
- idempotency tests
- delivery workflow tests

### Exit Criteria

- Alerts are built on top of stable `Location`, `History`, `Confidence`, and `Provenance`.
- Write APIs do not pull legacy assumptions into `v2`.

### Risks

- Building alerts on an unstable data model and re-breaking the API later.

## Stage 5: UX v2, Gradual Python SSR Migration

### Goal

Migrate the web layer gradually without a big bang rewrite.

### Deliverables

- Thin SSR entrypoint
- Web routing and view composition only in the web layer
- Data access moved to `application/query services`
- Progressive migration of pages:
  - city comparison
  - history charts
  - trends
  - explainability block
- Feature flags for route-by-route replacement
- Removal of business logic from templates
- Removal of legacy duplicated sources after migration completes

### Tests

- SSR smoke tests
- rendering tests
- accessibility smoke checks
- localization and encoding regression tests

### Exit Criteria

- The web layer runs on the new application core.
- Templates and SSR handlers no longer hold business logic.
- Migration is completed incrementally without a product freeze.

### Risks

- Mixing UX rewrite with core refactor in the same change set.

## Migration Safety Track

### Goal

Make schema and persistence changes safe by default.

### Deliverables

- migration apply tests against empty DB
- repository/schema contract tests
- app startup smoke test after migrations
- backfill idempotency coverage

### Exit Criteria

- DB changes are reproducible and testable.
- Schema drift is caught before merge.

## Encoding and Localization Track

### Goal

Prevent a return of encoding regressions and broken Russian text.

### Deliverables

- repository-wide UTF-8 checks
- no-BOM checks
- mojibake detection checks
- HTML and JSON encoding checks
- localized content integrity checks for user-facing responses

### Exit Criteria

- Encoding regressions are blocked by CI.
- User-facing Russian text stays valid and readable.

## SSR Smoke Track

### Goal

Keep gradual SSR migration safe and observable.

### Deliverables

- smoke pack for critical pages and flows
- content-type and status validation
- key-content assertions for HTML responses

### Exit Criteria

- Key SSR flows are validated automatically in CI.

## CI Gates

- encoding and localization checks
- migration tests
- `v2` contract tests
- SSR smoke tests
- OpenAPI snapshot diff
- baseline `v1` regression tests

## Recommended Execution Order

1. Stage 0 baseline and safety net
2. ADR track bootstrap
3. Stage 1 core refactor
4. Stage 2 data layer
5. Stage 3 readonly `v2`
6. Stage 4 alerts and write-paths
7. Stage 5 SSR migration

Cross-cutting tracks run throughout:

- migration safety
- encoding and localization
- SSR smoke
- CI gates

## Notes

- This roadmap is a working artifact, not a marketing document.
- Architecture changes should be reflected here when decisions are finalized in ADRs.
- If a stage cannot meet its exit criteria, it is not considered complete.
