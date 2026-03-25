# Stage 0 Baseline Checklist

This checklist defines the minimum baseline and safety-net work required before major v2 refactoring begins.

Protected surfaces are listed in [`stage0_protected_surfaces.md`](stage0_protected_surfaces.md).

## Baseline Inventory

- [ ] Identify all legacy `v1` endpoints that must remain behavior-compatible.
- [ ] Record current SSR routes and critical user flows.
- [ ] Capture OpenAPI snapshot for the current API.
- [ ] Capture representative response fixtures for key success and degraded paths.

## API Safety Net

- [ ] Add baseline integration tests for `current`, `forecast`, `history`, `health`, and `export`.
- [ ] Define which responses are contract-sensitive and must not drift silently.
- [ ] Decide where OpenAPI snapshot diffs will run in CI.

## SSR Safety Net

- [ ] Add smoke coverage for main page rendering.
- [ ] Add smoke coverage for city page rendering.
- [ ] Add smoke coverage for custom city flow.
- [ ] Add smoke coverage for error page rendering.
- [ ] Add smoke coverage for alert settings page.

## Encoding and Localization

- [ ] Add UTF-8 decode checks for critical application and template files.
- [ ] Add no-BOM checks for text assets.
- [ ] Add mojibake pattern detection for protected files.
- [ ] Validate `charset=utf-8` behavior for HTML responses.
- [ ] Keep `README.md` outside strict Stage 0 enforcement until legacy mojibake cleanup is done separately.

## Migration Readiness

- [ ] Confirm that no core refactor starts without ADR coverage.
- [ ] Confirm the integration branch for v2 is `airtrace-v2`.
- [ ] Confirm branch naming convention and merge policy.

## Exit Criteria

- [ ] Baseline behavior is documented.
- [ ] Critical flows are covered by tests or checklists.
- [ ] Regressions can be detected before large refactors begin.
