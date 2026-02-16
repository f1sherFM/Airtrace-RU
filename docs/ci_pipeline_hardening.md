# CI Pipeline Hardening

This document describes the CI hardening work from Issue #27.

## Stages

`/.github/workflows/contract-tests.yml` now contains four gated jobs:

- `lint`
- `tests`
- `contract-tests`
- `smoke-tests`

Each stage has `timeout-minutes` to keep CI bounded and predictable.

## Artifacts

Every stage uploads execution reports to GitHub Actions artifacts:

- `lint-report`
- `core-test-report`
- `contract-test-report`
- `smoke-test-report`

## Required checks

For branch protection, configure these required checks:

- `lint`
- `tests`
- `contract-tests`
- `smoke-tests`
