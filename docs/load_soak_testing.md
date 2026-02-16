# Load & Soak Testing Suite (Issue #24)

## Scope

- Core API scenarios:
  - `/weather/current`
  - `/weather/forecast`
  - `/history`
  - `/history/export/json`

## Runner

- Script: `tools/load/run_load_suite.py`
- Example:
```bash
python tools/load/run_load_suite.py --base-url http://127.0.0.1:8000 --requests 300 --concurrency 30
```

## Output

- JSON report: `reports/load/load_suite_result.json`
- Includes:
  - `p50/p95/p99` latency
  - success/failure counts
  - error rate

## Error Budget Guidance

- Warning: `error_rate > 0.01`
- Critical: `error_rate > 0.03`

## Degradation Scenarios

- Run with upstream weather provider disabled or rate-limited.
- Run with Redis unavailable (fallback mode).
- Run with reduced CPU/memory limits to validate graceful degradation behavior.
