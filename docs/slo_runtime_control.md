# SLO Runtime Control (Issue #23)

## Service Level Objectives

- Availability SLO (API): `99.5%` monthly
- Latency SLO (`/weather/current`, `/weather/forecast`, `/history`):
  - `p95 < 800ms`
  - `p99 < 1500ms`
- Error-rate SLO:
  - `5xx rate < 1.0%` (rolling 5m)

## Error Budget

- Monthly availability error budget: `0.5%`
- Burn-rate alert policy:
  - fast burn: budget consumption > `14x` over 5m
  - slow burn: budget consumption > `2x` over 1h

## Alert Thresholds

- `airtrace_error_rate > 0.01` for 5m => warning
- `airtrace_error_rate > 0.03` for 5m => critical
- `airtrace_request_duration_seconds{quantile="0.95"} > 0.8` for 10m => warning
- `airtrace_request_duration_seconds{quantile="0.99"} > 1.5` for 10m => critical
- `airtrace_external_api_success_rate < 0.90` for 10m => warning
- `airtrace_cache_hit_rate < 0.40` for 15m => warning

## Dashboard Source

- Grafana dashboard JSON:
  - `observability/grafana/airtrace_runtime_overview.json`

## Prometheus Scrape Target

- Metrics endpoint:
  - `GET /metrics/prometheus`
