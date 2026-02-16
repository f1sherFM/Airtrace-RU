# AirTrace RU Python SDK (starter)

Minimal Python SDK starter for AirTrace RU API v2.

## Install (local)

```bash
cd sdk/python
pip install -e .
```

## Usage

```python
from airtrace_sdk import AirTraceClient

with AirTraceClient(base_url="http://localhost:8000", retries=2) as client:
    health = client.get_health()
    current = client.get_current(lat=55.7558, lon=37.6176)
    print(health["status"], current.get("aqi", {}))
```

## Reliability behavior

- Retries for timeout/network errors.
- Raises `AirTraceError` on HTTP 4xx/5xx responses.

## Versioning

Starter package version: `0.1.0` in `pyproject.toml`.
