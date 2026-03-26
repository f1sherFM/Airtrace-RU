# AirTrace RU Python SDK

Generated Python SDK for the stable AirTrace RU public API v2.

Source of truth: `openapi/airtrace-v2.openapi.json`

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
    history = client.get_history_by_city(city="moscow", sort="desc")
    trends = client.get_trends_by_city(city="moscow", range="7d")
    alerts = client.list_alerts()
    print(health["status"], current.get("aqi", {}), len(history.get("items", [])), trends.get("trend"), len(alerts))
```

## Supported endpoints

- `/v2/health`
- `/v2/current`
- `/v2/forecast`
- `/v2/history`
- `/v2/trends`
- `/v2/alerts`
