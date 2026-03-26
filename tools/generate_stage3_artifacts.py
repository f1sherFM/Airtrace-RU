"""Generate Stage 3 public OpenAPI artifact and in-repo SDKs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as app_main

from api.v2.openapi import PUBLIC_V2_OPENAPI_PATH, build_public_v2_openapi


PYTHON_CLIENT_PATH = ROOT / "sdk" / "python" / "src" / "airtrace_sdk" / "client.py"
PYTHON_INIT_PATH = ROOT / "sdk" / "python" / "src" / "airtrace_sdk" / "__init__.py"
PYTHON_README_PATH = ROOT / "sdk" / "python" / "README.md"
JS_CLIENT_PATH = ROOT / "sdk" / "js" / "src" / "index.ts"
JS_README_PATH = ROOT / "sdk" / "js" / "README.md"
PYTHON_EXAMPLE_PATH = ROOT / "examples" / "python_sdk_example.py"
JS_EXAMPLE_PATH = ROOT / "examples" / "js_sdk_example.mjs"


def _load_spec() -> dict:
    return build_public_v2_openapi(app_main.app)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_supported_paths(spec: dict) -> None:
    required = {
        "/v2/current",
        "/v2/forecast",
        "/v2/history",
        "/v2/trends",
        "/v2/health",
        "/v2/alerts",
        "/v2/alerts/{subscription_id}",
    }
    available = set(spec.get("paths", {}).keys())
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"Public v2 OpenAPI is missing required paths: {missing}")


def _generate_python_client() -> str:
    return '''from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass
class AirTraceError(Exception):
    message: str
    status_code: int
    payload: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"{self.message} (status={self.status_code})"


class AirTraceClient:
    """Generated from openapi/airtrace-v2.openapi.json."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        retries: int = 2,
        retry_delay: float = 0.3,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(0, retries)
        self.retry_delay = max(0.0, retry_delay)
        self._client = httpx.Client(timeout=self.timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AirTraceClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        for attempt in range(self.retries + 1):
            try:
                response = self._client.request(method, url, params=params or {}, json=json_body, headers=headers)
                if response.status_code >= 400:
                    payload: Optional[Dict[str, Any]]
                    try:
                        payload = response.json()
                    except Exception:
                        payload = None
                    raise AirTraceError("AirTrace API error", response.status_code, payload)
                return response.json()
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt == self.retries:
                    raise
                time.sleep(self.retry_delay)
            except AirTraceError:
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unexpected SDK request flow")

    def get_health(self) -> Any:
        return self._request("GET", "/v2/health")

    def get_current(self, *, lat: float, lon: float) -> Any:
        return self._request("GET", "/v2/current", {"lat": lat, "lon": lon})

    def get_forecast(self, *, lat: float, lon: float, hours: int = 24) -> Any:
        return self._request("GET", "/v2/forecast", {"lat": lat, "lon": lon, "hours": hours})

    def get_history(
        self,
        *,
        range: str = "24h",
        page: int = 1,
        page_size: int = 50,
        sort: str = "desc",
        city: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Any:
        params: Dict[str, Any] = {
            "range": range,
            "page": page,
            "page_size": page_size,
            "sort": sort,
        }
        if city is not None:
            params["city"] = city
        if lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        return self._request("GET", "/v2/history", params)

    def get_history_by_city(
        self,
        *,
        city: str,
        range: str = "24h",
        page: int = 1,
        page_size: int = 50,
        sort: str = "desc",
    ) -> Any:
        return self.get_history(city=city, range=range, page=page, page_size=page_size, sort=sort)

    def get_trends(self, *, range: str = "7d", city: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None) -> Any:
        params: Dict[str, Any] = {"range": range}
        if city is not None:
            params["city"] = city
        if lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        return self._request("GET", "/v2/trends", params)

    def get_trends_by_city(self, *, city: str, range: str = "7d") -> Any:
        return self.get_trends(city=city, range=range)

    def list_alerts(self) -> Any:
        return self._request("GET", "/v2/alerts")

    def get_alert(self, *, subscription_id: str) -> Any:
        return self._request("GET", f"/v2/alerts/{subscription_id}")

    def create_alert(self, *, payload: Dict[str, Any], idempotency_key: Optional[str] = None) -> Any:
        return self._request("POST", "/v2/alerts", json_body=payload, idempotency_key=idempotency_key)

    def update_alert(self, *, subscription_id: str, payload: Dict[str, Any], idempotency_key: Optional[str] = None) -> Any:
        return self._request("PATCH", f"/v2/alerts/{subscription_id}", json_body=payload, idempotency_key=idempotency_key)

    def delete_alert(self, *, subscription_id: str) -> Any:
        return self._request("DELETE", f"/v2/alerts/{subscription_id}")
'''


def _generate_python_init() -> str:
    return '''from .client import AirTraceClient, AirTraceError

__all__ = ["AirTraceClient", "AirTraceError"]
'''


def _generate_python_readme() -> str:
    return """# AirTrace RU Python SDK

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
"""


def _generate_js_client() -> str:
    return """export type Coordinates = {
  lat: number;
  lon: number;
};

export type AirTraceClientOptions = {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
  retries?: number;
};

export type AirTraceErrorPayload = {
  code?: string;
  message?: string;
  details?: unknown;
  [key: string]: unknown;
};

export class AirTraceError extends Error {
  status: number;
  payload?: AirTraceErrorPayload;

  constructor(message: string, status: number, payload?: AirTraceErrorPayload) {
    super(message);
    this.name = "AirTraceError";
    this.status = status;
    this.payload = payload;
  }
}

type RequestInitWithTimeout = RequestInit & { timeoutMs?: number };

async function fetchWithTimeout(url: string, init: RequestInitWithTimeout): Promise<Response> {
  const controller = new AbortController();
  const timeoutMs = init.timeoutMs ?? 10_000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export class AirTraceClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeoutMs: number;
  private retries: number;

  constructor(options: AirTraceClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? "http://localhost:8000";
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 10_000;
    this.retries = options.retries ?? 2;
  }

  private async request<T>(
    method: string,
    path: string,
    params: Record<string, string>,
    body?: unknown,
    idempotencyKey?: string,
  ): Promise<T> {
    const url = new URL(`${this.baseUrl}${path}`);
    Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value));

    let lastError: unknown;
    for (let attempt = 0; attempt <= this.retries; attempt++) {
      try {
        const headers: Record<string, string> = {};
        if (this.apiKey) headers["X-API-Key"] = this.apiKey;
        if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
        if (body !== undefined) headers["Content-Type"] = "application/json";
        const response = await fetchWithTimeout(url.toString(), {
          method,
          timeoutMs: this.timeoutMs,
          headers,
          body: body === undefined ? undefined : JSON.stringify(body),
        });
        if (!response.ok) {
          let payload: AirTraceErrorPayload | undefined;
          try {
            payload = (await response.json()) as AirTraceErrorPayload;
          } catch {
            payload = undefined;
          }
          throw new AirTraceError(`HTTP ${response.status} for ${path}`, response.status, payload);
        }
        return (await response.json()) as T;
      } catch (error) {
        lastError = error;
        if (attempt === this.retries) {
          throw error;
        }
      }
    }
    throw lastError;
  }

  getHealth(): Promise<unknown> {
    return this.request("GET", "/v2/health", {});
  }

  getCurrent(coords: Coordinates): Promise<unknown> {
    return this.request("GET", "/v2/current", {
      lat: String(coords.lat),
      lon: String(coords.lon),
    });
  }

  getForecast(coords: Coordinates, hours = 24): Promise<unknown> {
    return this.request("GET", "/v2/forecast", {
      lat: String(coords.lat),
      lon: String(coords.lon),
      hours: String(hours),
    });
  }

  getHistory(options: {
    range?: string;
    page?: number;
    pageSize?: number;
    sort?: "asc" | "desc";
    city?: string;
    lat?: number;
    lon?: number;
  } = {}): Promise<unknown> {
    const params: Record<string, string> = {
      range: options.range ?? "24h",
      page: String(options.page ?? 1),
      page_size: String(options.pageSize ?? 50),
      sort: options.sort ?? "desc",
    };
    if (options.city) params.city = options.city;
    if (options.lat !== undefined && options.lon !== undefined) {
      params.lat = String(options.lat);
      params.lon = String(options.lon);
    }
    return this.request("GET", "/v2/history", params);
  }

  getHistoryByCity(city: string, range = "24h", page = 1, pageSize = 50, sort: "asc" | "desc" = "desc"): Promise<unknown> {
    return this.getHistory({ city, range, page, pageSize, sort });
  }

  getTrends(options: { range?: string; city?: string; lat?: number; lon?: number } = {}): Promise<unknown> {
    const params: Record<string, string> = {
      range: options.range ?? "7d",
    };
    if (options.city) params.city = options.city;
    if (options.lat !== undefined && options.lon !== undefined) {
      params.lat = String(options.lat);
      params.lon = String(options.lon);
    }
    return this.request("GET", "/v2/trends", params);
  }

  getTrendsByCity(city: string, range = "7d"): Promise<unknown> {
    return this.getTrends({ city, range });
  }

  listAlerts(): Promise<unknown> {
    return this.request("GET", "/v2/alerts", {});
  }

  getAlert(subscriptionId: string): Promise<unknown> {
    return this.request("GET", `/v2/alerts/${subscriptionId}`, {});
  }

  createAlert(payload: Record<string, unknown>, idempotencyKey?: string): Promise<unknown> {
    return this.request("POST", "/v2/alerts", {}, payload, idempotencyKey);
  }

  updateAlert(subscriptionId: string, payload: Record<string, unknown>, idempotencyKey?: string): Promise<unknown> {
    return this.request("PATCH", `/v2/alerts/${subscriptionId}`, {}, payload, idempotencyKey);
  }

  deleteAlert(subscriptionId: string): Promise<unknown> {
    return this.request("DELETE", `/v2/alerts/${subscriptionId}`, {});
  }
}
"""


def _generate_js_readme() -> str:
    return """# AirTrace RU JS SDK

Generated JavaScript SDK for the stable AirTrace RU public API v2.

Source of truth: `openapi/airtrace-v2.openapi.json`

## Install (local)

```bash
cd sdk/js
npm install
npm run build
```

## Usage example

```ts
import { AirTraceClient } from "@airtrace-ru/sdk-js";

const client = new AirTraceClient({ baseUrl: "http://localhost:8000", apiKey: "dev-key" });

const health = await client.getHealth();
const current = await client.getCurrent({ lat: 55.7558, lon: 37.6176 });
const history = await client.getHistoryByCity("moscow", "24h", 1, 50, "desc");
const trends = await client.getTrendsByCity("moscow", "7d");
const alerts = await client.listAlerts();
console.log({ health, current, history, trends, alerts });
```

## Supported endpoints

- `/v2/health`
- `/v2/current`
- `/v2/forecast`
- `/v2/history`
- `/v2/trends`
- `/v2/alerts`
"""


def _generate_python_example() -> str:
    return """from airtrace_sdk import AirTraceClient


def main() -> None:
    with AirTraceClient(base_url=\"http://localhost:8000\", retries=2, retry_delay=0.2) as client:
        health = client.get_health()
        current = client.get_current(lat=55.7558, lon=37.6176)
        history = client.get_history_by_city(city=\"moscow\", sort=\"desc\")
        trends = client.get_trends_by_city(city=\"moscow\", range=\"7d\")
        alerts = client.list_alerts()
        print({\"health\": health, \"current\": current, \"history_total\": history.get(\"total\"), \"trends\": trends.get(\"trend\"), \"alerts\": len(alerts)})


if __name__ == \"__main__\":
    main()
"""


def _generate_js_example() -> str:
    return """import { AirTraceClient } from \"../sdk/js/dist/index.js\";

async function main() {
  const client = new AirTraceClient({ baseUrl: \"http://localhost:8000\", apiKey: \"dev-key\", timeoutMs: 10000, retries: 2 });
  const health = await client.getHealth();
  const current = await client.getCurrent({ lat: 55.7558, lon: 37.6176 });
  const history = await client.getHistoryByCity(\"moscow\", \"24h\", 1, 50, \"desc\");
  const trends = await client.getTrendsByCity(\"moscow\", \"7d\");
  const alerts = await client.listAlerts();
  console.log(JSON.stringify({ health, current, history, trends, alerts }, null, 2));
}

main().catch((error) => {
  console.error(\"SDK example failed:\", error);
  process.exit(1);
});
"""


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def generate_stage3_artifacts() -> None:
    spec = _load_spec()
    _ensure_supported_paths(spec)
    _write_json(ROOT / PUBLIC_V2_OPENAPI_PATH, spec)
    _write_text(PYTHON_CLIENT_PATH, _generate_python_client())
    _write_text(PYTHON_INIT_PATH, _generate_python_init())
    _write_text(PYTHON_README_PATH, _generate_python_readme())
    _write_text(JS_CLIENT_PATH, _generate_js_client())
    _write_text(JS_README_PATH, _generate_js_readme())
    _write_text(PYTHON_EXAMPLE_PATH, _generate_python_example())
    _write_text(JS_EXAMPLE_PATH, _generate_js_example())


if __name__ == "__main__":
    generate_stage3_artifacts()
