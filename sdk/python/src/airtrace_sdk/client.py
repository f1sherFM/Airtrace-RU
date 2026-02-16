from __future__ import annotations

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
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        retries: int = 2,
        retry_delay: float = 0.3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(0, retries)
        self.retry_delay = max(0.0, retry_delay)
        self._client = httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AirTraceClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                response = self._client.get(url, params=params or {})
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
        return self._request("/v2/health")

    def get_current(self, *, lat: float, lon: float) -> Any:
        return self._request("/v2/current", {"lat": lat, "lon": lon})

    def get_forecast(self, *, lat: float, lon: float) -> Any:
        return self._request("/v2/forecast", {"lat": lat, "lon": lon})

    def get_history_by_city(self, *, city: str, range: str = "24h", page: int = 1, page_size: int = 20) -> Any:
        return self._request(
            "/v2/history",
            {"city": city, "range": range, "page": page, "page_size": page_size},
        )
