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
