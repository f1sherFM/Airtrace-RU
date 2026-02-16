#!/usr/bin/env python3
"""
Lightweight load/soak suite for core AirTrace endpoints (Issue #24).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import httpx


@dataclass
class ScenarioResult:
    name: str
    requests: int
    successes: int
    failures: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round((len(values_sorted) - 1) * p))
    return values_sorted[idx]


async def _hit_endpoint(client: httpx.AsyncClient, url: str) -> tuple[bool, float]:
    start = time.perf_counter()
    try:
        resp = await client.get(url)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (200 <= resp.status_code < 500), elapsed_ms
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return False, elapsed_ms


async def run_scenario(base_url: str, name: str, path: str, requests: int, concurrency: int) -> ScenarioResult:
    sem = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    successes = 0
    failures = 0

    async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
        async def one():
            nonlocal successes, failures
            async with sem:
                ok, ms = await _hit_endpoint(client, f"{base_url}{path}")
                latencies.append(ms)
                if ok:
                    successes += 1
                else:
                    failures += 1

        await asyncio.gather(*[one() for _ in range(requests)])

    error_rate = failures / requests if requests else 0.0
    return ScenarioResult(
        name=name,
        requests=requests,
        successes=successes,
        failures=failures,
        p50_ms=round(_percentile(latencies, 0.50), 2),
        p95_ms=round(_percentile(latencies, 0.95), 2),
        p99_ms=round(_percentile(latencies, 0.99), 2),
        error_rate=round(error_rate, 4),
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--out", default="reports/load/load_suite_result.json")
    args = parser.parse_args()

    scenarios = [
        ("current", "/weather/current?lat=55.7558&lon=37.6176"),
        ("forecast", "/weather/forecast?lat=55.7558&lon=37.6176"),
        ("history", "/history?range=24h&page=1&page_size=50&city=moscow"),
        ("export_json", "/history/export/json?hours=24&city=moscow"),
    ]

    results: List[ScenarioResult] = []
    for name, path in scenarios:
        result = await run_scenario(args.base_url, name, path, args.requests, args.concurrency)
        results.append(result)

    payload = {
        "base_url": args.base_url,
        "requests_per_scenario": args.requests,
        "concurrency": args.concurrency,
        "results": [asdict(r) for r in results],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
