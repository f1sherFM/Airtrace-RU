"""
Contract snapshot test for API schema drift gating (Issue #12).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import main


SNAPSHOT_PATH = Path(__file__).parent / "contract_snapshots" / "api_v2_contract_snapshot.json"


def _extract_operation(openapi: Dict[str, Any], path: str, method: str = "get") -> Dict[str, Any]:
    operation = openapi["paths"][path][method]
    return {
        "operationId": operation.get("operationId"),
        "parameters": operation.get("parameters", []),
        "responses": operation.get("responses", {}),
    }


def _build_contract_snapshot() -> Dict[str, Any]:
    openapi = main.app.openapi()

    selected_paths: List[str] = [
        "/weather/current",
        "/weather/forecast",
        "/history",
        "/health",
        "/v2/current",
        "/v2/forecast",
        "/v2/history",
        "/v2/health",
    ]
    selected_schemas: List[str] = [
        "AirQualityData",
        "ResponseMetadata",
        "HistoryQueryResponse",
        "HistoricalSnapshotRecord",
        "HealthCheckResponse",
        "HTTPValidationError",
        "ValidationError",
    ]

    snapshot: Dict[str, Any] = {
        "openapi": openapi.get("openapi"),
        "info": openapi.get("info"),
        "paths": {},
        "components": {"schemas": {}},
    }

    for path in selected_paths:
        snapshot["paths"][path] = {"get": _extract_operation(openapi, path)}

    schemas = openapi.get("components", {}).get("schemas", {})
    for schema_name in selected_schemas:
        if schema_name in schemas:
            snapshot["components"]["schemas"][schema_name] = schemas[schema_name]

    return snapshot


def test_api_contract_snapshot():
    """
    Fails when selected OpenAPI contract drifts from committed snapshot.

    To refresh snapshot intentionally:
    UPDATE_CONTRACT_SNAPSHOT=1 python -m pytest -q tests/test_contract_snapshot.py
    """
    current_snapshot = _build_contract_snapshot()

    if os.getenv("UPDATE_CONTRACT_SNAPSHOT") == "1":
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(
            json.dumps(current_snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    assert SNAPSHOT_PATH.exists(), (
        f"Snapshot file missing: {SNAPSHOT_PATH}. "
        "Generate it with UPDATE_CONTRACT_SNAPSHOT=1 python -m pytest -q tests/test_contract_snapshot.py"
    )

    expected_snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert current_snapshot == expected_snapshot
