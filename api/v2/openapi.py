"""Public OpenAPI builder for the stable read-only v2 contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi import FastAPI

PUBLIC_V2_OPENAPI_PATH = Path("openapi") / "airtrace-v2.openapi.json"

_MOJIBAKE_MARKERS = ("Ð", "Ñ", "â€™", "â€œ", "â€", "Р", "С", "\ufffd")


def _collect_schema_refs(payload: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(payload, dict):
        ref = payload.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            refs.add(ref.rsplit("/", 1)[-1])
        for value in payload.values():
            refs.update(_collect_schema_refs(value))
    elif isinstance(payload, list):
        for item in payload:
            refs.update(_collect_schema_refs(item))
    return refs


def _contains_mojibake(value: Any) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in _MOJIBAKE_MARKERS)
    if isinstance(value, dict):
        return any(_contains_mojibake(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_mojibake(item) for item in value)
    return False


def _sanitize_textual_metadata(payload: Any) -> Any:
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            if key in {"description", "summary", "title"} and isinstance(value, str) and _contains_mojibake(value):
                continue
            if key in {"example", "examples"} and _contains_mojibake(value):
                continue
            sanitized[key] = _sanitize_textual_metadata(value)
        return sanitized
    if isinstance(payload, list):
        return [_sanitize_textual_metadata(item) for item in payload]
    return payload


def build_public_v2_openapi(app: FastAPI) -> dict[str, Any]:
    source = deepcopy(app.openapi())
    public_paths = {path: item for path, item in source.get("paths", {}).items() if path.startswith("/v2/")}

    schemas = source.get("components", {}).get("schemas", {})
    required_schema_names = _collect_schema_refs(public_paths)
    selected_schemas: dict[str, Any] = {}
    pending = list(required_schema_names)
    while pending:
        schema_name = pending.pop()
        if schema_name in selected_schemas or schema_name not in schemas:
            continue
        schema_payload = deepcopy(schemas[schema_name])
        selected_schemas[schema_name] = schema_payload
        pending.extend(sorted(_collect_schema_refs(schema_payload) - selected_schemas.keys()))

    public_openapi = {
        "openapi": "3.1.0",
        "info": {
            "title": "AirTrace RU Public API v2",
            "version": source.get("info", {}).get("version", "0.3.1"),
            "description": "Stable read-only v2 contract for AirTrace RU.",
        },
        "paths": public_paths,
        "components": {
            "schemas": selected_schemas,
        },
    }
    return _sanitize_textual_metadata(public_openapi)
