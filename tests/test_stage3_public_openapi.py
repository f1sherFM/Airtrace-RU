"""Tests for the committed public v2 OpenAPI artifact."""

from __future__ import annotations

import json
from pathlib import Path

import main

from api.v2.openapi import build_public_v2_openapi


ARTIFACT_PATH = Path("openapi/airtrace-v2.openapi.json")


def test_public_v2_openapi_artifact_matches_generated_contract():
    generated = build_public_v2_openapi(main.app)
    committed = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert committed == generated


def test_public_v2_openapi_artifact_only_contains_v2_paths():
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert payload["openapi"] == "3.1.0"
    assert payload["info"]["title"] == "AirTrace RU Public API v2"
    assert payload["paths"]
    assert all(path.startswith("/v2/") for path in payload["paths"])
    for route in (
        "/v2/current",
        "/v2/forecast",
        "/v2/history",
        "/v2/trends",
        "/v2/health",
        "/v2/alerts",
        "/v2/alerts/{subscription_id}",
    ):
        assert route in payload["paths"]


def test_public_v2_openapi_artifact_is_free_from_common_mojibake_markers():
    content = ARTIFACT_PATH.read_text(encoding="utf-8")
    for marker in ("Гђ", "Г‘", "Гўв‚¬в„ў", "Гўв‚¬Е“", "Гўв‚¬вЂќ", "\ufffd"):
        assert marker not in content


def test_public_v2_openapi_health_schema_includes_public_status():
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    health_schema = payload["components"]["schemas"]["HealthCheckResponse"]
    properties = health_schema.get("properties", {})
    assert "status" in properties
    assert "public_status" in properties
    assert "services" in properties


def test_public_v2_openapi_alerts_document_auth_and_conflict_responses():
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    create_alert_responses = payload["paths"]["/v2/alerts"]["post"]["responses"]
    update_alert_responses = payload["paths"]["/v2/alerts/{subscription_id}"]["patch"]["responses"]

    for responses in (create_alert_responses, update_alert_responses):
        assert "401" in responses
        assert "409" in responses
        assert responses["401"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/ErrorResponse"
        assert responses["409"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/ErrorResponse"
