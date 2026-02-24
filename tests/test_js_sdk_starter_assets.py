"""
Validation tests for JS SDK starter assets (Issue #31).
"""

from pathlib import Path
import json


def test_js_sdk_package_has_version_and_build_scripts():
    package_payload = json.loads(Path("sdk/js/package.json").read_text(encoding="utf-8"))
    assert package_payload["name"] == "@airtrace-ru/sdk-js"
    assert package_payload["version"] == "0.3.1"
    assert "build" in package_payload["scripts"]
    assert "typecheck" in package_payload["scripts"]


def test_js_sdk_exports_v2_client_methods_and_example():
    sdk = Path("sdk/js/src/index.ts").read_text(encoding="utf-8")
    example = Path("examples/js_sdk_example.mjs").read_text(encoding="utf-8")

    assert "getCurrent" in sdk
    assert "getForecast" in sdk
    assert "getHistoryByCity" in sdk
    assert "getHealth" in sdk
    assert "/v2/current" in sdk
    assert "/v2/forecast" in sdk
    assert "/v2/history" in sdk
    assert "/v2/health" in sdk
    assert "new AirTraceClient" in example
