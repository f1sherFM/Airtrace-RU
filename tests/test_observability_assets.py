"""
Validation tests for SLO and dashboard observability assets (Issue #23).
"""

import json
from pathlib import Path


def test_grafana_dashboard_json_is_valid_and_contains_core_panels():
    path = Path("observability/grafana/airtrace_runtime_overview.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["title"] == "AirTrace Runtime Overview"
    assert isinstance(payload.get("panels"), list) and len(payload["panels"]) >= 5

    all_expr = []
    for panel in payload["panels"]:
        for target in panel.get("targets", []):
            all_expr.append(target.get("expr", ""))

    required_metrics = [
        "airtrace_requests_total",
        "airtrace_request_duration_seconds",
        "airtrace_error_rate",
        "airtrace_external_api_success_rate",
        "airtrace_cache_hit_rate",
    ]
    for metric in required_metrics:
        assert any(metric in expr for expr in all_expr)


def test_slo_doc_includes_thresholds_and_error_budget():
    path = Path("docs/slo_runtime_control.md")
    content = path.read_text(encoding="utf-8")

    assert "99.5%" in content
    assert "p95 < 800ms" in content
    assert "p99 < 1500ms" in content
    assert "Error Budget" in content
    assert "airtrace_error_rate" in content
