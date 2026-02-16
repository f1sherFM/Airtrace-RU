"""
Validation tests for incident runbooks and rollback docs (Issue #28).
"""

from pathlib import Path


def test_incident_runbook_contains_required_operational_sections():
    content = Path("docs/incident_runbooks.md").read_text(encoding="utf-8")

    assert "Runbook: API outage" in content
    assert "Runbook: Provider failure" in content
    assert "Runbook: Cache degradation" in content
    assert "Verified rollback path" in content
    assert "On-call first 10 minutes checklist" in content
    assert "Rollback verification checklist" in content


def test_deployment_checklist_links_incident_runbook():
    content = Path("DEPLOYMENT_CHECKLIST.md").read_text(encoding="utf-8")
    assert "docs/incident_runbooks.md" in content
