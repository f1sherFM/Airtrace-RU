"""
Validation tests for CI hardening assets (Issue #27).
"""

from pathlib import Path


def test_workflow_has_hardened_ci_stages_and_timeouts():
    content = Path(".github/workflows/contract-tests.yml").read_text(encoding="utf-8")

    assert "lint:" in content
    assert "tests:" in content
    assert "contract-tests:" in content
    assert "smoke-tests:" in content
    assert content.count("timeout-minutes:") >= 4
    assert content.count("actions/upload-artifact@v4") >= 4
    assert "needs: [lint]" in content
    assert "needs: [tests]" in content
    assert "needs: [contract-tests]" in content


def test_ci_doc_lists_required_checks():
    content = Path("docs/ci_pipeline_hardening.md").read_text(encoding="utf-8")
    assert "Required checks" in content
    assert "- `lint`" in content
    assert "- `tests`" in content
    assert "- `contract-tests`" in content
    assert "- `smoke-tests`" in content
