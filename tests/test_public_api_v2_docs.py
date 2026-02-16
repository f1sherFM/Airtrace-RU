"""
Validation tests for public API v2 docs (Issue #30).
"""

from pathlib import Path


def test_public_api_v2_doc_contains_verified_examples_and_migration_notes():
    content = Path("docs/public_api_v2.md").read_text(encoding="utf-8")

    assert "Verified curl snippets" in content
    assert "Migration notes (v1 -> v2)" in content
    assert "/v2/current" in content
    assert "/v2/forecast" in content
    assert "/v2/history" in content
    assert "/v2/health" in content
    assert "/weather/current" in content
    assert "/weather/forecast" in content
    assert "/history" in content
    assert "/health" in content


def test_readme_links_public_api_v2_guide():
    content = Path("README.md").read_text(encoding="utf-8")
    assert "docs/public_api_v2.md" in content
