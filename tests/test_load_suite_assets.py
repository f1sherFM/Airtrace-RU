"""
Validation tests for load/soak suite assets (Issue #24).
"""

from pathlib import Path
import runpy


def test_load_suite_script_exists_and_is_importable():
    path = Path("tools/load/run_load_suite.py")
    assert path.exists()
    # Ensure module can be loaded without executing as __main__
    runpy.run_path(str(path), run_name="load_suite_module")


def test_load_soak_docs_present():
    doc = Path("docs/load_soak_testing.md")
    content = doc.read_text(encoding="utf-8")
    assert "/weather/current" in content
    assert "/history/export/json" in content
    assert "error_rate > 0.01" in content
