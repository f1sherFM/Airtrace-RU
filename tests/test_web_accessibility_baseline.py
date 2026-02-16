"""
Accessibility baseline checks for web templates (Issue #16).
"""

from pathlib import Path


def _read_template(name: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "web" / "templates" / name).read_text(encoding="utf-8")


def test_base_template_has_semantic_landmarks_and_skip_link():
    content = _read_template("base.html")
    assert 'href="#main-content"' in content
    assert 'id="main-content"' in content
    assert 'role="banner"' in content
    assert 'role="main"' in content
    assert 'role="contentinfo"' in content


def test_city_template_has_keyboard_toggle_buttons_and_live_regions():
    content = _read_template("city.html")
    assert 'aria-pressed="true"' in content
    assert 'aria-pressed="false"' in content
    assert 'aria-live="polite"' in content
    assert 'aria-busy="false"' in content
