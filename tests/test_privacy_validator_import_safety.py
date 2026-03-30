"""
Regression tests for import safety in privacy compliance validator.
"""

from pathlib import Path


def test_privacy_validator_avoids_top_level_config_import():
    content = Path("core/privacy_validation.py").read_text(encoding="utf-8")
    assert "\nfrom config import config\n" not in content
    assert "def _get_cache_privacy_settings" in content
