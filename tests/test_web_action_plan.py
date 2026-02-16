"""
Unit tests for action-oriented recommendation plan (Issue #15).
"""

from pathlib import Path
import importlib.util
import os


def _load_web_app_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "web" / "web_app.py"
    previous_cwd = os.getcwd()
    try:
        os.chdir(repo_root / "web")
        spec = importlib.util.spec_from_file_location("web_app", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous_cwd)


web_app = _load_web_app_module()


def test_action_plan_low_scenario_includes_sensitive_group():
    plan = web_app.get_action_plan(aqi_value=40, nmu_risk="low")
    assert plan["risk_label"] == "low"
    assert "Что делать сейчас" in plan["title"]
    assert len(plan["general"]) >= 2
    assert len(plan["sensitive"]) >= 2
    assert len(plan["immediate"]) == 2


def test_action_plan_medium_scenario():
    plan = web_app.get_action_plan(aqi_value=110, nmu_risk="medium")
    assert plan["risk_label"] == "medium"
    assert "Умеренный" in plan["title"]


def test_action_plan_high_scenario():
    plan = web_app.get_action_plan(aqi_value=160, nmu_risk="high")
    assert plan["risk_label"] == "high"
    assert "Высокий" in plan["title"]


def test_action_plan_critical_scenario():
    plan = web_app.get_action_plan(aqi_value=220, nmu_risk="critical")
    assert plan["risk_label"] == "critical"
    assert "Критический" in plan["title"]
    assert any("медицинской" in x for x in plan["sensitive"])
