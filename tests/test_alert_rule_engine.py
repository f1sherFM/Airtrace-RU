"""
Unit tests for alert rule engine (Issue 5.1).
"""

from datetime import datetime, timezone, timedelta

from alert_rule_engine import AlertRuleEngine
from schemas import AlertRuleCreate
import pytest


def test_alert_rule_triggers_and_cooldown_suppresses_duplicates():
    engine = AlertRuleEngine()
    rule = engine.create_rule(
        AlertRuleCreate(
            name="AQI high",
            aqi_threshold=120,
            cooldown_minutes=60,
            nmu_levels=[],
        )
    )

    t0 = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)
    first = engine.evaluate(aqi=150, nmu_risk="low", now=t0)
    second = engine.evaluate(aqi=150, nmu_risk="low", now=t0 + timedelta(minutes=10))
    third = engine.evaluate(aqi=150, nmu_risk="low", now=t0 + timedelta(minutes=61))

    assert len(first) == 1 and first[0].suppressed is False and first[0].rule_id == rule.id
    assert len(second) == 1 and second[0].suppressed is True
    assert len(third) == 1 and third[0].suppressed is False


def test_alert_rule_respects_quiet_hours():
    engine = AlertRuleEngine()
    engine.create_rule(
        AlertRuleCreate(
            name="Night quiet hours",
            aqi_threshold=100,
            cooldown_minutes=30,
            quiet_hours_start=22,
            quiet_hours_end=7,
        )
    )
    quiet_time = datetime(2026, 2, 16, 23, 0, tzinfo=timezone.utc)
    events = engine.evaluate(aqi=180, nmu_risk="high", now=quiet_time)

    assert len(events) == 1
    assert events[0].suppressed is True
    assert "quiet_hours" in events[0].reasons


def test_alert_rule_nmu_trigger():
    engine = AlertRuleEngine()
    engine.create_rule(
        AlertRuleCreate(
            name="Critical NMU",
            nmu_levels=["critical", "high"],
            cooldown_minutes=30,
        )
    )
    events = engine.evaluate(
        aqi=80,
        nmu_risk="critical",
        now=datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc),
    )
    assert len(events) == 1
    assert events[0].suppressed is False
    assert any(reason.startswith("nmu=") for reason in events[0].reasons)


def test_alert_rule_cooldown_marks_suppressed_reason():
    engine = AlertRuleEngine()
    engine.create_rule(AlertRuleCreate(name="AQI>=100", aqi_threshold=100, cooldown_minutes=60))
    t0 = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)
    _ = engine.evaluate(aqi=150, nmu_risk="low", now=t0)
    second = engine.evaluate(aqi=150, nmu_risk="low", now=t0 + timedelta(minutes=5))
    assert len(second) == 1
    assert second[0].suppressed is True
    assert "cooldown" in second[0].reasons


def test_alert_rule_validation_requires_trigger():
    with pytest.raises(ValueError):
        AlertRuleCreate(name="invalid", aqi_threshold=None, nmu_levels=[])


def test_alert_rule_validation_rejects_invalid_nmu():
    with pytest.raises(ValueError):
        AlertRuleCreate(name="invalid nmu", nmu_levels=["extreme"])
