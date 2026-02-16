"""
Alert rule engine for Issue 5.1.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from schemas import AlertEvent, AlertRule, AlertRuleCreate, AlertRuleUpdate, AlertSeverity


class AlertRuleEngine:
    """In-memory alert rule engine with cooldown and quiet-hours suppression."""

    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._last_triggered_by_rule: Dict[str, datetime] = {}

    def create_rule(self, payload: AlertRuleCreate) -> AlertRule:
        rule = AlertRule(**payload.model_dump())
        self._rules[rule.id] = rule
        return rule

    def list_rules(self) -> List[AlertRule]:
        return sorted(self._rules.values(), key=lambda r: r.created_at, reverse=True)

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        return self._rules.get(rule_id)

    def delete_rule(self, rule_id: str) -> bool:
        existed = rule_id in self._rules
        if existed:
            self._rules.pop(rule_id, None)
            self._last_triggered_by_rule.pop(rule_id, None)
        return existed

    def update_rule(self, rule_id: str, payload: AlertRuleUpdate) -> Optional[AlertRule]:
        rule = self._rules.get(rule_id)
        if rule is None:
            return None
        updates = payload.model_dump(exclude_unset=True)
        merged = rule.model_dump()
        merged.update(updates)
        updated = AlertRule(**merged)
        if updated.aqi_threshold is None and not updated.nmu_levels:
            raise ValueError("At least one trigger is required: aqi_threshold or nmu_levels")
        self._rules[rule_id] = updated
        return updated

    @staticmethod
    def _is_in_quiet_hours(now: datetime, start: Optional[int], end: Optional[int]) -> bool:
        if start is None or end is None:
            return False
        hour = now.hour
        if start == end:
            return True
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    @staticmethod
    def _severity(aqi: int, nmu_risk: Optional[str]) -> AlertSeverity:
        if aqi >= 200 or (nmu_risk or "").lower() == "critical":
            return AlertSeverity.CRITICAL
        if aqi >= 150 or (nmu_risk or "").lower() == "high":
            return AlertSeverity.WARNING
        return AlertSeverity.INFO

    def evaluate(
        self,
        *,
        aqi: int,
        nmu_risk: Optional[str],
        now: Optional[datetime] = None,
    ) -> List[AlertEvent]:
        now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        events: List[AlertEvent] = []
        normalized_nmu = (nmu_risk or "").lower()

        for rule in self.list_rules():
            if not rule.enabled:
                continue

            reasons: List[str] = []
            if rule.aqi_threshold is not None and aqi >= rule.aqi_threshold:
                reasons.append(f"aqi>={rule.aqi_threshold}")
            if rule.nmu_levels and normalized_nmu in {x.lower() for x in rule.nmu_levels}:
                reasons.append(f"nmu={normalized_nmu}")
            if not reasons:
                continue

            if self._is_in_quiet_hours(now_utc, rule.quiet_hours_start, rule.quiet_hours_end):
                events.append(
                    AlertEvent(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        severity=self._severity(aqi, normalized_nmu),
                        reasons=reasons + ["quiet_hours"],
                        suppressed=True,
                    )
                )
                continue

            last = self._last_triggered_by_rule.get(rule.id)
            cooldown_until = last + timedelta(minutes=rule.cooldown_minutes) if last else None
            if cooldown_until and now_utc < cooldown_until:
                events.append(
                    AlertEvent(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        severity=self._severity(aqi, normalized_nmu),
                        reasons=reasons + ["cooldown"],
                        suppressed=True,
                    )
                )
                continue

            self._last_triggered_by_rule[rule.id] = now_utc
            events.append(
                AlertEvent(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    severity=self._severity(aqi, normalized_nmu),
                    reasons=reasons,
                    suppressed=False,
                )
            )

        return events
