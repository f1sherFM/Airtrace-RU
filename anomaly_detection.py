"""
Hourly anomaly detection for historical AQI series.
"""

from dataclasses import dataclass
from statistics import median
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class AnomalyResult:
    detected: bool
    anomaly_type: Optional[str]
    score: float
    baseline: float


class HourlyAnomalyDetector:
    """
    Detects AQI spikes/dropouts against a local rolling baseline.
    """

    def __init__(
        self,
        baseline_window: int = 6,
        min_absolute_delta: float = 35.0,
        min_relative_delta: float = 0.55,
    ):
        self.baseline_window = baseline_window
        self.min_absolute_delta = min_absolute_delta
        self.min_relative_delta = min_relative_delta

    def _baseline(self, history_values: Sequence[float]) -> float:
        if not history_values:
            return 0.0
        return float(median(history_values))

    def evaluate(self, current_value: float, previous_values: List[float]) -> AnomalyResult:
        """
        Evaluate one point against previous rolling baseline.
        """
        context = previous_values[-self.baseline_window :]
        baseline = self._baseline(context)
        if baseline <= 0:
            return AnomalyResult(False, None, 0.0, baseline)

        abs_delta = current_value - baseline
        rel_delta = abs(abs_delta) / baseline
        score = max(0.0, rel_delta)

        if abs(abs_delta) < self.min_absolute_delta:
            return AnomalyResult(False, None, score, baseline)

        if rel_delta < self.min_relative_delta:
            return AnomalyResult(False, None, score, baseline)

        if abs_delta > 0:
            return AnomalyResult(True, "spike", score, baseline)
        return AnomalyResult(True, "dropout", score, baseline)
