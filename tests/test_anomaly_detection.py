"""
Unit tests for hourly anomaly detection (Issue 2.3).
"""

from anomaly_detection import HourlyAnomalyDetector


def test_detects_spike_against_baseline():
    detector = HourlyAnomalyDetector()
    prev = [50, 52, 49, 51, 50, 48]
    result = detector.evaluate(current_value=120, previous_values=prev)
    assert result.detected is True
    assert result.anomaly_type == "spike"
    assert result.score > 0.55


def test_detects_dropout_against_baseline():
    detector = HourlyAnomalyDetector()
    prev = [120, 118, 122, 121, 119, 120]
    result = detector.evaluate(current_value=45, previous_values=prev)
    assert result.detected is True
    assert result.anomaly_type == "dropout"
    assert result.score > 0.55


def test_ignores_small_deviations():
    detector = HourlyAnomalyDetector()
    prev = [60, 62, 61, 59, 60, 58]
    result = detector.evaluate(current_value=80, previous_values=prev)
    assert result.detected is False


def test_false_positive_baseline_is_not_flagged():
    detector = HourlyAnomalyDetector()
    # Typical low-noise baseline with mild fluctuation should not create anomalies.
    prev = [74, 75, 73, 76, 74, 75]
    for value in [76, 73, 77, 72, 75]:
        result = detector.evaluate(current_value=value, previous_values=prev)
        assert result.detected is False
        prev.append(value)


def test_thresholds_are_configurable():
    detector = HourlyAnomalyDetector(min_absolute_delta=10.0, min_relative_delta=0.10)
    prev = [50, 52, 49, 51, 50, 48]
    result = detector.evaluate(current_value=60, previous_values=prev)
    assert result.detected is True
    assert result.anomaly_type == "spike"
