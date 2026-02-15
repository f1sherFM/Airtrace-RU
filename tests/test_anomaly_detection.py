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
