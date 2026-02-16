"""
Regression checks for pool saturation fallback behavior in services.
"""

from services import AirQualityService


def test_pool_saturation_detection_keywords():
    assert AirQualityService._is_pool_saturation_error(Exception("Request timed out in queue after 5.00s for open_meteo"))
    assert AirQualityService._is_pool_saturation_error(Exception("Request queue full for open_meteo"))
    assert AirQualityService._is_pool_saturation_error(Exception("Circuit breaker open for open_meteo"))
    assert not AirQualityService._is_pool_saturation_error(Exception("HTTP 500 from upstream"))
