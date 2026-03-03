"""
Targeted tests for WeatherAPI manager request statistics accounting.
"""

from unittest.mock import AsyncMock, patch

import pytest

from connection_pool import APIResponse
from weather_api_manager import WeatherAPIManager


def _create_manager() -> WeatherAPIManager:
    manager = WeatherAPIManager()
    manager.configure_api_key("test_api_key_123")
    return manager


@pytest.mark.asyncio
async def test_make_api_request_success_updates_counters_once():
    manager = _create_manager()
    response = APIResponse(
        status_code=200,
        data={"current": {"temp_c": 20.0}},
        headers={},
        response_time=0.1,
    )
    with patch("weather_api_manager.get_connection_pool_manager") as get_pool:
        get_pool.return_value.execute_request = AsyncMock(return_value=response)
        data = await manager._make_api_request("current", 55.7558, 37.6176)

    assert data["current"]["temp_c"] == 20.0
    assert manager.requests_made == 1
    assert manager.successful_requests == 1
    assert manager.failed_requests == 0


@pytest.mark.asyncio
async def test_make_api_request_non_200_counts_single_failure():
    manager = _create_manager()
    response = APIResponse(
        status_code=429,
        data={"error": {"message": "Rate limit exceeded"}},
        headers={},
        response_time=0.1,
    )
    with patch("weather_api_manager.get_connection_pool_manager") as get_pool:
        get_pool.return_value.execute_request = AsyncMock(return_value=response)
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await manager._make_api_request("current", 55.7558, 37.6176)

    assert manager.requests_made == 1
    assert manager.successful_requests == 0
    assert manager.failed_requests == 1


@pytest.mark.asyncio
async def test_make_api_request_transport_exception_counts_single_failure():
    manager = _create_manager()
    with patch("weather_api_manager.get_connection_pool_manager") as get_pool:
        get_pool.return_value.execute_request = AsyncMock(side_effect=Exception("network down"))
        with pytest.raises(Exception, match="network down"):
            await manager._make_api_request("current", 55.7558, 37.6176)

    assert manager.requests_made == 1
    assert manager.successful_requests == 0
    assert manager.failed_requests == 1
