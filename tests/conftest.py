"""
Pytest configuration and fixtures for AirTrace RU Backend tests.

Содержит общие фикстуры и настройки для unit и property-based тестов.
"""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
import httpx

from main import app
from services import AirQualityService, CacheManager
from utils import AQICalculator


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client for testing async endpoints."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def aqi_calculator() -> AQICalculator:
    """AQI calculator instance for testing."""
    return AQICalculator()


@pytest.fixture
async def air_quality_service() -> AsyncGenerator[AirQualityService, None]:
    """Air quality service instance for testing."""
    service = AirQualityService()
    yield service
    await service.cleanup()


@pytest.fixture
def cache_manager() -> CacheManager:
    """Cache manager instance for testing."""
    return CacheManager(ttl_minutes=1)  # Short TTL for tests


@pytest.fixture
def sample_pollutants():
    """Sample pollutant data for testing."""
    return {
        'pm2_5': 25.4,
        'pm10': 45.2,
        'no2': 35.1,
        'so2': 12.3,
        'o3': 85.7
    }


@pytest.fixture
def sample_coordinates():
    """Sample coordinates for Moscow."""
    return {'lat': 55.7558, 'lon': 37.6176}


@pytest.fixture
def invalid_coordinates():
    """Invalid coordinates for testing validation."""
    return [
        {'lat': 91.0, 'lon': 0.0},    # Invalid latitude
        {'lat': 0.0, 'lon': 181.0},   # Invalid longitude
        {'lat': -91.0, 'lon': 0.0},   # Invalid latitude
        {'lat': 0.0, 'lon': -181.0},  # Invalid longitude
    ]


# Hypothesis settings for property-based tests
from hypothesis import settings, Verbosity

# Configure Hypothesis for consistent test runs
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("ci", max_examples=1000, verbosity=Verbosity.verbose)
settings.load_profile("default")