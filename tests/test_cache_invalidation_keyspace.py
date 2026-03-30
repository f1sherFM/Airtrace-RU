import pytest

from cache import CacheLevel, MultiLevelCacheManager
from core.privacy_validation import validate_cache_key_privacy
from unified_weather_service import UnifiedWeatherService


@pytest.mark.asyncio
async def test_invalidate_by_coordinates_matches_set_get_keyspace_l1():
    cache = MultiLevelCacheManager()

    lat, lon = 55.7558, 37.6176
    payload = {"combined": {"aqi": 42}}

    ok = await cache.set(lat, lon, payload, ttl=60, levels=[CacheLevel.L1])
    assert ok is True

    cached = await cache.get(lat, lon, cache_levels=[CacheLevel.L1])
    assert cached == payload

    invalidated = await cache.invalidate_by_coordinates(lat, lon, levels=[CacheLevel.L1])
    assert invalidated == 1

    cached_after = await cache.get(lat, lon, cache_levels=[CacheLevel.L1])
    assert cached_after is None

    await cache.cleanup()


@pytest.mark.asyncio
async def test_unified_weather_service_uses_coordinate_invalidation_api(monkeypatch):
    service = UnifiedWeatherService()
    calls = {}

    async def fake_invalidate_by_coordinates(lat, lon, levels=None):
        calls["lat"] = lat
        calls["lon"] = lon
        calls["levels"] = levels
        return 1

    monkeypatch.setattr(service.cache_manager, "invalidate_by_coordinates", fake_invalidate_by_coordinates)

    result = await service.invalidate_location_cache(55.7558, 37.6176)

    assert result is True
    assert calls["lat"] == 55.7558
    assert calls["lon"] == 37.6176
    assert calls["levels"] == [CacheLevel.L1, CacheLevel.L2]

    await service.cleanup()


def test_unified_weather_service_combined_cache_key_is_privacy_safe(caplog):
    service = UnifiedWeatherService()

    with caplog.at_level("WARNING"):
        cache_key = service._generate_combined_cache_key(55.7558, 37.6176)

    assert cache_key.startswith("combined:")
    assert "55.7558" not in cache_key
    assert "37.6176" not in cache_key
    assert validate_cache_key_privacy(
        cache_key,
        "tests.test_unified_weather_service_combined_cache_key_is_privacy_safe",
    )
    assert "Combined cache key privacy validation failed" not in caplog.text
