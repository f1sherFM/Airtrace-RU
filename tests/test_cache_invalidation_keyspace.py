import pytest

from cache import CacheLevel, MultiLevelCacheManager
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
