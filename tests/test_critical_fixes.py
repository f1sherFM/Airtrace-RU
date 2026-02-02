"""
Tests for critical bug fixes

Tests for the following critical fixes:
1. API key validation (#1)
2. Race condition in cache (#2)
4. Memory growth limitation (#4)
5. Redis timeout (#5)
"""

import pytest
import asyncio
import time
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

# Test Fix #1: API Key Validation
class TestAPIKeyValidation:
    """Test API key validation in WeatherAPI configuration"""
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError when WeatherAPI is enabled"""
        from config import WeatherAPIConfig
        
        with patch.dict(os.environ, {
            'WEATHER_API_ENABLED': 'true',
            'WEATHER_API_KEY': ''
        }):
            with pytest.raises(ValueError, match="WEATHER_API_KEY is required"):
                config = WeatherAPIConfig()
    
    def test_short_api_key_raises_error(self):
        """Test that short API key raises ValueError"""
        from config import WeatherAPIConfig
        
        with patch.dict(os.environ, {
            'WEATHER_API_ENABLED': 'true',
            'WEATHER_API_KEY': 'short'
        }):
            with pytest.raises(ValueError, match="Invalid WEATHER_API_KEY format"):
                config = WeatherAPIConfig()
    
    def test_placeholder_api_key_raises_error(self):
        """Test that placeholder API key raises ValueError"""
        from config import WeatherAPIConfig
        
        placeholder_keys = [
            'your_api_key_here_12345678',
            'example_key_1234567890123',
            'test_api_key_12345678901234',
            'demo_key_123456789012345678'
        ]
        
        for key in placeholder_keys:
            with patch.dict(os.environ, {
                'WEATHER_API_ENABLED': 'true',
                'WEATHER_API_KEY': key
            }):
                with pytest.raises(ValueError, match="appears to be a placeholder"):
                    config = WeatherAPIConfig()
    
    def test_valid_api_key_accepted(self):
        """Test that valid API key is accepted"""
        from config import WeatherAPIConfig
        
        valid_key = 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6'  # 32 char valid-looking key
        
        with patch.dict(os.environ, {
            'WEATHER_API_ENABLED': 'true',
            'WEATHER_API_KEY': valid_key
        }):
            config = WeatherAPIConfig()
            assert config.enabled is True
            assert config.api_key == valid_key
    
    def test_disabled_weatherapi_no_validation(self):
        """Test that disabled WeatherAPI doesn't validate key"""
        from config import WeatherAPIConfig
        
        with patch.dict(os.environ, {
            'WEATHER_API_ENABLED': 'false',
            'WEATHER_API_KEY': ''
        }):
            config = WeatherAPIConfig()
            assert config.enabled is False


# Test Fix #2: Race Condition in Cache
class TestCacheRaceCondition:
    """Test race condition fix in cache eviction"""
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_eviction_no_race(self):
        """Test that concurrent cache operations don't cause race conditions"""
        from cache import MultiLevelCacheManager
        
        cache = MultiLevelCacheManager()
        cache._l1_max_size = 10  # Small size to trigger evictions
        
        # Fill cache to capacity
        for i in range(10):
            await cache.set(float(i), float(i), {"data": f"value_{i}"})
        
        # Concurrent operations that trigger evictions
        async def add_item(index):
            await cache.set(float(100 + index), float(100 + index), {"data": f"value_{100 + index}"})
            await asyncio.sleep(0.001)  # Small delay
        
        # Run 20 concurrent operations (should trigger evictions)
        tasks = [add_item(i) for i in range(20)]
        await asyncio.gather(*tasks)
        
        # Cache should not exceed max size
        assert len(cache._l1_cache) <= cache._l1_max_size
        
        # Stats should be consistent
        stats = await cache.get_stats()
        assert stats.eviction_count > 0
    
    @pytest.mark.asyncio
    async def test_eviction_with_lock(self):
        """Test that eviction properly uses lock"""
        from cache import MultiLevelCacheManager
        
        cache = MultiLevelCacheManager()
        cache._l1_max_size = 5
        
        # Fill cache
        for i in range(5):
            await cache.set(float(i), float(i), {"data": f"value_{i}"})
        
        # Trigger eviction
        await cache.set(10.0, 10.0, {"data": "new_value"})
        
        # Verify eviction happened
        stats = await cache.get_stats()
        assert stats.eviction_count == 1
        assert len(cache._l1_cache) == 5


# Test Fix #4: Memory Growth Limitation
class TestMemoryGrowthLimitation:
    """Test memory growth limitation in graceful degradation"""
    
    @pytest.mark.asyncio
    async def test_stale_data_cache_size_limit(self):
        """Test that stale data cache doesn't grow unbounded"""
        from graceful_degradation import GracefulDegradationManager
        
        manager = GracefulDegradationManager()
        max_entries = manager._max_stale_entries
        
        # Add more entries than the limit
        for i in range(max_entries + 500):
            await manager.store_stale_data(f"key_{i}", {"data": f"value_{i}"})
        
        # Cache should not exceed max size
        assert len(manager.stale_data_cache) <= max_entries
        
        # Oldest entries should be evicted (LRU)
        # The first entries should be gone
        assert "key_0" not in manager.stale_data_cache
        assert "key_100" not in manager.stale_data_cache
        
        # Recent entries should still be there
        assert f"key_{max_entries + 499}" in manager.stale_data_cache
    
    @pytest.mark.asyncio
    async def test_lru_eviction_order(self):
        """Test that LRU eviction works correctly"""
        from graceful_degradation import GracefulDegradationManager
        
        manager = GracefulDegradationManager()
        manager._max_stale_entries = 10  # Small limit for testing
        
        # Add 10 entries
        for i in range(10):
            await manager.store_stale_data(f"key_{i}", {"data": f"value_{i}"})
        
        # Verify all 10 entries are there
        assert len(manager.stale_data_cache) == 10
        
        # Add one more entry (should evict key_0, the oldest)
        await manager.store_stale_data("key_10", {"data": "value_10"})
        
        # key_0 should be evicted (oldest)
        assert "key_0" not in manager.stale_data_cache
        
        # key_10 should be there (newest)
        assert "key_10" in manager.stale_data_cache
        
        # Total should still be 10
        assert len(manager.stale_data_cache) == 10
    
    @pytest.mark.asyncio
    async def test_memory_bounded_under_load(self):
        """Test that memory stays bounded under heavy load"""
        from graceful_degradation import GracefulDegradationManager
        
        manager = GracefulDegradationManager()
        max_entries = manager._max_stale_entries
        
        # Simulate heavy load with many concurrent stores
        async def store_data(index):
            await manager.store_stale_data(f"key_{index}", {"data": f"value_{index}" * 100})
        
        tasks = [store_data(i) for i in range(max_entries * 2)]
        await asyncio.gather(*tasks)
        
        # Should never exceed limit
        assert len(manager.stale_data_cache) <= max_entries


# Test Fix #5: Redis Timeout
class TestRedisTimeout:
    """Test Redis timeout implementation"""
    
    @pytest.mark.asyncio
    async def test_redis_get_timeout(self):
        """Test that Redis get operations timeout properly"""
        from cache import MultiLevelCacheManager
        
        cache = MultiLevelCacheManager()
        cache._l2_enabled = True
        cache._redis_initialized = True
        
        # Mock Redis client that hangs
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=asyncio.sleep(10))  # Hangs for 10 seconds
        cache._redis_client = mock_redis
        cache._redis_healthy = True
        
        # Should timeout and return None
        start_time = time.time()
        result = await cache._get_from_l2("test_key")
        elapsed = time.time() - start_time
        
        assert result is None
        assert elapsed < 3.0  # Should timeout in ~2 seconds, not 10
    
    @pytest.mark.asyncio
    async def test_redis_set_timeout(self):
        """Test that Redis set operations timeout properly"""
        from cache import MultiLevelCacheManager
        
        cache = MultiLevelCacheManager()
        cache._l2_enabled = True
        cache._redis_initialized = True
        
        # Mock Redis client that hangs
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(side_effect=asyncio.sleep(10))
        cache._redis_client = mock_redis
        cache._redis_healthy = True
        
        # Should timeout and return False
        start_time = time.time()
        result = await cache._set_to_l2("test_key", {"data": "test"}, 300)
        elapsed = time.time() - start_time
        
        assert result is False
        assert elapsed < 3.0  # Should timeout in ~2 seconds
    
    @pytest.mark.asyncio
    async def test_redis_health_check_timeout(self):
        """Test that Redis health check timeouts properly"""
        from cache import MultiLevelCacheManager
        
        cache = MultiLevelCacheManager()
        cache._l2_enabled = True
        cache._redis_initialized = True
        
        # Mock Redis client that hangs on ping
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=asyncio.sleep(10))
        cache._redis_client = mock_redis
        
        # Should timeout and return False
        start_time = time.time()
        result = await cache._check_redis_health()
        elapsed = time.time() - start_time
        
        assert result is False
        assert cache._redis_healthy is False
        assert elapsed < 3.0  # Should timeout in ~2 seconds
    
    @pytest.mark.asyncio
    async def test_redis_operations_continue_after_timeout(self):
        """Test that cache continues working after Redis timeout"""
        from cache import MultiLevelCacheManager
        
        cache = MultiLevelCacheManager()
        cache._l1_enabled = True
        cache._l2_enabled = True
        cache._redis_initialized = True
        
        # Mock Redis that times out
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_redis.setex = AsyncMock(side_effect=asyncio.TimeoutError())
        cache._redis_client = mock_redis
        cache._redis_healthy = True
        
        # L1 cache should still work
        await cache.set(55.0, 37.0, {"data": "test"})
        result = await cache.get(55.0, 37.0)
        
        # Should get data from L1 even though L2 timed out
        assert result is not None
        assert result["data"] == "test"


# Integration test for all fixes
class TestCriticalFixesIntegration:
    """Integration tests for all critical fixes"""
    
    @pytest.mark.asyncio
    async def test_system_resilience_under_stress(self):
        """Test that system remains resilient with all fixes applied"""
        from cache import MultiLevelCacheManager
        from graceful_degradation import GracefulDegradationManager
        
        cache = MultiLevelCacheManager()
        cache._l1_enabled = True
        cache._l2_enabled = False  # Disable Redis for this test
        
        degradation = GracefulDegradationManager()
        
        # Simulate high load
        async def simulate_request(index):
            try:
                # Store in cache (L1 only)
                await cache.set(float(index), float(index), {"data": f"value_{index}"})
                
                # Store stale data
                await degradation.store_stale_data(f"key_{index}", {"data": f"value_{index}"})
                
                # Retrieve from cache
                result = await cache.get(float(index), float(index))
                return result is not None
            except Exception as e:
                return False
        
        # Run many concurrent requests
        tasks = [simulate_request(i) for i in range(2000)]
        results = await asyncio.gather(*tasks)
        
        # Most requests should succeed (L1 cache only, so some may fail due to eviction)
        success_rate = sum(results) / len(results)
        assert success_rate > 0.5  # At least 50% should succeed
        
        # Memory should be bounded
        assert len(cache._l1_cache) <= cache._l1_max_size
        assert len(degradation.stale_data_cache) <= degradation._max_stale_entries
        
        # Stats should be consistent
        stats = await cache.get_stats()
        assert stats.total_requests > 0
        assert stats.eviction_count >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
