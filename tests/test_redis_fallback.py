"""
Tests for Redis fallback mechanism

Verifies that the cache manager gracefully degrades to in-memory caching
when Redis is unavailable and automatically reconnects when Redis becomes available.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from cache import MultiLevelCacheManager, CacheLevel


class TestRedisFallback:
    """Tests for Redis fallback and reconnection mechanisms"""

    async def test_redis_unavailable_fallback_to_l1(self):
        """
        Test that cache operations fall back to L1 when Redis is unavailable.
        
        Validates: Requirements 1.4, 9.1
        """
        cache_manager = MultiLevelCacheManager()
        
        try:
            # Mock Redis to be unavailable
            with patch.object(cache_manager, '_check_redis_health', return_value=False):
                with patch.object(cache_manager, '_l2_enabled', True):  # L2 should be enabled but unhealthy
                    
                    # Store data - should succeed using L1 only
                    success = await cache_manager.set(55.7558, 37.6176, {"test": "data"}, ttl=300)
                    assert success, "Cache set should succeed even when Redis is unavailable"
                    
                    # Retrieve data - should succeed from L1
                    retrieved_data = await cache_manager.get(55.7558, 37.6176)
                    assert retrieved_data is not None, "Data should be retrievable from L1 cache"
                    assert retrieved_data == {"test": "data"}, "Retrieved data should match stored data"
                    
                    # Verify data is in L1 cache
                    key = cache_manager._generate_key(55.7558, 37.6176)
                    assert key in cache_manager._l1_cache, "Data should be stored in L1 cache"
                    
        finally:
            await cache_manager.cleanup()

    async def test_redis_connection_health_monitoring(self):
        """
        Test that Redis health monitoring works correctly.
        
        Validates: Requirements 1.4, 9.1
        """
        cache_manager = MultiLevelCacheManager()
        
        try:
            # Mock a Redis client for testing
            mock_redis_client = AsyncMock()
            cache_manager._redis_client = mock_redis_client
            cache_manager._l2_enabled = True
            
            # Test with healthy Redis connection (mocked)
            mock_redis_client.ping.return_value = True
            cache_manager._last_health_check = 0  # Force health check
            
            health_status = await cache_manager._check_redis_health()
            assert health_status is True, "Health check should return True for healthy Redis"
            assert cache_manager._redis_healthy is True, "Redis should be marked as healthy"
            
            # Test with unhealthy Redis connection (mocked)
            mock_redis_client.ping.side_effect = Exception("Connection failed")
            cache_manager._last_health_check = 0  # Force health check
            
            health_status = await cache_manager._check_redis_health()
            assert health_status is False, "Health check should return False for unhealthy Redis"
            assert cache_manager._redis_healthy is False, "Redis should be marked as unhealthy"
                
        finally:
            await cache_manager.cleanup()

    async def test_cache_status_reflects_redis_health(self):
        """
        Test that cache status correctly reflects Redis health.
        
        Validates: Requirements 1.4, 9.1
        """
        cache_manager = MultiLevelCacheManager()
        
        try:
            # Test with healthy Redis
            with patch.object(cache_manager, '_redis_healthy', True):
                with patch.object(cache_manager, '_l2_enabled', True):
                    status = cache_manager.get_status()
                    assert "L2:healthy" in status, "Status should show L2 as healthy when Redis is healthy"
            
            # Test with unhealthy Redis
            with patch.object(cache_manager, '_redis_healthy', False):
                with patch.object(cache_manager, '_l2_enabled', True):
                    status = cache_manager.get_status()
                    assert "L2:unhealthy" in status, "Status should show L2 as unhealthy when Redis is unhealthy"
            
            # Test with Redis disabled
            with patch.object(cache_manager, '_l2_enabled', False):
                status = cache_manager.get_status()
                assert "L2:disabled" in status, "Status should show L2 as disabled when Redis is disabled"
                
        finally:
            await cache_manager.cleanup()

    async def test_graceful_degradation_during_redis_failure(self):
        """
        Test that cache operations continue working when Redis fails during operation.
        
        Validates: Requirements 1.4, 9.1
        """
        cache_manager = MultiLevelCacheManager()
        
        try:
            # Initially store data successfully
            success = await cache_manager.set(55.7558, 37.6176, {"test": "data"}, ttl=300)
            assert success, "Initial cache set should succeed"
            
            # Simulate Redis failure during get operation
            with patch.object(cache_manager, '_get_from_l2', side_effect=Exception("Redis connection failed")):
                # Should still be able to get data from L1
                retrieved_data = await cache_manager.get(55.7558, 37.6176)
                assert retrieved_data is not None, "Data should still be retrievable from L1 after Redis failure"
                assert retrieved_data == {"test": "data"}, "Retrieved data should match stored data"
            
            # Simulate Redis failure during set operation
            with patch.object(cache_manager, '_set_to_l2', side_effect=Exception("Redis connection failed")):
                # Should still be able to set data to L1
                success = await cache_manager.set(55.7558, 37.6176, {"updated": "data"}, ttl=300)
                assert success, "Cache set should succeed even when Redis fails"
                
                # Verify data was updated in L1
                retrieved_data = await cache_manager.get(55.7558, 37.6176)
                assert retrieved_data == {"updated": "data"}, "Data should be updated in L1 cache"
                
        finally:
            await cache_manager.cleanup()

    async def test_automatic_redis_reconnection_attempt(self):
        """
        Test that Redis reconnection is attempted when initializing cache manager.
        
        Validates: Requirements 1.4, 9.1
        """
        # Test that Redis initialization is attempted
        with patch('cache.Redis') as mock_redis_class:
            with patch('cache.config.redis.cluster_enabled', False):
                with patch('cache.config.cache.l2_enabled', True):
                    with patch('cache.config.performance.redis_enabled', True):
                        
                        cache_manager = MultiLevelCacheManager()
                        
                        # Wait a bit for async initialization
                        await asyncio.sleep(0.1)
                        
                        # Verify Redis client creation was attempted
                        mock_redis_class.assert_called_once()
                        
                        await cache_manager.cleanup()

    async def test_cache_statistics_during_fallback(self):
        """
        Test that cache statistics are correctly maintained during Redis fallback.
        
        Validates: Requirements 1.4, 9.1
        """
        cache_manager = MultiLevelCacheManager()
        
        try:
            # Mock Redis as unavailable
            with patch.object(cache_manager, '_check_redis_health', return_value=False):
                with patch.object(cache_manager, '_l2_enabled', True):
                    
                    # Perform cache operations
                    await cache_manager.set(55.7558, 37.6176, {"test": "data"}, ttl=300)
                    await cache_manager.get(55.7558, 37.6176)
                    await cache_manager.get(55.7558, 37.6176)  # Second get for hit
                    await cache_manager.get(60.0, 30.0)  # Miss
                    
                    # Check statistics
                    stats = await cache_manager.get_stats()
                    
                    # Should have L1 hits and misses, but no L2 activity
                    assert stats.l1_hits > 0, "Should have L1 cache hits"
                    assert stats.l1_misses > 0, "Should have L1 cache misses"
                    assert stats.l2_hits == 0, "Should have no L2 hits when Redis is unavailable"
                    assert stats.l2_misses > 0, "Should have L2 misses when Redis is unavailable"
                    assert stats.total_requests > 0, "Should track total requests"
                    
        finally:
            await cache_manager.cleanup()

    async def test_configuration_based_fallback_control(self):
        """
        Test that fallback behavior can be controlled via configuration.
        
        Validates: Requirements 1.4, 9.1
        """
        # Test with fallback enabled
        with patch('cache.config.performance.fallback_to_memory', True):
            cache_manager = MultiLevelCacheManager()
            
            try:
                # Should be able to use L1 cache even when Redis is disabled
                with patch.object(cache_manager, '_l2_enabled', False):
                    success = await cache_manager.set(55.7558, 37.6176, {"test": "data"}, ttl=300)
                    assert success, "Cache should work with L1 when fallback is enabled"
                    
                    retrieved_data = await cache_manager.get(55.7558, 37.6176)
                    assert retrieved_data == {"test": "data"}, "Data should be retrievable from L1"
                    
            finally:
                await cache_manager.cleanup()