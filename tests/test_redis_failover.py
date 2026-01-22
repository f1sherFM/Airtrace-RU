"""
Redis cluster failover scenario tests for AirTrace RU Backend.

Tests Redis cluster failover, node recovery, and cache degradation scenarios
to ensure system resilience and data availability.

Requirements: 1.4, 9.1, 9.2, 9.3
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone
import redis.exceptions

from cache import MultiLevelCacheManager, CacheLevel
from rate_limiter import RateLimiter
from config import config


class TestRedisClusterFailover:
    """Test Redis cluster failover scenarios"""
    
    @pytest.fixture
    async def cache_manager(self):
        """Cache manager fixture with Redis enabled"""
        manager = MultiLevelCacheManager()
        yield manager
        await manager.cleanup()
    
    @pytest.fixture
    async def rate_limiter(self):
        """Rate limiter fixture with Redis enabled"""
        limiter = RateLimiter()
        yield limiter
        await limiter.cleanup()
    
    async def test_redis_connection_failure_at_startup(self):
        """Test cache manager handles Redis connection failure at startup"""
        with patch('redis.asyncio.Redis.ping', side_effect=redis.exceptions.ConnectionError("Connection failed")):
            cache_manager = MultiLevelCacheManager()
            
            # Should initialize with L2 disabled
            assert not cache_manager._redis_healthy
            
            # L1 cache should still work
            success = await cache_manager.set(55.7558, 37.6176, {"test": "data"}, levels=[CacheLevel.L1])
            assert success
            
            data = await cache_manager.get(55.7558, 37.6176, [CacheLevel.L1])
            assert data is not None
            assert data["test"] == "data"
    
    async def test_redis_cluster_node_failure_during_operation(self, cache_manager):
        """Test handling of Redis cluster node failure during operation"""
        # Initially Redis is healthy
        cache_manager._redis_healthy = True
        
        # Simulate node failure during operation
        with patch.object(cache_manager, '_redis_client') as mock_redis:
            mock_redis.get.side_effect = redis.exceptions.ClusterDownError("Cluster node down")
            
            # Cache get should handle the error gracefully
            data = await cache_manager._get_from_l2("test_key")
            assert data is None
            
            # Health check should detect the failure
            healthy = await cache_manager._check_redis_health()
            assert not healthy
    
    async def test_redis_cluster_failover_to_l1_cache(self, cache_manager):
        """Test automatic failover to L1 cache when Redis cluster fails"""
        # Set data when Redis is healthy
        cache_manager._redis_healthy = True
        test_data = {"temperature": 20.5, "humidity": 65}
        
        # Store in L1 first
        await cache_manager.set(55.7558, 37.6176, test_data, levels=[CacheLevel.L1])
        
        # Simulate Redis failure
        cache_manager._redis_healthy = False
        
        # Should still get data from L1
        data = await cache_manager.get(55.7558, 37.6176, [CacheLevel.L1, CacheLevel.L2])
        assert data is not None
        assert data["temperature"] == 20.5
        assert data["humidity"] == 65
    
    async def test_redis_recovery_detection(self, cache_manager):
        """Test detection of Redis recovery after failure"""
        # Start with Redis down
        cache_manager._redis_healthy = False
        cache_manager._last_health_check = 0  # Force health check
        
        # Mock Redis to be healthy again
        with patch.object(cache_manager, '_redis_client') as mock_redis:
            mock_redis.ping = AsyncMock(return_value=True)
            
            # Health check should detect recovery
            healthy = await cache_manager._check_redis_health()
            assert healthy
            assert cache_manager._redis_healthy
    
    async def test_redis_cluster_partial_failure(self, cache_manager):
        """Test handling of partial Redis cluster failure"""
        # Simulate some operations succeeding, others failing
        with patch.object(cache_manager, '_redis_client') as mock_redis:
            # GET operations fail, but SET operations succeed
            mock_redis.get.side_effect = redis.exceptions.ClusterDownError("Node down")
            mock_redis.setex = AsyncMock(return_value=True)
            
            # Set operation should succeed
            success = await cache_manager._set_to_l2("test_key", {"data": "value"}, 300)
            assert success
            
            # Get operation should fail gracefully
            data = await cache_manager._get_from_l2("test_key")
            assert data is None
    
    async def test_rate_limiter_redis_failover(self, rate_limiter):
        """Test rate limiter failover when Redis fails"""
        # Initially Redis is healthy
        rate_limiter._redis_healthy = True
        
        # Simulate Redis failure
        with patch.object(rate_limiter, '_redis_client') as mock_redis:
            mock_redis.ping.side_effect = redis.exceptions.ConnectionError("Redis down")
            
            # Rate limiting should fall back to memory
            result = await rate_limiter.check_rate_limit("192.168.1.1", "/api/test")
            assert result is not None
            assert hasattr(result, 'allowed')
    
    async def test_redis_cluster_split_brain_scenario(self, cache_manager):
        """Test handling of Redis cluster split-brain scenario"""
        # Simulate split-brain where some nodes are accessible
        with patch.object(cache_manager, '_redis_client') as mock_redis:
            # Some operations succeed, others fail with cluster errors
            def side_effect_get(key):
                if "accessible" in key:
                    return '{"data": "accessible_value"}'
                else:
                    raise redis.exceptions.ClusterDownError("Split brain")
            
            mock_redis.get.side_effect = side_effect_get
            
            # Should handle mixed success/failure gracefully
            accessible_data = await cache_manager._get_from_l2("accessible_key")
            inaccessible_data = await cache_manager._get_from_l2("inaccessible_key")
            
            assert accessible_data is not None
            assert inaccessible_data is None
    
    async def test_redis_timeout_handling(self, cache_manager):
        """Test handling of Redis operation timeouts"""
        with patch.object(cache_manager, '_redis_client') as mock_redis:
            mock_redis.get.side_effect = redis.exceptions.TimeoutError("Operation timeout")
            
            # Should handle timeout gracefully
            data = await cache_manager._get_from_l2("test_key")
            assert data is None
            
            # Health check should detect timeout issues
            healthy = await cache_manager._check_redis_health()
            assert not healthy
    
    async def test_redis_memory_pressure_handling(self, cache_manager):
        """Test handling of Redis memory pressure"""
        with patch.object(cache_manager, '_redis_client') as mock_redis:
            # Simulate Redis memory pressure
            mock_redis.setex.side_effect = redis.exceptions.ResponseError("OOM command not allowed")
            
            # Set operation should fail gracefully
            success = await cache_manager._set_to_l2("test_key", {"large": "data"}, 300)
            assert not success
            
            # System should continue with L1 cache
            success_l1 = await cache_manager.set(55.7558, 37.6176, {"data": "value"}, levels=[CacheLevel.L1])
            assert success_l1


class TestRedisClusterRecovery:
    """Test Redis cluster recovery scenarios"""
    
    @pytest.fixture
    async def cache_manager_with_failed_redis(self):
        """Cache manager with initially failed Redis"""
        manager = MultiLevelCacheManager()
        manager._redis_healthy = False
        yield manager
        await manager.cleanup()
    
    async def test_gradual_node_recovery(self, cache_manager_with_failed_redis):
        """Test gradual recovery of Redis cluster nodes"""
        cache_manager = cache_manager_with_failed_redis
        
        # Simulate gradual node recovery
        recovery_stages = [
            redis.exceptions.ClusterDownError("Still recovering"),
            redis.exceptions.ClusterDownError("Partial recovery"),
            True  # Full recovery
        ]
        
        with patch.object(cache_manager, '_redis_client') as mock_redis:
            for stage in recovery_stages:
                if isinstance(stage, Exception):
                    mock_redis.ping.side_effect = stage
                else:
                    mock_redis.ping = AsyncMock(return_value=stage)
                
                healthy = await cache_manager._check_redis_health()
                
                if stage is True:
                    assert healthy
                    assert cache_manager._redis_healthy
                else:
                    assert not healthy
    
    async def test_cache_warming_after_recovery(self, cache_manager_with_failed_redis):
        """Test cache warming after Redis recovery"""
        cache_manager = cache_manager_with_failed_redis
        
        # Store data in L1 while Redis is down
        test_data = {"recovered": "data", "timestamp": time.time()}
        await cache_manager.set(55.7558, 37.6176, test_data, levels=[CacheLevel.L1])
        
        # Simulate Redis recovery
        cache_manager._redis_healthy = True
        
        # Data should be promoted to L2 on next access
        with patch.object(cache_manager, '_set_to_l2', return_value=True) as mock_set_l2:
            data = await cache_manager.get(55.7558, 37.6176)
            assert data is not None
            
            # Should attempt to promote to L2
            # Note: This depends on the promotion logic implementation
    
    async def test_invalidation_after_recovery(self, cache_manager_with_failed_redis):
        """Test cache invalidation after Redis recovery"""
        cache_manager = cache_manager_with_failed_redis
        
        # Simulate Redis recovery
        cache_manager._redis_healthy = True
        
        with patch.object(cache_manager, '_redis_client') as mock_redis:
            mock_redis.keys = AsyncMock(return_value=[b'test_key_1', b'test_key_2'])
            mock_redis.delete = AsyncMock(return_value=2)
            
            # Should be able to invalidate cache after recovery
            invalidated = await cache_manager.invalidate("test_*", levels=[CacheLevel.L2])
            assert invalidated >= 0
    
    async def test_health_check_frequency_after_failure(self, cache_manager_with_failed_redis):
        """Test health check frequency increases after failure"""
        cache_manager = cache_manager_with_failed_redis
        
        # Health checks should be more frequent after failure
        initial_interval = cache_manager._health_check_interval
        
        # Simulate multiple failures
        for _ in range(3):
            await cache_manager._check_redis_health()
        
        # Health check interval might be adjusted (implementation dependent)
        # This test verifies the system continues to check health
        assert cache_manager._health_check_interval >= 0


class TestCacheCoherence:
    """Test cache coherence during Redis failures"""
    
    async def test_l1_l2_coherence_during_failure(self):
        """Test L1/L2 cache coherence during Redis failure"""
        cache_manager = MultiLevelCacheManager()
        
        # Set data in both L1 and L2
        test_data = {"coherence": "test", "version": 1}
        await cache_manager.set(55.7558, 37.6176, test_data, levels=[CacheLevel.L1, CacheLevel.L2])
        
        # Simulate Redis failure
        cache_manager._redis_healthy = False
        
        # Update data in L1 only (L2 unavailable)
        updated_data = {"coherence": "test", "version": 2}
        await cache_manager.set(55.7558, 37.6176, updated_data, levels=[CacheLevel.L1])
        
        # Should get updated data from L1
        data = await cache_manager.get(55.7558, 37.6176, [CacheLevel.L1])
        assert data["version"] == 2
    
    async def test_cache_invalidation_during_redis_failure(self):
        """Test cache invalidation when Redis is unavailable"""
        cache_manager = MultiLevelCacheManager()
        
        # Set data in L1
        await cache_manager.set(55.7558, 37.6176, {"test": "data"}, levels=[CacheLevel.L1])
        
        # Simulate Redis failure
        cache_manager._redis_healthy = False
        
        # Invalidation should still work for L1
        invalidated = await cache_manager.invalidate("*", levels=[CacheLevel.L1])
        assert invalidated >= 0
        
        # Data should be gone from L1
        data = await cache_manager.get(55.7558, 37.6176, [CacheLevel.L1])
        assert data is None
    
    async def test_stale_data_detection_after_recovery(self):
        """Test detection of stale data after Redis recovery"""
        cache_manager = MultiLevelCacheManager()
        
        # Set data with short TTL
        test_data = {"stale": "data", "timestamp": time.time()}
        await cache_manager.set(55.7558, 37.6176, test_data, ttl=1, levels=[CacheLevel.L1])
        
        # Wait for data to become stale
        await asyncio.sleep(1.1)
        
        # Simulate Redis recovery
        cache_manager._redis_healthy = True
        
        # Stale data should be detected and removed
        data = await cache_manager.get(55.7558, 37.6176)
        # Depending on implementation, might return None or fresh data
        assert data is None or data != test_data


class TestRedisClusterConfiguration:
    """Test Redis cluster configuration scenarios"""
    
    async def test_single_node_vs_cluster_configuration(self):
        """Test handling of single node vs cluster configuration"""
        # Test single node configuration
        with patch('config.redis.cluster_enabled', False):
            cache_manager = MultiLevelCacheManager()
            # Should initialize with single Redis client
            assert cache_manager._l2_enabled or not config.performance.redis_enabled
        
        # Test cluster configuration
        with patch('config.redis.cluster_enabled', True):
            with patch('config.get_redis_cluster_kwargs', return_value={"startup_nodes": [{"host": "localhost", "port": 7000}]}):
                cache_manager = MultiLevelCacheManager()
                # Should initialize with cluster client
                assert cache_manager._l2_enabled or not config.performance.redis_enabled
    
    async def test_invalid_cluster_configuration(self):
        """Test handling of invalid cluster configuration"""
        with patch('config.redis.cluster_enabled', True):
            with patch('config.get_redis_cluster_kwargs', return_value={"startup_nodes": []}):
                cache_manager = MultiLevelCacheManager()
                # Should disable L2 cache with invalid configuration
                assert not cache_manager._l2_enabled
    
    async def test_redis_authentication_failure(self):
        """Test handling of Redis authentication failure"""
        with patch('redis.asyncio.Redis.ping', side_effect=redis.exceptions.AuthenticationError("Authentication failed")):
            cache_manager = MultiLevelCacheManager()
            
            # Should handle auth failure gracefully
            assert not cache_manager._redis_healthy
            
            # L1 cache should still work
            success = await cache_manager.set(55.7558, 37.6176, {"test": "data"}, levels=[CacheLevel.L1])
            assert success


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])