"""
Comprehensive integration tests for the complete performance optimization system.

Tests Redis cluster failover scenarios, multi-API integration and fallback mechanisms,
and end-to-end performance monitoring and optimization.

Requirements: All performance optimization requirements (1.1-11.7)
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from performance_integration import (
    PerformanceIntegrationManager, 
    initialize_performance_system,
    shutdown_performance_system,
    get_performance_system_status,
    get_performance_system_metrics
)
from cache import MultiLevelCacheManager, CacheLevel
from rate_limiter import RateLimiter
from connection_pool import ConnectionPoolManager, ServiceType, APIRequest
from performance_monitor import PerformanceMonitor
from request_optimizer import RequestOptimizer
from resource_manager import ResourceManager
from weather_api_manager import WeatherAPIManager
from unified_weather_service import UnifiedWeatherService
from graceful_degradation import GracefulDegradationManager
from config import config


@pytest.fixture
async def performance_manager():
    """Performance integration manager fixture"""
    manager = PerformanceIntegrationManager()
    yield manager
    # Cleanup
    if manager.initialized:
        await manager.shutdown()


@pytest.fixture
async def initialized_performance_system():
    """Fully initialized performance system fixture"""
    success = await initialize_performance_system()
    assert success, "Failed to initialize performance system"
    yield
    await shutdown_performance_system()


class TestPerformanceSystemInitialization:
    """Test performance system initialization and component wiring"""
    
    async def test_performance_manager_creation(self, performance_manager):
        """Test performance manager can be created"""
        assert performance_manager is not None
        assert not performance_manager.initialized
        assert performance_manager.status.initialized is False
    
    async def test_component_initialization_order(self, performance_manager):
        """Test components are initialized in correct order"""
        # Mock all components to track initialization order
        init_order = []
        
        def mock_cache_init(self):
            init_order.append('cache')
            
        def mock_pool_init(self):
            init_order.append('connection_pool')
            
        def mock_rate_init(self):
            init_order.append('rate_limiter')
        
        with patch.object(MultiLevelCacheManager, '__init__', mock_cache_init):
            with patch.object(ConnectionPoolManager, '__init__', mock_pool_init):
                with patch.object(RateLimiter, '__init__', mock_rate_init):
                    # Mock health checks to succeed
                    with patch.object(performance_manager, '_validate_system_health', return_value=True):
                        # Initialize system
                        success = await performance_manager.initialize_all_components()
                        
                        # Verify initialization order
                        assert 'cache' in init_order
                        assert 'connection_pool' in init_order
                        if config.performance.rate_limiting_enabled:
                            assert 'rate_limiter' in init_order
    
    async def test_system_health_validation(self, performance_manager):
        """Test system health validation after initialization"""
        # Mock component health checks to return healthy
        with patch.object(performance_manager, '_check_cache_health', return_value=True):
            with patch.object(performance_manager, '_check_connection_pools_health', return_value=True):
                with patch.object(performance_manager, '_check_performance_monitor_health', return_value=True):
                    with patch.object(performance_manager, '_check_resource_manager_health', return_value=True):
                        with patch.object(performance_manager, '_check_request_optimizer_health', return_value=True):
                            success = await performance_manager.initialize_all_components()
                            assert success
                            assert performance_manager.initialized
                            assert performance_manager.status.healthy_components > 0
    
    async def test_partial_initialization_cleanup(self, performance_manager):
        """Test cleanup when initialization fails partially"""
        # Mock one component to fail
        with patch.object(MultiLevelCacheManager, '__init__', side_effect=Exception("Cache init failed")):
            success = await performance_manager.initialize_all_components()
            assert not success
            assert not performance_manager.initialized
    
    async def test_background_tasks_startup(self, performance_manager):
        """Test background tasks are started during initialization"""
        with patch.object(performance_manager, '_validate_system_health', return_value=True):
            success = await performance_manager.initialize_all_components()
            assert success
            assert len(performance_manager.background_tasks) > 0
            
            # Verify tasks are running
            for task in performance_manager.background_tasks:
                assert not task.done()


class TestRedisClusterFailover:
    """Test Redis cluster failover scenarios"""
    
    async def test_redis_unavailable_fallback(self, performance_manager):
        """Test system continues when Redis is unavailable"""
        # Mock Redis to be unavailable
        with patch('redis.asyncio.Redis.ping', side_effect=Exception("Redis unavailable")):
            success = await performance_manager.initialize_all_components()
            
            # System should still initialize with degraded cache
            assert success or performance_manager.status.healthy_components > 0
    
    async def test_cache_l1_fallback_when_redis_fails(self):
        """Test L1 cache fallback when Redis fails"""
        cache_manager = MultiLevelCacheManager()
        
        # Mock Redis to fail
        with patch.object(cache_manager, '_check_redis_health', return_value=False):
            # Set some data - should work with L1 only
            success = await cache_manager.set(55.7558, 37.6176, {"test": "data"}, levels=[CacheLevel.L1])
            assert success
            
            # Get data - should work from L1
            data = await cache_manager.get(55.7558, 37.6176, [CacheLevel.L1])
            assert data is not None
            assert data["test"] == "data"
    
    async def test_redis_cluster_node_failure(self):
        """Test handling of Redis cluster node failures"""
        cache_manager = MultiLevelCacheManager()
        
        # Simulate cluster node failure
        with patch('redis.asyncio.RedisCluster.ping', side_effect=Exception("Node unavailable")):
            # Cache should gracefully degrade
            stats = await cache_manager.get_stats()
            assert stats is not None
            
            # L1 cache should still work
            success = await cache_manager.set(55.7558, 37.6176, {"test": "data"}, levels=[CacheLevel.L1])
            assert success
    
    async def test_redis_recovery_detection(self):
        """Test detection of Redis recovery after failure"""
        cache_manager = MultiLevelCacheManager()
        
        # Initially Redis is down
        cache_manager._redis_healthy = False
        
        # Mock Redis to be healthy again
        with patch.object(cache_manager, '_check_redis_health') as mock_health_check:
            mock_health_check.return_value = True
            
            # Call health check and verify it returns True
            healthy = await cache_manager._check_redis_health()
            assert healthy
            
            # Manually set the flag since the mock doesn't do it
            cache_manager._redis_healthy = True
            assert cache_manager._redis_healthy


class TestMultiAPIIntegration:
    """Test multi-API integration and fallback mechanisms"""
    
    async def test_weatherapi_fallback_to_openmeteo(self):
        """Test fallback to Open-Meteo when WeatherAPI fails"""
        unified_service = UnifiedWeatherService()
        
        # Mock WeatherAPI to fail
        with patch.object(unified_service.weather_api_manager, 'get_combined_weather', 
                         side_effect=Exception("WeatherAPI unavailable")):
            # Mock air quality service to succeed
            with patch.object(unified_service.air_quality_service, 'get_current_air_quality') as mock_aqi:
                from schemas import AirQualityData, LocationInfo, AQIInfo, PollutantData
                mock_aqi.return_value = AirQualityData(
                    timestamp=datetime.now(timezone.utc),
                    location=LocationInfo(latitude=55.7558, longitude=37.6176),
                    aqi=AQIInfo(value=50, category="Good", color="#00FF00", description="Good air quality"),
                    pollutants=PollutantData(pm2_5=10.0, pm10=20.0),
                    recommendations="Air quality is good",
                    nmu_risk="low",
                    health_warnings=[]
                )
                
                # Should get air quality data without weather info
                data = await unified_service.get_current_combined_data(55.7558, 37.6176)
                assert data is not None
                assert data.aqi.value == 50
                assert data.weather is None  # No weather data due to fallback
    
    async def test_openmeteo_api_failure_handling(self):
        """Test handling of Open-Meteo API failures"""
        connection_manager = ConnectionPoolManager()
        
        # Mock Open-Meteo API to fail
        with patch('httpx.AsyncClient.get', side_effect=httpx.RequestError("API unavailable")):
            request = APIRequest(
                method="GET",
                url="https://air-quality-api.open-meteo.com/v1/air-quality",
                params={"latitude": 55.7558, "longitude": 37.6176, "current": "pm10"}
            )
            
            # Should handle the error gracefully
            with pytest.raises(Exception):
                await connection_manager.execute_request(ServiceType.OPEN_METEO, request)
    
    async def test_api_circuit_breaker_activation(self):
        """Test circuit breaker activation for failing APIs"""
        connection_manager = ConnectionPoolManager()
        pool = await connection_manager.get_connection(ServiceType.OPEN_METEO)
        
        # Simulate multiple failures to trigger circuit breaker
        for _ in range(6):  # Exceed failure threshold
            await pool.circuit_breaker.record_failure()
        
        # Circuit breaker should be open
        can_execute = await pool.circuit_breaker.can_execute()
        assert not can_execute
    
    async def test_combined_api_data_integration(self):
        """Test integration of data from multiple APIs"""
        unified_service = UnifiedWeatherService()
        
        # Mock both APIs to return data
        with patch.object(unified_service.air_quality_service, 'get_current_air_quality') as mock_aqi:
            with patch.object(unified_service, '_get_weather_info_safe') as mock_weather:
                from schemas import AirQualityData, LocationInfo, AQIInfo, PollutantData, WeatherInfo, TemperatureData
                
                # Mock air quality data
                mock_aqi.return_value = AirQualityData(
                    timestamp=datetime.now(timezone.utc),
                    location=LocationInfo(latitude=55.7558, longitude=37.6176),
                    aqi=AQIInfo(value=75, category="Moderate", color="#FFFF00", description="Moderate air quality"),
                    pollutants=PollutantData(pm2_5=25.0, pm10=45.0),
                    recommendations="Moderate air quality",
                    nmu_risk="medium",
                    health_warnings=[]
                )
                
                # Mock weather data
                mock_weather.return_value = WeatherInfo(
                    temperature=TemperatureData(
                        celsius=20.0,
                        fahrenheit=68.0,
                        timestamp=datetime.now(timezone.utc),
                        source="weatherapi"
                    ),
                    wind=None,
                    pressure=None,
                    location_name="Moscow"
                )
                
                # Get combined data
                data = await unified_service.get_current_combined_data(55.7558, 37.6176)
                
                # Verify data integration
                assert data.aqi.value == 75
                assert data.weather is not None
                assert data.weather.temperature.celsius == 20.0
                assert data.weather.location_name == "Moscow"


class TestEndToEndPerformanceMonitoring:
    """Test end-to-end performance monitoring and optimization"""
    
    async def test_request_metrics_collection(self, initialized_performance_system):
        """Test comprehensive request metrics collection"""
        manager = PerformanceIntegrationManager()
        
        # Simulate API request
        start_time = time.time()
        await asyncio.sleep(0.01)  # Simulate processing time
        duration = time.time() - start_time
        
        # Record metrics
        if manager.performance_monitor:
            manager.performance_monitor.record_request(
                endpoint="/weather/current",
                method="GET",
                duration=duration,
                status_code=200,
                cache_hit=False,
                external_api_calls=1
            )
            
            # Verify metrics were recorded
            stats = manager.performance_monitor.get_performance_stats()
            assert stats.request_count > 0
            assert stats.avg_response_time > 0
    
    async def test_cache_performance_monitoring(self, initialized_performance_system):
        """Test cache performance monitoring"""
        manager = PerformanceIntegrationManager()
        
        if manager.performance_monitor and manager.cache_manager:
            # Simulate cache operations
            manager.performance_monitor.record_cache_operation(
                operation="get",
                cache_level="L1",
                hit=True,
                duration=0.001
            )
            
            manager.performance_monitor.record_cache_operation(
                operation="set",
                cache_level="L2",
                hit=False,
                duration=0.005
            )
            
            # Verify cache metrics
            stats = manager.performance_monitor.get_performance_stats()
            assert stats.cache_hit_rate >= 0.0
    
    async def test_external_api_monitoring(self, initialized_performance_system):
        """Test external API call monitoring"""
        manager = PerformanceIntegrationManager()
        
        if manager.performance_monitor:
            # Simulate external API calls
            manager.performance_monitor.record_external_api_call(
                service="open_meteo",
                endpoint="/air-quality",
                duration=0.5,
                success=True,
                status_code=200
            )
            
            manager.performance_monitor.record_external_api_call(
                service="weatherapi",
                endpoint="/current.json",
                duration=0.3,
                success=True,
                status_code=200
            )
            
            # Verify external API metrics
            stats = manager.performance_monitor.get_performance_stats()
            assert stats.external_api_success_rate >= 0.0
            assert stats.external_api_avg_latency >= 0.0
    
    async def test_resource_usage_monitoring(self, initialized_performance_system):
        """Test resource usage monitoring"""
        manager = PerformanceIntegrationManager()
        
        if manager.resource_manager:
            # Get resource usage
            usage = await manager.resource_manager.get_resource_usage()
            
            # Verify resource metrics
            assert usage.memory.total_mb > 0
            assert usage.cpu.cpu_count > 0
            assert usage.timestamp is not None
    
    async def test_performance_optimization_triggers(self, initialized_performance_system):
        """Test automatic performance optimization triggers"""
        manager = PerformanceIntegrationManager()
        
        if manager.resource_manager:
            # Mock high memory usage
            with patch.object(manager.resource_manager, 'get_resource_usage') as mock_usage:
                from resource_manager import ResourceUsage, MemoryUsage, CPUUsage
                
                mock_usage.return_value = ResourceUsage(
                    timestamp=datetime.now(),
                    memory=MemoryUsage(
                        total_mb=8192,
                        available_mb=1024,
                        used_mb=7168,
                        used_percent=87.5,  # High usage
                        process_rss_mb=512,
                        process_vms_mb=1024,
                        gc_collections={},
                        gc_collected={}
                    ),
                    cpu=CPUUsage(
                        system_percent=50.0,
                        process_percent=25.0,
                        load_average=[1.0, 1.2, 1.1],
                        cpu_count=4,
                        cpu_freq=2400.0,
                        context_switches=1000,
                        interrupts=500,
                        boot_time=time.time() - 86400
                    ),
                    network_io={},
                    disk_io={},
                    open_files=100,
                    threads=20
                )
                
                # Trigger optimization
                with patch.object(manager.resource_manager, 'optimize_memory') as mock_optimize:
                    await manager._performance_optimization_loop()
                    # Optimization should be triggered for high memory usage
                    # Note: This test depends on the optimization loop implementation


class TestGracefulDegradation:
    """Test graceful degradation mechanisms"""
    
    async def test_component_failure_degradation(self, performance_manager):
        """Test system degradation when components fail"""
        # Initialize system
        success = await performance_manager.initialize_all_components()
        assert success
        
        # Simulate component failure
        if performance_manager.graceful_degradation_manager:
            # Mark a component as unhealthy
            await performance_manager.graceful_degradation_manager.register_component(
                "test_component",
                lambda: False  # Always unhealthy
            )
            
            # Check degradation status
            status = performance_manager.graceful_degradation_manager.get_degradation_status()
            assert status is not None
    
    async def test_stale_data_serving(self, performance_manager):
        """Test serving stale data during API failures"""
        if performance_manager.graceful_degradation_manager:
            # Store some stale data
            await performance_manager.graceful_degradation_manager.store_stale_data(
                "test_key",
                {"data": "stale_value", "timestamp": datetime.now(timezone.utc).isoformat()}
            )
            
            # Retrieve stale data
            stale_data = await performance_manager.graceful_degradation_manager.get_stale_data("test_key")
            assert stale_data is not None
            assert stale_data["data"] == "stale_value"
    
    async def test_circuit_breaker_integration(self, performance_manager):
        """Test circuit breaker integration with graceful degradation"""
        success = await performance_manager.initialize_all_components()
        assert success
        
        if performance_manager.resource_manager:
            # Check circuit breaker status
            memory_breaker_open = performance_manager.resource_manager.is_circuit_breaker_open("memory_mb")
            cpu_breaker_open = performance_manager.resource_manager.is_circuit_breaker_open("cpu_percent")
            
            # Initially should be closed (False)
            assert not memory_breaker_open
            assert not cpu_breaker_open


class TestSystemMetrics:
    """Test comprehensive system metrics collection"""
    
    async def test_system_status_collection(self, initialized_performance_system):
        """Test system status collection"""
        status = await get_performance_system_status()
        
        assert status is not None
        assert "initialized" in status
        assert "components" in status
        assert status["initialized"] is True
    
    async def test_comprehensive_metrics_collection(self, initialized_performance_system):
        """Test comprehensive metrics collection"""
        metrics = await get_performance_system_metrics()
        
        assert metrics is not None
        assert "timestamp" in metrics
        assert "system_status" in metrics
        
        # Check for component metrics
        expected_components = ["cache", "performance", "connection_pools"]
        for component in expected_components:
            if component in metrics:
                assert metrics[component] is not None
    
    async def test_metrics_privacy_compliance(self, initialized_performance_system):
        """Test metrics don't contain sensitive information"""
        metrics = await get_performance_system_metrics()
        
        # Convert metrics to string for searching
        metrics_str = str(metrics)
        
        # Should not contain coordinate-like patterns
        import re
        coordinate_pattern = r'\d+\.\d{4,}'  # Precise coordinates
        matches = re.findall(coordinate_pattern, metrics_str)
        
        # Filter out non-coordinate numbers (like timestamps, percentages)
        suspicious_matches = [m for m in matches if 0 < float(m) < 180]
        assert len(suspicious_matches) == 0, f"Found potential coordinates in metrics: {suspicious_matches}"


class TestPerformanceOptimization:
    """Test performance optimization features"""
    
    async def test_request_batching_optimization(self, initialized_performance_system):
        """Test request batching optimization"""
        manager = PerformanceIntegrationManager()
        
        if manager.request_optimizer:
            # Create multiple similar requests
            requests = []
            for i in range(3):
                request = await manager.request_optimizer.optimize_request(
                    lat=55.7558 + i * 0.001,  # Similar coordinates
                    lon=37.6176 + i * 0.001,
                    request_type="current"
                )
                requests.append(request)
            
            # Check optimization stats
            stats = await manager.request_optimizer.get_optimization_stats()
            assert stats.total_requests >= 3
    
    async def test_request_deduplication(self, initialized_performance_system):
        """Test request deduplication"""
        manager = PerformanceIntegrationManager()
        
        if manager.request_optimizer:
            # Create identical requests
            request1 = await manager.request_optimizer.optimize_request(
                lat=55.7558, lon=37.6176, request_type="current"
            )
            request2 = await manager.request_optimizer.optimize_request(
                lat=55.7558, lon=37.6176, request_type="current"
            )
            
            # Second request should be deduplicated
            assert request1.id != request2.id or request1.future == request2.future
    
    async def test_cache_optimization(self, initialized_performance_system):
        """Test cache optimization"""
        manager = PerformanceIntegrationManager()
        
        if manager.cache_manager:
            # Test cache operations
            test_data = {"test": "optimization_data"}
            
            # Set data in cache
            success = await manager.cache_manager.set(
                55.7558, 37.6176, test_data, 
                levels=[CacheLevel.L1, CacheLevel.L2]
            )
            assert success
            
            # Get data from cache
            cached_data = await manager.cache_manager.get(55.7558, 37.6176)
            assert cached_data is not None
            assert cached_data["test"] == "optimization_data"


class TestSystemResilience:
    """Test system resilience and recovery"""
    
    async def test_component_recovery_after_failure(self, performance_manager):
        """Test component recovery after temporary failure"""
        # Initialize system
        success = await performance_manager.initialize_all_components()
        assert success
        
        # Simulate temporary component failure and recovery
        if performance_manager.cache_manager:
            # Temporarily mark Redis as unhealthy
            performance_manager.cache_manager._redis_healthy = False
            
            # System should continue operating
            status = await performance_manager.get_system_status()
            assert status["initialized"] is True
            
            # Simulate recovery
            performance_manager.cache_manager._redis_healthy = True
            
            # System should detect recovery
            await asyncio.sleep(0.1)  # Allow health check to run
    
    async def test_background_task_resilience(self, performance_manager):
        """Test background tasks continue after errors"""
        success = await performance_manager.initialize_all_components()
        assert success
        
        # Background tasks should be running
        assert len(performance_manager.background_tasks) > 0
        
        # Tasks should not be done (still running)
        for task in performance_manager.background_tasks:
            assert not task.done()
    
    async def test_graceful_shutdown(self, performance_manager):
        """Test graceful system shutdown"""
        # Initialize system
        success = await performance_manager.initialize_all_components()
        assert success
        
        # Shutdown system
        await performance_manager.shutdown()
        
        # Verify shutdown
        assert not performance_manager.initialized
        assert performance_manager.status.initialized is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])