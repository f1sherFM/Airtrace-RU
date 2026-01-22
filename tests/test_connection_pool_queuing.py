"""
Test connection pool queuing and circuit breaker functionality.

Tests the enhanced connection pool implementation with request queuing,
circuit breaker patterns, and comprehensive metrics collection.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock

from connection_pool import (
    ConnectionPool, ConnectionPoolManager, PoolConfig, ServiceType,
    APIRequest, CircuitBreaker, CircuitBreakerState, ConnectionStatus
)


class TestCircuitBreaker:
    """Test enhanced circuit breaker functionality"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1, success_threshold=2)
        
        # Initially closed
        assert await cb.can_execute() == True
        metrics = await cb.get_metrics()
        assert metrics.state == CircuitBreakerState.CLOSED
        
        # Record failures to open circuit
        for _ in range(3):
            await cb.record_failure()
        
        # Should be open now
        assert await cb.can_execute() == False
        metrics = await cb.get_metrics()
        assert metrics.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should transition to half-open
        assert await cb.can_execute() == True
        metrics = await cb.get_metrics()
        assert metrics.state == CircuitBreakerState.HALF_OPEN
        
        # Record enough successes to close
        for _ in range(2):
            await cb.record_success()
        
        metrics = await cb.get_metrics()
        assert metrics.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_rate(self):
        """Test circuit breaker failure rate calculation"""
        cb = CircuitBreaker(failure_threshold=5, failure_rate_threshold=0.6)
        
        # Record mixed success/failure pattern
        for _ in range(3):
            await cb.record_success()
        for _ in range(7):
            await cb.record_failure()
        
        # Should open due to high failure rate (70%)
        metrics = await cb.get_metrics()
        assert metrics.state == CircuitBreakerState.OPEN
        assert metrics.get_failure_rate() == 0.7


class TestConnectionPoolQueuing:
    """Test connection pool request queuing functionality"""
    
    @pytest.fixture
    def pool_config(self):
        """Create test pool configuration"""
        return PoolConfig(
            max_connections=2,
            max_keepalive_connections=1,
            queue_timeout=1.0,
            max_queue_size=5,
            connect_timeout=0.1,
            read_timeout=0.5
        )
    
    @pytest.fixture
    def mock_client(self):
        """Create mock HTTP client"""
        client = AsyncMock()
        response = AsyncMock()
        response.status_code = 200
        response.json.return_value = {"test": "data"}
        response.headers = {"content-type": "application/json"}
        client.request.return_value = response
        client.get.return_value = response
        return client
    
    @pytest.mark.asyncio
    async def test_request_queuing_when_pool_exhausted(self, pool_config, mock_client):
        """Test that requests are queued when connection pool is exhausted"""
        with patch('connection_pool.AsyncClient', return_value=mock_client):
            pool = ConnectionPool(ServiceType.OPEN_METEO, "https://test.api", pool_config)
            
            # Create multiple concurrent requests that exceed pool capacity
            requests = [
                APIRequest("GET", "https://test.api/data", params={"id": i})
                for i in range(5)
            ]
            
            # Execute requests concurrently
            start_time = time.time()
            responses = await asyncio.gather(*[
                pool.execute_request(req) for req in requests
            ])
            execution_time = time.time() - start_time
            
            # All requests should succeed
            assert len(responses) == 5
            for response in responses:
                assert response.status_code == 200
            
            # Get metrics to verify queuing occurred
            metrics = await pool.get_metrics()
            assert metrics.queue_metrics.total_queued > 0
            assert metrics.queue_metrics.total_processed > 0
            
            await pool.cleanup()
    
    @pytest.mark.asyncio
    async def test_queue_timeout_handling(self, pool_config, mock_client):
        """Test queue timeout handling"""
        # Set very short queue timeout
        pool_config.queue_timeout = 0.1
        
        # Mock slow responses
        async def slow_request(*args, **kwargs):
            await asyncio.sleep(0.5)  # Longer than queue timeout
            response = AsyncMock()
            response.status_code = 200
            response.json.return_value = {"test": "data"}
            response.headers = {"content-type": "application/json"}
            return response
        
        mock_client.request.side_effect = slow_request
        
        with patch('connection_pool.AsyncClient', return_value=mock_client):
            pool = ConnectionPool(ServiceType.OPEN_METEO, "https://test.api", pool_config)
            
            # Create requests that will cause queuing
            requests = [
                APIRequest("GET", "https://test.api/data", params={"id": i})
                for i in range(4)
            ]
            
            # Some requests should timeout in queue
            with pytest.raises(Exception, match="timed out in queue"):
                await asyncio.gather(*[
                    pool.execute_request(req) for req in requests
                ])
            
            # Check timeout metrics
            metrics = await pool.get_metrics()
            assert metrics.queue_metrics.total_timeouts > 0
            
            await pool.cleanup()
    
    @pytest.mark.asyncio
    async def test_queue_full_rejection(self, pool_config, mock_client):
        """Test queue full rejection"""
        # Set small queue size
        pool_config.max_queue_size = 2
        
        # Mock slow responses to fill queue
        async def slow_request(*args, **kwargs):
            await asyncio.sleep(1.0)
            response = AsyncMock()
            response.status_code = 200
            response.json.return_value = {"test": "data"}
            response.headers = {"content-type": "application/json"}
            return response
        
        mock_client.request.side_effect = slow_request
        
        with patch('connection_pool.AsyncClient', return_value=mock_client):
            pool = ConnectionPool(ServiceType.OPEN_METEO, "https://test.api", pool_config)
            
            # Create more requests than queue can handle
            requests = [
                APIRequest("GET", "https://test.api/data", params={"id": i})
                for i in range(10)
            ]
            
            # Should get queue full exceptions
            with pytest.raises(Exception, match="Request queue full"):
                await asyncio.gather(*[
                    pool.execute_request(req) for req in requests
                ])
            
            await pool.cleanup()


class TestConnectionPoolMetrics:
    """Test enhanced connection pool metrics"""
    
    @pytest.fixture
    def pool_config(self):
        return PoolConfig(
            max_connections=3,
            queue_timeout=1.0,
            max_queue_size=10
        )
    
    @pytest.mark.asyncio
    async def test_comprehensive_metrics_collection(self, pool_config):
        """Test that all metrics are properly collected"""
        with patch('connection_pool.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock successful response
            response = AsyncMock()
            response.status_code = 200
            response.json.return_value = {"test": "data"}
            response.headers = {"content-type": "application/json"}
            mock_client.request.return_value = response
            mock_client.get.return_value = response
            
            pool = ConnectionPool(ServiceType.OPEN_METEO, "https://test.api", pool_config)
            
            # Execute some requests
            for i in range(5):
                request = APIRequest("GET", "https://test.api/data", params={"id": i})
                await pool.execute_request(request)
            
            # Get metrics
            metrics = await pool.get_metrics()
            
            # Verify basic metrics
            assert metrics.total_requests == 5
            assert metrics.successful_requests == 5
            assert metrics.failed_requests == 0
            assert metrics.avg_response_time >= 0
            
            # Verify queue metrics exist
            assert hasattr(metrics, 'queue_metrics')
            assert hasattr(metrics, 'circuit_breaker_metrics')
            
            # Verify circuit breaker metrics
            cb_metrics = metrics.circuit_breaker_metrics
            assert cb_metrics.state == CircuitBreakerState.CLOSED
            assert cb_metrics.total_requests >= 5
            
            await pool.cleanup()
    
    @pytest.mark.asyncio
    async def test_wait_time_metrics(self, pool_config):
        """Test wait time metrics collection"""
        with patch('connection_pool.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock response with delay to cause queuing
            async def delayed_response(*args, **kwargs):
                await asyncio.sleep(0.1)
                response = AsyncMock()
                response.status_code = 200
                response.json.return_value = {"test": "data"}
                response.headers = {"content-type": "application/json"}
                return response
            
            mock_client.request.side_effect = delayed_response
            mock_client.get.return_value = await delayed_response()
            
            pool = ConnectionPool(ServiceType.OPEN_METEO, "https://test.api", pool_config)
            
            # Execute concurrent requests to cause queuing
            requests = [
                APIRequest("GET", "https://test.api/data", params={"id": i})
                for i in range(6)  # More than max_connections
            ]
            
            await asyncio.gather(*[pool.execute_request(req) for req in requests])
            
            # Check wait time metrics
            metrics = await pool.get_metrics()
            if metrics.queue_metrics.total_queued > 0:
                assert metrics.avg_wait_time >= 0
                assert metrics.max_wait_time >= metrics.min_wait_time
            
            await pool.cleanup()


class TestConnectionPoolIntegration:
    """Test connection pool integration with circuit breaker and queuing"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_queuing(self):
        """Test that open circuit breaker prevents request queuing"""
        config = PoolConfig(max_connections=1, queue_timeout=0.5, max_retries=0)  # No retries to fail faster
        
        with patch('connection_pool.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock failing responses
            mock_client.request.side_effect = Exception("Connection failed")
            mock_client.get.side_effect = Exception("Health check failed")
            
            pool = ConnectionPool(ServiceType.OPEN_METEO, "https://test.api", config)
            
            # Cause circuit breaker to open
            for _ in range(6):  # More than failure threshold
                try:
                    request = APIRequest("GET", "https://test.api/data")
                    await pool.execute_request(request)
                except Exception:
                    pass
            
            # Circuit breaker should be open
            metrics = await pool.get_metrics()
            cb_metrics = metrics.circuit_breaker_metrics
            assert cb_metrics.state == CircuitBreakerState.OPEN
            
            # New requests should fail immediately without queuing
            with pytest.raises(Exception, match="Circuit breaker open"):
                request = APIRequest("GET", "https://test.api/data")
                await pool.execute_request(request)
            
            await pool.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])