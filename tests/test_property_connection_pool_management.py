"""
Property-based tests for connection pool management.

**Property 3: Connection Pool Management**
**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

Tests universal properties that should hold for connection pool operations
across all valid configurations and request patterns.
"""

import asyncio
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import Dict, List, Optional
import time
from unittest.mock import AsyncMock, patch, MagicMock

from connection_pool import (
    ConnectionPoolManager, 
    ConnectionPool, 
    ServiceType, 
    APIRequest, 
    PoolConfig,
    ConnectionStatus,
    CircuitBreaker,
    CircuitBreakerState
)


class TestConnectionPoolProperties:
    """Property-based tests for connection pool management"""
    
    @given(
        max_connections=st.integers(min_value=1, max_value=10),
        max_keepalive=st.integers(min_value=1, max_value=5),
        connect_timeout=st.floats(min_value=1.0, max_value=10.0),
        read_timeout=st.floats(min_value=5.0, max_value=20.0)
    )
    @settings(max_examples=5, deadline=3000)
    async def test_pool_configuration_consistency(
        self, max_connections, max_keepalive, connect_timeout, read_timeout
    ):
        """
        **Validates: Requirements 3.1, 3.2**
        
        Property: For any valid pool configuration, the pool should maintain
        the specified connection limits and timeout settings consistently.
        """
        assume(max_keepalive <= max_connections)
        
        config = PoolConfig(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout
        )
        
        # Create pool with mock HTTP client to avoid actual network calls
        with patch('connection_pool.AsyncClient') as mock_client:
            # Mock the client instance methods
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock()
            mock_client_instance.request = AsyncMock()
            mock_client_instance.aclose = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            pool = ConnectionPool(
                service_type=ServiceType.OPEN_METEO,
                base_url="https://test.example.com",
                config=config
            )
            
            try:
                # Verify configuration is applied
                assert pool.config.max_connections == max_connections
                assert pool.config.max_keepalive_connections == max_keepalive
                assert pool.config.connect_timeout == connect_timeout
                assert pool.config.read_timeout == read_timeout
                
                # Verify pool starts in healthy state
                assert pool.status == ConnectionStatus.HEALTHY
                
            finally:
                await pool.cleanup()
    
    @given(
        requests=st.lists(
            st.tuples(
                st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=10),  # A-Z only
                st.dictionaries(
                    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=5),  # a-z only
                    st.integers(min_value=0, max_value=100),
                    min_size=0,
                    max_size=2
                )  # Parameters
            ),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=5, deadline=5000)
    async def test_request_execution_consistency(self, requests):
        """
        **Validates: Requirements 3.1, 3.4, 3.5**
        
        Property: For any sequence of valid API requests, the connection pool
        should execute them consistently and maintain accurate metrics.
        """
        # Mock successful responses at the transport level
        with patch('connection_pool.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"test": "data"}
            mock_response.text = "test response"
            mock_response.headers = {"content-type": "application/json"}
            
            # Mock the client instance methods
            mock_client_instance = AsyncMock()
            mock_client_instance.request = AsyncMock(return_value=mock_response)
            mock_client_instance.aclose = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            manager = ConnectionPoolManager()
            
            try:
                initial_stats = await manager.get_pool_stats(ServiceType.OPEN_METEO)
                initial_requests = initial_stats.total_requests
                
                # Execute all requests
                for path, params in requests:
                    api_request = APIRequest(
                        method="GET",
                        url=f"https://test.example.com/{path}",
                        params=params
                    )
                    
                    response = await manager.execute_request(ServiceType.OPEN_METEO, api_request)
                    
                    # Verify response consistency
                    assert response.status_code == 200
                    assert response.retries >= 0
                    assert response.response_time >= 0
                
                # Verify metrics consistency
                final_stats = await manager.get_pool_stats(ServiceType.OPEN_METEO)
                
                # Total requests should increase by the number of requests made
                assert final_stats.total_requests == initial_requests + len(requests)
                
                # All requests should be successful (mocked)
                assert final_stats.successful_requests >= initial_stats.successful_requests + len(requests)
                
                # Success rate should be reasonable
                if final_stats.total_requests > 0:
                    success_rate = final_stats.calculate_success_rate()
                    assert 0.0 <= success_rate <= 1.0
                
            finally:
                await manager.cleanup()
    
    @given(
        failure_threshold=st.integers(min_value=1, max_value=3),
        recovery_timeout=st.integers(min_value=1, max_value=3)  # Reduced max to speed up tests
    )
    @settings(max_examples=3, deadline=10000)  # Increased deadline to 10 seconds
    async def test_circuit_breaker_behavior(self, failure_threshold, recovery_timeout):
        """
        **Validates: Requirements 3.6, 9.4**
        
        Property: For any circuit breaker configuration, the breaker should
        correctly track failures and prevent requests when threshold is exceeded.
        """
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
        # Initially should allow execution
        assert await circuit_breaker.can_execute() == True
        metrics = await circuit_breaker.get_metrics()
        assert metrics.state == CircuitBreakerState.CLOSED
        
        # Record failures up to threshold
        for i in range(failure_threshold):
            await circuit_breaker.record_failure()
            
            if i < failure_threshold - 1:
                # Should still allow execution before threshold
                assert await circuit_breaker.can_execute() == True
            else:
                # Should open circuit at threshold
                metrics = await circuit_breaker.get_metrics()
                assert metrics.state == CircuitBreakerState.OPEN
                assert await circuit_breaker.can_execute() == False
        
        # Test recovery after timeout
        await asyncio.sleep(recovery_timeout + 0.1)  # Wait for recovery timeout
        
        # Should transition to half-open and allow execution
        assert await circuit_breaker.can_execute() == True
        metrics = await circuit_breaker.get_metrics()
        assert metrics.state == CircuitBreakerState.HALF_OPEN
        
        # Record success should close circuit (need multiple successes)
        for _ in range(3):  # Default success_threshold is 3
            await circuit_breaker.record_success()
        
        metrics = await circuit_breaker.get_metrics()
        assert metrics.state == CircuitBreakerState.CLOSED
        assert await circuit_breaker.can_execute() == True
    
    @given(
        pool_sizes=st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=1,
            max_size=2
        ),
        timeouts=st.lists(
            st.floats(min_value=1.0, max_value=10.0),
            min_size=1,
            max_size=2
        )
    )
    @settings(max_examples=3, deadline=3000)
    async def test_pool_reconfiguration_consistency(self, pool_sizes, timeouts):
        """
        **Validates: Requirements 3.2, 3.7**
        
        Property: For any sequence of valid pool reconfigurations, the pool
        should maintain consistency and apply new settings correctly.
        """
        manager = ConnectionPoolManager()
        
        with patch('connection_pool.AsyncClient') as mock_client:
            # Mock the client instance methods
            mock_client_instance = AsyncMock()
            mock_client_instance.aclose = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            try:
                # Apply different configurations
                for i, (pool_size, timeout) in enumerate(zip(pool_sizes, timeouts)):
                    config = PoolConfig(
                        max_connections=pool_size,
                        connect_timeout=timeout,
                        read_timeout=timeout + 5.0
                    )
                    
                    manager.configure_pool(ServiceType.OPEN_METEO, config)
                    
                    # Verify configuration is applied
                    pool = await manager.get_connection(ServiceType.OPEN_METEO)
                    assert pool.config.max_connections == pool_size
                    assert pool.config.connect_timeout == timeout
                    assert pool.config.read_timeout == timeout + 5.0
                    
                    # Pool should be in healthy state after reconfiguration
                    assert pool.status in [ConnectionStatus.HEALTHY, ConnectionStatus.CONNECTING]
                
            finally:
                await manager.cleanup()
    
    @given(
        concurrent_requests=st.integers(min_value=1, max_value=3),
        pool_size=st.integers(min_value=1, max_value=2)
    )
    @settings(max_examples=2, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_concurrent_request_handling(self, concurrent_requests, pool_size):
        """
        **Validates: Requirements 3.3, 3.4**
        
        Property: For any number of concurrent requests within pool limits,
        all requests should be handled correctly without resource conflicts.
        """
        assume(concurrent_requests <= pool_size * 3)  # Allow more queuing
        
        config = PoolConfig(
            max_connections=pool_size,
            queue_timeout=8.0,  # Longer timeout
            max_queue_size=concurrent_requests + 10,
            max_retries=0  # No retries to speed up tests
        )
        
        with patch('connection_pool.AsyncClient') as mock_client_class:
            # Create a proper mock response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"concurrent": "test"}
            mock_response.headers = {"content-type": "application/json"}
            
            # Mock the client instance
            mock_client_instance = AsyncMock()
            mock_client_instance.request.return_value = mock_response
            mock_client_instance.get.return_value = mock_response
            mock_client_instance.aclose = AsyncMock()
            mock_client_class.return_value = mock_client_instance
            
            manager = ConnectionPoolManager()
            manager.configure_pool(ServiceType.OPEN_METEO, config)
            
            try:
                # Create concurrent requests
                tasks = []
                for i in range(concurrent_requests):
                    api_request = APIRequest(
                        method="GET",
                        url=f"https://test.example.com/concurrent/{i}",
                        params={"request_id": i}
                    )
                    
                    task = asyncio.create_task(
                        manager.execute_request(ServiceType.OPEN_METEO, api_request)
                    )
                    tasks.append(task)
                
                # Wait for all requests to complete
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count successful responses
                successful_responses = 0
                for response in responses:
                    if not isinstance(response, Exception):
                        assert response.status_code == 200
                        successful_responses += 1
                
                # With proper mocking, most requests should succeed
                success_rate = successful_responses / concurrent_requests
                assert success_rate >= 0.8  # At least 80% success rate with better mocking
                
            finally:
                await manager.cleanup()
    
    @given(
        health_check_interval=st.integers(min_value=1, max_value=3),
        max_age=st.integers(min_value=10, max_value=30)
    )
    @settings(max_examples=3, deadline=2000)
    async def test_connection_lifecycle_management(self, health_check_interval, max_age):
        """
        **Validates: Requirements 3.2, 3.5**
        
        Property: For any connection lifecycle configuration, the pool should
        correctly manage connection health checks and recycling.
        """
        config = PoolConfig(
            health_check_interval=health_check_interval,
            connection_max_age=max_age
        )
        
        with patch('connection_pool.AsyncClient') as mock_client:
            # Mock successful health check response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}
            
            # Mock the client instance methods
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.aclose = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            pool = ConnectionPool(
                service_type=ServiceType.OPEN_METEO,
                base_url="https://test.example.com",
                config=config
            )
            
            try:
                # Verify initial state
                assert pool.config.health_check_interval == health_check_interval
                assert pool.config.connection_max_age == max_age
                
                # Perform health check
                health_result = await pool._perform_health_check()
                assert isinstance(health_result, bool)
                
                # Check if connection should be recycled based on age
                initial_time = pool.connection_created_time
                
                # Simulate age by modifying creation time
                pool.connection_created_time = time.time() - max_age - 1
                should_recycle = pool.should_recycle()
                assert should_recycle == True
                
                # Reset time
                pool.connection_created_time = initial_time
                should_recycle = pool.should_recycle()
                assert should_recycle == False
                
            finally:
                await pool.cleanup()


@pytest.mark.asyncio
class TestConnectionPoolIntegration:
    """Integration tests for connection pool management"""
    
    async def test_property_connection_pool_management_integration(self):
        """
        **Property 3: Connection Pool Management**
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
        
        Integration test that validates the complete connection pool management
        property across multiple scenarios.
        """
        # Configure with longer timeouts to avoid queue timeouts
        config = PoolConfig(
            max_connections=2,
            queue_timeout=10.0,  # Longer timeout
            max_queue_size=10,
            max_retries=0  # No retries to speed up tests
        )
        
        with patch('connection_pool.AsyncClient') as mock_client_class:
            # Configure mock for successful responses
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"integration": "test"}
            mock_response.headers = {"content-type": "application/json"}
            
            # Mock the client instance methods
            mock_client_instance = AsyncMock()
            mock_client_instance.request.return_value = mock_response
            mock_client_instance.get.return_value = mock_response
            mock_client_instance.aclose = AsyncMock()
            mock_client_class.return_value = mock_client_instance
            
            manager = ConnectionPoolManager()
            manager.configure_pool(ServiceType.OPEN_METEO, config)
            
            try:
                # Test 1: Basic pool functionality
                request = APIRequest(
                    method="GET",
                    url="https://test.example.com/integration",
                    params={"test": "integration"}
                )
                
                response = await manager.execute_request(ServiceType.OPEN_METEO, request)
                assert response.status_code == 200
                
                # Test 2: Pool statistics
                stats = await manager.get_pool_stats(ServiceType.OPEN_METEO)
                assert stats.total_requests >= 1
                assert stats.successful_requests >= 1
                
                # Test 3: Health check (with proper mock setup)
                health_results = await manager.health_check_all()
                assert "open_meteo" in health_results
                
                # Test 4: Pool reconfiguration
                new_config = PoolConfig(max_connections=5, connect_timeout=15.0)
                manager.configure_pool(ServiceType.OPEN_METEO, new_config)
                
                pool = await manager.get_connection(ServiceType.OPEN_METEO)
                assert pool.config.max_connections == 5
                assert pool.config.connect_timeout == 15.0
                
                print("✓ Property 3: Connection Pool Management - All requirements validated")
                
            finally:
                await manager.cleanup()


if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--tb=short"])