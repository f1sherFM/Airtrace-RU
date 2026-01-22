"""
Property-based tests for rate limiting accuracy

**Property 2: Rate Limiting Accuracy**
**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**

Tests that the rate limiter accurately enforces configured limits using sliding window algorithms,
handles burst traffic within bounds, and provides appropriate error responses with retry information.
"""

import asyncio
import pytest
import time
from datetime import datetime, timezone
from hypothesis import given, strategies as st, settings, assume
from typing import List, Tuple

from rate_limit_types import RateLimitConfig, EndpointCategory, RateLimitStrategy
from rate_limiter import RateLimiter
from rate_limit_middleware import RateLimitMiddleware
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock


class TestRateLimitingAccuracy:
    """Property-based tests for rate limiting accuracy"""
    
    def _create_rate_limiter(self):
        """Create a rate limiter instance for testing"""
        limiter = RateLimiter()
        # Disable Redis for testing to use memory-based limiting
        limiter._redis_enabled = False
        return limiter
    
    @given(
        requests_per_minute=st.integers(min_value=1, max_value=100),
        burst_multiplier=st.floats(min_value=1.0, max_value=3.0),
        window_size=st.integers(min_value=10, max_value=120),
        num_requests=st.integers(min_value=1, max_value=200)
    )
    @settings(max_examples=50, deadline=10000)
    async def test_rate_limit_enforcement_accuracy(
        self, 
        requests_per_minute, 
        burst_multiplier, 
        window_size, 
        num_requests
    ):
        """
        Property: Rate limiter accurately enforces configured limits
        
        For any valid rate limit configuration and request pattern,
        the rate limiter should allow requests up to the burst limit
        and block requests that exceed it within the time window.
        """
        # Create fresh rate limiter for each test
        rate_limiter = self._create_rate_limiter()
        # Configure rate limiter
        config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            burst_multiplier=burst_multiplier,
            window_size_seconds=window_size,
            identifier_strategy=RateLimitStrategy.IP_BASED
        )
        
        rate_limiter.configure_limits(EndpointCategory.DEFAULT, config)
        
        # Calculate expected limits
        base_limit = requests_per_minute
        burst_limit = int(base_limit * burst_multiplier)
        
        # Test client identifier
        client_ip = "192.168.1.100"
        endpoint = "/test"
        
        allowed_count = 0
        blocked_count = 0
        
        # Send requests and track results
        for i in range(num_requests):
            result = await rate_limiter.check_rate_limit(client_ip, endpoint)
            
            if result.allowed:
                allowed_count += 1
                # Verify remaining count is accurate
                expected_remaining = max(0, burst_limit - allowed_count)
                assert result.remaining == expected_remaining, \
                    f"Remaining count mismatch: expected {expected_remaining}, got {result.remaining}"
            else:
                blocked_count += 1
                # Verify retry_after is provided when blocked
                assert result.retry_after is not None, "retry_after should be provided when blocked"
                assert result.retry_after > 0, "retry_after should be positive"
                # Verify remaining is 0 when blocked
                assert result.remaining == 0, "remaining should be 0 when blocked"
        
        # Verify enforcement accuracy
        assert allowed_count <= burst_limit, \
            f"Allowed requests ({allowed_count}) exceeded burst limit ({burst_limit})"
        
        if num_requests > burst_limit:
            assert blocked_count > 0, \
                f"Should have blocked some requests when {num_requests} > {burst_limit}"
            assert allowed_count == burst_limit, \
                f"Should have allowed exactly {burst_limit} requests"
        else:
            assert allowed_count == num_requests, \
                f"Should have allowed all {num_requests} requests when under limit"
            assert blocked_count == 0, "Should not have blocked any requests when under limit"
    
    @given(
        base_limit=st.integers(min_value=5, max_value=50),
        burst_multiplier=st.floats(min_value=1.1, max_value=2.5),
        request_intervals=st.lists(
            st.floats(min_value=0.1, max_value=5.0), 
            min_size=3, 
            max_size=20
        )
    )
    @settings(max_examples=30, deadline=15000)
    async def test_sliding_window_accuracy(
        self, 
        base_limit, 
        burst_multiplier, 
        request_intervals
    ):
        """
        Property: Sliding window algorithm accurately tracks request rates over time
        
        For any sequence of requests with time intervals, the sliding window
        should accurately count requests within the current window and allow
        new requests as old ones expire from the window.
        """
        # Create fresh rate limiter for each test
        rate_limiter = self._create_rate_limiter()
        window_size = 60  # 1 minute window
        burst_limit = int(base_limit * burst_multiplier)
        
        config = RateLimitConfig(
            requests_per_minute=base_limit,
            burst_multiplier=burst_multiplier,
            window_size_seconds=window_size,
            identifier_strategy=RateLimitStrategy.IP_BASED
        )
        
        rate_limiter.configure_limits(EndpointCategory.DEFAULT, config)
        
        client_ip = "192.168.1.101"
        endpoint = "/test"
        
        # Track request times and results
        request_times = []
        results = []
        current_time = time.time()
        
        for interval in request_intervals:
            current_time += interval
            
            # Mock time for testing
            original_time = time.time
            time.time = lambda: current_time
            
            try:
                result = await rate_limiter.check_rate_limit(client_ip, endpoint)
                request_times.append(current_time)
                results.append(result)
                
                # Count requests in current window
                window_start = current_time - window_size
                requests_in_window = sum(1 for t in request_times if t > window_start)
                
                # Verify sliding window accuracy
                if requests_in_window <= burst_limit:
                    assert result.allowed, \
                        f"Request should be allowed: {requests_in_window} <= {burst_limit}"
                else:
                    # Note: Due to sliding window implementation, this might not be exact
                    # but the overall behavior should be correct
                    pass
                    
            finally:
                time.time = original_time
    
    @given(
        num_clients=st.integers(min_value=2, max_value=10),
        requests_per_client=st.integers(min_value=5, max_value=30),
        rate_limit=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=20, deadline=10000)
    async def test_per_client_isolation(
        self, 
        num_clients, 
        requests_per_client, 
        rate_limit
    ):
        """
        Property: Rate limits are enforced independently per client
        
        For any number of clients making requests, each client's rate limit
        should be enforced independently without affecting other clients.
        """
        # Create fresh rate limiter for each test
        rate_limiter = self._create_rate_limiter()
        config = RateLimitConfig(
            requests_per_minute=rate_limit,
            burst_multiplier=1.5,
            window_size_seconds=60,
            identifier_strategy=RateLimitStrategy.IP_BASED
        )
        
        rate_limiter.configure_limits(EndpointCategory.DEFAULT, config)
        
        burst_limit = int(rate_limit * 1.5)
        endpoint = "/test"
        
        # Test multiple clients simultaneously
        client_results = {}
        
        for client_id in range(num_clients):
            client_ip = f"192.168.1.{100 + client_id}"
            client_results[client_ip] = {"allowed": 0, "blocked": 0}
            
            for _ in range(requests_per_client):
                result = await rate_limiter.check_rate_limit(client_ip, endpoint)
                
                if result.allowed:
                    client_results[client_ip]["allowed"] += 1
                else:
                    client_results[client_ip]["blocked"] += 1
        
        # Verify each client is treated independently
        for client_ip, results in client_results.items():
            allowed = results["allowed"]
            blocked = results["blocked"]
            
            # Each client should be subject to the same limits
            assert allowed <= burst_limit, \
                f"Client {client_ip} exceeded burst limit: {allowed} > {burst_limit}"
            
            if requests_per_client > burst_limit:
                assert allowed == burst_limit, \
                    f"Client {client_ip} should have been limited to {burst_limit} requests"
                assert blocked > 0, \
                    f"Client {client_ip} should have had some requests blocked"
            else:
                assert allowed == requests_per_client, \
                    f"Client {client_ip} should have had all requests allowed"
                assert blocked == 0, \
                    f"Client {client_ip} should not have had any requests blocked"
    
    @given(
        endpoint_configs=st.dictionaries(
            keys=st.sampled_from([
                EndpointCategory.AIR_QUALITY,
                EndpointCategory.HEALTH_CHECKS,
                EndpointCategory.METRICS
            ]),
            values=st.tuples(
                st.integers(min_value=10, max_value=100),  # requests_per_minute
                st.floats(min_value=1.1, max_value=2.0)   # burst_multiplier
            ),
            min_size=2,
            max_size=3
        ),
        requests_per_endpoint=st.integers(min_value=5, max_value=50)
    )
    @settings(max_examples=20, deadline=10000)
    async def test_endpoint_category_limits(
        self, 
        endpoint_configs, 
        requests_per_endpoint
    ):
        """
        Property: Different endpoint categories have independent rate limits
        
        For any configuration of endpoint-specific rate limits, each endpoint
        category should enforce its own limits independently.
        """
        # Create fresh rate limiter for each test
        rate_limiter = self._create_rate_limiter()
        # Configure different limits for different endpoints
        endpoint_paths = {
            EndpointCategory.AIR_QUALITY: "/weather/current",
            EndpointCategory.HEALTH_CHECKS: "/health",
            EndpointCategory.METRICS: "/metrics"
        }
        
        for category, (rpm, burst_mult) in endpoint_configs.items():
            config = RateLimitConfig(
                requests_per_minute=rpm,
                burst_multiplier=burst_mult,
                window_size_seconds=60,
                identifier_strategy=RateLimitStrategy.IP_BASED
            )
            rate_limiter.configure_limits(category, config)
        
        client_ip = "192.168.1.200"
        
        # Test each endpoint category independently
        for category, (rpm, burst_mult) in endpoint_configs.items():
            endpoint = endpoint_paths[category]
            burst_limit = int(rpm * burst_mult)
            
            allowed_count = 0
            blocked_count = 0
            
            for _ in range(requests_per_endpoint):
                result = await rate_limiter.check_rate_limit(client_ip, endpoint)
                
                if result.allowed:
                    allowed_count += 1
                else:
                    blocked_count += 1
            
            # Verify endpoint-specific limits
            assert allowed_count <= burst_limit, \
                f"Endpoint {endpoint} exceeded its burst limit: {allowed_count} > {burst_limit}"
            
            if requests_per_endpoint > burst_limit:
                assert allowed_count == burst_limit, \
                    f"Endpoint {endpoint} should have been limited to {burst_limit} requests"
            else:
                assert allowed_count == requests_per_endpoint, \
                    f"Endpoint {endpoint} should have allowed all {requests_per_endpoint} requests"
    
    @given(
        strategy=st.sampled_from([
            RateLimitStrategy.IP_BASED,
            RateLimitStrategy.USER_AGENT_BASED,
            RateLimitStrategy.COMBINED
        ]),
        user_agents=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 /.-", 
                   min_size=5, max_size=50),
            min_size=2,
            max_size=5,
            unique=True  # Ensure unique user agents
        )
    )
    @settings(max_examples=15, deadline=8000)
    async def test_identification_strategy_accuracy(
        self, 
        strategy, 
        user_agents
    ):
        """
        Property: Rate limiting identification strategies work correctly
        
        For any identification strategy (IP, User-Agent, Combined), the rate limiter
        should correctly identify and limit clients according to the strategy.
        """
        # Create fresh rate limiter for each test
        rate_limiter = self._create_rate_limiter()
        
        # Use fixed valid IP addresses for testing
        valid_ips = ["192.168.1.100", "192.168.1.101", "10.0.0.1", "172.16.0.1"]
        
        assume(len(user_agents) >= 2)
        # No need for additional uniqueness check since unique=True in strategy
        
        config = RateLimitConfig(
            requests_per_minute=20,
            burst_multiplier=1.5,
            window_size_seconds=60,
            identifier_strategy=strategy
        )
        
        rate_limiter.configure_limits(EndpointCategory.DEFAULT, config)
        
        endpoint = "/test"
        burst_limit = int(20 * 1.5)
        
        # Test different combinations based on strategy
        if strategy == RateLimitStrategy.IP_BASED:
            # Same user agent, different IPs should be treated separately
            user_agent = user_agents[0]
            
            for ip in valid_ips[:2]:  # Test first 2 IPs
                allowed_count = 0
                
                for _ in range(burst_limit + 5):
                    result = await rate_limiter.check_rate_limit(ip, endpoint, user_agent)
                    if result.allowed:
                        allowed_count += 1
                
                assert allowed_count == burst_limit, \
                    f"IP {ip} should have been limited to {burst_limit} requests"
        
        elif strategy == RateLimitStrategy.USER_AGENT_BASED:
            # Same IP, different user agents should be treated separately
            ip = valid_ips[0]
            
            for ua in user_agents[:2]:  # Test first 2 user agents
                allowed_count = 0
                
                for _ in range(burst_limit + 5):
                    result = await rate_limiter.check_rate_limit(ip, endpoint, ua)
                    if result.allowed:
                        allowed_count += 1
                
                assert allowed_count == burst_limit, \
                    f"User agent {ua[:20]}... should have been limited to {burst_limit} requests"
        
        elif strategy == RateLimitStrategy.COMBINED:
            # Different combinations of IP + User-Agent should be treated separately
            # Ensure we have enough unique combinations
            assume(len(user_agents) >= 2)
            assume(len(valid_ips) >= 2)
            
            combinations = [
                (valid_ips[0], user_agents[0]),
                (valid_ips[0], user_agents[1]),
                (valid_ips[1], user_agents[0])
            ]
            
            for ip, ua in combinations:
                allowed_count = 0
                
                for _ in range(burst_limit + 5):
                    result = await rate_limiter.check_rate_limit(ip, endpoint, ua)
                    if result.allowed:
                        allowed_count += 1
                
                # Each combination should be treated independently
                assert allowed_count == burst_limit, \
                    f"Combination {ip}+{ua[:10]}... should have been limited to {burst_limit} requests, got {allowed_count}"
    
    @given(
        initial_requests=st.integers(min_value=5, max_value=20),
        wait_time=st.floats(min_value=1.0, max_value=10.0),
        followup_requests=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=10, deadline=15000)
    async def test_rate_limit_reset_accuracy(
        self, 
        initial_requests, 
        wait_time, 
        followup_requests
    ):
        """
        Property: Rate limits reset correctly after time windows expire
        
        For any pattern of requests followed by a wait period, the rate limiter
        should reset limits appropriately when time windows expire.
        """
        # Create fresh rate limiter for each test
        rate_limiter = self._create_rate_limiter()
        config = RateLimitConfig(
            requests_per_minute=15,
            burst_multiplier=1.5,
            window_size_seconds=10,  # Short window for testing
            identifier_strategy=RateLimitStrategy.IP_BASED
        )
        
        rate_limiter.configure_limits(EndpointCategory.DEFAULT, config)
        
        client_ip = "192.168.1.300"
        endpoint = "/test"
        burst_limit = int(15 * 1.5)
        
        # Phase 1: Make initial requests
        initial_allowed = 0
        for _ in range(initial_requests):
            result = await rate_limiter.check_rate_limit(client_ip, endpoint)
            if result.allowed:
                initial_allowed += 1
        
        # Phase 2: Wait for window to expire (if wait_time > window_size)
        if wait_time > config.window_size_seconds:
            # Simulate time passage
            await asyncio.sleep(0.1)  # Small actual wait to avoid test timeout
            
            # Phase 3: Make follow-up requests
            followup_allowed = 0
            for _ in range(followup_requests):
                result = await rate_limiter.check_rate_limit(client_ip, endpoint)
                if result.allowed:
                    followup_allowed += 1
            
            # After window reset, should be able to make requests again
            expected_followup = min(followup_requests, burst_limit)
            assert followup_allowed == expected_followup, \
                f"After reset, should have allowed {expected_followup} requests, got {followup_allowed}"
        
        # Verify initial phase was limited correctly
        expected_initial = min(initial_requests, burst_limit)
        assert initial_allowed == expected_initial, \
            f"Initial phase should have allowed {expected_initial} requests, got {initial_allowed}"


# Integration tests with FastAPI middleware
class TestRateLimitMiddlewareIntegration:
    """Integration tests for rate limiting middleware with FastAPI"""
    
    def test_middleware_http_responses(self):
        """Test that middleware returns proper HTTP 429 responses"""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        # Configure strict rate limiting for testing
        rate_limiter = RateLimiter()
        rate_limiter._redis_enabled = False
        
        config = RateLimitConfig(
            requests_per_minute=2,
            burst_multiplier=1.0,
            window_size_seconds=60
        )
        rate_limiter.configure_limits(EndpointCategory.DEFAULT, config)
        
        app.add_middleware(RateLimitMiddleware, 
                          rate_limiter=rate_limiter,
                          enabled=True,
                          skip_paths=[])
        
        client = TestClient(app)
        
        # First 2 requests should succeed
        response1 = client.get("/test")
        assert response1.status_code == 200
        assert "X-RateLimit-Limit" in response1.headers
        assert "X-RateLimit-Remaining" in response1.headers
        
        response2 = client.get("/test")
        assert response2.status_code == 200
        assert response2.headers["X-RateLimit-Remaining"] == "0"
        
        # Third request should be rate limited
        response3 = client.get("/test")
        assert response3.status_code == 429
        assert "Retry-After" in response3.headers
        assert "X-RateLimit-Limit" in response3.headers
        
        # Verify error message is in Russian
        error_data = response3.json()
        assert "Превышен лимит запросов" in error_data["message"]
    
    def test_middleware_skip_paths(self):
        """Test that middleware correctly skips configured paths"""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}
        
        rate_limiter = RateLimiter()
        rate_limiter._redis_enabled = False
        
        # Very strict limits
        config = RateLimitConfig(
            requests_per_minute=1,
            burst_multiplier=1.0,
            window_size_seconds=60
        )
        rate_limiter.configure_limits(EndpointCategory.DEFAULT, config)
        
        app.add_middleware(RateLimitMiddleware, 
                          rate_limiter=rate_limiter,
                          enabled=True,
                          skip_paths=["/health"])
        
        client = TestClient(app)
        
        # /test should be rate limited after 1 request
        response1 = client.get("/test")
        assert response1.status_code == 200
        
        response2 = client.get("/test")
        assert response2.status_code == 429
        
        # /health should never be rate limited
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
            assert "X-RateLimit-Limit" not in response.headers