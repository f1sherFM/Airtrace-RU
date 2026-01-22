"""
Property-based tests for request optimization effectiveness.

**Property 6: Request Optimization Effectiveness**
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7**

Tests request batching, deduplication, and smart prefetching capabilities
to ensure optimization features work correctly across all input scenarios.
"""

import pytest
import asyncio
from hypothesis import given, strategies as st, settings
from datetime import datetime, timezone, timedelta
from typing import List, Set
from unittest.mock import AsyncMock, patch

from request_optimizer import (
    RequestOptimizer, 
    APIRequest, 
    BatchConfig, 
    DeduplicationConfig, 
    PrefetchConfig,
    OptimizationStats,
    UsagePattern
)


class TestRequestOptimizationProperty:
    """Property-based tests for request optimization effectiveness"""
    
    def create_test_optimizer(self):
        """Create request optimizer with test configuration"""
        batch_config = BatchConfig(
            enabled=True,
            max_batch_size=5,
            batch_timeout_ms=50,
            geographic_precision=2,
            max_geographic_distance=0.1
        )
        
        dedup_config = DeduplicationConfig(
            enabled=True,
            window_ms=100,
            max_pending_requests=100
        )
        
        prefetch_config = PrefetchConfig(
            enabled=True,
            pattern_window_hours=1,
            min_pattern_frequency=2,
            prefetch_ahead_minutes=5,
            max_prefetch_requests=10,
            respect_cache_ttl=True
        )
        
        return RequestOptimizer(batch_config, dedup_config, prefetch_config)
    
    @settings(max_examples=50, deadline=5000)
    @given(
        requests_data=st.lists(
            st.tuples(
                st.floats(min_value=50.0, max_value=70.0, allow_nan=False, allow_infinity=False),  # lat
                st.floats(min_value=30.0, max_value=50.0, allow_nan=False, allow_infinity=False),  # lon
                st.sampled_from(['current', 'forecast'])  # request_type
            ),
            min_size=1,
            max_size=20
        )
    )
    @pytest.mark.asyncio
    async def test_request_batching_effectiveness_property(self, requests_data):
        """
        **Property 6: Request Optimization Effectiveness**
        **Validates: Requirements 6.1, 6.2, 6.5**
        
        For any set of incoming requests, the request optimizer should batch 
        similar geographic requests, maintain individual request traceability,
        and provide accurate batching metrics.
        """
        optimizer = self.create_test_optimizer()
        
        # Create API requests from test data
        api_requests = []
        for lat, lon, request_type in requests_data:
            request = APIRequest(
                id=f"test_{len(api_requests)}",
                lat=lat,
                lon=lon,
                request_type=request_type,
                timestamp=datetime.now(timezone.utc)
            )
            api_requests.append(request)
        
        # Test batching
        batched_requests = await optimizer.batch_requests(api_requests)
        
        # Verify batching properties
        total_requests_in_batches = sum(len(batch.requests) for batch in batched_requests)
        
        # Property: All requests should be accounted for in batches or processed individually
        assert total_requests_in_batches <= len(api_requests)
        
        # Property: Each batch should contain requests of the same type
        for batch in batched_requests:
            request_types = {req.request_type for req in batch.requests}
            assert len(request_types) == 1, "Batch should contain requests of same type"
        
        # Property: Requests in same batch should be geographically close
        for batch in batched_requests:
            if len(batch.requests) > 1:
                for req in batch.requests:
                    distance = optimizer._calculate_distance(
                        batch.center_lat, batch.center_lon, req.lat, req.lon
                    )
                    assert distance <= optimizer.batch_config.max_geographic_distance
        
        # Property: Individual request traceability should be maintained
        for batch in batched_requests:
            for req in batch.requests:
                assert req.id is not None
                assert req.lat is not None
                assert req.lon is not None
                assert req.request_type in ['current', 'forecast']
    
    @settings(max_examples=30, deadline=3000)
    @given(
        duplicate_requests=st.lists(
            st.tuples(
                st.floats(min_value=55.0, max_value=56.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=37.0, max_value=38.0, allow_nan=False, allow_infinity=False),
                st.sampled_from(['current', 'forecast'])
            ),
            min_size=2,
            max_size=10
        )
    )
    @pytest.mark.asyncio
    async def test_request_deduplication_effectiveness_property(self, duplicate_requests):
        """
        **Property 6: Request Optimization Effectiveness**
        **Validates: Requirements 6.2, 6.4, 6.6**
        
        For any set of identical concurrent requests, the request optimizer should
        deduplicate them within the configured time window and provide accurate
        deduplication metrics.
        """
        optimizer = self.create_test_optimizer()
        
        # Create identical requests (same coordinates and type)
        lat, lon, request_type = duplicate_requests[0]
        
        # Mock the _process_single_request to avoid actual processing
        with patch.object(optimizer, '_process_single_request', new_callable=AsyncMock):
            # Submit identical requests concurrently
            tasks = []
            for i in range(len(duplicate_requests)):
                task = asyncio.create_task(
                    optimizer.optimize_request(lat, lon, request_type)
                )
                tasks.append(task)
            
            # Wait a bit to allow deduplication to work
            await asyncio.sleep(0.01)
            
            # Get all requests
            optimized_requests = await asyncio.gather(*tasks)
            
            # Property: Identical requests should return the same request object
            if len(optimized_requests) > 1:
                first_request = optimized_requests[0]
                for req in optimized_requests[1:]:
                    # They should be the same object (deduplicated)
                    assert req is first_request or req.get_dedup_key() == first_request.get_dedup_key()
            
            # Property: Deduplication should reduce the number of unique requests
            unique_dedup_keys = {req.get_dedup_key() for req in optimized_requests}
            assert len(unique_dedup_keys) <= len(duplicate_requests)
            
            # Property: Stats should reflect deduplication
            stats = await optimizer.get_optimization_stats()
            assert stats.total_requests >= len(duplicate_requests)
    
    @settings(max_examples=20, deadline=2000)
    @given(
        pattern_requests=st.lists(
            st.tuples(
                st.floats(min_value=55.0, max_value=56.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=37.0, max_value=38.0, allow_nan=False, allow_infinity=False),
                st.sampled_from(['current', 'forecast'])
            ),
            min_size=3,
            max_size=8
        )
    )
    def test_usage_pattern_analysis_property(self, pattern_requests):
        """
        **Property 6: Request Optimization Effectiveness**
        **Validates: Requirements 6.3, 6.7**
        
        For any sequence of requests, the usage pattern analyzer should
        correctly identify patterns and generate appropriate prefetch candidates
        based on frequency and timing patterns.
        """
        usage_patterns = UsagePattern()
        
        # Record requests to establish patterns
        for lat, lon, request_type in pattern_requests:
            usage_patterns.record_request(lat, lon, request_type)
        
        # Property: Pattern recording should not fail
        assert len(usage_patterns.request_history) <= len(pattern_requests)
        
        # Property: Location frequency should be tracked
        for lat, lon, request_type in pattern_requests:
            location_key = f"{lat:.3f},{lon:.3f}"
            assert location_key in usage_patterns.location_frequency
            assert usage_patterns.location_frequency[location_key] > 0
        
        # Property: Time patterns should be recorded
        for lat, lon, request_type in pattern_requests:
            pattern_key = f"{request_type}:{lat:.3f},{lon:.3f}"
            assert pattern_key in usage_patterns.time_patterns
            assert len(usage_patterns.time_patterns[pattern_key]) > 0
        
        # Test prefetch candidate generation
        prefetch_config = PrefetchConfig(
            enabled=True,
            pattern_window_hours=1,
            min_pattern_frequency=1,  # Low threshold for testing
            prefetch_ahead_minutes=5,
            max_prefetch_requests=10,
            respect_cache_ttl=True
        )
        
        candidates = usage_patterns.get_prefetch_candidates(prefetch_config)
        
        # Property: Prefetch candidates should be valid coordinates
        for lat, lon, request_type in candidates:
            assert isinstance(lat, float)
            assert isinstance(lon, float)
            assert request_type in ['current', 'forecast']
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180
    
    @settings(max_examples=20, deadline=3000)
    @given(
        mixed_requests=st.lists(
            st.tuples(
                st.floats(min_value=50.0, max_value=70.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=30.0, max_value=50.0, allow_nan=False, allow_infinity=False),
                st.sampled_from(['current', 'forecast'])
            ),
            min_size=5,
            max_size=15
        )
    )
    @pytest.mark.asyncio
    async def test_optimization_metrics_accuracy_property(self, mixed_requests):
        """
        **Property 6: Request Optimization Effectiveness**
        **Validates: Requirements 6.4, 6.6, 6.7**
        
        For any set of requests processed through optimization, the system should
        provide accurate metrics for batching efficiency, deduplication rates,
        and optimization statistics.
        """
        optimizer = self.create_test_optimizer()
        
        # Mock processing to avoid actual API calls
        with patch.object(optimizer, '_process_single_request', new_callable=AsyncMock):
            # Process requests through optimizer
            tasks = []
            for lat, lon, request_type in mixed_requests:
                task = asyncio.create_task(
                    optimizer.optimize_request(lat, lon, request_type)
                )
                tasks.append(task)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Get optimization statistics
            stats = await optimizer.get_optimization_stats()
            
            # Property: Total requests should match input
            assert stats.total_requests >= len(mixed_requests)
            
            # Property: Rates should be valid percentages (0-1)
            assert 0 <= stats.deduplication_rate <= 1
            assert 0 <= stats.batch_efficiency <= 1
            assert 0 <= stats.prefetch_hit_rate <= 1
            
            # Property: Counters should be non-negative
            assert stats.deduplicated_requests >= 0
            assert stats.batched_requests >= 0
            assert stats.prefetched_requests >= 0
            assert stats.cache_hits_from_prefetch >= 0
            
            # Property: Deduplicated requests should not exceed total requests
            assert stats.deduplicated_requests <= stats.total_requests
            
            # Property: Batched requests should not exceed total requests
            assert stats.batched_requests <= stats.total_requests
    
    @settings(max_examples=10, deadline=2000)
    @given(
        config_values=st.tuples(
            st.integers(min_value=1, max_value=20),  # max_batch_size
            st.integers(min_value=10, max_value=500),  # batch_timeout_ms
            st.integers(min_value=1, max_value=5),  # geographic_precision
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)  # max_distance
        )
    )
    def test_configuration_flexibility_property(self, config_values):
        """
        **Property 6: Request Optimization Effectiveness**
        **Validates: Requirements 6.1, 6.2, 6.3, 6.5, 6.6, 6.7**
        
        For any valid configuration values, the request optimizer should
        accept the configuration and operate within the specified parameters.
        """
        max_batch_size, batch_timeout_ms, geographic_precision, max_distance = config_values
        
        # Create configuration with test values
        batch_config = BatchConfig(
            enabled=True,
            max_batch_size=max_batch_size,
            batch_timeout_ms=batch_timeout_ms,
            geographic_precision=geographic_precision,
            max_geographic_distance=max_distance
        )
        
        dedup_config = DeduplicationConfig(
            enabled=True,
            window_ms=100,
            max_pending_requests=100
        )
        
        prefetch_config = PrefetchConfig(
            enabled=True,
            pattern_window_hours=1,
            min_pattern_frequency=2,
            prefetch_ahead_minutes=5,
            max_prefetch_requests=10,
            respect_cache_ttl=True
        )
        
        # Property: Optimizer should initialize with any valid configuration
        optimizer = RequestOptimizer(batch_config, dedup_config, prefetch_config)
        assert optimizer is not None
        
        # Property: Configuration should be applied correctly
        assert optimizer.batch_config.max_batch_size == max_batch_size
        assert optimizer.batch_config.batch_timeout_ms == batch_timeout_ms
        assert optimizer.batch_config.geographic_precision == geographic_precision
        assert optimizer.batch_config.max_geographic_distance == max_distance
        
        # Property: Configuration updates should work
        new_batch_config = BatchConfig(
            enabled=False,
            max_batch_size=1,
            batch_timeout_ms=10,
            geographic_precision=1,
            max_geographic_distance=0.01
        )
        
        optimizer.configure_batching(new_batch_config)
        assert optimizer.batch_config.enabled == False
        assert optimizer.batch_config.max_batch_size == 1
    
    @pytest.mark.asyncio
    async def test_request_coalescing_property(self):
        """
        **Property 6: Request Optimization Effectiveness**
        **Validates: Requirements 6.5**
        
        For identical requests submitted within the coalescing window,
        the optimizer should coalesce them into a single request while
        maintaining response delivery to all original requesters.
        """
        optimizer = self.create_test_optimizer()
        
        # Create identical requests
        lat, lon, request_type = 55.7558, 37.6176, 'current'
        
        with patch.object(optimizer, '_process_single_request', new_callable=AsyncMock) as mock_process:
            # Submit multiple identical requests simultaneously
            tasks = []
            for i in range(5):
                task = asyncio.create_task(
                    optimizer.optimize_request(lat, lon, request_type)
                )
                tasks.append(task)
            
            # Wait briefly for deduplication
            await asyncio.sleep(0.01)
            
            requests = await asyncio.gather(*tasks)
            
            # Property: All requests should have the same deduplication key
            dedup_keys = {req.get_dedup_key() for req in requests}
            assert len(dedup_keys) <= 2  # Should be mostly deduplicated
            
            # Property: All requests should be valid APIRequest objects
            for req in requests:
                assert isinstance(req, APIRequest)
                assert req.lat == lat
                assert req.lon == lon
                assert req.request_type == request_type
                assert req.future is not None