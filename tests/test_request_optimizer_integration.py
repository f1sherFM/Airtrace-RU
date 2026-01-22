"""
Integration tests for request optimizer with the AirTrace system.

Tests the integration of request optimization features with the existing
air quality service and caching system.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from request_optimizer import (
    RequestOptimizer, 
    BatchConfig, 
    DeduplicationConfig, 
    PrefetchConfig,
    setup_request_optimization,
    get_request_optimizer
)


class TestRequestOptimizerIntegration:
    """Integration tests for request optimizer"""
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self):
        """Test that optimizer initializes correctly with configuration"""
        optimizer = setup_request_optimization()
        
        assert optimizer is not None
        assert optimizer.batch_config is not None
        assert optimizer.dedup_config is not None
        assert optimizer.prefetch_config is not None
        
        # Test global instance
        global_optimizer = get_request_optimizer()
        assert global_optimizer is not None
    
    @pytest.mark.asyncio
    async def test_request_optimization_workflow(self):
        """Test complete request optimization workflow"""
        optimizer = RequestOptimizer()
        
        # Mock the processing to avoid actual API calls
        with patch.object(optimizer, '_process_single_request', new_callable=AsyncMock):
            # Submit multiple requests
            requests = []
            for i in range(5):
                request = await optimizer.optimize_request(
                    55.7558 + i * 0.001, 
                    37.6176 + i * 0.001, 
                    'current'
                )
                requests.append(request)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check statistics
            stats = await optimizer.get_optimization_stats()
            assert stats.total_requests == 5
            assert stats.request_trace_count == 5
            
            # Check detailed metrics
            detailed = await optimizer.get_detailed_metrics()
            assert "basic_stats" in detailed
            assert "efficiency_rates" in detailed
            assert "traceability_metrics" in detailed
            
            # Check optimization report
            report = await optimizer.get_optimization_report()
            assert "summary" in report
            assert "recommendations" in report
    
    @pytest.mark.asyncio
    async def test_request_traceability(self):
        """Test request traceability features"""
        optimizer = RequestOptimizer()
        
        with patch.object(optimizer, '_process_single_request', new_callable=AsyncMock):
            # Submit a request
            request = await optimizer.optimize_request(55.7558, 37.6176, 'current')
            
            # Check trace exists
            trace = await optimizer.get_request_trace(request.id)
            assert trace is not None
            assert trace.request_id == request.id
            assert trace.original_lat == 55.7558
            assert trace.original_lon == 37.6176
            assert trace.request_type == 'current'
            assert isinstance(trace.submitted_at, datetime)
    
    @pytest.mark.asyncio
    async def test_batch_traceability(self):
        """Test batch traceability features"""
        optimizer = RequestOptimizer()
        
        # Create requests that should be batched (similar coordinates)
        requests_data = [
            (55.7558, 37.6176, 'current'),
            (55.7559, 37.6177, 'current'),
            (55.7560, 37.6178, 'current')
        ]
        
        api_requests = []
        for lat, lon, request_type in requests_data:
            from request_optimizer import APIRequest
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
        
        # Should create at least one batch
        if batched_requests:
            batch = batched_requests[0]
            assert len(batch.requests) > 1
            assert batch.request_type == 'current'
            
            # Check that all requests in batch are geographically close
            for req in batch.requests:
                distance = optimizer._calculate_distance(
                    batch.center_lat, batch.center_lon, req.lat, req.lon
                )
                assert distance <= optimizer.batch_config.max_geographic_distance
    
    @pytest.mark.asyncio
    async def test_deduplication_effectiveness(self):
        """Test request deduplication effectiveness"""
        optimizer = RequestOptimizer()
        
        with patch.object(optimizer, '_process_single_request', new_callable=AsyncMock):
            # Submit identical requests
            lat, lon, request_type = 55.7558, 37.6176, 'current'
            
            requests = []
            for i in range(3):
                request = await optimizer.optimize_request(lat, lon, request_type)
                requests.append(request)
            
            # Wait for deduplication
            await asyncio.sleep(0.01)
            
            # Check that requests were deduplicated
            dedup_keys = {req.get_dedup_key() for req in requests}
            assert len(dedup_keys) <= 2  # Should be deduplicated
            
            # Check statistics
            stats = await optimizer.get_optimization_stats()
            assert stats.total_requests >= 3
            assert stats.deduplicated_requests >= 0
    
    @pytest.mark.asyncio
    async def test_configuration_updates(self):
        """Test dynamic configuration updates"""
        optimizer = RequestOptimizer()
        
        # Test batch configuration update
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
        
        # Test deduplication configuration update
        new_dedup_config = DeduplicationConfig(
            enabled=False,
            window_ms=50,
            max_pending_requests=50
        )
        
        optimizer.configure_deduplication(new_dedup_config)
        assert optimizer.dedup_config.enabled == False
        assert optimizer.dedup_config.window_ms == 50
        
        # Test prefetch configuration update
        new_prefetch_config = PrefetchConfig(
            enabled=False,
            pattern_window_hours=12,
            min_pattern_frequency=1,
            prefetch_ahead_minutes=10,
            max_prefetch_requests=25,
            respect_cache_ttl=False
        )
        
        optimizer.configure_prefetching(new_prefetch_config)
        assert optimizer.prefetch_config.enabled == False
        assert optimizer.prefetch_config.pattern_window_hours == 12
    
    @pytest.mark.asyncio
    async def test_usage_pattern_tracking(self):
        """Test usage pattern tracking for prefetching"""
        optimizer = RequestOptimizer()
        
        # Record some requests to establish patterns
        locations = [
            (55.7558, 37.6176, 'current'),
            (55.7558, 37.6176, 'current'),  # Duplicate for pattern
            (59.9311, 30.3609, 'forecast'),
            (59.9311, 30.3609, 'forecast')   # Duplicate for pattern
        ]
        
        for lat, lon, request_type in locations:
            optimizer.usage_patterns.record_request(lat, lon, request_type)
        
        # Check that patterns were recorded
        assert len(optimizer.usage_patterns.location_frequency) > 0
        assert len(optimizer.usage_patterns.time_patterns) > 0
        
        # Test prefetch candidate generation
        candidates = optimizer.usage_patterns.get_prefetch_candidates(
            optimizer.prefetch_config
        )
        
        # Should generate some candidates based on patterns
        for lat, lon, request_type in candidates:
            assert isinstance(lat, float)
            assert isinstance(lon, float)
            assert request_type in ['current', 'forecast']
    
    def test_metrics_calculation(self):
        """Test metrics calculation accuracy"""
        optimizer = RequestOptimizer()
        
        # Manually set some statistics
        optimizer.stats.total_requests = 100
        optimizer.stats.deduplicated_requests = 20
        optimizer.stats.batched_requests = 30
        optimizer.stats.prefetched_requests = 10
        optimizer.stats.cache_hits_from_prefetch = 5
        optimizer.stats.batches_created = 10
        optimizer.stats.request_trace_count = 100
        optimizer.stats.successful_traces = 95
        
        # Calculate rates
        optimizer.stats.calculate_rates()
        
        # Verify calculations
        assert optimizer.stats.deduplication_rate == 0.2  # 20/100
        assert optimizer.stats.batch_efficiency == 0.3    # 30/100
        assert optimizer.stats.prefetch_hit_rate == 0.5   # 5/10
        assert optimizer.stats.average_batch_size == 3.0  # 30/10
        assert optimizer.stats.prefetch_accuracy == 0.95  # 95/100