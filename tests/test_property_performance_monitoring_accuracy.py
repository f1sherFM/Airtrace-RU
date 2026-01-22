"""
Property-based tests for performance monitoring accuracy.

**Property 4: Performance Monitoring Accuracy**
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8**

Tests that the performance monitoring system accurately collects metrics
for response times, cache operations, external API calls, and system resources,
generating alerts when thresholds are exceeded.
"""

import asyncio
import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import statistics

from performance_monitor import (
    PerformanceMonitor, 
    RequestMetrics, 
    CacheMetrics, 
    ExternalAPIMetrics,
    PerformanceStats
)
from system_monitor import SystemResourceMonitor, SystemMetrics


class TestPerformanceMonitoringAccuracy:
    """Property tests for performance monitoring accuracy"""
    
    @given(
        endpoint=st.text(min_size=1, max_size=50),
        method=st.sampled_from(['GET', 'POST', 'PUT', 'DELETE']),
        duration=st.floats(min_value=0.001, max_value=30.0),
        status_code=st.integers(min_value=200, max_value=599),
        cache_hit=st.booleans(),
        external_api_calls=st.integers(min_value=0, max_value=10)
    )
    def test_request_metrics_recording_accuracy(self, endpoint, method, duration, 
                                              status_code, cache_hit, external_api_calls):
        """
        Property: Request metrics are recorded accurately with all provided data.
        
        For any valid request parameters, the performance monitor should:
        1. Store the metrics with correct values
        2. Update endpoint counters appropriately
        3. Calculate statistics accurately
        """
        monitor = PerformanceMonitor(max_metrics_history=1000)
        
        # Record the request
        monitor.record_request(
            endpoint=endpoint,
            method=method,
            duration=duration,
            status_code=status_code,
            cache_hit=cache_hit,
            external_api_calls=external_api_calls
        )
        
        # Verify metrics were recorded
        assert len(monitor.request_metrics) == 1
        
        recorded_metric = monitor.request_metrics[0]
        assert recorded_metric.endpoint == endpoint
        assert recorded_metric.method == method
        assert recorded_metric.duration == duration
        assert recorded_metric.status_code == status_code
        assert recorded_metric.cache_hit == cache_hit
        assert recorded_metric.external_api_calls == external_api_calls
        
        # Verify endpoint counters
        endpoint_stats = monitor.get_endpoint_stats(endpoint)
        assert endpoint_stats['total'] == 1
        
        if status_code >= 400:
            assert endpoint_stats['errors'] == 1
        else:
            assert endpoint_stats.get('errors', 0) == 0
        
        if cache_hit:
            assert endpoint_stats['cache_hits'] == 1
        else:
            assert endpoint_stats.get('cache_hits', 0) == 0
        
        # Verify performance stats calculation
        stats = monitor.get_performance_stats()
        assert stats.request_count == 1
        assert abs(stats.avg_response_time - duration) < 0.001
        assert abs(stats.p50_response_time - duration) < 0.001
        assert abs(stats.p95_response_time - duration) < 0.001
        assert abs(stats.p99_response_time - duration) < 0.001
        
        if status_code >= 400:
            assert stats.error_rate == 1.0
        else:
            assert stats.error_rate == 0.0
    
    @given(
        operations=st.lists(
            st.tuples(
                st.sampled_from(['get', 'set', 'invalidate']),  # operation
                st.sampled_from(['L1', 'L2', 'L3']),  # cache_level
                st.booleans(),  # hit
                st.floats(min_value=0.0001, max_value=1.0)  # duration
            ),
            min_size=1,
            max_size=100
        )
    )
    def test_cache_metrics_accuracy(self, operations):
        """
        Property: Cache metrics are recorded accurately and statistics are calculated correctly.
        
        For any sequence of cache operations, the monitor should:
        1. Record all operations with correct data
        2. Calculate hit rates accurately
        3. Track operations per cache level
        """
        monitor = PerformanceMonitor(max_metrics_history=1000)
        
        # Record all operations
        for operation, cache_level, hit, duration in operations:
            monitor.record_cache_operation(operation, cache_level, hit, duration)
        
        # Verify all operations were recorded
        assert len(monitor.cache_metrics) == len(operations)
        
        # Verify each recorded metric
        for i, (operation, cache_level, hit, duration) in enumerate(operations):
            recorded = monitor.cache_metrics[i]
            assert recorded.operation == operation
            assert recorded.cache_level == cache_level
            assert recorded.hit == hit
            assert abs(recorded.duration - duration) < 0.0001
        
        # Calculate expected hit rate
        total_ops = len(operations)
        hits = sum(1 for _, _, hit, _ in operations if hit)
        expected_hit_rate = hits / total_ops if total_ops > 0 else 0.0
        expected_miss_rate = 1.0 - expected_hit_rate
        
        # Verify performance stats
        stats = monitor.get_performance_stats()
        assert abs(stats.cache_hit_rate - expected_hit_rate) < 0.001
        assert abs(stats.cache_miss_rate - expected_miss_rate) < 0.001
    
    @given(
        api_calls=st.lists(
            st.tuples(
                st.sampled_from(['open_meteo', 'weatherapi']),  # service
                st.text(min_size=1, max_size=50),  # endpoint
                st.floats(min_value=0.1, max_value=60.0),  # duration
                st.booleans(),  # success
                st.integers(min_value=200, max_value=599)  # status_code
            ),
            min_size=1,
            max_size=50
        )
    )
    def test_external_api_metrics_accuracy(self, api_calls):
        """
        Property: External API metrics are recorded accurately with correct statistics.
        
        For any sequence of external API calls, the monitor should:
        1. Record all calls with correct data
        2. Calculate success rates accurately
        3. Calculate average latency correctly
        """
        monitor = PerformanceMonitor(max_metrics_history=1000)
        
        # Record all API calls
        for service, endpoint, duration, success, status_code in api_calls:
            monitor.record_external_api_call(
                service=service,
                endpoint=endpoint,
                duration=duration,
                success=success,
                status_code=status_code
            )
        
        # Verify all calls were recorded
        assert len(monitor.external_api_metrics) == len(api_calls)
        
        # Verify each recorded metric
        for i, (service, endpoint, duration, success, status_code) in enumerate(api_calls):
            recorded = monitor.external_api_metrics[i]
            assert recorded.service == service
            assert recorded.endpoint == endpoint
            assert abs(recorded.duration - duration) < 0.001
            assert recorded.success == success
            assert recorded.status_code == status_code
        
        # Calculate expected statistics
        total_calls = len(api_calls)
        successes = sum(1 for _, _, _, success, _ in api_calls if success)
        expected_success_rate = successes / total_calls if total_calls > 0 else 0.0
        
        durations = [duration for _, _, duration, _, _ in api_calls]
        expected_avg_latency = statistics.mean(durations) if durations else 0.0
        
        # Verify performance stats
        stats = monitor.get_performance_stats()
        assert abs(stats.external_api_success_rate - expected_success_rate) < 0.001
        assert abs(stats.external_api_avg_latency - expected_avg_latency) < 0.001
    
    @given(
        requests=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=20),  # endpoint
                st.floats(min_value=0.001, max_value=10.0),  # duration
                st.integers(min_value=200, max_value=599)  # status_code
            ),
            min_size=10,
            max_size=1000
        )
    )
    def test_percentile_calculations_accuracy(self, requests):
        """
        Property: Response time percentiles are calculated accurately.
        
        For any list of request durations, the calculated percentiles should
        match manual calculations within acceptable tolerance.
        """
        assume(len(requests) >= 10)  # Need sufficient data for percentiles
        
        monitor = PerformanceMonitor(max_metrics_history=2000)
        
        # Record all requests
        for i, (endpoint, duration, status_code) in enumerate(requests):
            monitor.record_request(
                endpoint=f"{endpoint}_{i % 5}",  # Vary endpoints
                method="GET",
                duration=duration,
                status_code=status_code
            )
        
        # Get performance stats
        stats = monitor.get_performance_stats()
        
        # Calculate expected percentiles manually using proper percentile calculation
        durations = [duration for _, duration, _ in requests]
        durations.sort()
        
        # Use the same method as statistics.quantiles for consistency
        expected_p50 = statistics.median(durations)
        expected_p95 = statistics.quantiles(durations, n=20)[18]  # 95th percentile (19/20)
        expected_p99 = statistics.quantiles(durations, n=100)[98]  # 99th percentile (99/100)
        expected_avg = statistics.mean(durations)
        
        # Verify percentiles are within reasonable tolerance
        # Use relative tolerance for better handling of different value ranges
        tolerance = max(0.1, expected_avg * 0.1)  # 10% or 0.1, whichever is larger
        assert abs(stats.avg_response_time - expected_avg) < tolerance
        assert abs(stats.p50_response_time - expected_p50) < tolerance
        assert abs(stats.p95_response_time - expected_p95) < tolerance
        assert abs(stats.p99_response_time - expected_p99) < tolerance
    
    @given(
        threshold=st.floats(min_value=0.1, max_value=5.0),
        duration=st.floats(min_value=0.001, max_value=10.0)
    )
    def test_alert_threshold_accuracy(self, threshold, duration):
        """
        Property: Alert thresholds are enforced accurately.
        
        When a metric exceeds its threshold, an alert should be triggered.
        When it doesn't exceed the threshold, no alert should be triggered.
        """
        monitor = PerformanceMonitor()
        
        # Set alert threshold
        monitor.set_alert_threshold('max_response_time', threshold)
        
        # Track alerts
        alerts_triggered = []
        def alert_callback(alert_data):
            alerts_triggered.append(alert_data)
        
        monitor.add_alert_callback(alert_callback)
        
        # Record request
        monitor.record_request(
            endpoint="/test",
            method="GET",
            duration=duration,
            status_code=200
        )
        
        # Check if alert was triggered correctly
        if duration > threshold:
            assert len(alerts_triggered) == 1
            assert alerts_triggered[0]['type'] == 'high_response_time'
            assert alerts_triggered[0]['data']['duration'] == duration
            assert alerts_triggered[0]['data']['threshold'] == threshold
        else:
            assert len(alerts_triggered) == 0
    
    @given(
        cpu_usage=st.floats(min_value=0.0, max_value=100.0),
        memory_percent=st.floats(min_value=0.0, max_value=100.0),
        disk_percent=st.floats(min_value=0.0, max_value=100.0)
    )
    def test_system_metrics_accuracy(self, cpu_usage, memory_percent, disk_percent):
        """
        Property: System metrics are recorded and calculated accurately.
        
        For any valid system resource values, the monitor should:
        1. Store metrics with correct values
        2. Calculate statistics accurately
        3. Maintain history within limits
        """
        monitor = SystemResourceMonitor(max_history=100)
        
        # Create test metrics
        test_metrics = SystemMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_usage=cpu_usage,
            memory_usage=int(memory_percent * 1024 * 1024 * 1024 / 100),  # Convert to bytes
            memory_percent=memory_percent,
            disk_usage=int(disk_percent * 1024 * 1024 * 1024 / 100),  # Convert to bytes
            disk_percent=disk_percent,
            network_io_sent=1000,
            network_io_recv=2000,
            active_connections=10,
            process_count=50
        )
        
        # Add metrics to history
        monitor.metrics_history.append(test_metrics)
        
        # Verify metrics were stored correctly
        assert len(monitor.metrics_history) == 1
        stored_metrics = monitor.metrics_history[0]
        
        assert abs(stored_metrics.cpu_usage - cpu_usage) < 0.001
        assert abs(stored_metrics.memory_percent - memory_percent) < 0.001
        assert abs(stored_metrics.disk_percent - disk_percent) < 0.001
        
        # Test statistics calculation
        stats = monitor.get_resource_usage_stats(timedelta(hours=1))
        
        if stats:  # Only check if we have stats
            assert abs(stats['cpu_stats']['avg'] - cpu_usage) < 0.001
            assert abs(stats['memory_stats']['avg'] - memory_percent) < 0.001
            assert abs(stats['disk_stats']['avg'] - disk_percent) < 0.001
    
    @given(
        metrics_count=st.integers(min_value=1, max_value=100),
        max_history=st.integers(min_value=10, max_value=50)
    )
    def test_metrics_history_management(self, metrics_count, max_history):
        """
        Property: Metrics history is managed correctly within specified limits.
        
        The monitor should maintain at most max_history metrics and
        automatically remove old metrics when the limit is exceeded.
        """
        assume(metrics_count > 0 and max_history > 0)
        
        monitor = PerformanceMonitor(max_metrics_history=max_history)
        
        # Add more metrics than the history limit
        for i in range(metrics_count):
            monitor.record_request(
                endpoint=f"/test_{i}",
                method="GET",
                duration=0.1,
                status_code=200
            )
        
        # Verify history limit is respected
        expected_count = min(metrics_count, max_history)
        assert len(monitor.request_metrics) == expected_count
        
        # If we exceeded the limit, verify the oldest metrics were removed
        if metrics_count > max_history:
            # The remaining metrics should be the most recent ones
            for i, metric in enumerate(monitor.request_metrics):
                expected_endpoint = f"/test_{metrics_count - max_history + i}"
                assert metric.endpoint == expected_endpoint
    
    @given(
        time_window_minutes=st.integers(min_value=1, max_value=120)
    )
    def test_time_window_filtering_accuracy(self, time_window_minutes):
        """
        Property: Time window filtering for statistics is accurate.
        
        Only metrics within the specified time window should be included
        in statistics calculations.
        """
        monitor = PerformanceMonitor(max_metrics_history=1000)
        
        now = datetime.now(timezone.utc)
        time_window = timedelta(minutes=time_window_minutes)
        
        # Add metrics both inside and outside the time window
        old_time = now - time_window - timedelta(minutes=10)
        recent_time = now - timedelta(minutes=time_window_minutes // 2)
        
        # Add old metric (should be excluded)
        old_metric = RequestMetrics(
            endpoint="/old",
            method="GET",
            duration=1.0,
            status_code=200,
            timestamp=old_time
        )
        monitor.request_metrics.append(old_metric)
        
        # Add recent metric (should be included)
        recent_metric = RequestMetrics(
            endpoint="/recent",
            method="GET",
            duration=2.0,
            status_code=200,
            timestamp=recent_time
        )
        monitor.request_metrics.append(recent_metric)
        
        # Get stats with the time window
        stats = monitor.get_performance_stats(time_window)
        
        # Only the recent metric should be included
        assert stats.request_count == 1
        assert abs(stats.avg_response_time - 2.0) < 0.001


if __name__ == "__main__":
    pytest.main([__file__])