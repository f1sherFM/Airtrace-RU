"""
Property-Based Tests for Resource Management Efficiency

**Property 7: Resource Management Efficiency**
**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8**

Tests that the resource management system efficiently monitors resource usage,
enforces configured limits, implements graceful degradation when approaching limits,
and optimizes resource allocation patterns under various load conditions.
"""

import asyncio
import gc
import pytest
import time
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock

from resource_manager import (
    ResourceManager, ResourceLimits, ResourceUsage, MemoryUsage, CPUUsage,
    MemoryOptimizationResult, ResourcePool, get_resource_manager
)
from resource_circuit_breaker import (
    ResourceCircuitBreaker, CircuitBreakerConfig, CircuitBreakerState,
    GracefulDegradationManager, GracefulDegradationConfig,
    ResourceLimitEnforcer, CircuitBreakerOpenError
)


class TestResourceManagementEfficiency:
    """Property-based tests for resource management efficiency"""
    
    @given(
        memory_limit_mb=st.floats(min_value=100.0, max_value=8192.0),
        cpu_limit_percent=st.floats(min_value=10.0, max_value=100.0),
        monitoring_interval=st.floats(min_value=1.0, max_value=60.0)
    )
    @settings(max_examples=50, deadline=30000)
    async def test_resource_monitoring_accuracy(self, memory_limit_mb: float, 
                                              cpu_limit_percent: float,
                                              monitoring_interval: float):
        """
        **Property 7: Resource Management Efficiency**
        **Validates: Requirements 7.1, 7.2**
        
        For any resource limits and monitoring interval, the resource manager should
        accurately monitor memory usage, track CPU usage patterns, and collect
        resource statistics within reasonable bounds.
        """
        # Create resource manager with limits
        limits = ResourceLimits(
            max_memory_mb=memory_limit_mb,
            max_cpu_percent=cpu_limit_percent
        )
        
        manager = ResourceManager(limits)
        
        try:
            # Get resource usage
            usage = await manager.get_resource_usage()
            
            # Verify resource usage structure and validity
            assert isinstance(usage, ResourceUsage)
            assert isinstance(usage.memory, MemoryUsage)
            assert isinstance(usage.cpu, CPUUsage)
            assert isinstance(usage.timestamp, datetime)
            
            # Memory usage should be non-negative and reasonable
            assert usage.memory.total_mb >= 0
            assert usage.memory.available_mb >= 0
            assert usage.memory.used_mb >= 0
            assert usage.memory.process_rss_mb >= 0
            assert usage.memory.process_vms_mb >= 0
            assert 0 <= usage.memory.used_percent <= 100
            
            # CPU usage should be within valid ranges
            assert 0 <= usage.cpu.system_percent <= 100
            assert usage.cpu.process_percent >= 0  # Can exceed 100% on multi-core
            assert usage.cpu.cpu_count >= 1
            
            # Resource counts should be non-negative
            assert usage.open_files >= 0
            assert usage.threads >= 0
            
            # GC statistics should be valid
            assert isinstance(usage.memory.gc_collections, dict)
            assert isinstance(usage.memory.gc_collected, dict)
            
            # Verify monitoring can be started and stopped
            await manager.start_monitoring(monitoring_interval)
            assert manager._monitoring_active
            
            # Wait briefly to ensure monitoring works
            await asyncio.sleep(0.1)
            
            await manager.stop_monitoring()
            assert not manager._monitoring_active
            
        finally:
            await manager.cleanup()
    
    @given(
        memory_limit_mb=st.floats(min_value=50.0, max_value=1000.0),
        cpu_limit_percent=st.floats(min_value=20.0, max_value=90.0),
        failure_threshold=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30, deadline=20000)
    async def test_resource_limits_enforcement(self, memory_limit_mb: float,
                                             cpu_limit_percent: float,
                                             failure_threshold: int):
        """
        **Property 7: Resource Management Efficiency**
        **Validates: Requirements 7.3, 7.4**
        
        For any resource limits configuration, the system should enforce
        configured limits, implement circuit breakers for resource exhaustion,
        and provide graceful degradation mechanisms.
        """
        # Create resource manager with strict limits
        limits = ResourceLimits(
            max_memory_mb=memory_limit_mb,
            max_cpu_percent=cpu_limit_percent,
            warning_threshold_percent=70.0,
            critical_threshold_percent=90.0
        )
        
        manager = ResourceManager(limits)
        
        try:
            # Test circuit breaker configuration
            breaker_config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=5.0,
                memory_threshold_mb=memory_limit_mb
            )
            
            breaker = ResourceCircuitBreaker("test_memory", breaker_config)
            
            # Verify initial state
            assert breaker.state == CircuitBreakerState.CLOSED
            assert breaker.failure_count == 0
            
            # Simulate failures to trigger circuit breaker
            for i in range(failure_threshold):
                breaker._record_failure(f"simulated_failure_{i}")
            
            # Circuit breaker should be open after threshold failures
            assert breaker.state == CircuitBreakerState.OPEN
            assert breaker.failure_count == failure_threshold
            
            # Test graceful degradation
            degradation_config = GracefulDegradationConfig(
                memory_warning_percent=70.0,
                memory_critical_percent=90.0,
                cpu_warning_percent=70.0,
                cpu_critical_percent=90.0
            )
            
            degradation_manager = GracefulDegradationManager(degradation_config)
            
            # Test degradation levels
            normal_level = degradation_manager.evaluate_degradation(50.0, 50.0)
            assert normal_level == 0
            
            warning_level = degradation_manager.evaluate_degradation(75.0, 75.0)
            assert warning_level == 1
            
            critical_level = degradation_manager.evaluate_degradation(95.0, 95.0)
            assert critical_level == 2
            
            # Verify feature disabling at critical level
            assert len(degradation_manager.disabled_features) > 0
            
        finally:
            await manager.cleanup()
    
    @given(
        initial_memory_mb=st.floats(min_value=100.0, max_value=500.0),
        gc_threshold_mb=st.floats(min_value=50.0, max_value=200.0)
    )
    @settings(max_examples=20, deadline=15000)
    async def test_memory_optimization_effectiveness(self, initial_memory_mb: float,
                                                   gc_threshold_mb: float):
        """
        **Property 7: Resource Management Efficiency**
        **Validates: Requirements 7.5, 7.6**
        
        For any memory usage scenario, the resource manager should optimize
        memory allocation patterns, implement resource pooling for expensive
        object creation, and provide effective garbage collection optimization.
        """
        limits = ResourceLimits(gc_threshold_mb=gc_threshold_mb)
        manager = ResourceManager(limits)
        
        try:
            # Test memory optimization
            optimization_result = await manager.optimize_memory()
            
            # Verify optimization result structure
            assert isinstance(optimization_result, MemoryOptimizationResult)
            assert optimization_result.memory_before_mb >= 0
            assert optimization_result.memory_after_mb >= 0
            assert optimization_result.memory_freed_mb >= 0
            assert optimization_result.duration_ms >= 0
            assert isinstance(optimization_result.optimizations_applied, list)
            
            # Memory freed should be non-negative
            assert optimization_result.memory_freed_mb >= 0
            
            # At least some optimization should be applied
            assert len(optimization_result.optimizations_applied) > 0
            
            # Test resource pool creation and usage
            def create_test_object():
                return {"data": "test", "timestamp": time.time()}
            
            def cleanup_test_object(obj):
                obj.clear()
            
            pool = manager.create_resource_pool(
                "test_pool",
                create_test_object,
                max_size=5,
                cleanup_func=cleanup_test_object
            )
            
            # Test pool operations
            obj1 = pool.acquire()
            assert obj1 is not None
            assert "data" in obj1
            
            obj2 = pool.acquire()
            assert obj2 is not None
            assert obj2 != obj1  # Should be different objects
            
            # Release objects back to pool
            pool.release(obj1)
            pool.release(obj2)
            
            # Verify pool statistics
            stats = pool.get_stats()
            assert stats["created_count"] >= 2
            assert stats["acquired_count"] >= 2
            assert stats["released_count"] >= 2
            assert stats["pool_size"] <= stats["max_size"]
            
        finally:
            await manager.cleanup()
    
    @given(
        load_factor=st.floats(min_value=0.1, max_value=2.0),
        resource_pressure=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=25, deadline=20000)
    async def test_dynamic_scaling_and_profiling(self, load_factor: float,
                                               resource_pressure: float):
        """
        **Property 7: Resource Management Efficiency**
        **Validates: Requirements 7.7, 7.8**
        
        For any load conditions, the resource manager should implement
        dynamic resource scaling based on load, provide detailed profiling
        and performance analysis capabilities.
        """
        # Calculate resource limits based on load factor
        base_memory_mb = 200.0
        base_cpu_percent = 50.0
        
        limits = ResourceLimits(
            max_memory_mb=base_memory_mb * load_factor,
            max_cpu_percent=base_cpu_percent * load_factor,
            gc_threshold_mb=base_memory_mb * 0.8 * load_factor
        )
        
        manager = ResourceManager(limits)
        
        try:
            # Test profiling capabilities
            profiling_duration = max(1, int(5 * resource_pressure))
            session = manager.enable_profiling(profiling_duration)
            
            # Verify profiling session
            assert session.session_id is not None
            assert session.duration_seconds == profiling_duration
            assert session.is_active
            assert session.tracemalloc_enabled
            
            # Simulate some memory activity during profiling
            test_data = []
            for i in range(int(100 * resource_pressure)):
                test_data.append(f"test_data_{i}" * 10)
            
            # Wait briefly for profiling
            await asyncio.sleep(0.1)
            
            # Verify profiling session is tracked
            retrieved_session = manager.get_profiling_session(session.session_id)
            assert retrieved_session == session
            
            # Test resource usage history
            usage_history = manager.get_resource_history(limit=5)
            assert isinstance(usage_history, list)
            
            # Test circuit breaker functionality under load
            circuit_breaker_open = manager.is_circuit_breaker_open("memory_mb")
            assert isinstance(circuit_breaker_open, bool)
            
            # Test degradation status
            degradation_status = manager.get_degradation_status()
            assert isinstance(degradation_status, dict)
            assert "level" in degradation_status
            assert "disabled_features" in degradation_status
            assert isinstance(degradation_status["level"], int)
            assert 0 <= degradation_status["level"] <= 2
            
            # Clean up test data
            del test_data
            gc.collect()
            
        finally:
            await manager.cleanup()
    
    @given(
        memory_usage_percent=st.floats(min_value=10.0, max_value=95.0),
        cpu_usage_percent=st.floats(min_value=5.0, max_value=95.0),
        feature_count=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30, deadline=15000)
    def test_graceful_degradation_consistency(self, memory_usage_percent: float,
                                            cpu_usage_percent: float,
                                            feature_count: int):
        """
        **Property 7: Resource Management Efficiency**
        **Validates: Requirements 7.3, 7.4**
        
        For any resource usage levels, graceful degradation should consistently
        disable features based on priority, maintain system stability,
        and provide predictable degradation behavior.
        """
        # Create feature priorities
        features = {f"feature_{i}": (i + 1) * 10 for i in range(feature_count)}
        
        config = GracefulDegradationConfig(
            feature_priorities=features,
            memory_warning_percent=70.0,
            memory_critical_percent=90.0,
            cpu_warning_percent=70.0,
            cpu_critical_percent=90.0
        )
        
        manager = GracefulDegradationManager(config)
        
        # Test degradation level calculation
        degradation_level = manager.evaluate_degradation(memory_usage_percent, cpu_usage_percent)
        
        # Verify degradation level is consistent with thresholds
        if memory_usage_percent >= 90.0 or cpu_usage_percent >= 90.0:
            assert degradation_level == 2  # Critical
        elif memory_usage_percent >= 70.0 or cpu_usage_percent >= 70.0:
            assert degradation_level >= 1  # Warning or Critical
        else:
            assert degradation_level == 0  # Normal
        
        # Verify feature disabling is consistent with priorities
        disabled_features = manager.disabled_features
        enabled_features = {f for f in features.keys() if manager.is_feature_enabled(f)}
        
        # All features should be either enabled or disabled
        assert len(disabled_features) + len(enabled_features) == feature_count
        
        # Higher priority features should be enabled before lower priority ones
        if degradation_level > 0:
            disabled_priorities = [features[f] for f in disabled_features]
            enabled_priorities = [features[f] for f in enabled_features]
            
            if disabled_priorities and enabled_priorities:
                min_enabled_priority = min(enabled_priorities)
                max_disabled_priority = max(disabled_priorities)
                
                # Higher priority features should remain enabled
                assert min_enabled_priority >= max_disabled_priority
        
        # Test status reporting
        status = manager.get_degradation_status()
        assert status["level"] == degradation_level
        assert len(status["disabled_features"]) == len(disabled_features)
        assert status["feature_count"] == feature_count
        assert status["disabled_count"] == len(disabled_features)
    
    @given(
        failure_threshold=st.integers(min_value=1, max_value=5),
        recovery_timeout=st.floats(min_value=1.0, max_value=10.0),
        success_threshold=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=20, deadline=15000)
    async def test_circuit_breaker_state_transitions(self, failure_threshold: int,
                                                   recovery_timeout: float,
                                                   success_threshold: int):
        """
        **Property 7: Resource Management Efficiency**
        **Validates: Requirements 7.3, 7.4**
        
        For any circuit breaker configuration, state transitions should be
        consistent, recovery should work properly, and statistics should
        accurately reflect the breaker's behavior.
        """
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            timeout=5.0
        )
        
        breaker = ResourceCircuitBreaker("test_breaker", config)
        
        # Initial state should be CLOSED
        assert breaker.state == CircuitBreakerState.CLOSED
        
        # Record failures to trigger opening
        for i in range(failure_threshold):
            breaker._record_failure(f"test_failure_{i}")
        
        # Should be OPEN after threshold failures
        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.failure_count == failure_threshold
        
        # Test that calls are blocked when open
        async def test_function():
            return "success"
        
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(test_function)
        
        # Simulate time passing for recovery
        breaker.last_failure_time = datetime.now() - timedelta(seconds=recovery_timeout + 1)
        
        # Next call should transition to HALF_OPEN, then possibly to CLOSED
        try:
            result = await breaker.call(test_function)
            assert result == "success"
            
            # After a successful call, the breaker should be either HALF_OPEN or CLOSED
            # depending on the success_threshold
            if success_threshold == 1:
                # With success_threshold=1, one success should close the breaker
                assert breaker.state == CircuitBreakerState.CLOSED
            else:
                # With success_threshold>1, should remain HALF_OPEN after first success
                assert breaker.state == CircuitBreakerState.HALF_OPEN
                
                # Record additional successes to close the breaker
                for i in range(success_threshold - 1):  # -1 because we already had one success
                    breaker._record_success()
                
                # Should be CLOSED after enough successes
                assert breaker.state == CircuitBreakerState.CLOSED
                
        except CircuitBreakerOpenError:
            # If still open, verify the recovery logic
            assert not breaker._should_attempt_reset()
        
        # Verify statistics
        stats = breaker.get_stats()
        assert stats.total_requests >= 0
        assert stats.total_failures >= failure_threshold
        assert stats.state_changes >= 1  # At least opened once
    
    @given(
        pool_size=st.integers(min_value=1, max_value=20),
        acquire_count=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=25, deadline=10000)
    def test_resource_pool_efficiency(self, pool_size: int, acquire_count: int):
        """
        **Property 7: Resource Management Efficiency**
        **Validates: Requirements 7.5, 7.6**
        
        For any pool configuration and usage pattern, resource pools should
        efficiently manage object creation and reuse, maintain accurate statistics,
        and properly handle cleanup operations.
        """
        created_objects = []
        cleanup_called = []
        
        def factory():
            obj = {"id": len(created_objects), "data": "test"}
            created_objects.append(obj)
            return obj
        
        def cleanup(obj):
            cleanup_called.append(obj["id"])
        
        pool = ResourcePool(factory, pool_size, cleanup)
        
        # Acquire objects
        acquired_objects = []
        for i in range(acquire_count):
            obj = pool.acquire()
            acquired_objects.append(obj)
            assert obj is not None
            assert "id" in obj
            assert "data" in obj
        
        # Verify object creation efficiency
        stats_after_acquire = pool.get_stats()
        assert stats_after_acquire["acquired_count"] == acquire_count
        assert stats_after_acquire["created_count"] <= acquire_count
        
        # Release objects back to pool
        for obj in acquired_objects:
            pool.release(obj)
        
        # Verify pool statistics
        final_stats = pool.get_stats()
        assert final_stats["released_count"] == acquire_count
        assert final_stats["pool_size"] <= pool_size
        assert final_stats["active_resources"] == 0
        
        # Objects beyond pool size should be cleaned up
        expected_cleanup_count = max(0, acquire_count - pool_size)
        if expected_cleanup_count > 0:
            assert len(cleanup_called) >= expected_cleanup_count
        
        # Clear pool and verify cleanup
        pool.clear()
        final_pool_stats = pool.get_stats()
        assert final_pool_stats["pool_size"] == 0