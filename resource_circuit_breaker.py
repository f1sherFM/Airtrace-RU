"""
Resource Circuit Breaker and Graceful Degradation for AirTrace RU Backend

Implements circuit breaker patterns for resource exhaustion scenarios,
graceful degradation mechanisms, and configurable resource limits
to maintain system stability under high load conditions.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker is open, requests are blocked
    HALF_OPEN = "half_open"  # Testing if the service has recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds
    
    # Resource-specific thresholds
    memory_threshold_mb: Optional[float] = None
    cpu_threshold_percent: Optional[float] = None
    response_time_threshold_ms: Optional[float] = None


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    state: CircuitBreakerState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    total_requests: int
    total_failures: int
    total_successes: int
    state_changes: int
    average_response_time_ms: float


class ResourceCircuitBreaker:
    """
    Circuit breaker for resource-based failures.
    
    Monitors resource usage and automatically opens when thresholds are exceeded,
    providing graceful degradation and automatic recovery.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        
        # Failure tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes = 0
        
        # Response time tracking
        self.response_times: deque = deque(maxlen=100)
        
        # State change callbacks
        self.state_change_callbacks: List[Callable] = []
        
        logger.info(f"Circuit breaker created: {name}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit breaker is open
            TimeoutError: When request times out
        """
        self.total_requests += 1
        
        # Check if circuit breaker is open
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._set_state(CircuitBreakerState.HALF_OPEN)
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        start_time = time.time()
        
        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(
                self._execute_function(func, *args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            response_time_ms = (time.time() - start_time) * 1000
            self.response_times.append(response_time_ms)
            self._record_success()
            
            return result
            
        except asyncio.TimeoutError:
            self._record_failure("timeout")
            raise TimeoutError(f"Request timed out after {self.config.timeout}s")
        except Exception as e:
            self._record_failure(str(e))
            raise
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function, handling both sync and async functions"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _record_success(self):
        """Record a successful request"""
        self.success_count += 1
        self.total_successes += 1
        self.last_success_time = datetime.now()
        
        # Reset failure count on success
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self._set_state(CircuitBreakerState.CLOSED)
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _record_failure(self, error: str):
        """Record a failed request"""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = datetime.now()
        
        logger.warning(f"Circuit breaker {self.name} recorded failure: {error}")
        
        # Check if we should open the circuit breaker
        if (self.state == CircuitBreakerState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self._set_state(CircuitBreakerState.OPEN)
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self._set_state(CircuitBreakerState.OPEN)
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _set_state(self, new_state: CircuitBreakerState):
        """Set circuit breaker state and notify callbacks"""
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self.state_changes += 1
            
            # Reset counters based on state
            if new_state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
                self.success_count = 0
            elif new_state == CircuitBreakerState.HALF_OPEN:
                self.success_count = 0
            
            logger.info(f"Circuit breaker {self.name} state changed: {old_state.value} -> {new_state.value}")
            
            # Notify callbacks
            for callback in self.state_change_callbacks:
                try:
                    callback(self.name, old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in circuit breaker callback: {e}")
    
    def add_state_change_callback(self, callback: Callable):
        """Add a callback for state changes"""
        self.state_change_callbacks.append(callback)
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics"""
        avg_response_time = 0.0
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
        
        return CircuitBreakerStats(
            state=self.state,
            failure_count=self.failure_count,
            success_count=self.success_count,
            last_failure_time=self.last_failure_time,
            last_success_time=self.last_success_time,
            total_requests=self.total_requests,
            total_failures=self.total_failures,
            total_successes=self.total_successes,
            state_changes=self.state_changes,
            average_response_time_ms=avg_response_time
        )
    
    def reset(self):
        """Manually reset the circuit breaker to closed state"""
        self._set_state(CircuitBreakerState.CLOSED)
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name} manually reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


@dataclass
class GracefulDegradationConfig:
    """Configuration for graceful degradation"""
    # Feature priorities (higher number = higher priority)
    feature_priorities: Dict[str, int] = field(default_factory=lambda: {
        "air_quality_current": 100,
        "air_quality_forecast": 80,
        "health_check": 90,
        "metrics": 50,
        "cache_warming": 30,
        "performance_monitoring": 40,
        "rate_limiting": 70
    })
    
    # Resource thresholds for degradation levels
    memory_warning_percent: float = 80.0
    memory_critical_percent: float = 95.0
    cpu_warning_percent: float = 80.0
    cpu_critical_percent: float = 95.0
    
    # Degradation actions
    disable_non_essential_features: bool = True
    reduce_cache_size: bool = True
    increase_gc_frequency: bool = True
    limit_concurrent_requests: bool = True


class GracefulDegradationManager:
    """
    Manages graceful degradation of system features under resource pressure.
    
    Automatically disables non-essential features and optimizes resource usage
    when system resources are under pressure.
    """
    
    def __init__(self, config: GracefulDegradationConfig):
        self.config = config
        self.degradation_level = 0  # 0=normal, 1=warning, 2=critical
        self.disabled_features: set = set()
        self.degradation_actions: Dict[str, bool] = {}
        
        # Callbacks for degradation events
        self.degradation_callbacks: List[Callable] = []
        
        logger.info("Graceful degradation manager initialized")
    
    def evaluate_degradation(self, memory_percent: float, cpu_percent: float) -> int:
        """
        Evaluate current degradation level based on resource usage.
        
        Args:
            memory_percent: Current memory usage percentage
            cpu_percent: Current CPU usage percentage
            
        Returns:
            int: Degradation level (0=normal, 1=warning, 2=critical)
        """
        new_level = 0
        
        # Check memory thresholds
        if memory_percent >= self.config.memory_critical_percent:
            new_level = max(new_level, 2)
        elif memory_percent >= self.config.memory_warning_percent:
            new_level = max(new_level, 1)
        
        # Check CPU thresholds
        if cpu_percent >= self.config.cpu_critical_percent:
            new_level = max(new_level, 2)
        elif cpu_percent >= self.config.cpu_warning_percent:
            new_level = max(new_level, 1)
        
        # Apply degradation if level changed
        if new_level != self.degradation_level:
            self._apply_degradation(new_level)
        
        return new_level
    
    def _apply_degradation(self, new_level: int):
        """Apply degradation measures for the new level"""
        old_level = self.degradation_level
        self.degradation_level = new_level
        
        logger.info(f"Degradation level changed: {old_level} -> {new_level}")
        
        if new_level > old_level:
            # Increasing degradation
            self._increase_degradation(new_level)
        else:
            # Decreasing degradation
            self._decrease_degradation(new_level)
        
        # Notify callbacks
        for callback in self.degradation_callbacks:
            try:
                callback(old_level, new_level)
            except Exception as e:
                logger.error(f"Error in degradation callback: {e}")
    
    def _increase_degradation(self, level: int):
        """Increase degradation measures"""
        if level >= 1:  # Warning level
            if self.config.disable_non_essential_features:
                self._disable_low_priority_features(threshold=50)
                self.degradation_actions["disabled_low_priority"] = True
            
            if self.config.increase_gc_frequency:
                self.degradation_actions["increased_gc"] = True
        
        if level >= 2:  # Critical level
            if self.config.disable_non_essential_features:
                self._disable_low_priority_features(threshold=80)
                self.degradation_actions["disabled_medium_priority"] = True
            
            if self.config.reduce_cache_size:
                self.degradation_actions["reduced_cache"] = True
            
            if self.config.limit_concurrent_requests:
                self.degradation_actions["limited_requests"] = True
    
    def _decrease_degradation(self, level: int):
        """Decrease degradation measures"""
        if level < 2:  # No longer critical
            if self.degradation_actions.get("limited_requests"):
                self.degradation_actions["limited_requests"] = False
            
            if self.degradation_actions.get("reduced_cache"):
                self.degradation_actions["reduced_cache"] = False
            
            if level < 1:  # Back to normal
                self._enable_all_features()
                self.degradation_actions.clear()
    
    def _disable_low_priority_features(self, threshold: int):
        """Disable features below priority threshold"""
        for feature, priority in self.config.feature_priorities.items():
            if priority < threshold and feature not in self.disabled_features:
                self.disabled_features.add(feature)
                logger.info(f"Disabled feature due to resource pressure: {feature}")
    
    def _enable_all_features(self):
        """Re-enable all disabled features"""
        if self.disabled_features:
            logger.info(f"Re-enabling features: {list(self.disabled_features)}")
            self.disabled_features.clear()
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is currently enabled"""
        return feature not in self.disabled_features
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status"""
        return {
            "level": self.degradation_level,
            "disabled_features": list(self.disabled_features),
            "active_actions": {k: v for k, v in self.degradation_actions.items() if v},
            "feature_count": len(self.config.feature_priorities),
            "disabled_count": len(self.disabled_features)
        }
    
    def add_degradation_callback(self, callback: Callable):
        """Add a callback for degradation level changes"""
        self.degradation_callbacks.append(callback)


class ResourceLimitEnforcer:
    """
    Enforces resource limits and triggers circuit breakers.
    
    Monitors resource usage and automatically triggers circuit breakers
    when limits are exceeded, providing system protection.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, ResourceCircuitBreaker] = {}
        self.degradation_manager: Optional[GracefulDegradationManager] = None
        self.resource_limits: Dict[str, float] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("Resource limit enforcer initialized")
    
    def add_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> ResourceCircuitBreaker:
        """Add a new circuit breaker"""
        breaker = ResourceCircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        
        # Add state change callback
        breaker.add_state_change_callback(self._on_circuit_breaker_state_change)
        
        logger.info(f"Circuit breaker added: {name}")
        return breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[ResourceCircuitBreaker]:
        """Get a circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def set_degradation_manager(self, manager: GracefulDegradationManager):
        """Set the graceful degradation manager"""
        self.degradation_manager = manager
    
    def set_resource_limit(self, resource: str, limit: float):
        """Set a resource limit"""
        self.resource_limits[resource] = limit
        logger.info(f"Resource limit set: {resource} = {limit}")
    
    def check_resource_limits(self, resource_usage: Dict[str, float]) -> List[str]:
        """
        Check resource usage against limits.
        
        Args:
            resource_usage: Current resource usage values
            
        Returns:
            List of violated limits
        """
        violations = []
        
        for resource, usage in resource_usage.items():
            limit = self.resource_limits.get(resource)
            if limit and usage > limit:
                violations.append(f"{resource}: {usage} > {limit}")
                
                # Trigger circuit breaker if exists
                breaker_name = f"{resource}_limit"
                if breaker_name in self.circuit_breakers:
                    breaker = self.circuit_breakers[breaker_name]
                    if breaker.state == CircuitBreakerState.CLOSED:
                        breaker._record_failure(f"Resource limit exceeded: {resource}")
        
        return violations
    
    def _on_circuit_breaker_state_change(self, name: str, old_state: CircuitBreakerState, 
                                       new_state: CircuitBreakerState):
        """Handle circuit breaker state changes"""
        logger.info(f"Circuit breaker {name} changed state: {old_state.value} -> {new_state.value}")
        
        # Trigger degradation if multiple breakers are open
        if new_state == CircuitBreakerState.OPEN:
            open_breakers = sum(1 for cb in self.circuit_breakers.values() 
                              if cb.state == CircuitBreakerState.OPEN)
            
            if open_breakers >= 2 and self.degradation_manager:
                # Force critical degradation when multiple breakers are open
                self.degradation_manager._apply_degradation(2)
    
    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in self.circuit_breakers.items()}
    
    async def cleanup(self):
        """Clean up resources"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Resource limit enforcer cleanup completed")


# Global instances
_resource_limit_enforcer: Optional[ResourceLimitEnforcer] = None
_graceful_degradation_manager: Optional[GracefulDegradationManager] = None


def get_resource_limit_enforcer() -> ResourceLimitEnforcer:
    """Get the global resource limit enforcer"""
    global _resource_limit_enforcer
    if _resource_limit_enforcer is None:
        _resource_limit_enforcer = ResourceLimitEnforcer()
    return _resource_limit_enforcer


def get_graceful_degradation_manager() -> GracefulDegradationManager:
    """Get the global graceful degradation manager"""
    global _graceful_degradation_manager
    if _graceful_degradation_manager is None:
        config = GracefulDegradationConfig()
        _graceful_degradation_manager = GracefulDegradationManager(config)
    return _graceful_degradation_manager


def setup_resource_protection(memory_limit_mb: Optional[float] = None,
                            cpu_limit_percent: Optional[float] = None) -> ResourceLimitEnforcer:
    """
    Set up resource protection with circuit breakers and graceful degradation.
    
    Args:
        memory_limit_mb: Memory limit in MB
        cpu_limit_percent: CPU limit percentage
        
    Returns:
        Configured ResourceLimitEnforcer
    """
    enforcer = get_resource_limit_enforcer()
    degradation_manager = get_graceful_degradation_manager()
    
    enforcer.set_degradation_manager(degradation_manager)
    
    # Set up memory circuit breaker
    if memory_limit_mb:
        memory_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=120.0,
            memory_threshold_mb=memory_limit_mb
        )
        enforcer.add_circuit_breaker("memory_limit", memory_config)
        enforcer.set_resource_limit("memory_mb", memory_limit_mb)
    
    # Set up CPU circuit breaker
    if cpu_limit_percent:
        cpu_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            cpu_threshold_percent=cpu_limit_percent
        )
        enforcer.add_circuit_breaker("cpu_limit", cpu_config)
        enforcer.set_resource_limit("cpu_percent", cpu_limit_percent)
    
    logger.info("Resource protection configured")
    return enforcer