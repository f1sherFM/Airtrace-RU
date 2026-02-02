"""
Graceful Degradation Manager for AirTrace RU Backend

Implements comprehensive fallback mechanisms for system resilience including:
- Stale data serving during API slowness
- Cached response serving during rate limiting
- Health check endpoints with component status
- Core functionality prioritization during resource constraints
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable
import json

from schemas import AirQualityData, HealthCheckResponse
from config import config

from collections import OrderedDict

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class FallbackStrategy(Enum):
    """Fallback strategy enumeration"""
    SERVE_STALE = "serve_stale"
    USE_CACHE = "use_cache"
    MINIMAL_RESPONSE = "minimal_response"
    FAIL_FAST = "fail_fast"
    RETRY_WITH_BACKOFF = "retry_with_backoff"


@dataclass
class ComponentHealth:
    """Component health information"""
    name: str
    status: ComponentStatus
    last_check: datetime
    error_count: int = 0
    success_count: int = 0
    response_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.error_count
        if total == 0:
            return 0.0
        return self.success_count / total


@dataclass
class FallbackConfig:
    """Fallback configuration for a component"""
    component_name: str
    max_stale_age: int = 300  # 5 minutes
    max_error_count: int = 5
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    max_retry_delay: float = 60.0
    health_check_interval: int = 30
    fallback_strategy: FallbackStrategy = FallbackStrategy.SERVE_STALE
    priority_level: int = 1  # 1=critical, 2=important, 3=optional


class GracefulDegradationManager:
    """
    Manages graceful degradation across all system components.
    
    Provides comprehensive fallback mechanisms including stale data serving,
    cached responses during rate limiting, and component health monitoring.
    """
    
    def __init__(self):
        self.component_health: Dict[str, ComponentHealth] = {}
        self.fallback_configs: Dict[str, FallbackConfig] = {}
        
        # ✅ FIX #4: Use OrderedDict with size limit for stale data cache
        self.stale_data_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_stale_entries = 1000  # Maximum number of stale data entries
        
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        
        # System state
        self.system_under_stress = False
        self.resource_constraints = False
        self.last_system_check = 0
        
        # Statistics
        self.fallback_usage_stats = {
            "stale_data_served": 0,
            "cached_responses_served": 0,
            "minimal_responses_served": 0,
            "fast_failures": 0,
            "successful_retries": 0
        }
        
        # Initialize default configurations
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default fallback configurations for system components"""
        
        # External API fallback config
        self.fallback_configs["external_api"] = FallbackConfig(
            component_name="external_api",
            max_stale_age=600,  # 10 minutes for external API data
            max_error_count=3,
            retry_delay=2.0,
            backoff_factor=2.0,
            health_check_interval=60,
            fallback_strategy=FallbackStrategy.SERVE_STALE,
            priority_level=1  # Critical
        )
        
        # Cache system fallback config
        self.fallback_configs["cache"] = FallbackConfig(
            component_name="cache",
            max_stale_age=1800,  # 30 minutes for cache data
            max_error_count=5,
            retry_delay=1.0,
            health_check_interval=30,
            fallback_strategy=FallbackStrategy.USE_CACHE,
            priority_level=1  # Critical
        )
        
        # Rate limiting fallback config
        self.fallback_configs["rate_limiting"] = FallbackConfig(
            component_name="rate_limiting",
            max_stale_age=300,  # 5 minutes
            max_error_count=10,
            retry_delay=0.5,
            health_check_interval=15,
            fallback_strategy=FallbackStrategy.USE_CACHE,
            priority_level=2  # Important
        )
        
        # WeatherAPI fallback config
        self.fallback_configs["weather_api"] = FallbackConfig(
            component_name="weather_api",
            max_stale_age=3600,  # 1 hour for weather data
            max_error_count=3,
            retry_delay=5.0,
            health_check_interval=120,
            fallback_strategy=FallbackStrategy.SERVE_STALE,
            priority_level=3  # Optional
        )
        
        # Performance monitoring fallback config
        self.fallback_configs["performance_monitoring"] = FallbackConfig(
            component_name="performance_monitoring",
            max_stale_age=0,  # No stale data for monitoring
            max_error_count=20,
            retry_delay=0.1,
            health_check_interval=10,
            fallback_strategy=FallbackStrategy.FAIL_FAST,
            priority_level=3  # Optional
        )
    
    async def register_component(self, component_name: str, 
                                health_check_func: Callable[[], bool],
                                config: Optional[FallbackConfig] = None):
        """Register a component for health monitoring"""
        
        if config:
            self.fallback_configs[component_name] = config
        elif component_name not in self.fallback_configs:
            # Use default config
            self.fallback_configs[component_name] = FallbackConfig(
                component_name=component_name
            )
        
        # Initialize health status
        self.component_health[component_name] = ComponentHealth(
            name=component_name,
            status=ComponentStatus.UNKNOWN,
            last_check=datetime.now(timezone.utc)
        )
        
        # Start health check task
        self.health_check_tasks[component_name] = asyncio.create_task(
            self._health_check_loop(component_name, health_check_func)
        )
        
        logger.info(f"Registered component for graceful degradation: {component_name}")
    
    async def _health_check_loop(self, component_name: str, health_check_func: Callable):
        """Background health check loop for a component"""
        config = self.fallback_configs[component_name]
        
        while True:
            try:
                await asyncio.sleep(config.health_check_interval)
                
                start_time = time.time()
                try:
                    is_healthy = await health_check_func() if asyncio.iscoroutinefunction(health_check_func) else health_check_func()
                    response_time = time.time() - start_time
                    
                    health = self.component_health[component_name]
                    health.last_check = datetime.now(timezone.utc)
                    health.response_time = response_time
                    
                    if is_healthy:
                        health.success_count += 1
                        health.status = ComponentStatus.HEALTHY
                        logger.debug(f"Health check passed for {component_name} ({response_time:.3f}s)")
                    else:
                        health.error_count += 1
                        health.status = ComponentStatus.UNHEALTHY
                        logger.warning(f"Health check failed for {component_name}")
                        
                except Exception as e:
                    health = self.component_health[component_name]
                    health.error_count += 1
                    health.last_check = datetime.now(timezone.utc)
                    health.status = ComponentStatus.UNHEALTHY
                    health.details["last_error"] = str(e)
                    logger.error(f"Health check error for {component_name}: {e}")
                
            except asyncio.CancelledError:
                logger.debug(f"Health check loop cancelled for {component_name}")
                break
            except Exception as e:
                logger.error(f"Health check loop error for {component_name}: {e}")
                await asyncio.sleep(5)  # Brief pause before continuing
    
    async def should_serve_stale_data(self, component_name: str, data_age: int) -> bool:
        """
        Determine if stale data should be served based on component health and configuration.
        
        Args:
            component_name: Name of the component
            data_age: Age of the data in seconds
            
        Returns:
            True if stale data should be served
        """
        if component_name not in self.fallback_configs:
            return False
        
        config = self.fallback_configs[component_name]
        health = self.component_health.get(component_name)
        
        # Don't serve stale data if it's too old
        if data_age > config.max_stale_age:
            return False
        
        # Serve stale data if component is unhealthy
        if health and health.status in [ComponentStatus.UNHEALTHY, ComponentStatus.DEGRADED]:
            return True
        
        # Serve stale data if system is under stress and data is relatively fresh
        if self.system_under_stress and data_age < config.max_stale_age // 2:
            return True
        
        return False
    
    async def get_stale_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get stale data from the fallback cache"""
        if cache_key in self.stale_data_cache:
            data_entry = self.stale_data_cache[cache_key]
            
            # Check if data is within acceptable staleness limits
            data_age = time.time() - data_entry.get("timestamp", 0)
            max_stale_age = max(config.max_stale_age for config in self.fallback_configs.values())
            
            if data_age <= max_stale_age:
                self.fallback_usage_stats["stale_data_served"] += 1
                logger.info(f"Serving stale data for key {cache_key} (age: {data_age:.0f}s)")
                return data_entry.get("data")
        
        return None
    
    async def store_stale_data(self, cache_key: str, data: Any):
        """
        Store data for potential stale serving with automatic size management.
        
        ✅ FIX #4: Implements LRU eviction to prevent unbounded memory growth
        """
        # Check if we need to evict oldest entry
        if len(self.stale_data_cache) >= self._max_stale_entries:
            # Remove oldest entry (FIFO from OrderedDict)
            self.stale_data_cache.popitem(last=False)
            logger.debug(f"Evicted oldest stale data entry (limit: {self._max_stale_entries})")
        
        # Store new data (will be added at the end)
        self.stale_data_cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
        
        # Move to end if key already exists (LRU behavior)
        if cache_key in self.stale_data_cache:
            self.stale_data_cache.move_to_end(cache_key)
    
    async def should_use_cached_response(self, component_name: str) -> bool:
        """
        Determine if cached response should be used during rate limiting or failures.
        
        Args:
            component_name: Name of the component experiencing issues
            
        Returns:
            True if cached response should be used
        """
        if component_name not in self.fallback_configs:
            return False
        
        config = self.fallback_configs[component_name]
        health = self.component_health.get(component_name)
        
        # Use cached response if component is unhealthy
        if health and health.status == ComponentStatus.UNHEALTHY:
            return True
        
        # Use cached response if error count is high
        if health and health.error_count >= config.max_error_count:
            return True
        
        # Use cached response during system stress
        if self.system_under_stress:
            return True
        
        return False
    
    async def get_cached_response_for_rate_limiting(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response to serve during rate limiting"""
        # Try to get from stale data cache first
        stale_data = await self.get_stale_data(cache_key)
        if stale_data:
            self.fallback_usage_stats["cached_responses_served"] += 1
            return stale_data
        
        return None
    
    async def should_prioritize_core_functionality(self) -> bool:
        """
        Determine if system should prioritize core functionality over optional features.
        
        Returns:
            True if core functionality should be prioritized
        """
        # Check system resource constraints
        if self.resource_constraints:
            return True
        
        # Check if critical components are unhealthy
        critical_components_unhealthy = 0
        total_critical_components = 0
        
        for name, config in self.fallback_configs.items():
            if config.priority_level == 1:  # Critical components
                total_critical_components += 1
                health = self.component_health.get(name)
                if health and health.status == ComponentStatus.UNHEALTHY:
                    critical_components_unhealthy += 1
        
        # Prioritize core functionality if >50% of critical components are unhealthy
        if total_critical_components > 0:
            unhealthy_ratio = critical_components_unhealthy / total_critical_components
            if unhealthy_ratio > 0.5:
                return True
        
        return False
    
    async def get_minimal_response(self, request_type: str) -> Dict[str, Any]:
        """
        Generate minimal response during system degradation.
        
        Args:
            request_type: Type of request (e.g., 'current', 'forecast')
            
        Returns:
            Minimal response data
        """
        self.fallback_usage_stats["minimal_responses_served"] += 1
        
        if request_type == "current":
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "location": {"latitude": 0.0, "longitude": 0.0},
                "aqi": {
                    "value": 0,
                    "category": "Данные недоступны",
                    "color": "#CCCCCC",
                    "description": "Система временно работает в ограниченном режиме"
                },
                "pollutants": {},
                "weather": None,
                "recommendations": "Данные о качестве воздуха временно недоступны. Система восстанавливается.",
                "nmu_risk": "unknown",
                "health_warnings": ["Система работает в ограниченном режиме"]
            }
        elif request_type == "forecast":
            return []
        else:
            return {
                "status": "degraded",
                "message": "Система работает в ограниченном режиме",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def update_system_stress_status(self):
        """Update system stress status based on component health"""
        current_time = time.time()
        
        # Check system status every 30 seconds
        if current_time - self.last_system_check < 30:
            return
        
        self.last_system_check = current_time
        
        # Calculate overall system health
        unhealthy_components = 0
        total_components = len(self.component_health)
        
        for health in self.component_health.values():
            if health.status in [ComponentStatus.UNHEALTHY, ComponentStatus.DEGRADED]:
                unhealthy_components += 1
        
        # System is under stress if >30% of components are unhealthy
        if total_components > 0:
            unhealthy_ratio = unhealthy_components / total_components
            self.system_under_stress = unhealthy_ratio > 0.3
            
            if self.system_under_stress:
                logger.warning(f"System under stress: {unhealthy_ratio:.1%} of components unhealthy")
            else:
                logger.debug(f"System healthy: {unhealthy_ratio:.1%} of components unhealthy")
    
    async def get_comprehensive_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status for all components.
        
        Returns detailed health information including component status,
        fallback usage statistics, and system recommendations.
        """
        await self.update_system_stress_status()
        
        component_statuses = {}
        for name, health in self.component_health.items():
            component_statuses[name] = {
                "status": health.status.value,
                "last_check": health.last_check.isoformat(),
                "success_rate": health.get_success_rate(),
                "error_count": health.error_count,
                "success_count": health.success_count,
                "response_time": health.response_time,
                "details": health.details
            }
        
        # Calculate overall system status
        healthy_count = sum(1 for h in self.component_health.values() 
                          if h.status == ComponentStatus.HEALTHY)
        total_count = len(self.component_health)
        
        if total_count == 0:
            overall_status = "unknown"
        elif healthy_count == total_count:
            overall_status = "healthy"
        elif healthy_count >= total_count * 0.7:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "overall_status": overall_status,
            "system_under_stress": self.system_under_stress,
            "resource_constraints": self.resource_constraints,
            "prioritize_core_functionality": await self.should_prioritize_core_functionality(),
            "components": component_statuses,
            "fallback_statistics": self.fallback_usage_stats.copy(),
            "stale_data_entries": len(self.stale_data_cache),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def cleanup(self):
        """Cleanup graceful degradation manager resources"""
        # Cancel all health check tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.health_check_tasks:
            await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)
        
        self.health_check_tasks.clear()
        self.stale_data_cache.clear()
        
        logger.info("Graceful degradation manager cleaned up")


# Global graceful degradation manager instance
graceful_degradation_manager = GracefulDegradationManager()


def get_graceful_degradation_manager() -> GracefulDegradationManager:
    """Get the global graceful degradation manager instance"""
    return graceful_degradation_manager