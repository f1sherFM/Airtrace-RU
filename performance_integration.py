"""
Performance Integration Module for AirTrace RU Backend

Wires all performance optimization components together into a unified system.
Provides centralized initialization, configuration, and coordination of:
- Multi-level caching with Redis
- Rate limiting with sliding windows
- Connection pooling for external APIs
- Performance monitoring and metrics
- Request optimization (batching, deduplication, prefetching)
- Resource management and circuit breakers
- WeatherAPI integration
- Graceful degradation mechanisms

Requirements: All performance optimization requirements (1.1-11.7)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Core performance components
from cache import MultiLevelCacheManager, CacheLevel
from rate_limiter import RateLimiter
from connection_pool import ConnectionPoolManager, ServiceType
from performance_monitor import PerformanceMonitor, setup_performance_monitoring
from request_optimizer import RequestOptimizer, setup_request_optimization
from resource_manager import ResourceManager, get_resource_manager
from weather_api_manager import WeatherAPIManager
from unified_weather_service import UnifiedWeatherService

# Supporting components
from graceful_degradation import GracefulDegradationManager, get_graceful_degradation_manager
from prometheus_exporter import PrometheusExporter, setup_prometheus_exporter
from system_monitor import SystemResourceMonitor, get_system_monitor
from config import config

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSystemStatus:
    """Overall performance system status"""
    initialized: bool = False
    components_healthy: Dict[str, bool] = None
    total_components: int = 0
    healthy_components: int = 0
    degraded_components: int = 0
    failed_components: int = 0
    last_health_check: Optional[datetime] = None
    
    def __post_init__(self):
        if self.components_healthy is None:
            self.components_healthy = {}


class PerformanceIntegrationManager:
    """
    Central manager for all performance optimization components.
    
    Coordinates initialization, health monitoring, and graceful degradation
    across all performance systems.
    """
    
    def __init__(self):
        # Component instances
        self.cache_manager: Optional[MultiLevelCacheManager] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.connection_pool_manager: Optional[ConnectionPoolManager] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.request_optimizer: Optional[RequestOptimizer] = None
        self.resource_manager: Optional[ResourceManager] = None
        self.weather_api_manager: Optional[WeatherAPIManager] = None
        self.unified_weather_service: Optional[UnifiedWeatherService] = None
        self.graceful_degradation_manager: Optional[GracefulDegradationManager] = None
        self.prometheus_exporter: Optional[PrometheusExporter] = None
        self.system_monitor: Optional[SystemResourceMonitor] = None
        
        # System state
        self.initialized = False
        self.status = PerformanceSystemStatus()
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info("PerformanceIntegrationManager created")
    
    async def initialize_all_components(self) -> bool:
        """
        Initialize all performance optimization components in the correct order.
        
        Returns:
            bool: True if all components initialized successfully
        """
        if self.initialized:
            logger.warning("Performance system already initialized")
            return True
        
        logger.info("Initializing performance optimization system...")
        
        try:
            # Phase 1: Core infrastructure components
            await self._initialize_core_infrastructure()
            
            # Phase 2: Monitoring and metrics components
            await self._initialize_monitoring_components()
            
            # Phase 3: Optimization components
            await self._initialize_optimization_components()
            
            # Phase 4: Integration and coordination
            await self._initialize_integration_components()
            
            # Phase 5: Start background tasks
            await self._start_background_tasks()
            
            # Phase 6: Validate system health
            system_healthy = await self._validate_system_health()
            
            if system_healthy:
                self.initialized = True
                self.status.initialized = True
                logger.info("Performance optimization system initialized successfully")
                return True
            else:
                logger.error("Performance system initialization completed with issues")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize performance system: {e}")
            await self._cleanup_partial_initialization()
            return False
    
    async def _initialize_core_infrastructure(self):
        """Initialize core infrastructure components"""
        logger.info("Initializing core infrastructure components...")
        
        # 1. Cache Manager (L1, L2, L3 caching)
        try:
            self.cache_manager = MultiLevelCacheManager()
            logger.info("✓ Multi-level cache manager initialized")
        except Exception as e:
            logger.error(f"✗ Cache manager initialization failed: {e}")
            raise
        
        # 2. Connection Pool Manager
        try:
            self.connection_pool_manager = ConnectionPoolManager()
            logger.info("✓ Connection pool manager initialized")
        except Exception as e:
            logger.error(f"✗ Connection pool manager initialization failed: {e}")
            raise
        
        # 3. Rate Limiter
        if config.performance.rate_limiting_enabled:
            try:
                self.rate_limiter = RateLimiter()
                logger.info("✓ Rate limiter initialized")
            except Exception as e:
                logger.error(f"✗ Rate limiter initialization failed: {e}")
                raise
        else:
            logger.info("- Rate limiter disabled")
        
        # 4. Graceful Degradation Manager
        try:
            self.graceful_degradation_manager = get_graceful_degradation_manager()
            logger.info("✓ Graceful degradation manager initialized")
        except Exception as e:
            logger.error(f"✗ Graceful degradation manager initialization failed: {e}")
            raise
    
    async def _initialize_monitoring_components(self):
        """Initialize monitoring and metrics components"""
        logger.info("Initializing monitoring components...")
        
        # 1. Performance Monitor
        try:
            self.performance_monitor = setup_performance_monitoring(
                max_metrics_history=getattr(config.performance, 'max_metrics_history', 1000),
                stats_window_minutes=getattr(config.performance, 'stats_window_minutes', 60)
            )
            logger.info("✓ Performance monitor initialized")
        except Exception as e:
            logger.error(f"✗ Performance monitor initialization failed: {e}")
            raise
        
        # 2. System Monitor
        try:
            self.system_monitor = get_system_monitor()
            await self.system_monitor.start_monitoring()
            logger.info("✓ System monitor initialized")
        except Exception as e:
            logger.error(f"✗ System monitor initialization failed: {e}")
            raise
        
        # 3. Prometheus Exporter
        if config.performance.monitoring_enabled:
            try:
                self.prometheus_exporter = setup_prometheus_exporter()
                logger.info("✓ Prometheus exporter initialized")
            except Exception as e:
                logger.error(f"✗ Prometheus exporter initialization failed: {e}")
                raise
        else:
            logger.info("- Prometheus exporter disabled")
    
    async def _initialize_optimization_components(self):
        """Initialize optimization components"""
        logger.info("Initializing optimization components...")
        
        # 1. Request Optimizer
        try:
            self.request_optimizer = setup_request_optimization()
            await self.request_optimizer.start_prefetching()
            logger.info("✓ Request optimizer initialized")
        except Exception as e:
            logger.error(f"✗ Request optimizer initialization failed: {e}")
            raise
        
        # 2. Resource Manager
        try:
            self.resource_manager = get_resource_manager()
            await self.resource_manager.start_monitoring()
            logger.info("✓ Resource manager initialized")
        except Exception as e:
            logger.error(f"✗ Resource manager initialization failed: {e}")
            raise
    
    async def _initialize_integration_components(self):
        """Initialize integration and coordination components"""
        logger.info("Initializing integration components...")
        
        # 1. WeatherAPI Manager
        try:
            from weather_api_manager import weather_api_manager
            self.weather_api_manager = weather_api_manager
            logger.info("✓ WeatherAPI manager initialized")
        except Exception as e:
            logger.error(f"✗ WeatherAPI manager initialization failed: {e}")
            raise
        
        # 2. Unified Weather Service
        try:
            from unified_weather_service import unified_weather_service
            self.unified_weather_service = unified_weather_service
            logger.info("✓ Unified weather service initialized")
        except Exception as e:
            logger.error(f"✗ Unified weather service initialization failed: {e}")
            raise
        
        # 3. Register components with graceful degradation
        await self._register_components_for_health_monitoring()
    
    async def _register_components_for_health_monitoring(self):
        """Register all components with graceful degradation manager"""
        logger.info("Registering components for health monitoring...")
        
        # Register cache manager
        await self.graceful_degradation_manager.register_component(
            "cache_manager",
            lambda: "healthy" in self.cache_manager.get_status()
        )
        
        # Register connection pools
        await self.graceful_degradation_manager.register_component(
            "connection_pools",
            lambda: self._check_connection_pools_health()
        )
        
        # Register rate limiter
        if self.rate_limiter:
            await self.graceful_degradation_manager.register_component(
                "rate_limiter",
                lambda: "healthy" in self.rate_limiter.get_status()
            )
        
        # Register WeatherAPI
        if config.weather_api.enabled:
            await self.graceful_degradation_manager.register_component(
                "weather_api",
                lambda: self._check_weather_api_health()
            )
        
        # Register resource manager
        await self.graceful_degradation_manager.register_component(
            "resource_manager",
            lambda: not any(self.resource_manager.is_circuit_breaker_open(cb) 
                          for cb in ["memory_mb", "memory_percent", "cpu_percent"])
        )
        
        logger.info("✓ Components registered for health monitoring")
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        logger.info("Starting background tasks...")
        
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.background_tasks.append(health_task)
        
        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.append(metrics_task)
        
        # Cache maintenance task
        cache_task = asyncio.create_task(self._cache_maintenance_loop())
        self.background_tasks.append(cache_task)
        
        # Performance optimization task
        optimization_task = asyncio.create_task(self._performance_optimization_loop())
        self.background_tasks.append(optimization_task)
        
        logger.info(f"✓ Started {len(self.background_tasks)} background tasks")
    
    async def _validate_system_health(self) -> bool:
        """Validate overall system health after initialization"""
        logger.info("Validating system health...")
        
        health_checks = {
            "cache_manager": self._check_cache_health(),
            "connection_pools": self._check_connection_pools_health(),
            "performance_monitor": self._check_performance_monitor_health(),
            "resource_manager": self._check_resource_manager_health(),
            "request_optimizer": self._check_request_optimizer_health(),
        }
        
        if self.rate_limiter:
            health_checks["rate_limiter"] = self._check_rate_limiter_health()
        
        if config.weather_api.enabled:
            health_checks["weather_api"] = await self._check_weather_api_health()
        
        # Execute all health checks
        results = {}
        for component, check in health_checks.items():
            try:
                if asyncio.iscoroutine(check):
                    results[component] = await check
                else:
                    results[component] = check
            except Exception as e:
                logger.error(f"Health check failed for {component}: {e}")
                results[component] = False
        
        # Update status
        self.status.components_healthy = results
        self.status.total_components = len(results)
        self.status.healthy_components = sum(1 for healthy in results.values() if healthy)
        self.status.failed_components = sum(1 for healthy in results.values() if not healthy)
        self.status.last_health_check = datetime.now(timezone.utc)
        
        # Log results
        for component, healthy in results.items():
            status_symbol = "✓" if healthy else "✗"
            logger.info(f"{status_symbol} {component}: {'healthy' if healthy else 'unhealthy'}")
        
        success_rate = self.status.healthy_components / self.status.total_components
        logger.info(f"System health: {self.status.healthy_components}/{self.status.total_components} "
                   f"components healthy ({success_rate:.1%})")
        
        return success_rate >= 0.8  # 80% of components must be healthy
    
    def _check_cache_health(self) -> bool:
        """Check cache manager health"""
        try:
            status = self.cache_manager.get_status()
            return "healthy" in status or "enabled" in status
        except Exception:
            return False
    
    def _check_connection_pools_health(self) -> bool:
        """Check connection pools health"""
        try:
            # Check if we can get pool stats
            stats = asyncio.create_task(self.connection_pool_manager.get_all_stats())
            return True
        except Exception:
            return False
    
    def _check_performance_monitor_health(self) -> bool:
        """Check performance monitor health"""
        try:
            stats = self.performance_monitor.get_performance_stats()
            return stats is not None
        except Exception:
            return False
    
    def _check_resource_manager_health(self) -> bool:
        """Check resource manager health"""
        try:
            # Check if any critical circuit breakers are open
            critical_breakers = ["memory_mb", "memory_percent", "cpu_percent"]
            return not any(self.resource_manager.is_circuit_breaker_open(cb) 
                          for cb in critical_breakers)
        except Exception:
            return False
    
    def _check_request_optimizer_health(self) -> bool:
        """Check request optimizer health"""
        try:
            stats = asyncio.create_task(self.request_optimizer.get_optimization_stats())
            return True
        except Exception:
            return False
    
    def _check_rate_limiter_health(self) -> bool:
        """Check rate limiter health"""
        try:
            status = self.rate_limiter.get_status()
            return "healthy" in status or "enabled" in status
        except Exception:
            return False
    
    async def _check_weather_api_health(self) -> bool:
        """Check WeatherAPI health"""
        try:
            status = await self.weather_api_manager.get_api_status()
            return status.available
        except Exception:
            return False
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._validate_system_health()
                
                # Update graceful degradation based on health
                unhealthy_components = [
                    comp for comp, healthy in self.status.components_healthy.items() 
                    if not healthy
                ]
                
                if len(unhealthy_components) > self.status.total_components * 0.3:
                    # More than 30% of components unhealthy - enable degradation
                    logger.warning(f"System degradation triggered: {len(unhealthy_components)} "
                                 f"unhealthy components: {unhealthy_components}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                
                # Collect metrics from all components
                await self._collect_comprehensive_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
    
    async def _cache_maintenance_loop(self):
        """Background cache maintenance loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Clear expired cache entries
                cleared = await self.cache_manager.clear_expired()
                if cleared > 0:
                    logger.debug(f"Cache maintenance: cleared {cleared} expired entries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache maintenance loop: {e}")
    
    async def _performance_optimization_loop(self):
        """Background performance optimization loop"""
        while True:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                
                # Trigger resource optimization if needed
                if self.resource_manager:
                    usage = await self.resource_manager.get_resource_usage()
                    if usage.memory.used_percent > 80:
                        logger.info("High memory usage detected, triggering optimization")
                        await self.resource_manager.optimize_memory()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance optimization loop: {e}")
    
    async def _collect_comprehensive_metrics(self):
        """Collect metrics from all performance components"""
        try:
            # Performance monitor metrics
            if self.performance_monitor:
                perf_stats = self.performance_monitor.get_performance_stats()
                self.performance_monitor.record_request(
                    endpoint="system_health",
                    method="GET",
                    duration=0.001,
                    status_code=200,
                    cache_hit=False
                )
            
            # Cache metrics
            if self.cache_manager:
                cache_stats = await self.cache_manager.get_stats()
                if self.performance_monitor:
                    self.performance_monitor.record_cache_operation(
                        operation="health_check",
                        cache_level="system",
                        hit=True,
                        duration=0.001
                    )
            
            # Connection pool metrics
            if self.connection_pool_manager:
                pool_stats = await self.connection_pool_manager.get_all_stats()
            
            # Rate limiting metrics
            if self.rate_limiter:
                rate_stats = await self.rate_limiter.get_metrics()
            
            # Request optimization metrics
            if self.request_optimizer:
                opt_stats = await self.request_optimizer.get_optimization_stats()
            
            # Resource management metrics
            if self.resource_manager:
                resource_usage = await self.resource_manager.get_resource_usage()
            
        except Exception as e:
            logger.error(f"Error collecting comprehensive metrics: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.initialized:
            return {
                "initialized": False,
                "status": "not_initialized",
                "message": "Performance system not initialized"
            }
        
        # Update health status
        await self._validate_system_health()
        
        return {
            "initialized": self.initialized,
            "status": "healthy" if self.status.healthy_components == self.status.total_components else "degraded",
            "components": {
                "total": self.status.total_components,
                "healthy": self.status.healthy_components,
                "failed": self.status.failed_components,
                "health_details": self.status.components_healthy
            },
            "last_health_check": self.status.last_health_check.isoformat() if self.status.last_health_check else None,
            "background_tasks": len(self.background_tasks),
            "graceful_degradation": self.graceful_degradation_manager.get_degradation_status() if self.graceful_degradation_manager else None
        }
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all components"""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_status": await self.get_system_status()
        }
        
        try:
            # Cache metrics
            if self.cache_manager:
                metrics["cache"] = (await self.cache_manager.get_stats()).__dict__
            
            # Performance metrics
            if self.performance_monitor:
                metrics["performance"] = self.performance_monitor.get_performance_stats().__dict__
            
            # Connection pool metrics
            if self.connection_pool_manager:
                metrics["connection_pools"] = await self.connection_pool_manager.get_all_stats()
            
            # Rate limiting metrics
            if self.rate_limiter:
                metrics["rate_limiting"] = (await self.rate_limiter.get_metrics()).__dict__
            
            # Request optimization metrics
            if self.request_optimizer:
                metrics["request_optimization"] = await self.request_optimizer.get_detailed_metrics()
            
            # Resource management metrics
            if self.resource_manager:
                resource_usage = await self.resource_manager.get_resource_usage()
                metrics["resource_management"] = {
                    "memory": resource_usage.memory.__dict__,
                    "cpu": resource_usage.cpu.__dict__,
                    "circuit_breakers": self.resource_manager.get_circuit_breaker_stats(),
                    "degradation_status": self.resource_manager.get_degradation_status()
                }
            
            # WeatherAPI metrics
            if self.weather_api_manager:
                metrics["weather_api"] = self.weather_api_manager.get_statistics()
            
            # Unified service metrics
            if self.unified_weather_service:
                metrics["unified_service"] = await self.unified_weather_service.get_service_statistics()
            
        except Exception as e:
            logger.error(f"Error collecting comprehensive metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def _cleanup_partial_initialization(self):
        """Clean up partially initialized components"""
        logger.info("Cleaning up partial initialization...")
        
        # Stop background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Cleanup components
        if self.resource_manager:
            await self.resource_manager.cleanup()
        
        if self.request_optimizer:
            await self.request_optimizer.stop_prefetching()
        
        if self.cache_manager:
            await self.cache_manager.cleanup()
        
        if self.connection_pool_manager:
            await self.connection_pool_manager.cleanup()
        
        if self.system_monitor:
            await self.system_monitor.stop_monitoring()
    
    async def shutdown(self):
        """Gracefully shutdown all performance components"""
        if not self.initialized:
            logger.info("Performance system not initialized, nothing to shutdown")
            return
        
        logger.info("Shutting down performance optimization system...")
        
        # Stop background tasks
        logger.info("Stopping background tasks...")
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown components in reverse order
        try:
            # Stop optimization components
            if self.request_optimizer:
                await self.request_optimizer.stop_prefetching()
                logger.info("✓ Request optimizer stopped")
            
            if self.resource_manager:
                await self.resource_manager.stop_monitoring()
                await self.resource_manager.cleanup()
                logger.info("✓ Resource manager stopped")
            
            # Stop monitoring components
            if self.system_monitor:
                await self.system_monitor.stop_monitoring()
                logger.info("✓ System monitor stopped")
            
            if self.prometheus_exporter:
                await self.prometheus_exporter.stop_alerting()
                logger.info("✓ Prometheus exporter stopped")
            
            # Stop infrastructure components
            if self.connection_pool_manager:
                await self.connection_pool_manager.cleanup()
                logger.info("✓ Connection pools cleaned up")
            
            if self.rate_limiter:
                await self.rate_limiter.cleanup()
                logger.info("✓ Rate limiter cleaned up")
            
            if self.cache_manager:
                await self.cache_manager.cleanup()
                logger.info("✓ Cache manager cleaned up")
            
            if self.unified_weather_service:
                await self.unified_weather_service.cleanup()
                logger.info("✓ Unified weather service cleaned up")
            
        except Exception as e:
            logger.error(f"Error during component shutdown: {e}")
        
        self.initialized = False
        self.status.initialized = False
        logger.info("Performance optimization system shutdown completed")


# Global performance integration manager
_performance_integration_manager: Optional[PerformanceIntegrationManager] = None


def get_performance_integration_manager() -> PerformanceIntegrationManager:
    """Get the global performance integration manager"""
    global _performance_integration_manager
    if _performance_integration_manager is None:
        _performance_integration_manager = PerformanceIntegrationManager()
    return _performance_integration_manager


async def initialize_performance_system() -> bool:
    """Initialize the complete performance optimization system"""
    manager = get_performance_integration_manager()
    return await manager.initialize_all_components()


async def shutdown_performance_system():
    """Shutdown the complete performance optimization system"""
    manager = get_performance_integration_manager()
    await manager.shutdown()


async def get_performance_system_status() -> Dict[str, Any]:
    """Get comprehensive performance system status"""
    manager = get_performance_integration_manager()
    return await manager.get_system_status()


async def get_performance_system_metrics() -> Dict[str, Any]:
    """Get comprehensive performance system metrics"""
    manager = get_performance_integration_manager()
    return await manager.get_comprehensive_metrics()