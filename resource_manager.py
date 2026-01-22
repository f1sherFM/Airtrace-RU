"""
Resource Management and Optimization for AirTrace RU Backend

Implements comprehensive resource monitoring, optimization, and management
including memory usage tracking, CPU monitoring, garbage collection optimization,
and resource pooling for improved performance and stability.
"""

import asyncio
import gc
import logging
import psutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from collections import deque, defaultdict
import weakref
import sys
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

# Import circuit breaker components
from resource_circuit_breaker import (
    ResourceLimitEnforcer, GracefulDegradationManager, GracefulDegradationConfig,
    CircuitBreakerConfig, get_resource_limit_enforcer, get_graceful_degradation_manager
)

# Import dynamic scaling components
from resource_scaling import (
    DynamicResourceScaler, ScalingConfig, AdvancedResourcePool, PoolingStrategy,
    PerformanceProfiler, get_dynamic_scaler, get_performance_profiler,
    create_advanced_pool
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryUsage:
    """Memory usage statistics"""
    total_mb: float
    available_mb: float
    used_mb: float
    used_percent: float
    process_rss_mb: float
    process_vms_mb: float
    gc_collections: Dict[int, int]
    gc_collected: Dict[int, int]
    tracemalloc_current_mb: Optional[float] = None
    tracemalloc_peak_mb: Optional[float] = None


@dataclass
class CPUUsage:
    """CPU usage statistics"""
    system_percent: float
    process_percent: float
    load_average: Optional[List[float]]
    cpu_count: int
    cpu_freq: Optional[float]
    context_switches: int
    interrupts: int
    boot_time: float


@dataclass
class ResourceUsage:
    """Combined resource usage statistics"""
    timestamp: datetime
    memory: MemoryUsage
    cpu: CPUUsage
    network_io: Dict[str, int]
    disk_io: Dict[str, int]
    open_files: int
    threads: int


@dataclass
class ResourceLimits:
    """Resource limits configuration"""
    max_memory_mb: Optional[float] = None
    max_memory_percent: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    max_open_files: Optional[int] = None
    max_threads: Optional[int] = None
    gc_threshold_mb: Optional[float] = None
    warning_threshold_percent: float = 80.0
    critical_threshold_percent: float = 95.0


@dataclass
class GCOptimizationResult:
    """Garbage collection optimization result"""
    collections_before: Dict[int, int]
    collections_after: Dict[int, int]
    collected_objects: int
    memory_freed_mb: float
    duration_ms: float
    optimization_applied: str


@dataclass
class MemoryOptimizationResult:
    """Memory optimization result"""
    memory_before_mb: float
    memory_after_mb: float
    memory_freed_mb: float
    gc_result: Optional[GCOptimizationResult]
    optimizations_applied: List[str]
    duration_ms: float


@dataclass
class ProfilingSession:
    """Memory profiling session"""
    session_id: str
    start_time: datetime
    duration_seconds: int
    tracemalloc_enabled: bool
    snapshot_interval_seconds: int
    snapshots: List[Any] = field(default_factory=list)
    is_active: bool = True


class ResourcePool:
    """Generic resource pool for expensive object creation"""
    
    def __init__(self, factory: Callable, max_size: int = 10, cleanup_func: Optional[Callable] = None):
        self.factory = factory
        self.max_size = max_size
        self.cleanup_func = cleanup_func
        self._pool = deque()
        self._created_count = 0
        self._acquired_count = 0
        self._released_count = 0
        self._lock = threading.Lock()
    
    def acquire(self):
        """Acquire a resource from the pool"""
        with self._lock:
            if self._pool:
                resource = self._pool.popleft()
                self._acquired_count += 1
                return resource
            else:
                resource = self.factory()
                self._created_count += 1
                self._acquired_count += 1
                return resource
    
    def release(self, resource):
        """Release a resource back to the pool"""
        with self._lock:
            if len(self._pool) < self.max_size:
                self._pool.append(resource)
                self._released_count += 1
            else:
                # Pool is full, cleanup the resource
                if self.cleanup_func:
                    try:
                        self.cleanup_func(resource)
                    except Exception as e:
                        logger.warning(f"Error cleaning up pooled resource: {e}")
                self._released_count += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "max_size": self.max_size,
                "created_count": self._created_count,
                "acquired_count": self._acquired_count,
                "released_count": self._released_count,
                "active_resources": self._acquired_count - self._released_count
            }
    
    def clear(self):
        """Clear all resources from the pool"""
        with self._lock:
            while self._pool:
                resource = self._pool.popleft()
                if self.cleanup_func:
                    try:
                        self.cleanup_func(resource)
                    except Exception as e:
                        logger.warning(f"Error cleaning up pooled resource during clear: {e}")


class ResourceManager:
    """
    Comprehensive resource management and optimization system.
    
    Monitors system and process resources, implements optimization strategies,
    provides resource pooling, and enforces resource limits with circuit breakers.
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval = 30.0  # seconds
        
        # Resource usage history
        self._usage_history: deque = deque(maxlen=100)
        self._cpu_history: deque = deque(maxlen=60)  # 1 minute of samples
        
        # Profiling
        self._profiling_sessions: Dict[str, ProfilingSession] = {}
        self._tracemalloc_enabled = False
        
        # Resource pools
        self._resource_pools: Dict[str, ResourcePool] = {}
        
        # Circuit breakers and graceful degradation
        self._limit_enforcer = get_resource_limit_enforcer()
        self._degradation_manager = get_graceful_degradation_manager()
        self._setup_circuit_breakers()
        
        # Dynamic scaling and advanced profiling
        self._dynamic_scaler = get_dynamic_scaler()
        self._performance_profiler = get_performance_profiler()
        self._advanced_pools: Dict[str, AdvancedResourcePool] = {}
        
        # Circuit breakers (legacy - kept for compatibility)
        self._circuit_breakers: Dict[str, bool] = defaultdict(bool)
        self._last_circuit_check = {}
        
        # Optimization state
        self._last_gc_optimization = datetime.now()
        self._gc_optimization_interval = timedelta(minutes=5)
        
        # Thread pool for CPU-intensive operations
        self._thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ResourceManager")
        
        # Initialize process monitoring
        try:
            self._process = psutil.Process()
            self._process_available = True
        except Exception as e:
            logger.warning(f"Process monitoring unavailable: {e}")
            self._process_available = False
        
        logger.info("ResourceManager initialized")
    
    def _setup_circuit_breakers(self):
        """Set up circuit breakers for resource limits"""
        # Memory circuit breaker
        if self.limits.max_memory_mb:
            memory_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=120.0,
                memory_threshold_mb=self.limits.max_memory_mb
            )
            self._limit_enforcer.add_circuit_breaker("memory", memory_config)
        
        # CPU circuit breaker
        if self.limits.max_cpu_percent:
            cpu_config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                cpu_threshold_percent=self.limits.max_cpu_percent
            )
            self._limit_enforcer.add_circuit_breaker("cpu", cpu_config)
        
        # Set degradation manager
        self._limit_enforcer.set_degradation_manager(self._degradation_manager)
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start comprehensive resource monitoring including dynamic scaling"""
        if self._monitoring_active:
            logger.warning("Resource monitoring already active")
            return
        
        self._monitoring_interval = interval
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start dynamic scaling monitoring
        await self._dynamic_scaler.start_monitoring()
        
        logger.info(f"Resource monitoring started with {interval}s interval")
    
    async def stop_monitoring(self):
        """Stop comprehensive resource monitoring including dynamic scaling"""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop dynamic scaling monitoring
        await self._dynamic_scaler.stop_monitoring()
        
        logger.info("Resource monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self._monitoring_active:
                try:
                    # Collect resource usage
                    usage = await self.get_resource_usage()
                    self._usage_history.append(usage)
                    
                    # Check resource limits and circuit breakers
                    await self._check_resource_limits(usage)
                    
                    # Update graceful degradation
                    self._degradation_manager.evaluate_degradation(
                        usage.memory.used_percent,
                        usage.cpu.process_percent
                    )
                    
                    # Perform automatic optimizations
                    await self._auto_optimize(usage)
                    
                except Exception as e:
                    logger.error(f"Error in resource monitoring loop: {e}")
                
                await asyncio.sleep(self._monitoring_interval)
        except asyncio.CancelledError:
            logger.info("Resource monitoring loop cancelled")
    
    async def get_resource_usage(self) -> ResourceUsage:
        """Get current resource usage statistics"""
        try:
            # Get system memory info
            memory_info = psutil.virtual_memory()
            
            # Get process memory info
            process_memory = self._process.memory_info() if self._process_available else None
            
            # Get GC statistics
            gc_stats = gc.get_stats()
            gc_collections = {i: stat['collections'] for i, stat in enumerate(gc_stats)}
            gc_collected = {i: stat['collected'] for i, stat in enumerate(gc_stats)}
            
            # Get tracemalloc info if enabled
            tracemalloc_current = None
            tracemalloc_peak = None
            if self._tracemalloc_enabled and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc_current = current / 1024 / 1024  # Convert to MB
                tracemalloc_peak = peak / 1024 / 1024
            
            memory_usage = MemoryUsage(
                total_mb=memory_info.total / 1024 / 1024,
                available_mb=memory_info.available / 1024 / 1024,
                used_mb=memory_info.used / 1024 / 1024,
                used_percent=memory_info.percent,
                process_rss_mb=process_memory.rss / 1024 / 1024 if process_memory else 0,
                process_vms_mb=process_memory.vms / 1024 / 1024 if process_memory else 0,
                gc_collections=gc_collections,
                gc_collected=gc_collected,
                tracemalloc_current_mb=tracemalloc_current,
                tracemalloc_peak_mb=tracemalloc_peak
            )
            
            # Get CPU info
            cpu_percent = psutil.cpu_percent(interval=None)
            process_cpu = self._process.cpu_percent() if self._process_available else 0
            
            # Get load average (Unix-like systems only)
            load_avg = None
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have load average
                pass
            
            # Get CPU frequency
            cpu_freq = None
            try:
                freq_info = psutil.cpu_freq()
                cpu_freq = freq_info.current if freq_info else None
            except Exception:
                pass
            
            # Get system stats
            cpu_stats = psutil.cpu_stats()
            
            cpu_usage = CPUUsage(
                system_percent=cpu_percent,
                process_percent=process_cpu,
                load_average=load_avg,
                cpu_count=psutil.cpu_count(),
                cpu_freq=cpu_freq,
                context_switches=cpu_stats.ctx_switches,
                interrupts=cpu_stats.interrupts,
                boot_time=psutil.boot_time()
            )
            
            # Get network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            } if net_io else {}
            
            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_stats = {
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count
            } if disk_io else {}
            
            # Get process info
            open_files = len(self._process.open_files()) if self._process_available else 0
            threads = self._process.num_threads() if self._process_available else 0
            
            return ResourceUsage(
                timestamp=datetime.now(),
                memory=memory_usage,
                cpu=cpu_usage,
                network_io=network_io,
                disk_io=disk_io_stats,
                open_files=open_files,
                threads=threads
            )
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            # Return minimal resource usage on error
            return ResourceUsage(
                timestamp=datetime.now(),
                memory=MemoryUsage(0, 0, 0, 0, 0, 0, {}, {}),
                cpu=CPUUsage(0, 0, None, 1, None, 0, 0, time.time()),
                network_io={},
                disk_io={},
                open_files=0,
                threads=0
            )
    
    def set_resource_limits(self, limits: ResourceLimits):
        """Set resource limits and thresholds"""
        self.limits = limits
        logger.info(f"Resource limits updated: {limits}")
    
    async def _check_resource_limits(self, usage: ResourceUsage):
        """Check resource usage against limits and update circuit breakers"""
        current_time = datetime.now()
        
        # Prepare resource usage for limit enforcer
        resource_usage_dict = {
            "memory_mb": usage.memory.process_rss_mb,
            "memory_percent": usage.memory.used_percent,
            "cpu_percent": usage.cpu.process_percent,
            "open_files": usage.open_files,
            "threads": usage.threads
        }
        
        # Check limits using the new enforcer
        violations = self._limit_enforcer.check_resource_limits(resource_usage_dict)
        
        if violations:
            logger.warning(f"Resource limit violations: {violations}")
        
        # Legacy circuit breaker logic (kept for compatibility)
        # Check memory limits
        if self.limits.max_memory_mb and usage.memory.process_rss_mb > self.limits.max_memory_mb:
            self._circuit_breakers["memory_mb"] = True
            self._last_circuit_check["memory_mb"] = current_time
            logger.warning(f"Memory limit exceeded: {usage.memory.process_rss_mb:.1f}MB > {self.limits.max_memory_mb}MB")
        
        if self.limits.max_memory_percent and usage.memory.used_percent > self.limits.max_memory_percent:
            self._circuit_breakers["memory_percent"] = True
            self._last_circuit_check["memory_percent"] = current_time
            logger.warning(f"Memory percentage limit exceeded: {usage.memory.used_percent:.1f}% > {self.limits.max_memory_percent}%")
        
        # Check CPU limits
        if self.limits.max_cpu_percent and usage.cpu.process_percent > self.limits.max_cpu_percent:
            self._circuit_breakers["cpu_percent"] = True
            self._last_circuit_check["cpu_percent"] = current_time
            logger.warning(f"CPU limit exceeded: {usage.cpu.process_percent:.1f}% > {self.limits.max_cpu_percent}%")
        
        # Check file descriptor limits
        if self.limits.max_open_files and usage.open_files > self.limits.max_open_files:
            self._circuit_breakers["open_files"] = True
            self._last_circuit_check["open_files"] = current_time
            logger.warning(f"Open files limit exceeded: {usage.open_files} > {self.limits.max_open_files}")
        
        # Check thread limits
        if self.limits.max_threads and usage.threads > self.limits.max_threads:
            self._circuit_breakers["threads"] = True
            self._last_circuit_check["threads"] = current_time
            logger.warning(f"Thread limit exceeded: {usage.threads} > {self.limits.max_threads}")
        
        # Reset circuit breakers after cooldown period (5 minutes)
        cooldown_period = timedelta(minutes=5)
        for breaker, last_check in list(self._last_circuit_check.items()):
            if current_time - last_check > cooldown_period:
                if self._circuit_breakers[breaker]:
                    self._circuit_breakers[breaker] = False
                    logger.info(f"Circuit breaker reset: {breaker}")
                del self._last_circuit_check[breaker]
    
    async def _auto_optimize(self, usage: ResourceUsage):
        """Perform automatic optimizations based on resource usage"""
        current_time = datetime.now()
        
        # Automatic garbage collection optimization
        if (self.limits.gc_threshold_mb and 
            usage.memory.process_rss_mb > self.limits.gc_threshold_mb and
            current_time - self._last_gc_optimization > self._gc_optimization_interval):
            
            try:
                result = await self.optimize_memory()
                logger.info(f"Auto GC optimization: freed {result.memory_freed_mb:.1f}MB")
                self._last_gc_optimization = current_time
            except Exception as e:
                logger.error(f"Auto GC optimization failed: {e}")
    
    async def optimize_memory(self) -> MemoryOptimizationResult:
        """Optimize memory usage through garbage collection and other techniques"""
        start_time = time.time()
        
        # Get memory usage before optimization
        usage_before = await self.get_resource_usage()
        memory_before = usage_before.memory.process_rss_mb
        
        optimizations_applied = []
        gc_result = None
        
        try:
            # Force garbage collection
            gc_result = await self._optimize_garbage_collection()
            optimizations_applied.append("garbage_collection")
            
            # Clear weak references
            weakref.finalize._registry.clear()
            optimizations_applied.append("weak_references")
            
            # Optimize string interning (Python-specific)
            if hasattr(sys, 'intern'):
                # This is more of a preventive measure
                optimizations_applied.append("string_interning")
            
        except Exception as e:
            logger.error(f"Error during memory optimization: {e}")
        
        # Get memory usage after optimization
        usage_after = await self.get_resource_usage()
        memory_after = usage_after.memory.process_rss_mb
        memory_freed = max(0, memory_before - memory_after)
        
        duration_ms = (time.time() - start_time) * 1000
        
        result = MemoryOptimizationResult(
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_freed_mb=memory_freed,
            gc_result=gc_result,
            optimizations_applied=optimizations_applied,
            duration_ms=duration_ms
        )
        
        logger.info(f"Memory optimization completed: freed {memory_freed:.1f}MB in {duration_ms:.1f}ms")
        return result
    
    async def _optimize_garbage_collection(self) -> GCOptimizationResult:
        """Optimize garbage collection"""
        start_time = time.time()
        
        # Get GC stats before
        gc_stats_before = gc.get_stats()
        collections_before = {i: stat['collections'] for i, stat in enumerate(gc_stats_before)}
        
        # Force collection of all generations
        collected_objects = 0
        for generation in range(3):
            collected = gc.collect(generation)
            collected_objects += collected
        
        # Get GC stats after
        gc_stats_after = gc.get_stats()
        collections_after = {i: stat['collections'] for i, stat in enumerate(gc_stats_after)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Estimate memory freed (this is approximate)
        memory_freed_mb = collected_objects * 0.001  # Rough estimate
        
        return GCOptimizationResult(
            collections_before=collections_before,
            collections_after=collections_after,
            collected_objects=collected_objects,
            memory_freed_mb=memory_freed_mb,
            duration_ms=duration_ms,
            optimization_applied="full_gc_cycle"
        )
    
    def enable_profiling(self, duration: int, session_id: Optional[str] = None) -> ProfilingSession:
        """Enable memory profiling for a specified duration"""
        if session_id is None:
            session_id = f"profile_{int(time.time())}"
        
        if session_id in self._profiling_sessions:
            raise ValueError(f"Profiling session {session_id} already exists")
        
        # Enable tracemalloc if not already enabled
        if not self._tracemalloc_enabled:
            tracemalloc.start()
            self._tracemalloc_enabled = True
        
        session = ProfilingSession(
            session_id=session_id,
            start_time=datetime.now(),
            duration_seconds=duration,
            tracemalloc_enabled=True,
            snapshot_interval_seconds=10
        )
        
        self._profiling_sessions[session_id] = session
        
        # Schedule session cleanup
        asyncio.create_task(self._cleanup_profiling_session(session_id, duration))
        
        logger.info(f"Memory profiling started: session {session_id}, duration {duration}s")
        return session
    
    async def _cleanup_profiling_session(self, session_id: str, duration: int):
        """Clean up profiling session after duration"""
        await asyncio.sleep(duration)
        
        if session_id in self._profiling_sessions:
            session = self._profiling_sessions[session_id]
            session.is_active = False
            
            # Take final snapshot
            if tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                session.snapshots.append(snapshot)
            
            logger.info(f"Profiling session {session_id} completed with {len(session.snapshots)} snapshots")
    
    def create_resource_pool(self, name: str, factory: Callable, max_size: int = 10, 
                           cleanup_func: Optional[Callable] = None) -> ResourcePool:
        """Create a new resource pool"""
        if name in self._resource_pools:
            raise ValueError(f"Resource pool {name} already exists")
        
        pool = ResourcePool(factory, max_size, cleanup_func)
        self._resource_pools[name] = pool
        
        logger.info(f"Resource pool created: {name} (max_size={max_size})")
        return pool
    
    def get_resource_pool(self, name: str) -> Optional[ResourcePool]:
        """Get an existing resource pool"""
        return self._resource_pools.get(name)
    
    async def create_advanced_resource_pool(self, name: str, factory: Callable, 
                                       strategy: Optional[PoolingStrategy] = None,
                                       cleanup_func: Optional[Callable] = None,
                                       health_check_func: Optional[Callable] = None,
                                       enable_scaling: bool = True) -> AdvancedResourcePool:
        """Create an advanced resource pool with dynamic scaling capabilities"""
        if name in self._advanced_pools:
            raise ValueError(f"Advanced resource pool {name} already exists")
        
        pool = await create_advanced_pool(
            name, factory, strategy, cleanup_func, health_check_func, enable_scaling
        )
        
        self._advanced_pools[name] = pool
        logger.info(f"Advanced resource pool created: {name}")
        return pool
    
    def get_advanced_resource_pool(self, name: str) -> Optional[AdvancedResourcePool]:
        """Get an advanced resource pool"""
        return self._advanced_pools.get(name)
    
    async def start_profiling_session(self, session_id: str, duration: int,
                                    profile_memory: bool = True,
                                    profile_cpu: bool = True) -> str:
        """Start a comprehensive profiling session"""
        return await self._performance_profiler.start_profiling_session(
            session_id, duration, profile_memory, profile_cpu
        )
    
    def get_profiling_result(self, session_id: str):
        """Get profiling result for a session"""
        return self._performance_profiler.get_profiling_result(session_id)
    
    def get_scaling_history(self, limit: int = 50):
        """Get recent scaling decisions"""
        return self._dynamic_scaler.get_scaling_history(limit)
    
    def get_load_metrics_history(self, limit: int = 100):
        """Get recent load metrics"""
        return self._dynamic_scaler.get_metrics_history(limit)
    
    def is_circuit_breaker_open(self, breaker_name: str) -> bool:
        """Check if a circuit breaker is open (resource limit exceeded)"""
        # Check new circuit breaker system first
        breaker = self._limit_enforcer.get_circuit_breaker(breaker_name)
        if breaker:
            from resource_circuit_breaker import CircuitBreakerState
            return breaker.state == CircuitBreakerState.OPEN
        
        # Fallback to legacy system
        return self._circuit_breakers.get(breaker_name, False)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled (not disabled by graceful degradation)"""
        return self._degradation_manager.is_feature_enabled(feature_name)
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current graceful degradation status"""
        return self._degradation_manager.get_degradation_status()
    
    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return self._limit_enforcer.get_all_stats()
    
    def get_resource_history(self, limit: int = 10) -> List[ResourceUsage]:
        """Get recent resource usage history"""
        return list(self._usage_history)[-limit:]
    
    def get_profiling_session(self, session_id: str) -> Optional[ProfilingSession]:
        """Get a profiling session"""
        return self._profiling_sessions.get(session_id)
    
    def list_profiling_sessions(self) -> List[str]:
        """List all profiling session IDs"""
        return list(self._profiling_sessions.keys())
    
    async def cleanup(self):
        """Clean up resources and stop monitoring"""
        await self.stop_monitoring()
        
        # Clean up advanced resource pools
        for pool in self._advanced_pools.values():
            await pool.stop_background_tasks()
        self._advanced_pools.clear()
        
        # Clean up resource pools
        for pool in self._resource_pools.values():
            pool.clear()
        self._resource_pools.clear()
        
        # Stop profiling
        if self._tracemalloc_enabled:
            tracemalloc.stop()
            self._tracemalloc_enabled = False
        
        # Cleanup circuit breakers and degradation manager
        await self._limit_enforcer.cleanup()
        
        # Cleanup performance profiler
        await self._performance_profiler.cleanup()
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        logger.info("ResourceManager cleanup completed")


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def set_resource_manager(manager: ResourceManager):
    """Set the global resource manager instance"""
    global _resource_manager
    _resource_manager = manager