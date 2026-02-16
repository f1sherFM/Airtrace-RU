"""
Connection Pool Manager for AirTrace RU Backend

Implements optimized connection pooling for external APIs with health checks,
automatic retry logic, connection recycling, and comprehensive monitoring.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable
from urllib.parse import urlparse
import json
import statistics

import httpx

from config import config
from http_transport import create_async_client

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics"""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    state_change_time: float = 0
    total_requests: int = 0
    
    def get_failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failure_count / self.total_requests


@dataclass
class QueueMetrics:
    """Request queue metrics"""
    current_size: int = 0
    max_size: int = 0
    total_queued: int = 0
    total_processed: int = 0
    total_timeouts: int = 0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0
    min_wait_time: float = 0.0
    
    def get_timeout_rate(self) -> float:
        """Calculate queue timeout rate"""
        if self.total_queued == 0:
            return 0.0
        return self.total_timeouts / self.total_queued
class ConnectionStatus(Enum):
    """Connection status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    CONNECTING = "connecting"
    RECYCLING = "recycling"
    CLOSED = "closed"


class ServiceType(Enum):
    """External service type enumeration"""
    OPEN_METEO = "open_meteo"
    WEATHER_API = "weather_api"


@dataclass
class PoolConfig:
    """Connection pool configuration"""
    # Pool size settings
    max_connections: int = 20
    max_keepalive_connections: int = 10
    
    # Timeout settings
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 10.0
    pool_timeout: float = 5.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    
    # Health check settings
    health_check_interval: int = 60
    health_check_timeout: float = 10.0
    
    # Connection recycling
    connection_max_age: int = 3600  # 1 hour
    connection_idle_timeout: int = 300  # 5 minutes
    
    # Queue settings
    queue_timeout: float = 5.0
    max_queue_size: int = 100
    trust_env: bool = False


@dataclass
class ConnectionMetrics:
    """Connection pool metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    queued_requests: int = 0
    
    # Request statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    
    # Timing statistics
    avg_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    
    # Wait time statistics
    avg_wait_time: float = 0.0
    min_wait_time: float = 0.0
    max_wait_time: float = 0.0
    
    # Health statistics
    health_check_count: int = 0
    health_check_failures: int = 0
    last_health_check: Optional[datetime] = None
    
    # Connection lifecycle
    connections_created: int = 0
    connections_recycled: int = 0
    connections_failed: int = 0
    
    # Queue metrics
    queue_metrics: QueueMetrics = field(default_factory=QueueMetrics)
    
    # Circuit breaker metrics
    circuit_breaker_metrics: CircuitBreakerMetrics = field(default_factory=CircuitBreakerMetrics)
    
    def calculate_success_rate(self) -> float:
        """Calculate request success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def calculate_failure_rate(self) -> float:
        """Calculate request failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


@dataclass
class APIRequest:
    """API request wrapper"""
    method: str
    url: str
    params: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    data: Optional[Any] = None
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.params is None:
            self.params = {}


@dataclass
class APIResponse:
    """API response wrapper"""
    status_code: int
    data: Any
    headers: Dict[str, str]
    response_time: float
    retries: int = 0
    from_cache: bool = False


class CircuitBreaker:
    """Enhanced circuit breaker for connection pool with comprehensive metrics"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 success_threshold: int = 3, failure_rate_threshold: float = 0.5):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold  # Successes needed to close from half-open
        self.failure_rate_threshold = failure_rate_threshold
        
        self.metrics = CircuitBreakerMetrics()
        self.metrics.state_change_time = time.time()
        
        # Request tracking for failure rate calculation
        self.request_window = []  # List of (timestamp, success) tuples
        self.window_size = 100  # Track last 100 requests
        
        self._lock = asyncio.Lock()
    
    async def can_execute(self) -> bool:
        """Check if request can be executed"""
        async with self._lock:
            current_time = time.time()
            
            if self.metrics.state == CircuitBreakerState.CLOSED:
                return True
            elif self.metrics.state == CircuitBreakerState.OPEN:
                if current_time - self.metrics.state_change_time > self.recovery_timeout:
                    self.metrics.state = CircuitBreakerState.HALF_OPEN
                    self.metrics.state_change_time = current_time
                    logger.info(f"Circuit breaker transitioning to HALF_OPEN")
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    async def record_success(self):
        """Record successful request"""
        async with self._lock:
            current_time = time.time()
            self.metrics.success_count += 1
            self.metrics.last_success_time = current_time
            self.metrics.total_requests += 1
            
            # Add to request window
            self.request_window.append((current_time, True))
            self._trim_request_window()
            
            if self.metrics.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.success_count >= self.success_threshold:
                    self.metrics.state = CircuitBreakerState.CLOSED
                    self.metrics.state_change_time = current_time
                    self.metrics.failure_count = 0  # Reset failure count
                    logger.info(f"Circuit breaker transitioning to CLOSED after {self.metrics.success_count} successes")
    
    async def record_failure(self):
        """Record failed request"""
        async with self._lock:
            current_time = time.time()
            self.metrics.failure_count += 1
            self.metrics.last_failure_time = current_time
            self.metrics.total_requests += 1
            
            # Add to request window
            self.request_window.append((current_time, False))
            self._trim_request_window()
            
            # Check if we should open the circuit
            should_open = False
            
            if self.metrics.state == CircuitBreakerState.HALF_OPEN:
                # In half-open, any failure opens the circuit
                should_open = True
            elif self.metrics.state == CircuitBreakerState.CLOSED:
                # Check failure threshold and failure rate
                if self.metrics.failure_count >= self.failure_threshold:
                    failure_rate = self._calculate_failure_rate()
                    if failure_rate >= self.failure_rate_threshold:
                        should_open = True
            
            if should_open:
                self.metrics.state = CircuitBreakerState.OPEN
                self.metrics.state_change_time = current_time
                self.metrics.success_count = 0  # Reset success count
                logger.warning(f"Circuit breaker OPENED after {self.metrics.failure_count} failures (rate: {self._calculate_failure_rate():.2%})")
    
    def _trim_request_window(self):
        """Trim request window to maintain size limit"""
        if len(self.request_window) > self.window_size:
            self.request_window = self.request_window[-self.window_size:]
    
    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate from request window"""
        if not self.request_window:
            return 0.0
        
        failures = sum(1 for _, success in self.request_window if not success)
        return failures / len(self.request_window)
    
    async def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics"""
        async with self._lock:
            # Update failure rate
            self.metrics.failure_count = sum(1 for _, success in self.request_window if not success)
            return CircuitBreakerMetrics(
                state=self.metrics.state,
                failure_count=self.metrics.failure_count,
                success_count=self.metrics.success_count,
                last_failure_time=self.metrics.last_failure_time,
                last_success_time=self.metrics.last_success_time,
                state_change_time=self.metrics.state_change_time,
                total_requests=self.metrics.total_requests
            )


class ConnectionPool:
    """Individual connection pool for a service with enhanced queuing and circuit breaker"""
    
    def __init__(self, service_type: ServiceType, base_url: str, config: PoolConfig):
        self.service_type = service_type
        self.base_url = base_url
        self.config = config
        
        # HTTP client with connection pooling
        self.client = create_async_client(
            max_connections=config.max_connections,
            max_keepalive_connections=config.max_keepalive_connections,
            connect_timeout=config.connect_timeout,
            read_timeout=config.read_timeout,
            write_timeout=config.write_timeout,
            pool_timeout=config.pool_timeout,
            trust_env=config.trust_env,
        )
        
        # Pool state
        self.status = ConnectionStatus.HEALTHY
        self.metrics = ConnectionMetrics()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=3,
            failure_rate_threshold=0.5
        )
        
        # Enhanced request queue with priority and timeout tracking
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.queue_workers: List[asyncio.Task] = []
        self.active_requests = 0
        self.max_concurrent_requests = config.max_connections
        
        # Queue metrics tracking
        self.queue_wait_times: List[float] = []
        self.queue_stats_lock = asyncio.Lock()
        
        # Health check
        self.last_health_check = 0
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Connection recycling
        self.connection_created_time = time.time()
        
        # Statistics tracking
        self.response_times: List[float] = []
        self.stats_lock = asyncio.Lock()
        
        # Start background tasks (deferred until first use)
        self._background_tasks_started = False
        
        # Request semaphore for connection limiting
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
    
    def _start_background_tasks(self):
        """Start background tasks for health checks and queue processing"""
        if not self._background_tasks_started:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            # Start queue worker tasks
            for i in range(min(3, self.max_concurrent_requests)):  # Start 3 workers or max connections, whichever is smaller
                worker = asyncio.create_task(self._queue_worker(f"worker-{i}"))
                self.queue_workers.append(worker)
            self._background_tasks_started = True
    
    async def _queue_worker(self, worker_id: str):
        """Background worker to process queued requests"""
        logger.debug(f"Queue worker {worker_id} started for {self.service_type.value}")
        
        while True:
            try:
                # Get request from queue with timeout
                try:
                    queue_item = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=1.0  # Check for shutdown every second
                    )
                except asyncio.TimeoutError:
                    continue
                
                request, future, queue_start_time = queue_item
                
                # Calculate wait time
                wait_time = time.time() - queue_start_time
                await self._update_queue_metrics(wait_time, False)  # False = not timeout
                
                try:
                    # Execute the request
                    result = await self._execute_request_internal(request)
                    if not future.cancelled():
                        future.set_result(result)
                except Exception as e:
                    if not future.cancelled():
                        future.set_exception(e)
                finally:
                    self.request_queue.task_done()
                    
            except asyncio.CancelledError:
                logger.debug(f"Queue worker {worker_id} cancelled for {self.service_type.value}")
                break
            except Exception as e:
                logger.error(f"Queue worker {worker_id} error for {self.service_type.value}: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def _update_queue_metrics(self, wait_time: float, is_timeout: bool):
        """Update queue metrics"""
        async with self.queue_stats_lock:
            self.queue_wait_times.append(wait_time)
            
            # Keep only last 1000 wait times
            if len(self.queue_wait_times) > 1000:
                self.queue_wait_times = self.queue_wait_times[-1000:]
            
            # Update queue metrics
            self.metrics.queue_metrics.total_processed += 1
            if is_timeout:
                self.metrics.queue_metrics.total_timeouts += 1
            
            if self.queue_wait_times:
                self.metrics.queue_metrics.avg_wait_time = statistics.mean(self.queue_wait_times)
                self.metrics.queue_metrics.min_wait_time = min(self.queue_wait_times)
                self.metrics.queue_metrics.max_wait_time = max(self.queue_wait_times)
    
    async def execute_request(self, request: APIRequest) -> APIResponse:
        """Execute API request with enhanced queuing and circuit breaker"""
        # Start background tasks on first use
        if not self._background_tasks_started:
            self._start_background_tasks()
        
        # Check circuit breaker
        if not await self.circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker open for {self.service_type.value}")
        
        # Try to acquire semaphore immediately (non-blocking)
        acquired = False
        try:
            # Use asyncio.wait_for with timeout=0 for non-blocking acquire
            await asyncio.wait_for(self.request_semaphore.acquire(), timeout=0)
            acquired = True
            # Execute immediately if we can get a connection
            try:
                return await self._execute_request_internal(request)
            finally:
                self.request_semaphore.release()
        except asyncio.TimeoutError:
            # No immediate connection available, queue the request
            return await self._queue_request(request)
    
    async def _queue_request(self, request: APIRequest) -> APIResponse:
        """Queue request when pool is exhausted"""
        queue_start_time = time.time()
        future = asyncio.Future()
        
        try:
            # Try to add to queue
            self.request_queue.put_nowait((request, future, queue_start_time))
            
            async with self.queue_stats_lock:
                self.metrics.queue_metrics.total_queued += 1
                self.metrics.queue_metrics.current_size = self.request_queue.qsize()
                if self.metrics.queue_metrics.current_size > self.metrics.queue_metrics.max_size:
                    self.metrics.queue_metrics.max_size = self.metrics.queue_metrics.current_size
            
            # Wait for result with timeout
            try:
                return await asyncio.wait_for(future, timeout=self.config.queue_timeout)
            except asyncio.TimeoutError:
                # Request timed out in queue
                wait_time = time.time() - queue_start_time
                await self._update_queue_metrics(wait_time, True)  # True = timeout
                
                # Try to cancel the future
                if not future.done():
                    future.cancel()
                
                raise Exception(f"Request timed out in queue after {wait_time:.2f}s for {self.service_type.value}")
                
        except asyncio.QueueFull:
            # Queue is full, reject immediately
            async with self.queue_stats_lock:
                self.metrics.queue_metrics.total_timeouts += 1
            
            raise Exception(f"Request queue full for {self.service_type.value} (size: {self.request_queue.qsize()})")
    
    async def _execute_request_internal(self, request: APIRequest) -> APIResponse:
        """Internal request execution with retry logic"""
        retries = 0
        last_exception = None
        
        while retries <= self.config.max_retries:
            try:
                start_time = time.time()
                
                # Execute HTTP request
                response = await self.client.request(
                    method=request.method,
                    url=request.url,
                    params=request.params,
                    headers=request.headers,
                    content=request.data,
                    timeout=request.timeout or self.config.read_timeout
                )
                
                response_time = time.time() - start_time
                
                # Update metrics
                await self._update_request_metrics(response_time, True, retries)
                
                # Record success in circuit breaker
                await self.circuit_breaker.record_success()
                
                return APIResponse(
                    status_code=response.status_code,
                    data=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    headers=dict(response.headers),
                    response_time=response_time,
                    retries=retries
                )
                
            except Exception as e:
                last_exception = e
                retries += 1
                
                if retries <= self.config.max_retries:
                    delay = self.config.retry_delay * (self.config.backoff_factor ** (retries - 1))
                    logger.warning(f"Request failed for {self.service_type.value}, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    # Record failure in circuit breaker
                    await self.circuit_breaker.record_failure()
                    
                    # Update metrics
                    await self._update_request_metrics(0, False, retries - 1)
                    
                    logger.error(f"Request failed after {retries - 1} retries for {self.service_type.value}: {e}")
        
        raise last_exception
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error for {self.service_type.value}: {e}")
    
    async def _perform_health_check(self) -> bool:
        """Perform health check on the connection pool"""
        # Start background tasks if not already started
        if not self._background_tasks_started:
            self._start_background_tasks()
        
        try:
            start_time = time.time()
            
            # Simple ping request based on service type
            if self.service_type == ServiceType.OPEN_METEO:
                # Test with minimal Open-Meteo request
                test_url = f"{self.base_url}?latitude=55.7558&longitude=37.6176&current=pm10"
            elif self.service_type == ServiceType.WEATHER_API:
                # Test with minimal WeatherAPI request
                from config import config
                test_url = f"{self.base_url}/current.json?key={config.weather_api.api_key}&q=55.7558,37.6176&aqi=no"
            else:
                # Generic health check
                test_url = self.base_url
            
            response = await self.client.get(
                test_url,
                timeout=self.config.health_check_timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.status = ConnectionStatus.HEALTHY
                await self.circuit_breaker.record_success()
                
                async with self.stats_lock:
                    self.metrics.health_check_count += 1
                    self.metrics.last_health_check = datetime.now(timezone.utc)
                
                logger.debug(f"Health check passed for {self.service_type.value} ({response_time:.3f}s)")
                return True
            else:
                raise httpx.HTTPStatusError(
                    f"Health check failed with status {response.status_code}",
                    request=response.request,
                    response=response
                )
                
        except Exception as e:
            self.status = ConnectionStatus.UNHEALTHY
            await self.circuit_breaker.record_failure()
            
            async with self.stats_lock:
                self.metrics.health_check_failures += 1
                self.metrics.last_health_check = datetime.now(timezone.utc)
            
            logger.warning(f"Health check failed for {self.service_type.value}: {e}")
            return False
    
    async def get_metrics(self) -> ConnectionMetrics:
        """Get current pool metrics with enhanced queue and circuit breaker metrics"""
        async with self.stats_lock:
            # Update connection counts from client
            try:
                # Get connection pool info from httpx client
                pool_info = self.client._transport._pool._pool_for_url(httpx.URL(self.base_url))
                if pool_info:
                    self.metrics.active_connections = len([c for c in pool_info._connections if c.is_connection_dropped() == False])
                    self.metrics.idle_connections = len([c for c in pool_info._connections if c.is_idle()])
                    self.metrics.total_connections = len(pool_info._connections)
            except Exception:
                # Fallback if we can't get detailed connection info
                pass
            
            # Update queue metrics
            self.metrics.queue_metrics.current_size = self.request_queue.qsize()
            
            # Update wait time statistics
            if self.queue_wait_times:
                self.metrics.avg_wait_time = statistics.mean(self.queue_wait_times)
                self.metrics.min_wait_time = min(self.queue_wait_times)
                self.metrics.max_wait_time = max(self.queue_wait_times)
            
            # Get circuit breaker metrics
            self.metrics.circuit_breaker_metrics = await self.circuit_breaker.get_metrics()
            
            return ConnectionMetrics(
                total_connections=self.metrics.total_connections,
                active_connections=self.metrics.active_connections,
                idle_connections=self.metrics.idle_connections,
                queued_requests=self.metrics.queue_metrics.current_size,
                total_requests=self.metrics.total_requests,
                successful_requests=self.metrics.successful_requests,
                failed_requests=self.metrics.failed_requests,
                retried_requests=self.metrics.retried_requests,
                avg_response_time=self.metrics.avg_response_time,
                min_response_time=self.metrics.min_response_time,
                max_response_time=self.metrics.max_response_time,
                avg_wait_time=self.metrics.avg_wait_time,
                min_wait_time=self.metrics.min_wait_time,
                max_wait_time=self.metrics.max_wait_time,
                health_check_count=self.metrics.health_check_count,
                health_check_failures=self.metrics.health_check_failures,
                last_health_check=self.metrics.last_health_check,
                connections_created=self.metrics.connections_created,
                connections_recycled=self.metrics.connections_recycled,
                connections_failed=self.metrics.connections_failed,
                queue_metrics=self.metrics.queue_metrics,
                circuit_breaker_metrics=self.metrics.circuit_breaker_metrics
            )
    
    async def _update_request_metrics(self, response_time: float, success: bool, retries: int):
        """Update request metrics"""
        async with self.stats_lock:
            self.metrics.total_requests += 1
            
            if success:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1
            
            if retries > 0:
                self.metrics.retried_requests += 1
            
            # Update response time statistics
            if response_time > 0:
                self.response_times.append(response_time)
                
                # Keep only last 1000 response times for statistics
                if len(self.response_times) > 1000:
                    self.response_times = self.response_times[-1000:]
                
                self.metrics.avg_response_time = statistics.mean(self.response_times) if self.response_times else 0.0
                self.metrics.min_response_time = min(self.response_times) if self.response_times else 0.0
                self.metrics.max_response_time = max(self.response_times) if self.response_times else 0.0
    
    def should_recycle(self) -> bool:
        """Check if connection pool should be recycled"""
        age = time.time() - self.connection_created_time
        return age > self.config.connection_max_age
    
    async def recycle_connections(self):
        """Recycle connections in the pool"""
        try:
            self.status = ConnectionStatus.RECYCLING
            
            # Close current client
            await self.client.aclose()
            
            # Create new client
            self.client = create_async_client(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
                connect_timeout=self.config.connect_timeout,
                read_timeout=self.config.read_timeout,
                write_timeout=self.config.write_timeout,
                pool_timeout=self.config.pool_timeout,
                trust_env=self.config.trust_env,
            )
            
            self.connection_created_time = time.time()
            self.status = ConnectionStatus.HEALTHY
            
            async with self.stats_lock:
                self.metrics.connections_recycled += 1
            
            logger.info(f"Recycled connections for {self.service_type.value}")
            
        except Exception as e:
            self.status = ConnectionStatus.UNHEALTHY
            logger.error(f"Connection recycling failed for {self.service_type.value}: {e}")
    
    async def cleanup(self):
        """Cleanup pool resources including queue workers"""
        try:
            # Cancel health check task
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel queue workers
            for worker in self.queue_workers:
                worker.cancel()
            
            # Wait for queue workers to finish
            if self.queue_workers:
                await asyncio.gather(*self.queue_workers, return_exceptions=True)
            
            # Clear any remaining queue items
            while not self.request_queue.empty():
                try:
                    _, future, _ = self.request_queue.get_nowait()
                    if not future.done():
                        future.cancel()
                    self.request_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            # Close HTTP client
            await self.client.aclose()
            
            self.status = ConnectionStatus.CLOSED
            logger.info(f"Connection pool cleaned up for {self.service_type.value}")
            
        except Exception as e:
            logger.error(f"Connection pool cleanup failed for {self.service_type.value}: {e}")


class ConnectionPoolManager:
    """
    Connection pool manager for external APIs.
    
    Manages multiple connection pools for different services with health monitoring,
    automatic retry logic, and comprehensive metrics collection.
    """
    
    def __init__(self):
        self.pools: Dict[ServiceType, ConnectionPool] = {}
        self.default_config = PoolConfig(
            connect_timeout=config.api.connect_timeout,
            read_timeout=config.api.read_timeout,
            write_timeout=config.api.write_timeout,
            pool_timeout=config.api.pool_timeout,
            max_retries=config.api.max_retries,
            retry_delay=config.api.retry_delay,
            backoff_factor=config.api.backoff_factor,
            trust_env=config.api.trust_env,
        )
        
        # Initialize Open-Meteo pool
        self._initialize_open_meteo_pool()
    
    def _initialize_open_meteo_pool(self):
        """Initialize connection pool for Open-Meteo API"""
        open_meteo_config = PoolConfig(
            max_connections=20,
            max_keepalive_connections=10,
            connect_timeout=config.api.connect_timeout,
            read_timeout=config.api.read_timeout,
            write_timeout=config.api.write_timeout,
            pool_timeout=config.api.pool_timeout,
            max_retries=config.api.max_retries,
            retry_delay=config.api.retry_delay,
            backoff_factor=config.api.backoff_factor,
            health_check_interval=60,
            connection_max_age=3600,
            queue_timeout=5.0,
            trust_env=config.api.trust_env,
        )
        
        self.pools[ServiceType.OPEN_METEO] = ConnectionPool(
            service_type=ServiceType.OPEN_METEO,
            base_url="https://air-quality-api.open-meteo.com/v1/air-quality",
            config=open_meteo_config
        )
        
        logger.info("Open-Meteo connection pool initialized")
        
        # Initialize WeatherAPI pool if enabled
        self._initialize_weather_api_pool()
    
    def _initialize_weather_api_pool(self):
        """Initialize connection pool for WeatherAPI.com"""
        from config import config
        
        if not config.weather_api.enabled:
            logger.info("WeatherAPI disabled - skipping pool initialization")
            return
        
        weather_api_config = PoolConfig(
            max_connections=15,
            max_keepalive_connections=8,
            connect_timeout=config.api.connect_timeout,
            read_timeout=config.weather_api.timeout,
            write_timeout=config.api.write_timeout,
            pool_timeout=config.api.pool_timeout,
            max_retries=config.weather_api.max_retries,
            retry_delay=config.api.retry_delay,
            backoff_factor=config.api.backoff_factor,
            health_check_interval=120,  # Less frequent health checks for external paid API
            connection_max_age=3600,
            queue_timeout=5.0,
            trust_env=config.api.trust_env,
        )
        
        self.pools[ServiceType.WEATHER_API] = ConnectionPool(
            service_type=ServiceType.WEATHER_API,
            base_url=config.weather_api.base_url,
            config=weather_api_config
        )
        
        logger.info("WeatherAPI connection pool initialized")
    
    async def get_connection(self, service: ServiceType) -> ConnectionPool:
        """Get connection pool for service"""
        if service not in self.pools:
            raise ValueError(f"No connection pool configured for service: {service}")
        
        pool = self.pools[service]
        
        # Check if pool needs recycling
        if pool.should_recycle():
            await pool.recycle_connections()
        
        return pool
    
    async def execute_request(self, service: ServiceType, request: APIRequest) -> APIResponse:
        """Execute request using appropriate connection pool"""
        pool = await self.get_connection(service)
        return await pool.execute_request(request)
    
    async def get_pool_stats(self, service: ServiceType) -> ConnectionMetrics:
        """Get statistics for specific connection pool"""
        if service not in self.pools:
            raise ValueError(f"No connection pool configured for service: {service}")
        
        return await self.pools[service].get_metrics()
    
    async def get_all_stats(self) -> Dict[str, ConnectionMetrics]:
        """Get statistics for all connection pools"""
        stats = {}
        for service_type, pool in self.pools.items():
            stats[service_type.value] = await pool.get_metrics()
        return stats
    
    def configure_pool(self, service: ServiceType, config: PoolConfig):
        """Configure connection pool for service"""
        if service in self.pools:
            # Close existing pool
            asyncio.create_task(self.pools[service].cleanup())
        
        # Create new pool with updated configuration
        if service == ServiceType.OPEN_METEO:
            base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        elif service == ServiceType.WEATHER_API:
            from config import config
            base_url = config.weather_api.base_url
        else:
            raise ValueError(f"Unknown service type: {service}")
        
        self.pools[service] = ConnectionPool(
            service_type=service,
            base_url=base_url,
            config=config
        )
        
        logger.info(f"Connection pool configured for {service.value}")
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all pools"""
        results = {}
        for service_type, pool in self.pools.items():
            try:
                results[service_type.value] = await pool._perform_health_check()
            except Exception as e:
                logger.error(f"Health check failed for {service_type.value}: {e}")
                results[service_type.value] = False
        return results
    
    async def cleanup(self):
        """Cleanup all connection pools"""
        cleanup_tasks = []
        for pool in self.pools.values():
            cleanup_tasks.append(pool.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.pools.clear()
        logger.info("All connection pools cleaned up")


# Global connection pool manager instance - initialized lazily
connection_pool_manager = None

def get_connection_pool_manager() -> ConnectionPoolManager:
    """Get or create the global connection pool manager"""
    global connection_pool_manager
    if connection_pool_manager is None:
        connection_pool_manager = ConnectionPoolManager()
    return connection_pool_manager
