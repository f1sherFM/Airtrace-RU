"""
Rate limiting system for AirTrace RU Backend

Implements intelligent rate limiting with sliding window algorithms,
burst traffic handling, and Redis-based distributed rate limiting.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timezone, timedelta

import redis.asyncio as redis
from redis.asyncio import Redis, RedisCluster
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from rate_limit_types import (
    RateLimitStrategy, EndpointCategory, RateLimitConfig, 
    RateLimitResult, RateLimitInfo, RateLimitMetrics
)
from config import config

logger = logging.getLogger(__name__)

# Import monitoring after logger to avoid circular imports
try:
    from rate_limit_monitoring import get_rate_limit_monitor, ViolationType, RateLimitConfigValidator
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning("Rate limit monitoring not available due to import error")


class RateLimiter:
    """
    Intelligent rate limiter with sliding window algorithm.
    
    Features:
    - Sliding window rate limiting using Redis
    - Burst traffic handling with configurable multipliers
    - Different limits per endpoint category
    - IP-based and User-Agent-based identification
    - Comprehensive metrics collection
    """
    
    def __init__(self):
        # Redis connection
        self._redis_client: Optional[Union[Redis, RedisCluster]] = None
        self._redis_enabled = config.performance.rate_limiting_enabled and config.performance.redis_enabled
        self._redis_healthy = False
        self._last_health_check = 0
        self._health_check_interval = config.redis.health_check_interval
        
        # Rate limit configurations
        self._endpoint_configs: Dict[EndpointCategory, RateLimitConfig] = {}
        self._setup_default_configs()
        
        # In-memory fallback for when Redis is unavailable
        self._memory_counters: Dict[str, Dict[str, Any]] = {}
        self._memory_cleanup_interval = 300  # 5 minutes
        self._last_memory_cleanup = time.time()
        
        # Metrics
        self._metrics = RateLimitMetrics()
        self._metrics_lock = asyncio.Lock()
        
        # Key prefix for Redis
        self._key_prefix = "airtrace:ratelimit:v1"
        
        # Initialize Redis connection if enabled
        if self._redis_enabled:
            asyncio.create_task(self._initialize_redis())
    
    def _setup_default_configs(self):
        """Setup default rate limit configurations"""
        self._endpoint_configs = {
            EndpointCategory.AIR_QUALITY: RateLimitConfig(
                requests_per_minute=100,
                burst_multiplier=1.5,
                window_size_seconds=60,
                identifier_strategy=RateLimitStrategy.IP_BASED
            ),
            EndpointCategory.BATCH_REQUESTS: RateLimitConfig(
                requests_per_minute=10,
                burst_multiplier=1.2,
                window_size_seconds=60,
                identifier_strategy=RateLimitStrategy.COMBINED
            ),
            EndpointCategory.HEALTH_CHECKS: RateLimitConfig(
                requests_per_minute=1000,
                burst_multiplier=2.0,
                window_size_seconds=60,
                identifier_strategy=RateLimitStrategy.IP_BASED
            ),
            EndpointCategory.METRICS: RateLimitConfig(
                requests_per_minute=60,
                burst_multiplier=1.3,
                window_size_seconds=60,
                identifier_strategy=RateLimitStrategy.IP_BASED
            ),
            EndpointCategory.DEFAULT: RateLimitConfig(
                requests_per_minute=60,
                burst_multiplier=1.5,
                window_size_seconds=60,
                identifier_strategy=RateLimitStrategy.IP_BASED
            )
        }
    
    async def _initialize_redis(self):
        """Initialize Redis connection for distributed rate limiting"""
        try:
            if config.redis.cluster_enabled:
                cluster_kwargs = config.get_redis_cluster_kwargs()
                if cluster_kwargs.get("startup_nodes"):
                    self._redis_client = RedisCluster(**cluster_kwargs)
                    logger.info("Rate limiter Redis cluster client initialized")
                else:
                    logger.warning("Redis cluster enabled but no nodes configured for rate limiter")
                    self._redis_enabled = False
                    return
            else:
                redis_kwargs = config.get_redis_connection_kwargs()
                self._redis_client = Redis(**redis_kwargs)
                logger.info("Rate limiter Redis client initialized")
            
            # Test connection
            await self._check_redis_health()
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis for rate limiter: {e}")
            self._redis_enabled = False
            self._redis_healthy = False
    
    async def _check_redis_health(self) -> bool:
        """Check Redis connection health"""
        if not self._redis_client or not self._redis_enabled:
            return False
        
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return self._redis_healthy
        
        try:
            await self._redis_client.ping()
            self._redis_healthy = True
            self._last_health_check = current_time
            return True
        except Exception as e:
            logger.warning(f"Rate limiter Redis health check failed: {e}")
            self._redis_healthy = False
            self._last_health_check = current_time
            return False
    
    def _generate_identifier(self, ip: str, user_agent: Optional[str], strategy: RateLimitStrategy) -> str:
        """Generate rate limit identifier based on strategy"""
        if strategy == RateLimitStrategy.IP_BASED:
            # Hash IP for privacy
            return hashlib.md5(ip.encode()).hexdigest()
        elif strategy == RateLimitStrategy.USER_AGENT_BASED:
            if user_agent:
                return hashlib.md5(user_agent.encode()).hexdigest()
            else:
                # Fallback to IP if no user agent
                return hashlib.md5(ip.encode()).hexdigest()
        elif strategy == RateLimitStrategy.COMBINED:
            combined = f"{ip}:{user_agent or 'unknown'}"
            return hashlib.md5(combined.encode()).hexdigest()
        else:
            return hashlib.md5(ip.encode()).hexdigest()
    
    def _generate_redis_key(self, identifier: str, endpoint: str, window_start: int) -> str:
        """Generate Redis key for rate limiting"""
        return f"{self._key_prefix}:{endpoint}:{identifier}:{window_start}"
    
    def _get_window_start(self, window_size: int) -> int:
        """Get current window start timestamp"""
        current_time = int(time.time())
        return (current_time // window_size) * window_size
    
    async def _check_rate_limit_redis(self, identifier: str, endpoint: str, config: RateLimitConfig) -> RateLimitResult:
        """Check rate limit using Redis sliding window"""
        if not await self._check_redis_health():
            # Fallback to memory-based rate limiting
            return await self._check_rate_limit_memory(identifier, endpoint, config)
        
        try:
            current_time = int(time.time())
            window_start = self._get_window_start(config.window_size_seconds)
            window_end = window_start + config.window_size_seconds
            
            # Use Redis pipeline for atomic operations
            pipe = self._redis_client.pipeline()
            
            # Get current count for this window
            redis_key = self._generate_redis_key(identifier, endpoint, window_start)
            pipe.get(redis_key)
            
            # Also check previous window for sliding window effect
            prev_window_start = window_start - config.window_size_seconds
            prev_redis_key = self._generate_redis_key(identifier, endpoint, prev_window_start)
            pipe.get(prev_redis_key)
            
            results = await pipe.execute()
            
            current_count = int(results[0] or 0)
            prev_count = int(results[1] or 0)
            
            # Calculate sliding window count
            time_into_window = current_time - window_start
            weight = time_into_window / config.window_size_seconds
            sliding_count = int(prev_count * (1 - weight) + current_count)
            
            # Calculate limits
            base_limit = config.requests_per_minute
            burst_limit = int(base_limit * config.burst_multiplier)
            
            # Determine if request is allowed
            allowed = sliding_count < burst_limit
            
            if allowed:
                # Increment counter
                pipe = self._redis_client.pipeline()
                pipe.incr(redis_key)
                pipe.expire(redis_key, config.window_size_seconds * 2)  # Keep for 2 windows
                await pipe.execute()
                
                current_count += 1
                sliding_count += 1
            
            # Calculate remaining requests
            remaining = max(0, burst_limit - sliding_count)
            
            # Calculate retry after if blocked
            retry_after = None
            if not allowed:
                retry_after = window_end - current_time
            
            # Update metrics
            await self._update_metrics(allowed, sliding_count > base_limit)
            
            # Record monitoring data
            if MONITORING_AVAILABLE:
                monitor = get_rate_limit_monitor()
                monitor.record_request(endpoint, allowed, identifier)
                
                if not allowed:
                    # Record violation
                    monitor.record_violation(
                        identifier_hash=identifier,
                        endpoint=endpoint,
                        violation_type=ViolationType.BURST_EXCEEDED,
                        current_usage=sliding_count,
                        limit=burst_limit,
                        window_start=datetime.fromtimestamp(window_start, timezone.utc),
                        window_end=datetime.fromtimestamp(window_end, timezone.utc)
                    )
            
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=datetime.fromtimestamp(window_end, timezone.utc),
                retry_after=retry_after,
                current_usage=sliding_count,
                limit=burst_limit
            )
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to memory-based rate limiting
            return await self._check_rate_limit_memory(identifier, endpoint, config)
    
    async def _check_rate_limit_memory(self, identifier: str, endpoint: str, config: RateLimitConfig) -> RateLimitResult:
        """Fallback memory-based rate limiting"""
        current_time = time.time()
        window_start = self._get_window_start(config.window_size_seconds)
        window_end = window_start + config.window_size_seconds
        
        # Cleanup old entries periodically
        if current_time - self._last_memory_cleanup > self._memory_cleanup_interval:
            await self._cleanup_memory_counters()
        
        # Get or create counter for this identifier/endpoint
        key = f"{identifier}:{endpoint}"
        if key not in self._memory_counters:
            self._memory_counters[key] = {
                "count": 0,
                "window_start": window_start,
                "burst_used": 0
            }
        
        counter = self._memory_counters[key]
        
        # Reset counter if we're in a new window
        if counter["window_start"] != window_start:
            counter["count"] = 0
            counter["window_start"] = window_start
            counter["burst_used"] = 0
        
        # Calculate limits
        base_limit = config.requests_per_minute
        burst_limit = int(base_limit * config.burst_multiplier)
        
        # Check if request is allowed
        allowed = counter["count"] < burst_limit
        
        if allowed:
            counter["count"] += 1
            if counter["count"] > base_limit:
                counter["burst_used"] += 1
        
        # Calculate remaining requests
        remaining = max(0, burst_limit - counter["count"])
        
        # Calculate retry after if blocked
        retry_after = None
        if not allowed:
            retry_after = int(window_end - current_time)
        
        # Update metrics
        await self._update_metrics(allowed, counter["count"] > base_limit)
        
        # Record monitoring data
        if MONITORING_AVAILABLE:
            monitor = get_rate_limit_monitor()
            monitor.record_request(endpoint, allowed, identifier)
            
            if not allowed:
                # Record violation
                monitor.record_violation(
                    identifier_hash=identifier,
                    endpoint=endpoint,
                    violation_type=ViolationType.BURST_EXCEEDED,
                    current_usage=counter["count"],
                    limit=burst_limit,
                    window_start=datetime.fromtimestamp(window_start, timezone.utc),
                    window_end=datetime.fromtimestamp(window_end, timezone.utc)
                )
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=datetime.fromtimestamp(window_end, timezone.utc),
            retry_after=retry_after,
            current_usage=counter["count"],
            limit=burst_limit
        )
    
    async def _cleanup_memory_counters(self):
        """Clean up expired memory counters"""
        current_time = time.time()
        expired_keys = []
        
        for key, counter in self._memory_counters.items():
            window_age = current_time - counter["window_start"]
            if window_age > 300:  # Remove counters older than 5 minutes
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._memory_counters[key]
        
        self._last_memory_cleanup = current_time
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit counters")
    
    async def _update_metrics(self, allowed: bool, is_burst: bool):
        """Update rate limiting metrics"""
        async with self._metrics_lock:
            self._metrics.total_requests += 1
            
            if allowed:
                self._metrics.allowed_requests += 1
                if is_burst:
                    self._metrics.burst_requests += 1
            else:
                self._metrics.blocked_requests += 1
    
    def configure_limits(self, endpoint: EndpointCategory, config: RateLimitConfig) -> None:
        """Configure rate limits for an endpoint category"""
        # Validate configuration
        if MONITORING_AVAILABLE:
            validation_errors = RateLimitConfigValidator.validate_config(config)
            if validation_errors:
                error_msg = f"Invalid rate limit configuration for {endpoint.value}: {'; '.join(validation_errors)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        self._endpoint_configs[endpoint] = config
        logger.info(f"Rate limit configured for {endpoint.value}: {config.requests_per_minute} req/min (burst: {config.burst_multiplier}x)")
        
        # Log configuration details
        burst_limit = int(config.requests_per_minute * config.burst_multiplier)
        logger.info(
            f"Rate limit details for {endpoint.value}: "
            f"Base: {config.requests_per_minute}/min, "
            f"Burst: {burst_limit}/min, "
            f"Window: {config.window_size_seconds}s, "
            f"Strategy: {config.identifier_strategy.value}"
        )
    
    def validate_all_configurations(self) -> Dict[str, List[str]]:
        """Validate all endpoint configurations"""
        if MONITORING_AVAILABLE:
            return RateLimitConfigValidator.validate_endpoint_configs(self._endpoint_configs)
        return {}
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all rate limit configurations"""
        summary = {}
        for category, config in self._endpoint_configs.items():
            burst_limit = int(config.requests_per_minute * config.burst_multiplier)
            summary[category.value] = {
                "requests_per_minute": config.requests_per_minute,
                "burst_limit": burst_limit,
                "burst_multiplier": config.burst_multiplier,
                "window_size_seconds": config.window_size_seconds,
                "identifier_strategy": config.identifier_strategy.value
            }
        
        # Add validation results
        validation_results = self.validate_all_configurations()
        if validation_results:
            summary["validation_errors"] = validation_results
        
        return summary
    
    def get_endpoint_category(self, endpoint_path: str) -> EndpointCategory:
        """Determine endpoint category from path"""
        if "/weather/" in endpoint_path:
            return EndpointCategory.AIR_QUALITY
        elif "/health" in endpoint_path:
            return EndpointCategory.HEALTH_CHECKS
        elif "/metrics" in endpoint_path:
            return EndpointCategory.METRICS
        elif "batch" in endpoint_path.lower():
            return EndpointCategory.BATCH_REQUESTS
        else:
            return EndpointCategory.DEFAULT
    
    async def check_rate_limit(self, ip: str, endpoint: str, user_agent: Optional[str] = None) -> RateLimitResult:
        """
        Check if request is within rate limits.
        
        Args:
            ip: Client IP address
            endpoint: API endpoint path
            user_agent: Client user agent string
            
        Returns:
            RateLimitResult with allow/deny decision and metadata
        """
        # Get endpoint category and configuration
        category = self.get_endpoint_category(endpoint)
        config = self._endpoint_configs.get(category, self._endpoint_configs[EndpointCategory.DEFAULT])
        
        # Generate identifier based on strategy
        identifier = self._generate_identifier(ip, user_agent, config.identifier_strategy)
        
        # Check rate limit using Redis or memory fallback
        if self._redis_enabled:
            return await self._check_rate_limit_redis(identifier, category.value, config)
        else:
            return await self._check_rate_limit_memory(identifier, category.value, config)
    
    async def get_rate_limit_info(self, ip: str, endpoint: str, user_agent: Optional[str] = None) -> RateLimitInfo:
        """Get current rate limit information for an identifier"""
        category = self.get_endpoint_category(endpoint)
        config = self._endpoint_configs.get(category, self._endpoint_configs[EndpointCategory.DEFAULT])
        identifier = self._generate_identifier(ip, user_agent, config.identifier_strategy)
        
        current_time = time.time()
        window_start = self._get_window_start(config.window_size_seconds)
        window_end = window_start + config.window_size_seconds
        
        # Get current count
        current_count = 0
        burst_used = 0
        
        if self._redis_enabled and await self._check_redis_health():
            try:
                redis_key = self._generate_redis_key(identifier, category.value, window_start)
                current_count = int(await self._redis_client.get(redis_key) or 0)
            except Exception as e:
                logger.warning(f"Failed to get rate limit info from Redis: {e}")
        else:
            # Use memory counters
            key = f"{identifier}:{category.value}"
            if key in self._memory_counters:
                counter = self._memory_counters[key]
                if counter["window_start"] == window_start:
                    current_count = counter["count"]
                    burst_used = counter.get("burst_used", 0)
        
        return RateLimitInfo(
            identifier=identifier,
            endpoint=endpoint,
            current_count=current_count,
            limit=int(config.requests_per_minute * config.burst_multiplier),
            window_start=datetime.fromtimestamp(window_start, timezone.utc),
            window_end=datetime.fromtimestamp(window_end, timezone.utc),
            burst_used=burst_used
        )
    
    async def get_metrics(self) -> RateLimitMetrics:
        """Get comprehensive rate limiting metrics"""
        async with self._metrics_lock:
            metrics = RateLimitMetrics(
                total_requests=self._metrics.total_requests,
                allowed_requests=self._metrics.allowed_requests,
                blocked_requests=self._metrics.blocked_requests,
                burst_requests=self._metrics.burst_requests,
                unique_identifiers=len(self._memory_counters),
                average_usage_per_window=0.0,  # TODO: Calculate from historical data
                peak_usage=0  # TODO: Track peak usage
            )
            
            return metrics
    
    def get_status(self) -> str:
        """Get rate limiter status for health checks"""
        try:
            redis_status = "healthy" if (self._redis_enabled and self._redis_healthy) else "unhealthy" if self._redis_enabled else "disabled"
            memory_counters = len(self._memory_counters)
            
            return f"Redis:{redis_status} MemoryCounters:{memory_counters} Enabled:{self._redis_enabled}"
        except Exception as e:
            logger.warning(f"Rate limiter status check failed: {e}")
            return "unhealthy (status check failed)"
    
    async def cleanup(self):
        """Cleanup rate limiter resources"""
        try:
            await self._cleanup_memory_counters()
            
            if self._redis_client:
                await self._redis_client.aclose()
                logger.info("Rate limiter Redis connection closed")
                
        except Exception as e:
            logger.warning(f"Rate limiter cleanup failed: {e}")