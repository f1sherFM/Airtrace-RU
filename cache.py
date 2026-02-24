"""
Multi-level cache manager for AirTrace RU Backend

Implements L1 (in-memory), L2 (Redis), and L3 (persistent) caching layers
with privacy-safe key generation and graceful degradation.

✅ FIX #7: Optimized JSON serialization with orjson
"""

import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field

import redis.asyncio as redis
from redis.asyncio import Redis, RedisCluster
from redis.exceptions import ConnectionError, TimeoutError, RedisError

# ✅ FIX #7: Use orjson for fast JSON serialization
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    import json
    HAS_ORJSON = False
    logging.warning("orjson not available, falling back to standard json")

from config import config
from schemas import CacheEntry
from privacy_compliance_validator import validate_cache_key_privacy, validate_metrics_privacy

logger = logging.getLogger(__name__)


def json_dumps(obj: Any) -> str:
    """
    ✅ FIX #7: Fast JSON serialization using orjson if available
    
    Falls back to standard json if orjson is not installed.
    """
    if HAS_ORJSON:
        # orjson returns bytes, decode to str
        return orjson.dumps(obj, default=str).decode('utf-8')
    else:
        import json as stdlib_json
        return stdlib_json.dumps(obj, default=str)


def json_loads(data: Union[str, bytes]) -> Any:
    """
    ✅ FIX #7: Fast JSON deserialization using orjson if available
    
    Falls back to standard json if orjson is not installed.
    """
    if HAS_ORJSON:
        if isinstance(data, str):
            data = data.encode('utf-8')
        return orjson.loads(data)
    else:
        import json as stdlib_json
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return stdlib_json.loads(data)


class CacheLevel(Enum):
    """Cache level enumeration"""
    L1 = "l1"  # In-memory cache
    L2 = "l2"  # Redis cache
    L3 = "l3"  # Persistent cache (future implementation)


@dataclass
class CacheStats:
    """Cache statistics for monitoring"""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_count: int = 0
    memory_usage: int = 0
    key_count: int = 0
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    total_requests: int = 0
    
    def calculate_rates(self):
        """Calculate hit and miss rates"""
        if self.total_requests > 0:
            self.hit_rate = (self.l1_hits + self.l2_hits + self.l3_hits) / self.total_requests
            self.miss_rate = 1.0 - self.hit_rate


@dataclass
class CacheLevelStats:
    """Statistics for a specific cache level"""
    level: str
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    total_requests: int = 0
    memory_usage: int = 0
    key_count: int = 0
    evictions: int = 0
    
    def calculate_rates(self):
        """Calculate hit and miss rates for this level"""
        if self.total_requests > 0:
            self.hit_rate = self.hits / self.total_requests
            self.miss_rate = self.misses / self.total_requests


@dataclass
class CachePerformanceAnalysis:
    """Cache performance analysis and recommendations"""
    overall_efficiency: float = 0.0
    l1_efficiency: float = 0.0
    l2_efficiency: float = 0.0
    l3_efficiency: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)


@dataclass
class CacheOperation:
    """Cache operation result"""
    success: bool
    level: Optional[CacheLevel] = None
    error: Optional[str] = None
    duration: float = 0.0


class MultiLevelCacheManager:
    """
    Multi-level cache manager with L1 (memory), L2 (Redis), and L3 (persistent) layers.
    
    Implements privacy-safe key generation, graceful degradation, and comprehensive
    statistics collection for performance monitoring.
    """
    
    def __init__(self):
        # L1 Cache (In-Memory)
        self._l1_cache: Dict[str, CacheEntry] = {}
        self._l1_max_size = config.cache.l1_max_size
        self._l1_enabled = config.cache.l1_enabled
        
        # L2 Cache (Redis)
        self._redis_client: Optional[Union[Redis, RedisCluster]] = None
        self._l2_enabled = config.cache.l2_enabled and config.performance.redis_enabled
        self._l2_key_prefix = config.cache.l2_key_prefix
        
        # L3 Cache (Persistent) 
        self._l3_enabled = config.cache.l3_enabled
        self._l3_cache_dir = config.cache.l3_cache_dir
        self._l3_max_files = config.cache.l3_max_files
        self._l3_cleanup_interval = config.cache.l3_cleanup_interval
        
        # Statistics
        self._stats = CacheStats()
        self._stats_lock = asyncio.Lock()
        
        # Connection health
        self._redis_healthy = False
        self._last_health_check = 0
        self._health_check_interval = config.redis.health_check_interval
        
        # Privacy settings
        self._hash_coordinates = config.cache.hash_coordinates
        self._coordinate_precision = config.cache.coordinate_precision
        
        # Initialization flags
        self._redis_initialized = False
        self._invalidation_listener_started = False
        
        # Initialize Redis connection if enabled (deferred to first use)
        # This avoids creating async tasks during module import
    
    async def _initialize_redis(self):
        """Initialize Redis connection with cluster support"""
        try:
            if config.redis.cluster_enabled:
                cluster_kwargs = config.get_redis_cluster_kwargs()
                if cluster_kwargs.get("startup_nodes"):
                    self._redis_client = RedisCluster(**cluster_kwargs)
                    logger.info("Redis cluster client initialized")
                else:
                    logger.warning("Redis cluster enabled but no nodes configured")
                    self._l2_enabled = False
                    return
            else:
                redis_kwargs = config.get_redis_connection_kwargs()
                self._redis_client = Redis(**redis_kwargs)
                logger.info("Redis client initialized")
            
            # Test connection
            await self._check_redis_health()
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self._l2_enabled = False
            self._redis_healthy = False
    
    async def _ensure_redis_initialized(self):
        """Ensure Redis is initialized before use"""
        if self._l2_enabled and not self._redis_initialized:
            await self._initialize_redis()
            self._redis_initialized = True
            
            # Start invalidation listener if not already started
            if not self._invalidation_listener_started:
                asyncio.create_task(self._start_invalidation_listener())
                self._invalidation_listener_started = True

    async def _check_redis_health(self) -> bool:
        """Check Redis connection health"""
        if not self._redis_client or not self._l2_enabled:
            return False
        
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return self._redis_healthy
        
        try:
            # ✅ FIX #5: Add timeout to Redis ping
            await asyncio.wait_for(
                self._redis_client.ping(),
                timeout=2.0  # 2 seconds timeout
            )
            self._redis_healthy = True
            self._last_health_check = current_time
            return True
        except asyncio.TimeoutError:
            logger.warning("Redis health check timed out")
            self._redis_healthy = False
            self._last_health_check = current_time
            return False
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self._redis_healthy = False
            self._last_health_check = current_time
            return False
    
    async def _start_invalidation_listener(self):
        """Start listening for cache invalidation messages from other instances"""
        if not self._l2_enabled:
            return
        
        try:
            # Wait for Redis to be initialized
            await asyncio.sleep(1)
            
            if not self._redis_client or not await self._check_redis_health():
                logger.warning("Cannot start invalidation listener: Redis not available")
                return
            
            # Subscribe to invalidation channel
            pubsub = self._redis_client.pubsub()
            await pubsub.subscribe(f"{self._l2_key_prefix}:invalidation")
            
            logger.info("Started cache invalidation listener")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        invalidation_data = json_loads(message['data'])
                        
                        # Don't process our own invalidation messages
                        if invalidation_data.get('instance_id') == os.getpid():
                            continue
                        
                        key = invalidation_data.get('key')
                        if key:
                            # Invalidate the key in local L1 cache
                            if key in self._l1_cache:
                                del self._l1_cache[key]
                                logger.debug(f"Invalidated L1 cache for key: {key}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process invalidation message: {e}")
                        
        except Exception as e:
            logger.error(f"Cache invalidation listener failed: {e}")
    
    async def _promote_to_higher_levels(self, key: str, value: Any, found_level: CacheLevel):
        """
        Promote cache data to higher levels when found in lower levels.
        
        This improves cache performance by moving frequently accessed data
        to faster cache levels.
        """
        try:
            if found_level == CacheLevel.L3:
                # Promote to L2 and L1
                await self._set_to_l2(key, value, config.cache.l2_default_ttl)
                await self._set_to_l1(key, value, config.cache.l1_default_ttl)
                logger.debug(f"Promoted key {key} from L3 to L2 and L1")
                
            elif found_level == CacheLevel.L2:
                # Promote to L1
                await self._set_to_l1(key, value, config.cache.l1_default_ttl)
                logger.debug(f"Promoted key {key} from L2 to L1")
                
        except Exception as e:
            logger.warning(f"Failed to promote cache data for key {key}: {e}")
        """Check Redis connection health"""
        if not self._redis_client or not self._l2_enabled:
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
            logger.warning(f"Redis health check failed: {e}")
            self._redis_healthy = False
            self._last_health_check = current_time
            return False
    
    def _generate_key(self, lat: float, lon: float) -> str:
        """
        Generate privacy-safe cache key.
        
        Rounds coordinates to specified precision and optionally hashes them
        to prevent storing actual coordinates.
        """
        # Round coordinates for grouping nearby requests
        rounded_lat = round(lat, self._coordinate_precision)
        rounded_lon = round(lon, self._coordinate_precision)
        
        key_string = f"{rounded_lat}:{rounded_lon}"
        
        if self._hash_coordinates:
            # Hash for privacy protection
            cache_key = hashlib.md5(key_string.encode()).hexdigest()
        else:
            # Use rounded coordinates directly (less privacy but more readable)
            cache_key = key_string.replace(".", "_").replace("-", "n")
        
        # Validate cache key privacy compliance
        if not validate_cache_key_privacy(cache_key, "MultiLevelCacheManager._generate_key"):
            logger.warning(f"Cache key privacy validation failed for key: {cache_key[:20]}...")
        
        return cache_key
    
    def _generate_l2_key(self, base_key: str) -> str:
        """Generate L2 (Redis) cache key with prefix"""
        return f"{self._l2_key_prefix}:{base_key}"
    
    def _get_l3_cache_file_path(self, key: str) -> str:
        """Generate L3 cache file path"""
        # Create subdirectories based on key hash to avoid too many files in one directory
        key_hash = hashlib.md5(key.encode()).hexdigest()
        subdir1 = key_hash[:2]
        subdir2 = key_hash[2:4]
        filename = f"{key_hash}.json"
        
        return os.path.join(self._l3_cache_dir, subdir1, subdir2, filename)
    
    async def _update_stats(self, level: CacheLevel, hit: bool):
        """Update cache statistics"""
        async with self._stats_lock:
            self._stats.total_requests += 1
            
            if level == CacheLevel.L1:
                if hit:
                    self._stats.l1_hits += 1
                else:
                    self._stats.l1_misses += 1
            elif level == CacheLevel.L2:
                if hit:
                    self._stats.l2_hits += 1
                else:
                    self._stats.l2_misses += 1
            elif level == CacheLevel.L3:
                if hit:
                    self._stats.l3_hits += 1
                else:
                    self._stats.l3_misses += 1
            
            self._stats.calculate_rates()
    
    async def _get_from_l1(self, key: str) -> Optional[Any]:
        """Get data from L1 (in-memory) cache"""
        if not self._l1_enabled:
            return None
        
        try:
            if key in self._l1_cache:
                entry = self._l1_cache[key]
                if not entry.is_expired():
                    await self._update_stats(CacheLevel.L1, True)
                    return entry.data
                else:
                    # Remove expired entry
                    del self._l1_cache[key]
                    await self._update_stats(CacheLevel.L1, False)
            else:
                await self._update_stats(CacheLevel.L1, False)
            
            return None
        except Exception as e:
            logger.warning(f"L1 cache get failed: {e}")
            await self._update_stats(CacheLevel.L1, False)
            return None
    
    async def _set_to_l1(self, key: str, value: Any, ttl: int) -> bool:
        """Set data to L1 (in-memory) cache"""
        if not self._l1_enabled:
            return False
        
        try:
            # Implement LRU eviction if cache is full
            if len(self._l1_cache) >= self._l1_max_size:
                await self._evict_l1_lru()
            
            entry = CacheEntry(
                data=value,
                timestamp=datetime.now(timezone.utc),
                ttl_seconds=ttl
            )
            self._l1_cache[key] = entry
            return True
        except Exception as e:
            logger.warning(f"L1 cache set failed: {e}")
            return False
    
    async def _evict_l1_lru(self):
        """Evict least recently used entries from L1 cache"""
        try:
            # ✅ FIX #2: Use lock to prevent race condition
            async with self._stats_lock:
                if not self._l1_cache:
                    return
                
                # Find oldest entry (simple LRU implementation)
                oldest_key = min(self._l1_cache.keys(), 
                               key=lambda k: self._l1_cache[k].timestamp)
                del self._l1_cache[oldest_key]
                self._stats.eviction_count += 1
                
        except Exception as e:
            logger.warning(f"L1 cache eviction failed: {e}")
    
    async def _get_from_l2(self, key: str) -> Optional[Any]:
        """Get data from L2 (Redis) cache"""
        if not self._l2_enabled:
            await self._update_stats(CacheLevel.L2, False)
            return None
        
        # Ensure Redis is initialized
        await self._ensure_redis_initialized()
        
        if not await self._check_redis_health():
            await self._update_stats(CacheLevel.L2, False)
            return None
        
        try:
            l2_key = self._generate_l2_key(key)
            
            # ✅ FIX #5: Add timeout to Redis operations
            data = await asyncio.wait_for(
                self._redis_client.get(l2_key),
                timeout=2.0  # 2 seconds timeout
            )
            
            if data:
                # Deserialize and check expiration
                cache_data = json_loads(data)
                entry = CacheEntry(
                    data=cache_data["data"],
                    timestamp=datetime.fromisoformat(cache_data["timestamp"]),
                    ttl_seconds=cache_data["ttl_seconds"]
                )
                
                if not entry.is_expired():
                    await self._update_stats(CacheLevel.L2, True)
                    # Also cache in L1 for faster access
                    await self._set_to_l1(key, entry.data, entry.ttl_seconds)
                    return entry.data
                else:
                    # Remove expired entry
                    await asyncio.wait_for(
                        self._redis_client.delete(l2_key),
                        timeout=1.0
                    )
                    await self._update_stats(CacheLevel.L2, False)
            else:
                await self._update_stats(CacheLevel.L2, False)
            
            return None
        except asyncio.TimeoutError:
            logger.warning("Redis get operation timed out")
            await self._update_stats(CacheLevel.L2, False)
            return None
        except Exception as e:
            logger.warning(f"L2 cache get failed: {e}")
            await self._update_stats(CacheLevel.L2, False)
            return None
    
    async def _set_to_l2(self, key: str, value: Any, ttl: int) -> bool:
        """Set data to L2 (Redis) cache"""
        if not self._l2_enabled:
            return False
        
        # Ensure Redis is initialized
        await self._ensure_redis_initialized()
        
        if not await self._check_redis_health():
            return False
        
        try:
            l2_key = self._generate_l2_key(key)
            entry_data = {
                "data": value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ttl_seconds": ttl
            }
            
            # ✅ FIX #5: Add timeout to Redis operations
            # Set with Redis TTL for automatic expiration
            await asyncio.wait_for(
                self._redis_client.setex(
                    l2_key, 
                    ttl, 
                    json_dumps(entry_data, default=str)
                ),
                timeout=2.0  # 2 seconds timeout
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Redis set operation timed out")
            return False
        except Exception as e:
            logger.warning(f"L2 cache set failed: {e}")
            return False
    
    async def _get_from_l3(self, key: str) -> Optional[Any]:
        """Get data from L3 (persistent) cache"""
        if not self._l3_enabled:
            await self._update_stats(CacheLevel.L3, False)
            return None
        
        try:
            cache_file = self._get_l3_cache_file_path(key)
            
            if not os.path.exists(cache_file):
                await self._update_stats(CacheLevel.L3, False)
                return None
            
            # Read and deserialize cache entry
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            entry = CacheEntry(
                data=cache_data["data"],
                timestamp=datetime.fromisoformat(cache_data["timestamp"]),
                ttl_seconds=cache_data["ttl_seconds"]
            )
            
            if not entry.is_expired():
                await self._update_stats(CacheLevel.L3, True)
                return entry.data
            else:
                # Remove expired file
                try:
                    os.remove(cache_file)
                except OSError:
                    pass
                await self._update_stats(CacheLevel.L3, False)
                return None
                
        except Exception as e:
            logger.warning(f"L3 cache get failed for key {key}: {e}")
            await self._update_stats(CacheLevel.L3, False)
            return None
    
    async def _set_to_l3(self, key: str, value: Any, ttl: int) -> bool:
        """Set data to L3 (persistent) cache"""
        if not self._l3_enabled:
            return False
        
        try:
            cache_file = self._get_l3_cache_file_path(key)
            
            # Ensure cache directory exists
            cache_dir = os.path.dirname(cache_file)
            os.makedirs(cache_dir, exist_ok=True)
            
            entry_data = {
                "data": value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ttl_seconds": ttl
            }
            
            # Write to temporary file first, then rename for atomicity
            temp_file = cache_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(entry_data, f, default=str, separators=(',', ':'))
            
            # Atomic rename
            os.rename(temp_file, cache_file)
            return True
            
        except Exception as e:
            logger.warning(f"L3 cache set failed for key {key}: {e}")
            return False
    
    async def get(self, lat: float, lon: float, cache_levels: Optional[List[CacheLevel]] = None) -> Optional[Any]:
        """
        Get data from multi-level cache.
        
        Checks L1 → L2 → L3 in order, returning first hit.
        Promotes data to higher levels on cache hits for improved performance.
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]
        
        key = self._generate_key(lat, lon)
        
        # Check each cache level in order
        for level in cache_levels:
            try:
                data = None
                if level == CacheLevel.L1:
                    data = await self._get_from_l1(key)
                elif level == CacheLevel.L2:
                    data = await self._get_from_l2(key)
                elif level == CacheLevel.L3:
                    data = await self._get_from_l3(key)
                else:
                    continue
                
                if data is not None:
                    logger.debug(f"Cache hit at level {level.value}")
                    
                    # Promote data to higher cache levels for better performance
                    if level != CacheLevel.L1:
                        await self._promote_to_higher_levels(key, data, level)
                    
                    return data
                    
            except Exception as e:
                logger.warning(f"Cache get failed at level {level.value}: {e}")
                continue
        
        logger.debug("Cache miss at all levels")
        return None
    
    async def set(self, lat: float, lon: float, value: Any, ttl: Optional[int] = None, 
                  levels: Optional[List[CacheLevel]] = None) -> bool:
        """
        Set data to multi-level cache.
        
        Stores data in specified cache levels with appropriate TTL values.
        """
        if levels is None:
            levels = [CacheLevel.L1, CacheLevel.L2]
        
        if ttl is None:
            ttl = config.cache.l1_default_ttl
        
        key = self._generate_key(lat, lon)
        success = False
        
        # Set data in each specified level
        for level in levels:
            try:
                level_ttl = ttl
                if level == CacheLevel.L2:
                    level_ttl = config.cache.l2_default_ttl
                elif level == CacheLevel.L3:
                    level_ttl = config.cache.l3_default_ttl
                
                if level == CacheLevel.L1:
                    result = await self._set_to_l1(key, value, level_ttl)
                elif level == CacheLevel.L2:
                    result = await self._set_to_l2(key, value, level_ttl)
                elif level == CacheLevel.L3:
                    result = await self._set_to_l3(key, value, level_ttl)
                else:
                    continue
                
                if result:
                    success = True
                    logger.debug(f"Cache set successful at level {level.value}")
                    
            except Exception as e:
                logger.warning(f"Cache set failed at level {level.value}: {e}")
                continue
        
        return success
    
    async def invalidate(self, pattern: str, levels: Optional[List[CacheLevel]] = None) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Returns number of invalidated entries.
        """
        if levels is None:
            levels = [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]
        
        total_invalidated = 0
        
        for level in levels:
            try:
                if level == CacheLevel.L1:
                    # Simple pattern matching for L1
                    keys_to_remove = [k for k in self._l1_cache.keys() if pattern in k]
                    for key in keys_to_remove:
                        del self._l1_cache[key]
                    total_invalidated += len(keys_to_remove)
                    
                elif level == CacheLevel.L2 and self._l2_enabled and await self._check_redis_health():
                    # Redis pattern matching
                    l2_pattern = self._generate_l2_key(pattern)
                    keys = await self._redis_client.keys(l2_pattern)
                    if keys:
                        deleted = await self._redis_client.delete(*keys)
                        total_invalidated += deleted
                        
                elif level == CacheLevel.L3:
                    # L3 pattern matching and cleanup
                    if self._l3_enabled:
                        l3_invalidated = await self._invalidate_l3_pattern(pattern)
                        total_invalidated += l3_invalidated
                    
            except Exception as e:
                logger.warning(f"Cache invalidation failed at level {level.value}: {e}")
                continue
        
        logger.info(f"Invalidated {total_invalidated} cache entries")
        return total_invalidated

    async def invalidate_by_coordinates(
        self,
        lat: float,
        lon: float,
        levels: Optional[List[CacheLevel]] = None,
    ) -> int:
        """
        Invalidate cache entries for a coordinate pair using the same keyspace as get/set.

        This avoids keyspace mismatches when coordinate keys are rounded/hashed.
        """
        key = self._generate_key(lat, lon)
        return await self.invalidate(key, levels=levels)
    
    async def clear_expired(self) -> int:
        """Clear expired entries from all cache levels"""
        total_cleared = 0
        
        # Clear L1 expired entries
        try:
            expired_keys = [
                key for key, entry in self._l1_cache.items() 
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._l1_cache[key]
            total_cleared += len(expired_keys)
        except Exception as e:
            logger.warning(f"L1 cache cleanup failed: {e}")
        
        # L2 (Redis) handles expiration automatically with TTL
        
        # Clear L3 expired entries
        if self._l3_enabled:
            try:
                l3_cleared = await self._cleanup_l3_expired()
                total_cleared += l3_cleared
            except Exception as e:
                logger.warning(f"L3 cache cleanup failed: {e}")
        
        if total_cleared > 0:
            logger.info(f"Cleared {total_cleared} expired cache entries")
        
        return total_cleared
    
    async def ensure_cache_coherence(self, key: str, value: Any, ttl: int) -> bool:
        """
        Ensure cache coherence across distributed instances.
        
        When data is updated, this method invalidates the key across all cache levels
        and instances to maintain consistency.
        """
        try:
            # First, invalidate the key in all local cache levels
            await self.invalidate(key, [CacheLevel.L1])
            
            # If Redis is available, use it to coordinate cache invalidation across instances
            if self._l2_enabled and await self._check_redis_health():
                # Publish cache invalidation message to other instances
                invalidation_message = {
                    "key": key,
                    "action": "invalidate",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "instance_id": os.getpid()  # Use process ID as instance identifier
                }
                
                try:
                    # Publish to Redis pub/sub for distributed cache invalidation
                    await self._redis_client.publish(
                        f"{self._l2_key_prefix}:invalidation",
                        json_dumps(invalidation_message)
                    )
                    logger.debug(f"Published cache invalidation for key: {key}")
                except Exception as e:
                    logger.warning(f"Failed to publish cache invalidation: {e}")
            
            # Set the new value in appropriate cache levels
            return await self.set_with_key(key, value, ttl)
            
        except Exception as e:
            logger.error(f"Cache coherence operation failed for key {key}: {e}")
            return False
    
    async def set_with_key(self, key: str, value: Any, ttl: Optional[int] = None, 
                          levels: Optional[List[CacheLevel]] = None) -> bool:
        """
        Set data to multi-level cache using a pre-generated key.
        
        This method is used internally for cache coherence operations.
        """
        if levels is None:
            levels = [CacheLevel.L1, CacheLevel.L2]
        
        if ttl is None:
            ttl = config.cache.l1_default_ttl
        
        success = False
        
        # Set data in each specified level
        for level in levels:
            try:
                level_ttl = ttl
                if level == CacheLevel.L2:
                    level_ttl = config.cache.l2_default_ttl
                elif level == CacheLevel.L3:
                    level_ttl = config.cache.l3_default_ttl
                
                if level == CacheLevel.L1:
                    result = await self._set_to_l1(key, value, level_ttl)
                elif level == CacheLevel.L2:
                    result = await self._set_to_l2(key, value, level_ttl)
                elif level == CacheLevel.L3:
                    result = await self._set_to_l3(key, value, level_ttl)
                else:
                    continue
                
                if result:
                    success = True
                    logger.debug(f"Cache set successful at level {level.value}")
                    
            except Exception as e:
                logger.warning(f"Cache set failed at level {level.value}: {e}")
                continue
        
        return success
    
    async def get_stats(self) -> CacheStats:
        """Get comprehensive cache statistics"""
        async with self._stats_lock:
            stats = CacheStats(
                hit_rate=self._stats.hit_rate,
                miss_rate=self._stats.miss_rate,
                eviction_count=self._stats.eviction_count,
                key_count=len(self._l1_cache),
                l1_hits=self._stats.l1_hits,
                l1_misses=self._stats.l1_misses,
                l2_hits=self._stats.l2_hits,
                l2_misses=self._stats.l2_misses,
                l3_hits=self._stats.l3_hits,
                l3_misses=self._stats.l3_misses,
                total_requests=self._stats.total_requests
            )
            
            # Calculate memory usage (approximate)
            try:
                import sys
                stats.memory_usage = sum(
                    sys.getsizeof(entry.data) for entry in self._l1_cache.values()
                )
            except Exception:
                stats.memory_usage = 0
            
            # Validate stats privacy compliance
            stats_dict = stats.__dict__
            if not validate_metrics_privacy(stats_dict, "CacheManager.get_stats"):
                logger.warning("Cache statistics privacy validation failed")
            
            return stats
    
    async def get_level_stats(self, level: CacheLevel) -> CacheLevelStats:
        """Get detailed statistics for a specific cache level"""
        async with self._stats_lock:
            if level == CacheLevel.L1:
                stats = CacheLevelStats(
                    level="L1",
                    hits=self._stats.l1_hits,
                    misses=self._stats.l1_misses,
                    total_requests=self._stats.l1_hits + self._stats.l1_misses,
                    key_count=len(self._l1_cache),
                    evictions=self._stats.eviction_count
                )
                
                # Calculate memory usage for L1
                try:
                    import sys
                    stats.memory_usage = sum(
                        sys.getsizeof(entry.data) for entry in self._l1_cache.values()
                    )
                except Exception:
                    stats.memory_usage = 0
                    
            elif level == CacheLevel.L2:
                stats = CacheLevelStats(
                    level="L2",
                    hits=self._stats.l2_hits,
                    misses=self._stats.l2_misses,
                    total_requests=self._stats.l2_hits + self._stats.l2_misses,
                    key_count=0,  # Redis key count would require separate query
                    evictions=0   # Redis handles evictions internally
                )
                
            elif level == CacheLevel.L3:
                stats = CacheLevelStats(
                    level="L3",
                    hits=self._stats.l3_hits,
                    misses=self._stats.l3_misses,
                    total_requests=self._stats.l3_hits + self._stats.l3_misses,
                    key_count=await self._get_l3_key_count(),
                    evictions=0  # L3 evictions are handled by cleanup
                )
            else:
                raise ValueError(f"Invalid cache level: {level}")
            
            stats.calculate_rates()
            return stats
    
    async def analyze_cache_performance(self) -> CachePerformanceAnalysis:
        """
        Analyze cache performance and provide optimization recommendations.
        
        Returns detailed analysis with efficiency metrics and actionable recommendations.
        """
        async with self._stats_lock:
            analysis = CachePerformanceAnalysis()
            
            # Calculate overall efficiency
            total_hits = self._stats.l1_hits + self._stats.l2_hits + self._stats.l3_hits
            if self._stats.total_requests > 0:
                analysis.overall_efficiency = total_hits / self._stats.total_requests
            
            # Calculate per-level efficiency
            l1_total = self._stats.l1_hits + self._stats.l1_misses
            if l1_total > 0:
                analysis.l1_efficiency = self._stats.l1_hits / l1_total
            
            l2_total = self._stats.l2_hits + self._stats.l2_misses
            if l2_total > 0:
                analysis.l2_efficiency = self._stats.l2_hits / l2_total
            
            l3_total = self._stats.l3_hits + self._stats.l3_misses
            if l3_total > 0:
                analysis.l3_efficiency = self._stats.l3_hits / l3_total
            
            # Generate recommendations based on performance patterns
            self._generate_performance_recommendations(analysis)
            
            return analysis
    
    def _generate_performance_recommendations(self, analysis: CachePerformanceAnalysis):
        """Generate performance recommendations based on cache statistics"""
        
        # Overall efficiency recommendations
        if analysis.overall_efficiency < 0.5:
            analysis.recommendations.append(
                "Overall cache hit rate is low (<50%). Consider increasing cache TTL values or cache size."
            )
            analysis.bottlenecks.append("Low overall cache efficiency")
        
        # L1 cache recommendations
        if analysis.l1_efficiency < 0.7 and self._stats.l1_hits + self._stats.l1_misses > 100:
            analysis.recommendations.append(
                "L1 cache hit rate is low. Consider increasing L1 cache size or implementing better cache warming."
            )
            analysis.bottlenecks.append("L1 cache inefficiency")
        
        if self._stats.eviction_count > self._stats.l1_hits * 0.1:
            analysis.recommendations.append(
                "High L1 cache eviction rate detected. Consider increasing L1 cache size."
            )
            analysis.bottlenecks.append("Frequent L1 cache evictions")
        
        # L2 cache recommendations
        if analysis.l2_efficiency < 0.8 and self._stats.l2_hits + self._stats.l2_misses > 50:
            analysis.recommendations.append(
                "L2 cache hit rate could be improved. Consider increasing L2 TTL or implementing cache pre-warming."
            )
        
        # Cache hierarchy optimization
        if self._stats.l2_hits > self._stats.l1_hits * 2:
            analysis.optimization_opportunities.append(
                "L2 cache is more effective than L1. Consider increasing L1 cache size or improving L1 cache strategy."
            )
        
        if self._stats.l1_hits < self._stats.total_requests * 0.3:
            analysis.optimization_opportunities.append(
                "L1 cache is underutilized. Consider implementing more aggressive cache warming or increasing L1 TTL."
            )
        
        # Redis health recommendations
        if not self._redis_healthy and self._l2_enabled:
            analysis.bottlenecks.append("Redis connection issues affecting L2 cache")
            analysis.recommendations.append(
                "Redis connection is unhealthy. Check Redis server status and network connectivity."
            )
        
        # Memory usage recommendations
        try:
            if len(self._l1_cache) > self._l1_max_size * 0.9:
                analysis.recommendations.append(
                    "L1 cache is near capacity. Consider increasing max size or implementing more aggressive eviction."
                )
        except Exception:
            pass
        
        # Performance pattern analysis
        if self._stats.total_requests > 1000:
            hit_distribution = {
                'L1': self._stats.l1_hits / max(1, total_hits) if (total_hits := self._stats.l1_hits + self._stats.l2_hits + self._stats.l3_hits) > 0 else 0,
                'L2': self._stats.l2_hits / max(1, total_hits) if total_hits > 0 else 0,
                'L3': self._stats.l3_hits / max(1, total_hits) if total_hits > 0 else 0
            }
            
            if hit_distribution['L1'] < 0.6:
                analysis.optimization_opportunities.append(
                    f"Only {hit_distribution['L1']:.1%} of hits come from L1 cache. Consider optimizing L1 cache strategy."
                )
    
    async def get_cache_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive cache health report.
        
        Returns a detailed report including statistics, performance analysis,
        and system health indicators.
        """
        stats = await self.get_stats()
        analysis = await self.analyze_cache_performance()
        
        # Get individual level statistics
        l1_stats = await self.get_level_stats(CacheLevel.L1)
        l2_stats = await self.get_level_stats(CacheLevel.L2)
        l3_stats = await self.get_level_stats(CacheLevel.L3)
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_stats": {
                "hit_rate": stats.hit_rate,
                "miss_rate": stats.miss_rate,
                "total_requests": stats.total_requests,
                "memory_usage_bytes": stats.memory_usage,
                "total_keys": stats.key_count
            },
            "level_stats": {
                "L1": {
                    "hit_rate": l1_stats.hit_rate,
                    "hits": l1_stats.hits,
                    "misses": l1_stats.misses,
                    "memory_usage": l1_stats.memory_usage,
                    "key_count": l1_stats.key_count,
                    "evictions": l1_stats.evictions,
                    "enabled": self._l1_enabled
                },
                "L2": {
                    "hit_rate": l2_stats.hit_rate,
                    "hits": l2_stats.hits,
                    "misses": l2_stats.misses,
                    "enabled": self._l2_enabled,
                    "healthy": self._redis_healthy
                },
                "L3": {
                    "hit_rate": l3_stats.hit_rate,
                    "hits": l3_stats.hits,
                    "misses": l3_stats.misses,
                    "enabled": self._l3_enabled
                }
            },
            "performance_analysis": {
                "overall_efficiency": analysis.overall_efficiency,
                "l1_efficiency": analysis.l1_efficiency,
                "l2_efficiency": analysis.l2_efficiency,
                "l3_efficiency": analysis.l3_efficiency,
                "recommendations": analysis.recommendations,
                "bottlenecks": analysis.bottlenecks,
                "optimization_opportunities": analysis.optimization_opportunities
            },
            "system_health": {
                "redis_healthy": self._redis_healthy,
                "l1_cache_utilization": len(self._l1_cache) / max(1, self._l1_max_size),
                "cache_levels_active": sum([
                    self._l1_enabled,
                    self._l2_enabled and self._redis_healthy,
                    self._l3_enabled
                ])
            }
        }
        
        return report
    
    async def warm_cache(self, keys: List[str]) -> None:
        """
        Warm cache with specified keys based on usage patterns.
        
        Implements intelligent cache warming by pre-loading frequently accessed data
        into higher cache levels for improved performance.
        """
        if not config.cache.warming_enabled:
            return
        
        if not keys:
            return
        
        logger.info(f"Starting cache warming for {len(keys)} keys")
        warmed_count = 0
        
        # Process keys in batches to avoid overwhelming the system
        batch_size = config.cache.warming_batch_size
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]
            
            for key in batch:
                try:
                    # Check if data exists in any cache level
                    data = await self._get_from_l3(key)
                    if data is None:
                        data = await self._get_from_l2(key)
                    
                    if data is not None:
                        # Promote data to higher cache levels
                        await self._set_to_l1(key, data, config.cache.l1_default_ttl)
                        if not await self._get_from_l2(key):
                            await self._set_to_l2(key, data, config.cache.l2_default_ttl)
                        warmed_count += 1
                        logger.debug(f"Warmed cache for key: {key}")
                    
                except Exception as e:
                    logger.warning(f"Failed to warm cache for key {key}: {e}")
                    continue
            
            # Small delay between batches to prevent overwhelming
            if i + batch_size < len(keys):
                await asyncio.sleep(0.1)
        
        logger.info(f"Cache warming completed: {warmed_count}/{len(keys)} keys warmed")
    
    async def _start_invalidation_listener(self):
        """Start listening for cache invalidation messages from other instances"""
        if not self._l2_enabled:
            return
        
        try:
            # Wait for Redis to be initialized
            await asyncio.sleep(1)
            
            if not self._redis_client or not await self._check_redis_health():
                logger.warning("Cannot start invalidation listener: Redis not available")
                return
            
            # Subscribe to invalidation channel
            pubsub = self._redis_client.pubsub()
            await pubsub.subscribe(f"{self._l2_key_prefix}:invalidation")
            
            logger.info("Started cache invalidation listener")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        invalidation_data = json_loads(message['data'])
                        
                        # Don't process our own invalidation messages
                        if invalidation_data.get('instance_id') == os.getpid():
                            continue
                        
                        key = invalidation_data.get('key')
                        if key:
                            # Invalidate the key in local L1 cache
                            if key in self._l1_cache:
                                del self._l1_cache[key]
                                logger.debug(f"Invalidated L1 cache for key: {key}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process invalidation message: {e}")
                        
        except Exception as e:
            logger.error(f"Cache invalidation listener failed: {e}")
    
    async def _promote_to_higher_levels(self, key: str, value: Any, found_level: CacheLevel):
        """
        Promote cache data to higher levels when found in lower levels.
        
        This improves cache performance by moving frequently accessed data
        to faster cache levels.
        """
        try:
            if found_level == CacheLevel.L3:
                # Promote to L2 and L1
                await self._set_to_l2(key, value, config.cache.l2_default_ttl)
                await self._set_to_l1(key, value, config.cache.l1_default_ttl)
                logger.debug(f"Promoted key {key} from L3 to L2 and L1")
                
            elif found_level == CacheLevel.L2:
                # Promote to L1
                await self._set_to_l1(key, value, config.cache.l1_default_ttl)
                logger.debug(f"Promoted key {key} from L2 to L1")
                
        except Exception as e:
            logger.warning(f"Failed to promote cache data for key {key}: {e}")
    
    def get_status(self) -> str:
        """Get cache status for health checks"""
        try:
            l1_status = "enabled" if self._l1_enabled else "disabled"
            l2_status = "healthy" if (self._l2_enabled and self._redis_healthy) else "unhealthy" if self._l2_enabled else "disabled"
            l3_status = "enabled" if self._l3_enabled else "disabled"
            
            total_entries = len(self._l1_cache)
            
            return f"L1:{l1_status} L2:{l2_status} L3:{l3_status} ({total_entries} entries)"
        except Exception as e:
            logger.warning(f"Cache status check failed: {e}")
            return "unhealthy (status check failed)"
    
    async def cleanup(self):
        """Cleanup cache resources"""
        try:
            await self.clear_expired()
            
            if self._redis_client:
                await self._redis_client.aclose()
                logger.info("Redis connection closed")
                
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    async def _cleanup_l3_expired(self) -> int:
        """Clean up expired L3 cache files"""
        if not self._l3_enabled or not os.path.exists(self._l3_cache_dir):
            return 0
        
        cleaned_count = 0
        current_time = time.time()
        
        try:
            # Walk through all cache files
            for root, dirs, files in os.walk(self._l3_cache_dir):
                for file in files:
                    if not file.endswith('.json'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Check if file is expired
                        with open(file_path, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                        
                        entry = CacheEntry(
                            data=cache_data["data"],
                            timestamp=datetime.fromisoformat(cache_data["timestamp"]),
                            ttl_seconds=cache_data["ttl_seconds"]
                        )
                        
                        if entry.is_expired():
                            os.remove(file_path)
                            cleaned_count += 1
                            
                    except (json.JSONDecodeError, KeyError, OSError) as e:
                        # Remove corrupted or inaccessible files
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                            logger.debug(f"Removed corrupted L3 cache file: {file_path}")
                        except OSError:
                            pass
            
            # Also enforce max files limit
            if self._l3_max_files > 0:
                cleaned_count += await self._enforce_l3_file_limit()
                
        except Exception as e:
            logger.error(f"L3 cache cleanup failed: {e}")
        
        return cleaned_count
    
    async def _enforce_l3_file_limit(self) -> int:
        """Enforce maximum file limit for L3 cache using LRU eviction"""
        if not self._l3_enabled or not os.path.exists(self._l3_cache_dir):
            return 0
        
        try:
            # Collect all cache files with their access times
            cache_files = []
            
            for root, dirs, files in os.walk(self._l3_cache_dir):
                for file in files:
                    if not file.endswith('.json'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        stat = os.stat(file_path)
                        cache_files.append((file_path, stat.st_atime))
                    except OSError:
                        continue
            
            # If under limit, no cleanup needed
            if len(cache_files) <= self._l3_max_files:
                return 0
            
            # Sort by access time (oldest first) and remove excess files
            cache_files.sort(key=lambda x: x[1])
            files_to_remove = len(cache_files) - self._l3_max_files
            removed_count = 0
            
            for file_path, _ in cache_files[:files_to_remove]:
                try:
                    os.remove(file_path)
                    removed_count += 1
                except OSError:
                    continue
            
            return removed_count
            
        except Exception as e:
            logger.error(f"L3 file limit enforcement failed: {e}")
            return 0
    
    async def _invalidate_l3_pattern(self, pattern: str) -> int:
        """Invalidate L3 cache files matching pattern"""
        if not self._l3_enabled or not os.path.exists(self._l3_cache_dir):
            return 0
        
        invalidated_count = 0
        
        try:
            # Walk through all cache files
            for root, dirs, files in os.walk(self._l3_cache_dir):
                for file in files:
                    if not file.endswith('.json'):
                        continue
                    
                    # Simple pattern matching on filename
                    if pattern in file:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            invalidated_count += 1
                        except OSError:
                            continue
                            
        except Exception as e:
            logger.error(f"L3 pattern invalidation failed: {e}")
        
        return invalidated_count
    
    async def _get_l3_key_count(self) -> int:
        """Get count of L3 cache files"""
        if not self._l3_enabled or not os.path.exists(self._l3_cache_dir):
            return 0
        
        try:
            count = 0
            for root, dirs, files in os.walk(self._l3_cache_dir):
                count += sum(1 for file in files if file.endswith('.json'))
            return count
        except Exception as e:
            logger.warning(f"Failed to count L3 cache files: {e}")
            return 0


# Backward compatibility alias
CacheManager = MultiLevelCacheManager
