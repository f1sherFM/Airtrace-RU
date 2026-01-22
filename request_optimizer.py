"""
Request Optimizer for AirTrace RU Backend

Implements request batching, deduplication, and smart prefetching capabilities
to optimize external API calls and improve system performance.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import asyncio
import logging
import hashlib
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import time

from schemas import AirQualityData, LocationInfo
from config import config

logger = logging.getLogger(__name__)


@dataclass
class APIRequest:
    """Represents an API request for optimization"""
    id: str
    lat: float
    lon: float
    request_type: str  # 'current' or 'forecast'
    timestamp: datetime
    future: Optional[asyncio.Future] = None
    
    def __post_init__(self):
        if self.future is None:
            self.future = asyncio.Future()
    
    def get_location_key(self, precision: int = 3) -> str:
        """Get location key for batching similar geographic regions"""
        # Round coordinates to specified precision for grouping
        rounded_lat = round(self.lat, precision)
        rounded_lon = round(self.lon, precision)
        return f"{rounded_lat},{rounded_lon}"
    
    def get_dedup_key(self) -> str:
        """Get deduplication key for identical requests"""
        # Use higher precision for exact deduplication
        return f"{self.request_type}:{self.lat:.6f},{self.lon:.6f}"


@dataclass
class BatchedRequest:
    """Represents a batched request combining multiple similar requests"""
    center_lat: float
    center_lon: float
    request_type: str
    requests: List[APIRequest]
    created_at: datetime
    
    def get_geographic_bounds(self) -> Tuple[float, float, float, float]:
        """Get geographic bounds of all requests in batch"""
        lats = [req.lat for req in self.requests]
        lons = [req.lon for req in self.requests]
        return min(lats), max(lats), min(lons), max(lons)


@dataclass
class OptimizationStats:
    """Statistics for request optimization"""
    total_requests: int = 0
    deduplicated_requests: int = 0
    batched_requests: int = 0
    prefetched_requests: int = 0
    cache_hits_from_prefetch: int = 0
    batch_efficiency: float = 0.0
    deduplication_rate: float = 0.0
    prefetch_hit_rate: float = 0.0
    
    # Detailed metrics
    batches_created: int = 0
    average_batch_size: float = 0.0
    max_batch_size: int = 0
    min_batch_size: int = 0
    batch_processing_time_ms: float = 0.0
    deduplication_window_hits: int = 0
    prefetch_accuracy: float = 0.0
    
    # Traceability metrics
    request_trace_count: int = 0
    successful_traces: int = 0
    failed_traces: int = 0
    
    def calculate_rates(self):
        """Calculate efficiency rates"""
        if self.total_requests > 0:
            self.deduplication_rate = self.deduplicated_requests / self.total_requests
            self.batch_efficiency = self.batched_requests / self.total_requests
        
        if self.prefetched_requests > 0:
            self.prefetch_hit_rate = self.cache_hits_from_prefetch / self.prefetched_requests
        
        if self.batches_created > 0:
            self.average_batch_size = self.batched_requests / self.batches_created
        
        if self.request_trace_count > 0:
            self.prefetch_accuracy = self.successful_traces / self.request_trace_count


@dataclass
class RequestTrace:
    """Traceability information for a request"""
    request_id: str
    original_lat: float
    original_lon: float
    request_type: str
    submitted_at: datetime
    optimization_applied: List[str]  # List of optimizations applied
    batch_id: Optional[str] = None
    dedup_key: Optional[str] = None
    processing_time_ms: float = 0.0
    final_status: str = "pending"  # pending, completed, failed
    error_message: Optional[str] = None
    
    def add_optimization(self, optimization: str):
        """Add an optimization to the trace"""
        self.optimization_applied.append(optimization)
    
    def set_batch_info(self, batch_id: str):
        """Set batch information"""
        self.batch_id = batch_id
        self.add_optimization("batched")
    
    def set_dedup_info(self, dedup_key: str):
        """Set deduplication information"""
        self.dedup_key = dedup_key
        self.add_optimization("deduplicated")
    
    def complete_trace(self, processing_time_ms: float, status: str = "completed", error: Optional[str] = None):
        """Complete the trace with final information"""
        self.processing_time_ms = processing_time_ms
        self.final_status = status
        self.error_message = error


@dataclass
class BatchConfig:
    """Configuration for request batching"""
    enabled: bool = True
    max_batch_size: int = 10
    batch_timeout_ms: int = 100  # 100ms window for batching
    geographic_precision: int = 2  # Precision for geographic grouping
    max_geographic_distance: float = 0.1  # Max distance in degrees for batching
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_batch_size <= 0:
            self.max_batch_size = 10
        if self.batch_timeout_ms <= 0:
            self.batch_timeout_ms = 100
        if self.geographic_precision < 1:
            self.geographic_precision = 2


@dataclass
class DeduplicationConfig:
    """Configuration for request deduplication"""
    enabled: bool = True
    window_ms: int = 100  # 100ms window for deduplication
    max_pending_requests: int = 1000
    
    def __post_init__(self):
        """Validate configuration"""
        if self.window_ms <= 0:
            self.window_ms = 100
        if self.max_pending_requests <= 0:
            self.max_pending_requests = 1000


@dataclass
class PrefetchConfig:
    """Configuration for smart prefetching"""
    enabled: bool = True
    pattern_window_hours: int = 24  # Look at last 24 hours for patterns
    min_pattern_frequency: int = 3  # Minimum frequency to trigger prefetch
    prefetch_ahead_minutes: int = 15  # How far ahead to prefetch
    max_prefetch_requests: int = 50  # Maximum concurrent prefetch requests
    respect_cache_ttl: bool = True  # Don't prefetch if cache is still valid
    
    def __post_init__(self):
        """Validate configuration"""
        if self.pattern_window_hours <= 0:
            self.pattern_window_hours = 24
        if self.min_pattern_frequency <= 0:
            self.min_pattern_frequency = 3
        if self.prefetch_ahead_minutes <= 0:
            self.prefetch_ahead_minutes = 15
        if self.max_prefetch_requests <= 0:
            self.max_prefetch_requests = 50


class UsagePattern:
    """Tracks usage patterns for predictive prefetching"""
    
    def __init__(self):
        self.request_history: deque = deque(maxlen=10000)  # Keep last 10k requests
        self.location_frequency: Dict[str, int] = defaultdict(int)
        self.time_patterns: Dict[str, List[datetime]] = defaultdict(list)
    
    def record_request(self, lat: float, lon: float, request_type: str):
        """Record a request for pattern analysis"""
        now = datetime.now(timezone.utc)
        location_key = f"{lat:.3f},{lon:.3f}"
        pattern_key = f"{request_type}:{location_key}"
        
        self.request_history.append({
            'timestamp': now,
            'location_key': location_key,
            'pattern_key': pattern_key,
            'lat': lat,
            'lon': lon,
            'request_type': request_type
        })
        
        self.location_frequency[location_key] += 1
        self.time_patterns[pattern_key].append(now)
        
        # Keep only recent patterns
        cutoff = now - timedelta(hours=24)
        self.time_patterns[pattern_key] = [
            ts for ts in self.time_patterns[pattern_key] if ts > cutoff
        ]
    
    def get_prefetch_candidates(self, config: PrefetchConfig) -> List[Tuple[float, float, str]]:
        """Get locations that should be prefetched based on patterns"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=config.pattern_window_hours)
        candidates = []
        
        for pattern_key, timestamps in self.time_patterns.items():
            # Filter recent timestamps
            recent_timestamps = [ts for ts in timestamps if ts > cutoff]
            
            if len(recent_timestamps) >= config.min_pattern_frequency:
                # Check if we should prefetch based on time patterns
                if self._should_prefetch_now(recent_timestamps, config):
                    request_type, location_key = pattern_key.split(':', 1)
                    lat_str, lon_str = location_key.split(',')
                    candidates.append((float(lat_str), float(lon_str), request_type))
        
        return candidates[:config.max_prefetch_requests]
    
    def _should_prefetch_now(self, timestamps: List[datetime], config: PrefetchConfig) -> bool:
        """Determine if we should prefetch based on time patterns"""
        if len(timestamps) < config.min_pattern_frequency:
            return False
        
        now = datetime.now(timezone.utc)
        
        # Look for patterns in request timing
        # Simple heuristic: if requests happen regularly, prefetch ahead
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # minutes
            intervals.append(interval)
        
        if not intervals:
            return False
        
        # If requests happen regularly (similar intervals), prefetch
        avg_interval = sum(intervals) / len(intervals)
        if avg_interval > 0:
            # Check if it's time to prefetch based on average interval
            last_request = timestamps[-1]
            time_since_last = (now - last_request).total_seconds() / 60
            
            # Prefetch if we're approaching the next expected request time
            return time_since_last >= (avg_interval - config.prefetch_ahead_minutes)
        
        return False


class RequestOptimizer:
    """
    Main request optimizer implementing batching, deduplication, and prefetching.
    
    Optimizes external API requests by:
    - Batching similar geographic requests
    - Deduplicating identical concurrent requests  
    - Smart prefetching based on usage patterns
    - Providing detailed optimization metrics and traceability
    """
    
    def __init__(self, 
                 batch_config: Optional[BatchConfig] = None,
                 dedup_config: Optional[DeduplicationConfig] = None,
                 prefetch_config: Optional[PrefetchConfig] = None):
        self.batch_config = batch_config or BatchConfig()
        self.dedup_config = dedup_config or DeduplicationConfig()
        self.prefetch_config = prefetch_config or PrefetchConfig()
        
        # Request tracking
        self.pending_requests: Dict[str, APIRequest] = {}
        self.batch_queues: Dict[str, List[APIRequest]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.stats = OptimizationStats()
        
        # Usage patterns for prefetching
        self.usage_patterns = UsagePattern()
        
        # Prefetch tracking
        self.active_prefetch_tasks: Set[asyncio.Task] = set()
        
        # Traceability
        self.request_traces: Dict[str, RequestTrace] = {}
        self.batch_traces: Dict[str, List[str]] = {}  # batch_id -> request_ids
        
        # Metrics tracking
        self.batch_sizes: List[int] = []
        self.batch_processing_times: List[float] = []
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("RequestOptimizer initialized with batching, deduplication, prefetching, and traceability")
    
    async def optimize_request(self, lat: float, lon: float, request_type: str) -> APIRequest:
        """
        Optimize a single request through batching and deduplication.
        
        Args:
            lat: Latitude
            lon: Longitude  
            request_type: Type of request ('current' or 'forecast')
            
        Returns:
            APIRequest: Optimized request object with future for result
        """
        start_time = time.time()
        
        async with self._lock:
            self.stats.total_requests += 1
            
            # Record request for pattern analysis
            self.usage_patterns.record_request(lat, lon, request_type)
            
            # Create request object
            request = APIRequest(
                id=self._generate_request_id(),
                lat=lat,
                lon=lon,
                request_type=request_type,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Create trace for request
            trace = RequestTrace(
                request_id=request.id,
                original_lat=lat,
                original_lon=lon,
                request_type=request_type,
                submitted_at=request.timestamp,
                optimization_applied=[]
            )
            self.request_traces[request.id] = trace
            self.stats.request_trace_count += 1
            
            # Check for deduplication first
            if self.dedup_config.enabled:
                dedup_key = request.get_dedup_key()
                if dedup_key in self.pending_requests:
                    existing_request = self.pending_requests[dedup_key]
                    # Check if within deduplication window
                    time_diff = (request.timestamp - existing_request.timestamp).total_seconds() * 1000
                    if time_diff <= self.dedup_config.window_ms:
                        self.stats.deduplicated_requests += 1
                        self.stats.deduplication_window_hits += 1
                        
                        # Update trace
                        trace.set_dedup_info(dedup_key)
                        trace.complete_trace(
                            processing_time_ms=(time.time() - start_time) * 1000,
                            status="deduplicated"
                        )
                        self.stats.successful_traces += 1
                        
                        logger.debug(f"Request deduplicated: {dedup_key}")
                        return existing_request
                
                # Add to pending requests
                self.pending_requests[dedup_key] = request
                
                # Clean up old pending requests
                await self._cleanup_pending_requests()
            
            # Add to batch queue if batching is enabled
            if self.batch_config.enabled:
                await self._add_to_batch_queue(request)
            else:
                # Process immediately if batching is disabled
                await self._process_single_request(request)
            
            return request
    
    async def batch_requests(self, requests: List[APIRequest]) -> List[BatchedRequest]:
        """
        Batch multiple requests by geographic proximity.
        
        Args:
            requests: List of API requests to batch
            
        Returns:
            List[BatchedRequest]: Batched requests grouped by location
        """
        if not requests:
            return []
        
        batch_start_time = time.time()
        
        # Group requests by type and geographic proximity
        batches: Dict[str, List[APIRequest]] = defaultdict(list)
        
        for request in requests:
            batch_key = f"{request.request_type}:{request.get_location_key(self.batch_config.geographic_precision)}"
            batches[batch_key].append(request)
        
        batched_requests = []
        for batch_key, batch_requests in batches.items():
            if len(batch_requests) > 1:
                # Calculate center point for batch
                center_lat = sum(req.lat for req in batch_requests) / len(batch_requests)
                center_lon = sum(req.lon for req in batch_requests) / len(batch_requests)
                
                # Verify all requests are within max distance
                valid_requests = []
                for req in batch_requests:
                    distance = self._calculate_distance(center_lat, center_lon, req.lat, req.lon)
                    if distance <= self.batch_config.max_geographic_distance:
                        valid_requests.append(req)
                
                if len(valid_requests) > 1:
                    batch_id = self._generate_batch_id()
                    
                    batched_request = BatchedRequest(
                        center_lat=center_lat,
                        center_lon=center_lon,
                        request_type=batch_requests[0].request_type,
                        requests=valid_requests,
                        created_at=datetime.now(timezone.utc)
                    )
                    batched_requests.append(batched_request)
                    
                    # Update statistics
                    self.stats.batched_requests += len(valid_requests)
                    self.stats.batches_created += 1
                    self.batch_sizes.append(len(valid_requests))
                    
                    # Update max/min batch sizes
                    if len(valid_requests) > self.stats.max_batch_size:
                        self.stats.max_batch_size = len(valid_requests)
                    if self.stats.min_batch_size == 0 or len(valid_requests) < self.stats.min_batch_size:
                        self.stats.min_batch_size = len(valid_requests)
                    
                    # Update traces for batched requests
                    request_ids = []
                    for req in valid_requests:
                        if req.id in self.request_traces:
                            self.request_traces[req.id].set_batch_info(batch_id)
                        request_ids.append(req.id)
                    
                    self.batch_traces[batch_id] = request_ids
        
        # Record batch processing time
        batch_processing_time = (time.time() - batch_start_time) * 1000
        self.batch_processing_times.append(batch_processing_time)
        self.stats.batch_processing_time_ms = sum(self.batch_processing_times) / len(self.batch_processing_times)
        
        return batched_requests
    
    async def start_prefetching(self):
        """Start background prefetching based on usage patterns"""
        if not self.prefetch_config.enabled:
            return
        
        # Create background task for prefetching
        task = asyncio.create_task(self._prefetch_worker())
        self.active_prefetch_tasks.add(task)
        task.add_done_callback(self.active_prefetch_tasks.discard)
        
        logger.info("Prefetching worker started")
    
    async def stop_prefetching(self):
        """Stop all prefetching tasks"""
        for task in list(self.active_prefetch_tasks):
            task.cancel()
        
        if self.active_prefetch_tasks:
            await asyncio.gather(*self.active_prefetch_tasks, return_exceptions=True)
        
        logger.info("Prefetching stopped")
    
    async def get_optimization_stats(self) -> OptimizationStats:
        """Get current optimization statistics"""
        self.stats.calculate_rates()
        return self.stats
    
    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed optimization metrics"""
        stats = await self.get_optimization_stats()
        
        return {
            "basic_stats": {
                "total_requests": stats.total_requests,
                "deduplicated_requests": stats.deduplicated_requests,
                "batched_requests": stats.batched_requests,
                "prefetched_requests": stats.prefetched_requests,
                "cache_hits_from_prefetch": stats.cache_hits_from_prefetch
            },
            "efficiency_rates": {
                "deduplication_rate": stats.deduplication_rate,
                "batch_efficiency": stats.batch_efficiency,
                "prefetch_hit_rate": stats.prefetch_hit_rate,
                "prefetch_accuracy": stats.prefetch_accuracy
            },
            "batching_metrics": {
                "batches_created": stats.batches_created,
                "average_batch_size": stats.average_batch_size,
                "max_batch_size": stats.max_batch_size,
                "min_batch_size": stats.min_batch_size,
                "batch_processing_time_ms": stats.batch_processing_time_ms,
                "batch_size_distribution": self._get_batch_size_distribution()
            },
            "deduplication_metrics": {
                "deduplication_window_hits": stats.deduplication_window_hits,
                "pending_requests_count": len(self.pending_requests)
            },
            "traceability_metrics": {
                "request_trace_count": stats.request_trace_count,
                "successful_traces": stats.successful_traces,
                "failed_traces": stats.failed_traces,
                "active_traces": len([t for t in self.request_traces.values() if t.final_status == "pending"])
            },
            "prefetch_metrics": {
                "active_prefetch_tasks": len(self.active_prefetch_tasks),
                "pattern_locations": len(self.usage_patterns.location_frequency),
                "pattern_keys": len(self.usage_patterns.time_patterns)
            }
        }
    
    async def get_request_trace(self, request_id: str) -> Optional[RequestTrace]:
        """Get trace information for a specific request"""
        return self.request_traces.get(request_id)
    
    async def get_batch_trace(self, batch_id: str) -> Optional[List[RequestTrace]]:
        """Get trace information for all requests in a batch"""
        if batch_id not in self.batch_traces:
            return None
        
        request_ids = self.batch_traces[batch_id]
        traces = []
        for request_id in request_ids:
            if request_id in self.request_traces:
                traces.append(self.request_traces[request_id])
        
        return traces
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        detailed_metrics = await self.get_detailed_metrics()
        
        # Calculate additional insights
        total_requests = detailed_metrics["basic_stats"]["total_requests"]
        if total_requests > 0:
            optimization_impact = {
                "requests_optimized": (
                    detailed_metrics["basic_stats"]["deduplicated_requests"] + 
                    detailed_metrics["basic_stats"]["batched_requests"]
                ),
                "optimization_percentage": (
                    (detailed_metrics["basic_stats"]["deduplicated_requests"] + 
                     detailed_metrics["basic_stats"]["batched_requests"]) / total_requests * 100
                ),
                "estimated_api_calls_saved": (
                    detailed_metrics["basic_stats"]["deduplicated_requests"] + 
                    max(0, detailed_metrics["basic_stats"]["batched_requests"] - detailed_metrics["batching_metrics"]["batches_created"])
                )
            }
        else:
            optimization_impact = {
                "requests_optimized": 0,
                "optimization_percentage": 0.0,
                "estimated_api_calls_saved": 0
            }
        
        return {
            "summary": optimization_impact,
            "detailed_metrics": detailed_metrics,
            "configuration": {
                "batching_enabled": self.batch_config.enabled,
                "deduplication_enabled": self.dedup_config.enabled,
                "prefetching_enabled": self.prefetch_config.enabled,
                "max_batch_size": self.batch_config.max_batch_size,
                "dedup_window_ms": self.dedup_config.window_ms,
                "prefetch_frequency_threshold": self.prefetch_config.min_pattern_frequency
            },
            "recommendations": self._generate_optimization_recommendations(detailed_metrics)
        }
    
    def _get_batch_size_distribution(self) -> Dict[str, int]:
        """Get distribution of batch sizes"""
        if not self.batch_sizes:
            return {}
        
        distribution = {}
        for size in self.batch_sizes:
            size_range = f"{size}-{size}"
            if size <= 2:
                size_range = "1-2"
            elif size <= 5:
                size_range = "3-5"
            elif size <= 10:
                size_range = "6-10"
            else:
                size_range = "10+"
            
            distribution[size_range] = distribution.get(size_range, 0) + 1
        
        return distribution
    
    def _generate_optimization_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on metrics"""
        recommendations = []
        
        # Batching recommendations
        if metrics["efficiency_rates"]["batch_efficiency"] < 0.3:
            recommendations.append(
                "Consider increasing batch timeout or reducing geographic precision to improve batching efficiency"
            )
        
        if metrics["batching_metrics"]["average_batch_size"] < 2.5:
            recommendations.append(
                "Average batch size is low. Consider adjusting geographic distance threshold"
            )
        
        # Deduplication recommendations
        if metrics["efficiency_rates"]["deduplication_rate"] > 0.5:
            recommendations.append(
                "High deduplication rate detected. Consider increasing deduplication window"
            )
        
        # Prefetching recommendations
        if metrics["efficiency_rates"]["prefetch_hit_rate"] < 0.2:
            recommendations.append(
                "Low prefetch hit rate. Consider adjusting pattern frequency threshold or prefetch timing"
            )
        
        if not recommendations:
            recommendations.append("Optimization is performing well. No immediate adjustments needed.")
        
        return recommendations
    
    def configure_batching(self, config: BatchConfig):
        """Update batching configuration"""
        self.batch_config = config
        logger.info(f"Batching configuration updated: {config}")
    
    def configure_deduplication(self, config: DeduplicationConfig):
        """Update deduplication configuration"""
        self.dedup_config = config
        logger.info(f"Deduplication configuration updated: {config}")
    
    def configure_prefetching(self, config: PrefetchConfig):
        """Update prefetching configuration"""
        self.prefetch_config = config
        logger.info(f"Prefetching configuration updated: {config}")
    
    async def _add_to_batch_queue(self, request: APIRequest):
        """Add request to appropriate batch queue"""
        batch_key = f"{request.request_type}:{request.get_location_key(self.batch_config.geographic_precision)}"
        
        self.batch_queues[batch_key].append(request)
        
        # Start batch timer if this is the first request in queue
        if len(self.batch_queues[batch_key]) == 1:
            timer_task = asyncio.create_task(
                self._batch_timer(batch_key, self.batch_config.batch_timeout_ms / 1000.0)
            )
            self.batch_timers[batch_key] = timer_task
        
        # Process immediately if batch is full
        if len(self.batch_queues[batch_key]) >= self.batch_config.max_batch_size:
            await self._process_batch_queue(batch_key)
    
    async def _batch_timer(self, batch_key: str, timeout_seconds: float):
        """Timer for batch processing"""
        try:
            await asyncio.sleep(timeout_seconds)
            async with self._lock:
                if batch_key in self.batch_queues and self.batch_queues[batch_key]:
                    await self._process_batch_queue(batch_key)
        except asyncio.CancelledError:
            pass
    
    async def _process_batch_queue(self, batch_key: str):
        """Process a batch queue"""
        if batch_key not in self.batch_queues or not self.batch_queues[batch_key]:
            return
        
        requests = self.batch_queues[batch_key].copy()
        self.batch_queues[batch_key].clear()
        
        # Cancel timer
        if batch_key in self.batch_timers:
            self.batch_timers[batch_key].cancel()
            del self.batch_timers[batch_key]
        
        # Process batch
        if len(requests) > 1:
            logger.debug(f"Processing batch of {len(requests)} requests for {batch_key}")
            # For now, process each request individually
            # In a real implementation, this would make a single API call for the batch
            for request in requests:
                asyncio.create_task(self._process_single_request(request))
        else:
            # Single request, process normally
            await self._process_single_request(requests[0])
    
    async def _process_single_request(self, request: APIRequest):
        """Process a single request (placeholder for actual API call)"""
        start_time = time.time()
        try:
            # This would normally call the actual API service
            # For now, we'll simulate processing
            await asyncio.sleep(0.01)  # Simulate API call
            
            # Set result (placeholder)
            if not request.future.done():
                request.future.set_result(f"Processed request for {request.lat}, {request.lon}")
            
            # Complete trace
            if request.id in self.request_traces:
                processing_time = (time.time() - start_time) * 1000
                self.request_traces[request.id].complete_trace(
                    processing_time_ms=processing_time,
                    status="completed"
                )
                self.stats.successful_traces += 1
            
            # Clean up from pending requests
            dedup_key = request.get_dedup_key()
            if dedup_key in self.pending_requests:
                del self.pending_requests[dedup_key]
                
        except Exception as e:
            if not request.future.done():
                request.future.set_exception(e)
            
            # Complete trace with error
            if request.id in self.request_traces:
                processing_time = (time.time() - start_time) * 1000
                self.request_traces[request.id].complete_trace(
                    processing_time_ms=processing_time,
                    status="failed",
                    error=str(e)
                )
                self.stats.failed_traces += 1
            
            logger.error(f"Error processing request {request.id}: {e}")
    
    async def _prefetch_worker(self):
        """Background worker for predictive prefetching"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                candidates = self.usage_patterns.get_prefetch_candidates(self.prefetch_config)
                
                for lat, lon, request_type in candidates:
                    # Check if we should prefetch (respect cache TTL)
                    if self.prefetch_config.respect_cache_ttl:
                        try:
                            from cache import MultiLevelCacheManager
                            cache_manager = MultiLevelCacheManager()
                            cached_data = await cache_manager.get(lat, lon)
                            if cached_data:
                                # Data is already cached, skip prefetch
                                continue
                        except Exception as e:
                            logger.debug(f"Error checking cache for prefetch: {e}")
                    
                    # Create prefetch request
                    prefetch_request = APIRequest(
                        id=self._generate_request_id(),
                        lat=lat,
                        lon=lon,
                        request_type=request_type,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    # Process prefetch request
                    asyncio.create_task(self._process_prefetch_request(prefetch_request))
                    self.stats.prefetched_requests += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prefetch worker: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _process_prefetch_request(self, request: APIRequest):
        """Process a prefetch request"""
        try:
            # Import here to avoid circular imports
            from services import AirQualityService
            from cache import MultiLevelCacheManager
            
            # Check if data is already cached
            cache_manager = MultiLevelCacheManager()
            cached_data = await cache_manager.get(request.lat, request.lon)
            
            if cached_data:
                # Data already cached, no need to prefetch
                return
            
            # Create air quality service and fetch data
            service = AirQualityService()
            
            if request.request_type == 'current':
                data = await service.get_current_air_quality(request.lat, request.lon)
                # Data will be automatically cached by the service
                logger.debug(f"Prefetched current data for {request.lat}, {request.lon}")
            elif request.request_type == 'forecast':
                data = await service.get_forecast_air_quality(request.lat, request.lon)
                # For forecast, we might want to cache the first hour's data
                if data:
                    await cache_manager.set(request.lat, request.lon, data[0].model_dump())
                logger.debug(f"Prefetched forecast data for {request.lat}, {request.lon}")
            
        except Exception as e:
            logger.error(f"Error processing prefetch request: {e}")
    
    async def _cleanup_pending_requests(self):
        """Clean up old pending requests"""
        if len(self.pending_requests) <= self.dedup_config.max_pending_requests:
            return
        
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(milliseconds=self.dedup_config.window_ms * 2)
        
        keys_to_remove = []
        for key, request in self.pending_requests.items():
            if request.timestamp < cutoff:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.pending_requests[key]
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        timestamp = int(time.time() * 1000000)  # microseconds
        return f"req_{timestamp}_{hash(timestamp) % 10000:04d}"
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch ID"""
        timestamp = int(time.time() * 1000000)  # microseconds
        return f"batch_{timestamp}_{hash(timestamp) % 10000:04d}"
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate approximate distance between two points in degrees"""
        return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)


# Global request optimizer instance
_request_optimizer: Optional[RequestOptimizer] = None


def get_request_optimizer() -> RequestOptimizer:
    """Get global request optimizer instance"""
    global _request_optimizer
    if _request_optimizer is None:
        _request_optimizer = RequestOptimizer()
    return _request_optimizer


def setup_request_optimization() -> RequestOptimizer:
    """Setup and configure request optimization"""
    global _request_optimizer
    
    from config import config
    
    # Create configuration from environment
    batch_config = BatchConfig(
        enabled=config.request_optimization.batching_enabled,
        max_batch_size=config.request_optimization.max_batch_size,
        batch_timeout_ms=config.request_optimization.batch_timeout_ms,
        geographic_precision=config.request_optimization.geographic_precision,
        max_geographic_distance=config.request_optimization.max_geographic_distance
    )
    
    dedup_config = DeduplicationConfig(
        enabled=config.request_optimization.deduplication_enabled,
        window_ms=config.request_optimization.dedup_window_ms,
        max_pending_requests=config.request_optimization.max_pending_requests
    )
    
    prefetch_config = PrefetchConfig(
        enabled=config.request_optimization.prefetching_enabled,
        pattern_window_hours=config.request_optimization.pattern_window_hours,
        min_pattern_frequency=config.request_optimization.min_pattern_frequency,
        prefetch_ahead_minutes=config.request_optimization.prefetch_ahead_minutes,
        max_prefetch_requests=config.request_optimization.max_prefetch_requests,
        respect_cache_ttl=config.request_optimization.respect_cache_ttl
    )
    
    _request_optimizer = RequestOptimizer(
        batch_config=batch_config,
        dedup_config=dedup_config,
        prefetch_config=prefetch_config
    )
    
    logger.info("Request optimization configured and ready")
    return _request_optimizer