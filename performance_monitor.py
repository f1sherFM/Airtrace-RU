"""
Performance monitoring system for AirTrace RU Backend

Implements comprehensive performance monitoring with metrics collection
for request latency, cache operations, and external API calls.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Deque
from threading import Lock
import statistics

from privacy_compliance_validator import validate_metrics_privacy, validate_log_privacy

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for individual requests"""
    endpoint: str
    method: str
    duration: float
    status_code: int
    timestamp: datetime
    cache_hit: bool = False
    external_api_calls: int = 0
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None


@dataclass
class CacheMetrics:
    """Metrics for cache operations"""
    operation: str  # get, set, invalidate
    cache_level: str  # L1, L2, L3
    hit: bool
    duration: float
    timestamp: datetime
    key_pattern: Optional[str] = None


@dataclass
class ExternalAPIMetrics:
    """Metrics for external API calls"""
    service: str  # open_meteo, weatherapi
    endpoint: str
    duration: float
    success: bool
    status_code: Optional[int]
    timestamp: datetime
    retry_count: int = 0
    error_type: Optional[str] = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics"""
    request_count: int = 0
    avg_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    external_api_success_rate: float = 0.0
    external_api_avg_latency: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Collects and analyzes metrics for:
    - Request latency tracking for all endpoints
    - Cache hit/miss rate monitoring  
    - External API call latency tracking
    - Performance statistics and percentiles
    """
    
    def __init__(self, max_metrics_history: int = 10000, stats_window_minutes: int = 60):
        """
        Initialize performance monitor.
        
        Args:
            max_metrics_history: Maximum number of metrics to keep in memory
            stats_window_minutes: Time window for statistics calculation
        """
        self.max_metrics_history = max_metrics_history
        self.stats_window = timedelta(minutes=stats_window_minutes)
        
        # Thread-safe storage for metrics
        self._lock = Lock()
        
        # Metrics storage with deques for efficient memory management
        self.request_metrics: Deque[RequestMetrics] = deque(maxlen=max_metrics_history)
        self.cache_metrics: Deque[CacheMetrics] = deque(maxlen=max_metrics_history)
        self.external_api_metrics: Deque[ExternalAPIMetrics] = deque(maxlen=max_metrics_history)
        
        # Endpoint-specific counters
        self.endpoint_counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Alert thresholds
        self.alert_thresholds = {
            'max_response_time': 5.0,  # seconds
            'min_cache_hit_rate': 0.7,  # 70%
            'max_error_rate': 0.05,     # 5%
            'max_external_api_latency': 10.0  # seconds
        }
        
        # Alert callbacks
        self.alert_callbacks: List[callable] = []
        
        logger.info(f"Performance monitor initialized with {max_metrics_history} metrics history "
                   f"and {stats_window_minutes} minute stats window")
    
    def record_request(self, endpoint: str, method: str, duration: float, 
                      status_code: int, cache_hit: bool = False, 
                      external_api_calls: int = 0, user_agent: Optional[str] = None,
                      ip_address: Optional[str] = None) -> None:
        """
        Record metrics for an API request.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            duration: Request duration in seconds
            status_code: HTTP status code
            cache_hit: Whether request was served from cache
            external_api_calls: Number of external API calls made
            user_agent: Client user agent (anonymized)
            ip_address: Client IP (anonymized)
        """
        try:
            # Validate privacy compliance before recording
            request_data = {
                'endpoint': endpoint,
                'method': method,
                'user_agent': user_agent,
                'ip_address': ip_address
            }
            
            if not validate_metrics_privacy(request_data, "PerformanceMonitor.record_request"):
                logger.warning("Request metrics privacy validation failed - anonymizing data")
                user_agent = "[ANONYMIZED]"
                ip_address = "[ANONYMIZED]"
            
            metrics = RequestMetrics(
                endpoint=endpoint,
                method=method,
                duration=duration,
                status_code=status_code,
                timestamp=datetime.now(timezone.utc),
                cache_hit=cache_hit,
                external_api_calls=external_api_calls,
                user_agent=self._anonymize_user_agent(user_agent),
                ip_address=self._anonymize_ip(ip_address)
            )
            
            with self._lock:
                self.request_metrics.append(metrics)
                
                # Update endpoint counters
                self.endpoint_counters[endpoint]['total'] += 1
                if status_code >= 400:
                    self.endpoint_counters[endpoint]['errors'] += 1
                if cache_hit:
                    self.endpoint_counters[endpoint]['cache_hits'] += 1
            
            # Check for alerts
            self._check_request_alerts(metrics)
            
            logger.debug(f"Recorded request metrics: {endpoint} {method} "
                        f"{duration:.3f}s {status_code}")
                        
        except Exception as e:
            logger.error(f"Error recording request metrics: {e}")
    
    def record_cache_operation(self, operation: str, cache_level: str, 
                             hit: bool, duration: float, 
                             key_pattern: Optional[str] = None) -> None:
        """
        Record metrics for cache operations.
        
        Args:
            operation: Cache operation type (get, set, invalidate)
            cache_level: Cache level (L1, L2, L3)
            hit: Whether operation was a cache hit
            duration: Operation duration in seconds
            key_pattern: Anonymized cache key pattern
        """
        try:
            metrics = CacheMetrics(
                operation=operation,
                cache_level=cache_level,
                hit=hit,
                duration=duration,
                timestamp=datetime.now(timezone.utc),
                key_pattern=self._anonymize_cache_key(key_pattern)
            )
            
            with self._lock:
                self.cache_metrics.append(metrics)
            
            logger.debug(f"Recorded cache metrics: {operation} {cache_level} "
                        f"hit={hit} {duration:.3f}s")
                        
        except Exception as e:
            logger.error(f"Error recording cache metrics: {e}")
    
    def record_external_api_call(self, service: str, endpoint: str, 
                                duration: float, success: bool,
                                status_code: Optional[int] = None,
                                retry_count: int = 0,
                                error_type: Optional[str] = None) -> None:
        """
        Record metrics for external API calls.
        
        Args:
            service: External service name (open_meteo, weatherapi)
            endpoint: API endpoint
            duration: Call duration in seconds
            success: Whether call was successful
            status_code: HTTP status code
            retry_count: Number of retries performed
            error_type: Type of error if failed
        """
        try:
            metrics = ExternalAPIMetrics(
                service=service,
                endpoint=endpoint,
                duration=duration,
                success=success,
                status_code=status_code,
                timestamp=datetime.now(timezone.utc),
                retry_count=retry_count,
                error_type=error_type
            )
            
            with self._lock:
                self.external_api_metrics.append(metrics)
            
            # Check for alerts
            self._check_external_api_alerts(metrics)
            
            logger.debug(f"Recorded external API metrics: {service} {endpoint} "
                        f"{duration:.3f}s success={success}")
                        
        except Exception as e:
            logger.error(f"Error recording external API metrics: {e}")
    
    def get_performance_stats(self, time_window: Optional[timedelta] = None) -> PerformanceStats:
        """
        Get aggregated performance statistics.
        
        Args:
            time_window: Time window for statistics (default: configured window)
            
        Returns:
            PerformanceStats: Aggregated statistics
        """
        if time_window is None:
            time_window = self.stats_window
        
        cutoff_time = datetime.now(timezone.utc) - time_window
        
        with self._lock:
            # Filter metrics by time window
            recent_requests = [m for m in self.request_metrics if m.timestamp >= cutoff_time]
            recent_cache = [m for m in self.cache_metrics if m.timestamp >= cutoff_time]
            recent_external = [m for m in self.external_api_metrics if m.timestamp >= cutoff_time]
        
        # Calculate request statistics
        request_count = len(recent_requests)
        if request_count > 0:
            durations = [m.duration for m in recent_requests]
            avg_response_time = statistics.mean(durations)
            p50_response_time = statistics.median(durations)
            p95_response_time = self._percentile(durations, 0.95)
            p99_response_time = self._percentile(durations, 0.99)
            
            error_count = sum(1 for m in recent_requests if m.status_code >= 400)
            error_rate = error_count / request_count
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0.0
            error_rate = 0.0
        
        # Calculate cache statistics
        if recent_cache:
            cache_operations = len(recent_cache)
            cache_hits = sum(1 for m in recent_cache if m.hit)
            cache_hit_rate = cache_hits / cache_operations
            cache_miss_rate = 1.0 - cache_hit_rate
        else:
            cache_hit_rate = cache_miss_rate = 0.0
        
        # Calculate external API statistics
        if recent_external:
            external_successes = sum(1 for m in recent_external if m.success)
            external_api_success_rate = external_successes / len(recent_external)
            external_durations = [m.duration for m in recent_external]
            external_api_avg_latency = statistics.mean(external_durations)
        else:
            external_api_success_rate = 0.0
            external_api_avg_latency = 0.0
        
        return PerformanceStats(
            request_count=request_count,
            avg_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            cache_hit_rate=cache_hit_rate,
            cache_miss_rate=cache_miss_rate,
            external_api_success_rate=external_api_success_rate,
            external_api_avg_latency=external_api_avg_latency,
            error_rate=error_rate
        )
    
    def get_endpoint_stats(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for specific endpoint or all endpoints.
        
        Args:
            endpoint: Specific endpoint to get stats for (None for all)
            
        Returns:
            Dict with endpoint statistics
        """
        with self._lock:
            if endpoint:
                if endpoint in self.endpoint_counters:
                    return dict(self.endpoint_counters[endpoint])
                else:
                    return {}
            else:
                return {ep: dict(stats) for ep, stats in self.endpoint_counters.items()}
    
    def set_alert_threshold(self, metric: str, threshold: float) -> None:
        """
        Set alert threshold for a metric.
        
        Args:
            metric: Metric name
            threshold: Threshold value
        """
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = threshold
            logger.info(f"Alert threshold updated: {metric} = {threshold}")
        else:
            logger.warning(f"Unknown metric for alert threshold: {metric}")
    
    def add_alert_callback(self, callback: callable) -> None:
        """
        Add callback function for alerts.
        
        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
        logger.info("Alert callback added")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collected metrics.
        
        Returns:
            Dict with metrics summary
        """
        with self._lock:
            return {
                'total_requests': len(self.request_metrics),
                'total_cache_operations': len(self.cache_metrics),
                'total_external_api_calls': len(self.external_api_metrics),
                'endpoint_stats': self.get_endpoint_stats(),
                'alert_thresholds': self.alert_thresholds.copy(),
                'memory_usage': {
                    'request_metrics_mb': len(self.request_metrics) * 0.001,  # Rough estimate
                    'cache_metrics_mb': len(self.cache_metrics) * 0.0005,
                    'external_api_metrics_mb': len(self.external_api_metrics) * 0.0008
                }
            }
    
    def clear_old_metrics(self, older_than: timedelta) -> int:
        """
        Clear metrics older than specified time.
        
        Args:
            older_than: Time threshold for clearing metrics
            
        Returns:
            Number of metrics cleared
        """
        cutoff_time = datetime.now(timezone.utc) - older_than
        cleared_count = 0
        
        with self._lock:
            # Clear old request metrics
            original_len = len(self.request_metrics)
            self.request_metrics = deque(
                (m for m in self.request_metrics if m.timestamp >= cutoff_time),
                maxlen=self.max_metrics_history
            )
            cleared_count += original_len - len(self.request_metrics)
            
            # Clear old cache metrics
            original_len = len(self.cache_metrics)
            self.cache_metrics = deque(
                (m for m in self.cache_metrics if m.timestamp >= cutoff_time),
                maxlen=self.max_metrics_history
            )
            cleared_count += original_len - len(self.cache_metrics)
            
            # Clear old external API metrics
            original_len = len(self.external_api_metrics)
            self.external_api_metrics = deque(
                (m for m in self.external_api_metrics if m.timestamp >= cutoff_time),
                maxlen=self.max_metrics_history
            )
            cleared_count += original_len - len(self.external_api_metrics)
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old metrics")
        
        return cleared_count
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data using proper statistical method"""
        if not data:
            return 0.0
        
        if len(data) == 1:
            return data[0]
        
        # Use statistics.quantiles for proper percentile calculation
        try:
            # Convert percentile to quantile (e.g., 0.95 -> 95th percentile out of 100)
            n = 100
            quantile_index = int(percentile * n) - 1  # Convert to 0-based index
            if quantile_index < 0:
                quantile_index = 0
            
            quantiles = statistics.quantiles(data, n=n)
            if quantile_index >= len(quantiles):
                return max(data)
            
            return quantiles[quantile_index]
        except statistics.StatisticsError:
            # Fallback for edge cases
            sorted_data = sorted(data)
            index = percentile * (len(sorted_data) - 1)
            lower_index = int(index)
            upper_index = min(lower_index + 1, len(sorted_data) - 1)
            
            if lower_index == upper_index:
                return sorted_data[lower_index]
            
            # Linear interpolation
            weight = index - lower_index
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    def _anonymize_user_agent(self, user_agent: Optional[str]) -> Optional[str]:
        """Anonymize user agent for privacy"""
        if not user_agent:
            return None
        
        # Extract only browser type, remove version details
        if 'Chrome' in user_agent:
            return 'Chrome'
        elif 'Firefox' in user_agent:
            return 'Firefox'
        elif 'Safari' in user_agent:
            return 'Safari'
        elif 'Edge' in user_agent:
            return 'Edge'
        else:
            return 'Other'
    
    def _anonymize_ip(self, ip_address: Optional[str]) -> Optional[str]:
        """Anonymize IP address for privacy"""
        if not ip_address:
            return None
        
        # Return only network portion for IPv4
        if '.' in ip_address:
            parts = ip_address.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.xxx.xxx"
        
        # For IPv6 or other formats, return generic identifier
        return "anonymized"
    
    def _anonymize_cache_key(self, key_pattern: Optional[str]) -> Optional[str]:
        """Anonymize cache key pattern for privacy"""
        if not key_pattern:
            return None
        
        # Replace coordinate-like patterns with placeholders
        import re
        anonymized = re.sub(r'\d+\.\d+', 'XX.XX', key_pattern)
        return anonymized
    
    def _check_request_alerts(self, metrics: RequestMetrics) -> None:
        """Check if request metrics trigger any alerts"""
        try:
            # Check response time alert
            if metrics.duration > self.alert_thresholds['max_response_time']:
                self._trigger_alert('high_response_time', {
                    'endpoint': metrics.endpoint,
                    'duration': metrics.duration,
                    'threshold': self.alert_thresholds['max_response_time']
                })
            
            # Check error rate (calculated over recent requests)
            recent_stats = self.get_performance_stats(timedelta(minutes=5))
            if recent_stats.error_rate > self.alert_thresholds['max_error_rate']:
                self._trigger_alert('high_error_rate', {
                    'error_rate': recent_stats.error_rate,
                    'threshold': self.alert_thresholds['max_error_rate']
                })
                
        except Exception as e:
            logger.error(f"Error checking request alerts: {e}")
    
    def _check_external_api_alerts(self, metrics: ExternalAPIMetrics) -> None:
        """Check if external API metrics trigger any alerts"""
        try:
            # Check external API latency alert
            if metrics.duration > self.alert_thresholds['max_external_api_latency']:
                self._trigger_alert('high_external_api_latency', {
                    'service': metrics.service,
                    'duration': metrics.duration,
                    'threshold': self.alert_thresholds['max_external_api_latency']
                })
                
        except Exception as e:
            logger.error(f"Error checking external API alerts: {e}")
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger alert with given type and data"""
        alert_data = {
            'type': alert_type,
            'timestamp': datetime.now(timezone.utc),
            'data': data
        }
        
        logger.warning(f"Performance alert triggered: {alert_type} - {data}")
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def setup_performance_monitoring(max_metrics_history: int = 10000, 
                                stats_window_minutes: int = 60) -> PerformanceMonitor:
    """
    Setup global performance monitoring.
    
    Args:
        max_metrics_history: Maximum metrics to keep in memory
        stats_window_minutes: Statistics calculation window
        
    Returns:
        PerformanceMonitor instance
    """
    global _performance_monitor
    _performance_monitor = PerformanceMonitor(max_metrics_history, stats_window_minutes)
    logger.info("Performance monitoring setup completed")
    return _performance_monitor