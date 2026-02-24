"""
Prometheus metrics exporter for AirTrace RU Backend

Implements Prometheus metrics format export, alerting, and anomaly detection
for performance monitoring data.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from threading import Lock
import statistics
import re

from performance_monitor import get_performance_monitor, PerformanceStats
from system_monitor import get_system_monitor

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: str
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    threshold: float
    duration: int  # seconds
    severity: str  # 'critical', 'warning', 'info'
    description: str
    enabled: bool = True


@dataclass
class Alert:
    """Active alert"""
    rule_name: str
    metric: str
    current_value: float
    threshold: float
    severity: str
    description: str
    started_at: datetime
    last_triggered: datetime
    trigger_count: int = 1


@dataclass
class MetricSample:
    """Metric sample for anomaly detection"""
    timestamp: datetime
    value: float
    metric_name: str


class PrometheusExporter:
    """
    Prometheus metrics exporter with alerting and anomaly detection.
    
    Exports performance metrics in Prometheus format and provides
    alerting capabilities with threshold monitoring and pattern analysis.
    """
    
    def __init__(self, alert_check_interval: int = 60):
        """
        Initialize Prometheus exporter.
        
        Args:
            alert_check_interval: Interval for checking alert rules in seconds
        """
        self.alert_check_interval = alert_check_interval
        
        # Thread-safe storage
        self._lock = Lock()
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # Anomaly detection
        self.metric_baselines: Dict[str, float] = {}
        self.metric_samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.anomaly_threshold = 2.0  # Standard deviations
        
        # Alerting state
        self.alerting_active = False
        self.alerting_task: Optional[asyncio.Task] = None
        
        logger.info("Prometheus exporter initialized")
    
    async def export_metrics(self) -> str:
        """
        Export all metrics in Prometheus format.
        
        Returns:
            str: Metrics in Prometheus exposition format
        """
        try:
            metrics_lines = []
            
            # Add metadata
            metrics_lines.append("# HELP airtrace_info AirTrace RU Backend information")
            metrics_lines.append("# TYPE airtrace_info gauge")
            metrics_lines.append('airtrace_info{version="0.3.1",service="airtrace-ru"} 1')
            metrics_lines.append("")
            
            # Export performance metrics
            performance_metrics = await self._export_performance_metrics()
            metrics_lines.extend(performance_metrics)
            
            # Export system metrics
            system_metrics = self._export_system_metrics()
            metrics_lines.extend(system_metrics)
            
            # Export alert metrics
            alert_metrics = self._export_alert_metrics()
            metrics_lines.extend(alert_metrics)
            
            return "\n".join(metrics_lines)
            
        except Exception as e:
            logger.error(f"Error exporting Prometheus metrics: {e}")
            return "# Error exporting metrics\n"
    
    async def _export_performance_metrics(self) -> List[str]:
        """Export performance monitoring metrics"""
        lines = []
        
        try:
            monitor = get_performance_monitor()
            stats = monitor.get_performance_stats()
            
            # Request metrics
            lines.extend([
                "# HELP airtrace_requests_total Total number of requests",
                "# TYPE airtrace_requests_total counter",
                f"airtrace_requests_total {stats.request_count}",
                "",
                "# HELP airtrace_request_duration_seconds Request duration percentiles",
                "# TYPE airtrace_request_duration_seconds histogram",
                f'airtrace_request_duration_seconds{{quantile="0.5"}} {stats.p50_response_time}',
                f'airtrace_request_duration_seconds{{quantile="0.95"}} {stats.p95_response_time}',
                f'airtrace_request_duration_seconds{{quantile="0.99"}} {stats.p99_response_time}',
                f"airtrace_request_duration_seconds_sum {stats.avg_response_time * stats.request_count}",
                f"airtrace_request_duration_seconds_count {stats.request_count}",
                "",
                "# HELP airtrace_error_rate Request error rate",
                "# TYPE airtrace_error_rate gauge",
                f"airtrace_error_rate {stats.error_rate}",
                ""
            ])
            
            # Cache metrics - get from cache manager if available
            cache_l1_hits = cache_l1_misses = cache_l2_hits = cache_l2_misses = cache_l3_hits = cache_l3_misses = 0
            
            try:
                # Try to get cache stats from unified weather service
                from unified_weather_service import unified_weather_service
                if hasattr(unified_weather_service, 'cache_manager'):
                    cache_stats = await unified_weather_service.cache_manager.get_stats()
                    cache_l1_hits = cache_stats.l1_hits
                    cache_l1_misses = cache_stats.l1_misses
                    cache_l2_hits = cache_stats.l2_hits
                    cache_l2_misses = cache_stats.l2_misses
                    cache_l3_hits = cache_stats.l3_hits
                    cache_l3_misses = cache_stats.l3_misses
            except Exception as e:
                logger.debug(f"Could not get cache level stats: {e}")
            
            lines.extend([
                "# HELP airtrace_cache_hit_rate Cache hit rate",
                "# TYPE airtrace_cache_hit_rate gauge",
                f"airtrace_cache_hit_rate {stats.cache_hit_rate}",
                "",
                "# HELP airtrace_cache_miss_rate Cache miss rate", 
                "# TYPE airtrace_cache_miss_rate gauge",
                f"airtrace_cache_miss_rate {stats.cache_miss_rate}",
                "",
                "# HELP airtrace_cache_hits_total Cache hits by level",
                "# TYPE airtrace_cache_hits_total counter",
                f'airtrace_cache_hits_total{{level="L1"}} {cache_l1_hits}',
                f'airtrace_cache_hits_total{{level="L2"}} {cache_l2_hits}',
                f'airtrace_cache_hits_total{{level="L3"}} {cache_l3_hits}',
                "",
                "# HELP airtrace_cache_misses_total Cache misses by level",
                "# TYPE airtrace_cache_misses_total counter",
                f'airtrace_cache_misses_total{{level="L1"}} {cache_l1_misses}',
                f'airtrace_cache_misses_total{{level="L2"}} {cache_l2_misses}',
                f'airtrace_cache_misses_total{{level="L3"}} {cache_l3_misses}',
                ""
            ])
            
            # Try to get detailed cache statistics from cache manager
            try:
                # Note: This would need to be called from an async context
                # For now, we'll skip detailed cache stats in Prometheus export
                # and rely on the performance monitor stats
                pass
                
            except Exception as e:
                logger.debug(f"Could not get detailed cache stats: {e}")
            
            # External API metrics
            lines.extend([
                "# HELP airtrace_external_api_success_rate External API success rate",
                "# TYPE airtrace_external_api_success_rate gauge",
                f"airtrace_external_api_success_rate {stats.external_api_success_rate}",
                "",
                "# HELP airtrace_external_api_duration_seconds External API call duration",
                "# TYPE airtrace_external_api_duration_seconds gauge",
                f"airtrace_external_api_duration_seconds {stats.external_api_avg_latency}",
                ""
            ])
            
            # Rate limiting metrics
            try:
                from rate_limiter import get_rate_limiter
                rate_limiter = get_rate_limiter()
                rate_metrics = await rate_limiter.get_metrics()
                
                lines.extend([
                    "# HELP airtrace_rate_limit_requests_total Total rate limit requests",
                    "# TYPE airtrace_rate_limit_requests_total counter",
                    f"airtrace_rate_limit_requests_total {rate_metrics.total_requests}",
                    "",
                    "# HELP airtrace_rate_limit_allowed_requests_total Allowed rate limit requests",
                    "# TYPE airtrace_rate_limit_allowed_requests_total counter",
                    f"airtrace_rate_limit_allowed_requests_total {rate_metrics.allowed_requests}",
                    "",
                    "# HELP airtrace_rate_limit_blocked_requests_total Blocked rate limit requests",
                    "# TYPE airtrace_rate_limit_blocked_requests_total counter",
                    f"airtrace_rate_limit_blocked_requests_total {rate_metrics.blocked_requests}",
                    "",
                    "# HELP airtrace_rate_limit_burst_requests_total Burst rate limit requests",
                    "# TYPE airtrace_rate_limit_burst_requests_total counter",
                    f"airtrace_rate_limit_burst_requests_total {rate_metrics.burst_requests}",
                    ""
                ])
                
                # Per-endpoint rate limiting metrics
                if hasattr(rate_metrics, 'endpoint_stats') and rate_metrics.endpoint_stats:
                    lines.extend([
                        "# HELP airtrace_rate_limit_endpoint_requests_total Rate limit requests per endpoint",
                        "# TYPE airtrace_rate_limit_endpoint_requests_total counter"
                    ])
                    
                    for endpoint, endpoint_stats in rate_metrics.endpoint_stats.items():
                        safe_endpoint = self._sanitize_label_value(endpoint)
                        lines.append(f'airtrace_rate_limit_endpoint_requests_total{{endpoint="{safe_endpoint}"}} {endpoint_stats.get("total", 0)}')
                        lines.append(f'airtrace_rate_limit_endpoint_blocked_total{{endpoint="{safe_endpoint}"}} {endpoint_stats.get("blocked", 0)}')
                    
                    lines.append("")
                    
            except Exception as e:
                logger.debug(f"Could not get rate limiting metrics: {e}")
            
            # Connection pool metrics
            try:
                from connection_pool import get_connection_pool_manager
                pool_manager = get_connection_pool_manager()
                pool_stats = await pool_manager.get_all_stats()
                
                lines.extend([
                    "# HELP airtrace_connection_pool_total_connections Total connections in pools",
                    "# TYPE airtrace_connection_pool_total_connections gauge",
                    "# HELP airtrace_connection_pool_active_connections Active connections in pools",
                    "# TYPE airtrace_connection_pool_active_connections gauge",
                    "# HELP airtrace_connection_pool_idle_connections Idle connections in pools",
                    "# TYPE airtrace_connection_pool_idle_connections gauge",
                    "# HELP airtrace_connection_pool_failed_connections Failed connections in pools",
                    "# TYPE airtrace_connection_pool_failed_connections counter"
                ])
                
                for pool_name, pool_stat in pool_stats.items():
                    safe_pool_name = self._sanitize_label_value(pool_name)
                    lines.extend([
                        f'airtrace_connection_pool_total_connections{{pool="{safe_pool_name}"}} {pool_stat.get("total_connections", 0)}',
                        f'airtrace_connection_pool_active_connections{{pool="{safe_pool_name}"}} {pool_stat.get("active_connections", 0)}',
                        f'airtrace_connection_pool_idle_connections{{pool="{safe_pool_name}"}} {pool_stat.get("idle_connections", 0)}',
                        f'airtrace_connection_pool_failed_connections{{pool="{safe_pool_name}"}} {pool_stat.get("failed_connections", 0)}'
                    ])
                
                lines.append("")
                
            except Exception as e:
                logger.debug(f"Could not get connection pool metrics: {e}")
            
            # Resource management metrics
            try:
                from resource_manager import get_resource_manager
                resource_manager = get_resource_manager()
                resource_usage = await resource_manager.get_resource_usage()
                
                lines.extend([
                    "# HELP airtrace_resource_memory_usage_bytes Current memory usage",
                    "# TYPE airtrace_resource_memory_usage_bytes gauge",
                    f"airtrace_resource_memory_usage_bytes {resource_usage.memory_usage_bytes}",
                    "",
                    "# HELP airtrace_resource_memory_limit_bytes Memory limit",
                    "# TYPE airtrace_resource_memory_limit_bytes gauge",
                    f"airtrace_resource_memory_limit_bytes {resource_usage.memory_limit_bytes}",
                    "",
                    "# HELP airtrace_resource_cpu_usage_percent Current CPU usage percentage",
                    "# TYPE airtrace_resource_cpu_usage_percent gauge",
                    f"airtrace_resource_cpu_usage_percent {resource_usage.cpu_usage_percent}",
                    "",
                    "# HELP airtrace_resource_gc_collections_total Total garbage collections",
                    "# TYPE airtrace_resource_gc_collections_total counter",
                    f"airtrace_resource_gc_collections_total {resource_usage.gc_collections}",
                    ""
                ])
                
            except Exception as e:
                logger.debug(f"Could not get resource management metrics: {e}")
            
            # WeatherAPI integration metrics
            try:
                from weather_api_manager import get_weather_api_manager
                weather_api_manager = get_weather_api_manager()
                api_stats = await weather_api_manager.get_api_status()
                
                lines.extend([
                    "# HELP airtrace_weatherapi_requests_total Total WeatherAPI requests",
                    "# TYPE airtrace_weatherapi_requests_total counter",
                    f"airtrace_weatherapi_requests_total {api_stats.get('total_requests', 0)}",
                    "",
                    "# HELP airtrace_weatherapi_success_rate WeatherAPI success rate",
                    "# TYPE airtrace_weatherapi_success_rate gauge",
                    f"airtrace_weatherapi_success_rate {api_stats.get('success_rate', 0.0)}",
                    "",
                    "# HELP airtrace_weatherapi_avg_latency_seconds WeatherAPI average latency",
                    "# TYPE airtrace_weatherapi_avg_latency_seconds gauge",
                    f"airtrace_weatherapi_avg_latency_seconds {api_stats.get('avg_latency', 0.0)}",
                    ""
                ])
                
            except Exception as e:
                logger.debug(f"Could not get WeatherAPI metrics: {e}")
            
            # Endpoint-specific metrics
            endpoint_stats = monitor.get_endpoint_stats()
            if endpoint_stats:
                lines.extend([
                    "# HELP airtrace_endpoint_requests_total Requests per endpoint",
                    "# TYPE airtrace_endpoint_requests_total counter",
                    "# HELP airtrace_endpoint_errors_total Errors per endpoint",
                    "# TYPE airtrace_endpoint_errors_total counter",
                    "# HELP airtrace_endpoint_cache_hits_total Cache hits per endpoint",
                    "# TYPE airtrace_endpoint_cache_hits_total counter"
                ])
                
                for endpoint, stats_dict in endpoint_stats.items():
                    safe_endpoint = self._sanitize_label_value(endpoint)
                    total = stats_dict.get('total', 0)
                    errors = stats_dict.get('errors', 0)
                    cache_hits = stats_dict.get('cache_hits', 0)
                    
                    lines.extend([
                        f'airtrace_endpoint_requests_total{{endpoint="{safe_endpoint}"}} {total}',
                        f'airtrace_endpoint_errors_total{{endpoint="{safe_endpoint}"}} {errors}',
                        f'airtrace_endpoint_cache_hits_total{{endpoint="{safe_endpoint}"}} {cache_hits}'
                    ])
                
                lines.append("")
            
            # Request optimization metrics
            try:
                from request_optimizer import get_request_optimizer
                optimizer = get_request_optimizer()
                optimization_stats = await optimizer.get_optimization_stats()
                
                lines.extend([
                    "# HELP airtrace_request_batching_efficiency Request batching efficiency ratio",
                    "# TYPE airtrace_request_batching_efficiency gauge",
                    f"airtrace_request_batching_efficiency {optimization_stats.get('batching_efficiency', 0.0)}",
                    "",
                    "# HELP airtrace_request_deduplication_rate Request deduplication rate",
                    "# TYPE airtrace_request_deduplication_rate gauge",
                    f"airtrace_request_deduplication_rate {optimization_stats.get('deduplication_rate', 0.0)}",
                    "",
                    "# HELP airtrace_request_prefetch_hits_total Prefetch cache hits",
                    "# TYPE airtrace_request_prefetch_hits_total counter",
                    f"airtrace_request_prefetch_hits_total {optimization_stats.get('prefetch_hits', 0)}",
                    "",
                    "# HELP airtrace_request_batched_total Total batched requests",
                    "# TYPE airtrace_request_batched_total counter",
                    f"airtrace_request_batched_total {optimization_stats.get('batched_requests', 0)}",
                    ""
                ])
                
            except Exception as e:
                logger.debug(f"Could not get request optimization metrics: {e}")
            
            # Multi-level cache detailed metrics
            try:
                from cache import get_cache_manager
                cache_manager = get_cache_manager()
                detailed_cache_stats = await cache_manager.get_detailed_stats()
                
                lines.extend([
                    "# HELP airtrace_cache_level_hit_rate Cache hit rate by level",
                    "# TYPE airtrace_cache_level_hit_rate gauge",
                    "# HELP airtrace_cache_level_size_bytes Cache size by level",
                    "# TYPE airtrace_cache_level_size_bytes gauge",
                    "# HELP airtrace_cache_level_evictions_total Cache evictions by level",
                    "# TYPE airtrace_cache_level_evictions_total counter"
                ])
                
                for level in ['L1', 'L2', 'L3']:
                    level_stats = detailed_cache_stats.get(level, {})
                    hit_rate = level_stats.get('hit_rate', 0.0)
                    size_bytes = level_stats.get('size_bytes', 0)
                    evictions = level_stats.get('evictions', 0)
                    
                    lines.extend([
                        f'airtrace_cache_level_hit_rate{{level="{level}"}} {hit_rate}',
                        f'airtrace_cache_level_size_bytes{{level="{level}"}} {size_bytes}',
                        f'airtrace_cache_level_evictions_total{{level="{level}"}} {evictions}'
                    ])
                
                lines.append("")
                
            except Exception as e:
                logger.debug(f"Could not get detailed cache metrics: {e}")
            
        except Exception as e:
            logger.error(f"Error exporting performance metrics: {e}")
            lines.append("# Error exporting performance metrics")
        
        return lines
    
    def _export_system_metrics(self) -> List[str]:
        """Export system resource metrics"""
        lines = []
        
        try:
            system_monitor = get_system_monitor()
            
            # Get latest system metrics
            if system_monitor.metrics_history:
                latest_metrics = system_monitor.metrics_history[-1]
                
                lines.extend([
                    "# HELP airtrace_cpu_usage_percent CPU usage percentage",
                    "# TYPE airtrace_cpu_usage_percent gauge",
                    f"airtrace_cpu_usage_percent {latest_metrics.cpu_usage}",
                    "",
                    "# HELP airtrace_memory_usage_bytes Memory usage in bytes",
                    "# TYPE airtrace_memory_usage_bytes gauge",
                    f"airtrace_memory_usage_bytes {latest_metrics.memory_usage}",
                    "",
                    "# HELP airtrace_memory_usage_percent Memory usage percentage",
                    "# TYPE airtrace_memory_usage_percent gauge",
                    f"airtrace_memory_usage_percent {latest_metrics.memory_percent}",
                    "",
                    "# HELP airtrace_disk_usage_bytes Disk usage in bytes",
                    "# TYPE airtrace_disk_usage_bytes gauge",
                    f"airtrace_disk_usage_bytes {latest_metrics.disk_usage}",
                    "",
                    "# HELP airtrace_disk_usage_percent Disk usage percentage",
                    "# TYPE airtrace_disk_usage_percent gauge",
                    f"airtrace_disk_usage_percent {latest_metrics.disk_percent}",
                    "",
                    "# HELP airtrace_network_bytes_sent Network bytes sent",
                    "# TYPE airtrace_network_bytes_sent counter",
                    f"airtrace_network_bytes_sent {latest_metrics.network_io_sent}",
                    "",
                    "# HELP airtrace_network_bytes_received Network bytes received",
                    "# TYPE airtrace_network_bytes_received counter",
                    f"airtrace_network_bytes_received {latest_metrics.network_io_recv}",
                    "",
                    "# HELP airtrace_active_connections Active network connections",
                    "# TYPE airtrace_active_connections gauge",
                    f"airtrace_active_connections {latest_metrics.active_connections}",
                    "",
                    "# HELP airtrace_process_count Total process count",
                    "# TYPE airtrace_process_count gauge",
                    f"airtrace_process_count {latest_metrics.process_count}",
                    ""
                ])
                
                # Add load average if available
                if latest_metrics.load_average is not None:
                    lines.extend([
                        "# HELP airtrace_load_average_1m Load average (1 minute)",
                        "# TYPE airtrace_load_average_1m gauge",
                        f"airtrace_load_average_1m {latest_metrics.load_average}",
                        ""
                    ])
                
                # Add additional system metrics
                lines.extend([
                    "# HELP airtrace_system_uptime_seconds System uptime in seconds",
                    "# TYPE airtrace_system_uptime_seconds counter",
                    f"airtrace_system_uptime_seconds {getattr(latest_metrics, 'uptime_seconds', 0)}",
                    "",
                    "# HELP airtrace_file_descriptors_open Open file descriptors",
                    "# TYPE airtrace_file_descriptors_open gauge",
                    f"airtrace_file_descriptors_open {getattr(latest_metrics, 'open_file_descriptors', 0)}",
                    ""
                ])
            
            # Add application-specific metrics
            lines.extend([
                "# HELP airtrace_application_start_time_seconds Application start time",
                "# TYPE airtrace_application_start_time_seconds gauge",
                f"airtrace_application_start_time_seconds {time.time()}",
                "",
                "# HELP airtrace_python_gc_objects_collected_total Objects collected during gc",
                "# TYPE airtrace_python_gc_objects_collected_total counter"
            ])
            
            # Add Python GC metrics
            import gc
            for i, count in enumerate(gc.get_count()):
                lines.append(f'airtrace_python_gc_objects_collected_total{{generation="{i}"}} {count}')
            
            lines.append("")
            
        except Exception as e:
            logger.error(f"Error exporting system metrics: {e}")
            lines.append("# Error exporting system metrics")
        
        return lines
    
    def _export_alert_metrics(self) -> List[str]:
        """Export alerting metrics"""
        lines = []
        
        try:
            with self._lock:
                lines.extend([
                    "# HELP airtrace_alerts_active Number of active alerts",
                    "# TYPE airtrace_alerts_active gauge",
                    f"airtrace_alerts_active {len(self.active_alerts)}",
                    "",
                    "# HELP airtrace_alert_rules_total Total number of alert rules",
                    "# TYPE airtrace_alert_rules_total gauge",
                    f"airtrace_alert_rules_total {len(self.alert_rules)}",
                    ""
                ])
                
                # Export individual alert states
                if self.active_alerts:
                    lines.extend([
                        "# HELP airtrace_alert_state Alert state (1=firing, 0=resolved)",
                        "# TYPE airtrace_alert_state gauge"
                    ])
                    
                    for alert_name, alert in self.active_alerts.items():
                        safe_name = self._sanitize_label_value(alert_name)
                        safe_metric = self._sanitize_label_value(alert.metric)
                        lines.append(
                            f'airtrace_alert_state{{alert="{safe_name}",metric="{safe_metric}",severity="{alert.severity}"}} 1'
                        )
                    
                    lines.append("")
            
        except Exception as e:
            logger.error(f"Error exporting alert metrics: {e}")
            lines.append("# Error exporting alert metrics")
        
        return lines
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """
        Add an alert rule.
        
        Args:
            rule: Alert rule configuration
        """
        with self._lock:
            self.alert_rules[rule.name] = rule
        
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            bool: True if rule was removed
        """
        with self._lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                # Also remove any active alerts for this rule
                if rule_name in self.active_alerts:
                    del self.active_alerts[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
        
        return False
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Add callback for alert notifications.
        
        Args:
            callback: Function to call when alerts are triggered
        """
        self.alert_callbacks.append(callback)
        logger.info("Added alert callback")
    
    async def start_alerting(self) -> None:
        """Start the alerting system"""
        if self.alerting_active:
            logger.warning("Alerting system is already active")
            return
        
        self.alerting_active = True
        self.alerting_task = asyncio.create_task(self._alerting_loop())
        logger.info("Alerting system started")
    
    async def stop_alerting(self) -> None:
        """Stop the alerting system"""
        if not self.alerting_active:
            return
        
        self.alerting_active = False
        
        if self.alerting_task:
            self.alerting_task.cancel()
            try:
                await self.alerting_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Alerting system stopped")
    
    async def _alerting_loop(self) -> None:
        """Main alerting loop"""
        logger.info("Alerting loop started")
        
        while self.alerting_active:
            try:
                await self._check_alert_rules()
                await self._detect_anomalies()
                await asyncio.sleep(self.alert_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alerting loop: {e}")
                await asyncio.sleep(self.alert_check_interval)
        
        logger.info("Alerting loop stopped")
    
    async def _check_alert_rules(self) -> None:
        """Check all alert rules against current metrics"""
        try:
            # Get current metrics
            performance_monitor = get_performance_monitor()
            system_monitor = get_system_monitor()
            
            perf_stats = performance_monitor.get_performance_stats()
            
            # Get latest system metrics
            system_metrics = None
            if system_monitor.metrics_history:
                system_metrics = system_monitor.metrics_history[-1]
            
            # Check each alert rule
            with self._lock:
                for rule_name, rule in self.alert_rules.items():
                    if not rule.enabled:
                        continue
                    
                    current_value = self._get_metric_value(rule.metric, perf_stats, system_metrics)
                    if current_value is None:
                        continue
                    
                    # Check if rule condition is met
                    should_alert = self._evaluate_condition(current_value, rule.condition, rule.threshold)
                    
                    if should_alert:
                        await self._trigger_alert(rule, current_value)
                    else:
                        await self._resolve_alert(rule_name)
            
        except Exception as e:
            logger.error(f"Error checking alert rules: {e}")
    
    def _get_metric_value(self, metric_name: str, perf_stats: PerformanceStats, system_metrics) -> Optional[float]:
        """Get current value for a metric"""
        # Performance metrics
        if metric_name == "response_time_avg":
            return perf_stats.avg_response_time
        elif metric_name == "response_time_p95":
            return perf_stats.p95_response_time
        elif metric_name == "response_time_p99":
            return perf_stats.p99_response_time
        elif metric_name == "error_rate":
            return perf_stats.error_rate
        elif metric_name == "cache_hit_rate":
            return perf_stats.cache_hit_rate
        elif metric_name == "external_api_success_rate":
            return perf_stats.external_api_success_rate
        elif metric_name == "external_api_latency":
            return perf_stats.external_api_avg_latency
        
        # System metrics
        elif system_metrics:
            if metric_name == "cpu_usage":
                return system_metrics.cpu_usage
            elif metric_name == "memory_percent":
                return system_metrics.memory_percent
            elif metric_name == "disk_percent":
                return system_metrics.disk_percent
            elif metric_name == "load_average" and system_metrics.load_average:
                return system_metrics.load_average
        
        return None
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.001
        elif condition == "ne":
            return abs(value - threshold) >= 0.001
        
        return False
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float) -> None:
        """Trigger an alert"""
        now = datetime.now(timezone.utc)
        
        if rule.name in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[rule.name]
            alert.current_value = current_value
            alert.last_triggered = now
            alert.trigger_count += 1
        else:
            # Create new alert
            alert = Alert(
                rule_name=rule.name,
                metric=rule.metric,
                current_value=current_value,
                threshold=rule.threshold,
                severity=rule.severity,
                description=rule.description,
                started_at=now,
                last_triggered=now
            )
            
            self.active_alerts[rule.name] = alert
            self.alert_history.append(alert)
            
            logger.warning(f"Alert triggered: {rule.name} - {rule.description}")
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    async def _resolve_alert(self, rule_name: str) -> None:
        """Resolve an active alert"""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            del self.active_alerts[rule_name]
            
            logger.info(f"Alert resolved: {rule_name}")
    
    async def _detect_anomalies(self) -> None:
        """Detect anomalies in metrics using statistical analysis"""
        try:
            # Get current metrics
            performance_monitor = get_performance_monitor()
            system_monitor = get_system_monitor()
            
            perf_stats = performance_monitor.get_performance_stats()
            
            # Collect metric samples for anomaly detection
            now = datetime.now(timezone.utc)
            
            # Performance metrics
            metrics_to_check = [
                ("response_time_avg", perf_stats.avg_response_time),
                ("error_rate", perf_stats.error_rate),
                ("cache_hit_rate", perf_stats.cache_hit_rate),
                ("external_api_latency", perf_stats.external_api_avg_latency)
            ]
            
            # System metrics
            if system_monitor.metrics_history:
                latest_system = system_monitor.metrics_history[-1]
                metrics_to_check.extend([
                    ("cpu_usage", latest_system.cpu_usage),
                    ("memory_percent", latest_system.memory_percent),
                    ("disk_percent", latest_system.disk_percent)
                ])
            
            # Check each metric for anomalies
            for metric_name, value in metrics_to_check:
                if value is not None:
                    await self._check_metric_anomaly(metric_name, value, now)
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
    
    async def _check_metric_anomaly(self, metric_name: str, value: float, timestamp: datetime) -> None:
        """Check if a metric value is anomalous"""
        # Add sample to history
        sample = MetricSample(timestamp, value, metric_name)
        self.metric_samples[metric_name].append(sample)
        
        # Need at least 10 samples for anomaly detection
        samples = self.metric_samples[metric_name]
        if len(samples) < 10:
            return
        
        # Calculate statistics
        values = [s.value for s in samples]
        mean = statistics.mean(values)
        
        if len(values) > 1:
            stdev = statistics.stdev(values)
            
            # Check if current value is anomalous
            if stdev > 0:
                z_score = abs(value - mean) / stdev
                
                if z_score > self.anomaly_threshold:
                    # Create anomaly alert
                    anomaly_rule = AlertRule(
                        name=f"anomaly_{metric_name}",
                        metric=metric_name,
                        condition="gt",
                        threshold=mean + (self.anomaly_threshold * stdev),
                        duration=0,
                        severity="warning",
                        description=f"Anomaly detected in {metric_name}: value {value:.2f} is {z_score:.2f} standard deviations from mean {mean:.2f}",
                        enabled=True
                    )
                    
                    await self._trigger_alert(anomaly_rule, value)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts"""
        with self._lock:
            return list(self.active_alerts.values())
    
    async def validate_metrics_completeness(self) -> Dict[str, Any]:
        """
        Validate that all expected metrics are being exported.
        
        Returns:
            Dict with validation results and missing metrics
        """
        validation_result = {
            "valid": True,
            "missing_metrics": [],
            "component_status": {},
            "total_metrics_exported": 0
        }
        
        try:
            # Export current metrics
            metrics_output = await self.export_metrics()
            lines = metrics_output.split('\n')
            
            # Count total metrics
            metric_lines = [line for line in lines if line and not line.startswith('#')]
            validation_result["total_metrics_exported"] = len(metric_lines)
            
            # Check for required metric categories
            required_metrics = [
                "airtrace_requests_total",
                "airtrace_request_duration_seconds",
                "airtrace_cache_hit_rate",
                "airtrace_external_api_success_rate",
                "airtrace_cpu_usage_percent",
                "airtrace_memory_usage_bytes",
                "airtrace_alerts_active"
            ]
            
            # Optional metrics that should be present if components are enabled
            optional_metrics = [
                ("airtrace_rate_limit_requests_total", "rate_limiting"),
                ("airtrace_connection_pool_total_connections", "connection_pooling"),
                ("airtrace_resource_memory_usage_bytes", "resource_management"),
                ("airtrace_weatherapi_requests_total", "weatherapi_integration"),
                ("airtrace_request_batching_efficiency", "request_optimization"),
                ("airtrace_cache_level_hit_rate", "multi_level_cache")
            ]
            
            # Check required metrics
            for metric in required_metrics:
                if metric not in metrics_output:
                    validation_result["missing_metrics"].append(metric)
                    validation_result["valid"] = False
                else:
                    validation_result["component_status"][metric] = "present"
            
            # Check optional metrics
            for metric, component in optional_metrics:
                if metric in metrics_output:
                    validation_result["component_status"][component] = "present"
                else:
                    validation_result["component_status"][component] = "missing"
            
            # Validate Prometheus format compliance
            format_issues = self._validate_prometheus_format(metrics_output)
            if format_issues:
                validation_result["format_issues"] = format_issues
                validation_result["valid"] = False
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["error"] = str(e)
        
        return validation_result
    
    def _validate_prometheus_format(self, metrics_output: str) -> List[str]:
        """Validate Prometheus format compliance"""
        issues = []
        lines = metrics_output.split('\n')
        
        for i, line in enumerate(lines):
            if line and not line.startswith('#') and line.strip():
                # Check metric name format
                if ' ' in line:
                    parts = line.split(' ')
                    metric_name_part = parts[0]
                    
                    # Validate metric name
                    if not re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{.*\})?$', metric_name_part):
                        issues.append(f"Line {i+1}: Invalid metric name format: {metric_name_part}")
                    
                    # Validate value
                    try:
                        value = parts[-1]
                        float(value)
                    except (ValueError, IndexError):
                        issues.append(f"Line {i+1}: Invalid metric value: {line}")
        
        return issues
    
    def _sanitize_label_value(self, value: str) -> str:
        """Sanitize label value for Prometheus format"""
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', value)
        return sanitized


# Global Prometheus exporter instance
_prometheus_exporter: Optional[PrometheusExporter] = None


def get_prometheus_exporter() -> PrometheusExporter:
    """Get global Prometheus exporter instance"""
    global _prometheus_exporter
    if _prometheus_exporter is None:
        _prometheus_exporter = PrometheusExporter()
    return _prometheus_exporter


def setup_prometheus_exporter(alert_check_interval: int = 60) -> PrometheusExporter:
    """
    Setup global Prometheus exporter.
    
    Args:
        alert_check_interval: Alert checking interval in seconds
        
    Returns:
        PrometheusExporter instance
    """
    global _prometheus_exporter
    _prometheus_exporter = PrometheusExporter(alert_check_interval)
    logger.info("Prometheus exporter setup completed")
    return _prometheus_exporter


# Default alert rules for common scenarios
DEFAULT_ALERT_RULES = [
    AlertRule(
        name="high_response_time",
        metric="response_time_p95",
        condition="gt",
        threshold=5.0,
        duration=300,
        severity="warning",
        description="95th percentile response time is above 5 seconds"
    ),
    AlertRule(
        name="high_error_rate",
        metric="error_rate",
        condition="gt",
        threshold=0.05,
        duration=300,
        severity="critical",
        description="Error rate is above 5%"
    ),
    AlertRule(
        name="low_cache_hit_rate",
        metric="cache_hit_rate",
        condition="lt",
        threshold=0.7,
        duration=600,
        severity="warning",
        description="Cache hit rate is below 70%"
    ),
    AlertRule(
        name="high_cpu_usage",
        metric="cpu_usage",
        condition="gt",
        threshold=80.0,
        duration=300,
        severity="warning",
        description="CPU usage is above 80%"
    ),
    AlertRule(
        name="high_memory_usage",
        metric="memory_percent",
        condition="gt",
        threshold=85.0,
        duration=300,
        severity="critical",
        description="Memory usage is above 85%"
    ),
    AlertRule(
        name="low_external_api_success_rate",
        metric="external_api_success_rate",
        condition="lt",
        threshold=0.95,
        duration=300,
        severity="warning",
        description="External API success rate is below 95%"
    )
]


def setup_default_alerts() -> None:
    """Setup default alert rules"""
    exporter = get_prometheus_exporter()
    
    for rule in DEFAULT_ALERT_RULES:
        exporter.add_alert_rule(rule)
    
    logger.info(f"Setup {len(DEFAULT_ALERT_RULES)} default alert rules")
