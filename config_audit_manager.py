"""
Configuration Change Logging and Monitoring for AirTrace RU Backend

Implements comprehensive configuration change tracking, performance impact monitoring,
and audit trail functionality for all configuration modifications.

Requirements: 10.7
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Deque, Union, Callable
from threading import Lock
from pathlib import Path
import hashlib
import copy

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """Aggregated performance statistics (local copy to avoid import issues)"""
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


@dataclass
class ConfigurationChange:
    """Record of a configuration change"""
    change_id: str
    timestamp: datetime
    component: str  # e.g., 'redis', 'cache', 'rate_limiting'
    setting_path: str  # e.g., 'redis.max_connections'
    old_value: Any
    new_value: Any
    change_type: str  # 'update', 'create', 'delete'
    source: str  # 'environment', 'runtime', 'config_file'
    user_context: Optional[str] = None
    reason: Optional[str] = None
    validation_status: str = 'pending'  # 'pending', 'valid', 'invalid'
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class PerformanceImpact:
    """Performance impact measurement for configuration changes"""
    change_id: str
    measurement_start: datetime
    measurement_end: Optional[datetime] = None
    baseline_stats: Optional[PerformanceStats] = None
    post_change_stats: Optional[PerformanceStats] = None
    impact_metrics: Dict[str, float] = field(default_factory=dict)
    impact_severity: str = 'unknown'  # 'none', 'low', 'medium', 'high', 'critical'
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ConfigurationSnapshot:
    """Complete configuration snapshot at a point in time"""
    snapshot_id: str
    timestamp: datetime
    configuration: Dict[str, Any]
    checksum: str
    performance_baseline: Optional[PerformanceStats] = None
    system_health: Dict[str, str] = field(default_factory=dict)


@dataclass
class AuditTrailEntry:
    """Audit trail entry combining change and impact data"""
    entry_id: str
    timestamp: datetime
    change: ConfigurationChange
    performance_impact: Optional[PerformanceImpact] = None
    rollback_available: bool = False
    rollback_snapshot_id: Optional[str] = None


class ConfigurationAuditManager:
    """
    Comprehensive configuration change logging and monitoring system.
    
    Features:
    - Configuration change logging with detailed audit trail
    - Performance impact tracking for configuration changes
    - Configuration snapshots for rollback capability
    - Automated validation and recommendations
    - Privacy-compliant logging (no sensitive data)
    """
    
    def __init__(self, audit_log_path: str = "logs/config_audit.log",
                 max_audit_entries: int = 10000,
                 performance_measurement_duration: int = 300):  # 5 minutes
        """
        Initialize configuration audit manager.
        
        Args:
            audit_log_path: Path to audit log file
            max_audit_entries: Maximum audit entries to keep in memory
            performance_measurement_duration: Duration to measure performance impact (seconds)
        """
        self.audit_log_path = Path(audit_log_path)
        self.max_audit_entries = max_audit_entries
        self.performance_measurement_duration = performance_measurement_duration
        
        # Ensure log directory exists
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe storage
        self._lock = Lock()
        
        # In-memory storage for recent changes
        self.audit_trail: Deque[AuditTrailEntry] = deque(maxlen=max_audit_entries)
        self.configuration_snapshots: Dict[str, ConfigurationSnapshot] = {}
        self.active_performance_measurements: Dict[str, PerformanceImpact] = {}
        
        # Configuration change callbacks
        self.change_callbacks: List[Callable[[ConfigurationChange], None]] = []
        self.impact_callbacks: List[Callable[[PerformanceImpact], None]] = []
        
        # Performance monitor integration
        try:
            from performance_monitor import get_performance_monitor
            self.performance_monitor = get_performance_monitor()
        except Exception as e:
            logger.warning(f"Failed to initialize performance monitor: {e}")
            # Create a mock performance monitor for basic functionality
            self.performance_monitor = None
        
        # Configuration validation rules
        self.validation_rules: Dict[str, Callable[[Any, Any], List[str]]] = {}
        
        # Initialize with current configuration snapshot
        self._initialize_baseline()
        
        logger.info(f"Configuration audit manager initialized with log path: {audit_log_path}")
    
    def _initialize_baseline(self) -> None:
        """Initialize baseline configuration snapshot"""
        try:
            # Skip baseline initialization during import to avoid circular imports
            # The baseline will be created when first needed
            logger.info("Configuration audit manager initialized - baseline will be created on first use")
            
        except Exception as e:
            logger.error(f"Failed to initialize baseline configuration snapshot: {e}")
    
    def log_configuration_change(self, component: str, setting_path: str,
                                old_value: Any, new_value: Any,
                                change_type: str = 'update',
                                source: str = 'runtime',
                                user_context: Optional[str] = None,
                                reason: Optional[str] = None) -> str:
        """
        Log a configuration change and start performance impact tracking.
        
        Args:
            component: Configuration component name
            setting_path: Dot-notation path to the setting
            old_value: Previous value
            new_value: New value
            change_type: Type of change ('update', 'create', 'delete')
            source: Source of change ('environment', 'runtime', 'config_file')
            user_context: User or system context making the change
            reason: Reason for the change
            
        Returns:
            str: Change ID for tracking
        """
        try:
            # Generate unique change ID
            change_id = self._generate_change_id(component, setting_path)
            
            # Create configuration change record
            change = ConfigurationChange(
                change_id=change_id,
                timestamp=datetime.now(timezone.utc),
                component=component,
                setting_path=setting_path,
                old_value=self._sanitize_value(old_value),
                new_value=self._sanitize_value(new_value),
                change_type=change_type,
                source=source,
                user_context=user_context,
                reason=reason
            )
            
            # Validate the change
            self._validate_configuration_change(change)
            
            # Start performance impact measurement
            self._start_performance_measurement(change_id)
            
            # Create audit trail entry
            audit_entry = AuditTrailEntry(
                entry_id=f"audit_{change_id}",
                timestamp=change.timestamp,
                change=change,
                rollback_available=self._can_rollback(component, setting_path)
            )
            
            # Store in memory and log to file
            with self._lock:
                self.audit_trail.append(audit_entry)
            
            self._write_audit_log(audit_entry)
            
            # Trigger callbacks
            for callback in self.change_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Error in configuration change callback: {e}")
            
            logger.info(f"Configuration change logged: {component}.{setting_path} "
                       f"changed from {old_value} to {new_value} (ID: {change_id})")
            
            return change_id
            
        except Exception as e:
            logger.error(f"Failed to log configuration change: {e}")
            return ""
    
    def complete_performance_measurement(self, change_id: str) -> Optional[PerformanceImpact]:
        """
        Complete performance impact measurement for a configuration change.
        
        Args:
            change_id: ID of the configuration change
            
        Returns:
            PerformanceImpact: Performance impact data if measurement exists
        """
        try:
            with self._lock:
                if change_id not in self.active_performance_measurements:
                    logger.warning(f"No active performance measurement for change ID: {change_id}")
                    return None
                
                impact = self.active_performance_measurements[change_id]
            
            # Get post-change performance statistics
            impact.measurement_end = datetime.now(timezone.utc)
            if self.performance_monitor:
                impact.post_change_stats = self.performance_monitor.get_performance_stats(
                    timedelta(seconds=self.performance_measurement_duration)
                )
            
            # Calculate impact metrics
            self._calculate_impact_metrics(impact)
            
            # Determine impact severity and generate recommendations
            self._assess_impact_severity(impact)
            self._generate_impact_recommendations(impact)
            
            # Update audit trail entry with performance impact
            self._update_audit_entry_with_impact(change_id, impact)
            
            # Remove from active measurements
            with self._lock:
                del self.active_performance_measurements[change_id]
            
            # Trigger impact callbacks
            for callback in self.impact_callbacks:
                try:
                    callback(impact)
                except Exception as e:
                    logger.error(f"Error in performance impact callback: {e}")
            
            logger.info(f"Performance measurement completed for change {change_id}: "
                       f"severity={impact.impact_severity}")
            
            return impact
            
        except Exception as e:
            logger.error(f"Failed to complete performance measurement: {e}")
            return None
    
    def create_configuration_snapshot(self, snapshot_id: Optional[str] = None) -> str:
        """
        Create a configuration snapshot for rollback purposes.
        
        Args:
            snapshot_id: Optional custom snapshot ID
            
        Returns:
            str: Snapshot ID
        """
        try:
            from config import config
            current_config = self._extract_configuration_dict(config)
            
            snapshot = self._create_configuration_snapshot(current_config, snapshot_id)
            
            with self._lock:
                self.configuration_snapshots[snapshot.snapshot_id] = snapshot
            
            logger.info(f"Configuration snapshot created: {snapshot.snapshot_id}")
            return snapshot.snapshot_id
            
        except Exception as e:
            logger.error(f"Failed to create configuration snapshot: {e}")
            return ""
    
    def get_audit_trail(self, component: Optional[str] = None,
                       time_range: Optional[timedelta] = None,
                       limit: Optional[int] = None) -> List[AuditTrailEntry]:
        """
        Get audit trail entries with optional filtering.
        
        Args:
            component: Filter by component name
            time_range: Filter by time range from now
            limit: Maximum number of entries to return
            
        Returns:
            List[AuditTrailEntry]: Filtered audit trail entries
        """
        try:
            with self._lock:
                entries = list(self.audit_trail)
            
            # Apply filters
            if component:
                entries = [e for e in entries if e.change.component == component]
            
            if time_range:
                cutoff_time = datetime.now(timezone.utc) - time_range
                entries = [e for e in entries if e.timestamp >= cutoff_time]
            
            # Sort by timestamp (most recent first)
            entries.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            if limit:
                entries = entries[:limit]
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to get audit trail: {e}")
            return []
    
    def get_performance_impact_summary(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get summary of performance impacts from configuration changes.
        
        Args:
            time_range: Time range to analyze
            
        Returns:
            Dict with performance impact summary
        """
        try:
            entries = self.get_audit_trail(time_range=time_range)
            
            # Filter entries with performance impact data
            impact_entries = [e for e in entries if e.performance_impact is not None]
            
            if not impact_entries:
                return {
                    'total_changes': len(entries),
                    'changes_with_impact_data': 0,
                    'impact_summary': {}
                }
            
            # Analyze impact severity distribution
            severity_counts = defaultdict(int)
            component_impacts = defaultdict(list)
            
            for entry in impact_entries:
                impact = entry.performance_impact
                severity_counts[impact.impact_severity] += 1
                component_impacts[entry.change.component].append(impact)
            
            # Calculate average impact metrics by component
            component_summary = {}
            for component, impacts in component_impacts.items():
                if impacts:
                    avg_metrics = {}
                    for metric_name in impacts[0].impact_metrics.keys():
                        values = [i.impact_metrics.get(metric_name, 0.0) for i in impacts]
                        avg_metrics[metric_name] = sum(values) / len(values)
                    
                    component_summary[component] = {
                        'change_count': len(impacts),
                        'average_impact_metrics': avg_metrics,
                        'severity_distribution': {
                            severity: sum(1 for i in impacts if i.impact_severity == severity)
                            for severity in ['none', 'low', 'medium', 'high', 'critical']
                        }
                    }
            
            return {
                'total_changes': len(entries),
                'changes_with_impact_data': len(impact_entries),
                'severity_distribution': dict(severity_counts),
                'component_summary': component_summary,
                'time_range_analyzed': str(time_range) if time_range else 'all_time'
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance impact summary: {e}")
            return {'error': str(e)}
    
    def add_validation_rule(self, setting_path: str, 
                           validator: Callable[[Any, Any], List[str]]) -> None:
        """
        Add custom validation rule for configuration changes.
        
        Args:
            setting_path: Dot-notation path to setting
            validator: Function that returns list of validation errors
        """
        self.validation_rules[setting_path] = validator
        logger.info(f"Validation rule added for {setting_path}")
    
    def add_change_callback(self, callback: Callable[[ConfigurationChange], None]) -> None:
        """Add callback for configuration changes"""
        self.change_callbacks.append(callback)
        logger.info("Configuration change callback added")
    
    def add_impact_callback(self, callback: Callable[[PerformanceImpact], None]) -> None:
        """Add callback for performance impact measurements"""
        self.impact_callbacks.append(callback)
        logger.info("Performance impact callback added")
    
    def export_audit_trail(self, output_path: str, format: str = 'json') -> bool:
        """
        Export audit trail to file.
        
        Args:
            output_path: Output file path
            format: Export format ('json', 'csv')
            
        Returns:
            bool: Success status
        """
        try:
            entries = self.get_audit_trail()
            
            if format.lower() == 'json':
                export_data = []
                for entry in entries:
                    entry_dict = asdict(entry)
                    # Convert datetime objects to ISO strings
                    entry_dict['timestamp'] = entry.timestamp.isoformat()
                    entry_dict['change']['timestamp'] = entry.change.timestamp.isoformat()
                    
                    if entry.performance_impact:
                        entry_dict['performance_impact']['measurement_start'] = \
                            entry.performance_impact.measurement_start.isoformat()
                        if entry.performance_impact.measurement_end:
                            entry_dict['performance_impact']['measurement_end'] = \
                                entry.performance_impact.measurement_end.isoformat()
                    
                    export_data.append(entry_dict)
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                import csv
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        'timestamp', 'change_id', 'component', 'setting_path',
                        'old_value', 'new_value', 'change_type', 'source',
                        'validation_status', 'impact_severity', 'rollback_available'
                    ])
                    
                    # Write data
                    for entry in entries:
                        writer.writerow([
                            entry.timestamp.isoformat(),
                            entry.change.change_id,
                            entry.change.component,
                            entry.change.setting_path,
                            str(entry.change.old_value),
                            str(entry.change.new_value),
                            entry.change.change_type,
                            entry.change.source,
                            entry.change.validation_status,
                            entry.performance_impact.impact_severity if entry.performance_impact else 'unknown',
                            entry.rollback_available
                        ])
            
            logger.info(f"Audit trail exported to {output_path} in {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export audit trail: {e}")
            return False
    
    def cleanup_old_data(self, older_than: timedelta) -> int:
        """
        Clean up old audit data and snapshots.
        
        Args:
            older_than: Age threshold for cleanup
            
        Returns:
            int: Number of items cleaned up
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - older_than
            cleanup_count = 0
            
            with self._lock:
                # Clean up old audit trail entries
                original_len = len(self.audit_trail)
                self.audit_trail = deque(
                    (entry for entry in self.audit_trail if entry.timestamp >= cutoff_time),
                    maxlen=self.max_audit_entries
                )
                cleanup_count += original_len - len(self.audit_trail)
                
                # Clean up old configuration snapshots
                old_snapshots = [
                    snapshot_id for snapshot_id, snapshot in self.configuration_snapshots.items()
                    if snapshot.timestamp < cutoff_time and snapshot_id != 'baseline_initialization'
                ]
                
                for snapshot_id in old_snapshots:
                    del self.configuration_snapshots[snapshot_id]
                    cleanup_count += 1
            
            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} old audit data items")
            
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old audit data: {e}")
            return 0
    
    def _extract_configuration_dict(self, config_obj) -> Dict[str, Any]:
        """Extract configuration as dictionary, sanitizing sensitive values"""
        try:
            config_dict = {}
            
            # Extract configuration from config object
            for attr_name in dir(config_obj):
                if not attr_name.startswith('_'):
                    attr_value = getattr(config_obj, attr_name)
                    if hasattr(attr_value, '__dict__'):
                        # Handle nested configuration objects
                        nested_dict = {}
                        for nested_attr in dir(attr_value):
                            if not nested_attr.startswith('_') and not callable(getattr(attr_value, nested_attr)):
                                nested_value = getattr(attr_value, nested_attr)
                                nested_dict[nested_attr] = self._sanitize_value(nested_value)
                        config_dict[attr_name] = nested_dict
                    elif not callable(attr_value):
                        config_dict[attr_name] = self._sanitize_value(attr_value)
            
            return config_dict
            
        except Exception as e:
            logger.error(f"Failed to extract configuration dictionary: {e}")
            return {}
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize configuration values to remove sensitive information"""
        if isinstance(value, str):
            # Sanitize potential passwords, API keys, etc.
            if any(keyword in value.lower() for keyword in ['password', 'key', 'secret', 'token']):
                if len(value) > 8:
                    return f"{value[:4]}***{value[-4:]}"
                else:
                    return "***"
            
            # Sanitize URLs with credentials
            if '://' in value and '@' in value:
                parts = value.split('@')
                if len(parts) == 2:
                    return f"{parts[0].split('://')[0]}://***@{parts[1]}"
        
        return value
    
    def _generate_change_id(self, component: str, setting_path: str) -> str:
        """Generate unique change ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"{component}_{setting_path}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _validate_configuration_change(self, change: ConfigurationChange) -> None:
        """Validate configuration change using registered rules"""
        try:
            errors = []
            
            # Check if there's a specific validation rule for this setting
            full_path = f"{change.component}.{change.setting_path}"
            if full_path in self.validation_rules:
                validator = self.validation_rules[full_path]
                validation_errors = validator(change.old_value, change.new_value)
                errors.extend(validation_errors)
            
            # Also check for just the setting path
            if change.setting_path in self.validation_rules:
                validator = self.validation_rules[change.setting_path]
                validation_errors = validator(change.old_value, change.new_value)
                errors.extend(validation_errors)
            
            # General validation rules
            if change.new_value is None and change.change_type != 'delete':
                errors.append("New value cannot be None for non-delete operations")
            
            # Update change record with validation results
            change.validation_errors = errors
            change.validation_status = 'invalid' if errors else 'valid'
            
            if errors:
                logger.warning(f"Configuration change validation failed: {errors}")
            
        except Exception as e:
            logger.error(f"Error validating configuration change: {e}")
            change.validation_status = 'error'
            change.validation_errors = [f"Validation error: {str(e)}"]
    
    def _start_performance_measurement(self, change_id: str) -> None:
        """Start performance impact measurement"""
        try:
            # Get baseline performance statistics if performance monitor is available
            baseline_stats = None
            if self.performance_monitor:
                baseline_stats = self.performance_monitor.get_performance_stats(
                    timedelta(seconds=self.performance_measurement_duration)
                )
            
            impact = PerformanceImpact(
                change_id=change_id,
                measurement_start=datetime.now(timezone.utc),
                baseline_stats=baseline_stats
            )
            
            with self._lock:
                self.active_performance_measurements[change_id] = impact
            
            # Schedule automatic completion only if we have an event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._auto_complete_measurement(change_id))
            except RuntimeError:
                # No event loop running, skip automatic completion
                logger.debug("No event loop running, skipping automatic performance measurement completion")
            
        except Exception as e:
            logger.error(f"Failed to start performance measurement: {e}")
    
    async def _auto_complete_measurement(self, change_id: str) -> None:
        """Automatically complete performance measurement after duration"""
        try:
            await asyncio.sleep(self.performance_measurement_duration)
            self.complete_performance_measurement(change_id)
        except Exception as e:
            logger.error(f"Error in auto-complete measurement: {e}")
    
    def _calculate_impact_metrics(self, impact: PerformanceImpact) -> None:
        """Calculate performance impact metrics"""
        try:
            if not impact.baseline_stats or not impact.post_change_stats:
                return
            
            baseline = impact.baseline_stats
            post_change = impact.post_change_stats
            
            # Calculate percentage changes
            metrics = {}
            
            if baseline.avg_response_time > 0:
                metrics['response_time_change_pct'] = (
                    (post_change.avg_response_time - baseline.avg_response_time) / 
                    baseline.avg_response_time * 100
                )
            
            if baseline.p95_response_time > 0:
                metrics['p95_response_time_change_pct'] = (
                    (post_change.p95_response_time - baseline.p95_response_time) / 
                    baseline.p95_response_time * 100
                )
            
            metrics['cache_hit_rate_change_pct'] = (
                post_change.cache_hit_rate - baseline.cache_hit_rate
            ) * 100
            
            metrics['error_rate_change_pct'] = (
                post_change.error_rate - baseline.error_rate
            ) * 100
            
            if baseline.external_api_avg_latency > 0:
                metrics['external_api_latency_change_pct'] = (
                    (post_change.external_api_avg_latency - baseline.external_api_avg_latency) / 
                    baseline.external_api_avg_latency * 100
                )
            
            impact.impact_metrics = metrics
            
        except Exception as e:
            logger.error(f"Error calculating impact metrics: {e}")
    
    def _assess_impact_severity(self, impact: PerformanceImpact) -> None:
        """Assess the severity of performance impact"""
        try:
            metrics = impact.impact_metrics
            
            # Define severity thresholds
            critical_threshold = 50.0  # 50% degradation
            high_threshold = 25.0      # 25% degradation
            medium_threshold = 10.0    # 10% degradation
            low_threshold = 5.0        # 5% degradation
            
            max_degradation = 0.0
            
            # Check response time degradation
            response_time_change = metrics.get('response_time_change_pct', 0.0)
            if response_time_change > 0:
                max_degradation = max(max_degradation, response_time_change)
            
            # Check P95 response time degradation
            p95_change = metrics.get('p95_response_time_change_pct', 0.0)
            if p95_change > 0:
                max_degradation = max(max_degradation, p95_change)
            
            # Check error rate increase
            error_rate_change = metrics.get('error_rate_change_pct', 0.0)
            if error_rate_change > 0:
                max_degradation = max(max_degradation, error_rate_change * 10)  # Weight errors more heavily
            
            # Check external API latency degradation
            api_latency_change = metrics.get('external_api_latency_change_pct', 0.0)
            if api_latency_change > 0:
                max_degradation = max(max_degradation, api_latency_change)
            
            # Determine severity
            if max_degradation >= critical_threshold:
                impact.impact_severity = 'critical'
            elif max_degradation >= high_threshold:
                impact.impact_severity = 'high'
            elif max_degradation >= medium_threshold:
                impact.impact_severity = 'medium'
            elif max_degradation >= low_threshold:
                impact.impact_severity = 'low'
            else:
                impact.impact_severity = 'none'
            
        except Exception as e:
            logger.error(f"Error assessing impact severity: {e}")
            impact.impact_severity = 'unknown'
    
    def _generate_impact_recommendations(self, impact: PerformanceImpact) -> None:
        """Generate recommendations based on performance impact"""
        try:
            recommendations = []
            metrics = impact.impact_metrics
            
            # Response time recommendations
            response_time_change = metrics.get('response_time_change_pct', 0.0)
            if response_time_change > 10.0:
                recommendations.append(
                    f"Response time increased by {response_time_change:.1f}%. "
                    "Consider reverting the change or optimizing the affected component."
                )
            
            # Cache hit rate recommendations
            cache_hit_change = metrics.get('cache_hit_rate_change_pct', 0.0)
            if cache_hit_change < -5.0:
                recommendations.append(
                    f"Cache hit rate decreased by {abs(cache_hit_change):.1f}%. "
                    "Review cache configuration and TTL settings."
                )
            
            # Error rate recommendations
            error_rate_change = metrics.get('error_rate_change_pct', 0.0)
            if error_rate_change > 1.0:
                recommendations.append(
                    f"Error rate increased by {error_rate_change:.1f}%. "
                    "Monitor error logs and consider immediate rollback if critical."
                )
            
            # External API latency recommendations
            api_latency_change = metrics.get('external_api_latency_change_pct', 0.0)
            if api_latency_change > 15.0:
                recommendations.append(
                    f"External API latency increased by {api_latency_change:.1f}%. "
                    "Check connection pool settings and network configuration."
                )
            
            # General recommendations based on severity
            if impact.impact_severity == 'critical':
                recommendations.append(
                    "CRITICAL: Immediate attention required. Consider emergency rollback."
                )
            elif impact.impact_severity == 'high':
                recommendations.append(
                    "HIGH: Significant performance degradation detected. Review change immediately."
                )
            elif impact.impact_severity == 'medium':
                recommendations.append(
                    "MEDIUM: Moderate performance impact. Monitor closely and consider optimization."
                )
            
            impact.recommendations = recommendations
            
        except Exception as e:
            logger.error(f"Error generating impact recommendations: {e}")
    
    def _create_configuration_snapshot(self, config_dict: Dict[str, Any], 
                                     snapshot_id: Optional[str] = None) -> ConfigurationSnapshot:
        """Create configuration snapshot"""
        if snapshot_id is None:
            snapshot_id = f"snapshot_{int(time.time())}"
        
        # Calculate checksum
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        checksum = hashlib.sha256(config_json.encode()).hexdigest()
        
        # Get current performance baseline
        performance_baseline = self.performance_monitor.get_performance_stats()
        
        return ConfigurationSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(timezone.utc),
            configuration=config_dict,
            checksum=checksum,
            performance_baseline=performance_baseline
        )
    
    def _can_rollback(self, component: str, setting_path: str) -> bool:
        """Check if rollback is available for a configuration change"""
        # For now, assume rollback is available for runtime changes
        # This could be enhanced with more sophisticated rollback logic
        return True
    
    def _update_audit_entry_with_impact(self, change_id: str, impact: PerformanceImpact) -> None:
        """Update audit trail entry with performance impact data"""
        try:
            with self._lock:
                for entry in self.audit_trail:
                    if entry.change.change_id == change_id:
                        entry.performance_impact = impact
                        break
            
            # Also write updated entry to log
            self._write_impact_log(impact)
            
        except Exception as e:
            logger.error(f"Error updating audit entry with impact: {e}")
    
    def _write_audit_log(self, entry: AuditTrailEntry) -> None:
        """Write audit trail entry to log file"""
        try:
            log_data = {
                'timestamp': entry.timestamp.isoformat(),
                'entry_id': entry.entry_id,
                'change': {
                    'change_id': entry.change.change_id,
                    'component': entry.change.component,
                    'setting_path': entry.change.setting_path,
                    'old_value': entry.change.old_value,
                    'new_value': entry.change.new_value,
                    'change_type': entry.change.change_type,
                    'source': entry.change.source,
                    'validation_status': entry.change.validation_status,
                    'validation_errors': entry.change.validation_errors
                },
                'rollback_available': entry.rollback_available
            }
            
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
                
        except Exception as e:
            logger.error(f"Error writing audit log: {e}")
    
    def _write_impact_log(self, impact: PerformanceImpact) -> None:
        """Write performance impact data to log file"""
        try:
            log_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'type': 'performance_impact',
                'change_id': impact.change_id,
                'impact_severity': impact.impact_severity,
                'impact_metrics': impact.impact_metrics,
                'recommendations': impact.recommendations
            }
            
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
                
        except Exception as e:
            logger.error(f"Error writing impact log: {e}")


# Global configuration audit manager instance
_config_audit_manager: Optional[ConfigurationAuditManager] = None


def get_config_audit_manager() -> ConfigurationAuditManager:
    """Get global configuration audit manager instance"""
    global _config_audit_manager
    if _config_audit_manager is None:
        _config_audit_manager = ConfigurationAuditManager()
    return _config_audit_manager


def setup_configuration_audit(audit_log_path: str = "logs/config_audit.log",
                             max_audit_entries: int = 10000,
                             performance_measurement_duration: int = 300) -> ConfigurationAuditManager:
    """
    Setup global configuration audit manager.
    
    Args:
        audit_log_path: Path to audit log file
        max_audit_entries: Maximum audit entries to keep in memory
        performance_measurement_duration: Duration to measure performance impact
        
    Returns:
        ConfigurationAuditManager instance
    """
    global _config_audit_manager
    _config_audit_manager = ConfigurationAuditManager(
        audit_log_path=audit_log_path,
        max_audit_entries=max_audit_entries,
        performance_measurement_duration=performance_measurement_duration
    )
    logger.info("Configuration audit manager setup completed")
    return _config_audit_manager