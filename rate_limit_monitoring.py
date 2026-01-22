"""
Rate limiting monitoring and logging system

Implements comprehensive monitoring, logging, and configuration validation
for the rate limiting system with privacy-safe logging.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque

from rate_limit_types import RateLimitConfig, EndpointCategory, RateLimitStrategy, RateLimitMetrics

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of rate limit violations"""
    BURST_EXCEEDED = "burst_exceeded"
    SUSTAINED_ABUSE = "sustained_abuse"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    CONFIGURATION_ERROR = "configuration_error"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class RateLimitViolation:
    """Rate limit violation record"""
    timestamp: datetime
    identifier_hash: str  # Privacy-safe hashed identifier
    endpoint: str
    violation_type: ViolationType
    current_usage: int
    limit: int
    window_start: datetime
    window_end: datetime
    user_agent_hash: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "identifier_hash": self.identifier_hash,
            "endpoint": self.endpoint,
            "violation_type": self.violation_type.value,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "user_agent_hash": self.user_agent_hash,
            "additional_info": self.additional_info
        }


@dataclass
class RateLimitAlert:
    """Rate limiting alert"""
    timestamp: datetime
    level: AlertLevel
    message: str
    endpoint: Optional[str] = None
    violation_count: int = 0
    time_window: Optional[timedelta] = None
    suggested_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "endpoint": self.endpoint,
            "violation_count": self.violation_count,
            "time_window_seconds": self.time_window.total_seconds() if self.time_window else None,
            "suggested_action": self.suggested_action
        }


@dataclass
class EndpointStats:
    """Statistics for a specific endpoint"""
    endpoint: str
    total_requests: int = 0
    allowed_requests: int = 0
    blocked_requests: int = 0
    unique_clients: int = 0
    peak_requests_per_minute: int = 0
    average_requests_per_minute: float = 0.0
    last_violation: Optional[datetime] = None
    violation_count: int = 0
    
    @property
    def block_rate(self) -> float:
        """Calculate block rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.blocked_requests / self.total_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "endpoint": self.endpoint,
            "total_requests": self.total_requests,
            "allowed_requests": self.allowed_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": self.block_rate,
            "unique_clients": self.unique_clients,
            "peak_requests_per_minute": self.peak_requests_per_minute,
            "average_requests_per_minute": self.average_requests_per_minute,
            "last_violation": self.last_violation.isoformat() if self.last_violation else None,
            "violation_count": self.violation_count
        }


class RateLimitConfigValidator:
    """Validates rate limiting configurations"""
    
    @staticmethod
    def validate_config(config: RateLimitConfig) -> List[str]:
        """
        Validate rate limit configuration.
        
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        # Validate requests per minute
        if config.requests_per_minute <= 0:
            errors.append("requests_per_minute must be positive")
        elif config.requests_per_minute > 10000:
            errors.append("requests_per_minute seems unusually high (>10000)")
        
        # Validate burst multiplier
        if config.burst_multiplier < 1.0:
            errors.append("burst_multiplier must be >= 1.0")
        elif config.burst_multiplier > 10.0:
            errors.append("burst_multiplier seems unusually high (>10.0)")
        
        # Validate window size
        if config.window_size_seconds <= 0:
            errors.append("window_size_seconds must be positive")
        elif config.window_size_seconds < 10:
            errors.append("window_size_seconds should be at least 10 seconds")
        elif config.window_size_seconds > 3600:
            errors.append("window_size_seconds should not exceed 1 hour")
        
        # Validate strategy
        if not isinstance(config.identifier_strategy, RateLimitStrategy):
            errors.append("identifier_strategy must be a valid RateLimitStrategy")
        
        # Check for reasonable burst limits
        burst_limit = int(config.requests_per_minute * config.burst_multiplier)
        if burst_limit > 1000:
            errors.append(f"Burst limit ({burst_limit}) seems unusually high")
        
        return errors
    
    @staticmethod
    def validate_endpoint_configs(configs: Dict[EndpointCategory, RateLimitConfig]) -> Dict[str, List[str]]:
        """
        Validate multiple endpoint configurations.
        
        Returns dictionary mapping endpoint names to validation errors.
        """
        validation_results = {}
        
        for category, config in configs.items():
            errors = RateLimitConfigValidator.validate_config(config)
            if errors:
                validation_results[category.value] = errors
        
        # Cross-validation checks
        if len(configs) > 1:
            # Check for inconsistent window sizes
            window_sizes = {config.window_size_seconds for config in configs.values()}
            if len(window_sizes) > 1:
                validation_results["cross_validation"] = validation_results.get("cross_validation", [])
                validation_results["cross_validation"].append(
                    "Different window sizes across endpoints may cause confusion"
                )
            
            # Check for extreme differences in limits
            limits = [config.requests_per_minute for config in configs.values()]
            if max(limits) / min(limits) > 100:
                validation_results["cross_validation"] = validation_results.get("cross_validation", [])
                validation_results["cross_validation"].append(
                    "Extreme differences in rate limits across endpoints detected"
                )
        
        return validation_results


class RateLimitMonitor:
    """
    Comprehensive rate limiting monitoring system.
    
    Tracks violations, generates alerts, collects metrics, and provides
    detailed analytics for rate limiting performance.
    """
    
    def __init__(self, alert_callback: Optional[Callable[[RateLimitAlert], None]] = None):
        # Violation tracking
        self._violations: deque = deque(maxlen=10000)  # Keep last 10k violations
        self._violation_counts: Dict[str, int] = defaultdict(int)
        
        # Endpoint statistics
        self._endpoint_stats: Dict[str, EndpointStats] = {}
        
        # Time-based metrics
        self._request_timeline: deque = deque(maxlen=1000)  # Last 1000 requests
        self._violation_timeline: deque = deque(maxlen=1000)  # Last 1000 violations
        
        # Alert system
        self._alert_callback = alert_callback
        self._alert_thresholds = {
            "violation_rate_per_minute": 10,
            "block_rate_threshold": 50.0,  # 50% block rate
            "sustained_violations_threshold": 5,  # 5 violations from same client
            "endpoint_abuse_threshold": 100  # 100 violations per endpoint per hour
        }
        
        # Monitoring state
        self._monitoring_start_time = datetime.now(timezone.utc)
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour
        
        # Privacy settings
        self._log_detailed_violations = True
        self._anonymize_identifiers = True
        
        logger.info("Rate limit monitor initialized")
    
    def record_violation(
        self,
        identifier_hash: str,
        endpoint: str,
        violation_type: ViolationType,
        current_usage: int,
        limit: int,
        window_start: datetime,
        window_end: datetime,
        user_agent_hash: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """Record a rate limit violation"""
        violation = RateLimitViolation(
            timestamp=datetime.now(timezone.utc),
            identifier_hash=identifier_hash,
            endpoint=endpoint,
            violation_type=violation_type,
            current_usage=current_usage,
            limit=limit,
            window_start=window_start,
            window_end=window_end,
            user_agent_hash=user_agent_hash,
            additional_info=additional_info or {}
        )
        
        # Store violation
        self._violations.append(violation)
        self._violation_counts[identifier_hash] += 1
        
        # Update endpoint statistics
        self._update_endpoint_stats(endpoint, blocked=True)
        
        # Log violation (privacy-safe)
        if self._log_detailed_violations:
            logger.warning(
                f"Rate limit violation - "
                f"Endpoint: {endpoint}, "
                f"Type: {violation_type.value}, "
                f"Usage: {current_usage}/{limit}, "
                f"Client: {identifier_hash[:8]}..."
            )
        
        # Check for alert conditions
        self._check_alert_conditions(violation)
        
        # Periodic cleanup
        self._periodic_cleanup()
    
    def record_request(self, endpoint: str, allowed: bool, identifier_hash: str):
        """Record a request (allowed or blocked)"""
        timestamp = datetime.now(timezone.utc)
        
        # Update timeline
        self._request_timeline.append({
            "timestamp": timestamp,
            "endpoint": endpoint,
            "allowed": allowed,
            "identifier_hash": identifier_hash
        })
        
        # Update endpoint statistics
        self._update_endpoint_stats(endpoint, blocked=not allowed)
    
    def _update_endpoint_stats(self, endpoint: str, blocked: bool):
        """Update statistics for an endpoint"""
        if endpoint not in self._endpoint_stats:
            self._endpoint_stats[endpoint] = EndpointStats(endpoint=endpoint)
        
        stats = self._endpoint_stats[endpoint]
        stats.total_requests += 1
        
        if blocked:
            stats.blocked_requests += 1
            stats.last_violation = datetime.now(timezone.utc)
            stats.violation_count += 1
        else:
            stats.allowed_requests += 1
        
        # Update peak requests per minute (simplified calculation)
        current_time = time.time()
        recent_requests = sum(
            1 for req in self._request_timeline
            if req["endpoint"] == endpoint and 
            (current_time - req["timestamp"].timestamp()) < 60
        )
        stats.peak_requests_per_minute = max(stats.peak_requests_per_minute, recent_requests)
    
    def _check_alert_conditions(self, violation: RateLimitViolation):
        """Check if violation triggers any alerts"""
        current_time = datetime.now(timezone.utc)
        
        # Check violation rate
        recent_violations = sum(
            1 for v in self._violations
            if (current_time - v.timestamp).total_seconds() < 60
        )
        
        if recent_violations >= self._alert_thresholds["violation_rate_per_minute"]:
            alert = RateLimitAlert(
                timestamp=current_time,
                level=AlertLevel.WARNING,
                message=f"High violation rate: {recent_violations} violations in last minute",
                violation_count=recent_violations,
                time_window=timedelta(minutes=1),
                suggested_action="Review rate limit configurations or investigate potential abuse"
            )
            self._send_alert(alert)
        
        # Check sustained violations from same client
        client_violations = sum(
            1 for v in self._violations
            if v.identifier_hash == violation.identifier_hash and
            (current_time - v.timestamp).total_seconds() < 300  # 5 minutes
        )
        
        if client_violations >= self._alert_thresholds["sustained_violations_threshold"]:
            alert = RateLimitAlert(
                timestamp=current_time,
                level=AlertLevel.ERROR,
                message=f"Sustained violations from client {violation.identifier_hash[:8]}...: {client_violations} in 5 minutes",
                violation_count=client_violations,
                time_window=timedelta(minutes=5),
                suggested_action="Consider blocking or investigating this client"
            )
            self._send_alert(alert)
        
        # Check endpoint abuse
        endpoint_violations = sum(
            1 for v in self._violations
            if v.endpoint == violation.endpoint and
            (current_time - v.timestamp).total_seconds() < 3600  # 1 hour
        )
        
        if endpoint_violations >= self._alert_thresholds["endpoint_abuse_threshold"]:
            alert = RateLimitAlert(
                timestamp=current_time,
                level=AlertLevel.CRITICAL,
                message=f"Endpoint under heavy abuse: {violation.endpoint} - {endpoint_violations} violations in last hour",
                endpoint=violation.endpoint,
                violation_count=endpoint_violations,
                time_window=timedelta(hours=1),
                suggested_action="Consider tightening rate limits for this endpoint or implementing additional protection"
            )
            self._send_alert(alert)
    
    def _send_alert(self, alert: RateLimitAlert):
        """Send alert through configured callback"""
        try:
            # Log alert
            logger.log(
                logging.WARNING if alert.level in [AlertLevel.WARNING, AlertLevel.INFO] else logging.ERROR,
                f"Rate limit alert [{alert.level.value.upper()}]: {alert.message}"
            )
            
            # Call external alert handler if configured
            if self._alert_callback:
                self._alert_callback(alert)
                
        except Exception as e:
            logger.error(f"Failed to send rate limit alert: {e}")
    
    def _periodic_cleanup(self):
        """Periodic cleanup of old data"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        # Clean up old violation counts
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        # Remove old violations from counts
        for identifier in list(self._violation_counts.keys()):
            recent_violations = sum(
                1 for v in self._violations
                if v.identifier_hash == identifier and v.timestamp > cutoff_time
            )
            if recent_violations == 0:
                del self._violation_counts[identifier]
            else:
                self._violation_counts[identifier] = recent_violations
        
        self._last_cleanup = current_time
        logger.debug("Rate limit monitor cleanup completed")
    
    def get_violation_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get summary of violations in specified time window"""
        if time_window is None:
            time_window = timedelta(hours=1)
        
        cutoff_time = datetime.now(timezone.utc) - time_window
        recent_violations = [v for v in self._violations if v.timestamp > cutoff_time]
        
        # Group by violation type
        violation_types = defaultdict(int)
        for violation in recent_violations:
            violation_types[violation.violation_type.value] += 1
        
        # Group by endpoint
        endpoint_violations = defaultdict(int)
        for violation in recent_violations:
            endpoint_violations[violation.endpoint] += 1
        
        # Top violating clients
        client_violations = defaultdict(int)
        for violation in recent_violations:
            client_violations[violation.identifier_hash] += 1
        
        top_clients = sorted(
            client_violations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "time_window_hours": time_window.total_seconds() / 3600,
            "total_violations": len(recent_violations),
            "violation_types": dict(violation_types),
            "endpoint_violations": dict(endpoint_violations),
            "top_violating_clients": [
                {"client_hash": client[:8] + "...", "violations": count}
                for client, count in top_clients
            ],
            "unique_violating_clients": len(client_violations)
        }
    
    def get_endpoint_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive endpoint statistics"""
        return {
            endpoint: stats.to_dict()
            for endpoint, stats in self._endpoint_stats.items()
        }
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics"""
        current_time = datetime.now(timezone.utc)
        uptime = current_time - self._monitoring_start_time
        
        # Calculate overall statistics
        total_requests = sum(stats.total_requests for stats in self._endpoint_stats.values())
        total_blocked = sum(stats.blocked_requests for stats in self._endpoint_stats.values())
        
        return {
            "monitoring_uptime_seconds": uptime.total_seconds(),
            "total_violations_recorded": len(self._violations),
            "total_requests_monitored": total_requests,
            "total_requests_blocked": total_blocked,
            "overall_block_rate": (total_blocked / max(1, total_requests)) * 100,
            "unique_violating_clients": len(self._violation_counts),
            "monitored_endpoints": len(self._endpoint_stats),
            "alert_thresholds": self._alert_thresholds,
            "violation_summary_1h": self.get_violation_summary(timedelta(hours=1)),
            "endpoint_statistics": self.get_endpoint_statistics()
        }
    
    def export_violations(
        self,
        time_window: Optional[timedelta] = None,
        include_sensitive_data: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Export violations for analysis.
        
        Args:
            time_window: Time window to export (None for all)
            include_sensitive_data: Whether to include full identifiers (not recommended)
        """
        if time_window:
            cutoff_time = datetime.now(timezone.utc) - time_window
            violations = [v for v in self._violations if v.timestamp > cutoff_time]
        else:
            violations = list(self._violations)
        
        exported = []
        for violation in violations:
            data = violation.to_dict()
            
            # Anonymize sensitive data if requested
            if not include_sensitive_data:
                if data["identifier_hash"]:
                    data["identifier_hash"] = data["identifier_hash"][:8] + "..."
                if data["user_agent_hash"]:
                    data["user_agent_hash"] = data["user_agent_hash"][:8] + "..."
            
            exported.append(data)
        
        return exported
    
    def configure_alerts(self, thresholds: Dict[str, Any]):
        """Configure alert thresholds"""
        self._alert_thresholds.update(thresholds)
        logger.info(f"Rate limit alert thresholds updated: {thresholds}")
    
    def set_alert_callback(self, callback: Callable[[RateLimitAlert], None]):
        """Set callback function for alerts"""
        self._alert_callback = callback
        logger.info("Rate limit alert callback configured")


# Global monitor instance
_rate_limit_monitor: Optional[RateLimitMonitor] = None


def get_rate_limit_monitor() -> RateLimitMonitor:
    """Get or create global rate limit monitor"""
    global _rate_limit_monitor
    if _rate_limit_monitor is None:
        _rate_limit_monitor = RateLimitMonitor()
    return _rate_limit_monitor


def setup_rate_limit_monitoring(alert_callback: Optional[Callable[[RateLimitAlert], None]] = None) -> RateLimitMonitor:
    """Setup rate limit monitoring with optional alert callback"""
    global _rate_limit_monitor
    _rate_limit_monitor = RateLimitMonitor(alert_callback=alert_callback)
    return _rate_limit_monitor


# Logging utilities
class RateLimitLogFormatter(logging.Formatter):
    """Custom log formatter for rate limiting events"""
    
    def format(self, record):
        # Add rate limiting context if available
        if hasattr(record, 'rate_limit_context'):
            context = record.rate_limit_context
            record.msg = f"[RateLimit] {record.msg} - Context: {context}"
        
        return super().format(record)


def setup_rate_limit_logging():
    """Setup specialized logging for rate limiting"""
    # Create rate limit logger
    rate_limit_logger = logging.getLogger('airtrace.rate_limit')
    rate_limit_logger.setLevel(logging.INFO)
    
    # Create handler if not exists
    if not rate_limit_logger.handlers:
        handler = logging.StreamHandler()
        formatter = RateLimitLogFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        rate_limit_logger.addHandler(handler)
    
    return rate_limit_logger