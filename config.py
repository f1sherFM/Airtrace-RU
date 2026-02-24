"""
Configuration module for AirTrace RU Backend

Manages environment-based configuration for Redis, caching, and performance settings.
Supports both development and production environments with appropriate defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Redis configuration settings"""
    # Connection settings
    url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    
    # Connection pool settings
    max_connections: int = field(default_factory=lambda: int(os.getenv("REDIS_MAX_CONNECTIONS", "20")))
    connection_timeout: float = field(default_factory=lambda: float(os.getenv("REDIS_CONNECTION_TIMEOUT", "5.0")))
    socket_timeout: float = field(default_factory=lambda: float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0")))
    
    # Cluster settings
    cluster_enabled: bool = field(default_factory=lambda: os.getenv("REDIS_CLUSTER_ENABLED", "false").lower() == "true")
    cluster_nodes: List[str] = field(default_factory=lambda: [
        node.strip() for node in os.getenv("REDIS_CLUSTER_NODES", "").split(",") if node.strip()
    ])
    
    # Health check settings
    health_check_interval: int = field(default_factory=lambda: int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("REDIS_MAX_RETRIES", "3")))
    retry_delay: float = field(default_factory=lambda: float(os.getenv("REDIS_RETRY_DELAY", "1.0")))
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization"""
        # Parse Redis URL if provided
        if self.url and self.url != "redis://localhost:6379/0":
            try:
                parsed = urlparse(self.url)
                if parsed.hostname:
                    self.host = parsed.hostname
                if parsed.port:
                    self.port = parsed.port
                if parsed.path and len(parsed.path) > 1:
                    self.db = int(parsed.path[1:])
                if parsed.password:
                    self.password = parsed.password
            except Exception as e:
                logger.warning(f"Failed to parse Redis URL: {e}")
        
        # Validate cluster configuration
        if self.cluster_enabled and not self.cluster_nodes:
            logger.warning("Redis cluster enabled but no cluster nodes specified")
            self.cluster_enabled = False
        
        # Validate numeric values
        if self.max_connections <= 0:
            self.max_connections = 20
        if self.connection_timeout <= 0:
            self.connection_timeout = 5.0
        if self.socket_timeout <= 0:
            self.socket_timeout = 5.0


@dataclass
class CacheConfig:
    """Multi-level cache configuration"""
    # L1 Cache (In-Memory) settings
    l1_enabled: bool = field(default_factory=lambda: os.getenv("CACHE_L1_ENABLED", "true").lower() == "true")
    l1_max_size: int = field(default_factory=lambda: int(os.getenv("CACHE_L1_MAX_SIZE", "1000")))
    l1_default_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_L1_DEFAULT_TTL", "300")))  # 5 minutes
    
    # L2 Cache (Redis) settings
    l2_enabled: bool = field(default_factory=lambda: os.getenv("CACHE_L2_ENABLED", "true").lower() == "true")
    l2_default_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_L2_DEFAULT_TTL", "1800")))  # 30 minutes
    l2_key_prefix: str = field(default_factory=lambda: os.getenv("CACHE_L2_KEY_PREFIX", "airtrace:v1"))
    
    # L3 Cache (Persistent) settings
    l3_enabled: bool = field(default_factory=lambda: os.getenv("CACHE_L3_ENABLED", "false").lower() == "true")
    l3_default_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_L3_DEFAULT_TTL", "86400")))  # 24 hours
    l3_cache_dir: str = field(default_factory=lambda: os.getenv("CACHE_L3_CACHE_DIR", ".cache/l3"))
    l3_max_files: int = field(default_factory=lambda: int(os.getenv("CACHE_L3_MAX_FILES", "10000")))
    l3_cleanup_interval: int = field(default_factory=lambda: int(os.getenv("CACHE_L3_CLEANUP_INTERVAL", "3600")))  # 1 hour
    
    # Cache warming settings
    warming_enabled: bool = field(default_factory=lambda: os.getenv("CACHE_WARMING_ENABLED", "true").lower() == "true")
    warming_batch_size: int = field(default_factory=lambda: int(os.getenv("CACHE_WARMING_BATCH_SIZE", "10")))
    
    # Privacy settings
    hash_coordinates: bool = field(default_factory=lambda: os.getenv("CACHE_HASH_COORDINATES", "true").lower() == "true")
    coordinate_precision: int = field(default_factory=lambda: int(os.getenv("CACHE_COORDINATE_PRECISION", "3")))
    
    def __post_init__(self):
        """Validate cache configuration"""
        if self.l1_max_size <= 0:
            self.l1_max_size = 1000
        if self.l1_default_ttl <= 0:
            self.l1_default_ttl = 300
        if self.l2_default_ttl <= 0:
            self.l2_default_ttl = 1800
        if self.l3_default_ttl <= 0:
            self.l3_default_ttl = 86400
        if self.l3_max_files <= 0:
            self.l3_max_files = 10000
        if self.l3_cleanup_interval <= 0:
            self.l3_cleanup_interval = 3600
        if self.coordinate_precision < 1:
            self.coordinate_precision = 3
        
        # Ensure L3 cache directory is absolute path
        if not os.path.isabs(self.l3_cache_dir):
            self.l3_cache_dir = os.path.abspath(self.l3_cache_dir)


@dataclass
class WeatherAPIConfig:
    """WeatherAPI.com configuration"""
    # API settings
    api_key: str = field(default_factory=lambda: os.getenv("WEATHER_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("WEATHER_API_BASE_URL", "https://api.weatherapi.com/v1"))
    
    # Connection settings
    timeout: float = field(default_factory=lambda: float(os.getenv("WEATHER_API_TIMEOUT", "25.0")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("WEATHER_API_MAX_RETRIES", "3")))
    
    # Rate limiting (WeatherAPI.com limits)
    requests_per_month: int = field(default_factory=lambda: int(os.getenv("WEATHER_API_REQUESTS_PER_MONTH", "1000000")))
    requests_per_minute: int = field(default_factory=lambda: int(os.getenv("WEATHER_API_REQUESTS_PER_MINUTE", "100")))
    
    # Feature flags
    enabled: bool = field(default_factory=lambda: os.getenv("WEATHER_API_ENABLED", "false").lower() == "true")
    fallback_enabled: bool = field(default_factory=lambda: os.getenv("WEATHER_API_FALLBACK_ENABLED", "true").lower() == "true")
    
    def __post_init__(self):
        """Validate WeatherAPI configuration"""
        # ✅ FIX #1: Add comprehensive API key validation
        if self.enabled:
            if not self.api_key:
                logger.error("WeatherAPI enabled but no API key provided")
                raise ValueError(
                    "WEATHER_API_KEY is required when WeatherAPI is enabled. "
                    "Set WEATHER_API_ENABLED=false to disable or provide a valid API key."
                )
            
            # Validate API key format (WeatherAPI keys are typically 32 characters)
            if len(self.api_key) < 20:
                logger.error(f"Invalid WEATHER_API_KEY format: key too short ({len(self.api_key)} chars)")
                raise ValueError(
                    f"Invalid WEATHER_API_KEY format: expected at least 20 characters, got {len(self.api_key)}. "
                    "Please check your API key."
                )
            
            # Check for placeholder/example keys
            placeholder_keys = ['your_api_key', 'example', 'test', 'demo', 'placeholder']
            if any(placeholder in self.api_key.lower() for placeholder in placeholder_keys):
                logger.error("WEATHER_API_KEY appears to be a placeholder value")
                raise ValueError(
                    "WEATHER_API_KEY appears to be a placeholder. "
                    "Please provide a valid API key from weatherapi.com"
                )
            
            logger.info(f"WeatherAPI enabled with valid key (length: {len(self.api_key)})")
        else:
            logger.info("WeatherAPI disabled")
        
        if self.timeout <= 0:
            self.timeout = 25.0
        if self.max_retries < 0:
            self.max_retries = 3


@dataclass
class RequestOptimizationConfig:
    """Request optimization configuration"""
    # Feature flags
    enabled: bool = field(default_factory=lambda: os.getenv("REQUEST_OPTIMIZATION_ENABLED", "true").lower() == "true")
    batching_enabled: bool = field(default_factory=lambda: os.getenv("REQUEST_BATCHING_ENABLED", "true").lower() == "true")
    deduplication_enabled: bool = field(default_factory=lambda: os.getenv("REQUEST_DEDUPLICATION_ENABLED", "true").lower() == "true")
    prefetching_enabled: bool = field(default_factory=lambda: os.getenv("REQUEST_PREFETCHING_ENABLED", "true").lower() == "true")
    
    # Batching settings
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("REQUEST_MAX_BATCH_SIZE", "10")))
    batch_timeout_ms: int = field(default_factory=lambda: int(os.getenv("REQUEST_BATCH_TIMEOUT_MS", "100")))
    geographic_precision: int = field(default_factory=lambda: int(os.getenv("REQUEST_GEOGRAPHIC_PRECISION", "2")))
    max_geographic_distance: float = field(default_factory=lambda: float(os.getenv("REQUEST_MAX_GEOGRAPHIC_DISTANCE", "0.1")))
    
    # Deduplication settings
    dedup_window_ms: int = field(default_factory=lambda: int(os.getenv("REQUEST_DEDUP_WINDOW_MS", "100")))
    max_pending_requests: int = field(default_factory=lambda: int(os.getenv("REQUEST_MAX_PENDING_REQUESTS", "1000")))
    
    # Prefetching settings
    pattern_window_hours: int = field(default_factory=lambda: int(os.getenv("REQUEST_PATTERN_WINDOW_HOURS", "24")))
    min_pattern_frequency: int = field(default_factory=lambda: int(os.getenv("REQUEST_MIN_PATTERN_FREQUENCY", "3")))
    prefetch_ahead_minutes: int = field(default_factory=lambda: int(os.getenv("REQUEST_PREFETCH_AHEAD_MINUTES", "15")))
    max_prefetch_requests: int = field(default_factory=lambda: int(os.getenv("REQUEST_MAX_PREFETCH_REQUESTS", "50")))
    respect_cache_ttl: bool = field(default_factory=lambda: os.getenv("REQUEST_RESPECT_CACHE_TTL", "true").lower() == "true")
    
    def __post_init__(self):
        """Validate request optimization configuration"""
        if self.max_batch_size <= 0:
            self.max_batch_size = 10
        if self.batch_timeout_ms <= 0:
            self.batch_timeout_ms = 100
        if self.geographic_precision < 1:
            self.geographic_precision = 2
        if self.max_geographic_distance <= 0:
            self.max_geographic_distance = 0.1
        if self.dedup_window_ms <= 0:
            self.dedup_window_ms = 100
        if self.max_pending_requests <= 0:
            self.max_pending_requests = 1000
        if self.pattern_window_hours <= 0:
            self.pattern_window_hours = 24
        if self.min_pattern_frequency <= 0:
            self.min_pattern_frequency = 3
        if self.prefetch_ahead_minutes <= 0:
            self.prefetch_ahead_minutes = 15
        if self.max_prefetch_requests <= 0:
            self.max_prefetch_requests = 50


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    # Feature flags
    redis_enabled: bool = field(default_factory=lambda: os.getenv("PERFORMANCE_REDIS_ENABLED", "true").lower() == "true")
    rate_limiting_enabled: bool = field(default_factory=lambda: os.getenv("PERFORMANCE_RATE_LIMITING_ENABLED", "false").lower() == "true")
    connection_pooling_enabled: bool = field(default_factory=lambda: os.getenv("PERFORMANCE_CONNECTION_POOLING_ENABLED", "true").lower() == "true")
    monitoring_enabled: bool = field(default_factory=lambda: os.getenv("PERFORMANCE_MONITORING_ENABLED", "false").lower() == "true")
    request_optimization_enabled: bool = field(default_factory=lambda: os.getenv("PERFORMANCE_REQUEST_OPTIMIZATION_ENABLED", "true").lower() == "true")
    rate_limit_trust_forwarded_headers: bool = field(
        default_factory=lambda: os.getenv("PERFORMANCE_RATE_LIMIT_TRUST_FORWARDED_HEADERS", "false").lower() == "true"
    )
    rate_limit_trusted_proxy_ips: List[str] = field(
        default_factory=lambda: [
            item.strip()
            for item in os.getenv("PERFORMANCE_RATE_LIMIT_TRUSTED_PROXY_IPS", "").split(",")
            if item.strip()
        ]
    )
    
    # Graceful degradation settings
    fallback_to_memory: bool = field(default_factory=lambda: os.getenv("PERFORMANCE_FALLBACK_TO_MEMORY", "true").lower() == "true")
    max_fallback_duration: int = field(default_factory=lambda: int(os.getenv("PERFORMANCE_MAX_FALLBACK_DURATION", "3600")))  # 1 hour
    
    # Environment detection
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")


@dataclass
class ApiTransportConfig:
    """Unified HTTP transport policy for external requests."""
    trust_env: bool = field(default_factory=lambda: os.getenv("API_TRANSPORT_TRUST_ENV", "false").lower() == "true")
    connect_timeout: float = field(default_factory=lambda: float(os.getenv("API_TRANSPORT_CONNECT_TIMEOUT", "10.0")))
    read_timeout: float = field(default_factory=lambda: float(os.getenv("API_TRANSPORT_READ_TIMEOUT", "30.0")))
    write_timeout: float = field(default_factory=lambda: float(os.getenv("API_TRANSPORT_WRITE_TIMEOUT", "10.0")))
    pool_timeout: float = field(default_factory=lambda: float(os.getenv("API_TRANSPORT_POOL_TIMEOUT", "5.0")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("API_TRANSPORT_MAX_RETRIES", "3")))
    retry_delay: float = field(default_factory=lambda: float(os.getenv("API_TRANSPORT_RETRY_DELAY", "1.0")))
    backoff_factor: float = field(default_factory=lambda: float(os.getenv("API_TRANSPORT_BACKOFF_FACTOR", "2.0")))
    response_timeout: float = field(default_factory=lambda: float(os.getenv("API_RESPONSE_TIMEOUT", "10.0")))

    def __post_init__(self):
        if self.connect_timeout <= 0:
            self.connect_timeout = 10.0
        if self.read_timeout <= 0:
            self.read_timeout = 30.0
        if self.write_timeout <= 0:
            self.write_timeout = 10.0
        if self.pool_timeout <= 0:
            self.pool_timeout = 5.0
        if self.max_retries < 0:
            self.max_retries = 3
        if self.retry_delay <= 0:
            self.retry_delay = 1.0
        if self.backoff_factor < 1.0:
            self.backoff_factor = 2.0
        if self.response_timeout <= 0:
            self.response_timeout = 10.0


@dataclass
class HistoryConfig:
    """Historical pipeline and anomaly detection configuration"""
    anomaly_baseline_window: int = field(default_factory=lambda: int(os.getenv("HISTORY_ANOMALY_BASELINE_WINDOW", "6")))
    anomaly_min_absolute_delta: float = field(default_factory=lambda: float(os.getenv("HISTORY_ANOMALY_MIN_ABSOLUTE_DELTA", "35.0")))
    anomaly_min_relative_delta: float = field(default_factory=lambda: float(os.getenv("HISTORY_ANOMALY_MIN_RELATIVE_DELTA", "0.55")))

    def __post_init__(self):
        if self.anomaly_baseline_window <= 0:
            self.anomaly_baseline_window = 6
        if self.anomaly_min_absolute_delta <= 0:
            self.anomaly_min_absolute_delta = 35.0
        if self.anomaly_min_relative_delta <= 0:
            self.anomaly_min_relative_delta = 0.55


class ConfigManager:
    """Central configuration manager for the application"""
    
    def __init__(self):
        self.redis = RedisConfig()
        self.cache = CacheConfig()
        self.performance = PerformanceConfig()
        self.api = ApiTransportConfig()
        self.history = HistoryConfig()
        self.weather_api = WeatherAPIConfig()
        self.request_optimization = RequestOptimizationConfig()
        self._validate_configuration()
        
        # Initialize configuration audit manager
        self._audit_manager = None
        self._initialize_audit_manager()
    
    def _validate_configuration(self):
        """Validate the complete configuration"""
        errors = []
        
        # Validate Redis configuration
        if self.performance.redis_enabled:
            if self.redis.cluster_enabled and not self.redis.cluster_nodes:
                errors.append("Redis cluster enabled but no cluster nodes specified")
            
            if not self.redis.host and not self.redis.url:
                errors.append("Redis host or URL must be specified when Redis is enabled")
        
        # Validate cache configuration
        if self.cache.l2_enabled and not self.performance.redis_enabled:
            logger.warning("L2 cache enabled but Redis is disabled - L2 cache will be unavailable")
        
        # Validate WeatherAPI configuration
        if self.weather_api.enabled and not self.weather_api.api_key:
            errors.append("WeatherAPI enabled but no API key provided")
        
        if errors:
            error_msg = "Configuration validation failed: " + "; ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def get_redis_connection_kwargs(self) -> Dict[str, Any]:
        """Get Redis connection parameters as kwargs"""
        kwargs = {
            "host": self.redis.host,
            "port": self.redis.port,
            "db": self.redis.db,
            "socket_timeout": self.redis.socket_timeout,
            "socket_connect_timeout": self.redis.connection_timeout,
            "max_connections": self.redis.max_connections,
            "health_check_interval": self.redis.health_check_interval,
        }
        
        if self.redis.password:
            kwargs["password"] = self.redis.password
        
        return kwargs
    
    def get_redis_cluster_kwargs(self) -> Dict[str, Any]:
        """Get Redis cluster connection parameters"""
        if not self.redis.cluster_enabled:
            return {}
        
        startup_nodes = []
        for node in self.redis.cluster_nodes:
            if ":" in node:
                host, port = node.split(":", 1)
                startup_nodes.append({"host": host.strip(), "port": int(port.strip())})
            else:
                startup_nodes.append({"host": node.strip(), "port": 6379})
        
        kwargs = {
            "startup_nodes": startup_nodes,
            "socket_timeout": self.redis.socket_timeout,
            "socket_connect_timeout": self.redis.connection_timeout,
            "max_connections": self.redis.max_connections,
        }
        
        if self.redis.password:
            kwargs["password"] = self.redis.password
        
        return kwargs
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.performance.environment.lower() in ["development", "dev", "local"]
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.performance.environment.lower() in ["production", "prod"]
    
    def log_configuration(self):
        """Log current configuration (without sensitive data)"""
        logger.info("Configuration loaded:")
        logger.info(f"  Environment: {self.performance.environment}")
        logger.info(f"  Redis enabled: {self.performance.redis_enabled}")
        logger.info(f"  Redis cluster: {self.redis.cluster_enabled}")
        logger.info(f"  Cache L1 enabled: {self.cache.l1_enabled}")
        logger.info(f"  Cache L2 enabled: {self.cache.l2_enabled}")
        logger.info(f"  Cache L3 enabled: {self.cache.l3_enabled}")
        logger.info(f"  Fallback to memory: {self.performance.fallback_to_memory}")
        logger.info(f"  WeatherAPI enabled: {self.weather_api.enabled}")
        logger.info(f"  Connection pooling enabled: {self.performance.connection_pooling_enabled}")
        logger.info(f"  Request optimization enabled: {self.performance.request_optimization_enabled}")
        logger.info(f"  Request batching enabled: {self.request_optimization.batching_enabled}")
        logger.info(f"  Request prefetching enabled: {self.request_optimization.prefetching_enabled}")
        logger.info(
            "  History anomaly thresholds: "
            f"window={self.history.anomaly_baseline_window}, "
            f"abs_delta={self.history.anomaly_min_absolute_delta}, "
            f"rel_delta={self.history.anomaly_min_relative_delta}"
        )
    
    def _initialize_audit_manager(self):
        """Initialize configuration audit manager"""
        try:
            from config_audit_manager import setup_configuration_audit
            self._audit_manager = setup_configuration_audit()
            
            # Add validation rules for critical configuration changes
            self._setup_validation_rules()
            
            # Add change callbacks for monitoring
            self._setup_change_callbacks()
            
            logger.info("Configuration audit manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize configuration audit manager: {e}")
            self._audit_manager = None
    
    def _setup_validation_rules(self):
        """Setup validation rules for configuration changes"""
        if not self._audit_manager:
            return
        
        def validate_redis_connections(old_value, new_value):
            """Validate Redis connection settings"""
            errors = []
            if isinstance(new_value, int) and new_value <= 0:
                errors.append("Redis connection count must be positive")
            if isinstance(new_value, int) and new_value > 1000:
                errors.append("Redis connection count seems too high (>1000)")
            return errors
        
        def validate_cache_ttl(old_value, new_value):
            """Validate cache TTL settings"""
            errors = []
            if isinstance(new_value, int) and new_value < 0:
                errors.append("Cache TTL cannot be negative")
            if isinstance(new_value, int) and new_value > 86400:  # 24 hours
                errors.append("Cache TTL seems too high (>24 hours)")
            return errors
        
        def validate_rate_limits(old_value, new_value):
            """Validate rate limiting settings"""
            errors = []
            if isinstance(new_value, int) and new_value <= 0:
                errors.append("Rate limit must be positive")
            if isinstance(new_value, int) and new_value > 10000:
                errors.append("Rate limit seems too high (>10000)")
            return errors
        
        # Register validation rules
        self._audit_manager.add_validation_rule("redis.max_connections", validate_redis_connections)
        self._audit_manager.add_validation_rule("cache.l1_default_ttl", validate_cache_ttl)
        self._audit_manager.add_validation_rule("cache.l2_default_ttl", validate_cache_ttl)
        self._audit_manager.add_validation_rule("rate_limiting.requests_per_minute", validate_rate_limits)
    
    def _setup_change_callbacks(self):
        """Setup callbacks for configuration changes"""
        if not self._audit_manager:
            return
        
        def log_critical_changes(change):
            """Log critical configuration changes"""
            # Check if there's performance impact data available
            impact_severity = 'unknown'
            if self._audit_manager:
                for entry in self._audit_manager.audit_trail:
                    if entry.change.change_id == change.change_id and entry.performance_impact:
                        impact_severity = entry.performance_impact.impact_severity
                        break
            
            if impact_severity in ['high', 'critical']:
                logger.warning(f"Critical configuration change detected: "
                             f"{change.component}.{change.setting_path} "
                             f"severity={impact_severity}")
        
        def alert_on_validation_failures(change):
            """Alert on configuration validation failures"""
            if change.validation_status == 'invalid':
                logger.error(f"Configuration validation failed: "
                           f"{change.component}.{change.setting_path} "
                           f"errors={change.validation_errors}")
        
        # Register callbacks
        self._audit_manager.add_change_callback(log_critical_changes)
        self._audit_manager.add_change_callback(alert_on_validation_failures)
    
    def update_configuration(self, component: str, setting_path: str, 
                           new_value: Any, source: str = 'runtime',
                           user_context: Optional[str] = None,
                           reason: Optional[str] = None) -> bool:
        """
        Update configuration with audit logging.
        
        Args:
            component: Configuration component name
            setting_path: Dot-notation path to setting
            new_value: New configuration value
            source: Source of change ('runtime', 'environment', 'config_file')
            user_context: User or system context making the change
            reason: Reason for the change
            
        Returns:
            bool: Success status
        """
        try:
            # Get current value
            old_value = self._get_configuration_value(component, setting_path)
            
            # Log the configuration change
            change_id = ""
            if self._audit_manager:
                change_id = self._audit_manager.log_configuration_change(
                    component=component,
                    setting_path=setting_path,
                    old_value=old_value,
                    new_value=new_value,
                    source=source,
                    user_context=user_context,
                    reason=reason
                )
            
            # Apply the configuration change
            success = self._apply_configuration_change(component, setting_path, new_value)
            
            if success:
                logger.info(f"Configuration updated: {component}.{setting_path} = {new_value} "
                           f"(change_id: {change_id})")
            else:
                logger.error(f"Failed to apply configuration change: {component}.{setting_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def _get_configuration_value(self, component: str, setting_path: str) -> Any:
        """Get current configuration value"""
        try:
            component_obj = getattr(self, component, None)
            if component_obj is None:
                return None
            
            # Handle nested paths
            path_parts = setting_path.split('.')
            current_obj = component_obj
            
            for part in path_parts:
                if hasattr(current_obj, part):
                    current_obj = getattr(current_obj, part)
                else:
                    return None
            
            return current_obj
            
        except Exception as e:
            logger.error(f"Error getting configuration value: {e}")
            return None
    
    def _apply_configuration_change(self, component: str, setting_path: str, new_value: Any) -> bool:
        """Apply configuration change to the actual configuration object"""
        try:
            component_obj = getattr(self, component, None)
            if component_obj is None:
                return False
            
            # Handle nested paths
            path_parts = setting_path.split('.')
            current_obj = component_obj
            
            # Navigate to the parent object
            for part in path_parts[:-1]:
                if hasattr(current_obj, part):
                    current_obj = getattr(current_obj, part)
                else:
                    return False
            
            # Set the final attribute
            final_attr = path_parts[-1]
            if hasattr(current_obj, final_attr):
                setattr(current_obj, final_attr, new_value)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error applying configuration change: {e}")
            return False
    
    def get_audit_manager(self):
        """Get configuration audit manager instance"""
        return self._audit_manager
    
    def create_configuration_snapshot(self, snapshot_id: Optional[str] = None) -> str:
        """Create configuration snapshot for rollback purposes"""
        if self._audit_manager:
            return self._audit_manager.create_configuration_snapshot(snapshot_id)
        return ""
    
    def get_configuration_audit_trail(self, component: Optional[str] = None,
                                    time_range: Optional[timedelta] = None) -> List[Any]:
        """Get configuration change audit trail"""
        if self._audit_manager:
            return self._audit_manager.get_audit_trail(component, time_range)
        return []


# Global configuration instance
config = ConfigManager()
