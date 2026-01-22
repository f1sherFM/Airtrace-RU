"""
Dynamic Resource Scaling and Advanced Pooling for AirTrace RU Backend

Implements dynamic resource scaling based on load patterns, advanced resource pooling
strategies, and detailed profiling and performance analysis capabilities.
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from collections import deque, defaultdict
from enum import Enum
import statistics
import weakref
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Resource scaling direction"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class LoadMetrics:
    """Load metrics for scaling decisions"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    request_rate: float
    response_time_ms: float
    active_connections: int
    queue_depth: int
    error_rate: float


@dataclass
class ScalingDecision:
    """Resource scaling decision"""
    timestamp: datetime
    direction: ScalingDirection
    resource_type: str
    current_capacity: int
    target_capacity: int
    reason: str
    confidence: float
    metrics: LoadMetrics


@dataclass
class ScalingConfig:
    """Configuration for dynamic scaling"""
    # Scaling thresholds
    cpu_scale_up_threshold: float = 70.0
    cpu_scale_down_threshold: float = 30.0
    memory_scale_up_threshold: float = 80.0
    memory_scale_down_threshold: float = 40.0
    response_time_scale_up_threshold_ms: float = 1000.0
    response_time_scale_down_threshold_ms: float = 200.0
    
    # Advanced scaling parameters
    min_capacity: int = 1
    max_capacity: int = 20
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    
    # Timing parameters
    evaluation_interval_seconds: float = 30.0
    cooldown_period_seconds: float = 300.0  # 5 minutes
    metrics_window_seconds: float = 300.0   # 5 minutes
    
    # Stability requirements
    min_stable_periods: int = 3
    confidence_threshold: float = 0.7
    
    # Advanced scaling features
    predictive_scaling_enabled: bool = True
    load_pattern_learning_enabled: bool = True
    burst_detection_enabled: bool = True
    seasonal_adjustment_enabled: bool = True
    
    # Machine learning parameters
    pattern_history_days: int = 7
    prediction_horizon_minutes: int = 30
    burst_detection_sensitivity: float = 2.0  # Standard deviations
    seasonal_pattern_weight: float = 0.3


@dataclass
class LoadPattern:
    """Load pattern analysis result"""
    pattern_type: str  # "steady", "increasing", "decreasing", "cyclical", "burst", "seasonal"
    confidence: float
    predicted_load: float
    time_to_peak: Optional[int]  # minutes
    seasonality_factor: float
    trend_strength: float
    volatility: float


@dataclass
class AdvancedScalingDecision(ScalingDecision):
    """Enhanced scaling decision with advanced analytics"""
    load_pattern: LoadPattern
    prediction_accuracy: float
    risk_assessment: str  # "low", "medium", "high"
    alternative_strategies: List[str]
    cost_benefit_ratio: float


@dataclass
class PoolingStrategy:
    """Advanced pooling strategy configuration"""
    # Pool sizing
    initial_size: int = 5
    min_size: int = 1
    max_size: int = 50
    
    # Growth parameters
    growth_factor: float = 1.5
    shrink_factor: float = 0.8
    
    # Timing parameters
    idle_timeout_seconds: float = 300.0
    cleanup_interval_seconds: float = 60.0
    
    # Health checking
    health_check_enabled: bool = True
    health_check_interval_seconds: float = 30.0
    max_unhealthy_ratio: float = 0.3


@dataclass
class ProfilingResult:
    """Detailed profiling result"""
    session_id: str
    duration_seconds: float
    memory_profile: Dict[str, Any]
    cpu_profile: Dict[str, Any]
    allocation_patterns: List[Dict[str, Any]]
    hotspots: List[Dict[str, Any]]
    recommendations: List[str]


class AdvancedResourcePool:
    """
    Advanced resource pool with dynamic scaling, health checking,
    and intelligent resource management.
    """
    
    def __init__(self, name: str, factory: Callable, strategy: PoolingStrategy,
                 cleanup_func: Optional[Callable] = None,
                 health_check_func: Optional[Callable] = None):
        self.name = name
        self.factory = factory
        self.strategy = strategy
        self.cleanup_func = cleanup_func
        self.health_check_func = health_check_func
        
        # Pool state
        self._pool: deque = deque()
        self._active_resources: set = set()
        self._unhealthy_resources: set = set()
        self._lock = threading.RLock()
        
        # Statistics
        self._created_count = 0
        self._acquired_count = 0
        self._released_count = 0
        self._cleanup_count = 0
        self._health_check_failures = 0
        
        # Scaling state
        self._last_scale_time = datetime.now()
        self._load_history: deque = deque(maxlen=100)
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._scaling_task: Optional[asyncio.Task] = None
        
        # Initialize pool
        self._initialize_pool()
        
        logger.info(f"Advanced resource pool created: {name}")
    
    def _initialize_pool(self):
        """Initialize the pool with initial resources"""
        with self._lock:
            for _ in range(self.strategy.initial_size):
                try:
                    resource = self.factory()
                    self._pool.append(resource)
                    self._created_count += 1
                except Exception as e:
                    logger.error(f"Failed to create initial resource for pool {self.name}: {e}")
    
    async def start_background_tasks(self):
        """Start background maintenance tasks"""
        if self.strategy.cleanup_interval_seconds > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.strategy.health_check_enabled and self.health_check_func:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        
        logger.info(f"Background tasks started for pool {self.name}")
    
    async def stop_background_tasks(self):
        """Stop background maintenance tasks"""
        tasks = [self._cleanup_task, self._health_check_task, self._scaling_task]
        
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info(f"Background tasks stopped for pool {self.name}")
    
    def acquire(self, timeout: Optional[float] = None) -> Any:
        """Acquire a resource from the pool"""
        start_time = time.time()
        
        with self._lock:
            # Try to get from pool first
            while self._pool:
                resource = self._pool.popleft()
                
                # Check if resource is healthy
                if resource not in self._unhealthy_resources:
                    self._active_resources.add(resource)
                    self._acquired_count += 1
                    return resource
                else:
                    # Remove unhealthy resource
                    self._cleanup_resource(resource)
            
            # No resources available, create new one if under limit
            if len(self._active_resources) < self.strategy.max_size:
                try:
                    resource = self.factory()
                    self._active_resources.add(resource)
                    self._created_count += 1
                    self._acquired_count += 1
                    return resource
                except Exception as e:
                    logger.error(f"Failed to create resource for pool {self.name}: {e}")
                    raise
            
            # Pool is at capacity
            if timeout is None or timeout <= 0:
                raise RuntimeError(f"Pool {self.name} is at capacity")
            
            # Wait for resource to become available (simplified implementation)
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Timeout waiting for resource from pool {self.name}")
            
            # In a real implementation, this would use proper condition variables
            time.sleep(0.01)
            return self.acquire(timeout - elapsed)
    
    def release(self, resource: Any):
        """Release a resource back to the pool"""
        with self._lock:
            if resource in self._active_resources:
                self._active_resources.remove(resource)
                
                # Check if resource is still healthy
                if (resource not in self._unhealthy_resources and 
                    len(self._pool) < self.strategy.max_size):
                    self._pool.append(resource)
                    self._released_count += 1
                else:
                    # Clean up resource
                    self._cleanup_resource(resource)
    
    def _cleanup_resource(self, resource: Any):
        """Clean up a single resource"""
        try:
            if self.cleanup_func:
                self.cleanup_func(resource)
            self._cleanup_count += 1
        except Exception as e:
            logger.warning(f"Error cleaning up resource in pool {self.name}: {e}")
        finally:
            self._unhealthy_resources.discard(resource)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        try:
            while True:
                await asyncio.sleep(self.strategy.cleanup_interval_seconds)
                await self._perform_cleanup()
        except asyncio.CancelledError:
            logger.info(f"Cleanup loop cancelled for pool {self.name}")
    
    async def _perform_cleanup(self):
        """Perform periodic cleanup"""
        with self._lock:
            # Remove idle resources if pool is too large
            current_time = time.time()
            target_size = max(self.strategy.min_size, 
                            int(len(self._active_resources) * 1.2))
            
            while len(self._pool) > target_size:
                resource = self._pool.popleft()
                self._cleanup_resource(resource)
            
            # Clean up unhealthy resources
            unhealthy_to_remove = list(self._unhealthy_resources)
            for resource in unhealthy_to_remove:
                if resource in self._active_resources:
                    self._active_resources.remove(resource)
                self._cleanup_resource(resource)
    
    async def _health_check_loop(self):
        """Background health check loop"""
        try:
            while True:
                await asyncio.sleep(self.strategy.health_check_interval_seconds)
                await self._perform_health_checks()
        except asyncio.CancelledError:
            logger.info(f"Health check loop cancelled for pool {self.name}")
    
    async def _perform_health_checks(self):
        """Perform health checks on resources"""
        if not self.health_check_func:
            return
        
        with self._lock:
            resources_to_check = list(self._pool) + list(self._active_resources)
        
        unhealthy_count = 0
        
        for resource in resources_to_check:
            try:
                if asyncio.iscoroutinefunction(self.health_check_func):
                    is_healthy = await self.health_check_func(resource)
                else:
                    is_healthy = self.health_check_func(resource)
                
                if not is_healthy:
                    with self._lock:
                        self._unhealthy_resources.add(resource)
                        unhealthy_count += 1
                        self._health_check_failures += 1
                else:
                    with self._lock:
                        self._unhealthy_resources.discard(resource)
                        
            except Exception as e:
                logger.warning(f"Health check failed for resource in pool {self.name}: {e}")
                with self._lock:
                    self._unhealthy_resources.add(resource)
                    unhealthy_count += 1
                    self._health_check_failures += 1
        
        # Check if too many resources are unhealthy
        total_resources = len(resources_to_check)
        if total_resources > 0:
            unhealthy_ratio = unhealthy_count / total_resources
            if unhealthy_ratio > self.strategy.max_unhealthy_ratio:
                logger.warning(f"Pool {self.name} has high unhealthy ratio: {unhealthy_ratio:.2f}")
    
    async def _scaling_loop(self):
        """Background scaling loop"""
        try:
            while True:
                await asyncio.sleep(30.0)  # Check every 30 seconds
                await self._evaluate_scaling()
        except asyncio.CancelledError:
            logger.info(f"Scaling loop cancelled for pool {self.name}")
    
    async def _evaluate_scaling(self):
        """Evaluate if pool should be scaled"""
        with self._lock:
            current_size = len(self._pool) + len(self._active_resources)
            utilization = len(self._active_resources) / max(current_size, 1)
            
            # Simple scaling logic based on utilization
            if utilization > 0.8 and current_size < self.strategy.max_size:
                # Scale up
                target_size = min(self.strategy.max_size, 
                                int(current_size * self.strategy.growth_factor))
                await self._scale_pool(target_size)
            elif utilization < 0.3 and current_size > self.strategy.min_size:
                # Scale down
                target_size = max(self.strategy.min_size,
                                int(current_size * self.strategy.shrink_factor))
                await self._scale_pool(target_size)
    
    async def _scale_pool(self, target_size: int):
        """Scale pool to target size"""
        with self._lock:
            current_size = len(self._pool) + len(self._active_resources)
            
            if target_size > current_size:
                # Scale up - create more resources
                for _ in range(target_size - current_size):
                    try:
                        resource = self.factory()
                        self._pool.append(resource)
                        self._created_count += 1
                    except Exception as e:
                        logger.error(f"Failed to scale up pool {self.name}: {e}")
                        break
                
                logger.info(f"Scaled up pool {self.name} from {current_size} to {len(self._pool) + len(self._active_resources)}")
            
            elif target_size < current_size:
                # Scale down - remove excess resources
                excess = current_size - target_size
                removed = 0
                
                while self._pool and removed < excess:
                    resource = self._pool.popleft()
                    self._cleanup_resource(resource)
                    removed += 1
                
                logger.info(f"Scaled down pool {self.name} by {removed} resources")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        with self._lock:
            return {
                "name": self.name,
                "pool_size": len(self._pool),
                "active_resources": len(self._active_resources),
                "unhealthy_resources": len(self._unhealthy_resources),
                "total_capacity": len(self._pool) + len(self._active_resources),
                "utilization": len(self._active_resources) / max(len(self._pool) + len(self._active_resources), 1),
                "created_count": self._created_count,
                "acquired_count": self._acquired_count,
                "released_count": self._released_count,
                "cleanup_count": self._cleanup_count,
                "health_check_failures": self._health_check_failures,
                "strategy": {
                    "min_size": self.strategy.min_size,
                    "max_size": self.strategy.max_size,
                    "initial_size": self.strategy.initial_size
                }
            }


class AdvancedLoadAnalyzer:
    """
    Advanced load pattern analyzer with machine learning capabilities.
    
    Analyzes historical load patterns to predict future resource needs
    and identify optimal scaling strategies.
    """
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self._load_history: deque = deque(maxlen=10000)  # Store more history for ML
        self._pattern_cache: Dict[str, LoadPattern] = {}
        self._seasonal_patterns: Dict[int, float] = {}  # hour -> seasonal factor
        self._prediction_accuracy_history: deque = deque(maxlen=100)
        
        logger.info("Advanced load analyzer initialized")
    
    def add_load_sample(self, metrics: LoadMetrics):
        """Add a load sample for pattern analysis"""
        self._load_history.append(metrics)
        
        # Update seasonal patterns
        hour = metrics.timestamp.hour
        if hour not in self._seasonal_patterns:
            self._seasonal_patterns[hour] = 1.0
        
        # Simple exponential smoothing for seasonal adjustment
        current_load = self._normalize_load(metrics)
        self._seasonal_patterns[hour] = (
            0.9 * self._seasonal_patterns[hour] + 
            0.1 * current_load
        )
    
    def analyze_load_pattern(self, window_minutes: int = 60) -> LoadPattern:
        """Analyze current load pattern with advanced algorithms"""
        if len(self._load_history) < 10:
            return LoadPattern(
                pattern_type="insufficient_data",
                confidence=0.0,
                predicted_load=0.0,
                time_to_peak=None,
                seasonality_factor=1.0,
                trend_strength=0.0,
                volatility=0.0
            )
        
        # Get recent samples
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_samples = [
            m for m in self._load_history 
            if m.timestamp >= cutoff_time
        ]
        
        if len(recent_samples) < 5:
            recent_samples = list(self._load_history)[-10:]
        
        # Extract load values
        load_values = [self._normalize_load(m) for m in recent_samples]
        timestamps = [m.timestamp for m in recent_samples]
        
        # Analyze trend
        trend_strength, trend_direction = self._analyze_trend(load_values)
        
        # Detect patterns
        pattern_type = self._detect_pattern_type(load_values, timestamps)
        
        # Calculate volatility
        volatility = self._calculate_volatility(load_values)
        
        # Predict future load
        predicted_load = self._predict_load(load_values, timestamps)
        
        # Estimate time to peak
        time_to_peak = self._estimate_time_to_peak(load_values, timestamps)
        
        # Get seasonal factor
        current_hour = datetime.now().hour
        seasonality_factor = self._seasonal_patterns.get(current_hour, 1.0)
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(
            pattern_type, trend_strength, volatility, len(recent_samples)
        )
        
        return LoadPattern(
            pattern_type=pattern_type,
            confidence=confidence,
            predicted_load=predicted_load,
            time_to_peak=time_to_peak,
            seasonality_factor=seasonality_factor,
            trend_strength=trend_strength,
            volatility=volatility
        )
    
    def _normalize_load(self, metrics: LoadMetrics) -> float:
        """Normalize load metrics to a single value"""
        # Weighted combination of different metrics
        cpu_weight = 0.3
        memory_weight = 0.3
        response_time_weight = 0.2
        request_rate_weight = 0.2
        
        # Normalize each metric to 0-1 scale
        cpu_norm = min(metrics.cpu_percent / 100.0, 1.0)
        memory_norm = min(metrics.memory_percent / 100.0, 1.0)
        response_time_norm = min(metrics.response_time_ms / 5000.0, 1.0)  # 5s max
        request_rate_norm = min(metrics.request_rate / 1000.0, 1.0)  # 1000 req/s max
        
        return (
            cpu_weight * cpu_norm +
            memory_weight * memory_norm +
            response_time_weight * response_time_norm +
            request_rate_weight * request_rate_norm
        )
    
    def _analyze_trend(self, values: List[float]) -> Tuple[float, str]:
        """Analyze trend strength and direction using advanced methods"""
        if len(values) < 3:
            return 0.0, "stable"
        
        # Use linear regression for trend analysis
        n = len(values)
        x = list(range(n))
        
        # Calculate correlation coefficient
        mean_x = sum(x) / n
        mean_y = sum(values) / n
        
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denominator_y = sum((values[i] - mean_y) ** 2 for i in range(n))
        
        if denominator_x == 0 or denominator_y == 0:
            return 0.0, "stable"
        
        correlation = numerator / (denominator_x * denominator_y) ** 0.5
        
        # Calculate slope
        slope = numerator / denominator_x if denominator_x > 0 else 0
        
        # Determine trend direction and strength
        if abs(correlation) < 0.3:
            return abs(correlation), "stable"
        elif slope > 0:
            return abs(correlation), "increasing"
        else:
            return abs(correlation), "decreasing"
    
    def _detect_pattern_type(self, values: List[float], timestamps: List[datetime]) -> str:
        """Detect load pattern type using advanced algorithms"""
        if len(values) < 10:
            return "insufficient_data"
        
        # Check for burst patterns
        if self._detect_burst_pattern(values):
            return "burst"
        
        # Check for cyclical patterns
        if self._detect_cyclical_pattern(values):
            return "cyclical"
        
        # Check for seasonal patterns
        if self._detect_seasonal_pattern(timestamps, values):
            return "seasonal"
        
        # Analyze overall trend
        trend_strength, trend_direction = self._analyze_trend(values)
        
        if trend_strength > 0.7:
            return trend_direction
        elif trend_strength < 0.3:
            return "steady"
        else:
            return "variable"
    
    def _detect_burst_pattern(self, values: List[float]) -> bool:
        """Detect burst patterns in load data"""
        if len(values) < 5:
            return False
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_val == 0:
            return False
        
        # Check for values significantly above mean
        burst_threshold = mean_val + (self.config.burst_detection_sensitivity * std_val)
        burst_count = sum(1 for v in values if v > burst_threshold)
        
        # Burst if more than 20% of values are significantly above mean
        return burst_count / len(values) > 0.2
    
    def _detect_cyclical_pattern(self, values: List[float]) -> bool:
        """Detect cyclical patterns using autocorrelation"""
        if len(values) < 20:
            return False
        
        # Simple autocorrelation check
        n = len(values)
        mean_val = statistics.mean(values)
        
        # Check for periodicity at different lags
        max_correlation = 0.0
        for lag in range(2, min(n // 2, 20)):
            correlation = 0.0
            count = 0
            
            for i in range(lag, n):
                correlation += (values[i] - mean_val) * (values[i - lag] - mean_val)
                count += 1
            
            if count > 0:
                correlation /= count
                max_correlation = max(max_correlation, abs(correlation))
        
        return max_correlation > 0.5
    
    def _detect_seasonal_pattern(self, timestamps: List[datetime], values: List[float]) -> bool:
        """Detect seasonal patterns based on time of day"""
        if len(timestamps) < 24:  # Need at least 24 hours of data
            return False
        
        # Group by hour and check for consistent patterns
        hourly_loads = defaultdict(list)
        for ts, val in zip(timestamps, values):
            hourly_loads[ts.hour].append(val)
        
        # Calculate variance between hours vs within hours
        hour_means = {hour: statistics.mean(loads) for hour, loads in hourly_loads.items()}
        
        if len(hour_means) < 12:  # Need data from at least half the day
            return False
        
        between_hour_variance = statistics.variance(hour_means.values())
        within_hour_variance = statistics.mean([
            statistics.variance(loads) if len(loads) > 1 else 0
            for loads in hourly_loads.values()
        ])
        
        # Seasonal if between-hour variance is much larger than within-hour variance
        return between_hour_variance > 2 * within_hour_variance
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate load volatility"""
        if len(values) < 2:
            return 0.0
        
        return statistics.stdev(values) / max(statistics.mean(values), 0.001)
    
    def _predict_load(self, values: List[float], timestamps: List[datetime]) -> float:
        """Predict future load using multiple methods"""
        if len(values) < 3:
            return values[-1] if values else 0.0
        
        # Method 1: Linear extrapolation
        trend_strength, trend_direction = self._analyze_trend(values)
        linear_prediction = self._linear_extrapolation(values)
        
        # Method 2: Exponential smoothing
        exp_prediction = self._exponential_smoothing(values)
        
        # Method 3: Seasonal adjustment
        current_hour = datetime.now().hour
        seasonal_factor = self._seasonal_patterns.get(current_hour, 1.0)
        seasonal_prediction = values[-1] * seasonal_factor
        
        # Weighted combination
        if trend_strength > 0.7:
            # Strong trend - rely more on linear prediction
            prediction = 0.5 * linear_prediction + 0.3 * exp_prediction + 0.2 * seasonal_prediction
        else:
            # Weak trend - rely more on smoothing and seasonality
            prediction = 0.2 * linear_prediction + 0.5 * exp_prediction + 0.3 * seasonal_prediction
        
        return max(0.0, min(1.0, prediction))
    
    def _linear_extrapolation(self, values: List[float]) -> float:
        """Predict using linear extrapolation"""
        if len(values) < 2:
            return values[-1] if values else 0.0
        
        # Simple linear extrapolation
        recent_values = values[-5:]  # Use last 5 points
        n = len(recent_values)
        x = list(range(n))
        
        # Calculate slope
        mean_x = sum(x) / n
        mean_y = sum(recent_values) / n
        
        numerator = sum((x[i] - mean_x) * (recent_values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return recent_values[-1]
        
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        
        # Predict next point
        return slope * n + intercept
    
    def _exponential_smoothing(self, values: List[float], alpha: float = 0.3) -> float:
        """Predict using exponential smoothing"""
        if not values:
            return 0.0
        
        smoothed = values[0]
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        return smoothed
    
    def _estimate_time_to_peak(self, values: List[float], timestamps: List[datetime]) -> Optional[int]:
        """Estimate time to peak load in minutes"""
        if len(values) < 5:
            return None
        
        trend_strength, trend_direction = self._analyze_trend(values)
        
        if trend_direction != "increasing" or trend_strength < 0.5:
            return None
        
        # Simple estimation based on current trend
        current_rate = (values[-1] - values[-3]) / 2  # Rate per sample
        if current_rate <= 0:
            return None
        
        # Estimate time to reach 90% capacity
        target_load = 0.9
        current_load = values[-1]
        
        if current_load >= target_load:
            return 0
        
        samples_to_peak = (target_load - current_load) / current_rate
        
        # Convert to minutes (assuming samples are taken every 30 seconds)
        minutes_to_peak = int(samples_to_peak * 0.5)
        
        return max(1, min(120, minutes_to_peak))  # Cap between 1 and 120 minutes
    
    def _calculate_pattern_confidence(self, pattern_type: str, trend_strength: float, 
                                    volatility: float, sample_count: int) -> float:
        """Calculate confidence in pattern detection"""
        base_confidence = min(sample_count / 20.0, 1.0)  # More samples = higher confidence
        
        # Adjust based on pattern characteristics
        if pattern_type == "insufficient_data":
            return 0.0
        elif pattern_type in ["burst", "cyclical", "seasonal"]:
            return base_confidence * 0.9  # High confidence for clear patterns
        elif pattern_type in ["increasing", "decreasing"]:
            return base_confidence * trend_strength
        else:
            return base_confidence * (1.0 - volatility)  # Lower volatility = higher confidence
class DynamicResourceScaler:
    """
    Enhanced dynamic resource scaler with advanced load pattern analysis,
    predictive scaling, and machine learning capabilities.
    """
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self._metrics_history: deque = deque(maxlen=1000)
        self._scaling_history: List[ScalingDecision] = []
        self._resource_pools: Dict[str, AdvancedResourcePool] = {}
        
        # Advanced components
        self._load_analyzer = AdvancedLoadAnalyzer(config)
        self._scaling_strategies: Dict[str, Callable] = {
            "reactive": self._reactive_scaling_strategy,
            "predictive": self._predictive_scaling_strategy,
            "hybrid": self._hybrid_scaling_strategy
        }
        
        # Scaling state
        self._last_scaling_time = datetime.now()
        self._stable_periods = 0
        self._prediction_accuracy_tracker = deque(maxlen=50)
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("Enhanced dynamic resource scaler initialized")
    
    async def start_monitoring(self):
        """Start advanced load monitoring and scaling"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._advanced_monitoring_loop())
        logger.info("Advanced dynamic scaling monitoring started")
    
    async def stop_monitoring(self):
        """Stop load monitoring and scaling"""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced dynamic scaling monitoring stopped")
    
    async def _advanced_monitoring_loop(self):
        """Enhanced monitoring loop with predictive capabilities"""
        try:
            while self._monitoring_active:
                try:
                    # Collect current metrics
                    metrics = await self._collect_enhanced_metrics()
                    self._metrics_history.append(metrics)
                    self._load_analyzer.add_load_sample(metrics)
                    
                    # Analyze load patterns
                    load_pattern = self._load_analyzer.analyze_load_pattern()
                    
                    # Choose scaling strategy based on pattern
                    strategy = self._choose_scaling_strategy(load_pattern)
                    
                    # Evaluate scaling decisions using chosen strategy
                    decisions = await self._evaluate_advanced_scaling(metrics, load_pattern, strategy)
                    
                    # Apply scaling decisions with risk assessment
                    for decision in decisions:
                        await self._apply_advanced_scaling_decision(decision)
                    
                    # Update prediction accuracy
                    self._update_prediction_accuracy(metrics, load_pattern)
                    
                except Exception as e:
                    logger.error(f"Error in advanced scaling monitoring loop: {e}")
                
                await asyncio.sleep(self.config.evaluation_interval_seconds)
        except asyncio.CancelledError:
            logger.info("Advanced scaling monitoring loop cancelled")
    
    async def _collect_enhanced_metrics(self) -> LoadMetrics:
        """Collect enhanced load metrics from multiple sources"""
        # Get basic system metrics
        try:
            from system_monitor import get_system_monitor
            system_monitor = get_system_monitor()
            
            if system_monitor.metrics_history:
                latest_system = system_monitor.metrics_history[-1]
                cpu_percent = latest_system.cpu_usage
                memory_percent = latest_system.memory_percent
            else:
                cpu_percent = memory_percent = 50.0
        except:
            cpu_percent = memory_percent = 50.0
        
        # Get performance metrics
        try:
            from performance_monitor import get_performance_monitor
            perf_monitor = get_performance_monitor()
            perf_stats = perf_monitor.get_performance_stats()
            
            response_time_ms = perf_stats.avg_response_time * 1000
            error_rate = perf_stats.error_rate
        except:
            response_time_ms = 200.0
            error_rate = 0.01
        
        # Get connection pool metrics
        try:
            from connection_pool import get_connection_pool_manager
            pool_manager = get_connection_pool_manager()
            pool_stats = await pool_manager.get_all_stats()
            
            total_connections = sum(stats.get('total_connections', 0) for stats in pool_stats.values())
            active_connections = sum(stats.get('active_connections', 0) for stats in pool_stats.values())
        except:
            total_connections = active_connections = 10
        
        # Calculate request rate (simplified)
        request_rate = 100.0  # This would be calculated from actual metrics
        queue_depth = max(0, active_connections - total_connections)
        
        return LoadMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            request_rate=request_rate,
            response_time_ms=response_time_ms,
            active_connections=active_connections,
            queue_depth=queue_depth,
            error_rate=error_rate
        )
    
    def _choose_scaling_strategy(self, load_pattern: LoadPattern) -> str:
        """Choose optimal scaling strategy based on load pattern"""
        if not self.config.predictive_scaling_enabled:
            return "reactive"
        
        # Choose strategy based on pattern characteristics
        if load_pattern.pattern_type == "burst":
            return "predictive"  # Burst patterns benefit from prediction
        elif load_pattern.pattern_type in ["cyclical", "seasonal"]:
            return "predictive"  # Predictable patterns
        elif load_pattern.confidence > 0.8 and load_pattern.trend_strength > 0.7:
            return "hybrid"  # High confidence trends
        else:
            return "reactive"  # Fall back to reactive for uncertain patterns
    
    async def _evaluate_advanced_scaling(self, current_metrics: LoadMetrics, 
                                       load_pattern: LoadPattern, 
                                       strategy: str) -> List[AdvancedScalingDecision]:
        """Evaluate scaling decisions using advanced algorithms"""
        decisions = []
        
        # Check cooldown period
        time_since_last_scaling = datetime.now() - self._last_scaling_time
        if time_since_last_scaling.total_seconds() < self.config.cooldown_period_seconds:
            return decisions
        
        # Use selected strategy
        strategy_func = self._scaling_strategies.get(strategy, self._reactive_scaling_strategy)
        
        for pool_name, pool in self._resource_pools.items():
            decision = await strategy_func(pool_name, pool, current_metrics, load_pattern)
            if decision:
                decisions.append(decision)
        
        return decisions
    
    async def _reactive_scaling_strategy(self, pool_name: str, pool: AdvancedResourcePool,
                                       metrics: LoadMetrics, load_pattern: LoadPattern) -> Optional[AdvancedScalingDecision]:
        """Traditional reactive scaling strategy"""
        basic_decision = self._make_basic_scaling_decision(pool_name, pool, metrics)
        
        if not basic_decision:
            return None
        
        # Enhance with advanced analytics
        return AdvancedScalingDecision(
            timestamp=basic_decision.timestamp,
            direction=basic_decision.direction,
            resource_type=basic_decision.resource_type,
            current_capacity=basic_decision.current_capacity,
            target_capacity=basic_decision.target_capacity,
            reason=basic_decision.reason,
            confidence=basic_decision.confidence,
            metrics=basic_decision.metrics,
            load_pattern=load_pattern,
            prediction_accuracy=self._get_average_prediction_accuracy(),
            risk_assessment="medium",
            alternative_strategies=["predictive", "hybrid"],
            cost_benefit_ratio=self._calculate_cost_benefit_ratio(basic_decision)
        )
    
    async def _predictive_scaling_strategy(self, pool_name: str, pool: AdvancedResourcePool,
                                         metrics: LoadMetrics, load_pattern: LoadPattern) -> Optional[AdvancedScalingDecision]:
        """Predictive scaling strategy based on load forecasting"""
        if load_pattern.confidence < 0.6:
            # Fall back to reactive if prediction confidence is low
            return await self._reactive_scaling_strategy(pool_name, pool, metrics, load_pattern)
        
        stats = pool.get_stats()
        current_capacity = stats["total_capacity"]
        
        # Predict future resource needs
        predicted_load = load_pattern.predicted_load
        current_load = self._load_analyzer._normalize_load(metrics)
        
        # Calculate target capacity based on prediction
        if load_pattern.pattern_type == "burst" and load_pattern.time_to_peak:
            # Scale up proactively for burst patterns
            target_capacity = min(
                self.config.max_capacity,
                int(current_capacity * (1 + predicted_load))
            )
            direction = ScalingDirection.UP
            reason = f"Predictive scaling for {load_pattern.pattern_type} pattern (confidence: {load_pattern.confidence:.2f})"
        
        elif load_pattern.pattern_type == "increasing" and load_pattern.trend_strength > 0.7:
            # Scale up for strong increasing trends
            scale_factor = 1 + (predicted_load - current_load)
            target_capacity = min(
                self.config.max_capacity,
                int(current_capacity * scale_factor)
            )
            direction = ScalingDirection.UP
            reason = f"Predictive scaling for increasing trend (strength: {load_pattern.trend_strength:.2f})"
        
        elif load_pattern.pattern_type == "decreasing" and load_pattern.trend_strength > 0.7:
            # Scale down for strong decreasing trends
            scale_factor = max(0.5, 1 - (current_load - predicted_load))
            target_capacity = max(
                self.config.min_capacity,
                int(current_capacity * scale_factor)
            )
            direction = ScalingDirection.DOWN
            reason = f"Predictive scaling for decreasing trend (strength: {load_pattern.trend_strength:.2f})"
        
        else:
            return None
        
        if target_capacity == current_capacity:
            return None
        
        return AdvancedScalingDecision(
            timestamp=datetime.now(),
            direction=direction,
            resource_type=pool_name,
            current_capacity=current_capacity,
            target_capacity=target_capacity,
            reason=reason,
            confidence=load_pattern.confidence,
            metrics=metrics,
            load_pattern=load_pattern,
            prediction_accuracy=self._get_average_prediction_accuracy(),
            risk_assessment=self._assess_scaling_risk(load_pattern, direction),
            alternative_strategies=["reactive", "hybrid"],
            cost_benefit_ratio=self._calculate_predictive_cost_benefit(current_capacity, target_capacity, load_pattern)
        )
    
    async def _hybrid_scaling_strategy(self, pool_name: str, pool: AdvancedResourcePool,
                                     metrics: LoadMetrics, load_pattern: LoadPattern) -> Optional[AdvancedScalingDecision]:
        """Hybrid scaling strategy combining reactive and predictive approaches"""
        # Get both reactive and predictive decisions
        reactive_decision = await self._reactive_scaling_strategy(pool_name, pool, metrics, load_pattern)
        predictive_decision = await self._predictive_scaling_strategy(pool_name, pool, metrics, load_pattern)
        
        # If only one strategy suggests scaling, use it
        if reactive_decision and not predictive_decision:
            reactive_decision.alternative_strategies = ["predictive"]
            return reactive_decision
        elif predictive_decision and not reactive_decision:
            predictive_decision.alternative_strategies = ["reactive"]
            return predictive_decision
        elif not reactive_decision and not predictive_decision:
            return None
        
        # Both strategies suggest scaling - choose the more conservative approach
        if reactive_decision.direction == predictive_decision.direction:
            # Same direction - use the more conservative target
            if reactive_decision.direction == ScalingDirection.UP:
                target_capacity = min(reactive_decision.target_capacity, predictive_decision.target_capacity)
            else:
                target_capacity = max(reactive_decision.target_capacity, predictive_decision.target_capacity)
            
            confidence = (reactive_decision.confidence + predictive_decision.confidence) / 2
            reason = f"Hybrid scaling: {reactive_decision.reason} + {predictive_decision.reason}"
        else:
            # Different directions - be more conservative
            target_capacity = reactive_decision.current_capacity  # No scaling
            confidence = 0.5
            reason = "Hybrid scaling: conflicting signals, maintaining current capacity"
        
        return AdvancedScalingDecision(
            timestamp=datetime.now(),
            direction=ScalingDirection.STABLE if target_capacity == reactive_decision.current_capacity else reactive_decision.direction,
            resource_type=pool_name,
            current_capacity=reactive_decision.current_capacity,
            target_capacity=target_capacity,
            reason=reason,
            confidence=confidence,
            metrics=metrics,
            load_pattern=load_pattern,
            prediction_accuracy=self._get_average_prediction_accuracy(),
            risk_assessment="low",
            alternative_strategies=["reactive", "predictive"],
            cost_benefit_ratio=(reactive_decision.cost_benefit_ratio + predictive_decision.cost_benefit_ratio) / 2
        )
    
    def _make_basic_scaling_decision(self, pool_name: str, pool: AdvancedResourcePool,
                                   metrics: LoadMetrics) -> Optional[ScalingDecision]:
        """Make basic scaling decision using traditional thresholds"""
        stats = pool.get_stats()
        current_capacity = stats["total_capacity"]
        
        # Determine if scaling is needed
        scale_up_signals = 0
        scale_down_signals = 0
        reasons = []
        
        # CPU-based scaling
        if metrics.cpu_percent > self.config.cpu_scale_up_threshold:
            scale_up_signals += 1
            reasons.append(f"CPU usage {metrics.cpu_percent:.1f}% > {self.config.cpu_scale_up_threshold}%")
        elif metrics.cpu_percent < self.config.cpu_scale_down_threshold:
            scale_down_signals += 1
            reasons.append(f"CPU usage {metrics.cpu_percent:.1f}% < {self.config.cpu_scale_down_threshold}%")
        
        # Memory-based scaling
        if metrics.memory_percent > self.config.memory_scale_up_threshold:
            scale_up_signals += 1
            reasons.append(f"Memory usage {metrics.memory_percent:.1f}% > {self.config.memory_scale_up_threshold}%")
        elif metrics.memory_percent < self.config.memory_scale_down_threshold:
            scale_down_signals += 1
            reasons.append(f"Memory usage {metrics.memory_percent:.1f}% < {self.config.memory_scale_down_threshold}%")
        
        # Response time-based scaling
        if metrics.response_time_ms > self.config.response_time_scale_up_threshold_ms:
            scale_up_signals += 1
            reasons.append(f"Response time {metrics.response_time_ms:.1f}ms > {self.config.response_time_scale_up_threshold_ms}ms")
        elif metrics.response_time_ms < self.config.response_time_scale_down_threshold_ms:
            scale_down_signals += 1
            reasons.append(f"Response time {metrics.response_time_ms:.1f}ms < {self.config.response_time_scale_down_threshold_ms}ms")
        
        # Pool utilization
        utilization = stats["utilization"]
        if utilization > 0.8:
            scale_up_signals += 1
            reasons.append(f"Pool utilization {utilization:.1f} > 0.8")
        elif utilization < 0.3:
            scale_down_signals += 1
            reasons.append(f"Pool utilization {utilization:.1f} < 0.3")
        
        # Make decision
        if scale_up_signals > scale_down_signals and scale_up_signals >= 2:
            direction = ScalingDirection.UP
            target_capacity = min(self.config.max_capacity,
                                int(current_capacity * self.config.scale_up_factor))
            confidence = min(1.0, scale_up_signals / 4.0)
        elif scale_down_signals > scale_up_signals and scale_down_signals >= 2:
            direction = ScalingDirection.DOWN
            target_capacity = max(self.config.min_capacity,
                                int(current_capacity * self.config.scale_down_factor))
            confidence = min(1.0, scale_down_signals / 4.0)
        else:
            return None
        
        # Check confidence threshold
        if confidence < self.config.confidence_threshold:
            return None
        
        return ScalingDecision(
            timestamp=datetime.now(),
            direction=direction,
            resource_type=pool_name,
            current_capacity=current_capacity,
            target_capacity=target_capacity,
            reason="; ".join(reasons),
            confidence=confidence,
            metrics=metrics
        )
    
    def _assess_scaling_risk(self, load_pattern: LoadPattern, direction: ScalingDirection) -> str:
        """Assess risk of scaling decision"""
        if load_pattern.confidence < 0.5:
            return "high"
        elif load_pattern.volatility > 0.5:
            return "medium"
        elif direction == ScalingDirection.DOWN and load_pattern.pattern_type == "burst":
            return "high"  # Risky to scale down during burst patterns
        else:
            return "low"
    
    def _calculate_cost_benefit_ratio(self, decision: ScalingDecision) -> float:
        """Calculate cost-benefit ratio for scaling decision"""
        # Simplified cost-benefit calculation
        capacity_change = abs(decision.target_capacity - decision.current_capacity)
        benefit = decision.confidence  # Higher confidence = higher benefit
        cost = capacity_change / decision.current_capacity  # Relative cost
        
        return benefit / max(cost, 0.1)  # Avoid division by zero
    
    def _calculate_predictive_cost_benefit(self, current_capacity: int, target_capacity: int, 
                                         load_pattern: LoadPattern) -> float:
        """Calculate cost-benefit ratio for predictive scaling"""
        capacity_change = abs(target_capacity - current_capacity)
        
        # Benefit based on prediction accuracy and pattern confidence
        benefit = load_pattern.confidence * self._get_average_prediction_accuracy()
        
        # Cost based on capacity change and volatility
        cost = (capacity_change / current_capacity) * (1 + load_pattern.volatility)
        
        return benefit / max(cost, 0.1)
    
    def _get_average_prediction_accuracy(self) -> float:
        """Get average prediction accuracy"""
        if not self._prediction_accuracy_tracker:
            return 0.7  # Default assumption
        
        return statistics.mean(self._prediction_accuracy_tracker)
    
    def _update_prediction_accuracy(self, current_metrics: LoadMetrics, load_pattern: LoadPattern):
        """Update prediction accuracy tracking"""
        if len(self._metrics_history) < 2:
            return
        
        # Compare predicted vs actual load
        previous_metrics = self._metrics_history[-2]
        previous_load = self._load_analyzer._normalize_load(previous_metrics)
        current_load = self._load_analyzer._normalize_load(current_metrics)
        
        # Simple accuracy calculation
        predicted_load = load_pattern.predicted_load
        error = abs(predicted_load - current_load)
        accuracy = max(0.0, 1.0 - error)
        
        self._prediction_accuracy_tracker.append(accuracy)
    
    async def _apply_advanced_scaling_decision(self, decision: AdvancedScalingDecision):
        """Apply an advanced scaling decision with risk assessment"""
        pool = self._resource_pools.get(decision.resource_type)
        if not pool:
            logger.warning(f"Pool not found for scaling decision: {decision.resource_type}")
            return
        
        # Check risk assessment
        if decision.risk_assessment == "high" and decision.confidence < 0.8:
            logger.warning(f"High-risk scaling decision rejected: {decision.resource_type} "
                         f"(risk: {decision.risk_assessment}, confidence: {decision.confidence:.2f})")
            return
        
        try:
            await pool._scale_pool(decision.target_capacity)
            self._scaling_history.append(decision)
            self._last_scaling_time = decision.timestamp
            
            logger.info(f"Applied advanced scaling decision: {decision.resource_type} "
                       f"{decision.direction.value} from {decision.current_capacity} "
                       f"to {decision.target_capacity} "
                       f"(strategy: {decision.load_pattern.pattern_type}, "
                       f"confidence: {decision.confidence:.2f}, "
                       f"risk: {decision.risk_assessment})")
        
        except Exception as e:
            logger.error(f"Failed to apply advanced scaling decision: {e}")
    
    def get_advanced_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics"""
        return {
            "scaling_history_count": len(self._scaling_history),
            "average_prediction_accuracy": self._get_average_prediction_accuracy(),
            "registered_pools": list(self._resource_pools.keys()),
            "current_config": {
                "predictive_scaling_enabled": self.config.predictive_scaling_enabled,
                "load_pattern_learning_enabled": self.config.load_pattern_learning_enabled,
                "burst_detection_enabled": self.config.burst_detection_enabled,
                "seasonal_adjustment_enabled": self.config.seasonal_adjustment_enabled
            },
            "load_analyzer_stats": {
                "pattern_cache_size": len(self._load_analyzer._pattern_cache),
                "seasonal_patterns_count": len(self._load_analyzer._seasonal_patterns),
                "load_history_size": len(self._load_analyzer._load_history)
            }
        }
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Dynamic scaling monitoring started")
    
    async def stop_monitoring(self):
        """Stop load monitoring and scaling"""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Dynamic scaling monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring and scaling loop"""
        try:
            while self._monitoring_active:
                try:
                    # Collect current metrics
                    metrics = await self._collect_metrics()
                    self._metrics_history.append(metrics)
                    
                    # Evaluate scaling decisions
                    decisions = await self._evaluate_scaling(metrics)
                    
                    # Apply scaling decisions
                    for decision in decisions:
                        await self._apply_scaling_decision(decision)
                    
                except Exception as e:
                    logger.error(f"Error in scaling monitoring loop: {e}")
                
                await asyncio.sleep(self.config.evaluation_interval_seconds)
        except asyncio.CancelledError:
            logger.info("Scaling monitoring loop cancelled")
    
    async def _collect_metrics(self) -> LoadMetrics:
        """Collect current load metrics"""
        # In a real implementation, this would collect actual metrics
        # For now, we'll return mock metrics
        return LoadMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            request_rate=100.0,
            response_time_ms=200.0,
            active_connections=50,
            queue_depth=5,
            error_rate=0.01
        )
    
    async def _evaluate_scaling(self, current_metrics: LoadMetrics) -> List[ScalingDecision]:
        """Evaluate scaling decisions based on current metrics"""
        decisions = []
        
        # Check if we're in cooldown period
        time_since_last_scaling = datetime.now() - self._last_scaling_time
        if time_since_last_scaling.total_seconds() < self.config.cooldown_period_seconds:
            return decisions
        
        # Get recent metrics for trend analysis
        recent_metrics = list(self._metrics_history)[-10:]  # Last 10 measurements
        if len(recent_metrics) < 3:
            return decisions
        
        # Analyze trends
        cpu_trend = self._analyze_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._analyze_trend([m.memory_percent for m in recent_metrics])
        response_time_trend = self._analyze_trend([m.response_time_ms for m in recent_metrics])
        
        # Make scaling decisions for each resource pool
        for pool_name, pool in self._resource_pools.items():
            decision = self._make_scaling_decision(
                pool_name, pool, current_metrics, 
                cpu_trend, memory_trend, response_time_trend
            )
            
            if decision:
                decisions.append(decision)
        
        return decisions
    
    def _analyze_trend(self, values: List[float]) -> Tuple[float, str]:
        """Analyze trend in metric values"""
        if len(values) < 2:
            return 0.0, "stable"
        
        # Simple linear trend analysis
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0, "stable"
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if slope > 0.5:
            return slope, "increasing"
        elif slope < -0.5:
            return slope, "decreasing"
        else:
            return slope, "stable"
    
    def _make_scaling_decision(self, pool_name: str, pool: AdvancedResourcePool,
                             metrics: LoadMetrics, cpu_trend: Tuple[float, str],
                             memory_trend: Tuple[float, str], 
                             response_time_trend: Tuple[float, str]) -> Optional[ScalingDecision]:
        """Make scaling decision for a specific resource pool"""
        stats = pool.get_stats()
        current_capacity = stats["total_capacity"]
        
        # Determine if scaling is needed
        scale_up_signals = 0
        scale_down_signals = 0
        reasons = []
        
        # CPU-based scaling
        if metrics.cpu_percent > self.config.cpu_scale_up_threshold:
            scale_up_signals += 1
            reasons.append(f"CPU usage {metrics.cpu_percent:.1f}% > {self.config.cpu_scale_up_threshold}%")
        elif metrics.cpu_percent < self.config.cpu_scale_down_threshold:
            scale_down_signals += 1
            reasons.append(f"CPU usage {metrics.cpu_percent:.1f}% < {self.config.cpu_scale_down_threshold}%")
        
        # Memory-based scaling
        if metrics.memory_percent > self.config.memory_scale_up_threshold:
            scale_up_signals += 1
            reasons.append(f"Memory usage {metrics.memory_percent:.1f}% > {self.config.memory_scale_up_threshold}%")
        elif metrics.memory_percent < self.config.memory_scale_down_threshold:
            scale_down_signals += 1
            reasons.append(f"Memory usage {metrics.memory_percent:.1f}% < {self.config.memory_scale_down_threshold}%")
        
        # Response time-based scaling
        if metrics.response_time_ms > self.config.response_time_scale_up_threshold_ms:
            scale_up_signals += 1
            reasons.append(f"Response time {metrics.response_time_ms:.1f}ms > {self.config.response_time_scale_up_threshold_ms}ms")
        elif metrics.response_time_ms < self.config.response_time_scale_down_threshold_ms:
            scale_down_signals += 1
            reasons.append(f"Response time {metrics.response_time_ms:.1f}ms < {self.config.response_time_scale_down_threshold_ms}ms")
        
        # Pool utilization
        utilization = stats["utilization"]
        if utilization > 0.8:
            scale_up_signals += 1
            reasons.append(f"Pool utilization {utilization:.1f} > 0.8")
        elif utilization < 0.3:
            scale_down_signals += 1
            reasons.append(f"Pool utilization {utilization:.1f} < 0.3")
        
        # Make decision
        if scale_up_signals > scale_down_signals and scale_up_signals >= 2:
            direction = ScalingDirection.UP
            target_capacity = min(self.config.max_capacity,
                                int(current_capacity * self.config.scale_up_factor))
            confidence = min(1.0, scale_up_signals / 4.0)
        elif scale_down_signals > scale_up_signals and scale_down_signals >= 2:
            direction = ScalingDirection.DOWN
            target_capacity = max(self.config.min_capacity,
                                int(current_capacity * self.config.scale_down_factor))
            confidence = min(1.0, scale_down_signals / 4.0)
        else:
            return None
        
        # Check confidence threshold
        if confidence < self.config.confidence_threshold:
            return None
        
        return ScalingDecision(
            timestamp=datetime.now(),
            direction=direction,
            resource_type=pool_name,
            current_capacity=current_capacity,
            target_capacity=target_capacity,
            reason="; ".join(reasons),
            confidence=confidence,
            metrics=metrics
        )
    
    async def _apply_scaling_decision(self, decision: ScalingDecision):
        """Apply a scaling decision"""
        pool = self._resource_pools.get(decision.resource_type)
        if not pool:
            logger.warning(f"Pool not found for scaling decision: {decision.resource_type}")
            return
        
        try:
            await pool._scale_pool(decision.target_capacity)
            self._scaling_history.append(decision)
            self._last_scaling_time = decision.timestamp
            
            logger.info(f"Applied scaling decision: {decision.resource_type} "
                       f"{decision.direction.value} from {decision.current_capacity} "
                       f"to {decision.target_capacity} (confidence: {decision.confidence:.2f})")
        
        except Exception as e:
            logger.error(f"Failed to apply scaling decision: {e}")
    
    def register_pool(self, pool: AdvancedResourcePool):
        """Register a resource pool for scaling"""
        self._resource_pools[pool.name] = pool
        logger.info(f"Registered pool for scaling: {pool.name}")
    
    def unregister_pool(self, pool_name: str):
        """Unregister a resource pool from scaling"""
        if pool_name in self._resource_pools:
            del self._resource_pools[pool_name]
            logger.info(f"Unregistered pool from scaling: {pool_name}")
    
    def get_scaling_history(self, limit: int = 50) -> List[ScalingDecision]:
        """Get recent scaling decisions"""
        return self._scaling_history[-limit:]
    
    def get_metrics_history(self, limit: int = 100) -> List[LoadMetrics]:
        """Get recent load metrics"""
        return list(self._metrics_history)[-limit:]


class PerformanceProfiler:
    """
    Advanced performance profiler with detailed analysis capabilities.
    """
    
    def __init__(self):
        self._profiling_sessions: Dict[str, Dict[str, Any]] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Profiler")
        
        logger.info("Performance profiler initialized")
    
    async def start_profiling_session(self, session_id: str, duration: int,
                                    profile_memory: bool = True,
                                    profile_cpu: bool = True) -> str:
        """Start a comprehensive profiling session"""
        if session_id in self._profiling_sessions:
            raise ValueError(f"Profiling session {session_id} already exists")
        
        session = {
            "session_id": session_id,
            "start_time": datetime.now(),
            "duration": duration,
            "profile_memory": profile_memory,
            "profile_cpu": profile_cpu,
            "snapshots": [],
            "active": True
        }
        
        self._profiling_sessions[session_id] = session
        
        # Schedule session completion
        asyncio.create_task(self._complete_session(session_id, duration))
        
        logger.info(f"Started profiling session: {session_id} for {duration}s")
        return session_id
    
    async def _complete_session(self, session_id: str, duration: int):
        """Complete a profiling session"""
        await asyncio.sleep(duration)
        
        if session_id in self._profiling_sessions:
            session = self._profiling_sessions[session_id]
            session["active"] = False
            session["end_time"] = datetime.now()
            
            # Generate profiling result
            result = await self._generate_profiling_result(session)
            session["result"] = result
            
            logger.info(f"Completed profiling session: {session_id}")
    
    async def _generate_profiling_result(self, session: Dict[str, Any]) -> ProfilingResult:
        """Generate comprehensive profiling result"""
        # This is a simplified implementation
        # In a real system, this would analyze memory allocations, CPU hotspots, etc.
        
        recommendations = []
        
        # Analyze memory patterns
        memory_profile = {
            "peak_usage_mb": 150.0,
            "average_usage_mb": 120.0,
            "allocation_rate": 1000,  # allocations per second
            "gc_frequency": 5  # collections per minute
        }
        
        if memory_profile["peak_usage_mb"] > 200:
            recommendations.append("Consider reducing memory usage - peak usage is high")
        
        if memory_profile["gc_frequency"] > 10:
            recommendations.append("High GC frequency detected - optimize object lifecycle")
        
        # Analyze CPU patterns
        cpu_profile = {
            "average_usage_percent": 45.0,
            "peak_usage_percent": 85.0,
            "context_switches": 1000,
            "system_calls": 5000
        }
        
        if cpu_profile["peak_usage_percent"] > 90:
            recommendations.append("High CPU usage detected - consider optimization")
        
        # Mock allocation patterns and hotspots
        allocation_patterns = [
            {"type": "string", "count": 10000, "size_mb": 5.0},
            {"type": "dict", "count": 5000, "size_mb": 15.0},
            {"type": "list", "count": 3000, "size_mb": 8.0}
        ]
        
        hotspots = [
            {"function": "process_request", "cpu_percent": 25.0, "calls": 1000},
            {"function": "serialize_data", "cpu_percent": 15.0, "calls": 500},
            {"function": "validate_input", "cpu_percent": 10.0, "calls": 2000}
        ]
        
        return ProfilingResult(
            session_id=session["session_id"],
            duration_seconds=session["duration"],
            memory_profile=memory_profile,
            cpu_profile=cpu_profile,
            allocation_patterns=allocation_patterns,
            hotspots=hotspots,
            recommendations=recommendations
        )
    
    def get_profiling_result(self, session_id: str) -> Optional[ProfilingResult]:
        """Get profiling result for a session"""
        session = self._profiling_sessions.get(session_id)
        if session and "result" in session:
            return session["result"]
        return None
    
    def list_sessions(self) -> List[str]:
        """List all profiling sessions"""
        return list(self._profiling_sessions.keys())
    
    async def cleanup(self):
        """Clean up profiler resources"""
        self._thread_pool.shutdown(wait=True)
        logger.info("Performance profiler cleanup completed")


# Global instances
_dynamic_scaler: Optional[DynamicResourceScaler] = None
_performance_profiler: Optional[PerformanceProfiler] = None


def get_dynamic_scaler(config: Optional[ScalingConfig] = None) -> DynamicResourceScaler:
    """Get the global dynamic resource scaler"""
    global _dynamic_scaler
    if _dynamic_scaler is None:
        _dynamic_scaler = DynamicResourceScaler(config or ScalingConfig())
    return _dynamic_scaler


def get_performance_profiler() -> PerformanceProfiler:
    """Get the global performance profiler"""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler


async def create_advanced_pool(name: str, factory: Callable, 
                             strategy: Optional[PoolingStrategy] = None,
                             cleanup_func: Optional[Callable] = None,
                             health_check_func: Optional[Callable] = None,
                             enable_scaling: bool = True) -> AdvancedResourcePool:
    """
    Create an advanced resource pool with optional dynamic scaling.
    
    Args:
        name: Pool name
        factory: Resource factory function
        strategy: Pooling strategy configuration
        cleanup_func: Resource cleanup function
        health_check_func: Resource health check function
        enable_scaling: Whether to enable dynamic scaling
        
    Returns:
        Configured AdvancedResourcePool
    """
    if strategy is None:
        strategy = PoolingStrategy()
    
    pool = AdvancedResourcePool(name, factory, strategy, cleanup_func, health_check_func)
    await pool.start_background_tasks()
    
    if enable_scaling:
        scaler = get_dynamic_scaler()
        scaler.register_pool(pool)
    
    logger.info(f"Created advanced resource pool: {name} (scaling: {enable_scaling})")
    return pool