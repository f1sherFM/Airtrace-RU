"""
Shared types and enums for rate limiting system

Contains common data structures used across rate limiting components
to avoid circular imports.
"""

from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Dict


class RateLimitStrategy(Enum):
    """Rate limiting identification strategies"""
    IP_BASED = "ip"
    USER_AGENT_BASED = "user_agent"
    COMBINED = "combined"


class EndpointCategory(Enum):
    """Endpoint categories with different rate limits"""
    AIR_QUALITY = "air_quality"
    BATCH_REQUESTS = "batch"
    HEALTH_CHECKS = "health"
    METRICS = "metrics"
    DEFAULT = "default"


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an endpoint category"""
    requests_per_minute: int
    burst_multiplier: float = 1.5
    window_size_seconds: int = 60
    identifier_strategy: RateLimitStrategy = RateLimitStrategy.IP_BASED
    
    def __post_init__(self):
        """Validate configuration values"""
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if self.burst_multiplier < 1.0:
            raise ValueError("burst_multiplier must be >= 1.0")
        if self.window_size_seconds <= 0:
            raise ValueError("window_size_seconds must be positive")


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: int = None
    current_usage: int = 0
    limit: int = 0
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers"""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_time.timestamp())),
            "X-RateLimit-Used": str(self.current_usage)
        }
        
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        
        return headers


@dataclass
class RateLimitInfo:
    """Rate limit information for an identifier"""
    identifier: str
    endpoint: str
    current_count: int
    limit: int
    window_start: datetime
    window_end: datetime
    burst_used: int = 0


@dataclass
class RateLimitMetrics:
    """Rate limiting metrics for monitoring"""
    total_requests: int = 0
    allowed_requests: int = 0
    blocked_requests: int = 0
    burst_requests: int = 0
    unique_identifiers: int = 0
    average_usage_per_window: float = 0.0
    peak_usage: int = 0
    
    @property
    def block_rate(self) -> float:
        """Calculate block rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.blocked_requests / self.total_requests) * 100
    
    @property
    def burst_rate(self) -> float:
        """Calculate burst usage rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.burst_requests / self.total_requests) * 100