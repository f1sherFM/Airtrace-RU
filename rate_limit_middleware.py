"""
Rate limiting middleware for FastAPI integration

Provides seamless integration of rate limiting with FastAPI applications,
including proper HTTP 429 responses and rate limit headers.
"""

import logging
import time
from ipaddress import ip_address, ip_network
from typing import Callable, Optional

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from rate_limit_types import RateLimitResult
from rate_limiter import RateLimiter
from rate_limit_monitoring import get_rate_limit_monitor, setup_rate_limit_logging
from schemas import ErrorResponse

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Integrates with the RateLimiter to provide automatic rate limiting
    for all API endpoints with proper HTTP responses and headers.
    
    ✅ FIX #8: Added IP-based rate limiting
    """
    
    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: Optional[RateLimiter] = None,
        enabled: bool = True,
        skip_paths: Optional[list] = None,
        custom_identifier: Optional[Callable[[Request], str]] = None,
        on_instance_ready: Optional[Callable[["RateLimitMiddleware"], None]] = None,
        # ✅ NEW: IP-based rate limiting configuration
        ip_rate_limit_enabled: bool = True,
        max_requests_per_ip_per_minute: int = 100,
        ip_burst_multiplier: float = 1.5,
        trust_forwarded_headers: bool = False,
        trusted_proxy_ips: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()
        self.enabled = enabled
        # Contract: each skip path is matched by exact path or path-segment prefix.
        # Examples:
        # - "/docs" matches "/docs" and "/docs/oauth2-redirect"
        # - "/docs" does NOT match "/api/docs-backup" or "/docsx"
        # - "/" matches only the root path
        self.skip_paths = skip_paths or ["/docs", "/redoc", "/openapi.json"]
        self.custom_identifier = custom_identifier
        self.on_instance_ready = on_instance_ready
        
        # ✅ NEW: IP-based rate limiting
        self.ip_rate_limit_enabled = ip_rate_limit_enabled
        self.max_requests_per_ip = max_requests_per_ip_per_minute
        self.ip_burst_limit = int(max_requests_per_ip_per_minute * ip_burst_multiplier)
        self.ip_request_counts: dict[str, dict] = {}  # IP -> {count, reset_time, burst_count}
        self.trust_forwarded_headers = trust_forwarded_headers
        self.trusted_proxy_networks = []
        for entry in trusted_proxy_ips or []:
            try:
                self.trusted_proxy_networks.append(ip_network(entry, strict=False))
            except ValueError:
                logger.warning(f"Ignoring invalid trusted proxy network: {entry}")
        if self.trust_forwarded_headers and not self.trusted_proxy_networks:
            logger.warning(
                "RateLimitMiddleware trusts forwarded headers without trusted_proxy_ips; "
                "this is unsafe unless the app is reachable only through a trusted reverse proxy."
            )
        
        # Metrics
        self.total_requests = 0
        self.blocked_requests = 0
        self.ip_blocked_requests = 0
        self.start_time = time.time()
        
        logger.info(
            f"Rate limiting middleware initialized - "
            f"Enabled: {enabled}, IP limiting: {ip_rate_limit_enabled}, "
            f"IP limit: {max_requests_per_ip_per_minute}/min"
        )
        if self.on_instance_ready:
            try:
                self.on_instance_ready(self)
            except Exception as e:
                logger.warning(f"Failed to register live rate limit middleware instance: {e}")
    
    def _should_skip_path(self, path: str) -> bool:
        """Check if path should skip rate limiting (exact or path-segment prefix)."""
        return any(self._path_matches_skip_rule(path, skip_path) for skip_path in self.skip_paths)

    @staticmethod
    def _path_matches_skip_rule(path: str, skip_path: str) -> bool:
        """Exact match or prefix on a path boundary, never substring match."""
        if not skip_path:
            return False
        if skip_path == "/":
            return path == "/"
        return path == skip_path or path.startswith(f"{skip_path}/")

    def _is_trusted_proxy(self, proxy_ip: str) -> bool:
        if not self.trust_forwarded_headers:
            return False
        if not self.trusted_proxy_networks:
            return True
        try:
            addr = ip_address(proxy_ip)
        except ValueError:
            return False
        return any(addr in network for network in self.trusted_proxy_networks)
    
    def _extract_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        client_host = request.client.host if request.client else "unknown"

        if not self._is_trusted_proxy(client_host):
            return client_host

        # Check for forwarded headers (proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fallback to direct client IP
        return client_host
    
    def _extract_user_agent(self, request: Request) -> Optional[str]:
        """Extract user agent from request"""
        return request.headers.get("User-Agent")
    
    def _create_rate_limit_response(self, result: RateLimitResult, endpoint: str) -> JSONResponse:
        """Create HTTP 429 response with rate limit information"""
        error_response = ErrorResponse(
            code="RATE_LIMIT_EXCEEDED",
            message=f"Превышен лимит запросов. Попробуйте через {result.retry_after} секунд."
        )
        
        headers = result.to_headers()
        
        # Add additional context headers
        headers["X-RateLimit-Endpoint"] = endpoint
        headers["X-RateLimit-Policy"] = "sliding-window"
        
        logger.warning(
            f"Rate limit exceeded - Endpoint: {endpoint}, "
            f"Usage: {result.current_usage}/{result.limit}, "
            f"Retry after: {result.retry_after}s"
        )
        
        return JSONResponse(
            status_code=429,
            content=error_response.model_dump(mode='json'),
            headers=headers
        )
    
    def _check_ip_rate_limit(self, client_ip: str) -> tuple[bool, int]:
        """
        ✅ FIX #8: Check IP-based rate limit
        
        Returns:
            tuple: (allowed, retry_after_seconds)
        """
        if not self.ip_rate_limit_enabled:
            return True, 0
        
        current_time = time.time()
        
        # Get or create IP tracking entry
        if client_ip not in self.ip_request_counts:
            self.ip_request_counts[client_ip] = {
                "count": 0,
                "reset_time": current_time + 60,  # 1 minute window
                "burst_count": 0,
                "burst_reset_time": current_time + 10  # 10 second burst window
            }
        
        ip_data = self.ip_request_counts[client_ip]
        
        # Reset counter if window expired
        if current_time >= ip_data["reset_time"]:
            ip_data["count"] = 0
            ip_data["reset_time"] = current_time + 60
        
        # Reset burst counter if burst window expired
        if current_time >= ip_data["burst_reset_time"]:
            ip_data["burst_count"] = 0
            ip_data["burst_reset_time"] = current_time + 10
        
        # Check burst limit (short-term)
        if ip_data["burst_count"] >= self.ip_burst_limit:
            retry_after = int(ip_data["burst_reset_time"] - current_time) + 1
            logger.warning(
                f"IP burst limit exceeded - IP: {client_ip[:10]}..., "
                f"Burst: {ip_data['burst_count']}/{self.ip_burst_limit}"
            )
            return False, retry_after
        
        # Check regular limit (per minute)
        if ip_data["count"] >= self.max_requests_per_ip:
            retry_after = int(ip_data["reset_time"] - current_time) + 1
            logger.warning(
                f"IP rate limit exceeded - IP: {client_ip[:10]}..., "
                f"Count: {ip_data['count']}/{self.max_requests_per_ip}"
            )
            return False, retry_after
        
        # Increment counters
        ip_data["count"] += 1
        ip_data["burst_count"] += 1
        
        # Cleanup old entries (keep only last 1000 IPs)
        if len(self.ip_request_counts) > 1000:
            # First try to remove expired entries
            expired_ips = [
                ip for ip, data in self.ip_request_counts.items()
                if current_time >= data["reset_time"]
            ]
            
            if expired_ips:
                # Remove expired entries
                for ip in expired_ips[:100]:  # Remove up to 100 expired
                    del self.ip_request_counts[ip]
            else:
                # If no expired entries, remove oldest by reset_time
                sorted_ips = sorted(
                    self.ip_request_counts.items(),
                    key=lambda x: x[1]["reset_time"]
                )
                # Remove oldest 100 entries
                for ip, _ in sorted_ips[:100]:
                    del self.ip_request_counts[ip]
        
        return True, 0
    
    def _create_ip_rate_limit_response(self, client_ip: str, retry_after: int) -> JSONResponse:
        """Create HTTP 429 response for IP rate limit"""
        error_response = ErrorResponse(
            code="IP_RATE_LIMIT_EXCEEDED",
            message=f"Превышен лимит запросов с вашего IP. Попробуйте через {retry_after} секунд."
        )
        
        headers = {
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": str(self.max_requests_per_ip),
            "X-RateLimit-Policy": "per-ip",
            "X-RateLimit-Window": "60"
        }
        
        logger.warning(
            f"IP rate limit exceeded - IP: {client_ip[:10]}..., "
            f"Retry after: {retry_after}s"
        )
        
        return JSONResponse(
            status_code=429,
            content=error_response.model_dump(mode='json'),
            headers=headers
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiting middleware"""
        # Skip if middleware is disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip certain paths
        if self._should_skip_path(request.url.path):
            return await call_next(request)
        
        # Extract client information
        client_ip = self._extract_client_ip(request)
        user_agent = self._extract_user_agent(request)
        endpoint = request.url.path
        
        # ✅ FIX #8: Check IP-based rate limit first
        ip_allowed, ip_retry_after = self._check_ip_rate_limit(client_ip)
        if not ip_allowed:
            self.total_requests += 1
            self.blocked_requests += 1
            self.ip_blocked_requests += 1
            return self._create_ip_rate_limit_response(client_ip, ip_retry_after)
        
        # Use custom identifier if provided
        if self.custom_identifier:
            try:
                identifier_info = self.custom_identifier(request)
                # Custom identifier should return IP or IP:user_agent format
                if ":" in identifier_info:
                    client_ip, user_agent = identifier_info.split(":", 1)
                else:
                    client_ip = identifier_info
            except Exception as e:
                logger.warning(f"Custom identifier function failed: {e}")
                # Continue with extracted IP
        
        # Check endpoint-based rate limit
        try:
            start_time = time.time()
            result = await self.rate_limiter.check_rate_limit(
                ip=client_ip,
                endpoint=endpoint,
                user_agent=user_agent
            )
            check_duration = time.time() - start_time
            
            # Update middleware metrics
            self.total_requests += 1
            
            # Log rate limit check (privacy-safe)
            logger.debug(
                f"Rate limit check - Endpoint: {endpoint}, "
                f"Allowed: {result.allowed}, "
                f"Usage: {result.current_usage}/{result.limit}, "
                f"Duration: {check_duration:.3f}s"
            )
            
            # Block request if rate limit exceeded
            if not result.allowed:
                self.blocked_requests += 1
                return self._create_rate_limit_response(result, endpoint)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to successful responses
            rate_limit_headers = result.to_headers()
            for header_name, header_value in rate_limit_headers.items():
                response.headers[header_name] = header_value
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            # Continue without rate limiting on error
            return await call_next(request)
    
    def get_stats(self) -> dict:
        """Get middleware statistics"""
        uptime = time.time() - self.start_time
        return {
            "enabled": self.enabled,
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "ip_blocked_requests": self.ip_blocked_requests,  # ✅ NEW
            "block_rate": (self.blocked_requests / max(1, self.total_requests)) * 100,
            "ip_block_rate": (self.ip_blocked_requests / max(1, self.total_requests)) * 100,  # ✅ NEW
            "uptime_seconds": uptime,
            "requests_per_second": self.total_requests / max(1, uptime),
            "ip_rate_limiting": {  # ✅ NEW
                "enabled": self.ip_rate_limit_enabled,
                "max_per_minute": self.max_requests_per_ip,
                "burst_limit": self.ip_burst_limit,
                "tracked_ips": len(self.ip_request_counts)
            }
        }


class RateLimitManager:
    """
    Manager for rate limiting functionality.
    
    Provides a centralized interface for configuring and managing
    rate limiting across the application.
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.middleware: Optional[RateLimitMiddleware] = None
        self._enabled = False

    def _bind_live_middleware(self, middleware: RateLimitMiddleware) -> None:
        """Bind the live middleware instance created by FastAPI/Starlette."""
        self.middleware = middleware
        self._enabled = middleware.enabled
    
    def setup_middleware(
        self,
        app: FastAPI,
        enabled: bool = True,
        skip_paths: Optional[list] = None,
        custom_identifier: Optional[Callable[[Request], str]] = None,
        trust_forwarded_headers: bool = False,
        trusted_proxy_ips: Optional[list[str]] = None,
    ) -> "RateLimitManager":
        """Setup rate limiting middleware for FastAPI app"""
        app.add_middleware(RateLimitMiddleware, 
                          rate_limiter=self.rate_limiter,
                          enabled=enabled,
                          skip_paths=skip_paths,
                          custom_identifier=custom_identifier,
                          trust_forwarded_headers=trust_forwarded_headers,
                          trusted_proxy_ips=trusted_proxy_ips,
                          on_instance_ready=self._bind_live_middleware)
        
        self._enabled = enabled
        logger.info("Rate limiting middleware added to FastAPI app")
        # The actual middleware instance is created lazily when FastAPI builds the middleware stack.
        return self
    
    def enable(self):
        """Enable rate limiting"""
        if self.middleware:
            self.middleware.enabled = True
            self._enabled = True
            logger.info("Rate limiting enabled")
    
    def disable(self):
        """Disable rate limiting"""
        if self.middleware:
            self.middleware.enabled = False
            self._enabled = False
            logger.info("Rate limiting disabled")
    
    def is_enabled(self) -> bool:
        """Check if rate limiting is enabled"""
        return self._enabled
    
    async def check_rate_limit_manual(
        self,
        request: Request,
        endpoint_override: Optional[str] = None
    ) -> RateLimitResult:
        """
        Manually check rate limit for a request.
        
        Useful for custom rate limiting logic or testing.
        """
        client_ip = self._extract_client_ip_from_request(request)
        user_agent = request.headers.get("User-Agent")
        endpoint = endpoint_override or request.url.path
        
        return await self.rate_limiter.check_rate_limit(
            ip=client_ip,
            endpoint=endpoint,
            user_agent=user_agent
        )
    
    def _extract_client_ip_from_request(self, request: Request) -> str:
        """Extract client IP from request for manual checks (safe-by-default)."""
        return request.client.host if request.client else "unknown"
    
    async def get_comprehensive_stats(self) -> dict:
        """Get comprehensive rate limiting statistics"""
        stats = {
            "enabled": self._enabled,
            "rate_limiter_status": self.rate_limiter.get_status(),
            "rate_limiter_metrics": await self.rate_limiter.get_metrics(),
            "rate_limiter_config": self.rate_limiter.get_configuration_summary()
        }
        
        if self.middleware:
            stats["middleware_stats"] = self.middleware.get_stats()
        
        # Add monitoring data
        try:
            monitor = get_rate_limit_monitor()
            stats["monitoring_metrics"] = monitor.get_monitoring_metrics()
        except Exception as e:
            logger.warning(f"Failed to get monitoring metrics: {e}")
            stats["monitoring_error"] = str(e)
        
        return stats
    
    async def cleanup(self):
        """Cleanup rate limiting resources"""
        if self.rate_limiter:
            await self.rate_limiter.cleanup()
        logger.info("Rate limiting manager cleaned up")


# Global rate limit manager instance
rate_limit_manager = RateLimitManager()


def setup_rate_limiting(
    app: FastAPI,
    enabled: bool = True,
    skip_paths: Optional[list] = None,
    custom_identifier: Optional[Callable[[Request], str]] = None,
    trust_forwarded_headers: bool = False,
    trusted_proxy_ips: Optional[list[str]] = None,
) -> RateLimitManager:
    """
    Convenience function to setup rate limiting for a FastAPI app.
    
    Args:
        app: FastAPI application instance
        enabled: Whether rate limiting is enabled
        skip_paths: List of path rules skipped by rate limiting (exact match or path-segment prefix, never substring)
        custom_identifier: Custom function to extract client identifier
        trust_forwarded_headers: Trust X-Forwarded-For/X-Real-IP only when requests come from trusted proxies
        trusted_proxy_ips: Trusted proxy IPs/CIDRs. Empty list with trust_forwarded_headers=True trusts any proxy.
        
    Returns:
        RateLimitManager instance
    """
    rate_limit_manager.setup_middleware(
        app=app,
        enabled=enabled,
        skip_paths=skip_paths,
        custom_identifier=custom_identifier,
        trust_forwarded_headers=trust_forwarded_headers,
        trusted_proxy_ips=trusted_proxy_ips,
    )
    
    return rate_limit_manager


def get_rate_limit_manager() -> RateLimitManager:
    """Get the global rate limit manager instance"""
    return rate_limit_manager


# FastAPI dependency for rate limiting
async def rate_limit_dependency(request: Request) -> RateLimitResult:
    """
    FastAPI dependency for manual rate limiting checks.
    
    Usage:
        @app.get("/api/endpoint")
        async def endpoint(rate_limit: RateLimitResult = Depends(rate_limit_dependency)):
            if not rate_limit.allowed:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
    """
    return await rate_limit_manager.check_rate_limit_manual(request)


# Decorator for rate limiting specific endpoints
def rate_limited(
    requests_per_minute: Optional[int] = None,
    burst_multiplier: Optional[float] = None,
    custom_endpoint_name: Optional[str] = None
):
    """
    Decorator for applying custom rate limits to specific endpoints.
    
    Args:
        requests_per_minute: Custom rate limit for this endpoint
        burst_multiplier: Custom burst multiplier for this endpoint
        custom_endpoint_name: Custom name for rate limiting (defaults to function name)
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would need to be implemented with FastAPI's dependency system
            # For now, it's a placeholder for future enhancement
            return await func(*args, **kwargs)
        return wrapper
    return decorator
