"""
Tests for medium priority fixes

Tests for:
- #8: IP-based rate limiting
- #9: Coordinate masking in logs
- #7: JSON serialization optimization
- #6: Centralized validation
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request
from fastapi.testclient import TestClient

# Test Fix #6: Centralized Validation
class TestCoordinateValidator:
    """Test centralized coordinate validation"""
    
    def test_valid_coordinates(self):
        """Test valid coordinates"""
        from validators import CoordinateValidator
        
        is_valid, error = CoordinateValidator.validate(55.7558, 37.6176)
        assert is_valid
        assert error == ""
    
    def test_invalid_latitude_too_high(self):
        """Test latitude above 90"""
        from validators import CoordinateValidator
        
        is_valid, error = CoordinateValidator.validate(100, 37.6176)
        assert not is_valid
        assert "Latitude" in error
        assert "90" in error
    
    def test_invalid_latitude_too_low(self):
        """Test latitude below -90"""
        from validators import CoordinateValidator
        
        is_valid, error = CoordinateValidator.validate(-100, 37.6176)
        assert not is_valid
        assert "Latitude" in error
    
    def test_invalid_longitude(self):
        """Test invalid longitude"""
        from validators import CoordinateValidator
        
        is_valid, error = CoordinateValidator.validate(55.7558, 200)
        assert not is_valid
        assert "Longitude" in error
    
    def test_russian_territory_moscow(self):
        """Test Moscow coordinates (should be valid)"""
        from validators import CoordinateValidator
        
        is_valid, error = CoordinateValidator.validate_russian_territory(55.7558, 37.6176)
        assert is_valid
    
    def test_russian_territory_vladivostok(self):
        """Test Vladivostok coordinates (should be valid)"""
        from validators import CoordinateValidator
        
        is_valid, error = CoordinateValidator.validate_russian_territory(43.1056, 131.8735)
        assert is_valid
    
    def test_russian_territory_outside(self):
        """Test coordinates outside Russia (New York)"""
        from validators import CoordinateValidator
        
        is_valid, error = CoordinateValidator.validate_russian_territory(40.7128, -74.0060)
        assert not is_valid
        assert "outside Russian territory" in error
    
    def test_validate_or_raise_valid(self):
        """Test validate_or_raise with valid coordinates"""
        from validators import CoordinateValidator
        
        # Should not raise
        CoordinateValidator.validate_or_raise(55.7558, 37.6176)
    
    def test_validate_or_raise_invalid(self):
        """Test validate_or_raise with invalid coordinates"""
        from validators import CoordinateValidator, ValidationError
        
        with pytest.raises(ValidationError):
            CoordinateValidator.validate_or_raise(100, 37.6176)


class TestPollutantValidator:
    """Test centralized pollutant validation"""
    
    def test_valid_pollutants(self):
        """Test valid pollutant values"""
        from validators import PollutantValidator
        
        pollutants = {"pm2_5": 25.0, "pm10": 50.0, "no2": 30.0}
        is_valid, error = PollutantValidator.validate_dict(pollutants)
        assert is_valid
        assert error == ""
    
    def test_negative_value(self):
        """Test negative pollutant value"""
        from validators import PollutantValidator
        
        pollutants = {"pm2_5": -10.0}
        is_valid, error = PollutantValidator.validate_dict(pollutants)
        assert not is_valid
        assert "negative" in error.lower()
    
    def test_excessive_value(self):
        """Test excessively high pollutant value"""
        from validators import PollutantValidator
        
        pollutants = {"pm2_5": 10000.0}  # Way too high
        is_valid, error = PollutantValidator.validate_dict(pollutants)
        assert not is_valid
        assert "exceeds maximum" in error
    
    def test_none_value_allowed(self):
        """Test that None values are allowed (missing data)"""
        from validators import PollutantValidator
        
        pollutants = {"pm2_5": 25.0, "pm10": None}
        is_valid, error = PollutantValidator.validate_dict(pollutants)
        assert is_valid
    
    def test_empty_dict(self):
        """Test empty pollutants dictionary"""
        from validators import PollutantValidator
        
        pollutants = {}
        is_valid, error = PollutantValidator.validate_dict(pollutants)
        assert not is_valid
        assert "empty" in error.lower()
    
    def test_validate_or_raise(self):
        """Test validate_or_raise"""
        from validators import PollutantValidator, ValidationError
        
        # Valid - should not raise
        pollutants = {"pm2_5": 25.0}
        PollutantValidator.validate_or_raise(pollutants)
        
        # Invalid - should raise
        with pytest.raises(ValidationError):
            PollutantValidator.validate_or_raise({"pm2_5": -10.0})


class TestAPIResponseValidator:
    """Test API response validation"""
    
    def test_valid_open_meteo_response(self):
        """Test valid Open-Meteo response"""
        from validators import APIResponseValidator
        
        response = {
            "latitude": 55.7558,
            "longitude": 37.6176,
            "current": {"pm2_5": 25.0}
        }
        is_valid, error = APIResponseValidator.validate_open_meteo_response(response)
        assert is_valid
    
    def test_missing_latitude(self):
        """Test response missing latitude"""
        from validators import APIResponseValidator
        
        response = {"longitude": 37.6176}
        is_valid, error = APIResponseValidator.validate_open_meteo_response(response)
        assert not is_valid
        assert "latitude" in error.lower()
    
    def test_invalid_coordinate_type(self):
        """Test response with invalid coordinate type"""
        from validators import APIResponseValidator
        
        response = {"latitude": "invalid", "longitude": 37.6176}
        is_valid, error = APIResponseValidator.validate_open_meteo_response(response)
        assert not is_valid


# Test Fix #9: Coordinate Masking
class TestCoordinateMasking:
    """Test coordinate masking in logs"""
    
    def test_mask_url_parameters(self):
        """Test masking coordinates in URL parameters"""
        from services import mask_coordinates
        
        url = "http://api.example.com?latitude=55.7558&longitude=37.6176"
        masked = mask_coordinates(url)
        
        assert "55.7558" not in masked
        assert "37.6176" not in masked
        assert "latitude=***" in masked
        assert "longitude=***" in masked
    
    def test_mask_short_parameters(self):
        """Test masking short parameter names"""
        from services import mask_coordinates
        
        url = "http://api.example.com?lat=55.7558&lon=37.6176"
        masked = mask_coordinates(url)
        
        assert "55.7558" not in masked
        assert "37.6176" not in masked
        assert "lat=***" in masked
        assert "lon=***" in masked
    
    def test_mask_tuple_coordinates(self):
        """Test masking coordinate tuples"""
        from services import mask_coordinates
        
        text = "Location: (55.7558, 37.6176)"
        masked = mask_coordinates(text)
        
        assert "55.7558" not in masked
        assert "37.6176" not in masked
        assert "(***, ***)" in masked
    
    def test_mask_json_coordinates(self):
        """Test masking coordinates in JSON"""
        from services import mask_coordinates
        
        json_text = '{"latitude": 55.7558, "longitude": 37.6176}'
        masked = mask_coordinates(json_text)
        
        assert "55.7558" not in masked
        assert "37.6176" not in masked
        assert '"latitude": ***' in masked
        assert '"longitude": ***' in masked
    
    def test_mask_negative_coordinates(self):
        """Test masking negative coordinates"""
        from services import mask_coordinates
        
        url = "http://api.example.com?lat=-33.8688&lon=-151.2093"
        masked = mask_coordinates(url)
        
        assert "-33.8688" not in masked
        assert "-151.2093" not in masked
        assert "lat=***" in masked
    
    def test_mask_preserves_other_text(self):
        """Test that masking preserves non-coordinate text"""
        from services import mask_coordinates
        
        text = "Request for lat=55.7558 with status=success"
        masked = mask_coordinates(text)
        
        assert "Request for" in masked
        assert "with status=success" in masked
        assert "55.7558" not in masked


# Test Fix #7: JSON Optimization
class TestJSONOptimization:
    """Test JSON serialization optimization"""
    
    def test_json_dumps_basic(self):
        """Test json_dumps with basic data"""
        from cache import json_dumps
        
        data = {"key": "value", "number": 42}
        result = json_dumps(data)
        
        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result
    
    def test_json_loads_basic(self):
        """Test json_loads with basic data"""
        from cache import json_loads
        
        json_str = '{"key": "value", "number": 42}'
        result = json_loads(json_str)
        
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["number"] == 42
    
    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip"""
        from cache import json_dumps, json_loads
        
        original = {"key": "value", "nested": {"number": 42}}
        serialized = json_dumps(original)
        deserialized = json_loads(serialized)
        
        assert deserialized == original
    
    def test_json_loads_bytes(self):
        """Test json_loads with bytes input"""
        from cache import json_loads
        
        json_bytes = b'{"key": "value"}'
        result = json_loads(json_bytes)
        
        assert isinstance(result, dict)
        assert result["key"] == "value"
    
    def test_json_with_datetime(self):
        """Test JSON serialization with datetime (uses default=str)"""
        from cache import json_dumps
        from datetime import datetime
        
        data = {"timestamp": datetime.now()}
        result = json_dumps(data)
        
        assert isinstance(result, str)
        assert "timestamp" in result


# Test Fix #8: IP Rate Limiting
class TestIPRateLimiting:
    """Test IP-based rate limiting"""
    
    @pytest.mark.asyncio
    async def test_ip_rate_limit_allows_under_limit(self):
        """Test that requests under limit are allowed"""
        from rate_limit_middleware import RateLimitMiddleware
        from fastapi import FastAPI
        
        app = FastAPI()
        middleware = RateLimitMiddleware(
            app=app,
            ip_rate_limit_enabled=True,
            max_requests_per_ip_per_minute=10
        )
        
        # Make 5 requests (under limit of 10)
        for i in range(5):
            allowed, retry_after = middleware._check_ip_rate_limit("192.168.1.1")
            assert allowed
            assert retry_after == 0
    
    @pytest.mark.asyncio
    async def test_ip_rate_limit_blocks_over_limit(self):
        """Test that requests over limit are blocked"""
        from rate_limit_middleware import RateLimitMiddleware
        from fastapi import FastAPI
        
        app = FastAPI()
        middleware = RateLimitMiddleware(
            app=app,
            ip_rate_limit_enabled=True,
            max_requests_per_ip_per_minute=5
        )
        
        # Make 5 requests (at limit)
        for i in range(5):
            allowed, _ = middleware._check_ip_rate_limit("192.168.1.2")
            assert allowed
        
        # 6th request should be blocked
        allowed, retry_after = middleware._check_ip_rate_limit("192.168.1.2")
        assert not allowed
        assert retry_after > 0
    
    @pytest.mark.asyncio
    async def test_ip_rate_limit_burst_protection(self):
        """Test burst protection"""
        from rate_limit_middleware import RateLimitMiddleware
        from fastapi import FastAPI
        
        app = FastAPI()
        middleware = RateLimitMiddleware(
            app=app,
            ip_rate_limit_enabled=True,
            max_requests_per_ip_per_minute=200,  # High regular limit
            ip_burst_multiplier=0.75  # 150 burst limit (200 * 0.75)
        )
        
        # Make 150 rapid requests (at burst limit)
        for i in range(150):
            allowed, _ = middleware._check_ip_rate_limit("192.168.1.3")
            assert allowed
        
        # 151st request should be blocked by burst limit
        allowed, retry_after = middleware._check_ip_rate_limit("192.168.1.3")
        assert not allowed
        assert retry_after <= 10  # Burst window is 10 seconds
    
    @pytest.mark.asyncio
    async def test_ip_rate_limit_different_ips(self):
        """Test that different IPs have separate limits"""
        from rate_limit_middleware import RateLimitMiddleware
        from fastapi import FastAPI
        
        app = FastAPI()
        middleware = RateLimitMiddleware(
            app=app,
            ip_rate_limit_enabled=True,
            max_requests_per_ip_per_minute=5
        )
        
        # IP 1: Make 5 requests
        for i in range(5):
            allowed, _ = middleware._check_ip_rate_limit("192.168.1.10")
            assert allowed
        
        # IP 1: 6th request blocked
        allowed, _ = middleware._check_ip_rate_limit("192.168.1.10")
        assert not allowed
        
        # IP 2: Should still be allowed
        allowed, _ = middleware._check_ip_rate_limit("192.168.1.11")
        assert allowed
    
    @pytest.mark.asyncio
    async def test_ip_rate_limit_cleanup(self):
        """Test that old IP entries are cleaned up"""
        from rate_limit_middleware import RateLimitMiddleware
        from fastapi import FastAPI
        import time
        
        app = FastAPI()
        middleware = RateLimitMiddleware(
            app=app,
            ip_rate_limit_enabled=True,
            max_requests_per_ip_per_minute=10
        )
        
        # Add 1100 IPs (over the 1000 limit)
        for i in range(1100):
            middleware._check_ip_rate_limit(f"192.168.{i // 256}.{i % 256}")
        
        # Cleanup happens during check, so make one more request
        middleware._check_ip_rate_limit("192.168.255.255")
        
        # Should have cleaned up to ~1000 or less
        assert len(middleware.ip_request_counts) <= 1001  # Allow small margin


# Integration tests
class TestMediumPriorityIntegration:
    """Integration tests for medium priority fixes"""
    
    def test_validators_integration(self):
        """Test that validators work together"""
        from validators import CoordinateValidator, PollutantValidator
        
        # Valid coordinates and pollutants
        lat, lon = 55.7558, 37.6176
        pollutants = {"pm2_5": 25.0, "pm10": 50.0}
        
        coord_valid, _ = CoordinateValidator.validate(lat, lon)
        poll_valid, _ = PollutantValidator.validate_dict(pollutants)
        
        assert coord_valid
        assert poll_valid
    
    def test_masking_with_validation(self):
        """Test coordinate masking with validation"""
        from validators import CoordinateValidator
        from services import mask_coordinates
        
        lat, lon = 55.7558, 37.6176
        
        # Validate
        is_valid, _ = CoordinateValidator.validate(lat, lon)
        assert is_valid
        
        # Mask for logging
        log_message = f"Valid coordinates: lat={lat}, lon={lon}"
        masked = mask_coordinates(log_message)
        
        assert "lat=***" in masked
        assert str(lat) not in masked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
