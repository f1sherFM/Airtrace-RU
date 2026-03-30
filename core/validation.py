"""
✅ FIX #6: Centralized validation module for AirTrace RU Backend

Provides unified validation functions for coordinates, pollutants, and other data
to avoid code duplication and ensure consistency across the application.
"""

import logging
import re
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class CoordinateValidator:
    """
    Centralized coordinate validation.
    
    ✅ FIX #6: Replaces duplicate validation logic across services.py and utils.py
    """
    
    # Russian territory boundaries (approximate)
    RUSSIA_LAT_MIN = 41.0
    RUSSIA_LAT_MAX = 82.0
    RUSSIA_LON_MIN = 19.0
    RUSSIA_LON_MAX = 169.0
    
    @staticmethod
    def validate_basic(lat: float, lon: float) -> Tuple[bool, str]:
        """
        Basic coordinate validation (world boundaries).
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(lat, (int, float)):
            return False, f"Latitude must be a number, got {type(lat).__name__}"
        
        if not isinstance(lon, (int, float)):
            return False, f"Longitude must be a number, got {type(lon).__name__}"
        
        if not (-90 <= lat <= 90):
            return False, f"Latitude must be between -90 and 90, got {lat}"
        
        if not (-180 <= lon <= 180):
            return False, f"Longitude must be between -180 and 180, got {lon}"
        
        return True, ""
    
    @staticmethod
    def validate_russian_territory(lat: float, lon: float) -> Tuple[bool, str]:
        """
        Validate coordinates are within Russian territory.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # First check basic validation
        is_valid, error = CoordinateValidator.validate_basic(lat, lon)
        if not is_valid:
            return False, error
        
        # Check Russian territory boundaries
        if not (CoordinateValidator.RUSSIA_LAT_MIN <= lat <= CoordinateValidator.RUSSIA_LAT_MAX):
            return False, f"Latitude {lat} is outside Russian territory (41-82°N)"
        
        if not (CoordinateValidator.RUSSIA_LON_MIN <= lon <= CoordinateValidator.RUSSIA_LON_MAX):
            return False, f"Longitude {lon} is outside Russian territory (19-169°E)"
        
        return True, ""
    
    @staticmethod
    def validate(lat: float, lon: float, strict: bool = False) -> Tuple[bool, str]:
        """
        Unified coordinate validation.
        
        Args:
            lat: Latitude
            lon: Longitude
            strict: If True, validates Russian territory boundaries
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if strict:
            return CoordinateValidator.validate_russian_territory(lat, lon)
        else:
            return CoordinateValidator.validate_basic(lat, lon)
    
    @staticmethod
    def validate_or_raise(lat: float, lon: float, strict: bool = False):
        """
        Validate coordinates and raise ValidationError if invalid.
        
        Args:
            lat: Latitude
            lon: Longitude
            strict: If True, validates Russian territory boundaries
            
        Raises:
            ValidationError: If coordinates are invalid
        """
        is_valid, error = CoordinateValidator.validate(lat, lon, strict)
        if not is_valid:
            raise ValidationError(error)


class PollutantValidator:
    """
    Centralized pollutant data validation.
    
    ✅ FIX #6: Ensures pollutant values are valid and within reasonable ranges
    """
    
    # Maximum reasonable values for pollutants (μg/m³)
    MAX_VALUES = {
        'pm2_5': 1000.0,  # Extreme pollution
        'pm10': 2000.0,
        'no2': 1000.0,
        'so2': 1000.0,
        'o3': 500.0
    }
    
    @staticmethod
    def validate_value(pollutant: str, value: Optional[float]) -> Tuple[bool, str]:
        """
        Validate a single pollutant value.
        
        Args:
            pollutant: Pollutant name (pm2_5, pm10, no2, so2, o3)
            value: Pollutant concentration in μg/m³
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if value is None:
            return True, ""  # None is acceptable (missing data)
        
        if not isinstance(value, (int, float)):
            return False, f"{pollutant}: value must be a number, got {type(value).__name__}"
        
        if value < 0:
            return False, f"{pollutant}: value cannot be negative, got {value}"
        
        max_value = PollutantValidator.MAX_VALUES.get(pollutant, 10000.0)
        if value > max_value:
            return False, f"{pollutant}: value {value} exceeds maximum {max_value} μg/m³"
        
        return True, ""
    
    @staticmethod
    def validate_dict(pollutants: Dict[str, Optional[float]]) -> Tuple[bool, str]:
        """
        Validate a dictionary of pollutant values.
        
        Args:
            pollutants: Dictionary of pollutant name -> concentration
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(pollutants, dict):
            return False, f"Pollutants must be a dictionary, got {type(pollutants).__name__}"
        
        if not pollutants:
            return False, "Pollutants dictionary cannot be empty"
        
        for pollutant, value in pollutants.items():
            is_valid, error = PollutantValidator.validate_value(pollutant, value)
            if not is_valid:
                return False, error
        
        return True, ""
    
    @staticmethod
    def validate_or_raise(pollutants: Dict[str, Optional[float]]):
        """
        Validate pollutants and raise ValidationError if invalid.
        
        Args:
            pollutants: Dictionary of pollutant name -> concentration
            
        Raises:
            ValidationError: If pollutants are invalid
        """
        is_valid, error = PollutantValidator.validate_dict(pollutants)
        if not is_valid:
            raise ValidationError(error)


class APIResponseValidator:
    """
    Centralized API response validation.
    
    ✅ FIX #6: Validates external API responses for required fields and structure
    """
    
    @staticmethod
    def validate_open_meteo_response(data: Any) -> Tuple[bool, str]:
        """
        Validate Open-Meteo API response structure.
        
        Args:
            data: API response data
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "API response must be a dictionary"
        
        # Check required fields
        required_fields = ["latitude", "longitude"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate coordinate types
        try:
            lat = float(data["latitude"])
            lon = float(data["longitude"])
        except (ValueError, TypeError) as e:
            return False, f"Invalid coordinate types: {e}"
        
        # Validate coordinate values
        is_valid, error = CoordinateValidator.validate_basic(lat, lon)
        if not is_valid:
            return False, f"Invalid coordinates in response: {error}"
        
        return True, ""
    
    @staticmethod
    def validate_or_raise(data: Any, api_type: str = "open_meteo"):
        """
        Validate API response and raise ValidationError if invalid.
        
        Args:
            data: API response data
            api_type: Type of API (open_meteo, weather_api, etc.)
            
        Raises:
            ValidationError: If response is invalid
        """
        if api_type == "open_meteo":
            is_valid, error = APIResponseValidator.validate_open_meteo_response(data)
        else:
            is_valid, error = True, ""  # Unknown API type, skip validation
        
        if not is_valid:
            raise ValidationError(f"Invalid {api_type} API response: {error}")


class ConfigValidator:
    """
    Centralized configuration validation.
    
    ✅ FIX #6: Validates configuration values for consistency
    """
    
    @staticmethod
    def validate_ttl(ttl: int, min_value: int = 0, max_value: int = 86400) -> Tuple[bool, str]:
        """
        Validate TTL (Time To Live) value.
        
        Args:
            ttl: TTL in seconds
            min_value: Minimum allowed TTL
            max_value: Maximum allowed TTL (default 24 hours)
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(ttl, int):
            return False, f"TTL must be an integer, got {type(ttl).__name__}"
        
        if ttl < min_value:
            return False, f"TTL {ttl} is below minimum {min_value}"
        
        if ttl > max_value:
            return False, f"TTL {ttl} exceeds maximum {max_value}"
        
        return True, ""
    
    @staticmethod
    def validate_rate_limit(limit: int, min_value: int = 1, max_value: int = 10000) -> Tuple[bool, str]:
        """
        Validate rate limit value.
        
        Args:
            limit: Rate limit (requests per time window)
            min_value: Minimum allowed limit
            max_value: Maximum allowed limit
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(limit, int):
            return False, f"Rate limit must be an integer, got {type(limit).__name__}"
        
        if limit < min_value:
            return False, f"Rate limit {limit} is below minimum {min_value}"
        
        if limit > max_value:
            return False, f"Rate limit {limit} exceeds maximum {max_value}"
        
        return True, ""


# Convenience functions for backward compatibility
def validate_coordinates(lat: float, lon: float, strict: bool = False) -> bool:
    """
    Validate coordinates (backward compatible function).
    
    Args:
        lat: Latitude
        lon: Longitude
        strict: If True, validates Russian territory boundaries
        
    Returns:
        bool: True if valid
    """
    is_valid, _ = CoordinateValidator.validate(lat, lon, strict)
    return is_valid


def validate_pollutants(pollutants: Dict[str, Optional[float]]) -> bool:
    """
    Validate pollutants (backward compatible function).
    
    Args:
        pollutants: Dictionary of pollutant name -> concentration
        
    Returns:
        bool: True if valid
    """
    is_valid, _ = PollutantValidator.validate_dict(pollutants)
    return is_valid
