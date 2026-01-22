"""
WeatherAPI Manager for AirTrace RU Backend

Manages integration with WeatherAPI.com for temperature, wind, and pressure data.
Provides unified weather data response format and automatic fallback handling.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import json

from connection_pool import get_connection_pool_manager, ServiceType, APIRequest, APIResponse
from config import config

logger = logging.getLogger(__name__)


@dataclass
class TemperatureData:
    """Temperature data from WeatherAPI"""
    celsius: float
    fahrenheit: float
    timestamp: datetime
    source: str = "weatherapi"


@dataclass
class WindData:
    """Wind data from WeatherAPI"""
    speed_kmh: float
    speed_mph: float
    direction_degrees: int
    direction_compass: str
    timestamp: datetime


@dataclass
class PressureData:
    """Pressure data from WeatherAPI"""
    pressure_mb: float
    pressure_in: float
    timestamp: datetime


@dataclass
class WeatherData:
    """Combined weather data"""
    temperature: TemperatureData
    wind: Optional[WindData]
    pressure: Optional[PressureData]
    location_name: str
    timestamp: datetime


@dataclass
class APIStatus:
    """WeatherAPI status information"""
    available: bool
    last_check: datetime
    error_message: Optional[str] = None
    requests_remaining: Optional[int] = None


class WeatherAPIManager:
    """
    Manager for WeatherAPI.com integration.
    
    Provides temperature, wind, and pressure data with automatic fallback
    when WeatherAPI is unavailable. Uses connection pooling for optimal performance.
    """
    
    def __init__(self):
        self.enabled = config.weather_api.enabled
        self.api_key = config.weather_api.api_key
        self.base_url = config.weather_api.base_url
        self.fallback_enabled = config.weather_api.fallback_enabled
        
        # API status tracking
        self.last_status_check = datetime.min.replace(tzinfo=timezone.utc)
        self.api_available = True
        self.last_error = None
        
        # Request statistics
        self.requests_made = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        if not self.enabled:
            logger.info("WeatherAPI manager initialized but disabled")
        else:
            logger.info("WeatherAPI manager initialized and enabled")
    
    def _build_request_url(self, endpoint: str, lat: float, lon: float, **params) -> str:
        """Build WeatherAPI request URL"""
        base_params = {
            "key": self.api_key,
            "q": f"{lat},{lon}",
            "aqi": "no"  # We get AQI from Open-Meteo
        }
        base_params.update(params)
        
        param_string = "&".join([f"{k}={v}" for k, v in base_params.items()])
        return f"{self.base_url}/{endpoint}.json?{param_string}"
    
    async def _make_api_request(self, endpoint: str, lat: float, lon: float, **params) -> Dict[str, Any]:
        """Make request to WeatherAPI with error handling"""
        if not self.enabled:
            raise Exception("WeatherAPI is disabled")
        
        if not self.api_key:
            raise Exception("WeatherAPI key not configured")
        
        try:
            url = self._build_request_url(endpoint, lat, lon, **params)
            
            # Use connection pool for the request
            api_request = APIRequest(
                method="GET",
                url=url,
                timeout=config.weather_api.timeout
            )
            
            response = await get_connection_pool_manager().execute_request(
                ServiceType.WEATHER_API,
                api_request
            )
            
            self.requests_made += 1
            
            if response.status_code == 200:
                self.successful_requests += 1
                self.api_available = True
                self.last_error = None
                return response.data
            else:
                self.failed_requests += 1
                error_msg = f"WeatherAPI returned status {response.status_code}"
                
                # Try to extract error message from response
                if isinstance(response.data, dict) and "error" in response.data:
                    error_msg = response.data["error"].get("message", error_msg)
                
                self.last_error = error_msg
                raise Exception(error_msg)
                
        except Exception as e:
            self.failed_requests += 1
            self.last_error = str(e)
            
            # Mark API as unavailable on repeated failures
            if self.failed_requests > self.successful_requests and self.requests_made > 5:
                self.api_available = False
            
            logger.error(f"WeatherAPI request failed: {e}")
            raise
    
    async def get_temperature(self, lat: float, lon: float) -> TemperatureData:
        """Get temperature data for location"""
        try:
            data = await self._make_api_request("current", lat, lon)
            
            current = data.get("current", {})
            if not current:
                raise ValueError("No current weather data in response")
            
            return TemperatureData(
                celsius=current.get("temp_c", 0.0),
                fahrenheit=current.get("temp_f", 0.0),
                timestamp=datetime.now(timezone.utc),
                source="weatherapi"
            )
            
        except Exception as e:
            logger.error(f"Failed to get temperature data: {e}")
            if self.fallback_enabled:
                # Return default/fallback temperature data
                return TemperatureData(
                    celsius=0.0,
                    fahrenheit=32.0,
                    timestamp=datetime.now(timezone.utc),
                    source="fallback"
                )
            raise
    
    async def get_wind_data(self, lat: float, lon: float) -> WindData:
        """Get wind data for location"""
        try:
            data = await self._make_api_request("current", lat, lon)
            
            current = data.get("current", {})
            if not current:
                raise ValueError("No current weather data in response")
            
            return WindData(
                speed_kmh=current.get("wind_kph", 0.0),
                speed_mph=current.get("wind_mph", 0.0),
                direction_degrees=current.get("wind_degree", 0),
                direction_compass=current.get("wind_dir", "N"),
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to get wind data: {e}")
            if self.fallback_enabled:
                # Return default/fallback wind data
                return WindData(
                    speed_kmh=0.0,
                    speed_mph=0.0,
                    direction_degrees=0,
                    direction_compass="N",
                    timestamp=datetime.now(timezone.utc)
                )
            raise
    
    async def get_pressure_data(self, lat: float, lon: float) -> PressureData:
        """Get atmospheric pressure data for location"""
        try:
            data = await self._make_api_request("current", lat, lon)
            
            current = data.get("current", {})
            if not current:
                raise ValueError("No current weather data in response")
            
            return PressureData(
                pressure_mb=current.get("pressure_mb", 1013.25),
                pressure_in=current.get("pressure_in", 29.92),
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to get pressure data: {e}")
            if self.fallback_enabled:
                # Return default/fallback pressure data (standard atmospheric pressure)
                return PressureData(
                    pressure_mb=1013.25,
                    pressure_in=29.92,
                    timestamp=datetime.now(timezone.utc)
                )
            raise
    
    async def get_combined_weather(self, lat: float, lon: float) -> WeatherData:
        """Get combined weather data (temperature, wind, pressure) for location"""
        try:
            # Make single API call for all data
            data = await self._make_api_request("current", lat, lon)
            
            current = data.get("current", {})
            location = data.get("location", {})
            
            if not current:
                raise ValueError("No current weather data in response")
            
            # Extract temperature data
            temperature = TemperatureData(
                celsius=current.get("temp_c", 0.0),
                fahrenheit=current.get("temp_f", 0.0),
                timestamp=datetime.now(timezone.utc),
                source="weatherapi"
            )
            
            # Extract wind data
            wind = WindData(
                speed_kmh=current.get("wind_kph", 0.0),
                speed_mph=current.get("wind_mph", 0.0),
                direction_degrees=current.get("wind_degree", 0),
                direction_compass=current.get("wind_dir", "N"),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Extract pressure data
            pressure = PressureData(
                pressure_mb=current.get("pressure_mb", 1013.25),
                pressure_in=current.get("pressure_in", 29.92),
                timestamp=datetime.now(timezone.utc)
            )
            
            return WeatherData(
                temperature=temperature,
                wind=wind,
                pressure=pressure,
                location_name=location.get("name", f"{lat:.2f},{lon:.2f}"),
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to get combined weather data: {e}")
            if self.fallback_enabled:
                # Return fallback data
                return WeatherData(
                    temperature=TemperatureData(
                        celsius=0.0,
                        fahrenheit=32.0,
                        timestamp=datetime.now(timezone.utc),
                        source="fallback"
                    ),
                    wind=WindData(
                        speed_kmh=0.0,
                        speed_mph=0.0,
                        direction_degrees=0,
                        direction_compass="N",
                        timestamp=datetime.now(timezone.utc)
                    ),
                    pressure=PressureData(
                        pressure_mb=1013.25,
                        pressure_in=29.92,
                        timestamp=datetime.now(timezone.utc)
                    ),
                    location_name=f"{lat:.2f},{lon:.2f}",
                    timestamp=datetime.now(timezone.utc)
                )
            raise
    
    def configure_api_key(self, api_key: str):
        """Configure WeatherAPI key"""
        self.api_key = api_key
        if api_key:
            self.enabled = True
            logger.info("WeatherAPI key configured and enabled")
        else:
            self.enabled = False
            logger.warning("WeatherAPI key removed - disabling WeatherAPI")
    
    async def get_api_status(self) -> APIStatus:
        """Get current API status"""
        # Update status if it's been a while since last check
        now = datetime.now(timezone.utc)
        if (now - self.last_status_check).total_seconds() > 300:  # 5 minutes
            await self._check_api_status()
        
        return APIStatus(
            available=self.api_available and self.enabled,
            last_check=self.last_status_check,
            error_message=self.last_error,
            requests_remaining=None  # WeatherAPI doesn't provide this in response
        )
    
    async def _check_api_status(self):
        """Check API status with a simple request"""
        try:
            # Simple status check request
            await self._make_api_request("current", 55.7558, 37.6176)
            self.api_available = True
            self.last_error = None
        except Exception as e:
            self.api_available = False
            self.last_error = str(e)
        finally:
            self.last_status_check = datetime.now(timezone.utc)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get request statistics"""
        success_rate = 0.0
        if self.requests_made > 0:
            success_rate = self.successful_requests / self.requests_made
        
        return {
            "enabled": self.enabled,
            "api_available": self.api_available,
            "requests_made": self.requests_made,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "last_error": self.last_error,
            "last_status_check": self.last_status_check.isoformat() if self.last_status_check else None
        }


# Global WeatherAPI manager instance
weather_api_manager = WeatherAPIManager()