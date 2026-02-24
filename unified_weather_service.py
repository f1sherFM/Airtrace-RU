"""
Unified Weather Service for AirTrace RU Backend

Combines air quality data from Open-Meteo API with weather data from WeatherAPI.com
to provide comprehensive environmental information with fallback handling.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from schemas import (
    AirQualityData, 
    WeatherInfo,
    TemperatureData,
    WindData,
    PressureData,
    LocationInfo,
    ResponseMetadata,
)
from services import AirQualityService
from weather_api_manager import weather_api_manager
from cache import MultiLevelCacheManager, CacheLevel
from config import config
from privacy_compliance_validator import validate_cache_key_privacy, validate_metrics_privacy

logger = logging.getLogger(__name__)


class UnifiedWeatherService:
    """
    Unified service that combines air quality data with weather information.
    
    Provides a single interface for getting comprehensive environmental data
    with automatic fallback when WeatherAPI is unavailable.
    """
    
    def __init__(self):
        self.air_quality_service = AirQualityService()
        self.cache_manager = MultiLevelCacheManager()
        
        # Cache TTL for combined data (shorter than individual components)
        self.combined_cache_ttl = 600  # 10 minutes
        
        # Statistics
        self.requests_with_weather = 0
        self.requests_fallback_only = 0
        self.weather_api_failures = 0
        
    def _generate_combined_cache_key(self, lat: float, lon: float) -> str:
        """Generate cache key for combined weather data"""
        # Use higher precision for combined data to avoid conflicts
        rounded_lat = round(lat, 4)
        rounded_lon = round(lon, 4)
        cache_key = f"combined:{rounded_lat}:{rounded_lon}"
        
        # Validate cache key privacy compliance
        if not validate_cache_key_privacy(cache_key, "UnifiedWeatherService._generate_combined_cache_key"):
            logger.warning(f"Combined cache key privacy validation failed for key: {cache_key[:20]}...")
        
        return cache_key
    
    async def get_current_combined_data(self, lat: float, lon: float) -> AirQualityData:
        """
        Get current air quality data combined with weather information.
        
        First checks cache for combined data, then fetches from both APIs
        and combines the results. Falls back gracefully if WeatherAPI fails.
        """
        # Check cache for combined data first
        cache_key = self._generate_combined_cache_key(lat, lon)
        try:
            cached_data = await self.cache_manager.get(lat, lon, [CacheLevel.L1, CacheLevel.L2])
            if cached_data and isinstance(cached_data, dict) and 'combined' in cached_data:
                logger.debug(f"Combined data cache hit for {lat}, {lon}")
                return AirQualityData(**cached_data['combined'])
        except Exception as e:
            logger.warning(f"Combined data cache get failed: {e}")
        
        # Fetch air quality data (this is the primary data)
        try:
            air_quality_data = await self.air_quality_service.get_current_air_quality(lat, lon)
        except Exception as e:
            logger.error(f"Failed to get air quality data: {e}")
            raise
        
        # Try to enhance with weather data
        weather_info = await self._get_weather_info_safe(lat, lon)
        
        # Combine the data
        combined_data = AirQualityData(
            timestamp=air_quality_data.timestamp,
            location=air_quality_data.location,
            aqi=air_quality_data.aqi,
            pollutants=air_quality_data.pollutants,
            weather=weather_info,
            recommendations=air_quality_data.recommendations,
            nmu_risk=air_quality_data.nmu_risk,
            health_warnings=air_quality_data.health_warnings,
            metadata=ResponseMetadata(**air_quality_data.metadata.model_dump()),
        )
        
        # Cache the combined result
        try:
            cache_data = {'combined': combined_data.model_dump()}
            await self.cache_manager.set(
                lat, lon, 
                cache_data, 
                ttl=self.combined_cache_ttl,
                levels=[CacheLevel.L1, CacheLevel.L2]
            )
        except Exception as e:
            logger.warning(f"Failed to cache combined data: {e}")
        
        # Update statistics
        if weather_info:
            self.requests_with_weather += 1
        else:
            self.requests_fallback_only += 1
        
        return combined_data
    
    async def get_forecast_combined_data(self, lat: float, lon: float, hours: int = 24) -> List[AirQualityData]:
        """
        Get forecast air quality data combined with weather information.
        
        Returns 24-hour forecast with weather data included where available.
        """
        # Get air quality forecast (this is the primary data)
        try:
            forecast_data = await self.air_quality_service.get_forecast_air_quality(lat, lon, hours=hours)
        except Exception as e:
            logger.error(f"Failed to get air quality forecast: {e}")
            raise
        
        # Try to enhance with weather data (current weather applies to forecast)
        weather_info = await self._get_weather_info_safe(lat, lon)
        
        # Add weather info to each forecast item
        enhanced_forecast = []
        for forecast_item in forecast_data:
            enhanced_item = AirQualityData(
                timestamp=forecast_item.timestamp,
                location=forecast_item.location,
                aqi=forecast_item.aqi,
                pollutants=forecast_item.pollutants,
                weather=weather_info,
                recommendations=forecast_item.recommendations,
                nmu_risk=forecast_item.nmu_risk,
                health_warnings=forecast_item.health_warnings,
                metadata=ResponseMetadata(**forecast_item.metadata.model_dump()),
            )
            enhanced_forecast.append(enhanced_item)
        
        # Update statistics
        if weather_info:
            self.requests_with_weather += 1
        else:
            self.requests_fallback_only += 1
        
        return enhanced_forecast
    
    async def _get_weather_info_safe(self, lat: float, lon: float) -> Optional[WeatherInfo]:
        """
        Safely get weather information with fallback handling.
        
        Returns None if WeatherAPI is disabled or fails, allowing the system
        to continue with air quality data only.
        """
        if not config.weather_api.enabled:
            logger.debug("WeatherAPI disabled, skipping weather data")
            return None
        
        try:
            # Get combined weather data from WeatherAPI
            weather_data = await weather_api_manager.get_combined_weather(lat, lon)
            
            # Convert to schema format
            weather_info = WeatherInfo(
                temperature=TemperatureData(
                    celsius=weather_data.temperature.celsius,
                    fahrenheit=weather_data.temperature.fahrenheit,
                    timestamp=weather_data.temperature.timestamp,
                    source=weather_data.temperature.source
                ),
                wind=WindData(
                    speed_kmh=weather_data.wind.speed_kmh,
                    speed_mph=weather_data.wind.speed_mph,
                    direction_degrees=weather_data.wind.direction_degrees,
                    direction_compass=weather_data.wind.direction_compass,
                    timestamp=weather_data.wind.timestamp
                ) if weather_data.wind else None,
                pressure=PressureData(
                    pressure_mb=weather_data.pressure.pressure_mb,
                    pressure_in=weather_data.pressure.pressure_in,
                    timestamp=weather_data.pressure.timestamp
                ) if weather_data.pressure else None,
                location_name=weather_data.location_name
            )
            
            logger.debug(f"Successfully retrieved weather data for {lat}, {lon}")
            return weather_info
            
        except Exception as e:
            logger.warning(f"Failed to get weather data from WeatherAPI: {e}")
            self.weather_api_failures += 1
            
            # Return fallback weather data if enabled
            if config.weather_api.fallback_enabled:
                logger.debug("Using fallback weather data")
                return WeatherInfo(
                    temperature=TemperatureData(
                        celsius=0.0,
                        fahrenheit=32.0,
                        timestamp=datetime.now(timezone.utc),
                        source="fallback"
                    ),
                    wind=None,
                    pressure=None,
                    location_name=f"{lat:.2f},{lon:.2f}"
                )
            
            return None
    
    async def check_weather_api_health(self) -> Dict[str, Any]:
        """Check WeatherAPI health and return status information"""
        if not config.weather_api.enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "message": "WeatherAPI is disabled in configuration"
            }
        
        try:
            api_status = await weather_api_manager.get_api_status()
            return {
                "enabled": True,
                "status": "healthy" if api_status.available else "unhealthy",
                "available": api_status.available,
                "last_check": api_status.last_check.isoformat() if api_status.last_check else None,
                "error_message": api_status.error_message,
                "requests_remaining": api_status.requests_remaining
            }
        except Exception as e:
            logger.error(f"WeatherAPI health check failed: {e}")
            return {
                "enabled": True,
                "status": "unhealthy",
                "available": False,
                "error_message": str(e)
            }
    
    async def get_service_statistics(self) -> Dict[str, Any]:
        """Get service usage statistics"""
        total_requests = self.requests_with_weather + self.requests_fallback_only
        weather_success_rate = 0.0
        
        if total_requests > 0:
            weather_success_rate = self.requests_with_weather / total_requests
        
        # Get WeatherAPI manager statistics
        weather_api_stats = weather_api_manager.get_statistics()
        
        stats = {
            "total_requests": total_requests,
            "requests_with_weather": self.requests_with_weather,
            "requests_fallback_only": self.requests_fallback_only,
            "weather_success_rate": weather_success_rate,
            "weather_api_failures": self.weather_api_failures,
            "weather_api_stats": weather_api_stats,
            "cache_ttl_seconds": self.combined_cache_ttl
        }
        
        # Validate statistics privacy compliance
        if not validate_metrics_privacy(stats, "UnifiedWeatherService.get_service_statistics"):
            logger.warning("Service statistics privacy validation failed")
        
        return stats
    
    async def invalidate_location_cache(self, lat: float, lon: float) -> bool:
        """Invalidate cache for specific location"""
        try:
            invalidated = await self.cache_manager.invalidate_by_coordinates(
                lat,
                lon,
                levels=[CacheLevel.L1, CacheLevel.L2]
            )
            logger.info(f"Invalidated {invalidated} cache entries for location {lat}, {lon}")
            return invalidated > 0
        except Exception as e:
            logger.error(f"Failed to invalidate cache for location {lat}, {lon}: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup service resources"""
        try:
            await self.air_quality_service.cleanup()
            await self.cache_manager.cleanup()
            logger.info("Unified weather service cleaned up")
        except Exception as e:
            logger.error(f"Unified weather service cleanup failed: {e}")


# Global unified weather service instance
unified_weather_service = UnifiedWeatherService()
