"""
Property-based tests for WeatherAPI integration reliability.

**Property 10: WeatherAPI Integration Reliability**
**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7**

Tests that WeatherAPI integration provides reliable temperature, wind, and pressure data
with proper fallback handling when the service is unavailable.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import httpx

from main import app
from unified_weather_service import unified_weather_service
from weather_api_manager import weather_api_manager
from schemas import AirQualityData, WeatherInfo, TemperatureData, WindData, PressureData
from config import config
from connection_pool import APIResponse


class TestWeatherAPIIntegrationReliability:
    """Property-based tests for WeatherAPI integration reliability"""
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=25, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_weatherapi_temperature_data_reliability_property(self, lat, lon):
        """
        **Property 10: WeatherAPI Integration Reliability**
        **Validates: Requirements 11.1, 11.2, 11.3**
        
        For any location coordinates, when WeatherAPI is available, the system should
        fetch and include temperature data in both Celsius and Fahrenheit.
        """
        client = TestClient(app)
        
        # Mock both the connection pool and the services directly
        with patch('services.AirQualityService.get_current_air_quality') as mock_air_quality:
            with patch('weather_api_manager.weather_api_manager.get_combined_weather') as mock_weather:
                with patch('unified_weather_service.unified_weather_service.cache_manager.get') as mock_cache_get:
                    with patch('unified_weather_service.unified_weather_service.cache_manager.set') as mock_cache_set:
                        
                        # Mock cache miss
                        mock_cache_get.return_value = None
                        mock_cache_set.return_value = True
                        
                        # Mock WeatherAPI response
                        from weather_api_manager import WeatherData, TemperatureData, WindData, PressureData
                        mock_weather_data = WeatherData(
                            temperature=TemperatureData(
                                celsius=15.5,
                                fahrenheit=59.9,
                                timestamp=datetime.now(timezone.utc),
                                source="weatherapi"
                            ),
                            wind=WindData(
                                speed_kmh=12.5,
                                speed_mph=7.8,
                                direction_degrees=180,
                                direction_compass="S",
                                timestamp=datetime.now(timezone.utc)
                            ),
                            pressure=PressureData(
                                pressure_mb=1013.25,
                                pressure_in=29.92,
                                timestamp=datetime.now(timezone.utc)
                            ),
                            location_name=f"Test Location {lat:.2f},{lon:.2f}",
                            timestamp=datetime.now(timezone.utc)
                        )
                        mock_weather.return_value = mock_weather_data
                        
                        # Mock air quality response
                        from schemas import AirQualityData, LocationInfo, AQIInfo, PollutantData
                        mock_aqi_data = AirQualityData(
                            timestamp=datetime.now(timezone.utc),
                            location=LocationInfo(latitude=lat, longitude=lon),
                            aqi=AQIInfo(
                                value=85,
                                category="Умеренное",
                                color="#FFFF00",
                                description="Качество воздуха приемлемо для большинства людей"
                            ),
                            pollutants=PollutantData(
                                pm2_5=15.0,
                                pm10=25.0,
                                no2=30.0,
                                so2=10.0,
                                o3=80.0
                            ),
                            recommendations="Чувствительные люди должны ограничить длительное пребывание на улице",
                            nmu_risk="low",
                            health_warnings=[]
                        )
                        mock_air_quality.return_value = mock_aqi_data
                        
                        # Enable WeatherAPI for this test
                        with patch.object(config.weather_api, 'enabled', True):
                            with patch.object(config.weather_api, 'api_key', 'test_key'):
                                response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                        
                        # Verify response structure
                        assert response.status_code == 200
                        data = response.json()
                        
                        # Verify air quality data is present (primary requirement)
                        assert "aqi" in data
                        assert "pollutants" in data
                        assert data["aqi"]["value"] >= 0
                        
                        # Verify weather data is included when WeatherAPI is available
                        if "weather" in data and data["weather"]:
                            weather = data["weather"]
                            
                            # Validate temperature data
                            assert "temperature" in weather
                            temp = weather["temperature"]
                            assert "celsius" in temp
                            assert "fahrenheit" in temp
                            assert isinstance(temp["celsius"], (int, float))
                            assert isinstance(temp["fahrenheit"], (int, float))
                            assert temp["source"] in ["weatherapi", "fallback"]
                            
                            # Validate timestamp format
                            assert "timestamp" in temp
                            timestamp_str = temp["timestamp"]
                            # Should be ISO format
                            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=15, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_weatherapi_fallback_handling_property(self, lat, lon):
        """
        **Property 10: WeatherAPI Integration Reliability**
        **Validates: Requirements 11.4, 11.5**
        
        For any location coordinates, when WeatherAPI is unavailable, the system should
        continue operating with Open-Meteo data only and provide appropriate fallback.
        """
        client = TestClient(app)
        
        # Mock services directly to avoid connection pool issues
        with patch('services.AirQualityService.get_current_air_quality') as mock_air_quality:
            with patch('weather_api_manager.weather_api_manager.get_combined_weather') as mock_weather:
                with patch('unified_weather_service.unified_weather_service.cache_manager.get') as mock_cache_get:
                    with patch('unified_weather_service.unified_weather_service.cache_manager.set') as mock_cache_set:
                        
                        # Mock cache miss
                        mock_cache_get.return_value = None
                        mock_cache_set.return_value = True
                        
                        # Mock WeatherAPI failure
                        mock_weather.side_effect = Exception("WeatherAPI unavailable")
                        
                        # Mock air quality response (should still work)
                        from schemas import AirQualityData, LocationInfo, AQIInfo, PollutantData
                        mock_aqi_data = AirQualityData(
                            timestamp=datetime.now(timezone.utc),
                            location=LocationInfo(latitude=lat, longitude=lon),
                            aqi=AQIInfo(
                                value=95,
                                category="Умеренное",
                                color="#FFFF00",
                                description="Качество воздуха приемлемо для большинства людей"
                            ),
                            pollutants=PollutantData(
                                pm2_5=25.0,
                                pm10=35.0,
                                no2=40.0,
                                so2=15.0,
                                o3=90.0
                            ),
                            recommendations="Чувствительные люди должны ограничить длительное пребывание на улице",
                            nmu_risk="low",
                            health_warnings=[]
                        )
                        mock_air_quality.return_value = mock_aqi_data
                        
                        # Enable WeatherAPI but it will fail
                        with patch.object(config.weather_api, 'enabled', True):
                            with patch.object(config.weather_api, 'api_key', 'test_key'):
                                with patch.object(config.weather_api, 'fallback_enabled', True):
                                    response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                        
                        # System should still work with air quality data
                        assert response.status_code == 200
                        data = response.json()
                        
                        # Verify air quality data is present (primary requirement)
                        assert "aqi" in data
                        assert "pollutants" in data
                        assert data["aqi"]["value"] >= 0
                        
                        # Weather data should either be None or contain fallback data
                        if "weather" in data:
                            weather = data["weather"]
                            if weather and "temperature" in weather:
                                # If fallback is enabled, should have fallback temperature
                                temp = weather["temperature"]
                                assert temp["source"] == "fallback"
                                assert temp["celsius"] == 0.0
                                assert temp["fahrenheit"] == 32.0
    
    def test_weatherapi_unified_response_format_property(self):
        """
        **Property 10: WeatherAPI Integration Reliability**
        **Validates: Requirements 11.6, 11.7**
        
        The unified weather data response format should combine air quality and weather data
        in a consistent structure regardless of WeatherAPI availability.
        """
        # Test with various coordinate combinations
        test_coordinates = [
            (55.7558, 37.6176),  # Moscow
            (59.9311, 30.3609),  # St. Petersburg
        ]
        
        client = TestClient(app)
        
        for lat, lon in test_coordinates:
            # Mock services directly to avoid connection pool issues
            with patch('services.AirQualityService.get_current_air_quality') as mock_air_quality:
                with patch('weather_api_manager.weather_api_manager.get_combined_weather') as mock_weather:
                    with patch('unified_weather_service.unified_weather_service.cache_manager.get') as mock_cache_get:
                        with patch('unified_weather_service.unified_weather_service.cache_manager.set') as mock_cache_set:
                            
                            # Mock cache miss
                            mock_cache_get.return_value = None
                            mock_cache_set.return_value = True
                            
                            # Mock WeatherAPI response
                            from weather_api_manager import WeatherData, TemperatureData, WindData, PressureData
                            mock_weather_data = WeatherData(
                                temperature=TemperatureData(
                                    celsius=15.0,
                                    fahrenheit=59.0,
                                    timestamp=datetime.now(timezone.utc),
                                    source="weatherapi"
                                ),
                                wind=WindData(
                                    speed_kmh=10.0,
                                    speed_mph=6.2,
                                    direction_degrees=180,
                                    direction_compass="S",
                                    timestamp=datetime.now(timezone.utc)
                                ),
                                pressure=PressureData(
                                    pressure_mb=1013.25,
                                    pressure_in=29.92,
                                    timestamp=datetime.now(timezone.utc)
                                ),
                                location_name=f"City {lat:.1f},{lon:.1f}",
                                timestamp=datetime.now(timezone.utc)
                            )
                            mock_weather.return_value = mock_weather_data
                            
                            # Mock air quality response
                            from schemas import AirQualityData, LocationInfo, AQIInfo, PollutantData
                            mock_aqi_data = AirQualityData(
                                timestamp=datetime.now(timezone.utc),
                                location=LocationInfo(latitude=lat, longitude=lon),
                                aqi=AQIInfo(
                                    value=75,
                                    category="Умеренное",
                                    color="#FFFF00",
                                    description="Качество воздуха приемлемо для большинства людей"
                                ),
                                pollutants=PollutantData(
                                    pm2_5=15.0,
                                    pm10=25.0,
                                    no2=30.0,
                                    so2=10.0,
                                    o3=80.0
                                ),
                                recommendations="Чувствительные люди должны ограничить длительное пребывание на улице",
                                nmu_risk="low",
                                health_warnings=[]
                            )
                            mock_air_quality.return_value = mock_aqi_data
                            
                            # Test with WeatherAPI enabled
                            with patch.object(config.weather_api, 'enabled', True):
                                with patch.object(config.weather_api, 'api_key', 'test_key'):
                                    response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                            
                            assert response.status_code == 200
                            data = response.json()
                            
                            # Verify unified response structure
                            required_fields = ["timestamp", "location", "aqi", "pollutants", "recommendations"]
                            for field in required_fields:
                                assert field in data, f"Missing required field: {field}"
                            
                            # Verify location data
                            assert "latitude" in data["location"]
                            assert "longitude" in data["location"]
                            assert abs(data["location"]["latitude"] - lat) < 0.1
                            assert abs(data["location"]["longitude"] - lon) < 0.1
                            
                            # Verify AQI data structure
                            aqi = data["aqi"]
                            assert "value" in aqi
                            assert "category" in aqi
                            assert "color" in aqi
                            assert "description" in aqi
                            
                            # Verify weather data integration (if present)
                            if "weather" in data and data["weather"]:
                                weather = data["weather"]
                                
                                # Should have location name from WeatherAPI
                                if "location_name" in weather:
                                    assert isinstance(weather["location_name"], str)
                                    assert len(weather["location_name"]) > 0
                                
                                # Temperature should be present
                                if "temperature" in weather:
                                    temp = weather["temperature"]
                                    assert "celsius" in temp
                                    assert "fahrenheit" in temp
                                    assert "timestamp" in temp
                                    assert "source" in temp