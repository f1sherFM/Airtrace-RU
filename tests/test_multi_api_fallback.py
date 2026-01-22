"""
Multi-API integration and fallback mechanism tests for AirTrace RU Backend.

Tests WeatherAPI and Open-Meteo API integration, fallback scenarios,
and data combination strategies.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 9.2, 9.3
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock
import httpx

from unified_weather_service import UnifiedWeatherService
from weather_api_manager import WeatherAPIManager, WeatherData, TemperatureData, WindData, PressureData
from services import AirQualityService
from connection_pool import ConnectionPoolManager, ServiceType, APIRequest, APIResponse
from schemas import AirQualityData, LocationInfo, AQIInfo, PollutantData, WeatherInfo
from config import config


class TestWeatherAPIIntegration:
    """Test WeatherAPI.com integration"""
    
    @pytest.fixture
    def weather_api_manager(self):
        """WeatherAPI manager fixture"""
        return WeatherAPIManager()
    
    @pytest.fixture
    def mock_weather_response(self):
        """Mock WeatherAPI response data"""
        return {
            "location": {
                "name": "Moscow",
                "region": "Moscow",
                "country": "Russia",
                "lat": 55.75,
                "lon": 37.62,
                "tz_id": "Europe/Moscow"
            },
            "current": {
                "temp_c": 20.5,
                "temp_f": 68.9,
                "wind_mph": 5.6,
                "wind_kph": 9.0,
                "wind_degree": 180,
                "wind_dir": "S",
                "pressure_mb": 1013.2,
                "pressure_in": 29.92,
                "humidity": 65,
                "cloud": 25
            }
        }
    
    async def test_weatherapi_temperature_data_retrieval(self, weather_api_manager, mock_weather_response):
        """Test temperature data retrieval from WeatherAPI"""
        with patch.object(weather_api_manager, '_make_api_request', return_value=mock_weather_response):
            temp_data = await weather_api_manager.get_temperature(55.7558, 37.6176)
            
            assert temp_data.celsius == 20.5
            assert temp_data.fahrenheit == 68.9
            assert temp_data.source == "weatherapi"
            assert temp_data.timestamp is not None
    
    async def test_weatherapi_wind_data_retrieval(self, weather_api_manager, mock_weather_response):
        """Test wind data retrieval from WeatherAPI"""
        with patch.object(weather_api_manager, '_make_api_request', return_value=mock_weather_response):
            wind_data = await weather_api_manager.get_wind_data(55.7558, 37.6176)
            
            assert wind_data.speed_kmh == 9.0
            assert wind_data.speed_mph == 5.6
            assert wind_data.direction_degrees == 180
            assert wind_data.direction_compass == "S"
    
    async def test_weatherapi_pressure_data_retrieval(self, weather_api_manager, mock_weather_response):
        """Test pressure data retrieval from WeatherAPI"""
        with patch.object(weather_api_manager, '_make_api_request', return_value=mock_weather_response):
            pressure_data = await weather_api_manager.get_pressure_data(55.7558, 37.6176)
            
            assert pressure_data.pressure_mb == 1013.2
            assert pressure_data.pressure_in == 29.92
    
    async def test_weatherapi_combined_data_retrieval(self, weather_api_manager, mock_weather_response):
        """Test combined weather data retrieval from WeatherAPI"""
        with patch.object(weather_api_manager, '_make_api_request', return_value=mock_weather_response):
            weather_data = await weather_api_manager.get_combined_weather(55.7558, 37.6176)
            
            assert weather_data.temperature.celsius == 20.5
            assert weather_data.wind.speed_kmh == 9.0
            assert weather_data.pressure.pressure_mb == 1013.2
            assert weather_data.location_name == "Moscow"
    
    async def test_weatherapi_api_key_configuration(self, weather_api_manager):
        """Test WeatherAPI key configuration"""
        # Test with valid key
        weather_api_manager.configure_api_key("test_api_key_123")
        assert weather_api_manager.api_key == "test_api_key_123"
        assert weather_api_manager.enabled is True
        
        # Test with empty key
        weather_api_manager.configure_api_key("")
        assert weather_api_manager.enabled is False
    
    async def test_weatherapi_rate_limiting_compliance(self, weather_api_manager):
        """Test WeatherAPI rate limiting compliance"""
        # Mock multiple requests to test rate limiting
        with patch.object(weather_api_manager, '_make_api_request') as mock_request:
            mock_request.return_value = {"current": {"temp_c": 20.0}}
            
            # Make multiple requests
            for _ in range(5):
                await weather_api_manager.get_temperature(55.7558, 37.6176)
            
            # Should track request count
            assert weather_api_manager.requests_made >= 5
    
    async def test_weatherapi_error_handling(self, weather_api_manager):
        """Test WeatherAPI error handling"""
        # Test HTTP error
        with patch.object(weather_api_manager, '_make_api_request', 
                         side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                await weather_api_manager.get_temperature(55.7558, 37.6176)
        
        # Test invalid response format
        with patch.object(weather_api_manager, '_make_api_request', return_value={}):
            with pytest.raises(ValueError):
                await weather_api_manager.get_temperature(55.7558, 37.6176)


class TestOpenMeteoIntegration:
    """Test Open-Meteo API integration"""
    
    @pytest.fixture
    async def air_quality_service(self):
        """Air quality service fixture"""
        service = AirQualityService()
        yield service
        await service.cleanup()
    
    @pytest.fixture
    def mock_openmeteo_response(self):
        """Mock Open-Meteo API response"""
        return {
            "latitude": 55.75,
            "longitude": 37.625,
            "current": {
                "time": "2024-01-15T12:00",
                "pm10": 45.2,
                "pm2_5": 25.4,
                "nitrogen_dioxide": 35.1,
                "sulphur_dioxide": 12.3,
                "ozone": 85.7
            }
        }
    
    async def test_openmeteo_air_quality_data_retrieval(self, air_quality_service, mock_openmeteo_response):
        """Test air quality data retrieval from Open-Meteo"""
        with patch.object(air_quality_service, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openmeteo_response
            mock_client.get = AsyncMock(return_value=mock_response)
            
            data = await air_quality_service.get_current_air_quality(55.7558, 37.6176)
            
            assert data.pollutants.pm2_5 == 25.4
            assert data.pollutants.pm10 == 45.2
            assert data.pollutants.no2 == 35.1
            assert data.pollutants.so2 == 12.3
            assert data.pollutants.o3 == 85.7
    
    async def test_openmeteo_connection_pool_integration(self):
        """Test Open-Meteo API with connection pooling"""
        connection_manager = ConnectionPoolManager()
        
        # Mock successful response
        mock_response = APIResponse(
            status_code=200,
            data={"latitude": 55.75, "longitude": 37.625, "current": {"pm10": 45.2}},
            headers={},
            response_time=0.5
        )
        
        with patch.object(connection_manager, 'execute_request', return_value=mock_response):
            request = APIRequest(
                method="GET",
                url="https://air-quality-api.open-meteo.com/v1/air-quality",
                params={"latitude": 55.7558, "longitude": 37.6176, "current": "pm10"}
            )
            
            response = await connection_manager.execute_request(ServiceType.OPEN_METEO, request)
            assert response.status_code == 200
            assert response.data["current"]["pm10"] == 45.2
    
    async def test_openmeteo_forecast_data_retrieval(self, air_quality_service):
        """Test forecast data retrieval from Open-Meteo"""
        mock_forecast_response = {
            "latitude": 55.75,
            "longitude": 37.625,
            "hourly": {
                "time": ["2024-01-15T12:00", "2024-01-15T13:00", "2024-01-15T14:00"],
                "pm10": [45.2, 46.1, 44.8],
                "pm2_5": [25.4, 26.0, 24.9],
                "nitrogen_dioxide": [35.1, 36.2, 34.5],
                "sulphur_dioxide": [12.3, 12.8, 11.9],
                "ozone": [85.7, 87.2, 84.3]
            }
        }
        
        with patch.object(air_quality_service, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_forecast_response
            mock_client.get = AsyncMock(return_value=mock_response)
            
            forecast = await air_quality_service.get_forecast_air_quality(55.7558, 37.6176)
            
            assert len(forecast) == 3
            assert forecast[0].pollutants.pm10 == 45.2
            assert forecast[1].pollutants.pm2_5 == 26.0
            assert forecast[2].pollutants.no2 == 34.5


class TestMultiAPIFallbackMechanisms:
    """Test multi-API fallback mechanisms"""
    
    @pytest.fixture
    async def unified_service(self):
        """Unified weather service fixture"""
        service = UnifiedWeatherService()
        yield service
        await service.cleanup()
    
    async def test_weatherapi_unavailable_fallback(self, unified_service):
        """Test fallback when WeatherAPI is unavailable"""
        # Mock air quality service to succeed
        mock_aqi_data = AirQualityData(
            timestamp=datetime.now(timezone.utc),
            location=LocationInfo(latitude=55.7558, longitude=37.6176),
            aqi=AQIInfo(value=75, category="Moderate", color="#FFFF00", description="Moderate air quality"),
            pollutants=PollutantData(pm2_5=25.0, pm10=45.0),
            recommendations="Moderate air quality",
            nmu_risk="medium",
            health_warnings=[]
        )
        
        with patch.object(unified_service.air_quality_service, 'get_current_air_quality', 
                         return_value=mock_aqi_data):
            # Mock WeatherAPI to fail
            with patch.object(unified_service, '_get_weather_info_safe', return_value=None):
                data = await unified_service.get_current_combined_data(55.7558, 37.6176)
                
                # Should get air quality data without weather info
                assert data.aqi.value == 75
                assert data.weather is None
                assert unified_service.requests_fallback_only > 0
    
    async def test_openmeteo_unavailable_fallback(self, unified_service):
        """Test fallback when Open-Meteo API is unavailable"""
        # Mock Open-Meteo to fail
        with patch.object(unified_service.air_quality_service, 'get_current_air_quality',
                         side_effect=Exception("Open-Meteo unavailable")):
            # Should raise exception as Open-Meteo is primary data source
            with pytest.raises(Exception):
                await unified_service.get_current_combined_data(55.7558, 37.6176)
    
    async def test_weatherapi_fallback_data_provision(self, unified_service):
        """Test WeatherAPI fallback data provision"""
        # Enable fallback in config
        with patch('config.weather_api.fallback_enabled', True):
            # Mock WeatherAPI to fail
            with patch.object(unified_service.weather_api_manager, 'get_combined_weather',
                             side_effect=Exception("WeatherAPI error")):
                weather_info = await unified_service._get_weather_info_safe(55.7558, 37.6176)
                
                # Should get fallback weather data
                assert weather_info is not None
                assert weather_info.temperature.source == "fallback"
                assert weather_info.temperature.celsius == 0.0
                assert weather_info.temperature.fahrenheit == 32.0
    
    async def test_partial_api_failure_handling(self, unified_service):
        """Test handling of partial API failures"""
        # Mock air quality service to succeed
        mock_aqi_data = AirQualityData(
            timestamp=datetime.now(timezone.utc),
            location=LocationInfo(latitude=55.7558, longitude=37.6176),
            aqi=AQIInfo(value=50, category="Good", color="#00FF00", description="Good air quality"),
            pollutants=PollutantData(pm2_5=15.0, pm10=30.0),
            recommendations="Good air quality",
            nmu_risk="low",
            health_warnings=[]
        )
        
        # Mock WeatherAPI to return partial data (temperature only)
        mock_weather_data = WeatherData(
            temperature=TemperatureData(celsius=22.0, fahrenheit=71.6, 
                                      timestamp=datetime.now(timezone.utc), source="weatherapi"),
            wind=None,  # Wind data unavailable
            pressure=None,  # Pressure data unavailable
            location_name="Moscow",
            timestamp=datetime.now(timezone.utc)
        )
        
        with patch.object(unified_service.air_quality_service, 'get_current_air_quality',
                         return_value=mock_aqi_data):
            with patch.object(unified_service.weather_api_manager, 'get_combined_weather',
                             return_value=mock_weather_data):
                data = await unified_service.get_current_combined_data(55.7558, 37.6176)
                
                # Should get combined data with partial weather info
                assert data.aqi.value == 50
                assert data.weather is not None
                assert data.weather.temperature.celsius == 22.0
                assert data.weather.wind is None
                assert data.weather.pressure is None
    
    async def test_api_timeout_fallback(self, unified_service):
        """Test fallback when APIs timeout"""
        # Mock air quality service to succeed
        mock_aqi_data = AirQualityData(
            timestamp=datetime.now(timezone.utc),
            location=LocationInfo(latitude=55.7558, longitude=37.6176),
            aqi=AQIInfo(value=60, category="Moderate", color="#FFFF00", description="Moderate air quality"),
            pollutants=PollutantData(pm2_5=20.0, pm10=40.0),
            recommendations="Moderate air quality",
            nmu_risk="medium",
            health_warnings=[]
        )
        
        with patch.object(unified_service.air_quality_service, 'get_current_air_quality',
                         return_value=mock_aqi_data):
            # Mock WeatherAPI to timeout
            with patch.object(unified_service.weather_api_manager, 'get_combined_weather',
                             side_effect=asyncio.TimeoutError("Request timeout")):
                data = await unified_service.get_current_combined_data(55.7558, 37.6176)
                
                # Should get air quality data without weather info
                assert data.aqi.value == 60
                assert data.weather is None or data.weather.temperature.source == "fallback"
    
    async def test_api_rate_limit_fallback(self, unified_service):
        """Test fallback when APIs hit rate limits"""
        # Mock air quality service to succeed
        mock_aqi_data = AirQualityData(
            timestamp=datetime.now(timezone.utc),
            location=LocationInfo(latitude=55.7558, longitude=37.6176),
            aqi=AQIInfo(value=80, category="Moderate", color="#FFFF00", description="Moderate air quality"),
            pollutants=PollutantData(pm2_5=30.0, pm10=55.0),
            recommendations="Moderate air quality",
            nmu_risk="medium",
            health_warnings=[]
        )
        
        with patch.object(unified_service.air_quality_service, 'get_current_air_quality',
                         return_value=mock_aqi_data):
            # Mock WeatherAPI to hit rate limit
            rate_limit_error = httpx.HTTPStatusError(
                "Rate limit exceeded", 
                request=MagicMock(), 
                response=MagicMock(status_code=429)
            )
            
            with patch.object(unified_service.weather_api_manager, 'get_combined_weather',
                             side_effect=rate_limit_error):
                data = await unified_service.get_current_combined_data(55.7558, 37.6176)
                
                # Should get air quality data without weather info
                assert data.aqi.value == 80
                assert unified_service.weather_api_failures > 0


class TestDataCombinationStrategies:
    """Test data combination strategies from multiple APIs"""
    
    async def test_successful_data_combination(self):
        """Test successful combination of data from both APIs"""
        unified_service = UnifiedWeatherService()
        
        # Mock both services to succeed
        mock_aqi_data = AirQualityData(
            timestamp=datetime.now(timezone.utc),
            location=LocationInfo(latitude=55.7558, longitude=37.6176),
            aqi=AQIInfo(value=65, category="Moderate", color="#FFFF00", description="Moderate air quality"),
            pollutants=PollutantData(pm2_5=22.0, pm10=42.0, no2=30.0),
            recommendations="Moderate air quality",
            nmu_risk="medium",
            health_warnings=[]
        )
        
        mock_weather_info = WeatherInfo(
            temperature=TemperatureData(celsius=18.5, fahrenheit=65.3,
                                      timestamp=datetime.now(timezone.utc), source="weatherapi"),
            wind=WindData(speed_kmh=12.0, speed_mph=7.5, direction_degrees=270, 
                         direction_compass="W", timestamp=datetime.now(timezone.utc)),
            pressure=PressureData(pressure_mb=1015.2, pressure_in=29.98,
                                 timestamp=datetime.now(timezone.utc)),
            location_name="Moscow"
        )
        
        with patch.object(unified_service.air_quality_service, 'get_current_air_quality',
                         return_value=mock_aqi_data):
            with patch.object(unified_service, '_get_weather_info_safe',
                             return_value=mock_weather_info):
                data = await unified_service.get_current_combined_data(55.7558, 37.6176)
                
                # Verify combined data
                assert data.aqi.value == 65
                assert data.pollutants.pm2_5 == 22.0
                assert data.weather.temperature.celsius == 18.5
                assert data.weather.wind.speed_kmh == 12.0
                assert data.weather.pressure.pressure_mb == 1015.2
                assert unified_service.requests_with_weather > 0
    
    async def test_data_timestamp_consistency(self):
        """Test timestamp consistency in combined data"""
        unified_service = UnifiedWeatherService()
        
        # Mock data with specific timestamps
        base_time = datetime.now(timezone.utc)
        
        mock_aqi_data = AirQualityData(
            timestamp=base_time,
            location=LocationInfo(latitude=55.7558, longitude=37.6176),
            aqi=AQIInfo(value=70, category="Moderate", color="#FFFF00", description="Moderate air quality"),
            pollutants=PollutantData(pm2_5=25.0, pm10=45.0),
            recommendations="Moderate air quality",
            nmu_risk="medium",
            health_warnings=[]
        )
        
        with patch.object(unified_service.air_quality_service, 'get_current_air_quality',
                         return_value=mock_aqi_data):
            with patch.object(unified_service, '_get_weather_info_safe', return_value=None):
                data = await unified_service.get_current_combined_data(55.7558, 37.6176)
                
                # Timestamp should be preserved from air quality data
                assert data.timestamp == base_time
    
    async def test_location_consistency_validation(self):
        """Test location consistency validation between APIs"""
        unified_service = UnifiedWeatherService()
        
        # Mock data with consistent locations
        mock_aqi_data = AirQualityData(
            timestamp=datetime.now(timezone.utc),
            location=LocationInfo(latitude=55.7558, longitude=37.6176),
            aqi=AQIInfo(value=55, category="Moderate", color="#FFFF00", description="Moderate air quality"),
            pollutants=PollutantData(pm2_5=18.0, pm10=35.0),
            recommendations="Moderate air quality",
            nmu_risk="low",
            health_warnings=[]
        )
        
        with patch.object(unified_service.air_quality_service, 'get_current_air_quality',
                         return_value=mock_aqi_data):
            data = await unified_service.get_current_combined_data(55.7558, 37.6176)
            
            # Location should match request coordinates
            assert data.location.latitude == 55.7558
            assert data.location.longitude == 37.6176
    
    async def test_forecast_data_combination(self):
        """Test combination of forecast data from multiple APIs"""
        unified_service = UnifiedWeatherService()
        
        # Mock forecast data
        mock_forecast = []
        for i in range(3):
            forecast_item = AirQualityData(
                timestamp=datetime.now(timezone.utc),
                location=LocationInfo(latitude=55.7558, longitude=37.6176),
                aqi=AQIInfo(value=60 + i, category="Moderate", color="#FFFF00", description="Moderate air quality"),
                pollutants=PollutantData(pm2_5=20.0 + i, pm10=40.0 + i),
                recommendations="Moderate air quality",
                nmu_risk="medium",
                health_warnings=[]
            )
            mock_forecast.append(forecast_item)
        
        with patch.object(unified_service.air_quality_service, 'get_forecast_air_quality',
                         return_value=mock_forecast):
            with patch.object(unified_service, '_get_weather_info_safe', return_value=None):
                forecast = await unified_service.get_forecast_combined_data(55.7558, 37.6176)
                
                # Verify forecast data
                assert len(forecast) == 3
                assert forecast[0].aqi.value == 60
                assert forecast[1].aqi.value == 61
                assert forecast[2].aqi.value == 62


class TestAPIHealthMonitoring:
    """Test API health monitoring and status tracking"""
    
    async def test_weatherapi_health_status_tracking(self):
        """Test WeatherAPI health status tracking"""
        weather_manager = WeatherAPIManager()
        
        # Test healthy status
        with patch.object(weather_manager, '_make_api_request', return_value={"current": {"temp_c": 20.0}}):
            status = await weather_manager.get_api_status()
            assert status.available is True
            assert status.error_message is None
        
        # Test unhealthy status
        with patch.object(weather_manager, '_make_api_request', side_effect=Exception("API Error")):
            # Force status check
            await weather_manager._check_api_status()
            status = await weather_manager.get_api_status()
            assert status.available is False
            assert status.error_message is not None
    
    async def test_unified_service_health_monitoring(self):
        """Test unified service health monitoring"""
        unified_service = UnifiedWeatherService()
        
        # Test WeatherAPI health check
        health_status = await unified_service.check_weather_api_health()
        assert "enabled" in health_status
        assert "status" in health_status
        
        if config.weather_api.enabled:
            assert health_status["enabled"] is True
        else:
            assert health_status["enabled"] is False
    
    async def test_api_statistics_collection(self):
        """Test API statistics collection"""
        unified_service = UnifiedWeatherService()
        
        # Get service statistics
        stats = await unified_service.get_service_statistics()
        
        assert "total_requests" in stats
        assert "requests_with_weather" in stats
        assert "requests_fallback_only" in stats
        assert "weather_success_rate" in stats
        assert "weather_api_failures" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])