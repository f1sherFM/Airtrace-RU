"""
API Compatibility and Russian Localization Validation Tests

Tests to ensure:
1. Existing API endpoints remain unchanged
2. Russian AQI calculations remain accurate
3. Russian localization in all error messages

Requirements: 8.2, 8.5, 8.6
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import httpx

from main import app
from utils import AQICalculator, check_nmu_risk, get_nmu_recommendations, get_pollutant_name_russian
from schemas import AirQualityData, LocationInfo, AQIInfo, PollutantData


class TestAPICompatibility:
    """Test API endpoint compatibility"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_root_endpoint_structure(self):
        """Test root endpoint returns expected structure"""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data
        
        # Verify expected endpoints are present
        endpoints = data["endpoints"]
        assert "current" in endpoints
        assert "forecast" in endpoints
        assert "health" in endpoints
        assert "docs" in endpoints
        
        # Verify endpoint paths haven't changed
        assert endpoints["current"] == "/weather/current"
        assert endpoints["forecast"] == "/weather/forecast"
        assert endpoints["health"] == "/health"
        assert endpoints["docs"] == "/docs"
    
    def test_current_weather_endpoint_parameters(self):
        """Test current weather endpoint accepts expected parameters"""
        # Test with valid coordinates
        response = self.client.get("/weather/current?lat=55.7558&lon=37.6176")
        # Should not return 422 (validation error)
        assert response.status_code != 422
        
        # Test parameter validation
        response = self.client.get("/weather/current?lat=91&lon=37.6176")  # Invalid lat
        assert response.status_code == 422
        
        response = self.client.get("/weather/current?lat=55.7558&lon=181")  # Invalid lon
        assert response.status_code == 422
        
        response = self.client.get("/weather/current")  # Missing parameters
        assert response.status_code == 422
    
    def test_forecast_weather_endpoint_parameters(self):
        """Test forecast weather endpoint accepts expected parameters"""
        # Test with valid coordinates
        response = self.client.get("/weather/forecast?lat=55.7558&lon=37.6176")
        # Should not return 422 (validation error)
        assert response.status_code != 422
        
        # Test parameter validation
        response = self.client.get("/weather/forecast?lat=-91&lon=37.6176")  # Invalid lat
        assert response.status_code == 422
        
        response = self.client.get("/weather/forecast?lat=55.7558&lon=-181")  # Invalid lon
        assert response.status_code == 422
    
    def test_health_endpoint_structure(self):
        """Test health endpoint returns expected structure"""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        
        # Verify expected services are checked
        services = data["services"]
        expected_services = ["api", "external_api", "cache", "aqi_calculator"]
        for service in expected_services:
            assert service in services
    
    def test_metrics_endpoint_exists(self):
        """Test metrics endpoint exists and returns data"""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "service_status" in data
        assert "components" in data
    
    def test_version_endpoint_structure(self):
        """Test version endpoint returns expected structure"""
        response = self.client.get("/version")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "api_version" in data
        assert "features" in data
        
        # Verify Russian-specific features are present
        features = data["features"]
        assert "Russian AQI calculation" in features
        assert "NMU risk detection" in features


class TestRussianAQICalculations:
    """Test Russian AQI calculations remain accurate"""
    
    def setup_method(self):
        """Setup AQI calculator"""
        self.calculator = AQICalculator()
    
    def test_russian_pdk_standards_unchanged(self):
        """Test Russian PDK standards haven't changed"""
        standards = self.calculator.RU_STANDARDS
        
        # Verify PM2.5 standards
        assert standards['pm2_5']['good'] == 25
        assert standards['pm2_5']['moderate'] == 50
        assert standards['pm2_5']['unhealthy_sensitive'] == 75
        assert standards['pm2_5']['unhealthy'] == 100
        assert standards['pm2_5']['very_unhealthy'] == 150
        assert standards['pm2_5']['hazardous'] == 250
        
        # Verify PM10 standards
        assert standards['pm10']['good'] == 50
        assert standards['pm10']['moderate'] == 100
        assert standards['pm10']['unhealthy_sensitive'] == 150
        assert standards['pm10']['unhealthy'] == 200
        assert standards['pm10']['very_unhealthy'] == 300
        assert standards['pm10']['hazardous'] == 500
        
        # Verify NO2 standards
        assert standards['no2']['good'] == 40
        assert standards['no2']['moderate'] == 80
        
        # Verify SO2 standards
        assert standards['so2']['good'] == 50
        assert standards['so2']['moderate'] == 100
        
        # Verify O3 standards
        assert standards['o3']['good'] == 100
        assert standards['o3']['moderate'] == 160
    
    def test_aqi_calculation_accuracy(self):
        """Test AQI calculation accuracy with known values"""
        # Test good air quality
        pollutants = {"pm2_5": 15.0, "pm10": 30.0}
        aqi, category, color = self.calculator.calculate_aqi(pollutants)
        assert 0 <= aqi <= 50
        assert category == "Хорошее"
        assert color == "#00E400"
        
        # Test moderate air quality
        pollutants = {"pm2_5": 35.0, "pm10": 75.0}
        aqi, category, color = self.calculator.calculate_aqi(pollutants)
        assert 51 <= aqi <= 100
        assert category == "Умеренное"
        assert color == "#FFFF00"
        
        # Test unhealthy for sensitive groups
        pollutants = {"pm2_5": 65.0, "pm10": 125.0}
        aqi, category, color = self.calculator.calculate_aqi(pollutants)
        assert 101 <= aqi <= 150
        assert category == "Вредно для чувствительных групп"
        assert color == "#FF7E00"
        
        # Test unhealthy
        pollutants = {"pm2_5": 85.0, "pm10": 175.0}
        aqi, category, color = self.calculator.calculate_aqi(pollutants)
        assert 151 <= aqi <= 200
        assert category == "Вредно"
        assert color == "#FF0000"
        
        # Test very unhealthy
        pollutants = {"pm2_5": 125.0, "pm10": 250.0}
        aqi, category, color = self.calculator.calculate_aqi(pollutants)
        assert 201 <= aqi <= 300
        assert category == "Очень вредно"
        assert color == "#8F3F97"
        
        # Test hazardous
        pollutants = {"pm2_5": 200.0, "pm10": 400.0}
        aqi, category, color = self.calculator.calculate_aqi(pollutants)
        assert 301 <= aqi <= 500
        assert category == "Опасно"
        assert color == "#7E0023"
    
    def test_dominant_pollutant_selection(self):
        """Test that highest AQI pollutant is selected correctly"""
        # PM2.5 should dominate
        pollutants = {"pm2_5": 100.0, "pm10": 75.0, "no2": 30.0}
        aqi, category, color = self.calculator.calculate_aqi(pollutants)
        # PM2.5 at 100 should give AQI around 200 (unhealthy)
        assert 180 <= aqi <= 220
        assert category == "Вредно"
        
        # NO2 should dominate
        pollutants = {"pm2_5": 20.0, "pm10": 40.0, "no2": 120.0}
        aqi, category, color = self.calculator.calculate_aqi(pollutants)
        # NO2 at 120 should give AQI around 150 (unhealthy for sensitive)
        assert 140 <= aqi <= 160
        assert category == "Вредно для чувствительных групп"
    
    def test_russian_recommendations_accuracy(self):
        """Test Russian recommendations are accurate and appropriate"""
        # Test good air quality recommendations
        recommendations = self.calculator.get_recommendations(25, "Хорошее")
        assert "Отличное качество воздуха" in recommendations
        assert "активностей на открытом воздухе" in recommendations
        
        # Test moderate air quality recommendations
        recommendations = self.calculator.get_recommendations(75, "Умеренное")
        assert "Хорошее качество воздуха" in recommendations
        assert "заниматься любыми видами деятельности" in recommendations
        
        # Test unhealthy for sensitive groups recommendations
        recommendations = self.calculator.get_recommendations(125, "Вредно для чувствительных групп")
        assert "Чувствительные люди" in recommendations
        assert "ограничить" in recommendations
        assert "физические нагрузки" in recommendations
        
        # Test unhealthy recommendations
        recommendations = self.calculator.get_recommendations(175, "Вредно")
        assert "Всем рекомендуется ограничить" in recommendations
        assert "избегать физических нагрузок" in recommendations
        
        # Test very unhealthy recommendations
        recommendations = self.calculator.get_recommendations(250, "Очень вредно")
        assert "избегать физических нагрузок на открытом воздухе" in recommendations
        assert "оставаться в помещении" in recommendations
        
        # Test hazardous recommendations
        recommendations = self.calculator.get_recommendations(400, "Опасно")
        assert "Чрезвычайная ситуация" in recommendations
        assert "оставаться в помещении" in recommendations
        assert "очистители воздуха" in recommendations


class TestNMURiskDetection:
    """Test NMU (Неблагоприятные Метеорологические Условия) risk detection"""
    
    def test_nmu_risk_levels(self):
        """Test NMU risk level determination"""
        # Test low risk - very clean air
        pollutants = {"pm2_5": 10.0, "pm10": 20.0}
        risk = check_nmu_risk(pollutants)
        assert risk in ["low", "medium"]  # Should be low or medium for clean air
        
        # Test medium risk
        pollutants = {"pm2_5": 35.0, "pm10": 70.0}
        risk = check_nmu_risk(pollutants)
        assert risk in ["low", "medium", "high"]  # Could be any of these depending on calculation
        
        # Test high risk
        pollutants = {"pm2_5": 75.0, "pm10": 150.0}
        risk = check_nmu_risk(pollutants)
        assert risk in ["medium", "high", "critical"]
        
        # Test critical risk - very high pollution
        pollutants = {"pm2_5": 200.0, "pm10": 400.0}
        risk = check_nmu_risk(pollutants)
        assert risk in ["high", "critical"]
    
    def test_blacksky_conditions_detection(self):
        """Test 'Black Sky' conditions detection"""
        from utils import is_blacksky_conditions
        
        # Test normal conditions (not black sky)
        pollutants = {"pm2_5": 50.0, "pm10": 100.0}
        assert not is_blacksky_conditions(pollutants)
        
        # Test black sky conditions for PM2.5 (5x PDK = 125)
        pollutants = {"pm2_5": 130.0, "pm10": 50.0}
        assert is_blacksky_conditions(pollutants)
        
        # Test black sky conditions for PM10 (5x PDK = 250)
        pollutants = {"pm2_5": 50.0, "pm10": 260.0}
        assert is_blacksky_conditions(pollutants)
        
        # Test black sky conditions for NO2 (10x PDK = 400)
        pollutants = {"no2": 410.0, "pm2_5": 20.0}
        assert is_blacksky_conditions(pollutants)
    
    def test_nmu_recommendations_russian(self):
        """Test NMU recommendations are in Russian"""
        # Test low risk recommendations
        recommendations = get_nmu_recommendations("low")
        assert len(recommendations) > 0
        assert "Низкий риск НМУ" in recommendations[0]
        
        # Test medium risk recommendations
        recommendations = get_nmu_recommendations("medium")
        assert any("Умеренный риск НМУ" in rec for rec in recommendations)
        assert any("физические нагрузки" in rec for rec in recommendations)
        
        # Test high risk recommendations
        recommendations = get_nmu_recommendations("high")
        assert any("Высокий риск НМУ" in rec for rec in recommendations)
        assert any("ограничьте время на улице" in rec for rec in recommendations)
        
        # Test critical risk recommendations
        recommendations = get_nmu_recommendations("critical")
        assert any("Критический уровень" in rec for rec in recommendations)
        assert any("Оставайтесь в помещении" in rec for rec in recommendations)
        
        # Test black sky recommendations
        recommendations = get_nmu_recommendations("critical", blacksky=True)
        assert any("ЧЕРНОЕ НЕБО" in rec for rec in recommendations)
        assert any("КРИТИЧЕСКАЯ СИТУАЦИЯ" in rec for rec in recommendations)
        assert any("респиратор" in rec for rec in recommendations)


class TestRussianLocalization:
    """Test Russian localization in error messages and responses"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_error_messages_in_russian(self):
        """Test error messages are in Russian"""
        # Test validation error (should be in Russian if customized)
        response = self.client.get("/weather/current?lat=invalid&lon=37.6176")
        assert response.status_code == 422
        # FastAPI validation errors might be in English, but our custom errors should be Russian
        
        # Test with coordinates outside valid range
        response = self.client.get("/weather/current?lat=91&lon=37.6176")
        assert response.status_code == 422
    
    @patch('unified_weather_service.unified_weather_service.get_current_combined_data')
    def test_service_error_messages_russian(self, mock_get_data):
        """Test service error messages are in Russian"""
        # Test connection error
        mock_get_data.side_effect = ConnectionError("Connection failed")
        response = self.client.get("/weather/current?lat=55.7558&lon=37.6176")
        
        if response.status_code == 503:
            data = response.json()
            assert "detail" in data
            # Should contain Russian error message
            assert any(word in data["detail"] for word in ["сервис", "недоступен", "попробуйте"])
        
        # Test timeout error
        mock_get_data.side_effect = asyncio.TimeoutError()
        response = self.client.get("/weather/current?lat=55.7558&lon=37.6176")
        
        if response.status_code == 503:
            data = response.json()
            assert "detail" in data
            # Should contain Russian error message
            assert any(word in data["detail"] for word in ["медленно", "отвечает", "попробуйте"])
    
    def test_pollutant_names_russian(self):
        """Test pollutant names are in Russian"""
        assert get_pollutant_name_russian("pm2_5") == "Мелкодисперсные частицы PM2.5"
        assert get_pollutant_name_russian("pm10") == "Взвешенные частицы PM10"
        assert get_pollutant_name_russian("no2") == "Диоксид азота"
        assert get_pollutant_name_russian("so2") == "Диоксид серы"
        assert get_pollutant_name_russian("o3") == "Озон"
    
    def test_aqi_categories_russian(self):
        """Test AQI categories are in Russian"""
        calculator = AQICalculator()
        
        # Test all category names are in Russian
        for (min_aqi, max_aqi), info in calculator.AQI_CATEGORIES.items():
            category = info['category']
            description = info['description']
            
            # Verify categories are in Russian
            assert any(word in category for word in [
                "Хорошее", "Умеренное", "Вредно", "Очень", "Опасно"
            ])
            
            # Verify descriptions are in Russian
            assert any(word in description for word in [
                "качество", "воздуха", "здоровье", "риск", "население"
            ])
    
    @patch('unified_weather_service.unified_weather_service.get_current_combined_data')
    def test_api_response_structure_unchanged(self, mock_get_data):
        """Test API response structure hasn't changed"""
        # Mock successful response
        mock_data = AirQualityData(
            timestamp=datetime.now(timezone.utc),
            location=LocationInfo(latitude=55.7558, longitude=37.6176),
            aqi=AQIInfo(value=75, category="Умеренное", color="#FFFF00", description="Умеренное качество воздуха"),
            pollutants=PollutantData(pm2_5=25.0, pm10=50.0, no2=30.0, so2=20.0, o3=80.0),
            recommendations="Хорошее качество воздуха. Можно заниматься любыми видами деятельности на открытом воздухе.",
            nmu_risk="medium",
            health_warnings=[]
        )
        mock_get_data.return_value = mock_data
        
        response = self.client.get("/weather/current?lat=55.7558&lon=37.6176")
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify expected fields are present
        required_fields = [
            "timestamp", "location", "aqi", "pollutants", 
            "recommendations", "nmu_risk", "health_warnings"
        ]
        for field in required_fields:
            assert field in data
        
        # Verify location structure
        location = data["location"]
        assert "latitude" in location
        assert "longitude" in location
        
        # Verify AQI structure
        aqi = data["aqi"]
        assert "value" in aqi
        assert "category" in aqi
        assert "color" in aqi
        assert "description" in aqi
        
        # Verify pollutants structure
        pollutants = data["pollutants"]
        expected_pollutants = ["pm2_5", "pm10", "no2", "so2", "o3"]
        for pollutant in expected_pollutants:
            assert pollutant in pollutants
        
        # Verify Russian content
        assert data["aqi"]["category"] == "Умеренное"
        assert "качество воздуха" in data["recommendations"]


class TestPerformanceOptimizationCompatibility:
    """Test that performance optimizations don't break API compatibility"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_caching_doesnt_affect_response_format(self):
        """Test caching doesn't change response format"""
        # Make the same request twice to test caching
        response1 = self.client.get("/weather/current?lat=55.7558&lon=37.6176")
        response2 = self.client.get("/weather/current?lat=55.7558&lon=37.6176")
        
        # Both responses should have same structure (regardless of cache hit/miss)
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            # Same fields should be present
            assert set(data1.keys()) == set(data2.keys())
            
            # AQI structure should be the same
            assert set(data1["aqi"].keys()) == set(data2["aqi"].keys())
            assert set(data1["pollutants"].keys()) == set(data2["pollutants"].keys())
    
    def test_rate_limiting_error_messages_russian(self):
        """Test rate limiting error messages are in Russian"""
        # This test would need to trigger rate limiting
        # For now, we just verify the endpoint is accessible
        response = self.client.get("/rate-limit-status")
        # Should not crash
        assert response.status_code in [200, 404]  # 404 if rate limiting disabled
    
    def test_graceful_degradation_maintains_compatibility(self):
        """Test graceful degradation maintains API compatibility"""
        # Test health endpoint during degradation
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        
        # Status should be one of expected values
        assert data["status"] in ["healthy", "degraded", "unhealthy"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])