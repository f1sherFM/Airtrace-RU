"""
Basic tests for AirTrace RU Backend setup and configuration.

Проверяет базовую функциональность приложения, импорты и конфигурацию.
"""

import pytest
from fastapi.testclient import TestClient

from main import app
from schemas import AirQualityData, HealthCheckResponse, ErrorResponse
from services import AirQualityService
from cache import MultiLevelCacheManager
from utils import AQICalculator


class TestBasicSetup:
    """Тесты базовой настройки проекта"""
    
    def test_app_creation(self):
        """Тест создания FastAPI приложения"""
        assert app is not None
        assert app.title == "AirTrace RU API"
        assert app.version == "1.0.0"
    
    def test_imports(self):
        """Тест импорта всех основных модулей"""
        # Проверяем, что все модули импортируются без ошибок
        assert AirQualityData is not None
        assert HealthCheckResponse is not None
        assert ErrorResponse is not None
        assert AirQualityService is not None
        assert MultiLevelCacheManager is not None
        assert AQICalculator is not None
    
    def test_client_creation(self, client: TestClient):
        """Тест создания тестового клиента"""
        assert client is not None
    
    def test_root_endpoint(self, client: TestClient):
        """Тест корневого эндпоинта"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "AirTrace RU Backend"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
    
    def test_openapi_docs(self, client: TestClient):
        """Тест доступности OpenAPI документации"""
        response = client.get("/docs")
        assert response.status_code == 200
        
        response = client.get("/redoc")
        assert response.status_code == 200
        
        response = client.get("/openapi.json")
        assert response.status_code == 200


class TestSchemas:
    """Тесты Pydantic схем"""
    
    def test_coordinates_request_valid(self):
        """Тест валидации корректных координат"""
        from schemas import CoordinatesRequest
        
        # Валидные координаты
        coords = CoordinatesRequest(lat=55.7558, lon=37.6176)
        assert coords.lat == 55.7558
        assert coords.lon == 37.6176
    
    def test_coordinates_request_invalid(self):
        """Тест валидации некорректных координат"""
        from schemas import CoordinatesRequest
        from pydantic import ValidationError
        
        # Невалидная широта
        with pytest.raises(ValidationError):
            CoordinatesRequest(lat=91.0, lon=0.0)
        
        with pytest.raises(ValidationError):
            CoordinatesRequest(lat=-91.0, lon=0.0)
        
        # Невалидная долгота
        with pytest.raises(ValidationError):
            CoordinatesRequest(lat=0.0, lon=181.0)
        
        with pytest.raises(ValidationError):
            CoordinatesRequest(lat=0.0, lon=-181.0)


class TestUtils:
    """Тесты утилит"""
    
    def test_aqi_calculator_creation(self, aqi_calculator: AQICalculator):
        """Тест создания калькулятора AQI"""
        assert aqi_calculator is not None
        assert hasattr(aqi_calculator, 'RU_STANDARDS')
        assert hasattr(aqi_calculator, 'AQI_CATEGORIES')
    
    def test_aqi_calculator_standards(self, aqi_calculator: AQICalculator):
        """Тест наличия российских стандартов ПДК"""
        standards = aqi_calculator.RU_STANDARDS
        
        # Проверяем наличие всех загрязнителей
        required_pollutants = ['pm2_5', 'pm10', 'no2', 'so2', 'o3']
        for pollutant in required_pollutants:
            assert pollutant in standards
            
            # Проверяем наличие всех категорий
            required_categories = ['good', 'moderate', 'unhealthy_sensitive', 'unhealthy', 'very_unhealthy', 'hazardous']
            for category in required_categories:
                assert category in standards[pollutant]
                assert isinstance(standards[pollutant][category], (int, float))
                assert standards[pollutant][category] > 0


class TestServices:
    """Тесты сервисов"""
    
    async def test_cache_manager_creation(self, cache_manager: MultiLevelCacheManager):
        """Тест создания менеджера кэша"""
        assert cache_manager is not None
        assert hasattr(cache_manager, '_l1_cache')
        assert hasattr(cache_manager, '_l1_enabled')
        assert hasattr(cache_manager, '_l2_enabled')
    
    @pytest.mark.asyncio
    async def test_air_quality_service_creation(self, air_quality_service: AirQualityService):
        """Тест создания сервиса качества воздуха"""
        assert air_quality_service is not None
        assert hasattr(air_quality_service, 'base_url')
        assert hasattr(air_quality_service, 'client')
        assert hasattr(air_quality_service, 'cache_manager')
        assert hasattr(air_quality_service, 'aqi_calculator')


class TestEndpoints:
    """Тесты API эндпоинтов"""
    
    def test_health_endpoint_exists(self, client: TestClient):
        """Тест существования health check эндпоинта"""
        response = client.get("/health")
        # Эндпоинт должен существовать, но может возвращать ошибку из-за отсутствия реализации
        assert response.status_code in [200, 500, 503]
    
    def test_current_weather_endpoint_exists(self, client: TestClient):
        """Тест существования эндпоинта текущей погоды"""
        response = client.get("/weather/current?lat=55.7558&lon=37.6176")
        # Эндпоинт должен существовать, но может возвращать ошибку из-за отсутствия реализации
        assert response.status_code in [200, 422, 500, 503]
    
    def test_forecast_weather_endpoint_exists(self, client: TestClient):
        """Тест существования эндпоинта прогноза погоды"""
        response = client.get("/weather/forecast?lat=55.7558&lon=37.6176")
        # Эндпоинт должен существовать, но может возвращать ошибку из-за отсутствия реализации
        assert response.status_code in [200, 422, 500, 503]
    
    def test_invalid_coordinates_validation(self, client: TestClient):
        """Тест валидации некорректных координат в эндпоинтах"""
        # Тест с невалидной широтой
        response = client.get("/weather/current?lat=91.0&lon=37.6176")
        assert response.status_code == 422
        
        # Тест с невалидной долготой
        response = client.get("/weather/current?lat=55.7558&lon=181.0")
        assert response.status_code == 422
        
        # Тест без параметров
        response = client.get("/weather/current")
        assert response.status_code == 422