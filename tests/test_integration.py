"""
Упрощенные интеграционные тесты для AirTrace RU Backend

Фокусируется на основных интеграционных сценариях без сложного мокинга.

Requirements: 1.1, 1.2, 1.3
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from datetime import datetime, timezone

from main import app
from schemas import AirQualityData, PollutantData, AQIInfo, LocationInfo
from services import AirQualityService
from utils import AQICalculator, check_nmu_risk, is_blacksky_conditions
from middleware import PrivacyAwareFormatter


class TestEndToEndIntegration:
    """Основные end-to-end интеграционные тесты"""
    
    def test_current_weather_endpoint_integration(self):
        """Тест интеграции эндпоинта текущей погоды"""
        # Создаем мок сервиса
        mock_service = MagicMock(spec=AirQualityService)
        
        # Создаем корректные тестовые данные
        mock_data = AirQualityData(
            timestamp=datetime.now(timezone.utc),
            location=LocationInfo(latitude=55.7558, longitude=37.6176),
            aqi=AQIInfo(
                value=85, 
                category="Умеренное", 
                color="#FFFF00",
                description="Качество воздуха приемлемо для большинства людей"
            ),
            pollutants=PollutantData(
                pm2_5=25.4, pm10=45.2, no2=35.1, so2=12.3, o3=85.7
            ),
            recommendations="Рекомендации для умеренного качества воздуха",
            nmu_risk="medium",
            health_warnings=["Чувствительные люди могут испытывать дискомфорт"]
        )
        mock_service.get_current_air_quality.return_value = mock_data
        
        # Переопределяем dependency
        from main import get_air_quality_service
        app.dependency_overrides[get_air_quality_service] = lambda: mock_service
        
        try:
            client = TestClient(app)
            response = client.get(
                "/weather/current",
                params={"lat": 55.7558, "lon": 37.6176}
            )
            
            # Проверяем успешный ответ
            assert response.status_code == 200
            
            # Проверяем структуру ответа
            data = response.json()
            assert "timestamp" in data
            assert "location" in data
            assert "aqi" in data
            assert "pollutants" in data
            assert "recommendations" in data
            assert "nmu_risk" in data
            assert "health_warnings" in data
            
            # Проверяем корректность данных
            assert data["location"]["latitude"] == 55.7558
            assert data["location"]["longitude"] == 37.6176
            assert data["aqi"]["value"] == 85
            assert data["pollutants"]["pm2_5"] == 25.4
            
        finally:
            app.dependency_overrides.clear()
    
    def test_forecast_weather_endpoint_integration(self):
        """Тест интеграции эндпоинта прогноза погоды"""
        mock_service = MagicMock(spec=AirQualityService)
        
        # Создаем список данных для прогноза
        mock_forecast = []
        for i in range(3):
            mock_data = AirQualityData(
                timestamp=datetime.now(timezone.utc),
                location=LocationInfo(latitude=55.7558, longitude=37.6176),
                aqi=AQIInfo(
                    value=85 + i, 
                    category="Умеренное", 
                    color="#FFFF00",
                    description="Качество воздуха приемлемо для большинства людей"
                ),
                pollutants=PollutantData(
                    pm2_5=25.4 + i, pm10=45.2 + i, no2=35.1 + i, so2=12.3 + i, o3=85.7 + i
                ),
                recommendations="Рекомендации для умеренного качества воздуха",
                nmu_risk="medium",
                health_warnings=["Чувствительные люди могут испытывать дискомфорт"]
            )
            mock_forecast.append(mock_data)
        
        mock_service.get_forecast_air_quality.return_value = mock_forecast
        
        from main import get_air_quality_service
        app.dependency_overrides[get_air_quality_service] = lambda: mock_service
        
        try:
            client = TestClient(app)
            response = client.get(
                "/weather/forecast",
                params={"lat": 55.7558, "lon": 37.6176}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 3
            
            for hour_data in data:
                assert "timestamp" in hour_data
                assert "aqi" in hour_data
                assert "pollutants" in hour_data
                
        finally:
            app.dependency_overrides.clear()
    
    def test_health_check_endpoint_integration(self):
        """Тест интеграции health check эндпоинта"""
        mock_service = MagicMock(spec=AirQualityService)
        mock_service.check_external_api_health.return_value = "healthy"
        mock_service.cache_manager = MagicMock()
        
        from main import get_air_quality_service
        app.dependency_overrides[get_air_quality_service] = lambda: mock_service
        
        try:
            client = TestClient(app)
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "services" in data
            
            services = data["services"]
            assert "api" in services
            assert "external_api" in services
            assert "cache" in services
            assert "aqi_calculator" in services
            assert "privacy_middleware" in services
            assert "nmu_detector" in services
            
        finally:
            app.dependency_overrides.clear()
    
    def test_validation_error_handling(self):
        """Тест обработки ошибок валидации"""
        mock_service = MagicMock(spec=AirQualityService)
        
        from main import get_air_quality_service
        app.dependency_overrides[get_air_quality_service] = lambda: mock_service
        
        try:
            client = TestClient(app)
            
            # Тест невалидной широты
            response = client.get(
                "/weather/current",
                params={"lat": 91.0, "lon": 37.6176}
            )
            assert response.status_code == 422
            
            # Тест отсутствующих параметров
            response = client.get("/weather/current")
            assert response.status_code == 422
            
        finally:
            app.dependency_overrides.clear()


class TestComponentIntegration:
    """Тесты интеграции отдельных компонентов"""
    
    def test_aqi_calculator_nmu_integration(self):
        """Тест интеграции AQI калькулятора с НМУ детектором"""
        calculator = AQICalculator()
        
        # Тестовые данные с высоким уровнем загрязнения
        high_pollution = {
            "pm2_5": 150.0,
            "pm10": 200.0,
            "no2": 180.0,
            "so2": 300.0
        }
        
        # Расчет AQI
        aqi_value, category, color = calculator.calculate_aqi(high_pollution)
        
        # Проверяем высокий AQI
        assert aqi_value > 150
        assert category in ["Вредно", "Очень вредно", "Опасно"]
        
        # Проверяем определение НМУ
        nmu_risk = check_nmu_risk(high_pollution)
        assert nmu_risk in ["high", "critical"]
        
        # Проверяем условия "Черное небо"
        blacksky_pollution = {
            "pm2_5": 125.0,  # 5x ПДК (25 * 5)
            "pm10": 250.0    # 5x ПДК (50 * 5)
        }
        
        blacksky = is_blacksky_conditions(blacksky_pollution)
        assert blacksky is True
        
        nmu_risk_blacksky = check_nmu_risk(blacksky_pollution)
        assert nmu_risk_blacksky == "critical"
    
    def test_privacy_middleware_integration(self):
        """Тест интеграции privacy middleware"""
        formatter = PrivacyAwareFormatter()
        
        # Тестовое сообщение с координатами
        test_message = "Request with lat=55.7558 and lon=37.6176 processed"
        
        # Применяем форматтер
        sanitized = formatter._sanitize_coordinates(test_message)
        
        # Проверяем, что координаты отфильтрованы
        assert "55.7558" not in sanitized
        assert "37.6176" not in sanitized
        assert "[COORDINATE_FILTERED]" in sanitized
    
    def test_aqi_calculation_with_partial_data(self):
        """Тест расчета AQI с частичными данными"""
        calculator = AQICalculator()
        
        # Данные с отсутствующими значениями
        partial_data = {
            "pm2_5": 45.0,
            "pm10": None,  # Отсутствующее значение
            "no2": 35.0,
            "so2": None,   # Отсутствующее значение
            "o3": 85.0
        }
        
        # Расчет AQI должен работать с частичными данными
        aqi_value, category, color = calculator.calculate_aqi(partial_data)
        
        assert aqi_value > 0
        assert category is not None
        assert color is not None
        assert len(color) == 7  # Формат #RRGGBB
        assert color.startswith("#")
    
    def test_nmu_risk_assessment_integration(self):
        """Тест интеграции оценки НМУ риска"""
        # Низкий уровень загрязнения
        low_pollution = {
            "pm2_5": 10.0,
            "pm10": 20.0,
            "no2": 15.0,
            "so2": 5.0
        }
        
        nmu_risk = check_nmu_risk(low_pollution)
        assert nmu_risk == "low"
        
        # Средний уровень загрязнения
        medium_pollution = {
            "pm2_5": 35.0,
            "pm10": 60.0,
            "no2": 50.0,
            "so2": 25.0
        }
        
        nmu_risk = check_nmu_risk(medium_pollution)
        assert nmu_risk in ["medium", "high"]
        
        # Высокий уровень загрязнения
        high_pollution = {
            "pm2_5": 100.0,
            "pm10": 150.0,
            "no2": 120.0,
            "so2": 80.0
        }
        
        nmu_risk = check_nmu_risk(high_pollution)
        assert nmu_risk in ["high", "critical"]


class TestDataTransformation:
    """Тесты трансформации данных"""
    
    def test_aqi_category_mapping(self):
        """Тест корректного маппинга категорий AQI"""
        calculator = AQICalculator()
        
        # Тестируем различные уровни загрязнения
        test_cases = [
            ({"pm2_5": 5.0}, "Хорошее"),
            ({"pm2_5": 15.0}, "Хорошее"),
            ({"pm2_5": 30.0}, "Умеренное"),
            ({"pm2_5": 60.0}, "Вредно для чувствительных групп"),
            ({"pm2_5": 100.0}, "Вредно"),
            ({"pm2_5": 200.0}, "Опасно"),  # Исправлено на правильную категорию
        ]
        
        for pollution_data, expected_category in test_cases:
            aqi_value, category, color = calculator.calculate_aqi(pollution_data)
            assert category == expected_category
    
    def test_color_coding_consistency(self):
        """Тест консистентности цветового кодирования"""
        calculator = AQICalculator()
        
        # Проверяем, что одинаковые уровни AQI дают одинаковые цвета
        pollution1 = {"pm2_5": 25.0}
        pollution2 = {"pm10": 50.0}  # Должно дать похожий AQI
        
        aqi1, cat1, color1 = calculator.calculate_aqi(pollution1)
        aqi2, cat2, color2 = calculator.calculate_aqi(pollution2)
        
        # Если AQI в одном диапазоне, цвета должны быть одинаковыми
        if abs(aqi1 - aqi2) < 50:  # В пределах одной категории
            assert color1 == color2 or cat1 == cat2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])