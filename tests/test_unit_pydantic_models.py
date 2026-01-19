"""
Unit tests for Pydantic models.

Тестирует edge cases валидации, сериализацию/десериализацию
и корректность работы всех Pydantic моделей.
**Validates: Requirements 8.1, 8.2**
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
import json

from schemas import (
    CoordinatesRequest,
    PollutantData,
    AQIInfo,
    LocationInfo,
    AirQualityData,
    HealthCheckResponse,
    ErrorResponse,
    OpenMeteoResponse,
    CacheEntry
)


class TestCoordinatesRequest:
    """Unit тесты для модели CoordinatesRequest"""
    
    def test_valid_coordinates(self):
        """Тест валидных координат"""
        coords = CoordinatesRequest(lat=55.7558, lon=37.6176)
        assert coords.lat == 55.7558
        assert coords.lon == 37.6176
    
    def test_boundary_coordinates(self):
        """Тест граничных значений координат"""
        # Максимальные значения
        coords_max = CoordinatesRequest(lat=90.0, lon=180.0)
        assert coords_max.lat == 90.0
        assert coords_max.lon == 180.0
        
        # Минимальные значения
        coords_min = CoordinatesRequest(lat=-90.0, lon=-180.0)
        assert coords_min.lat == -90.0
        assert coords_min.lon == -180.0
        
        # Нулевые значения
        coords_zero = CoordinatesRequest(lat=0.0, lon=0.0)
        assert coords_zero.lat == 0.0
        assert coords_zero.lon == 0.0
    
    def test_invalid_latitude(self):
        """Тест невалидной широты"""
        with pytest.raises(ValidationError) as exc_info:
            CoordinatesRequest(lat=91.0, lon=0.0)
        
        errors = exc_info.value.errors()
        assert any('lat' in str(error.get('loc', [])) for error in errors)
        
        with pytest.raises(ValidationError) as exc_info:
            CoordinatesRequest(lat=-91.0, lon=0.0)
        
        errors = exc_info.value.errors()
        assert any('lat' in str(error.get('loc', [])) for error in errors)
    
    def test_invalid_longitude(self):
        """Тест невалидной долготы"""
        with pytest.raises(ValidationError) as exc_info:
            CoordinatesRequest(lat=0.0, lon=181.0)
        
        errors = exc_info.value.errors()
        assert any('lon' in str(error.get('loc', [])) for error in errors)
        
        with pytest.raises(ValidationError) as exc_info:
            CoordinatesRequest(lat=0.0, lon=-181.0)
        
        errors = exc_info.value.errors()
        assert any('lon' in str(error.get('loc', [])) for error in errors)
    
    def test_missing_fields(self):
        """Тест отсутствующих обязательных полей"""
        with pytest.raises(ValidationError):
            CoordinatesRequest(lat=55.7558)  # Отсутствует lon
        
        with pytest.raises(ValidationError):
            CoordinatesRequest(lon=37.6176)  # Отсутствует lat
    
    def test_serialization(self):
        """Тест сериализации в JSON"""
        coords = CoordinatesRequest(lat=55.7558, lon=37.6176)
        json_data = coords.model_dump()
        
        assert json_data['lat'] == 55.7558
        assert json_data['lon'] == 37.6176
        
        # Тест JSON строки
        json_str = coords.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed['lat'] == 55.7558
        assert parsed['lon'] == 37.6176


class TestPollutantData:
    """Unit тесты для модели PollutantData"""
    
    def test_all_pollutants_present(self):
        """Тест с всеми загрязнителями"""
        pollutants = PollutantData(
            pm2_5=25.4,
            pm10=45.2,
            no2=35.1,
            so2=12.3,
            o3=85.7
        )
        
        assert pollutants.pm2_5 == 25.4
        assert pollutants.pm10 == 45.2
        assert pollutants.no2 == 35.1
        assert pollutants.so2 == 12.3
        assert pollutants.o3 == 85.7
    
    def test_partial_pollutants(self):
        """Тест с частичными данными о загрязнителях"""
        pollutants = PollutantData(pm2_5=25.4, no2=35.1)
        
        assert pollutants.pm2_5 == 25.4
        assert pollutants.pm10 is None
        assert pollutants.no2 == 35.1
        assert pollutants.so2 is None
        assert pollutants.o3 is None
    
    def test_empty_pollutants(self):
        """Тест с пустыми данными"""
        pollutants = PollutantData()
        
        assert pollutants.pm2_5 is None
        assert pollutants.pm10 is None
        assert pollutants.no2 is None
        assert pollutants.so2 is None
        assert pollutants.o3 is None
    
    def test_zero_values(self):
        """Тест с нулевыми значениями"""
        pollutants = PollutantData(
            pm2_5=0.0,
            pm10=0.0,
            no2=0.0,
            so2=0.0,
            o3=0.0
        )
        
        assert pollutants.pm2_5 == 0.0
        assert pollutants.pm10 == 0.0
        assert pollutants.no2 == 0.0
        assert pollutants.so2 == 0.0
        assert pollutants.o3 == 0.0
    
    def test_negative_values(self):
        """Тест с отрицательными значениями (должны быть отклонены)"""
        with pytest.raises(ValidationError):
            PollutantData(pm2_5=-1.0)
        
        with pytest.raises(ValidationError):
            PollutantData(pm10=-5.0)
        
        with pytest.raises(ValidationError):
            PollutantData(no2=-10.0)


class TestAQIInfo:
    """Unit тесты для модели AQIInfo"""
    
    def test_valid_aqi_info(self):
        """Тест валидной информации об AQI"""
        aqi = AQIInfo(
            value=85,
            category="Умеренное",
            color="#FFFF00",
            description="Качество воздуха приемлемо для большинства людей"
        )
        
        assert aqi.value == 85
        assert aqi.category == "Умеренное"
        assert aqi.color == "#FFFF00"
        assert aqi.description == "Качество воздуха приемлемо для большинства людей"
    
    def test_boundary_aqi_values(self):
        """Тест граничных значений AQI"""
        # Минимальное значение
        aqi_min = AQIInfo(
            value=0,
            category="Хорошее",
            color="#00E400",
            description="Отличное качество воздуха"
        )
        assert aqi_min.value == 0
        
        # Максимальное значение
        aqi_max = AQIInfo(
            value=500,
            category="Опасно",
            color="#7E0023",
            description="Чрезвычайная ситуация"
        )
        assert aqi_max.value == 500
    
    def test_invalid_aqi_values(self):
        """Тест невалидных значений AQI"""
        with pytest.raises(ValidationError):
            AQIInfo(
                value=-1,
                category="Хорошее",
                color="#00E400",
                description="Отличное качество воздуха"
            )
        
        with pytest.raises(ValidationError):
            AQIInfo(
                value=501,
                category="Опасно",
                color="#7E0023",
                description="Чрезвычайная ситуация"
            )
    
    def test_invalid_color_format(self):
        """Тест невалидного формата цвета"""
        with pytest.raises(ValidationError):
            AQIInfo(
                value=85,
                category="Умеренное",
                color="FFFF00",  # Отсутствует #
                description="Описание"
            )
        
        with pytest.raises(ValidationError):
            AQIInfo(
                value=85,
                category="Умеренное",
                color="#GGGG00",  # Невалидные hex символы
                description="Описание"
            )
        
        with pytest.raises(ValidationError):
            AQIInfo(
                value=85,
                category="Умеренное",
                color="#FFF",  # Неправильная длина
                description="Описание"
            )


class TestAirQualityData:
    """Unit тесты для модели AirQualityData"""
    
    def test_complete_air_quality_data(self):
        """Тест полных данных о качестве воздуха"""
        location = LocationInfo(latitude=55.7558, longitude=37.6176)
        aqi = AQIInfo(
            value=85,
            category="Умеренное",
            color="#FFFF00",
            description="Качество воздуха приемлемо"
        )
        pollutants = PollutantData(pm2_5=25.4, pm10=45.2, no2=35.1)
        
        air_quality = AirQualityData(
            location=location,
            aqi=aqi,
            pollutants=pollutants,
            recommendations="Ограничить активность на улице",
            nmu_risk="medium",
            health_warnings=["Чувствительные люди должны быть осторожны"]
        )
        
        assert air_quality.location.latitude == 55.7558
        assert air_quality.aqi.value == 85
        assert air_quality.pollutants.pm2_5 == 25.4
        assert air_quality.recommendations == "Ограничить активность на улице"
        assert air_quality.nmu_risk == "medium"
        assert len(air_quality.health_warnings) == 1
    
    def test_default_timestamp(self):
        """Тест автоматической генерации timestamp"""
        location = LocationInfo(latitude=55.7558, longitude=37.6176)
        aqi = AQIInfo(value=50, category="Хорошее", color="#00E400", description="Хорошо")
        pollutants = PollutantData()
        
        air_quality = AirQualityData(
            location=location,
            aqi=aqi,
            pollutants=pollutants,
            recommendations="Нет ограничений"
        )
        
        assert air_quality.timestamp is not None
        assert isinstance(air_quality.timestamp, datetime)
        # Проверяем, что timestamp близок к текущему времени (в пределах 1 минуты)
        now = datetime.now(timezone.utc)
        time_diff = abs((now - air_quality.timestamp).total_seconds())
        assert time_diff < 60
    
    def test_timestamp_serialization(self):
        """Тест сериализации timestamp в ISO формат"""
        location = LocationInfo(latitude=55.7558, longitude=37.6176)
        aqi = AQIInfo(value=50, category="Хорошее", color="#00E400", description="Хорошо")
        pollutants = PollutantData()
        
        air_quality = AirQualityData(
            location=location,
            aqi=aqi,
            pollutants=pollutants,
            recommendations="Нет ограничений"
        )
        
        json_data = air_quality.model_dump()
        assert isinstance(json_data['timestamp'], str)
        assert 'T' in json_data['timestamp']  # ISO формат содержит T
    
    def test_empty_health_warnings(self):
        """Тест с пустыми предупреждениями о здоровье"""
        location = LocationInfo(latitude=55.7558, longitude=37.6176)
        aqi = AQIInfo(value=50, category="Хорошее", color="#00E400", description="Хорошо")
        pollutants = PollutantData()
        
        air_quality = AirQualityData(
            location=location,
            aqi=aqi,
            pollutants=pollutants,
            recommendations="Нет ограничений"
        )
        
        assert air_quality.health_warnings == []


class TestHealthCheckResponse:
    """Unit тесты для модели HealthCheckResponse"""
    
    def test_healthy_status(self):
        """Тест здорового статуса"""
        health = HealthCheckResponse(
            status="healthy",
            services={
                "api": "healthy",
                "external_api": "healthy",
                "cache": "healthy"
            }
        )
        
        assert health.status == "healthy"
        assert health.services["api"] == "healthy"
        assert health.services["external_api"] == "healthy"
        assert health.services["cache"] == "healthy"
    
    def test_unhealthy_status(self):
        """Тест нездорового статуса"""
        health = HealthCheckResponse(
            status="unhealthy",
            services={
                "api": "healthy",
                "external_api": "unhealthy",
                "cache": "healthy"
            }
        )
        
        assert health.status == "unhealthy"
        assert health.services["external_api"] == "unhealthy"
    
    def test_timestamp_generation(self):
        """Тест автоматической генерации timestamp"""
        health = HealthCheckResponse(
            status="healthy",
            services={"api": "healthy"}
        )
        
        assert health.timestamp is not None
        assert isinstance(health.timestamp, datetime)


class TestErrorResponse:
    """Unit тесты для модели ErrorResponse"""
    
    def test_basic_error(self):
        """Тест базовой ошибки"""
        error = ErrorResponse(
            code="VALIDATION_ERROR",
            message="Некорректные данные"
        )
        
        assert error.code == "VALIDATION_ERROR"
        assert error.message == "Некорректные данные"
        assert error.details is None
        assert error.timestamp is not None
    
    def test_error_with_details(self):
        """Тест ошибки с деталями"""
        error = ErrorResponse(
            code="VALIDATION_ERROR",
            message="Некорректные координаты",
            details={
                "latitude": "Значение должно быть между -90 и 90",
                "longitude": "Значение должно быть между -180 и 180"
            }
        )
        
        assert error.code == "VALIDATION_ERROR"
        assert error.message == "Некорректные координаты"
        assert error.details["latitude"] == "Значение должно быть между -90 и 90"
        assert error.details["longitude"] == "Значение должно быть между -180 и 180"
    
    def test_timestamp_serialization(self):
        """Тест сериализации timestamp"""
        error = ErrorResponse(
            code="TEST_ERROR",
            message="Тестовая ошибка"
        )
        
        json_data = error.model_dump()
        assert isinstance(json_data['timestamp'], str)


class TestOpenMeteoResponse:
    """Unit тесты для модели OpenMeteoResponse"""
    
    def test_basic_response(self):
        """Тест базового ответа от Open-Meteo API"""
        response = OpenMeteoResponse(
            latitude=55.7558,
            longitude=37.6176,
            elevation=156.0,
            hourly={
                "time": ["2024-01-15T00:00", "2024-01-15T01:00"],
                "pm10": [15.2, 18.1],
                "pm2_5": [8.5, 12.3]
            },
            hourly_units={
                "pm10": "μg/m³",
                "pm2_5": "μg/m³"
            }
        )
        
        assert response.latitude == 55.7558
        assert response.longitude == 37.6176
        assert response.elevation == 156.0
        assert len(response.hourly["time"]) == 2
        assert response.hourly_units["pm10"] == "μg/m³"
    
    def test_response_without_elevation(self):
        """Тест ответа без elevation"""
        response = OpenMeteoResponse(
            latitude=55.7558,
            longitude=37.6176,
            hourly={"time": ["2024-01-15T00:00"]},
            hourly_units={"time": "iso8601"}
        )
        
        assert response.elevation is None
    
    def test_extra_fields_allowed(self):
        """Тест разрешения дополнительных полей"""
        response = OpenMeteoResponse(
            latitude=55.7558,
            longitude=37.6176,
            hourly={"time": ["2024-01-15T00:00"]},
            hourly_units={"time": "iso8601"},
            extra_field="extra_value"  # Дополнительное поле
        )
        
        assert hasattr(response, 'extra_field')
        assert response.extra_field == "extra_value"


class TestCacheEntry:
    """Unit тесты для модели CacheEntry"""
    
    def test_basic_cache_entry(self):
        """Тест базовой записи кэша"""
        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            data={"aqi": 85, "category": "Умеренное"},
            timestamp=now
        )
        
        assert entry.data["aqi"] == 85
        assert entry.timestamp == now
        assert entry.ttl_seconds == 900  # Значение по умолчанию
    
    def test_custom_ttl(self):
        """Тест с пользовательским TTL"""
        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            data={"test": "data"},
            timestamp=now,
            ttl_seconds=1800  # 30 минут
        )
        
        assert entry.ttl_seconds == 1800
    
    def test_is_expired_fresh(self):
        """Тест проверки свежести записи"""
        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            data={"test": "data"},
            timestamp=now,
            ttl_seconds=900
        )
        
        assert not entry.is_expired()  # Только что созданная запись не должна быть просроченной
    
    def test_is_expired_old(self):
        """Тест проверки просроченной записи"""
        from datetime import timedelta
        
        old_time = datetime.now(timezone.utc) - timedelta(seconds=1000)  # 1000 секунд назад
        entry = CacheEntry(
            data={"test": "data"},
            timestamp=old_time,
            ttl_seconds=900  # TTL 900 секунд
        )
        
        assert entry.is_expired()  # Запись должна быть просроченной


class TestModelIntegration:
    """Интеграционные тесты моделей"""
    
    def test_full_json_serialization_deserialization(self):
        """Тест полной сериализации/десериализации в JSON"""
        # Создаем полную модель AirQualityData
        location = LocationInfo(latitude=55.7558, longitude=37.6176)
        aqi = AQIInfo(
            value=85,
            category="Умеренное",
            color="#FFFF00",
            description="Качество воздуха приемлемо"
        )
        pollutants = PollutantData(pm2_5=25.4, pm10=45.2, no2=35.1)
        
        original = AirQualityData(
            location=location,
            aqi=aqi,
            pollutants=pollutants,
            recommendations="Ограничить активность на улице",
            nmu_risk="medium",
            health_warnings=["Предупреждение для чувствительных людей"]
        )
        
        # Сериализация в JSON
        json_str = original.model_dump_json()
        json_data = json.loads(json_str)
        
        # Проверяем основные поля
        assert json_data['location']['latitude'] == 55.7558
        assert json_data['aqi']['value'] == 85
        assert json_data['pollutants']['pm2_5'] == 25.4
        assert json_data['recommendations'] == "Ограничить активность на улице"
        
        # Десериализация из JSON
        restored = AirQualityData.model_validate(json_data)
        
        assert restored.location.latitude == original.location.latitude
        assert restored.aqi.value == original.aqi.value
        assert restored.pollutants.pm2_5 == original.pollutants.pm2_5
        assert restored.recommendations == original.recommendations
    
    def test_nested_validation_errors(self):
        """Тест ошибок валидации во вложенных моделях"""
        with pytest.raises(ValidationError) as exc_info:
            AirQualityData(
                location=LocationInfo(latitude=91.0, longitude=37.6176),  # Невалидная широта
                aqi=AQIInfo(
                    value=600,  # Невалидное значение AQI
                    category="Тест",
                    color="invalid_color",  # Невалидный цвет
                    description="Тест"
                ),
                pollutants=PollutantData(pm2_5=-1.0),  # Отрицательное значение
                recommendations="Тест"
            )
        
        errors = exc_info.value.errors()
        # Проверяем, что есть ошибки валидации
        assert len(errors) > 0
        
        # Проверяем наличие ошибок - достаточно того, что ValidationError был поднят
        # Это означает, что валидация работает корректно