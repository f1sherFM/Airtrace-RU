"""
Property-based tests for external API response parsing.

**Property 3: External API Response Parsing**
**Validates: Requirements 2.2**

Тестирует парсинг ответов от Open-Meteo API с использованием property-based testing
для проверки корректности обработки всех возможных форматов ответов.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timezone

from services import AirQualityService
from schemas import OpenMeteoResponse, AirQualityData


class TestExternalAPIParsingProperty:
    """Property-based тесты для парсинга внешних API"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.service = AirQualityService()
    
    @given(
        latitude=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        longitude=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        elevation=st.one_of(st.none(), st.floats(min_value=-500.0, max_value=9000.0, allow_nan=False, allow_infinity=False)),
        pm10=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        pm2_5=st.one_of(st.none(), st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False)),
        no2=st.one_of(st.none(), st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False)),
        so2=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        o3=st.one_of(st.none(), st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False))
    )
    async def test_current_data_parsing_property(self, latitude: float, longitude: float, elevation: Optional[float],
                                               pm10: Optional[float], pm2_5: Optional[float], no2: Optional[float],
                                               so2: Optional[float], o3: Optional[float]):
        """
        **Property 3: External API Response Parsing**
        **Validates: Requirements 2.2**
        
        For any valid response structure from Open-Meteo API,
        the system should successfully parse and validate the response without errors.
        """
        # Создаем валидную структуру ответа Open-Meteo API
        api_response = {
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
            "current": {
                "time": "2024-01-15T12:00",
                "pm10": pm10,
                "pm2_5": pm2_5,
                "nitrogen_dioxide": no2,
                "sulphur_dioxide": so2,
                "ozone": o3
            },
            "current_units": {
                "time": "iso8601",
                "pm10": "μg/m³",
                "pm2_5": "μg/m³",
                "nitrogen_dioxide": "μg/m³",
                "sulphur_dioxide": "μg/m³",
                "ozone": "μg/m³"
            }
        }
        
        # Тестируем парсинг текущих данных
        try:
            processed_data = await self.service._process_current_data(api_response, latitude, longitude)
            
            # Проверяем, что результат является валидным объектом AirQualityData
            assert isinstance(processed_data, AirQualityData), f"Result should be AirQualityData, got {type(processed_data)}"
            
            # Проверяем основные поля
            assert processed_data.location.latitude == latitude, f"Latitude should match: expected {latitude}, got {processed_data.location.latitude}"
            assert processed_data.location.longitude == longitude, f"Longitude should match: expected {longitude}, got {processed_data.location.longitude}"
            
            # Проверяем, что AQI рассчитан корректно
            assert isinstance(processed_data.aqi.value, int), f"AQI value should be integer, got {type(processed_data.aqi.value)}"
            assert 0 <= processed_data.aqi.value <= 500, f"AQI value should be between 0 and 500, got {processed_data.aqi.value}"
            
            # Проверяем, что категория не пустая
            assert len(processed_data.aqi.category) > 0, "AQI category should not be empty"
            
            # Проверяем цветовой код
            assert processed_data.aqi.color.startswith('#'), f"Color should start with #, got {processed_data.aqi.color}"
            assert len(processed_data.aqi.color) == 7, f"Color should be 7 characters, got {len(processed_data.aqi.color)}"
            
            # Проверяем данные о загрязнителях
            if pm2_5 is not None:
                assert processed_data.pollutants.pm2_5 == pm2_5, f"PM2.5 should match: expected {pm2_5}, got {processed_data.pollutants.pm2_5}"
            if pm10 is not None:
                assert processed_data.pollutants.pm10 == pm10, f"PM10 should match: expected {pm10}, got {processed_data.pollutants.pm10}"
            if no2 is not None:
                assert processed_data.pollutants.no2 == no2, f"NO2 should match: expected {no2}, got {processed_data.pollutants.no2}"
            if so2 is not None:
                assert processed_data.pollutants.so2 == so2, f"SO2 should match: expected {so2}, got {processed_data.pollutants.so2}"
            if o3 is not None:
                assert processed_data.pollutants.o3 == o3, f"O3 should match: expected {o3}, got {processed_data.pollutants.o3}"
            
            # Проверяем рекомендации
            assert isinstance(processed_data.recommendations, str), f"Recommendations should be string, got {type(processed_data.recommendations)}"
            assert len(processed_data.recommendations) > 0, "Recommendations should not be empty"
            
            # Проверяем, что рекомендации на русском языке
            assert any(ord(char) >= 1040 and ord(char) <= 1103 for char in processed_data.recommendations), \
                f"Recommendations should contain Russian text: '{processed_data.recommendations}'"
            
            # Проверяем НМУ риск
            assert processed_data.nmu_risk in ["low", "medium", "high", "critical", "unknown"], \
                f"NMU risk should be valid level, got '{processed_data.nmu_risk}'"
            
            # Проверяем предупреждения о здоровье
            assert isinstance(processed_data.health_warnings, list), f"Health warnings should be list, got {type(processed_data.health_warnings)}"
            for warning in processed_data.health_warnings:
                assert isinstance(warning, str), f"Each warning should be string, got {type(warning)}"
                assert len(warning) > 0, "Warning should not be empty"
            
        except Exception as e:
            pytest.fail(f"Failed to parse valid API response: {e}")
    
    @given(
        latitude=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        longitude=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        hours_count=st.integers(min_value=1, max_value=24),
        pm10_values=st.lists(
            st.one_of(st.none(), st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
            min_size=1, max_size=24
        ),
        pm2_5_values=st.lists(
            st.one_of(st.none(), st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False)),
            min_size=1, max_size=24
        )
    )
    async def test_forecast_data_parsing_property(self, latitude: float, longitude: float, hours_count: int,
                                                pm10_values: List[Optional[float]], pm2_5_values: List[Optional[float]]):
        """
        **Property 3: External API Response Parsing**
        **Validates: Requirements 2.2**
        
        For any valid forecast response structure from Open-Meteo API,
        the system should successfully parse hourly data and return list of AirQualityData.
        """
        assume(len(pm10_values) >= hours_count)
        assume(len(pm2_5_values) >= hours_count)
        
        # Генерируем временные метки
        base_time = "2024-01-15T00:00"
        times = [f"2024-01-15T{i:02d}:00" for i in range(hours_count)]
        
        # Обрезаем массивы до нужного размера
        pm10_values = pm10_values[:hours_count]
        pm2_5_values = pm2_5_values[:hours_count]
        
        # Создаем валидную структуру ответа прогноза
        api_response = {
            "latitude": latitude,
            "longitude": longitude,
            "elevation": 100.0,
            "hourly": {
                "time": times,
                "pm10": pm10_values,
                "pm2_5": pm2_5_values,
                "nitrogen_dioxide": [20.0] * hours_count,  # Фиксированные значения для простоты
                "sulphur_dioxide": [10.0] * hours_count,
                "ozone": [50.0] * hours_count
            },
            "hourly_units": {
                "time": "iso8601",
                "pm10": "μg/m³",
                "pm2_5": "μg/m³",
                "nitrogen_dioxide": "μg/m³",
                "sulphur_dioxide": "μg/m³",
                "ozone": "μg/m³"
            }
        }
        
        # Тестируем парсинг прогнозных данных
        try:
            forecast_data = await self.service._process_forecast_data(api_response, latitude, longitude)
            
            # Проверяем, что результат является списком
            assert isinstance(forecast_data, list), f"Result should be list, got {type(forecast_data)}"
            
            # Проверяем количество элементов (должно быть не больше 24)
            expected_count = min(hours_count, 24)
            assert len(forecast_data) == expected_count, f"Should return {expected_count} items, got {len(forecast_data)}"
            
            # Проверяем каждый элемент прогноза
            for i, item in enumerate(forecast_data):
                assert isinstance(item, AirQualityData), f"Item {i} should be AirQualityData, got {type(item)}"
                
                # Проверяем координаты
                assert item.location.latitude == latitude, f"Item {i} latitude should match"
                assert item.location.longitude == longitude, f"Item {i} longitude should match"
                
                # Проверяем AQI
                assert isinstance(item.aqi.value, int), f"Item {i} AQI should be integer"
                assert 0 <= item.aqi.value <= 500, f"Item {i} AQI should be valid range"
                
                # Проверяем временную метку
                assert isinstance(item.timestamp, datetime), f"Item {i} timestamp should be datetime"
                
                # Проверяем данные о загрязнителях
                if i < len(pm10_values) and pm10_values[i] is not None:
                    assert item.pollutants.pm10 == pm10_values[i], f"Item {i} PM10 should match"
                if i < len(pm2_5_values) and pm2_5_values[i] is not None:
                    assert item.pollutants.pm2_5 == pm2_5_values[i], f"Item {i} PM2.5 should match"
                
                # Проверяем рекомендации
                assert isinstance(item.recommendations, str), f"Item {i} recommendations should be string"
                assert len(item.recommendations) > 0, f"Item {i} recommendations should not be empty"
                
        except Exception as e:
            pytest.fail(f"Failed to parse valid forecast response: {e}")
    
    @given(
        missing_field=st.sampled_from(['latitude', 'longitude', 'current', 'current_units'])
    )
    async def test_malformed_response_handling_property(self, missing_field: str):
        """
        **Property 3: External API Response Parsing**
        **Validates: Requirements 2.2**
        
        For any malformed response from Open-Meteo API (missing required fields),
        the system should handle the error gracefully without crashing.
        """
        # Создаем базовую структуру ответа
        api_response = {
            "latitude": 55.7558,
            "longitude": 37.6176,
            "elevation": 156.0,
            "current": {
                "time": "2024-01-15T12:00",
                "pm10": 25.0,
                "pm2_5": 15.0,
                "nitrogen_dioxide": 30.0,
                "sulphur_dioxide": 10.0,
                "ozone": 60.0
            },
            "current_units": {
                "time": "iso8601",
                "pm10": "μg/m³",
                "pm2_5": "μg/m³",
                "nitrogen_dioxide": "μg/m³",
                "sulphur_dioxide": "μg/m³",
                "ozone": "μg/m³"
            }
        }
        
        # Удаляем указанное поле для создания некорректного ответа
        if missing_field in api_response:
            del api_response[missing_field]
        
        # Тестируем обработку некорректного ответа
        try:
            processed_data = await self.service._process_current_data(api_response, 55.7558, 37.6176)
            
            # Если обработка прошла успешно, проверяем, что результат валиден
            if processed_data:
                assert isinstance(processed_data, AirQualityData), "Result should be AirQualityData if processing succeeds"
                
        except (KeyError, ValueError, TypeError) as e:
            # Ожидаемые исключения при некорректных данных - это нормально
            assert True, f"Expected error for malformed response: {e}"
        except Exception as e:
            # Неожиданные исключения должны быть обработаны
            pytest.fail(f"Unexpected error type for malformed response: {type(e).__name__}: {e}")
    
    @given(
        pollutant_values=st.dictionaries(
            keys=st.sampled_from(['pm10', 'pm2_5', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']),
            values=st.one_of(
                st.none(),
                st.floats(min_value=-100.0, max_value=-0.1),  # Отрицательные значения
                st.floats(min_value=0.0, max_value=10000.0),  # Экстремально высокие значения
                st.text(min_size=1, max_size=10),  # Строковые значения вместо чисел
                st.booleans()  # Булевы значения
            ),
            min_size=1,
            max_size=5
        )
    )
    async def test_invalid_pollutant_values_property(self, pollutant_values: Dict[str, Any]):
        """
        **Property 3: External API Response Parsing**
        **Validates: Requirements 2.2**
        
        For any response with invalid pollutant values (negative, non-numeric, extreme),
        the system should handle them gracefully and produce valid AirQualityData.
        """
        api_response = {
            "latitude": 55.7558,
            "longitude": 37.6176,
            "elevation": 156.0,
            "current": {
                "time": "2024-01-15T12:00",
                **pollutant_values
            },
            "current_units": {
                "time": "iso8601",
                "pm10": "μg/m³",
                "pm2_5": "μg/m³",
                "nitrogen_dioxide": "μg/m³",
                "sulphur_dioxide": "μg/m³",
                "ozone": "μg/m³"
            }
        }
        
        try:
            processed_data = await self.service._process_current_data(api_response, 55.7558, 37.6176)
            
            # Если обработка прошла успешно, проверяем результат
            assert isinstance(processed_data, AirQualityData), "Result should be AirQualityData"
            
            # AQI должен быть валидным даже при некорректных входных данных
            assert isinstance(processed_data.aqi.value, int), "AQI should be integer"
            assert 0 <= processed_data.aqi.value <= 500, "AQI should be in valid range"
            
            # Проверяем, что некорректные значения были отфильтрованы
            pollutants_dict = processed_data.pollutants.model_dump(exclude_none=True)
            for pollutant, value in pollutants_dict.items():
                assert isinstance(value, (int, float)), f"Pollutant {pollutant} should be numeric, got {type(value)}"
                assert value >= 0, f"Pollutant {pollutant} should be non-negative, got {value}"
                
        except Exception as e:
            # Некоторые исключения могут быть ожидаемыми при совсем некорректных данных
            assert isinstance(e, (ValueError, TypeError, KeyError)), f"Should handle invalid data gracefully, got {type(e).__name__}: {e}"
    
    def test_openmeteo_response_model_property(self):
        """
        **Property 3: External API Response Parsing**
        **Validates: Requirements 2.2**
        
        The OpenMeteoResponse Pydantic model should correctly validate
        typical response structures from Open-Meteo API.
        """
        # Тестируем валидную структуру ответа
        valid_response_data = {
            "latitude": 55.7558,
            "longitude": 37.6176,
            "elevation": 156.0,
            "hourly": {
                "time": ["2024-01-15T00:00", "2024-01-15T01:00"],
                "pm10": [25.0, 30.0],
                "pm2_5": [15.0, 18.0]
            },
            "hourly_units": {
                "time": "iso8601",
                "pm10": "μg/m³",
                "pm2_5": "μg/m³"
            }
        }
        
        try:
            response_model = OpenMeteoResponse(**valid_response_data)
            
            assert response_model.latitude == 55.7558, "Latitude should match"
            assert response_model.longitude == 37.6176, "Longitude should match"
            assert response_model.elevation == 156.0, "Elevation should match"
            assert "time" in response_model.hourly, "Hourly data should contain time"
            assert "pm10" in response_model.hourly, "Hourly data should contain pm10"
            
        except Exception as e:
            pytest.fail(f"Valid OpenMeteo response should parse successfully: {e}")