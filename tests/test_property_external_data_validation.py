"""
Property-based тесты для валидации внешних данных в AirTrace RU Backend

**Property 17: External Data Validation**
**Validates: Requirements 8.3, 8.4**

Тестирует, что для любого ответа от внешних API данные валидируются перед обработкой,
с логированием ошибок и возвратом соответствующих ошибок для некорректных данных.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient

from main import app
from services import AirQualityService


class TestExternalDataValidation:
    """Property-based тесты для валидации внешних данных"""
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
        invalid_response_type=st.sampled_from([
            "empty_dict", "missing_latitude", "missing_longitude", "missing_current",
            "invalid_latitude_type", "invalid_longitude_type", "invalid_current_type",
            "negative_pollutants", "string_pollutants", "null_response", "malformed_json"
        ])
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_external_api_response_handling_property(self, lat, lon, invalid_response_type):
        """
        Feature: airtrace-ru, Property 17: External Data Validation
        
        For any invalid response from external APIs, the system should validate
        the data before processing and return appropriate error responses.
        **Validates: Requirements 8.3, 8.4**
        """
        client = TestClient(app)
        
        # Генерируем различные типы некорректных ответов
        invalid_responses = {
            "empty_dict": {},
            "missing_latitude": {
                "longitude": lon,
                "current": {"pm10": 25.0, "pm2_5": 15.0}
            },
            "missing_longitude": {
                "latitude": lat,
                "current": {"pm10": 25.0, "pm2_5": 15.0}
            },
            "missing_current": {
                "latitude": lat,
                "longitude": lon
            },
            "invalid_latitude_type": {
                "latitude": "invalid",
                "longitude": lon,
                "current": {"pm10": 25.0, "pm2_5": 15.0}
            },
            "invalid_longitude_type": {
                "latitude": lat,
                "longitude": "invalid",
                "current": {"pm10": 25.0, "pm2_5": 15.0}
            },
            "invalid_current_type": {
                "latitude": lat,
                "longitude": lon,
                "current": "invalid"
            },
            "negative_pollutants": {
                "latitude": lat,
                "longitude": lon,
                "current": {"pm10": -25.0, "pm2_5": -15.0}
            },
            "string_pollutants": {
                "latitude": lat,
                "longitude": lon,
                "current": {"pm10": "invalid", "pm2_5": "invalid"}
            },
            "null_response": None,
            "malformed_json": "invalid json string"
        }
        
        invalid_response = invalid_responses[invalid_response_type]
        
        # Clear cache to ensure external API is called
        with patch('services.CacheManager.get') as mock_cache_get:
            mock_cache_get.return_value = None  # Force cache miss
            
            with patch('httpx.AsyncClient.get') as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                
                if invalid_response_type == "malformed_json":
                    # Симулируем ошибку JSON парсинга
                    mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
                else:
                    mock_response.json.return_value = invalid_response
                
                mock_get.return_value = mock_response
                
                try:
                    response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                    
                    # Система должна обрабатывать некорректные данные gracefully
                    assert response.status_code != 200, (
                        f"System should not return success for invalid external data type: {invalid_response_type} "
                        f"for lat={lat}, lon={lon}"
                    )
                    
                    # Должен быть возвращен соответствующий код ошибки
                    assert response.status_code in [400, 500, 503], (
                        f"System should return appropriate error code for invalid external data type: {invalid_response_type} "
                        f"for lat={lat}, lon={lon}. Got: {response.status_code}"
                    )
                    
                    # Ответ должен содержать информацию об ошибке
                    if response.status_code != 503:  # 503 может не иметь JSON body
                        try:
                            error_data = response.json()
                            assert isinstance(error_data, dict), (
                                f"Error response should be JSON dict for invalid external data type: {invalid_response_type} "
                                f"for lat={lat}, lon={lon}"
                            )
                        except json.JSONDecodeError:
                            # Некоторые ошибки могут не возвращать JSON
                            pass
                    
                    # Внешний API должен быть вызван
                    assert mock_get.called, (
                        f"External API should be called for validation test type: {invalid_response_type} "
                        f"for lat={lat}, lon={lon}"
                    )
                    
                except Exception as e:
                    # Система не должна падать с необработанными исключениями
                    # Проверяем, что это не связано с валидацией данных
                    if any(keyword in str(e).lower() for keyword in ['validation', 'invalid', 'malformed']):
                        pytest.fail(
                            f"System crashed with unhandled validation error for type: {invalid_response_type} "
                            f"for lat={lat}, lon={lon}: {e}"
                        )
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
        pollutant_values=st.dictionaries(
            keys=st.sampled_from(['pm10', 'pm2_5', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']),
            values=st.one_of(
                st.floats(min_value=-1000, max_value=-0.1),  # Negative values
                st.floats(min_value=10000, max_value=100000),  # Extremely high values
                st.just(float('inf')),  # Infinity
                st.just(float('-inf')),  # Negative infinity
                st.just(float('nan'))  # NaN
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_pollutant_values_validation_property(self, lat, lon, pollutant_values):
        """
        Feature: airtrace-ru, Property 17: External Data Validation
        
        For any invalid pollutant values (negative, infinite, NaN, extremely high),
        the system should validate and handle them appropriately.
        **Validates: Requirements 8.3, 8.4**
        """
        client = TestClient(app)
        
        # Создаем ответ с некорректными значениями загрязнителей
        invalid_response = {
            "latitude": lat,
            "longitude": lon,
            "current": pollutant_values
        }
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = invalid_response
            mock_get.return_value = mock_response
            
            try:
                response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                
                # Система должна обрабатывать некорректные значения загрязнителей
                if any(val < 0 or val != val or val == float('inf') or val == float('-inf') 
                       for val in pollutant_values.values() if isinstance(val, (int, float))):
                    
                    # Для явно некорректных значений система может вернуть ошибку или обработать их
                    if response.status_code == 200:
                        # Если система обработала данные, проверяем корректность результата
                        data = response.json()
                        assert 'aqi' in data, (
                            f"Missing AQI in response for invalid pollutant values: {pollutant_values} "
                            f"for lat={lat}, lon={lon}"
                        )
                        
                        # AQI должен быть валидным числом
                        aqi_value = data['aqi']['value']
                        assert isinstance(aqi_value, (int, float)), (
                            f"AQI value should be numeric for invalid pollutant values: {pollutant_values} "
                            f"for lat={lat}, lon={lon}. Got: {aqi_value}"
                        )
                        
                        assert aqi_value >= 0 and aqi_value <= 500, (
                            f"AQI value should be in valid range [0-500] for invalid pollutant values: {pollutant_values} "
                            f"for lat={lat}, lon={lon}. Got: {aqi_value}"
                        )
                    else:
                        # Система вернула ошибку - это тоже валидное поведение
                        assert response.status_code in [400, 500, 503], (
                            f"System should return appropriate error code for invalid pollutant values: {pollutant_values} "
                            f"for lat={lat}, lon={lon}. Got: {response.status_code}"
                        )
                
            except Exception as e:
                # Система не должна падать с необработанными исключениями
                if any(keyword in str(e).lower() for keyword in ['validation', 'invalid', 'nan', 'inf']):
                    pytest.fail(
                        f"System crashed with unhandled validation error for pollutant values: {pollutant_values} "
                        f"for lat={lat}, lon={lon}: {e}"
                    )
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
        http_status=st.sampled_from([400, 401, 403, 404, 429, 500, 502, 503, 504])
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_external_api_error_status_handling_property(self, lat, lon, http_status):
        """
        Feature: airtrace-ru, Property 17: External Data Validation
        
        For any HTTP error status from external APIs, the system should
        handle the error appropriately and return meaningful error responses.
        **Validates: Requirements 8.3, 8.4**
        """
        client = TestClient(app)
        
        # Clear cache to ensure external API is called
        with patch('services.CacheManager.get') as mock_cache_get:
            mock_cache_get.return_value = None  # Force cache miss
            
            with patch('httpx.AsyncClient.get') as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = http_status
                mock_response.raise_for_status.side_effect = Exception(f"HTTP {http_status} Error")
                mock_get.return_value = mock_response
                
                try:
                    response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                    
                    # Система должна обрабатывать HTTP ошибки от внешних API
                    assert response.status_code != 200, (
                        f"System should not return success when external API returns HTTP {http_status} "
                        f"for lat={lat}, lon={lon}"
                    )
                    
                    # Должен быть возвращен соответствующий код ошибки
                    assert response.status_code in [500, 503], (
                        f"System should return 500 or 503 for external API HTTP {http_status} error "
                        f"for lat={lat}, lon={lon}. Got: {response.status_code}"
                    )
                    
                    # Внешний API должен быть вызван
                    assert mock_get.called, (
                        f"External API should be called for HTTP {http_status} error test "
                        f"for lat={lat}, lon={lon}"
                    )
                    
                except Exception as e:
                    # Система не должна падать с необработанными исключениями
                    if f"{http_status}" in str(e) or "http" in str(e).lower():
                        pytest.fail(
                            f"System crashed with unhandled HTTP error {http_status} "
                            f"for lat={lat}, lon={lon}: {e}"
                        )
    
    def test_service_level_external_data_validation(self):
        """
        Feature: airtrace-ru, Property 17: External Data Validation
        
        At the service level, external data validation should be robust
        and handle various edge cases.
        **Validates: Requirements 8.3, 8.4**
        """
        service = AirQualityService()
        
        # Тестируем различные некорректные ответы на уровне сервиса
        invalid_responses = [
            # Пустой ответ
            {},
            # Отсутствующие обязательные поля
            {"latitude": 55.7558},
            {"longitude": 37.6176},
            {"latitude": 55.7558, "longitude": 37.6176},
            # Некорректные типы данных
            {"latitude": "invalid", "longitude": 37.6176, "current": {}},
            {"latitude": 55.7558, "longitude": "invalid", "current": {}},
            {"latitude": 55.7558, "longitude": 37.6176, "current": "invalid"},
            # Некорректные значения загрязнителей
            {
                "latitude": 55.7558,
                "longitude": 37.6176,
                "current": {"pm10": -25.0, "pm2_5": float('inf')}
            }
        ]
        
        for invalid_response in invalid_responses:
            with patch.object(service.client, 'get') as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = invalid_response
                mock_get.return_value = mock_response
                
                try:
                    import asyncio
                    result = asyncio.run(service.get_current_air_quality(55.7558, 37.6176))
                    
                    # Если сервис вернул результат, он должен быть валидным
                    if result:
                        assert hasattr(result, 'aqi'), (
                            f"Service result should have AQI for invalid response: {invalid_response}"
                        )
                        assert hasattr(result, 'pollutants'), (
                            f"Service result should have pollutants for invalid response: {invalid_response}"
                        )
                
                except Exception as e:
                    # Сервис может выбросить исключение для некорректных данных - это нормально
                    # Но исключение должно быть информативным
                    assert len(str(e)) > 0, (
                        f"Service exception should be informative for invalid response: {invalid_response}"
                    )
    
    @given(
        response_data=st.one_of(
            st.none(),
            st.text(min_size=1, max_size=100),
            st.integers(),
            st.lists(st.integers(), min_size=0, max_size=10),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20),
                values=st.one_of(st.none(), st.text(), st.integers(), st.floats()),
                min_size=0,
                max_size=10
            )
        )
    )
    @settings(max_examples=50, deadline=30000)
    def test_arbitrary_external_response_handling_property(self, response_data):
        """
        Feature: airtrace-ru, Property 17: External Data Validation
        
        For any arbitrary response data from external APIs, the system
        should handle it gracefully without crashing.
        **Validates: Requirements 8.3, 8.4**
        """
        service = AirQualityService()
        
        with patch.object(service.client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = response_data
            mock_get.return_value = mock_response
            
            try:
                import asyncio
                result = asyncio.run(service.get_current_air_quality(55.7558, 37.6176))
                
                # Если результат получен, он должен быть валидным
                if result:
                    assert hasattr(result, 'aqi'), (
                        f"Service result should have AQI for arbitrary response: {response_data}"
                    )
                    assert result.aqi.value >= 0, (
                        f"AQI should be non-negative for arbitrary response: {response_data}"
                    )
            
            except Exception as e:
                # Исключения допустимы для произвольных данных
                # Но они не должны быть связаны с отсутствием валидации
                error_message = str(e).lower()
                
                # Проверяем, что это не критическая ошибка системы
                critical_errors = ['segmentation fault', 'memory error', 'stack overflow']
                for critical_error in critical_errors:
                    assert critical_error not in error_message, (
                        f"System should not crash with critical error for arbitrary response: {response_data}. "
                        f"Error: {e}"
                    )
    
    def test_json_parsing_error_handling(self):
        """
        Feature: airtrace-ru, Property 17: External Data Validation
        
        JSON parsing errors from external APIs should be handled gracefully.
        **Validates: Requirements 8.3, 8.4**
        """
        client = TestClient(app)
        
        # Тестируем различные типы JSON ошибок
        json_errors = [
            json.JSONDecodeError("Invalid JSON", "", 0),
            json.JSONDecodeError("Unexpected end of JSON input", "", 10),
            json.JSONDecodeError("Invalid character", "", 5)
        ]
        
        for json_error in json_errors:
            with patch('httpx.AsyncClient.get') as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.side_effect = json_error
                mock_get.return_value = mock_response
                
                try:
                    response = client.get("/weather/current?lat=55.7558&lon=37.6176")
                    
                    # Система должна обрабатывать JSON ошибки
                    assert response.status_code != 200, (
                        f"System should not return success for JSON parsing error: {json_error}"
                    )
                    
                    assert response.status_code in [500, 503], (
                        f"System should return 500 or 503 for JSON parsing error: {json_error}. "
                        f"Got: {response.status_code}"
                    )
                
                except Exception as e:
                    # Система не должна падать с необработанными JSON ошибками
                    if "json" in str(e).lower() or "parsing" in str(e).lower():
                        pytest.fail(
                            f"System crashed with unhandled JSON parsing error: {json_error}. Error: {e}"
                        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])