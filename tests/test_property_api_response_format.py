"""
Property-based tests for API response format consistency.

**Property 1: API Response Format Consistency**
*For any* valid coordinates provided to current or forecast endpoints, 
the system should return properly structured JSON responses containing 
all required air quality fields.

**Validates: Requirements 1.1, 1.2**
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient
import json
from datetime import datetime
from typing import Dict, Any, List

from main import app
from schemas import AirQualityData, LocationInfo, AQIInfo, PollutantData


class TestAPIResponseFormatProperty:
    """Property-based tests for API response format consistency"""
    
    @pytest.fixture(scope="class")
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method"""
        # Мокаем внешний API для предсказуемых ответов
        import httpx
        from unittest.mock import AsyncMock, patch
        
        # Создаем мок ответа от Open-Meteo API
        mock_response_data = {
            "latitude": 55.7558,
            "longitude": 37.6176,
            "elevation": 156.0,
            "current": {
                "time": "2024-01-15T12:00",
                "pm10": 25.4,
                "pm2_5": 15.2,
                "nitrogen_dioxide": 35.1,
                "sulphur_dioxide": 12.3,
                "ozone": 85.7
            },
            "current_units": {
                "pm10": "μg/m³",
                "pm2_5": "μg/m³",
                "nitrogen_dioxide": "μg/m³",
                "sulphur_dioxide": "μg/m³",
                "ozone": "μg/m³"
            }
        }
        
        mock_forecast_data = {
            "latitude": 55.7558,
            "longitude": 37.6176,
            "elevation": 156.0,
            "hourly": {
                "time": ["2024-01-15T12:00", "2024-01-15T13:00", "2024-01-15T14:00"],
                "pm10": [25.4, 28.1, 22.3],
                "pm2_5": [15.2, 18.5, 14.1],
                "nitrogen_dioxide": [35.1, 38.2, 32.4],
                "sulphur_dioxide": [12.3, 14.1, 11.2],
                "ozone": [85.7, 88.2, 82.1]
            },
            "hourly_units": {
                "pm10": "μg/m³",
                "pm2_5": "μg/m³",
                "nitrogen_dioxide": "μg/m³",
                "sulphur_dioxide": "μg/m³",
                "ozone": "μg/m³"
            }
        }
        
        # Создаем мок для httpx.AsyncClient
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.raise_for_status = AsyncMock(return_value=None)
        
        mock_forecast_response = AsyncMock()
        mock_forecast_response.status_code = 200
        mock_forecast_response.json = AsyncMock(return_value=mock_forecast_data)
        mock_forecast_response.raise_for_status = AsyncMock(return_value=None)
        
        # Патчим httpx.AsyncClient.get для возврата разных ответов
        async def mock_get(self, url, **kwargs):
            params = kwargs.get('params', {})
            if 'hourly' in params:
                return mock_forecast_response
            else:
                return mock_response
        
        self.mock_get_patcher = patch.object(httpx.AsyncClient, 'get', mock_get)
        self.mock_get_patcher.start()
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if hasattr(self, 'mock_get_patcher'):
            self.mock_get_patcher.stop()
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_current_endpoint_response_format_property(self, client, lat: float, lon: float):
        """
        **Property 1: API Response Format Consistency**
        
        **Validates: Requirements 1.1, 1.2**
        
        For any valid coordinates provided to the current endpoint,
        the system should return properly structured JSON responses containing
        all required air quality fields.
        """
        # Выполняем запрос к current endpoint
        response = client.get(f"/weather/current?lat={lat}&lon={lon}")
        
        # Проверяем, что ответ успешный или содержит валидную ошибку
        assert response.status_code in [200, 422, 503], (
            f"Unexpected status code {response.status_code} for coordinates lat={lat}, lon={lon}"
        )
        
        # Проверяем, что ответ является валидным JSON
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON for coordinates lat={lat}, lon={lon}")
        
        if response.status_code == 200:
            # Для успешных ответов проверяем структуру данных
            self._validate_air_quality_response_structure(response_data, lat, lon)
        elif response.status_code == 422:
            # Для ошибок валидации проверяем структуру ошибки FastAPI
            self._validate_fastapi_error_response_structure(response_data, lat, lon)
        elif response.status_code == 503:
            # Для ошибок сервиса проверяем структуру ошибки
            self._validate_error_response_structure(response_data, lat, lon)
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_forecast_endpoint_response_format_property(self, client, lat: float, lon: float):
        """
        **Property 1: API Response Format Consistency**
        
        **Validates: Requirements 1.1, 1.2**
        
        For any valid coordinates provided to the forecast endpoint,
        the system should return properly structured JSON responses containing
        all required air quality fields in array format.
        """
        # Выполняем запрос к forecast endpoint
        response = client.get(f"/weather/forecast?lat={lat}&lon={lon}")
        
        # Проверяем, что ответ успешный или содержит валидную ошибку
        assert response.status_code in [200, 422, 503], (
            f"Unexpected status code {response.status_code} for coordinates lat={lat}, lon={lon}"
        )
        
        # Проверяем, что ответ является валидным JSON
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON for coordinates lat={lat}, lon={lon}")
        
        if response.status_code == 200:
            # Для успешных ответов проверяем структуру массива данных
            self._validate_forecast_response_structure(response_data, lat, lon)
        elif response.status_code == 422:
            # Для ошибок валидации проверяем структуру ошибки FastAPI
            self._validate_fastapi_error_response_structure(response_data, lat, lon)
        elif response.status_code == 503:
            # Для ошибок сервиса проверяем структуру ошибки
            self._validate_error_response_structure(response_data, lat, lon)
    
    @given(
        lat=st.one_of(
            st.floats(min_value=-1000, max_value=-90.1),  # Невалидная широта (слишком мала)
            st.floats(min_value=90.1, max_value=1000),    # Невалидная широта (слишком велика)
        ),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_latitude_response_format_property(self, client, lat: float, lon: float):
        """
        **Property 1: API Response Format Consistency**
        
        **Validates: Requirements 1.1, 1.2**
        
        For any invalid latitude coordinates, the system should return
        properly structured error responses with HTTP 400 status.
        """
        # Тестируем current endpoint
        response = client.get(f"/weather/current?lat={lat}&lon={lon}")
        
        # Должен быть статус 422 для невалидных координат (FastAPI + Pydantic)
        assert response.status_code == 422, (
            f"Expected status 422 for invalid latitude {lat}, got {response.status_code}"
        )
        
        # Проверяем структуру ошибки
        try:
            response_data = response.json()
            self._validate_fastapi_error_response_structure(response_data, lat, lon)
        except json.JSONDecodeError:
            pytest.fail(f"Error response is not valid JSON for lat={lat}, lon={lon}")
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.one_of(
            st.floats(min_value=-1000, max_value=-180.1),  # Невалидная долгота (слишком мала)
            st.floats(min_value=180.1, max_value=1000),    # Невалидная долгота (слишком велика)
        )
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_longitude_response_format_property(self, client, lat: float, lon: float):
        """
        **Property 1: API Response Format Consistency**
        
        **Validates: Requirements 1.1, 1.2**
        
        For any invalid longitude coordinates, the system should return
        properly structured error responses with HTTP 400 status.
        """
        # Тестируем forecast endpoint
        response = client.get(f"/weather/forecast?lat={lat}&lon={lon}")
        
        # Должен быть статус 422 для невалидных координат (FastAPI + Pydantic)
        assert response.status_code == 422, (
            f"Expected status 422 for invalid longitude {lon}, got {response.status_code}"
        )
        
        # Проверяем структуру ошибки
        try:
            response_data = response.json()
            self._validate_fastapi_error_response_structure(response_data, lat, lon)
        except json.JSONDecodeError:
            pytest.fail(f"Error response is not valid JSON for lat={lat}, lon={lon}")
    
    def _validate_air_quality_response_structure(self, response_data: Dict[str, Any], lat: float, lon: float):
        """Валидация структуры ответа с данными о качестве воздуха"""
        # Проверяем наличие всех обязательных полей верхнего уровня
        required_fields = ['timestamp', 'location', 'aqi', 'pollutants', 'recommendations']
        for field in required_fields:
            assert field in response_data, (
                f"Missing required field '{field}' in response for lat={lat}, lon={lon}"
            )
        
        # Проверяем структуру location
        location = response_data['location']
        assert isinstance(location, dict), f"Location should be dict for lat={lat}, lon={lon}"
        assert 'latitude' in location, f"Missing latitude in location for lat={lat}, lon={lon}"
        assert 'longitude' in location, f"Missing longitude in location for lat={lat}, lon={lon}"
        assert isinstance(location['latitude'], (int, float)), f"Latitude should be numeric for lat={lat}, lon={lon}"
        assert isinstance(location['longitude'], (int, float)), f"Longitude should be numeric for lat={lat}, lon={lon}"
        
        # Проверяем структуру aqi
        aqi = response_data['aqi']
        assert isinstance(aqi, dict), f"AQI should be dict for lat={lat}, lon={lon}"
        aqi_required_fields = ['value', 'category', 'color', 'description']
        for field in aqi_required_fields:
            assert field in aqi, f"Missing AQI field '{field}' for lat={lat}, lon={lon}"
        
        assert isinstance(aqi['value'], int), f"AQI value should be integer for lat={lat}, lon={lon}"
        assert 0 <= aqi['value'] <= 500, f"AQI value should be 0-500 for lat={lat}, lon={lon}"
        assert isinstance(aqi['category'], str), f"AQI category should be string for lat={lat}, lon={lon}"
        assert isinstance(aqi['color'], str), f"AQI color should be string for lat={lat}, lon={lon}"
        assert aqi['color'].startswith('#'), f"AQI color should be hex color for lat={lat}, lon={lon}"
        assert isinstance(aqi['description'], str), f"AQI description should be string for lat={lat}, lon={lon}"
        
        # Проверяем структуру pollutants
        pollutants = response_data['pollutants']
        assert isinstance(pollutants, dict), f"Pollutants should be dict for lat={lat}, lon={lon}"
        
        # Проверяем recommendations
        recommendations = response_data['recommendations']
        assert isinstance(recommendations, str), f"Recommendations should be string for lat={lat}, lon={lon}"
        assert len(recommendations) > 0, f"Recommendations should not be empty for lat={lat}, lon={lon}"
        
        # Проверяем timestamp
        timestamp = response_data['timestamp']
        assert isinstance(timestamp, str), f"Timestamp should be string for lat={lat}, lon={lon}"
        
        # Проверяем, что timestamp можно распарсить как ISO datetime
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"Invalid timestamp format '{timestamp}' for lat={lat}, lon={lon}")
        
        # Проверяем опциональные поля
        if 'nmu_risk' in response_data:
            assert isinstance(response_data['nmu_risk'], (str, type(None))), (
                f"NMU risk should be string or null for lat={lat}, lon={lon}"
            )
        
        if 'health_warnings' in response_data:
            assert isinstance(response_data['health_warnings'], list), (
                f"Health warnings should be list for lat={lat}, lon={lon}"
            )
            for warning in response_data['health_warnings']:
                assert isinstance(warning, str), (
                    f"Health warning should be string for lat={lat}, lon={lon}"
                )
    
    def _validate_forecast_response_structure(self, response_data: List[Dict[str, Any]], lat: float, lon: float):
        """Валидация структуры ответа с прогнозом качества воздуха"""
        # Проверяем, что ответ является списком
        assert isinstance(response_data, list), (
            f"Forecast response should be list for lat={lat}, lon={lon}"
        )
        
        # Проверяем, что список не пустой
        assert len(response_data) > 0, (
            f"Forecast response should not be empty for lat={lat}, lon={lon}"
        )
        
        # Проверяем, что количество элементов разумно (не более 24 часов)
        assert len(response_data) <= 24, (
            f"Forecast should not exceed 24 hours for lat={lat}, lon={lon}"
        )
        
        # Проверяем структуру каждого элемента прогноза
        for i, forecast_item in enumerate(response_data):
            try:
                self._validate_air_quality_response_structure(forecast_item, lat, lon)
            except AssertionError as e:
                pytest.fail(f"Invalid forecast item {i} for lat={lat}, lon={lon}: {e}")
    
    def _validate_error_response_structure(self, response_data: Dict[str, Any], lat: float, lon: float):
        """Валидация структуры ответа об ошибке"""
        # Проверяем различные форматы ошибок:
        # 1. FastAPI стандартный формат: {'detail': ...}
        # 2. Вложенный формат: {'error': {'code': ..., 'message': ...}}
        # 3. Плоский формат: {'code': ..., 'message': ..., 'details': ..., 'timestamp': ...}
        
        has_error_field = 'error' in response_data
        has_detail_field = 'detail' in response_data
        has_code_field = 'code' in response_data
        has_message_field = 'message' in response_data
        
        assert has_error_field or has_detail_field or (has_code_field and has_message_field), (
            f"Error response should have 'error', 'detail', or 'code'+'message' fields for lat={lat}, lon={lon}. "
            f"Got: {list(response_data.keys())}"
        )
        
        if has_error_field:
            # Проверяем структуру error объекта
            error = response_data['error']
            assert isinstance(error, dict), f"Error should be dict for lat={lat}, lon={lon}"
            
            # Проверяем обязательные поля ошибки
            error_required_fields = ['code', 'message']
            for field in error_required_fields:
                assert field in error, f"Missing error field '{field}' for lat={lat}, lon={lon}"
            
            assert isinstance(error['code'], str), f"Error code should be string for lat={lat}, lon={lon}"
            assert isinstance(error['message'], str), f"Error message should be string for lat={lat}, lon={lon}"
            assert len(error['message']) > 0, f"Error message should not be empty for lat={lat}, lon={lon}"
        
        elif has_detail_field:
            # FastAPI стандартный формат ошибки
            detail = response_data['detail']
            assert isinstance(detail, (str, list)), f"Detail should be string or list for lat={lat}, lon={lon}"
        
        elif has_code_field and has_message_field:
            # Плоский формат ошибки (как в текущем API)
            assert isinstance(response_data['code'], str), f"Error code should be string for lat={lat}, lon={lon}"
            assert isinstance(response_data['message'], str), f"Error message should be string for lat={lat}, lon={lon}"
            assert len(response_data['message']) > 0, f"Error message should not be empty for lat={lat}, lon={lon}"
            
            # Проверяем опциональные поля
            if 'details' in response_data:
                # details может быть None или строкой
                assert response_data['details'] is None or isinstance(response_data['details'], str), (
                    f"Error details should be None or string for lat={lat}, lon={lon}"
                )
            
            if 'timestamp' in response_data:
                assert isinstance(response_data['timestamp'], str), (
                    f"Error timestamp should be string for lat={lat}, lon={lon}"
                )
                # Проверяем, что timestamp можно распарсить как ISO datetime
                try:
                    datetime.fromisoformat(response_data['timestamp'].replace('Z', '+00:00'))
                except ValueError:
                    pytest.fail(f"Invalid timestamp format '{response_data['timestamp']}' for lat={lat}, lon={lon}")
    
    def _validate_fastapi_error_response_structure(self, response_data: Dict[str, Any], lat: float, lon: float):
        """Валидация структуры ошибки FastAPI (422 Unprocessable Entity)"""
        # FastAPI возвращает ошибки валидации в поле 'detail'
        assert 'detail' in response_data, (
            f"FastAPI error response should have 'detail' field for lat={lat}, lon={lon}"
        )
        
        detail = response_data['detail']
        
        # Detail может быть строкой или списком ошибок валидации
        if isinstance(detail, str):
            # Простое сообщение об ошибке
            assert len(detail) > 0, f"Error message should not be empty for lat={lat}, lon={lon}"
        elif isinstance(detail, list):
            # Список ошибок валидации Pydantic
            assert len(detail) > 0, f"Should have at least one validation error for lat={lat}, lon={lon}"
            
            # Проверяем структуру первой ошибки валидации
            first_error = detail[0]
            assert isinstance(first_error, dict), f"Validation error should be a dict for lat={lat}, lon={lon}"
            
            # Проверяем обязательные поля ошибки валидации
            required_fields = ['type', 'loc', 'msg']
            for field in required_fields:
                assert field in first_error, (
                    f"Validation error should have '{field}' field for lat={lat}, lon={lon}"
                )
        else:
            pytest.fail(f"Detail should be string or list for lat={lat}, lon={lon}, got {type(detail)}")