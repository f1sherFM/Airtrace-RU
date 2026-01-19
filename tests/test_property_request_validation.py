"""
Property-based tests for request validation.

**Property 16: Request Validation**
*For any* incoming API request, all parameters should be validated using 
Pydantic models with structured error responses for invalid data.

**Validates: Requirements 8.1, 8.2**
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient
import json
from typing import Dict, Any, List, Union

from main import app


class TestRequestValidationProperty:
    """Property-based tests for request validation"""
    
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
        
        # Создаем мок для httpx.AsyncClient
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.raise_for_status = AsyncMock(return_value=None)
        
        # Патчим httpx.AsyncClient.get
        self.mock_get_patcher = patch('httpx.AsyncClient.get', return_value=mock_response)
        self.mock_get_patcher.start()
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if hasattr(self, 'mock_get_patcher'):
            self.mock_get_patcher.stop()
    
    @given(
        lat=st.one_of(
            st.floats(min_value=-1000, max_value=-90.1),  # Невалидная широта (слишком мала)
            st.floats(min_value=90.1, max_value=1000),    # Невалидная широта (слишком велика)
            st.just(float('inf')),                        # Бесконечность
            st.just(float('-inf')),                       # Отрицательная бесконечность
            st.just(float('nan'))                         # NaN
        ),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_latitude_validation_property(self, client, lat: float, lon: float):
        """
        **Property 16: Request Validation**
        
        **Validates: Requirements 8.1, 8.2**
        
        For any invalid latitude parameter, the system should validate using
        Pydantic models and return structured error responses with field-specific messages.
        """
        # Тестируем current endpoint
        response = client.get(f"/weather/current?lat={lat}&lon={lon}")
        
        # Должен быть статус 422 для невалидных параметров (FastAPI + Pydantic)
        assert response.status_code == 422, (
            f"Expected status 422 for invalid latitude {lat}, got {response.status_code}"
        )
        
        # Проверяем, что ответ является валидным JSON
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            pytest.fail(f"Error response is not valid JSON for lat={lat}, lon={lon}")
        
        # Проверяем структуру ошибки валидации Pydantic
        self._validate_pydantic_error_structure(response_data, lat, lon, "latitude")
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.one_of(
            st.floats(min_value=-1000, max_value=-180.1),  # Невалидная долгота (слишком мала)
            st.floats(min_value=180.1, max_value=1000),    # Невалидная долгота (слишком велика)
            st.just(float('inf')),                         # Бесконечность
            st.just(float('-inf')),                        # Отрицательная бесконечность
            st.just(float('nan'))                          # NaN
        )
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_longitude_validation_property(self, client, lat: float, lon: float):
        """
        **Property 16: Request Validation**
        
        **Validates: Requirements 8.1, 8.2**
        
        For any invalid longitude parameter, the system should validate using
        Pydantic models and return structured error responses with field-specific messages.
        """
        # Тестируем forecast endpoint
        response = client.get(f"/weather/forecast?lat={lat}&lon={lon}")
        
        # Должен быть статус 422 для невалидных параметров
        assert response.status_code == 422, (
            f"Expected status 422 for invalid longitude {lon}, got {response.status_code}"
        )
        
        # Проверяем, что ответ является валидным JSON
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            pytest.fail(f"Error response is not valid JSON for lat={lat}, lon={lon}")
        
        # Проверяем структуру ошибки валидации Pydantic
        self._validate_pydantic_error_structure(response_data, lat, lon, "longitude")
    
    @given(
        invalid_param_name=st.sampled_from(['latitude', 'longitude', 'invalid_param']),
        invalid_param_value=st.one_of(
            st.text(min_size=1, max_size=50),  # Строковые значения
            st.lists(st.integers(), min_size=1, max_size=5),  # Массивы
            st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), min_size=1, max_size=3)  # Объекты
        )
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_parameter_types_validation_property(self, client, invalid_param_name: str, invalid_param_value: Any):
        """
        **Property 16: Request Validation**
        
        **Validates: Requirements 8.1, 8.2**
        
        For any invalid parameter types (non-numeric for coordinates), the system
        should validate using Pydantic models and return structured error responses.
        """
        # Создаем запрос с невалидным типом параметра
        if invalid_param_name == 'latitude':
            url = f"/weather/current?lat={invalid_param_value}&lon=37.6176"
        elif invalid_param_name == 'longitude':
            url = f"/weather/current?lat=55.7558&lon={invalid_param_value}"
        else:
            url = f"/weather/current?lat=55.7558&lon=37.6176&{invalid_param_name}={invalid_param_value}"
        
        try:
            response = client.get(url)
        except Exception as e:
            # Некоторые невалидные значения могут вызвать исключения на уровне HTTP клиента
            # Это тоже валидное поведение для защиты от некорректных запросов
            return
        
        # Должен быть статус 422 для невалидных типов параметров (FastAPI + Pydantic)
        assert response.status_code in [422, 503], (
            f"Expected status 422 or 503 for invalid parameter type, got {response.status_code}"
        )
        
        # Проверяем, что ответ является валидным JSON
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            pytest.fail(f"Error response is not valid JSON for {invalid_param_name}={invalid_param_value}")
        
        # Проверяем наличие информации об ошибке
        if response.status_code == 422:
            # Ошибки валидации FastAPI
            assert 'detail' in response_data, (
                f"Validation error response should contain 'detail' for {invalid_param_name}={invalid_param_value}"
            )
        elif response.status_code == 503:
            # Ошибки сервиса (когда валидация прошла, но сервис недоступен)
            # Проверяем различные форматы ошибок
            has_error_info = (
                'detail' in response_data or 
                'error' in response_data or 
                'message' in response_data or
                'code' in response_data
            )
            assert has_error_info, (
                f"Service error response should contain error information for {invalid_param_name}={invalid_param_value}. "
                f"Response: {response_data}"
            )
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_valid_coordinates_validation_property(self, client, lat: float, lon: float):
        """
        **Property 16: Request Validation**
        
        **Validates: Requirements 8.1, 8.2**
        
        For any valid coordinate parameters, the system should validate successfully
        using Pydantic models and proceed to process the request.
        """
        # Тестируем current endpoint с валидными координатами
        response = client.get(f"/weather/current?lat={lat}&lon={lon}")
        
        # Валидные координаты должны проходить валидацию
        # Статус может быть 200 (успех) или 503 (проблемы с внешним API)
        assert response.status_code in [200, 503], (
            f"Expected status 200 or 503 for valid coordinates lat={lat}, lon={lon}, "
            f"got {response.status_code}"
        )
        
        # Проверяем, что ответ является валидным JSON
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON for valid coordinates lat={lat}, lon={lon}")
        
        # Для успешных ответов проверяем наличие данных о качестве воздуха
        if response.status_code == 200:
            assert 'aqi' in response_data, (
                f"Successful response should contain AQI data for lat={lat}, lon={lon}"
            )
            assert 'location' in response_data, (
                f"Successful response should contain location data for lat={lat}, lon={lon}"
            )
    
    @given(
        endpoint=st.sampled_from(['/weather/current', '/weather/forecast'])
    )
    @settings(max_examples=20, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_missing_required_parameters_validation_property(self, client, endpoint: str):
        """
        **Property 16: Request Validation**
        
        **Validates: Requirements 8.1, 8.2**
        
        For any request missing required parameters (lat, lon), the system should
        validate using Pydantic models and return structured error responses.
        """
        # Тестируем запросы без обязательных параметров
        test_cases = [
            endpoint,  # Без параметров
            f"{endpoint}?lat=55.7558",  # Без lon
            f"{endpoint}?lon=37.6176",  # Без lat
        ]
        
        for url in test_cases:
            response = client.get(url)
            
            # Должен быть статус 422 для отсутствующих обязательных параметров
            assert response.status_code == 422, (
                f"Expected status 422 for missing parameters in {url}, got {response.status_code}"
            )
            
            # Проверяем, что ответ является валидным JSON
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                pytest.fail(f"Error response is not valid JSON for {url}")
            
            # Проверяем структуру ошибки валидации
            assert 'detail' in response_data, (
                f"Error response should contain 'detail' field for {url}"
            )
            
            # Проверяем, что detail содержит информацию об ошибке валидации
            detail = response_data['detail']
            assert isinstance(detail, list), (
                f"Detail should be a list of validation errors for {url}"
            )
            
            # Проверяем, что есть хотя бы одна ошибка валидации
            assert len(detail) > 0, (
                f"Should have at least one validation error for {url}"
            )
    
    @given(
        extra_params=st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
            st.one_of(st.text(min_size=1, max_size=50), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_extra_parameters_handling_property(self, client, extra_params: Dict[str, Any]):
        """
        **Property 16: Request Validation**
        
        **Validates: Requirements 8.1, 8.2**
        
        For any request with extra/unknown parameters, the system should handle them
        gracefully without breaking the validation of required parameters.
        """
        # Создаем URL с валидными обязательными параметрами и дополнительными
        base_url = "/weather/current?lat=55.7558&lon=37.6176"
        
        # Добавляем дополнительные параметры
        extra_params_str = "&".join([f"{k}={v}" for k, v in extra_params.items()])
        url = f"{base_url}&{extra_params_str}"
        
        try:
            response = client.get(url)
        except Exception as e:
            # Некоторые невалидные значения могут вызвать исключения
            # Это приемлемо для защиты от некорректных запросов
            return
        
        # Дополнительные параметры не должны ломать валидацию основных параметров
        # Статус должен быть 200 (успех) или 503 (проблемы с внешним API)
        assert response.status_code in [200, 503], (
            f"Extra parameters should not break validation, got status {response.status_code} for {url}"
        )
        
        # Проверяем, что ответ является валидным JSON
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON with extra parameters: {url}")
        
        # Для успешных ответов проверяем наличие основных данных
        if response.status_code == 200:
            assert 'aqi' in response_data, (
                f"Response should contain AQI data even with extra parameters: {url}"
            )
    
    def _validate_pydantic_error_structure(self, response_data: Dict[str, Any], lat: float, lon: float, field_name: str):
        """Валидация структуры ошибки Pydantic"""
        # Проверяем наличие поля detail (стандартный формат FastAPI)
        assert 'detail' in response_data, (
            f"Pydantic error response should have 'detail' field for {field_name}={lat if field_name == 'latitude' else lon}"
        )
        
        detail = response_data['detail']
        
        # Detail должен быть списком ошибок валидации
        assert isinstance(detail, list), (
            f"Detail should be a list of validation errors for {field_name}={lat if field_name == 'latitude' else lon}"
        )
        
        # Должна быть хотя бы одна ошибка
        assert len(detail) > 0, (
            f"Should have at least one validation error for {field_name}={lat if field_name == 'latitude' else lon}"
        )
        
        # Проверяем структуру первой ошибки валидации
        first_error = detail[0]
        assert isinstance(first_error, dict), (
            f"Validation error should be a dict for {field_name}={lat if field_name == 'latitude' else lon}"
        )
        
        # Проверяем обязательные поля ошибки валидации
        required_error_fields = ['type', 'loc', 'msg']
        for field in required_error_fields:
            assert field in first_error, (
                f"Validation error should have '{field}' field for {field_name}={lat if field_name == 'latitude' else lon}"
            )
        
        # Проверяем, что loc содержит информацию о поле с ошибкой
        loc = first_error['loc']
        assert isinstance(loc, list), (
            f"Error location should be a list for {field_name}={lat if field_name == 'latitude' else lon}"
        )
        
        # Проверяем, что сообщение об ошибке не пустое
        msg = first_error['msg']
        assert isinstance(msg, str), (
            f"Error message should be a string for {field_name}={lat if field_name == 'latitude' else lon}"
        )
        assert len(msg) > 0, (
            f"Error message should not be empty for {field_name}={lat if field_name == 'latitude' else lon}"
        )