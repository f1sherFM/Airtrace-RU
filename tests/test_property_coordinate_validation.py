"""
Property-based tests for coordinate validation.

**Property 2: Coordinate Validation**
**Validates: Requirements 1.4, 1.5**

Тестирует валидацию координат с использованием property-based testing
для проверки корректности обработки всех возможных значений координат.
"""

import pytest
from hypothesis import given, strategies as st, settings
from fastapi.testclient import TestClient
from pydantic import ValidationError

from schemas import CoordinatesRequest
from main import app


class TestCoordinateValidationProperty:
    """Property-based тесты для валидации координат"""
    
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
    )
    def test_valid_coordinates_property(self, lat: float, lon: float):
        """
        **Property 2: Coordinate Validation**
        **Validates: Requirements 1.4, 1.5**
        
        For any valid coordinate values within geographic ranges 
        (latitude: -90 to 90, longitude: -180 to 180), 
        the CoordinatesRequest model should validate successfully.
        """
        # Создание модели с валидными координатами должно проходить без ошибок
        coords = CoordinatesRequest(lat=lat, lon=lon)
        assert coords.lat == lat
        assert coords.lon == lon
        assert -90.0 <= coords.lat <= 90.0
        assert -180.0 <= coords.lon <= 180.0
    
    @given(
        lat=st.one_of(
            st.floats(min_value=-1000.0, max_value=-90.01, allow_nan=False, allow_infinity=False),
            st.floats(min_value=90.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
        ),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
    )
    def test_invalid_latitude_property(self, lat: float, lon: float):
        """
        **Property 2: Coordinate Validation**
        **Validates: Requirements 1.4, 1.5**
        
        For any latitude values outside valid geographic range (-90 to 90),
        the CoordinatesRequest model should raise ValidationError.
        """
        with pytest.raises(ValidationError) as exc_info:
            CoordinatesRequest(lat=lat, lon=lon)
        
        # Проверяем, что ошибка связана с полем lat
        errors = exc_info.value.errors()
        lat_errors = [error for error in errors if 'lat' in str(error.get('loc', []))]
        assert len(lat_errors) > 0, f"Expected latitude validation error for lat={lat}"
    
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.one_of(
            st.floats(min_value=-1000.0, max_value=-180.01, allow_nan=False, allow_infinity=False),
            st.floats(min_value=180.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
        )
    )
    def test_invalid_longitude_property(self, lat: float, lon: float):
        """
        **Property 2: Coordinate Validation**
        **Validates: Requirements 1.4, 1.5**
        
        For any longitude values outside valid geographic range (-180 to 180),
        the CoordinatesRequest model should raise ValidationError.
        """
        with pytest.raises(ValidationError) as exc_info:
            CoordinatesRequest(lat=lat, lon=lon)
        
        # Проверяем, что ошибка связана с полем lon
        errors = exc_info.value.errors()
        lon_errors = [error for error in errors if 'lon' in str(error.get('loc', []))]
        assert len(lon_errors) > 0, f"Expected longitude validation error for lon={lon}"
    
    @given(
        lat=st.one_of(
            st.floats(min_value=-1000.0, max_value=-90.01, allow_nan=False, allow_infinity=False),
            st.floats(min_value=90.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
        ),
        lon=st.one_of(
            st.floats(min_value=-1000.0, max_value=-180.01, allow_nan=False, allow_infinity=False),
            st.floats(min_value=180.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
        )
    )
    def test_both_coordinates_invalid_property(self, lat: float, lon: float):
        """
        **Property 2: Coordinate Validation**
        **Validates: Requirements 1.4, 1.5**
        
        For any coordinate values where both latitude and longitude are outside 
        valid geographic ranges, the CoordinatesRequest model should raise ValidationError
        with errors for both fields.
        """
        with pytest.raises(ValidationError) as exc_info:
            CoordinatesRequest(lat=lat, lon=lon)
        
        # Проверяем, что есть ошибки для обоих полей
        errors = exc_info.value.errors()
        lat_errors = [error for error in errors if 'lat' in str(error.get('loc', []))]
        lon_errors = [error for error in errors if 'lon' in str(error.get('loc', []))]
        
        assert len(lat_errors) > 0, f"Expected latitude validation error for lat={lat}"
        assert len(lon_errors) > 0, f"Expected longitude validation error for lon={lon}"


class TestCoordinateValidationAPIProperty:
    """Property-based тесты для валидации координат через API"""
    
    @settings(deadline=5000, max_examples=20)  # Увеличиваем deadline и уменьшаем количество примеров
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
    )
    def test_api_valid_coordinates_property(self, lat: float, lon: float):
        """
        **Property 2: Coordinate Validation**
        **Validates: Requirements 1.4, 1.5**
        
        For any valid coordinates provided to API endpoints,
        the system should not return HTTP 422 validation errors.
        """
        client = TestClient(app)
        
        # Тестируем current endpoint
        response = client.get(f"/weather/current?lat={lat}&lon={lon}")
        assert response.status_code != 422, f"Unexpected validation error for valid coordinates lat={lat}, lon={lon}"
        
        # Тестируем forecast endpoint
        response = client.get(f"/weather/forecast?lat={lat}&lon={lon}")
        assert response.status_code != 422, f"Unexpected validation error for valid coordinates lat={lat}, lon={lon}"
    
    @settings(max_examples=10)  # Уменьшаем количество примеров для быстрых тестов
    @given(
        lat=st.one_of(
            st.floats(min_value=-1000.0, max_value=-90.01, allow_nan=False, allow_infinity=False),
            st.floats(min_value=90.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
        ),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
    )
    def test_api_invalid_latitude_property(self, lat: float, lon: float):
        """
        **Property 2: Coordinate Validation**
        **Validates: Requirements 1.4, 1.5**
        
        For any latitude values outside valid geographic range,
        the API should return HTTP 422 error with descriptive validation messages.
        """
        client = TestClient(app)
        
        # Тестируем current endpoint
        response = client.get(f"/weather/current?lat={lat}&lon={lon}")
        assert response.status_code == 422, f"Expected validation error for invalid latitude lat={lat}"
        
        # Проверяем, что в ответе есть информация об ошибке валидации
        error_data = response.json()
        assert "detail" in error_data
        
        # Тестируем forecast endpoint
        response = client.get(f"/weather/forecast?lat={lat}&lon={lon}")
        assert response.status_code == 422, f"Expected validation error for invalid latitude lat={lat}"
    
    @settings(max_examples=10)  # Уменьшаем количество примеров для быстрых тестов
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.one_of(
            st.floats(min_value=-1000.0, max_value=-180.01, allow_nan=False, allow_infinity=False),
            st.floats(min_value=180.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
        )
    )
    def test_api_invalid_longitude_property(self, lat: float, lon: float):
        """
        **Property 2: Coordinate Validation**
        **Validates: Requirements 1.4, 1.5**
        
        For any longitude values outside valid geographic range,
        the API should return HTTP 422 error with descriptive validation messages.
        """
        client = TestClient(app)
        
        # Тестируем current endpoint
        response = client.get(f"/weather/current?lat={lat}&lon={lon}")
        assert response.status_code == 422, f"Expected validation error for invalid longitude lon={lon}"
        
        # Проверяем, что в ответе есть информация об ошибке валидации
        error_data = response.json()
        assert "detail" in error_data
        
        # Тестируем forecast endpoint
        response = client.get(f"/weather/forecast?lat={lat}&lon={lon}")
        assert response.status_code == 422, f"Expected validation error for invalid longitude lon={lon}"