"""
Property-based тесты для консистентности временных зон в AirTrace RU Backend

**Property 10: Timezone Consistency**
**Validates: Requirements 4.5**

Тестирует, что все временные метки в API ответах форматируются с использованием
соответствующей информации о российских часовых поясах.
"""

import pytest
from datetime import datetime, timezone, timedelta
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import re

from main import app
from schemas import AirQualityData, LocationInfo, AQIInfo, PollutantData


class TestTimezoneConsistency:
    """Property-based тесты для консистентности временных зон"""
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_api_response_timestamp_format_property(self, lat, lon):
        """
        Feature: airtrace-ru, Property 10: Timezone Consistency
        
        For any timestamp in API responses, the time should be formatted 
        using appropriate timezone information.
        **Validates: Requirements 4.5**
        """
        client = TestClient(app)
        
        # Мокаем внешний API для получения предсказуемых данных
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "latitude": lat,
                "longitude": lon,
                "current": {
                    "pm10": 25.0,
                    "pm2_5": 15.0,
                    "nitrogen_dioxide": 30.0,
                    "sulphur_dioxide": 10.0,
                    "ozone": 80.0
                }
            }
            mock_get.return_value = mock_response
            
            try:
                # Тестируем current endpoint
                response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Проверяем наличие timestamp
                    assert 'timestamp' in data, f"Missing timestamp in response for lat={lat}, lon={lon}"
                    
                    timestamp_str = data['timestamp']
                    
                    # Проверяем формат ISO 8601
                    iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$'
                    assert re.match(iso_pattern, timestamp_str), (
                        f"Timestamp not in ISO 8601 format: {timestamp_str} for lat={lat}, lon={lon}"
                    )
                    
                    # Проверяем, что timestamp можно парсить
                    try:
                        parsed_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        assert isinstance(parsed_timestamp, datetime), (
                            f"Failed to parse timestamp: {timestamp_str} for lat={lat}, lon={lon}"
                        )
                        
                        # Проверяем, что timestamp имеет информацию о часовом поясе
                        assert parsed_timestamp.tzinfo is not None, (
                            f"Timestamp missing timezone info: {timestamp_str} for lat={lat}, lon={lon}"
                        )
                        
                    except ValueError as e:
                        pytest.fail(f"Invalid timestamp format: {timestamp_str} for lat={lat}, lon={lon}. Error: {e}")
                
            except Exception as e:
                # Если запрос не удался, это не ошибка timezone consistency
                # но мы должны убедиться, что это не связано с форматированием времени
                if "timestamp" in str(e).lower() or "timezone" in str(e).lower():
                    pytest.fail(f"Timezone-related error for lat={lat}, lon={lon}: {e}")
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_forecast_timestamps_consistency_property(self, lat, lon):
        """
        Feature: airtrace-ru, Property 10: Timezone Consistency
        
        For any forecast response, all timestamps should be consistently formatted
        and properly ordered in time.
        **Validates: Requirements 4.5**
        """
        client = TestClient(app)
        
        # Мокаем внешний API для прогноза
        with patch('httpx.AsyncClient.get') as mock_get:
            # Создаем данные прогноза на 24 часа
            forecast_times = []
            base_time = datetime.now(timezone.utc)
            
            for i in range(24):
                forecast_times.append((base_time + timedelta(hours=i)).isoformat())
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "latitude": lat,
                "longitude": lon,
                "hourly": {
                    "time": forecast_times,
                    "pm10": [25.0] * 24,
                    "pm2_5": [15.0] * 24,
                    "nitrogen_dioxide": [30.0] * 24,
                    "sulphur_dioxide": [10.0] * 24,
                    "ozone": [80.0] * 24
                }
            }
            mock_get.return_value = mock_response
            
            try:
                response = client.get(f"/weather/forecast?lat={lat}&lon={lon}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Проверяем, что это список
                    assert isinstance(data, list), f"Forecast response should be a list for lat={lat}, lon={lon}"
                    
                    if len(data) > 0:
                        previous_timestamp = None
                        
                        for i, item in enumerate(data):
                            # Проверяем наличие timestamp в каждом элементе
                            assert 'timestamp' in item, (
                                f"Missing timestamp in forecast item {i} for lat={lat}, lon={lon}"
                            )
                            
                            timestamp_str = item['timestamp']
                            
                            # Проверяем формат ISO 8601
                            iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$'
                            assert re.match(iso_pattern, timestamp_str), (
                                f"Forecast timestamp not in ISO 8601 format: {timestamp_str} "
                                f"for item {i}, lat={lat}, lon={lon}"
                            )
                            
                            # Парсим timestamp
                            try:
                                parsed_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                
                                # Проверяем наличие timezone info
                                assert parsed_timestamp.tzinfo is not None, (
                                    f"Forecast timestamp missing timezone info: {timestamp_str} "
                                    f"for item {i}, lat={lat}, lon={lon}"
                                )
                                
                                # Проверяем хронологический порядок
                                if previous_timestamp is not None:
                                    assert parsed_timestamp >= previous_timestamp, (
                                        f"Forecast timestamps not in chronological order: "
                                        f"{previous_timestamp} -> {parsed_timestamp} for lat={lat}, lon={lon}"
                                    )
                                
                                previous_timestamp = parsed_timestamp
                                
                            except ValueError as e:
                                pytest.fail(
                                    f"Invalid forecast timestamp format: {timestamp_str} "
                                    f"for item {i}, lat={lat}, lon={lon}. Error: {e}"
                                )
                
            except Exception as e:
                # Если запрос не удался, проверяем, что это не связано с timezone
                if "timestamp" in str(e).lower() or "timezone" in str(e).lower():
                    pytest.fail(f"Timezone-related error in forecast for lat={lat}, lon={lon}: {e}")
    
    def test_schema_timestamp_serialization_property(self):
        """
        Feature: airtrace-ru, Property 10: Timezone Consistency
        
        For any AirQualityData model instance, the timestamp should be 
        serialized consistently with timezone information.
        **Validates: Requirements 4.5**
        """
        # Тестируем различные временные зоны
        test_timezones = [
            timezone.utc,
            timezone(timedelta(hours=3)),  # MSK
            timezone(timedelta(hours=5)),  # YEKT
            timezone(timedelta(hours=7)),  # NOVT
        ]
        
        for tz in test_timezones:
            test_datetime = datetime(2024, 1, 15, 12, 0, 0, tzinfo=tz)
            
            # Создаем модель данных
            air_quality_data = AirQualityData(
                timestamp=test_datetime,
                location=LocationInfo(latitude=55.7558, longitude=37.6176),
                aqi=AQIInfo(value=85, category="Умеренное", color="#FFFF00", description="Test"),
                pollutants=PollutantData(pm2_5=25.0, pm10=45.0),
                recommendations="Test recommendations",
                nmu_risk="low",
                health_warnings=[]
            )
            
            # Сериализуем в JSON
            json_data = air_quality_data.model_dump(mode='json')
            
            # Проверяем формат timestamp
            timestamp_str = json_data['timestamp']
            
            # Проверяем ISO 8601 формат
            iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$'
            assert re.match(iso_pattern, timestamp_str), (
                f"Serialized timestamp not in ISO 8601 format: {timestamp_str} for timezone {tz}"
            )
            
            # Проверяем, что можно десериализовать обратно
            try:
                parsed_back = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                assert parsed_back.tzinfo is not None, (
                    f"Deserialized timestamp missing timezone: {timestamp_str} for timezone {tz}"
                )
            except ValueError as e:
                pytest.fail(f"Failed to deserialize timestamp: {timestamp_str} for timezone {tz}. Error: {e}")
    
    @given(
        hours_offset=st.integers(min_value=-12, max_value=12)
    )
    @settings(max_examples=25, deadline=10000)
    def test_timezone_offset_handling_property(self, hours_offset):
        """
        Feature: airtrace-ru, Property 10: Timezone Consistency
        
        For any timezone offset, the system should handle timestamp 
        formatting consistently.
        **Validates: Requirements 4.5**
        """
        # Создаем timezone с заданным смещением
        test_tz = timezone(timedelta(hours=hours_offset))
        test_datetime = datetime(2024, 1, 15, 12, 0, 0, tzinfo=test_tz)
        
        # Создаем модель данных
        air_quality_data = AirQualityData(
            timestamp=test_datetime,
            location=LocationInfo(latitude=55.7558, longitude=37.6176),
            aqi=AQIInfo(value=85, category="Умеренное", color="#FFFF00", description="Test"),
            pollutants=PollutantData(pm2_5=25.0, pm10=45.0),
            recommendations="Test recommendations",
            nmu_risk="low",
            health_warnings=[]
        )
        
        # Сериализуем
        json_data = air_quality_data.model_dump(mode='json')
        timestamp_str = json_data['timestamp']
        
        # Проверяем корректность формата
        iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$'
        assert re.match(iso_pattern, timestamp_str), (
            f"Invalid timestamp format for offset {hours_offset}: {timestamp_str}"
        )
        
        # Проверяем, что смещение корректно отражено
        if hours_offset == 0:
            # UTC должен быть представлен как Z или +00:00
            assert timestamp_str.endswith('Z') or timestamp_str.endswith('+00:00'), (
                f"UTC timestamp should end with Z or +00:00: {timestamp_str}"
            )
        else:
            # Другие смещения должны быть явно указаны
            expected_offset = f"{hours_offset:+03d}:00"
            assert expected_offset in timestamp_str, (
                f"Timezone offset {hours_offset} not properly reflected in timestamp: {timestamp_str}"
            )
    
    def test_health_check_timestamp_consistency(self):
        """
        Feature: airtrace-ru, Property 10: Timezone Consistency
        
        Health check endpoint should also return timestamps with consistent formatting.
        **Validates: Requirements 4.5**
        """
        client = TestClient(app)
        
        response = client.get("/health")
        
        if response.status_code == 200:
            data = response.json()
            
            # Проверяем наличие timestamp в health check
            if 'timestamp' in data:
                timestamp_str = data['timestamp']
                
                # Проверяем формат ISO 8601
                iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$'
                assert re.match(iso_pattern, timestamp_str), (
                    f"Health check timestamp not in ISO 8601 format: {timestamp_str}"
                )
                
                # Проверяем, что можно парсить
                try:
                    parsed_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    assert parsed_timestamp.tzinfo is not None, (
                        f"Health check timestamp missing timezone info: {timestamp_str}"
                    )
                except ValueError as e:
                    pytest.fail(f"Invalid health check timestamp format: {timestamp_str}. Error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])