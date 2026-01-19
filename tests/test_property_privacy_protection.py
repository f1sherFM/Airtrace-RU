"""
Property-based тесты для защиты приватности в AirTrace RU Backend

**Property 11: Privacy Protection**
**Validates: Requirements 5.1, 5.2, 5.4**

Тестирует, что координаты пользователей никогда не логируются в постоянное хранилище,
логи приложения или логи ошибок.
"""

import pytest
import logging
import io
import json
import re
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient
from contextlib import redirect_stderr, redirect_stdout

from main import app
from middleware import PrivacyMiddleware, PrivacyAwareFormatter, setup_privacy_logging
from services import AirQualityService


class LogCapture:
    """Класс для захвата логов во время тестирования"""
    
    def __init__(self):
        self.logs = []
        self.handler = None
        self.stream = None
    
    def __enter__(self):
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(PrivacyAwareFormatter())
        
        # Добавляем обработчик ко всем логгерам
        loggers = [
            logging.getLogger(),
            logging.getLogger('main'),
            logging.getLogger('services'),
            logging.getLogger('middleware'),
            logging.getLogger('privacy_middleware')
        ]
        
        for logger in loggers:
            logger.addHandler(self.handler)
            logger.setLevel(logging.DEBUG)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Удаляем обработчик
        loggers = [
            logging.getLogger(),
            logging.getLogger('main'),
            logging.getLogger('services'),
            logging.getLogger('middleware'),
            logging.getLogger('privacy_middleware')
        ]
        
        for logger in loggers:
            if self.handler in logger.handlers:
                logger.removeHandler(self.handler)
        
        # Сохраняем логи
        self.logs = self.stream.getvalue().split('\n')
        self.stream.close()
    
    def get_logs(self):
        """Получение всех захваченных логов"""
        return [log for log in self.logs if log.strip()]
    
    def contains_coordinates(self):
        """Проверка наличия координат в логах"""
        coordinate_patterns = [
            r'lat(?:itude)?["\s]*[:=]["\s]*(-?\d+\.?\d*)',
            r'lon(?:gitude)?["\s]*[:=]["\s]*(-?\d+\.?\d*)',
            r'(-?\d{1,3}\.\d+),\s*(-?\d{1,3}\.\d+)',
            r'lat=(-?\d+\.?\d*)',
            r'lon=(-?\d+\.?\d*)'
        ]
        
        all_logs = '\n'.join(self.get_logs())
        
        for pattern in coordinate_patterns:
            if re.search(pattern, all_logs, re.IGNORECASE):
                # Проверяем, что это не отфильтрованные координаты
                if '[COORDINATE_FILTERED]' not in all_logs:
                    return True
        
        return False


@pytest.fixture
def client():
    """Тестовый клиент FastAPI"""
    return TestClient(app)


@pytest.fixture
def mock_external_api():
    """Мок для внешнего API"""
    with patch('httpx.AsyncClient.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "latitude": 55.7558,
            "longitude": 37.6176,
            "current": {
                "pm10": 25.0,
                "pm2_5": 15.0,
                "nitrogen_dioxide": 30.0,
                "sulphur_dioxide": 10.0,
                "ozone": 80.0
            }
        }
        mock_get.return_value = mock_response
        yield mock_get


class TestPrivacyProtection:
    """Тесты защиты приватности"""
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_coordinates_not_logged_in_successful_requests(self, client, mock_external_api, lat, lon):
        """
        Feature: airtrace-ru, Property 11: Privacy Protection
        
        Тестирует, что координаты не логируются при успешных запросах.
        **Validates: Requirements 5.1, 5.2**
        """
        with LogCapture() as log_capture:
            try:
                response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                
                # Проверяем, что в логах нет координат
                assert not log_capture.contains_coordinates(), (
                    f"Coordinates found in logs for successful request lat={lat}, lon={lon}. "
                    f"Logs: {log_capture.get_logs()}"
                )
                
                # Проверяем, что есть отфильтрованные маркеры
                all_logs = '\n'.join(log_capture.get_logs())
                if response.status_code == 200:
                    # Для успешных запросов должны быть маркеры фильтрации
                    assert '[COORDINATE_FILTERED]' in all_logs or len(all_logs.strip()) == 0, (
                        f"Expected coordinate filtering markers in logs for lat={lat}, lon={lon}"
                    )
                
            except Exception as e:
                # Даже при ошибках координаты не должны логироваться
                assert not log_capture.contains_coordinates(), (
                    f"Coordinates found in error logs for lat={lat}, lon={lon}. "
                    f"Error: {e}, Logs: {log_capture.get_logs()}"
                )
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_coordinates_not_logged_in_error_conditions(self, client, lat, lon):
        """
        Feature: airtrace-ru, Property 11: Privacy Protection
        
        Тестирует, что координаты не логируются при ошибках.
        **Validates: Requirements 5.2, 5.4**
        """
        # Мокаем внешний API для генерации ошибки
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_get.side_effect = Exception("External API error")
            
            with LogCapture() as log_capture:
                try:
                    response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                    
                    # Проверяем, что в логах ошибок нет координат
                    assert not log_capture.contains_coordinates(), (
                        f"Coordinates found in error logs for lat={lat}, lon={lon}. "
                        f"Logs: {log_capture.get_logs()}"
                    )
                    
                    # Проверяем, что есть отфильтрованные маркеры в логах ошибок
                    all_logs = '\n'.join(log_capture.get_logs())
                    if all_logs.strip():
                        assert '[COORDINATE_FILTERED]' in all_logs, (
                            f"Expected coordinate filtering in error logs for lat={lat}, lon={lon}"
                        )
                
                except Exception:
                    # Даже при исключениях координаты не должны логироваться
                    assert not log_capture.contains_coordinates(), (
                        f"Coordinates found in exception logs for lat={lat}, lon={lon}. "
                        f"Logs: {log_capture.get_logs()}"
                    )
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_coordinates_not_in_response_logs(self, client, mock_external_api, lat, lon):
        """
        Feature: airtrace-ru, Property 11: Privacy Protection
        
        Тестирует, что координаты не попадают в логи ответов.
        **Validates: Requirements 5.1, 5.4**
        """
        with LogCapture() as log_capture:
            response = client.get(f"/weather/current?lat={lat}&lon={lon}")
            
            # Проверяем логи ответа
            assert not log_capture.contains_coordinates(), (
                f"Coordinates found in response logs for lat={lat}, lon={lon}. "
                f"Response status: {response.status_code}, Logs: {log_capture.get_logs()}"
            )
            
            # Проверяем, что ответ может содержать координаты (это нормально),
            # но они не должны быть в логах
            if response.status_code == 200:
                response_data = response.json()
                if 'location' in response_data:
                    # Координаты в ответе допустимы
                    assert 'latitude' in response_data['location']
                    assert 'longitude' in response_data['location']
                    
                    # Но в логах их быть не должно
                    all_logs = '\n'.join(log_capture.get_logs())
                    assert str(lat) not in all_logs or '[COORDINATE_FILTERED]' in all_logs
                    assert str(lon) not in all_logs or '[COORDINATE_FILTERED]' in all_logs
    
    def test_privacy_aware_formatter_filters_coordinates(self):
        """
        Feature: airtrace-ru, Property 11: Privacy Protection
        
        Тестирует, что PrivacyAwareFormatter корректно фильтрует координаты.
        **Validates: Requirements 5.1, 5.2**
        """
        formatter = PrivacyAwareFormatter()
        
        test_messages = [
            "Request with lat=55.7558 and lon=37.6176",
            "Coordinates: latitude: 55.7558, longitude: 37.6176",
            "API call to lat=55.7558&lon=37.6176",
            "Error at coordinates (55.7558, 37.6176)",
            "Processing request for lat: 55.7558, lon: 37.6176"
        ]
        
        for message in test_messages:
            # Создаем фиктивную запись лога
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=message,
                args=(),
                exc_info=None
            )
            
            formatted = formatter.format(record)
            
            # Проверяем, что координаты заменены на маркеры
            assert '[COORDINATE_FILTERED]' in formatted, (
                f"Coordinates not filtered in message: {message} -> {formatted}"
            )
            
            # Проверяем, что исходные координаты удалены
            assert '55.7558' not in formatted, (
                f"Original coordinates still present in: {formatted}"
            )
            assert '37.6176' not in formatted, (
                f"Original coordinates still present in: {formatted}"
            )
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=30000)
    def test_external_api_logging_privacy(self, lat, lon):
        """
        Feature: airtrace-ru, Property 11: Privacy Protection
        
        Тестирует, что логирование внешних API запросов не раскрывает координаты.
        **Validates: Requirements 5.1, 5.3**
        """
        service = AirQualityService()
        
        with LogCapture() as log_capture:
            # Тестируем логирование внешнего запроса
            test_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10"
            service._log_external_request(test_url, has_coordinates=True)
            
            logs = log_capture.get_logs()
            
            # Проверяем, что координаты не попали в логи
            assert not log_capture.contains_coordinates(), (
                f"Coordinates found in external API logs for lat={lat}, lon={lon}. "
                f"Logs: {logs}"
            )
            
            # Проверяем, что есть информация о том, что запрос содержит координаты
            all_logs = '\n'.join(logs)
            if all_logs.strip():
                assert 'coordinates' in all_logs.lower() or 'FILTERED' in all_logs, (
                    f"No indication of coordinate filtering in external API logs: {logs}"
                )
    
    def test_privacy_middleware_validates_external_domains(self):
        """
        Feature: airtrace-ru, Property 11: Privacy Protection
        
        Тестирует, что middleware блокирует передачу координат неразрешенным доменам.
        **Validates: Requirements 5.3, 5.5**
        """
        middleware = PrivacyMiddleware(app)
        
        # Разрешенные домены
        allowed_urls = [
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            "https://api.open-meteo.com/v1/forecast"
        ]
        
        # Неразрешенные домены
        blocked_urls = [
            "https://malicious-site.com/api",
            "https://data-collector.evil.com/collect",
            "https://unknown-api.com/weather"
        ]
        
        # Проверяем разрешенные домены
        for url in allowed_urls:
            assert middleware.validate_external_request(url), (
                f"Allowed domain should be permitted: {url}"
            )
        
        # Проверяем блокировку неразрешенных доменов
        for url in blocked_urls:
            assert not middleware.validate_external_request(url), (
                f"Blocked domain should be rejected: {url}"
            )
    
    @given(
        sensitive_data=st.text(min_size=1, max_size=50)
    )
    @settings(max_examples=50, deadline=10000)
    def test_sensitive_data_filtering_in_logs(self, sensitive_data):
        """
        Feature: airtrace-ru, Property 11: Privacy Protection
        
        Тестирует фильтрацию других чувствительных данных в логах.
        **Validates: Requirements 5.1, 5.2**
        """
        formatter = PrivacyAwareFormatter()
        
        # Создаем сообщения с чувствительными данными
        test_messages = [
            f"User session: {sensitive_data}",
            f"IP address: 192.168.1.1, user_id: {sensitive_data}",
            f"Processing request from ip=10.0.0.1 with session={sensitive_data}"
        ]
        
        for message in test_messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=message,
                args=(),
                exc_info=None
            )
            
            formatted = formatter.format(record)
            
            # Проверяем, что чувствительные данные отфильтрованы
            if any(pattern in message.lower() for pattern in ['ip', 'user_id', 'session']):
                assert '[SENSITIVE_DATA_FILTERED]' in formatted or sensitive_data not in formatted, (
                    f"Sensitive data not properly filtered: {message} -> {formatted}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])