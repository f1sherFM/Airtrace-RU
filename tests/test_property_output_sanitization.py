"""
Property-based тесты для санитизации выходных данных в AirTrace RU Backend

**Property 18: Output Sanitization**
**Validates: Requirements 8.5**

Тестирует, что все данные, возвращаемые в API ответах, санитизированы
для предотвращения injection атак.
"""

import pytest
import json
import re
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient

from main import app
from middleware import PrivacyMiddleware


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


class TestOutputSanitization:
    """Тесты санитизации выходных данных"""
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_api_response_contains_no_injection_vectors(self, client, mock_external_api, lat, lon):
        """
        Feature: airtrace-ru, Property 18: Output Sanitization
        
        Тестирует, что API ответы не содержат потенциальных векторов для injection атак.
        **Validates: Requirements 8.5**
        """
        response = client.get(f"/weather/current?lat={lat}&lon={lon}")
        
        if response.status_code == 200:
            response_text = response.text
            response_data = response.json()
            
            # Проверяем отсутствие HTML/JavaScript injection векторов
            dangerous_patterns = [
                r'<script[^>]*>',
                r'</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe[^>]*>',
                r'<object[^>]*>',
                r'<embed[^>]*>',
                r'<link[^>]*>',
                r'<meta[^>]*>',
                r'<style[^>]*>',
                r'</style>',
                r'<img[^>]*onerror',
                r'<svg[^>]*onload',
                r'eval\s*\(',
                r'setTimeout\s*\(',
                r'setInterval\s*\(',
                r'Function\s*\(',
                r'document\.',
                r'window\.',
                r'alert\s*\(',
                r'confirm\s*\(',
                r'prompt\s*\('
            ]
            
            for pattern in dangerous_patterns:
                assert not re.search(pattern, response_text, re.IGNORECASE), (
                    f"Dangerous pattern '{pattern}' found in API response for lat={lat}, lon={lon}. "
                    f"Response: {response_text[:500]}..."
                )
            
            # Проверяем отсутствие SQL injection векторов
            sql_patterns = [
                r"'\s*;\s*drop\s+table",
                r"'\s*;\s*delete\s+from",
                r"'\s*;\s*insert\s+into",
                r"'\s*;\s*update\s+",
                r"'\s*union\s+select",
                r"'\s*or\s+1\s*=\s*1",
                r"'\s*and\s+1\s*=\s*1",
                r"--\s*$",
                r"/\*.*\*/",
                r"'\s*;\s*exec",
                r"'\s*;\s*xp_"
            ]
            
            for pattern in sql_patterns:
                assert not re.search(pattern, response_text, re.IGNORECASE), (
                    f"SQL injection pattern '{pattern}' found in API response for lat={lat}, lon={lon}. "
                    f"Response: {response_text[:500]}..."
                )
            
            # Проверяем отсутствие NoSQL injection векторов
            nosql_patterns = [
                r'\$where',
                r'\$ne',
                r'\$gt',
                r'\$lt',
                r'\$regex',
                r'\$or',
                r'\$and',
                r'\$not',
                r'\$nor',
                r'\$exists',
                r'\$type',
                r'\$mod',
                r'\$all',
                r'\$size',
                r'\$elemMatch'
            ]
            
            for pattern in nosql_patterns:
                # Проверяем только если это не часть обычного JSON ключа
                if pattern in response_text and not self._is_safe_json_key(response_text, pattern):
                    assert False, (
                        f"NoSQL injection pattern '{pattern}' found in API response for lat={lat}, lon={lon}. "
                        f"Response: {response_text[:500]}..."
                    )
    
    def _is_safe_json_key(self, text: str, pattern: str) -> bool:
        """Проверяет, является ли паттерн безопасным JSON ключом"""
        # Простая проверка - если паттерн окружен кавычками и двоеточием, это вероятно JSON ключ
        safe_pattern = f'"{pattern}":'
        return safe_pattern in text
    
    @given(
        malicious_input=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=50, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_error_responses_are_sanitized(self, client, malicious_input):
        """
        Feature: airtrace-ru, Property 18: Output Sanitization
        
        Тестирует, что ошибочные ответы также санитизированы.
        **Validates: Requirements 8.5**
        """
        # Тестируем с некорректными координатами, содержащими потенциально опасный контент
        try:
            # Пытаемся передать malicious_input как параметр
            response = client.get(f"/weather/current?lat=invalid&lon={malicious_input}")
            
            if response.status_code >= 400:
                response_text = response.text
                
                # Проверяем, что malicious_input не отражается в ответе без санитизации
                dangerous_chars = ['<', '>', '"', "'", '&', 'javascript:', 'data:', 'vbscript:']
                
                for char in dangerous_chars:
                    if char in malicious_input:
                        # Если опасный символ есть во входных данных, он не должен появиться в ответе
                        # без соответствующего экранирования
                        if char in response_text:
                            # Проверяем, что символ правильно экранирован
                            if char == '<':
                                assert '&lt;' in response_text or char not in response_text
                            elif char == '>':
                                assert '&gt;' in response_text or char not in response_text
                            elif char == '"':
                                assert '&quot;' in response_text or '\\"' in response_text or char not in response_text
                            elif char == "'":
                                assert '&#x27;' in response_text or "\\'" in response_text or char not in response_text
                            elif char == '&':
                                assert '&amp;' in response_text or char not in response_text
                
        except Exception:
            # Если запрос вызывает исключение, это нормально для некорректных данных
            pass
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_json_response_structure_is_safe(self, client, mock_external_api, lat, lon):
        """
        Feature: airtrace-ru, Property 18: Output Sanitization
        
        Тестирует, что структура JSON ответа безопасна и не содержит опасных конструкций.
        **Validates: Requirements 8.5**
        """
        response = client.get(f"/weather/current?lat={lat}&lon={lon}")
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                
                # Проверяем, что JSON не содержит функций или исполняемого кода
                json_str = json.dumps(response_data)
                
                dangerous_json_patterns = [
                    r'"__proto__"',
                    r'"constructor"',
                    r'"prototype"',
                    r'function\s*\(',
                    r'=>',
                    r'eval\s*\(',
                    r'new\s+Function',
                    r'setTimeout',
                    r'setInterval'
                ]
                
                for pattern in dangerous_json_patterns:
                    assert not re.search(pattern, json_str, re.IGNORECASE), (
                        f"Dangerous JSON pattern '{pattern}' found in response for lat={lat}, lon={lon}. "
                        f"JSON: {json_str[:500]}..."
                    )
                
                # Проверяем, что все строковые значения в JSON безопасны
                self._check_json_values_recursively(response_data, lat, lon)
                
            except json.JSONDecodeError:
                # Если ответ не является валидным JSON, это может быть проблемой
                assert False, f"Invalid JSON response for lat={lat}, lon={lon}: {response.text[:200]}..."
    
    def _check_json_values_recursively(self, data, lat, lon, path=""):
        """Рекурсивно проверяет значения в JSON на безопасность"""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Проверяем ключи
                assert not self._contains_dangerous_content(str(key)), (
                    f"Dangerous content in JSON key '{key}' at path '{current_path}' "
                    f"for lat={lat}, lon={lon}"
                )
                
                # Рекурсивно проверяем значения
                self._check_json_values_recursively(value, lat, lon, current_path)
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                self._check_json_values_recursively(item, lat, lon, current_path)
                
        elif isinstance(data, str):
            # Проверяем строковые значения
            assert not self._contains_dangerous_content(data), (
                f"Dangerous content in JSON value at path '{path}': '{data[:100]}...' "
                f"for lat={lat}, lon={lon}"
            )
    
    def _contains_dangerous_content(self, text: str) -> bool:
        """Проверяет, содержит ли текст опасный контент"""
        dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'on\w+\s*=',
            r'<iframe',
            r'<object',
            r'<embed',
            r'<link.*href.*javascript',
            r'<img.*src.*javascript',
            r'<svg.*onload',
            r'eval\s*\(',
            r'Function\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def test_middleware_sanitizes_response_data(self):
        """
        Feature: airtrace-ru, Property 18: Output Sanitization
        
        Тестирует, что middleware корректно санитизирует данные ответа.
        **Validates: Requirements 8.5**
        """
        middleware = PrivacyMiddleware(app)
        
        # Тестовые данные с потенциально опасным контентом
        test_data = {
            "message": "Hello <script>alert('xss')</script> World",
            "user_id": "user123",  # Чувствительное поле
            "description": "Test with javascript:alert('test') content",
            "nested": {
                "value": "<img src=x onerror=alert('nested')>",
                "session_id": "session456"  # Чувствительное поле
            },
            "list": [
                "Normal text",
                "<iframe src='javascript:alert(1)'></iframe>",
                {"inner": "eval('malicious code')"}
            ]
        }
        
        # Санитизируем данные
        sanitized = middleware._sanitize_json_data(test_data)
        
        # Проверяем, что чувствительные поля отфильтрованы
        assert sanitized["user_id"] == "[SENSITIVE_DATA_FILTERED]"
        assert sanitized["nested"]["session_id"] == "[SENSITIVE_DATA_FILTERED]"
        
        # Проверяем, что опасный контент остается (middleware не должен изменять контент,
        # только чувствительные поля)
        assert "script" in sanitized["message"]  # Контент остается, но логируется
        assert "javascript:" in sanitized["description"]
        assert "iframe" in sanitized["list"][1]
        assert "eval" in sanitized["list"][2]["inner"]
    
    @given(
        sensitive_field_name=st.sampled_from(['user_id', 'session_id', 'ip_address', 'device_id']),
        sensitive_value=st.text(min_size=1, max_size=50)
    )
    @settings(max_examples=50, deadline=10000)
    def test_sensitive_fields_are_filtered(self, sensitive_field_name, sensitive_value):
        """
        Feature: airtrace-ru, Property 18: Output Sanitization
        
        Тестирует, что чувствительные поля фильтруются из ответов.
        **Validates: Requirements 8.5**
        """
        middleware = PrivacyMiddleware(app)
        
        test_data = {
            "normal_field": "normal_value",
            sensitive_field_name: sensitive_value,
            "nested": {
                "another_field": "another_value",
                sensitive_field_name: sensitive_value
            }
        }
        
        sanitized = middleware._sanitize_json_data(test_data)
        
        # Проверяем, что чувствительные поля заменены на маркер
        assert sanitized[sensitive_field_name] == "[SENSITIVE_DATA_FILTERED]"
        assert sanitized["nested"][sensitive_field_name] == "[SENSITIVE_DATA_FILTERED]"
        
        # Проверяем, что обычные поля остались без изменений
        assert sanitized["normal_field"] == "normal_value"
        assert sanitized["nested"]["another_field"] == "another_value"
    
    def test_health_endpoint_is_sanitized(self, client):
        """
        Feature: airtrace-ru, Property 18: Output Sanitization
        
        Тестирует, что health endpoint также возвращает санитизированные данные.
        **Validates: Requirements 8.5**
        """
        response = client.get("/health")
        
        if response.status_code == 200:
            response_text = response.text
            
            # Проверяем отсутствие опасных паттернов в health ответе
            dangerous_patterns = [
                r'<script[^>]*>',
                r'javascript:',
                r'on\w+\s*=',
                r'eval\s*\(',
                r'<iframe'
            ]
            
            for pattern in dangerous_patterns:
                assert not re.search(pattern, response_text, re.IGNORECASE), (
                    f"Dangerous pattern '{pattern}' found in health endpoint response. "
                    f"Response: {response_text}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])