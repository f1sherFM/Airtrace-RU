"""
Property-based tests for health check completeness.

**Property 19: Health Check Completeness**
*For any* health check request, the response should include service status, 
Open-Meteo API connectivity status, and cache system status, returning 
unhealthy status when critical components are unavailable.

**Validates: Requirements 9.2, 9.3, 9.4**
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient
import json
from datetime import datetime
from typing import Dict, Any
from unittest.mock import AsyncMock, patch

from main import app
from schemas import HealthCheckResponse


class TestHealthCheckCompletenessProperty:
    """Property-based tests for health check completeness"""
    
    @pytest.fixture(scope="class")
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @given(
        external_api_healthy=st.booleans(),
        cache_healthy=st.booleans(),
        aqi_calculator_healthy=st.booleans()
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_health_check_completeness_property(
        self, 
        client, 
        external_api_healthy: bool, 
        cache_healthy: bool, 
        aqi_calculator_healthy: bool
    ):
        """
        **Property 19: Health Check Completeness**
        
        **Validates: Requirements 9.2, 9.3, 9.4**
        
        For any health check request, the response should include service status,
        Open-Meteo API connectivity status, and cache system status, returning
        unhealthy status when critical components are unavailable.
        """
        # Мокаем различные состояния компонентов системы
        with patch('services.AirQualityService.check_external_api_health') as mock_external_api:
            with patch('services.CacheManager.get_status') as mock_cache_status:
                with patch('utils.AQICalculator.calculate_aqi') as mock_aqi_calc:
                    
                    # Настраиваем моки в зависимости от входных параметров
                    if external_api_healthy:
                        mock_external_api.return_value = "healthy"
                    else:
                        mock_external_api.return_value = "unhealthy"
                    
                    if cache_healthy:
                        mock_cache_status.return_value = "healthy (5 entries)"
                    else:
                        mock_cache_status.return_value = "unhealthy"
                    
                    if aqi_calculator_healthy:
                        mock_aqi_calc.return_value = (85, "Умеренное", "#FFFF00")
                    else:
                        mock_aqi_calc.side_effect = Exception("AQI calculation failed")
                    
                    # Выполняем запрос к health check endpoint
                    response = client.get("/health")
                    
                    # Проверяем, что ответ всегда успешный (health check не должен падать)
                    assert response.status_code == 200, (
                        f"Health check should always return 200, got {response.status_code}"
                    )
                    
                    # Проверяем, что ответ является валидным JSON
                    try:
                        response_data = response.json()
                    except json.JSONDecodeError:
                        pytest.fail("Health check response is not valid JSON")
                    
                    # Валидируем структуру ответа
                    self._validate_health_check_response_structure(response_data)
                    
                    # Проверяем логику определения общего статуса
                    expected_overall_status = self._calculate_expected_overall_status(
                        external_api_healthy, cache_healthy, aqi_calculator_healthy
                    )
                    
                    assert response_data['status'] == expected_overall_status, (
                        f"Expected overall status '{expected_overall_status}', "
                        f"got '{response_data['status']}' for external_api={external_api_healthy}, "
                        f"cache={cache_healthy}, aqi_calc={aqi_calculator_healthy}"
                    )
                    
                    # Проверяем статусы отдельных компонентов
                    services = response_data['services']
                    
                    # API всегда должен быть healthy (если мы можем ответить)
                    assert services['api'] == 'healthy', (
                        "API service should always be healthy in health check response"
                    )
                    
                    # Проверяем статус внешнего API
                    expected_external_status = "healthy" if external_api_healthy else "unhealthy"
                    assert services['external_api'] == expected_external_status, (
                        f"Expected external_api status '{expected_external_status}', "
                        f"got '{services['external_api']}'"
                    )
                    
                    # Проверяем статус кэша
                    if cache_healthy:
                        assert "healthy" in services['cache'], (
                            f"Expected cache status to contain 'healthy', got '{services['cache']}'"
                        )
                    else:
                        assert services['cache'] == "unhealthy", (
                            f"Expected cache status 'unhealthy', got '{services['cache']}'"
                        )
                    
                    # Проверяем статус AQI калькулятора
                    expected_aqi_status = "healthy" if aqi_calculator_healthy else "unhealthy"
                    assert services['aqi_calculator'] == expected_aqi_status, (
                        f"Expected aqi_calculator status '{expected_aqi_status}', "
                        f"got '{services['aqi_calculator']}'"
                    )
    
    @given(
        failure_scenario=st.sampled_from(['network_timeout', 'http_error', 'invalid_response', 'healthy'])
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_health_check_external_api_failure_scenarios_property(
        self, 
        client, 
        failure_scenario: str
    ):
        """
        **Property 19: Health Check Completeness**
        
        **Validates: Requirements 9.2, 9.3, 9.4**
        
        For any external API failure scenario, the health check should still
        return a complete response with appropriate error status.
        """
        # Мокаем метод проверки внешнего API напрямую
        with patch('services.AirQualityService.check_external_api_health') as mock_external_api:
            with patch('services.CacheManager.get_status') as mock_cache_status:
                with patch('utils.AQICalculator.calculate_aqi') as mock_aqi_calc:
                    
                    # Настраиваем моки в зависимости от сценария
                    if failure_scenario == 'network_timeout':
                        mock_external_api.return_value = "unhealthy"
                    elif failure_scenario == 'http_error':
                        mock_external_api.return_value = "unhealthy"
                    elif failure_scenario == 'invalid_response':
                        mock_external_api.return_value = "degraded"
                    else:  # healthy case
                        mock_external_api.return_value = "healthy"
                    
                    # Кэш и AQI калькулятор всегда здоровы в этом тесте
                    mock_cache_status.return_value = "healthy (5 entries)"
                    mock_aqi_calc.return_value = (85, "Умеренное", "#FFFF00")
                    
                    # Выполняем запрос к health check endpoint
                    response = client.get("/health")
                    
                    # Проверяем, что ответ всегда успешный
                    assert response.status_code == 200, (
                        f"Health check should always return 200 even with external API failures, "
                        f"got {response.status_code}"
                    )
                    
                    # Проверяем структуру ответа
                    try:
                        response_data = response.json()
                    except json.JSONDecodeError:
                        pytest.fail("Health check response is not valid JSON")
                    
                    self._validate_health_check_response_structure(response_data)
                    
                    # Проверяем, что при ошибках внешнего API общий статус unhealthy
                    if failure_scenario in ['network_timeout', 'http_error', 'invalid_response']:
                        assert response_data['status'] == 'unhealthy', (
                            f"Expected overall status 'unhealthy' when external API fails with {failure_scenario}, "
                            f"got '{response_data['status']}'"
                        )
                        
                        # Проверяем, что статус внешнего API отражает проблему
                        external_api_status = response_data['services']['external_api']
                        assert external_api_status in ['unhealthy', 'degraded'], (
                            f"Expected external_api status 'unhealthy' or 'degraded' when API fails with {failure_scenario}, "
                            f"got '{external_api_status}'"
                        )
                    else:
                        # В нормальном случае статус должен быть healthy
                        assert response_data['status'] == 'healthy', (
                            f"Expected overall status 'healthy' when all components work, "
                            f"got '{response_data['status']}'"
                        )
    
    def test_health_check_response_time_property(self, client):
        """
        **Property 19: Health Check Completeness**
        
        **Validates: Requirements 9.2, 9.3, 9.4**
        
        Health check should respond within reasonable time limits (< 5 seconds)
        as specified in requirements.
        """
        import time
        
        # Мокаем нормальное поведение компонентов
        with patch('services.AirQualityService.check_external_api_health') as mock_external_api:
            with patch('services.CacheManager.get_status') as mock_cache_status:
                with patch('utils.AQICalculator.calculate_aqi') as mock_aqi_calc:
                    
                    mock_external_api.return_value = "healthy"
                    mock_cache_status.return_value = "healthy (10 entries)"
                    mock_aqi_calc.return_value = (75, "Умеренное", "#FFFF00")
                    
                    # Измеряем время выполнения
                    start_time = time.time()
                    response = client.get("/health")
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    
                    # Проверяем, что ответ получен в разумное время
                    assert response_time < 5.0, (
                        f"Health check should respond within 5 seconds, took {response_time:.2f}s"
                    )
                    
                    # Проверяем успешность ответа
                    assert response.status_code == 200, (
                        f"Health check should return 200, got {response.status_code}"
                    )
    
    def _validate_health_check_response_structure(self, response_data: Dict[str, Any]):
        """Валидация структуры ответа health check"""
        # Проверяем наличие всех обязательных полей верхнего уровня
        required_fields = ['status', 'timestamp', 'services']
        for field in required_fields:
            assert field in response_data, (
                f"Missing required field '{field}' in health check response"
            )
        
        # Проверяем тип и значения поля status
        status = response_data['status']
        assert isinstance(status, str), "Status should be string"
        assert status in ['healthy', 'unhealthy'], (
            f"Status should be 'healthy' or 'unhealthy', got '{status}'"
        )
        
        # Проверяем timestamp
        timestamp = response_data['timestamp']
        assert isinstance(timestamp, str), "Timestamp should be string"
        
        # Проверяем, что timestamp можно распарсить как ISO datetime
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"Invalid timestamp format '{timestamp}'")
        
        # Проверяем структуру services
        services = response_data['services']
        assert isinstance(services, dict), "Services should be dict"
        
        # Проверяем наличие всех обязательных сервисов
        required_services = ['api', 'external_api', 'cache', 'aqi_calculator']
        for service in required_services:
            assert service in services, (
                f"Missing required service '{service}' in health check response"
            )
            
            service_status = services[service]
            assert isinstance(service_status, str), (
                f"Service '{service}' status should be string, got {type(service_status)}"
            )
            assert len(service_status) > 0, (
                f"Service '{service}' status should not be empty"
            )
        
        # Проверяем валидные значения статусов сервисов
        valid_statuses = ['healthy', 'unhealthy', 'degraded', 'unknown']
        
        # API всегда должен быть healthy
        assert services['api'] == 'healthy', (
            "API service should always be healthy in health check response"
        )
        
        # Внешний API может быть healthy, unhealthy или degraded
        external_api_status = services['external_api']
        assert external_api_status in valid_statuses, (
            f"External API status should be one of {valid_statuses}, got '{external_api_status}'"
        )
        
        # Кэш может быть healthy (с деталями) или unhealthy
        cache_status = services['cache']
        assert (cache_status in valid_statuses or 
                any(status in cache_status for status in ['healthy', 'unhealthy'])), (
            f"Cache status should contain valid status, got '{cache_status}'"
        )
        
        # AQI калькулятор может быть healthy или unhealthy
        aqi_calc_status = services['aqi_calculator']
        assert aqi_calc_status in valid_statuses, (
            f"AQI calculator status should be one of {valid_statuses}, got '{aqi_calc_status}'"
        )
    
    def _calculate_expected_overall_status(
        self, 
        external_api_healthy: bool, 
        cache_healthy: bool, 
        aqi_calculator_healthy: bool
    ) -> str:
        """Вычисление ожидаемого общего статуса на основе состояния компонентов"""
        # Если любой критический компонент нездоров, общий статус unhealthy
        if not external_api_healthy or not cache_healthy or not aqi_calculator_healthy:
            return "unhealthy"
        else:
            return "healthy"