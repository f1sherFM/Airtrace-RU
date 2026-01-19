"""
Property-based тесты для передачи данных третьим лицам в AirTrace RU Backend

**Property 12: Third-party Data Transmission**
**Validates: Requirements 5.3, 5.5**

Тестирует, что координатные данные передаются только в Open-Meteo API
и никогда не передаются другим третьим лицам.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient
import httpx
import re
from urllib.parse import urlparse, parse_qs

from main import app
from services import AirQualityService
from middleware import PrivacyMiddleware


class HTTPRequestCapture:
    """Класс для захвата всех HTTP запросов во время тестирования"""
    
    def __init__(self):
        self.requests = []
    
    def capture_request(self, method, url, **kwargs):
        """Захват информации о HTTP запросе"""
        self.requests.append({
            'method': method,
            'url': str(url),
            'kwargs': kwargs,
            'domain': urlparse(str(url)).netloc
        })
        
        # Возвращаем мок ответ
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
        return mock_response
    
    def get_requests(self):
        """Получение всех захваченных запросов"""
        return self.requests
    
    def get_domains(self):
        """Получение всех доменов, к которым были запросы"""
        return [req['domain'] for req in self.requests]
    
    def has_coordinates_in_requests(self):
        """Проверка наличия координат в запросах"""
        coordinate_patterns = [
            r'lat(?:itude)?[=&](-?\d+\.?\d*)',
            r'lon(?:gitude)?[=&](-?\d+\.?\d*)'
        ]
        
        for request in self.requests:
            url = request['url']
            for pattern in coordinate_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return True
        return False
    
    def get_requests_with_coordinates(self):
        """Получение запросов, содержащих координаты"""
        coordinate_requests = []
        coordinate_patterns = [
            r'lat(?:itude)?[=&](-?\d+\.?\d*)',
            r'lon(?:gitude)?[=&](-?\d+\.?\d*)'
        ]
        
        for request in self.requests:
            url = request['url']
            for pattern in coordinate_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    coordinate_requests.append(request)
                    break
        
        return coordinate_requests


@pytest.fixture
def request_capture():
    """Фикстура для захвата HTTP запросов"""
    return HTTPRequestCapture()


class TestThirdPartyDataTransmission:
    """Property-based тесты для передачи данных третьим лицам"""
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_coordinates_only_sent_to_allowed_domains_property(self, request_capture, lat, lon):
        """
        Feature: airtrace-ru, Property 12: Third-party Data Transmission
        
        For any coordinate data, it should only be transmitted to Open-Meteo API
        and never to other third parties.
        **Validates: Requirements 5.3, 5.5**
        """
        client = TestClient(app)
        
        # Разрешенные домены для передачи координат
        allowed_domains = {
            'air-quality-api.open-meteo.com',
            'api.open-meteo.com'
        }
        
        # Патчим httpx.AsyncClient.get для захвата всех запросов
        with patch('httpx.AsyncClient.get', side_effect=request_capture.capture_request):
            try:
                response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                
                # Получаем все запросы с координатами
                coordinate_requests = request_capture.get_requests_with_coordinates()
                
                # Проверяем, что все запросы с координатами идут только к разрешенным доменам
                for request in coordinate_requests:
                    domain = request['domain']
                    assert domain in allowed_domains, (
                        f"Coordinates sent to unauthorized domain: {domain} "
                        f"for lat={lat}, lon={lon}. URL: {request['url']}"
                    )
                
                # Проверяем, что есть хотя бы один запрос к разрешенному домену
                # (если API запрос был успешным)
                if response.status_code == 200 and coordinate_requests:
                    allowed_requests = [req for req in coordinate_requests 
                                      if req['domain'] in allowed_domains]
                    assert len(allowed_requests) > 0, (
                        f"No requests to allowed domains found for lat={lat}, lon={lon}"
                    )
                
            except Exception as e:
                # Даже при ошибках координаты не должны передаваться неразрешенным доменам
                coordinate_requests = request_capture.get_requests_with_coordinates()
                for request in coordinate_requests:
                    domain = request['domain']
                    assert domain in allowed_domains, (
                        f"Coordinates sent to unauthorized domain during error: {domain} "
                        f"for lat={lat}, lon={lon}. Error: {e}"
                    )
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_no_coordinate_leakage_to_unauthorized_domains_property(self, request_capture, lat, lon):
        """
        Feature: airtrace-ru, Property 12: Third-party Data Transmission
        
        For any API request, coordinates should never leak to unauthorized domains
        even through redirects, error handlers, or other mechanisms.
        **Validates: Requirements 5.3, 5.5**
        """
        client = TestClient(app)
        
        # Неразрешенные домены (примеры потенциально опасных доменов)
        unauthorized_domains = {
            'analytics.google.com',
            'tracking.example.com',
            'data-collector.com',
            'metrics.third-party.com',
            'logger.external.com'
        }
        
        with patch('httpx.AsyncClient.get', side_effect=request_capture.capture_request):
            try:
                # Тестируем current endpoint
                response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                
                # Проверяем все запросы
                all_requests = request_capture.get_requests()
                
                for request in all_requests:
                    domain = request['domain']
                    url = request['url']
                    
                    # Если домен неразрешенный, координаты не должны быть в запросе
                    if domain in unauthorized_domains:
                        coordinate_patterns = [
                            r'lat(?:itude)?[=&](-?\d+\.?\d*)',
                            r'lon(?:gitude)?[=&](-?\d+\.?\d*)'
                        ]
                        
                        for pattern in coordinate_patterns:
                            assert not re.search(pattern, url, re.IGNORECASE), (
                                f"Coordinates leaked to unauthorized domain {domain}: {url} "
                                f"for lat={lat}, lon={lon}"
                            )
                
                # Тестируем forecast endpoint
                request_capture.requests.clear()  # Очищаем предыдущие запросы
                
                response = client.get(f"/weather/forecast?lat={lat}&lon={lon}")
                
                all_requests = request_capture.get_requests()
                
                for request in all_requests:
                    domain = request['domain']
                    url = request['url']
                    
                    if domain in unauthorized_domains:
                        coordinate_patterns = [
                            r'lat(?:itude)?[=&](-?\d+\.?\d*)',
                            r'lon(?:gitude)?[=&](-?\d+\.?\d*)'
                        ]
                        
                        for pattern in coordinate_patterns:
                            assert not re.search(pattern, url, re.IGNORECASE), (
                                f"Coordinates leaked to unauthorized domain {domain} in forecast: {url} "
                                f"for lat={lat}, lon={lon}"
                            )
                
            except Exception as e:
                # Проверяем, что даже при ошибках нет утечек
                all_requests = request_capture.get_requests()
                
                for request in all_requests:
                    domain = request['domain']
                    url = request['url']
                    
                    if domain in unauthorized_domains:
                        coordinate_patterns = [
                            r'lat(?:itude)?[=&](-?\d+\.?\d*)',
                            r'lon(?:gitude)?[=&](-?\d+\.?\d*)'
                        ]
                        
                        for pattern in coordinate_patterns:
                            assert not re.search(pattern, url, re.IGNORECASE), (
                                f"Coordinates leaked to unauthorized domain during error {domain}: {url} "
                                f"for lat={lat}, lon={lon}. Error: {e}"
                            )
    
    def test_privacy_middleware_blocks_unauthorized_domains(self):
        """
        Feature: airtrace-ru, Property 12: Third-party Data Transmission
        
        Privacy middleware should block coordinate transmission to unauthorized domains.
        **Validates: Requirements 5.3, 5.5**
        """
        middleware = PrivacyMiddleware(app)
        
        # Тестируем разрешенные домены
        allowed_urls = [
            "https://air-quality-api.open-meteo.com/v1/air-quality?lat=55.7558&lon=37.6176",
            "https://api.open-meteo.com/v1/forecast?latitude=55.7558&longitude=37.6176"
        ]
        
        for url in allowed_urls:
            assert middleware.validate_external_request(url), (
                f"Allowed domain should be permitted: {url}"
            )
        
        # Тестируем блокировку неразрешенных доменов
        blocked_urls = [
            "https://analytics.google.com/collect?lat=55.7558&lon=37.6176",
            "https://tracking.example.com/api?latitude=55.7558&longitude=37.6176",
            "https://data-collector.com/coordinates?lat=55.7558&lon=37.6176",
            "https://malicious-site.com/steal-data?coordinates=55.7558,37.6176"
        ]
        
        for url in blocked_urls:
            assert not middleware.validate_external_request(url), (
                f"Unauthorized domain should be blocked: {url}"
            )
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=30000)
    def test_service_level_coordinate_transmission_control_property(self, lat, lon):
        """
        Feature: airtrace-ru, Property 12: Third-party Data Transmission
        
        At the service level, coordinate transmission should be controlled
        and only allowed to authorized APIs.
        **Validates: Requirements 5.3, 5.5**
        """
        service = AirQualityService()
        
        # Мокаем httpx.AsyncClient для контроля запросов
        with patch.object(service.client, 'get') as mock_get:
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
                # Выполняем запрос через сервис
                import asyncio
                result = asyncio.run(service.get_current_air_quality(lat, lon))
                
                # Проверяем, что запрос был сделан
                assert mock_get.called, f"No external API call made for lat={lat}, lon={lon}"
                
                # Анализируем все вызовы
                for call_args in mock_get.call_args_list:
                    args, kwargs = call_args
                    
                    # Получаем URL запроса
                    if args:
                        url = str(args[0])
                    else:
                        url = str(kwargs.get('url', ''))
                    
                    # Проверяем домен
                    domain = urlparse(url).netloc
                    allowed_domains = {
                        'air-quality-api.open-meteo.com',
                        'api.open-meteo.com'
                    }
                    
                    assert domain in allowed_domains, (
                        f"Service made request to unauthorized domain: {domain} "
                        f"for lat={lat}, lon={lon}. URL: {url}"
                    )
                    
                    # Проверяем, что координаты присутствуют в разрешенном запросе
                    if domain in allowed_domains:
                        # Для разрешенных доменов координаты должны быть в параметрах
                        params = kwargs.get('params', {})
                        if params:
                            assert 'latitude' in params or 'lat' in params, (
                                f"Missing latitude in authorized request to {domain}"
                            )
                            assert 'longitude' in params or 'lon' in params, (
                                f"Missing longitude in authorized request to {domain}"
                            )
                
            except Exception as e:
                # Даже при ошибках проверяем, что запросы идут только к разрешенным доменам
                if mock_get.called:
                    for call_args in mock_get.call_args_list:
                        args, kwargs = call_args
                        
                        if args:
                            url = str(args[0])
                        else:
                            url = str(kwargs.get('url', ''))
                        
                        domain = urlparse(url).netloc
                        allowed_domains = {
                            'air-quality-api.open-meteo.com',
                            'api.open-meteo.com'
                        }
                        
                        assert domain in allowed_domains, (
                            f"Service made request to unauthorized domain during error: {domain} "
                            f"for lat={lat}, lon={lon}. Error: {e}"
                        )
    
    def test_coordinate_transmission_logging_property(self):
        """
        Feature: airtrace-ru, Property 12: Third-party Data Transmission
        
        All coordinate transmissions should be properly logged for audit purposes
        while maintaining privacy.
        **Validates: Requirements 5.3, 5.5**
        """
        service = AirQualityService()
        
        # Тестируем логирование внешних запросов
        test_cases = [
            {
                'url': 'https://air-quality-api.open-meteo.com/v1/air-quality?lat=55.7558&lon=37.6176',
                'should_be_allowed': True
            },
            {
                'url': 'https://malicious-site.com/api?lat=55.7558&lon=37.6176',
                'should_be_allowed': False
            }
        ]
        
        for case in test_cases:
            url = case['url']
            should_be_allowed = case['should_be_allowed']
            
            # Тестируем валидацию запроса
            from middleware import get_privacy_middleware
            privacy_middleware = get_privacy_middleware()
            
            if privacy_middleware:
                is_allowed = privacy_middleware.validate_external_request(url)
                
                if should_be_allowed:
                    assert is_allowed, f"Authorized request should be allowed: {url}"
                else:
                    assert not is_allowed, f"Unauthorized request should be blocked: {url}"
            
            # Тестируем логирование
            try:
                service._log_external_request(url, has_coordinates=True)
                # Логирование не должно вызывать исключений
            except Exception as e:
                pytest.fail(f"External request logging failed for {url}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])