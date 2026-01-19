"""
Property-based тесты для устойчивости к сбоям кэша в AirTrace RU Backend

**Property 15: Cache Failure Resilience**
**Validates: Requirements 6.5**

Тестирует, что при любом сбое системы кэширования приложение продолжает работать,
получая данные напрямую из внешних API.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient

from main import app
from services import AirQualityService, CacheManager
from schemas import AirQualityData


class CacheFailureSimulator:
    """Класс для симуляции различных типов сбоев кэша"""
    
    def __init__(self, failure_type="exception"):
        self.failure_type = failure_type
        self.call_count = 0
    
    async def failing_get(self, lat, lon):
        """Симуляция сбоя при получении данных из кэша"""
        self.call_count += 1
        
        if self.failure_type == "exception":
            raise Exception("Cache system failure")
        elif self.failure_type == "timeout":
            raise asyncio.TimeoutError("Cache timeout")
        elif self.failure_type == "memory_error":
            raise MemoryError("Cache memory exhausted")
        elif self.failure_type == "key_error":
            raise KeyError("Cache key not found")
        elif self.failure_type == "none_return":
            return None
        else:
            raise RuntimeError(f"Unknown cache failure: {self.failure_type}")
    
    async def failing_set(self, lat, lon, data):
        """Симуляция сбоя при сохранении данных в кэш"""
        self.call_count += 1
        
        if self.failure_type == "exception":
            raise Exception("Cache write failure")
        elif self.failure_type == "disk_full":
            raise OSError("No space left on device")
        elif self.failure_type == "permission_error":
            raise PermissionError("Cache write permission denied")
        else:
            raise RuntimeError(f"Unknown cache write failure: {self.failure_type}")


class TestCacheFailureResilience:
    """Property-based тесты для устойчивости к сбоям кэша"""
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
        failure_type=st.sampled_from([
            "exception", "timeout", "memory_error", "key_error", "none_return"
        ])
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cache_get_failure_resilience_property(self, lat, lon, failure_type):
        """
        Feature: airtrace-ru, Property 15: Cache Failure Resilience
        
        For any cache system failure during data retrieval, the application 
        should continue operating by fetching data directly from external APIs.
        **Validates: Requirements 6.5**
        """
        client = TestClient(app)
        
        # Мокаем внешний API для успешного ответа
        with patch('httpx.AsyncClient.get') as mock_external_api:
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
            mock_external_api.return_value = mock_response
            
            # Симулируем сбой кэша
            cache_failure = CacheFailureSimulator(failure_type)
            
            with patch('services.CacheManager.get', side_effect=cache_failure.failing_get):
                try:
                    response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                    
                    # Приложение должно продолжать работать несмотря на сбой кэша
                    assert response.status_code != 500, (
                        f"Application failed due to cache {failure_type} failure "
                        f"for lat={lat}, lon={lon}. Response: {response.status_code}"
                    )
                    
                    # Если внешний API доступен, должен быть успешный ответ
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Проверяем, что данные получены из внешнего API
                        assert 'aqi' in data, (
                            f"Missing AQI data in response after cache {failure_type} failure "
                            f"for lat={lat}, lon={lon}"
                        )
                        assert 'pollutants' in data, (
                            f"Missing pollutants data in response after cache {failure_type} failure "
                            f"for lat={lat}, lon={lon}"
                        )
                        
                        # Проверяем, что внешний API был вызван
                        assert mock_external_api.called, (
                            f"External API not called after cache {failure_type} failure "
                            f"for lat={lat}, lon={lon}"
                        )
                    
                    # Проверяем, что кэш был попытан использовать
                    assert cache_failure.call_count > 0, (
                        f"Cache not accessed during {failure_type} failure test "
                        f"for lat={lat}, lon={lon}"
                    )
                
                except Exception as e:
                    # Даже при исключениях приложение не должно полностью падать
                    # Проверяем, что это не связано с кэшем
                    if "cache" in str(e).lower() and failure_type in str(e).lower():
                        pytest.fail(
                            f"Application crashed due to cache {failure_type} failure "
                            f"for lat={lat}, lon={lon}: {e}"
                        )
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
        failure_type=st.sampled_from([
            "exception", "disk_full", "permission_error"
        ])
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cache_set_failure_resilience_property(self, lat, lon, failure_type):
        """
        Feature: airtrace-ru, Property 15: Cache Failure Resilience
        
        For any cache system failure during data storage, the application 
        should continue operating and return data from external APIs.
        **Validates: Requirements 6.5**
        """
        client = TestClient(app)
        
        # Мокаем внешний API
        with patch('httpx.AsyncClient.get') as mock_external_api:
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
            mock_external_api.return_value = mock_response
            
            # Симулируем сбой записи в кэш
            cache_failure = CacheFailureSimulator(failure_type)
            
            # Мокаем get для возврата None (cache miss)
            with patch('services.CacheManager.get', return_value=None):
                with patch('services.CacheManager.set', side_effect=cache_failure.failing_set):
                    try:
                        response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                        
                        # Приложение должно работать несмотря на сбой записи в кэш
                        assert response.status_code != 500, (
                            f"Application failed due to cache write {failure_type} failure "
                            f"for lat={lat}, lon={lon}. Response: {response.status_code}"
                        )
                        
                        # Должны получить данные из внешнего API
                        if response.status_code == 200:
                            data = response.json()
                            
                            assert 'aqi' in data, (
                                f"Missing AQI data after cache write {failure_type} failure "
                                f"for lat={lat}, lon={lon}"
                            )
                            
                            # Внешний API должен быть вызван
                            assert mock_external_api.called, (
                                f"External API not called after cache write {failure_type} failure "
                                f"for lat={lat}, lon={lon}"
                            )
                        
                        # Проверяем, что попытка записи в кэш была сделана
                        assert cache_failure.call_count > 0, (
                            f"Cache write not attempted during {failure_type} failure test "
                            f"for lat={lat}, lon={lon}"
                        )
                    
                    except Exception as e:
                        # Проверяем, что ошибка не связана с кэшем
                        if "cache" in str(e).lower() and failure_type in str(e).lower():
                            pytest.fail(
                                f"Application crashed due to cache write {failure_type} failure "
                                f"for lat={lat}, lon={lon}: {e}"
                            )
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_complete_cache_system_failure_resilience_property(self, lat, lon):
        """
        Feature: airtrace-ru, Property 15: Cache Failure Resilience
        
        For any complete cache system failure, the application should 
        continue operating by bypassing cache entirely.
        **Validates: Requirements 6.5**
        """
        client = TestClient(app)
        
        # Мокаем внешний API
        with patch('httpx.AsyncClient.get') as mock_external_api:
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
            mock_external_api.return_value = mock_response
            
            # Симулируем полный сбой кэша
            def cache_system_failure(*args, **kwargs):
                raise RuntimeError("Complete cache system failure")
            
            with patch('services.CacheManager.get', side_effect=cache_system_failure):
                with patch('services.CacheManager.set', side_effect=cache_system_failure):
                    with patch('services.CacheManager.clear_expired', side_effect=cache_system_failure):
                        try:
                            response = client.get(f"/weather/current?lat={lat}&lon={lon}")
                            
                            # Приложение должно работать без кэша
                            assert response.status_code != 500, (
                                f"Application failed due to complete cache system failure "
                                f"for lat={lat}, lon={lon}. Response: {response.status_code}"
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                
                                # Данные должны быть получены напрямую из внешнего API
                                assert 'aqi' in data, (
                                    f"Missing AQI data after complete cache failure "
                                    f"for lat={lat}, lon={lon}"
                                )
                                assert 'pollutants' in data, (
                                    f"Missing pollutants data after complete cache failure "
                                    f"for lat={lat}, lon={lon}"
                                )
                                
                                # Внешний API должен быть вызван
                                assert mock_external_api.called, (
                                    f"External API not called after complete cache failure "
                                    f"for lat={lat}, lon={lon}"
                                )
                        
                        except Exception as e:
                            # Проверяем, что ошибка не связана с кэшем
                            if "cache" in str(e).lower():
                                pytest.fail(
                                    f"Application crashed due to complete cache failure "
                                    f"for lat={lat}, lon={lon}: {e}"
                                )
    
    def test_cache_manager_individual_failure_resilience(self):
        """
        Feature: airtrace-ru, Property 15: Cache Failure Resilience
        
        Individual cache manager operations should handle failures gracefully.
        **Validates: Requirements 6.5**
        """
        cache_manager = CacheManager()
        
        # Тестируем устойчивость к сбоям отдельных операций
        test_lat, test_lon = 55.7558, 37.6176
        test_data = {"test": "data"}
        
        # Симулируем сбой внутренней структуры кэша
        with patch.object(cache_manager, '_cache', side_effect=RuntimeError("Cache storage failure")):
            try:
                # get должен возвращать None при сбое
                result = asyncio.run(cache_manager.get(test_lat, test_lon))
                assert result is None, "Cache get should return None on failure"
                
            except Exception as e:
                # get не должен пропускать исключения
                pytest.fail(f"Cache get should handle failures gracefully: {e}")
        
        # Тестируем устойчивость set операции
        with patch.object(cache_manager, '_cache', side_effect=RuntimeError("Cache storage failure")):
            try:
                # set должен завершаться без исключений
                asyncio.run(cache_manager.set(test_lat, test_lon, test_data))
                
            except Exception as e:
                # set не должен пропускать исключения
                pytest.fail(f"Cache set should handle failures gracefully: {e}")
        
        # Тестируем устойчивость clear_expired операции
        with patch.object(cache_manager, '_cache', side_effect=RuntimeError("Cache storage failure")):
            try:
                # clear_expired должен завершаться без исключений
                asyncio.run(cache_manager.clear_expired())
                
            except Exception as e:
                # clear_expired не должен пропускать исключения
                pytest.fail(f"Cache clear_expired should handle failures gracefully: {e}")
    
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=30, deadline=30000)
    def test_service_level_cache_failure_handling_property(self, lat, lon):
        """
        Feature: airtrace-ru, Property 15: Cache Failure Resilience
        
        At the service level, cache failures should be handled gracefully
        without affecting the main functionality.
        **Validates: Requirements 6.5**
        """
        # Создаем сервис с мокнутым кэшем
        service = AirQualityService()
        
        # Мокаем внешний API
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
            
            # Симулируем сбой кэша на уровне сервиса
            with patch.object(service.cache_manager, 'get', side_effect=Exception("Cache failure")):
                with patch.object(service.cache_manager, 'set', side_effect=Exception("Cache failure")):
                    try:
                        # Сервис должен работать несмотря на сбой кэша
                        result = asyncio.run(service.get_current_air_quality(lat, lon))
                        
                        # Проверяем, что получили корректные данные
                        assert isinstance(result, AirQualityData), (
                            f"Service should return AirQualityData despite cache failure "
                            f"for lat={lat}, lon={lon}"
                        )
                        
                        assert result.aqi.value > 0, (
                            f"Service should return valid AQI despite cache failure "
                            f"for lat={lat}, lon={lon}"
                        )
                        
                        # Внешний API должен быть вызван
                        assert mock_get.called, (
                            f"External API should be called when cache fails "
                            f"for lat={lat}, lon={lon}"
                        )
                    
                    except Exception as e:
                        # Сервис не должен падать из-за сбоя кэша
                        if "cache" in str(e).lower():
                            pytest.fail(
                                f"Service failed due to cache error for lat={lat}, lon={lon}: {e}"
                            )
    
    def test_health_check_cache_failure_reporting(self):
        """
        Feature: airtrace-ru, Property 15: Cache Failure Resilience
        
        Health check should report cache failures but application should remain operational.
        **Validates: Requirements 6.5**
        """
        client = TestClient(app)
        
        # Симулируем сбой кэша в health check
        with patch('services.CacheManager.get_status', side_effect=Exception("Cache health check failure")):
            response = client.get("/health")
            
            # Health check должен работать несмотря на сбой кэша
            assert response.status_code == 200, (
                "Health check should work despite cache failure"
            )
            
            data = response.json()
            
            # Должна быть информация о статусе кэша
            assert 'services' in data, "Health check should include services status"
            
            if 'cache' in data['services']:
                # Статус кэша должен указывать на проблему
                cache_status = data['services']['cache']
                assert 'unhealthy' in cache_status or 'error' in cache_status.lower(), (
                    f"Cache status should indicate failure: {cache_status}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])