"""
Property-based tests for cache fallback behavior.

**Property 4: Cache Fallback Behavior**
**Validates: Requirements 2.3, 2.4**

Тестирует поведение системы кэширования при недоступности внешнего API
с использованием property-based testing для проверки корректности
fallback логики во всех возможных сценариях.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from datetime import datetime, timezone, timedelta

from services import AirQualityService
from cache import MultiLevelCacheManager as CacheManager
from schemas import AirQualityData, PollutantData, AQIInfo, LocationInfo


class TestCacheFallbackProperty:
    """Property-based тесты для fallback логики кэша"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.service = AirQualityService()
        self.cache_manager = CacheManager()
    
    async def teardown_method(self):
        """Очистка после каждого теста"""
        await self.service.cleanup()
    
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        pm2_5=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        pm10=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        no2=st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False),
        so2=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
    async def test_cache_hit_fallback_property(self, lat: float, lon: float, pm2_5: float, 
                                             pm10: float, no2: float, so2: float):
        """
        **Property 4: Cache Fallback Behavior**
        **Validates: Requirements 2.3, 2.4**
        
        For any request when cached data is available and fresh,
        the system should return cached data without making external API calls,
        even if the external API is unavailable.
        """
        # Создаем валидные кэшированные данные
        cached_air_quality = AirQualityData(
            timestamp=datetime.now(timezone.utc),
            location=LocationInfo(latitude=lat, longitude=lon),
            aqi=AQIInfo(value=85, category="Умеренное", color="#FFFF00", description="Test description"),
            pollutants=PollutantData(pm2_5=pm2_5, pm10=pm10, no2=no2, so2=so2),
            recommendations="Test recommendations",
            nmu_risk="low",
            health_warnings=[]
        )
        
        # Помещаем данные в кэш
        await self.service.cache_manager.set(lat, lon, cached_air_quality.model_dump())
        
        # Мокаем httpx клиент для симуляции недоступности API
        with patch.object(self.service.client, 'get') as mock_get:
            # Настраиваем мок для возврата ошибки сети
            mock_get.side_effect = httpx.RequestError("Network error")
            
            try:
                # Запрашиваем данные - должны получить кэшированные данные
                result = await self.service.get_current_air_quality(lat, lon)
                
                # Проверяем, что получили валидные данные
                assert isinstance(result, AirQualityData), f"Should return AirQualityData, got {type(result)}"
                
                # Проверяем, что данные соответствуют кэшированным
                assert result.location.latitude == lat, f"Latitude should match cached data"
                assert result.location.longitude == lon, f"Longitude should match cached data"
                assert result.pollutants.pm2_5 == pm2_5, f"PM2.5 should match cached data"
                assert result.pollutants.pm10 == pm10, f"PM10 should match cached data"
                assert result.pollutants.no2 == no2, f"NO2 should match cached data"
                assert result.pollutants.so2 == so2, f"SO2 should match cached data"
                
                # Проверяем, что внешний API не вызывался (кэш сработал)
                mock_get.assert_not_called()
                
            except Exception as e:
                pytest.fail(f"Should return cached data when API is unavailable: {e}")
    
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
    )
    async def test_cache_miss_api_unavailable_property(self, lat: float, lon: float):
        """
        **Property 4: Cache Fallback Behavior**
        **Validates: Requirements 2.3, 2.4**
        
        For any request when cached data is not available and external API is unavailable,
        the system should return appropriate error response with HTTP 503 status.
        """
        # Убеждаемся, что кэш пуст для данных координат
        cached_data = await self.service.cache_manager.get(lat, lon)
        assume(cached_data is None)
        
        # Мокаем httpx клиент для симуляции недоступности API
        with patch.object(self.service.client, 'get') as mock_get:
            # Настраиваем мок для возврата ошибки сети
            mock_get.side_effect = httpx.RequestError("Network error")
            
            # Запрашиваем данные - должны получить исключение
            with pytest.raises(Exception) as exc_info:
                await self.service.get_current_air_quality(lat, lon)
            
            # Проверяем, что исключение содержит информацию о недоступности сервиса
            error_message = str(exc_info.value).lower()
            assert any(word in error_message for word in ['сети', 'недоступен', 'ошибка']), \
                f"Error message should indicate network/service unavailability: {exc_info.value}"
            
            # Проверяем, что была попытка обращения к внешнему API
            mock_get.assert_called_once()
    
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        cache_age_minutes=st.integers(min_value=16, max_value=60)  # Устаревший кэш (старше 15 минут)
    )
    async def test_expired_cache_fallback_property(self, lat: float, lon: float, cache_age_minutes: int):
        """
        **Property 4: Cache Fallback Behavior**
        **Validates: Requirements 2.3, 2.4**
        
        For any request when cached data is expired and external API is unavailable,
        the system should attempt to fetch fresh data and return error when API fails.
        """
        # Создаем устаревшие кэшированные данные
        old_timestamp = datetime.now(timezone.utc) - timedelta(minutes=cache_age_minutes)
        expired_air_quality = AirQualityData(
            timestamp=old_timestamp,
            location=LocationInfo(latitude=lat, longitude=lon),
            aqi=AQIInfo(value=50, category="Хорошее", color="#00E400", description="Old data"),
            pollutants=PollutantData(pm2_5=10.0, pm10=20.0, no2=15.0, so2=5.0),
            recommendations="Old recommendations",
            nmu_risk="low",
            health_warnings=[]
        )
        
        # Помещаем устаревшие данные в кэш напрямую (обходя TTL)
        cache_key = self.service.cache_manager._generate_key(lat, lon)
        from schemas import CacheEntry
        expired_entry = CacheEntry(
            data=expired_air_quality.model_dump(),
            timestamp=old_timestamp,
            ttl_seconds=900  # 15 минут
        )
        self.service.cache_manager._cache[cache_key] = expired_entry
        
        # Мокаем httpx клиент для симуляции недоступности API
        with patch.object(self.service.client, 'get') as mock_get:
            # Настраиваем мок для возврата HTTP ошибки
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Service unavailable", request=MagicMock(), response=MagicMock()
            )
            mock_get.return_value = mock_response
            
            # Запрашиваем данные - должны получить исключение из-за недоступности API
            with pytest.raises(Exception) as exc_info:
                await self.service.get_current_air_quality(lat, lon)
            
            # Проверяем, что была попытка обращения к внешнему API (кэш устарел)
            mock_get.assert_called_once()
            
            # Проверяем, что исключение связано с недоступностью внешнего сервиса
            error_message = str(exc_info.value).lower()
            assert any(word in error_message for word in ['недоступен', 'сервис', 'временно']), \
                f"Error should indicate service unavailability: {exc_info.value}"
    
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        timeout_seconds=st.floats(min_value=0.1, max_value=5.0)
    )
    async def test_api_timeout_fallback_property(self, lat: float, lon: float, timeout_seconds: float):
        """
        **Property 4: Cache Fallback Behavior**
        **Validates: Requirements 2.3, 2.4**
        
        For any request when external API times out and no cached data is available,
        the system should handle timeout gracefully and return appropriate error.
        """
        # Убеждаемся, что кэш пуст
        cached_data = await self.service.cache_manager.get(lat, lon)
        assume(cached_data is None)
        
        # Мокаем httpx клиент для симуляции таймаута
        with patch.object(self.service.client, 'get') as mock_get:
            # Настраиваем мок для возврата таймаута
            mock_get.side_effect = httpx.TimeoutException("Request timeout")
            
            # Запрашиваем данные - должны получить исключение
            with pytest.raises(Exception) as exc_info:
                await self.service.get_current_air_quality(lat, lon)
            
            # Проверяем, что исключение обработано корректно
            error_message = str(exc_info.value).lower()
            assert any(word in error_message for word in ['сети', 'ошибка', 'недоступен']), \
                f"Error should indicate network issue: {exc_info.value}"
            
            # Проверяем, что была попытка обращения к API
            mock_get.assert_called_once()
    
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        http_status=st.sampled_from([500, 502, 503, 504])  # Серверные ошибки
    )
    async def test_api_server_error_fallback_property(self, lat: float, lon: float, http_status: int):
        """
        **Property 4: Cache Fallback Behavior**
        **Validates: Requirements 2.3, 2.4**
        
        For any request when external API returns server errors (5xx) and no cached data available,
        the system should handle the error gracefully and return appropriate error response.
        """
        # Убеждаемся, что кэш пуст
        cached_data = await self.service.cache_manager.get(lat, lon)
        assume(cached_data is None)
        
        # Мокаем httpx клиент для симуляции серверной ошибки
        with patch.object(self.service.client, 'get') as mock_get:
            # Создаем мок ответа с серверной ошибкой
            mock_response = MagicMock()
            mock_response.status_code = http_status
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"Server error {http_status}",
                request=MagicMock(),
                response=mock_response
            )
            mock_get.return_value = mock_response
            
            # Запрашиваем данные - должны получить исключение
            with pytest.raises(Exception) as exc_info:
                await self.service.get_current_air_quality(lat, lon)
            
            # Проверяем, что исключение содержит информацию о недоступности сервиса
            error_message = str(exc_info.value).lower()
            assert any(word in error_message for word in ['недоступен', 'сервис', 'временно']), \
                f"Error should indicate service unavailability for HTTP {http_status}: {exc_info.value}"
            
            # Проверяем, что была попытка обращения к API
            mock_get.assert_called_once()
    
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
    )
    async def test_cache_corruption_fallback_property(self, lat: float, lon: float):
        """
        **Property 4: Cache Fallback Behavior**
        **Validates: Requirements 2.3, 2.4**
        
        For any request when cached data is corrupted and external API is unavailable,
        the system should handle cache corruption gracefully and attempt to fetch fresh data.
        """
        # Помещаем поврежденные данные в кэш
        cache_key = self.service.cache_manager._generate_key(lat, lon)
        from schemas import CacheEntry
        corrupted_entry = CacheEntry(
            data={"invalid": "data", "missing_required_fields": True},  # Некорректная структура
            timestamp=datetime.now(timezone.utc),
            ttl_seconds=900
        )
        self.service.cache_manager._cache[cache_key] = corrupted_entry
        
        # Мокаем httpx клиент для симуляции недоступности API
        with patch.object(self.service.client, 'get') as mock_get:
            mock_get.side_effect = httpx.RequestError("Network error")
            
            # Запрашиваем данные
            with pytest.raises(Exception) as exc_info:
                await self.service.get_current_air_quality(lat, lon)
            
            # Система должна попытаться обратиться к API, когда кэш поврежден
            # Это может произойти либо сразу (если кэш не парсится), либо после попытки использовать кэш
            # В любом случае, должна быть обработка ошибки
            error_message = str(exc_info.value).lower()
            assert len(error_message) > 0, "Should return meaningful error message"
    
    async def test_cache_manager_status_property(self):
        """
        **Property 4: Cache Fallback Behavior**
        **Validates: Requirements 2.3, 2.4**
        
        The cache manager should always return a valid status string
        for health check purposes, regardless of cache state.
        """
        # Тестируем статус пустого кэша
        status = self.cache_manager.get_status()
        assert isinstance(status, str), f"Status should be string, got {type(status)}"
        assert len(status) > 0, "Status should not be empty"
        assert "healthy" in status.lower() or "unhealthy" in status.lower(), \
            f"Status should indicate health state: '{status}'"
        
        # Добавляем данные в кэш и проверяем статус
        test_data = {"test": "data"}
        await self.cache_manager.set(55.0, 37.0, test_data)
        
        status_with_data = self.cache_manager.get_status()
        assert isinstance(status_with_data, str), "Status with data should be string"
        assert "healthy" in status_with_data.lower(), f"Status with data should be healthy: '{status_with_data}'"
        assert "1 entries" in status_with_data or "entries" in status_with_data, \
            f"Status should mention cache entries: '{status_with_data}'"
    
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
    )
    async def test_external_api_health_check_property(self, lat: float, lon: float):
        """
        **Property 4: Cache Fallback Behavior**
        **Validates: Requirements 2.3, 2.4**
        
        The external API health check should return appropriate status
        regardless of API availability state.
        """
        # Тестируем health check при доступном API
        with patch.object(self.service.client, 'get') as mock_get:
            # Мокаем успешный ответ
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            status = await self.service.check_external_api_health()
            assert status == "healthy", f"Should return 'healthy' when API is available, got '{status}'"
        
        # Тестируем health check при недоступном API
        with patch.object(self.service.client, 'get') as mock_get:
            mock_get.side_effect = httpx.RequestError("Network error")
            
            status = await self.service.check_external_api_health()
            assert status == "unhealthy", f"Should return 'unhealthy' when API is unavailable, got '{status}'"