"""
Property-based тесты для жизненного цикла кэша AirTrace RU Backend

Тестирует Property 13: Cache Lifecycle Management
Validates: Requirements 6.1, 6.2, 6.3
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from hypothesis import given, strategies as st, settings
from typing import Dict, Any

from services import CacheManager
from schemas import CacheEntry


class TestCacheLifecycleProperty:
    """
    Property 13: Cache Lifecycle Management
    
    For any geographic location, air quality data should be cached for exactly 15 minutes,
    with fresh data automatically fetched when cache expires.
    
    **Validates: Requirements 6.1, 6.2, 6.3**
    """

    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.integers(min_value=0, max_value=500),
                st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
                st.text(min_size=1, max_size=100)
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=5000)
    async def test_cache_stores_data_for_15_minutes(self, lat, lon, data):
        """
        Property 13: Cache Lifecycle Management
        
        Тестирует, что данные кэшируются ровно на 15 минут для любой географической локации.
        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        # Сохранение данных в кэш
        await cache_manager.set(lat, lon, data)
        
        # Проверка, что данные доступны сразу после сохранения
        cached_data = await cache_manager.get(lat, lon)
        assert cached_data is not None
        assert cached_data == data
        
        # Проверка TTL записи в кэше
        key = cache_manager._generate_key(lat, lon)
        assert key in cache_manager._cache
        
        cache_entry = cache_manager._cache[key]
        assert isinstance(cache_entry, CacheEntry)
        assert cache_entry.ttl_seconds == 15 * 60  # 15 минут в секундах
        assert cache_entry.data == data
        
        # Проверка, что запись не истекла сразу после создания
        assert not cache_entry.is_expired()

    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.integers(min_value=0, max_value=500),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=50, deadline=10000)
    async def test_cache_expiration_behavior(self, lat, lon, data):
        """
        Property 13: Cache Lifecycle Management
        
        Тестирует автоматическое истечение срока действия кэша через 15 минут.
        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        # Сохранение данных в кэш
        await cache_manager.set(lat, lon, data)
        
        # Получение ключа и записи кэша
        key = cache_manager._generate_key(lat, lon)
        cache_entry = cache_manager._cache[key]
        
        # Симуляция истечения времени путем изменения timestamp
        expired_timestamp = datetime.now(timezone.utc) - timedelta(minutes=16)
        cache_entry.timestamp = expired_timestamp
        
        # Проверка, что запись помечена как истекшая
        assert cache_entry.is_expired()
        
        # Проверка, что get() возвращает None для истекших данных
        cached_data = await cache_manager.get(lat, lon)
        assert cached_data is None
        
        # Проверка, что истекшая запись удалена из кэша
        assert key not in cache_manager._cache

    @given(
        coordinates_list=st.lists(
            st.tuples(
                st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
            ),
            min_size=1,
            max_size=10
        ).filter(lambda coords: len(set((round(lat, 2), round(lon, 2)) for lat, lon in coords)) == len(coords)),
        data_list=st.lists(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.integers(min_value=0, max_value=100),
                min_size=1,
                max_size=3
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=30, deadline=10000)
    async def test_multiple_locations_cache_independently(self, coordinates_list, data_list):
        """
        Property 13: Cache Lifecycle Management
        
        Тестирует, что разные географические локации кэшируются независимо.
        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        # Обеспечиваем одинаковое количество координат и данных
        min_length = min(len(coordinates_list), len(data_list))
        coordinates_list = coordinates_list[:min_length]
        data_list = data_list[:min_length]
        
        # Сохранение данных для всех локаций
        for (lat, lon), data in zip(coordinates_list, data_list):
            await cache_manager.set(lat, lon, data)
        
        # Проверка, что все данные доступны
        for (lat, lon), expected_data in zip(coordinates_list, data_list):
            cached_data = await cache_manager.get(lat, lon)
            assert cached_data is not None
            assert cached_data == expected_data
        
        # Проверка, что каждая локация имеет уникальный ключ кэша (с учетом округления)
        cache_keys = set()
        for lat, lon in coordinates_list:
            key = cache_manager._generate_key(lat, lon)
            assert key not in cache_keys, f"Cache keys should be unique for locations with different rounded coordinates"
            cache_keys.add(key)
        
        # Проверка количества записей в кэше
        assert len(cache_manager._cache) == len(coordinates_list)

    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
        data_pair=st.tuples(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.integers(min_value=0, max_value=100),
                min_size=1,
                max_size=3
            ),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.integers(min_value=100, max_value=200),
                min_size=1,
                max_size=3
            )
        ).filter(lambda pair: pair[0] != pair[1])
    )
    @settings(max_examples=50, deadline=5000)
    async def test_cache_update_behavior(self, lat, lon, data_pair):
        """
        Property 13: Cache Lifecycle Management
        
        Тестирует поведение кэша при обновлении данных для той же локации.
        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        initial_data, updated_data = data_pair
        cache_manager = CacheManager(ttl_minutes=15)
        # Сохранение первоначальных данных
        await cache_manager.set(lat, lon, initial_data)
        
        # Проверка первоначальных данных
        cached_data = await cache_manager.get(lat, lon)
        assert cached_data == initial_data
        
        # Обновление данных для той же локации
        await cache_manager.set(lat, lon, updated_data)
        
        # Проверка, что данные обновились
        cached_data = await cache_manager.get(lat, lon)
        assert cached_data == updated_data
        assert cached_data != initial_data
        
        # Проверка, что в кэше только одна запись для данной локации
        key = cache_manager._generate_key(lat, lon)
        assert key in cache_manager._cache
        
        # Подсчет записей с тем же ключом (должна быть только одна)
        matching_keys = [k for k in cache_manager._cache.keys() if k == key]
        assert len(matching_keys) == 1

    @given(
        coordinates_list=st.lists(
            st.tuples(
                st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
            ),
            min_size=2,
            max_size=5
        ).filter(lambda coords: len(set((round(lat, 2), round(lon, 2)) for lat, lon in coords)) == len(coords))
    )
    @settings(max_examples=20, deadline=10000)
    async def test_automatic_expired_cleanup(self, coordinates_list):
        """
        Property 13: Cache Lifecycle Management
        
        Тестирует автоматическую очистку истекших записей кэша.
        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        test_data = {"aqi": 50, "category": "good"}
        
        # Сохранение данных для всех локаций
        for lat, lon in coordinates_list:
            await cache_manager.set(lat, lon, test_data)
        
        # Проверка, что все данные сохранены (с учетом округления координат)
        assert len(cache_manager._cache) == len(coordinates_list)
        
        # Симуляция истечения времени для половины записей
        expired_count = len(coordinates_list) // 2
        cache_keys = list(cache_manager._cache.keys())
        
        for i in range(expired_count):
            key = cache_keys[i]
            cache_entry = cache_manager._cache[key]
            cache_entry.timestamp = datetime.now(timezone.utc) - timedelta(minutes=16)
        
        # Выполнение очистки истекших записей
        await cache_manager.clear_expired()
        
        # Проверка, что истекшие записи удалены
        remaining_count = len(coordinates_list) - expired_count
        assert len(cache_manager._cache) == remaining_count
        
        # Проверка, что оставшиеся записи не истекли
        for cache_entry in cache_manager._cache.values():
            assert not cache_entry.is_expired()

    async def test_cache_miss_returns_none(self):
        """
        Property 13: Cache Lifecycle Management
        
        Тестирует, что cache miss возвращает None для любых координат.
        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        # Попытка получить данные из пустого кэша
        cached_data = await cache_manager.get(55.7558, 37.6176)
        assert cached_data is None
        
        # Попытка получить данные для несуществующих координат
        cached_data = await cache_manager.get(0.0, 0.0)
        assert cached_data is None

    async def test_cache_status_reporting(self):
        """
        Property 13: Cache Lifecycle Management
        
        Тестирует корректность отчета о статусе кэша.
        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        # Проверка статуса пустого кэша
        status = cache_manager.get_status()
        assert "healthy" in status
        assert "0 entries" in status
        
        # Добавление данных в кэш
        test_data = {"test": "data"}
        await cache_manager.set(55.7558, 37.6176, test_data)
        await cache_manager.set(59.9311, 30.3609, test_data)
        
        # Проверка статуса с данными
        status = cache_manager.get_status()
        assert "healthy" in status
        assert "2 entries" in status


# Запуск тестов с pytest-asyncio
if __name__ == "__main__":
    pytest.main([__file__, "-v"])