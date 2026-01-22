"""
Property-based тесты для приватности ключей кэша AirTrace RU Backend

Тестирует Property 14: Cache Key Privacy
Validates: Requirements 6.4
"""

import pytest
import re
import hashlib
from hypothesis import given, strategies as st, settings
from typing import Set

from cache import MultiLevelCacheManager as CacheManager


class TestCacheKeyPrivacyProperty:
    """
    Property 14: Cache Key Privacy
    
    For any cache operations, geographic coordinates should be used as cache keys 
    without permanent storage of the actual coordinate values.
    
    **Validates: Requirements 6.4**
    """

    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=3000)
    async def test_cache_keys_do_not_contain_raw_coordinates(self, lat, lon):
        """
        Property 14: Cache Key Privacy
        
        Тестирует, что ключи кэша не содержат исходные координаты в читаемом виде.
        **Validates: Requirements 6.4**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        test_data = {"aqi": 50, "category": "good"}
        
        # Сохранение данных в кэш
        await cache_manager.set(lat, lon, test_data)
        
        # Получение ключа кэша
        cache_key = cache_manager._generate_key(lat, lon)
        
        # Проверка, что ключ не содержит исходные координаты
        lat_str = str(lat)
        lon_str = str(lon)
        
        assert lat_str not in cache_key, f"Cache key contains raw latitude: {lat_str}"
        assert lon_str not in cache_key, f"Cache key contains raw longitude: {lon_str}"
        
        # Проверка, что ключ не содержит округленные координаты
        rounded_lat = str(round(lat, 2))
        rounded_lon = str(round(lon, 2))
        
        assert rounded_lat not in cache_key, f"Cache key contains rounded latitude: {rounded_lat}"
        assert rounded_lon not in cache_key, f"Cache key contains rounded longitude: {rounded_lon}"
        
        # Проверка, что ключ выглядит как хэш (только hex символы)
        assert re.match(r'^[a-f0-9]+$', cache_key), f"Cache key should be hexadecimal hash: {cache_key}"
        
        # Проверка длины ключа (MD5 хэш должен быть 32 символа)
        assert len(cache_key) == 32, f"Cache key should be 32 characters (MD5 hash): {len(cache_key)}"

    @given(
        coordinates_list=st.lists(
            st.tuples(
                st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
            ),
            min_size=2,
            max_size=10
        ).filter(lambda coords: len(set((round(lat, 2), round(lon, 2)) for lat, lon in coords)) == len(coords))
    )
    @settings(max_examples=50, deadline=5000)
    async def test_cache_keys_are_deterministic_and_unique(self, coordinates_list):
        """
        Property 14: Cache Key Privacy
        
        Тестирует, что ключи кэша детерминированы и уникальны для координат с разными округленными значениями.
        **Validates: Requirements 6.4**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        
        # Генерация ключей для всех координат
        cache_keys = []
        for lat, lon in coordinates_list:
            key = cache_manager._generate_key(lat, lon)
            cache_keys.append(key)
        
        # Проверка уникальности ключей (учитывая, что координаты округляются до 2 знаков)
        unique_keys = set(cache_keys)
        assert len(unique_keys) == len(cache_keys), f"Cache keys should be unique for coordinates with different rounded values. Got {len(unique_keys)} unique keys for {len(cache_keys)} coordinates"
        
        # Проверка детерминированности - повторная генерация должна дать те же ключи
        for i, (lat, lon) in enumerate(coordinates_list):
            key_again = cache_manager._generate_key(lat, lon)
            assert key_again == cache_keys[i], f"Cache key generation should be deterministic"

    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=3000)
    async def test_cache_key_generation_is_privacy_safe(self, lat, lon):
        """
        Property 14: Cache Key Privacy
        
        Тестирует, что генерация ключей кэша безопасна с точки зрения приватности.
        **Validates: Requirements 6.4**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        
        # Генерация ключа
        cache_key = cache_manager._generate_key(lat, lon)
        
        # Проверка, что ключ является результатом хэширования
        # Воссоздаем процесс генерации ключа для проверки
        rounded_lat = round(lat, 2)
        rounded_lon = round(lon, 2)
        expected_key = hashlib.md5(f"{rounded_lat}:{rounded_lon}".encode()).hexdigest()
        
        assert cache_key == expected_key, "Cache key should be MD5 hash of rounded coordinates"
        
        # Проверка, что обратное восстановление координат невозможно
        # (это концептуальная проверка - MD5 необратим)
        assert len(cache_key) == 32, "MD5 hash should be 32 characters"
        assert cache_key.isalnum(), "Hash should contain only alphanumeric characters"

    @given(
        lat1=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon1=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
        lat2=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon2=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=3000)
    async def test_similar_coordinates_have_different_keys(self, lat1, lon1, lat2, lon2):
        """
        Property 14: Cache Key Privacy
        
        Тестирует, что даже похожие координаты генерируют разные ключи кэша.
        **Validates: Requirements 6.4**
        """
        # Пропускаем случаи, когда координаты одинаковые после округления
        rounded_lat1, rounded_lon1 = round(lat1, 2), round(lon1, 2)
        rounded_lat2, rounded_lon2 = round(lat2, 2), round(lon2, 2)
        
        if rounded_lat1 == rounded_lat2 and rounded_lon1 == rounded_lon2:
            return  # Пропускаем этот случай
        
        cache_manager = CacheManager(ttl_minutes=15)
        
        # Генерация ключей для обеих пар координат
        key1 = cache_manager._generate_key(lat1, lon1)
        key2 = cache_manager._generate_key(lat2, lon2)
        
        # Проверка, что ключи разные
        assert key1 != key2, f"Different coordinates should generate different cache keys"
        
        # Проверка, что оба ключа являются валидными хэшами
        assert len(key1) == 32 and len(key2) == 32, "Both keys should be valid MD5 hashes"
        assert re.match(r'^[a-f0-9]+$', key1), f"Key1 should be hexadecimal: {key1}"
        assert re.match(r'^[a-f0-9]+$', key2), f"Key2 should be hexadecimal: {key2}"

    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.integers(min_value=0, max_value=100),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=50, deadline=5000)
    async def test_cache_operations_do_not_expose_coordinates(self, lat, lon, data):
        """
        Property 14: Cache Key Privacy
        
        Тестирует, что операции с кэшем не раскрывают исходные координаты.
        **Validates: Requirements 6.4**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        
        # Сохранение данных в кэш
        await cache_manager.set(lat, lon, data)
        
        # Получение данных из кэша
        cached_data = await cache_manager.get(lat, lon)
        assert cached_data == data
        
        # Проверка внутреннего состояния кэша
        assert len(cache_manager._cache) == 1, "Should have exactly one cache entry"
        
        # Получение единственного ключа из кэша
        cache_key = list(cache_manager._cache.keys())[0]
        
        # Проверка, что ключ не содержит координаты
        lat_variations = [str(lat), str(round(lat, 2)), f"{lat:.2f}", f"{lat:.1f}"]
        lon_variations = [str(lon), str(round(lon, 2)), f"{lon:.2f}", f"{lon:.1f}"]
        
        for lat_var in lat_variations:
            assert lat_var not in cache_key, f"Cache key contains latitude variation: {lat_var}"
        
        for lon_var in lon_variations:
            assert lon_var not in cache_key, f"Cache key contains longitude variation: {lon_var}"
        
        # Проверка, что данные в кэше не содержат координаты
        cache_entry = cache_manager._cache[cache_key]
        cache_data_str = str(cache_entry.data)
        
        # Координаты не должны попасть в кэшированные данные
        for lat_var in lat_variations:
            if lat_var and lat_var != "0" and lat_var != "0.0":  # Исключаем общие значения
                assert lat_var not in cache_data_str, f"Cached data contains latitude: {lat_var}"
        
        for lon_var in lon_variations:
            if lon_var and lon_var != "0" and lon_var != "0.0":  # Исключаем общие значения
                assert lon_var not in cache_data_str, f"Cached data contains longitude: {lon_var}"

    async def test_coordinate_rounding_for_privacy(self):
        """
        Property 14: Cache Key Privacy
        
        Тестирует, что координаты округляются для группировки близких запросов.
        **Validates: Requirements 6.4**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        
        # Тестирование близких координат, которые должны давать один ключ
        lat_base, lon_base = 55.7558, 37.6176
        
        # Координаты с небольшими различиями (в пределах точности округления)
        coordinates_same_key = [
            (lat_base, lon_base),
            (lat_base + 0.001, lon_base),  # Различие меньше 0.01
            (lat_base, lon_base + 0.001),
            (lat_base + 0.004, lon_base + 0.004)
        ]
        
        # Все эти координаты должны давать один ключ
        keys = [cache_manager._generate_key(lat, lon) for lat, lon in coordinates_same_key]
        unique_keys = set(keys)
        
        assert len(unique_keys) == 1, f"Close coordinates should generate same cache key, got {len(unique_keys)} keys"
        
        # Координаты с большими различиями должны давать разные ключи
        coordinates_different_keys = [
            (lat_base, lon_base),
            (lat_base + 0.01, lon_base),  # Различие равно точности округления
            (lat_base, lon_base + 0.01),
            (lat_base + 0.02, lon_base + 0.02)
        ]
        
        keys_different = [cache_manager._generate_key(lat, lon) for lat, lon in coordinates_different_keys]
        unique_keys_different = set(keys_different)
        
        assert len(unique_keys_different) > 1, "Coordinates with significant differences should generate different keys"

    async def test_cache_key_collision_resistance(self):
        """
        Property 14: Cache Key Privacy
        
        Тестирует устойчивость к коллизиям ключей кэша.
        **Validates: Requirements 6.4**
        """
        cache_manager = CacheManager(ttl_minutes=15)
        
        # Генерация большого количества ключей для разных координат
        keys: Set[str] = set()
        coordinate_pairs = []
        
        # Создание сетки координат
        for lat in range(-90, 91, 10):  # Каждые 10 градусов
            for lon in range(-180, 181, 10):
                coordinate_pairs.append((float(lat), float(lon)))
        
        # Генерация ключей
        for lat, lon in coordinate_pairs:
            key = cache_manager._generate_key(lat, lon)
            keys.add(key)
        
        # Проверка отсутствия коллизий
        assert len(keys) == len(coordinate_pairs), f"Key collision detected: {len(keys)} unique keys for {len(coordinate_pairs)} coordinates"
        
        # Проверка формата всех ключей
        for key in keys:
            assert len(key) == 32, f"Invalid key length: {len(key)}"
            assert re.match(r'^[a-f0-9]+$', key), f"Invalid key format: {key}"


# Запуск тестов с pytest-asyncio
if __name__ == "__main__":
    pytest.main([__file__, "-v"])