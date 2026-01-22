"""
Property-based tests for cache operation consistency

Tests Property 1: Cache Operation Consistency
Validates: Requirements 1.2, 1.3, 1.5, 1.6, 1.7

**Property 1: Cache Operation Consistency**
For any cache key and data value, storing data in the cache and then retrieving it 
should return the same value if within TTL, and cache operations should maintain 
consistency across distributed instances.
"""

import asyncio
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import Any, Dict, List
import json
import time

from cache import MultiLevelCacheManager, CacheLevel
from config import config


class TestCacheOperationConsistency:
    """Property-based tests for cache operation consistency"""

    async def create_cache_manager(self):
        """Create a fresh cache manager for testing"""
        manager = MultiLevelCacheManager()
        # Disable L2 (Redis) for testing to avoid connection issues
        manager._l2_enabled = False
        manager._redis_enabled = False
        # Disable L3 for testing
        manager._l3_enabled = False
        return manager

    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.one_of(
                st.text(max_size=100),
                st.integers(min_value=-1000, max_value=1000),
                st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
                st.booleans()
            ),
            min_size=1,
            max_size=10
        ),
        ttl=st.integers(min_value=1, max_value=3600)
    )
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_cache_set_get_consistency(self, lat, lon, data, ttl):
        """
        **Property 1: Cache Operation Consistency**
        
        For any valid coordinates and data, storing data in cache and immediately 
        retrieving it should return the same data.
        
        **Validates: Requirements 1.2, 1.3, 1.5, 1.6, 1.7**
        """
        cache_manager = await self.create_cache_manager()
        try:
            # Ensure data is JSON serializable (for Redis compatibility)
            try:
                json.dumps(data)
            except (TypeError, ValueError):
                assume(False)  # Skip non-serializable data
            
            # Store data in cache
            success = await cache_manager.set(lat, lon, data, ttl)
            assert success, "Cache set operation should succeed"
            
            # Retrieve data immediately
            retrieved_data = await cache_manager.get(lat, lon)
            
            # Verify consistency
            assert retrieved_data is not None, "Data should be retrievable immediately after setting"
            assert retrieved_data == data, "Retrieved data should match stored data exactly"
        finally:
            await cache_manager.cleanup()

    @given(
        coordinates=st.lists(
            st.tuples(
                st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False).filter(
                    lambda x: abs(x) > 1e-3 or x == 0.0  # Ensure coordinates are either 0 or significantly different
                ),
                st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False).filter(
                    lambda x: abs(x) > 1e-3 or x == 0.0  # Ensure coordinates are either 0 or significantly different
                )
            ),
            min_size=1,
            max_size=10,
            unique_by=lambda coord: (round(coord[0], 3), round(coord[1], 3))  # Ensure unique after rounding
        ),
        data_values=st.lists(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20),
                values=st.one_of(st.text(max_size=50), st.integers(min_value=0, max_value=1000)),
                min_size=1,
                max_size=5
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_cache_multiple_entries_consistency(self, coordinates, data_values):
        """
        **Property 1: Cache Operation Consistency**
        
        For multiple cache entries, each should maintain consistency independently.
        
        **Validates: Requirements 1.2, 1.3, 1.5, 1.6, 1.7**
        """
        cache_manager = await self.create_cache_manager()
        try:
            # Ensure we have matching coordinates and data
            min_length = min(len(coordinates), len(data_values))
            coordinates = coordinates[:min_length]
            data_values = data_values[:min_length]
            
            # Ensure all data is JSON serializable
            for data in data_values:
                try:
                    json.dumps(data)
                except (TypeError, ValueError):
                    assume(False)
            
            # Store all entries
            for (lat, lon), data in zip(coordinates, data_values):
                success = await cache_manager.set(lat, lon, data, ttl=300)
                assert success, f"Cache set should succeed for coordinates ({lat}, {lon})"
            
            # Verify all entries can be retrieved consistently
            for (lat, lon), expected_data in zip(coordinates, data_values):
                retrieved_data = await cache_manager.get(lat, lon)
                assert retrieved_data is not None, f"Data should be retrievable for coordinates ({lat}, {lon})"
                assert retrieved_data == expected_data, f"Retrieved data should match stored data for coordinates ({lat}, {lon})"
        finally:
            await cache_manager.cleanup()

    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        initial_data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.text(max_size=50),
            min_size=1,
            max_size=5
        ),
        updated_data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.text(max_size=50),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=30, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_cache_update_consistency(self, lat, lon, initial_data, updated_data):
        """
        **Property 1: Cache Operation Consistency**
        
        When cache data is updated, the new data should be consistently retrievable.
        
        **Validates: Requirements 1.2, 1.3, 1.5, 1.6, 1.7**
        """
        cache_manager = await self.create_cache_manager()
        try:
            assume(initial_data != updated_data)  # Ensure we're actually updating
            
            # Store initial data
            success = await cache_manager.set(lat, lon, initial_data, ttl=300)
            assert success, "Initial cache set should succeed"
            
            # Verify initial data
            retrieved_data = await cache_manager.get(lat, lon)
            assert retrieved_data == initial_data, "Initial data should be retrievable"
            
            # Update with new data
            success = await cache_manager.set(lat, lon, updated_data, ttl=300)
            assert success, "Cache update should succeed"
            
            # Verify updated data
            retrieved_data = await cache_manager.get(lat, lon)
            assert retrieved_data == updated_data, "Updated data should be retrievable"
            assert retrieved_data != initial_data, "Retrieved data should be the updated version"
        finally:
            await cache_manager.cleanup()

    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.text(max_size=50),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=30, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_cache_key_privacy_consistency(self, lat, lon, data):
        """
        **Property 1: Cache Operation Consistency**
        
        Cache keys should be generated consistently for the same coordinates
        while maintaining privacy (no raw coordinates stored).
        
        **Validates: Requirements 1.2, 1.3, 1.5, 1.6, 1.7**
        """
        cache_manager = await self.create_cache_manager()
        try:
            # Store data
            success = await cache_manager.set(lat, lon, data, ttl=300)
            assert success, "Cache set should succeed"
            
            # Generate key using the same method
            key1 = cache_manager._generate_key(lat, lon)
            key2 = cache_manager._generate_key(lat, lon)
            
            # Keys should be consistent
            assert key1 == key2, "Cache keys should be consistent for same coordinates"
            
            # Keys should not contain raw coordinates (privacy check)
            assert str(lat) not in key1, "Cache key should not contain raw latitude"
            assert str(lon) not in key1, "Cache key should not contain raw longitude"
            
            # Data should be retrievable using the same coordinates
            retrieved_data = await cache_manager.get(lat, lon)
            assert retrieved_data == data, "Data should be retrievable with consistent key generation"
        finally:
            await cache_manager.cleanup()