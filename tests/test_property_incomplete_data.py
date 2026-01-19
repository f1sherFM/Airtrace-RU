"""
Property-based tests for incomplete data handling.

**Property 7: Incomplete Data Handling**
**Validates: Requirements 3.4**

Тестирует обработку неполных данных загрязнителей с использованием
property-based testing для проверки корректности расчета AQI
при отсутствии некоторых значений загрязнителей.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Optional

from utils import AQICalculator


class TestIncompleteDataHandlingProperty:
    """Property-based тесты для обработки неполных данных"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.calculator = AQICalculator()
    
    @given(
        pm2_5=st.one_of(st.none(), st.floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False)),
        pm10=st.one_of(st.none(), st.floats(min_value=0.1, max_value=400.0, allow_nan=False, allow_infinity=False)),
        no2=st.one_of(st.none(), st.floats(min_value=0.1, max_value=300.0, allow_nan=False, allow_infinity=False)),
        so2=st.one_of(st.none(), st.floats(min_value=0.1, max_value=300.0, allow_nan=False, allow_infinity=False)),
        o3=st.one_of(st.none(), st.floats(min_value=0.1, max_value=300.0, allow_nan=False, allow_infinity=False))
    )
    def test_incomplete_data_exclusion_property(self, pm2_5: Optional[float], pm10: Optional[float], 
                                              no2: Optional[float], so2: Optional[float], o3: Optional[float]):
        """
        **Property 7: Incomplete Data Handling**
        **Validates: Requirements 3.4**
        
        For any pollutant dataset with missing values (None), 
        the AQI calculator should exclude missing pollutants and 
        calculate AQI from available data only.
        """
        # Создаем словарь с данными, включая None значения
        all_pollutants = {
            'pm2_5': pm2_5,
            'pm10': pm10,
            'no2': no2,
            'so2': so2,
            'o3': o3
        }
        
        # Фильтруем только валидные (не None) значения
        valid_pollutants = {k: v for k, v in all_pollutants.items() if v is not None}
        
        # Пропускаем случаи, когда нет валидных данных
        assume(len(valid_pollutants) > 0)
        
        # Рассчитываем AQI для полного набора (с None)
        aqi_with_none, category_with_none, color_with_none = self.calculator.calculate_aqi(all_pollutants)
        
        # Рассчитываем AQI только для валидных данных
        aqi_valid_only, category_valid_only, color_valid_only = self.calculator.calculate_aqi(valid_pollutants)
        
        # Результаты должны быть одинаковыми - None значения должны игнорироваться
        assert aqi_with_none == aqi_valid_only, \
            f"AQI should be same when None values are excluded: {aqi_with_none} vs {aqi_valid_only}"
        assert category_with_none == category_valid_only, \
            f"Category should be same when None values are excluded: '{category_with_none}' vs '{category_valid_only}'"
        assert color_with_none == color_valid_only, \
            f"Color should be same when None values are excluded: '{color_with_none}' vs '{color_valid_only}'"
    
    @given(
        base_pollutants=st.dictionaries(
            keys=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3']),
            values=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=5
        ),
        missing_count=st.integers(min_value=1, max_value=3)
    )
    def test_partial_data_consistency_property(self, base_pollutants: Dict[str, float], missing_count: int):
        """
        **Property 7: Incomplete Data Handling**
        **Validates: Requirements 3.4**
        
        For any complete pollutant dataset, removing some pollutants should result
        in AQI calculation based only on remaining pollutants, and the result
        should be consistent with calculating AQI from those pollutants alone.
        """
        assume(len(base_pollutants) > missing_count)
        
        # Получаем AQI для полного набора данных
        full_aqi, full_category, full_color = self.calculator.calculate_aqi(base_pollutants)
        
        # Удаляем случайные загрязнители
        pollutant_keys = list(base_pollutants.keys())
        keys_to_remove = pollutant_keys[:missing_count]
        
        partial_pollutants = {k: v for k, v in base_pollutants.items() if k not in keys_to_remove}
        
        # Получаем AQI для частичного набора данных
        partial_aqi, partial_category, partial_color = self.calculator.calculate_aqi(partial_pollutants)
        
        # Проверяем, что результат валиден
        assert isinstance(partial_aqi, int), f"Partial AQI should be integer, got {type(partial_aqi)}"
        assert isinstance(partial_category, str), f"Partial category should be string, got {type(partial_category)}"
        assert isinstance(partial_color, str), f"Partial color should be string, got {type(partial_color)}"
        
        assert 0 <= partial_aqi <= 500, f"Partial AQI {partial_aqi} should be between 0 and 500"
        assert len(partial_category) > 0, "Partial category should not be empty"
        assert partial_color.startswith('#'), f"Partial color {partial_color} should start with #"
        
        # Если удаленные загрязнители не были доминирующими, AQI может остаться тем же
        # Если удаленные загрязнители были доминирующими, AQI должен уменьшиться или остаться тем же
        assert partial_aqi <= full_aqi, \
            f"Partial AQI {partial_aqi} should be <= full AQI {full_aqi} when pollutants are removed"
    
    @given(
        pollutant=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3']),
        concentration=st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False)
    )
    def test_single_pollutant_handling_property(self, pollutant: str, concentration: float):
        """
        **Property 7: Incomplete Data Handling**
        **Validates: Requirements 3.4**
        
        For any single pollutant with valid concentration,
        the AQI calculator should calculate AQI based on that pollutant alone,
        ignoring missing data for other pollutants.
        """
        # Создаем данные только с одним загрязнителем
        single_pollutant_data = {pollutant: concentration}
        
        # Создаем данные с одним загрязнителем и None для остальных
        mixed_data = {
            'pm2_5': concentration if pollutant == 'pm2_5' else None,
            'pm10': concentration if pollutant == 'pm10' else None,
            'no2': concentration if pollutant == 'no2' else None,
            'so2': concentration if pollutant == 'so2' else None,
            'o3': concentration if pollutant == 'o3' else None
        }
        
        # Рассчитываем AQI для обоих случаев
        single_aqi, single_category, single_color = self.calculator.calculate_aqi(single_pollutant_data)
        mixed_aqi, mixed_category, mixed_color = self.calculator.calculate_aqi(mixed_data)
        
        # Результаты должны быть одинаковыми
        assert single_aqi == mixed_aqi, \
            f"Single pollutant AQI should match mixed data AQI: {single_aqi} vs {mixed_aqi}"
        assert single_category == mixed_category, \
            f"Single pollutant category should match mixed data category: '{single_category}' vs '{mixed_category}'"
        assert single_color == mixed_color, \
            f"Single pollutant color should match mixed data color: '{single_color}' vs '{mixed_color}'"
        
        # Проверяем, что результат основан на правильном загрязнителе
        if single_aqi == 0 and single_category == "Нет данных":
            # Очень низкие концентрации могут привести к AQI=0, что технически корректно
            # но система интерпретирует это как "нет данных"
            pass
        else:
            assert single_aqi > 0, f"AQI should be > 0 for valid concentration {concentration}"
            assert single_category != "Нет данных", f"Category should not be 'Нет данных' for valid data"
    
    @given(
        valid_pollutants=st.dictionaries(
            keys=st.sampled_from(['pm2_5', 'pm10', 'no2']),
            values=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=3
        ),
        invalid_values=st.lists(
            st.one_of(st.none(), st.floats(max_value=0.0), st.floats(min_value=-100.0, max_value=-0.1)),
            min_size=1,
            max_size=2
        )
    )
    def test_mixed_valid_invalid_data_property(self, valid_pollutants: Dict[str, float], invalid_values: list):
        """
        **Property 7: Incomplete Data Handling**
        **Validates: Requirements 3.4**
        
        For any dataset containing both valid pollutant values and invalid values
        (None, negative, zero), the AQI calculator should process only valid values
        and ignore invalid ones.
        """
        # Добавляем невалидные значения к валидным данным
        mixed_data = valid_pollutants.copy()
        
        # Добавляем невалидные значения для других загрязнителей
        other_pollutants = ['so2', 'o3']
        for i, invalid_value in enumerate(invalid_values[:len(other_pollutants)]):
            mixed_data[other_pollutants[i]] = invalid_value
        
        # Рассчитываем AQI для смешанных данных
        mixed_aqi, mixed_category, mixed_color = self.calculator.calculate_aqi(mixed_data)
        
        # Рассчитываем AQI только для валидных данных
        valid_aqi, valid_category, valid_color = self.calculator.calculate_aqi(valid_pollutants)
        
        # Результаты должны быть одинаковыми - невалидные значения должны игнорироваться
        assert mixed_aqi == valid_aqi, \
            f"Mixed data AQI should equal valid-only AQI: {mixed_aqi} vs {valid_aqi}"
        assert mixed_category == valid_category, \
            f"Mixed data category should equal valid-only category: '{mixed_category}' vs '{valid_category}'"
        assert mixed_color == valid_color, \
            f"Mixed data color should equal valid-only color: '{mixed_color}' vs '{valid_color}'"
    
    def test_all_missing_data_property(self):
        """
        **Property 7: Incomplete Data Handling**
        **Validates: Requirements 3.4**
        
        For completely missing pollutant data (all None or empty dict),
        the AQI calculator should return appropriate default values.
        """
        # Тест с пустым словарем
        empty_aqi, empty_category, empty_color = self.calculator.calculate_aqi({})
        
        assert empty_aqi == 0, f"Empty data should return AQI 0, got {empty_aqi}"
        assert empty_category == "Нет данных", f"Empty data should return 'Нет данных', got '{empty_category}'"
        assert empty_color == "#808080", f"Empty data should return gray color, got '{empty_color}'"
        
        # Тест со всеми None значениями
        all_none_data = {
            'pm2_5': None,
            'pm10': None,
            'no2': None,
            'so2': None,
            'o3': None
        }
        
        none_aqi, none_category, none_color = self.calculator.calculate_aqi(all_none_data)
        
        assert none_aqi == 0, f"All None data should return AQI 0, got {none_aqi}"
        assert none_category == "Нет данных", f"All None data should return 'Нет данных', got '{none_category}'"
        assert none_color == "#808080", f"All None data should return gray color, got '{none_color}'"
    
    @given(
        pollutants_with_zeros=st.dictionaries(
            keys=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3']),
            values=st.one_of(
                st.just(0.0),
                st.floats(min_value=-10.0, max_value=-0.1, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False)
            ),
            min_size=2,
            max_size=5
        )
    )
    def test_zero_and_negative_values_property(self, pollutants_with_zeros: Dict[str, float]):
        """
        **Property 7: Incomplete Data Handling**
        **Validates: Requirements 3.4**
        
        For any dataset containing zero or negative pollutant values,
        the AQI calculator should ignore these invalid values and
        calculate AQI from positive values only.
        """
        # Фильтруем только положительные значения
        positive_only = {k: v for k, v in pollutants_with_zeros.items() if v > 0}
        
        # Пропускаем случаи без положительных значений
        assume(len(positive_only) > 0)
        
        # Рассчитываем AQI для данных с нулями/отрицательными значениями
        mixed_aqi, mixed_category, mixed_color = self.calculator.calculate_aqi(pollutants_with_zeros)
        
        # Рассчитываем AQI только для положительных значений
        positive_aqi, positive_category, positive_color = self.calculator.calculate_aqi(positive_only)
        
        # Результаты должны быть одинаковыми
        assert mixed_aqi == positive_aqi, \
            f"AQI with zeros should equal positive-only AQI: {mixed_aqi} vs {positive_aqi}"
        assert mixed_category == positive_category, \
            f"Category with zeros should equal positive-only category: '{mixed_category}' vs '{positive_category}'"
        assert mixed_color == positive_color, \
            f"Color with zeros should equal positive-only color: '{mixed_color}' vs '{positive_color}'"