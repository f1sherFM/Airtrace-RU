"""
Property-based tests for AQI calculation consistency.

**Property 5: AQI Calculation Consistency**
**Validates: Requirements 3.1, 3.2**

Тестирует расчет AQI на основе российских стандартов ПДК с использованием
property-based testing для проверки корректности расчетов для всех возможных
значений концентраций загрязнителей.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Any

from utils import AQICalculator


class TestAQICalculationProperty:
    """Property-based тесты для расчета AQI"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.calculator = AQICalculator()
    
    @given(
        pm2_5=st.one_of(st.none(), st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False)),
        pm10=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        no2=st.one_of(st.none(), st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False)),
        so2=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        o3=st.one_of(st.none(), st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False))
    )
    def test_aqi_calculation_consistency_property(self, pm2_5: float, pm10: float, no2: float, so2: float, o3: float):
        """
        **Property 5: AQI Calculation Consistency**
        **Validates: Requirements 3.1, 3.2**
        
        For any set of valid pollutant values (PM2.5, PM10, NO2, SO2, O3),
        the AQI calculator should produce consistent AQI values using Russian ПДК standards.
        """
        pollutants = {}
        if pm2_5 is not None:
            pollutants['pm2_5'] = pm2_5
        if pm10 is not None:
            pollutants['pm10'] = pm10
        if no2 is not None:
            pollutants['no2'] = no2
        if so2 is not None:
            pollutants['so2'] = so2
        if o3 is not None:
            pollutants['o3'] = o3
        
        # Пропускаем тесты с пустыми данными
        assume(len(pollutants) > 0)
        assume(any(v > 0 for v in pollutants.values() if v is not None))
        
        aqi_value, category, color = self.calculator.calculate_aqi(pollutants)
        
        # Проверяем базовые свойства результата
        assert isinstance(aqi_value, int), f"AQI value should be integer, got {type(aqi_value)}"
        assert isinstance(category, str), f"Category should be string, got {type(category)}"
        assert isinstance(color, str), f"Color should be string, got {type(color)}"
        
        # AQI должен быть в разумных пределах
        assert 0 <= aqi_value <= 500, f"AQI value {aqi_value} should be between 0 and 500"
        
        # Категория не должна быть пустой
        assert len(category) > 0, "Category should not be empty"
        
        # Цвет должен быть в формате hex
        assert color.startswith('#'), f"Color {color} should start with #"
        assert len(color) == 7, f"Color {color} should be 7 characters long"
        
        # Проверяем соответствие AQI и категории
        if aqi_value == 0 and category == "Нет данных":
            # Специальный случай для отсутствия данных - это валидно
            pass
        else:
            expected_categories = {
                (0, 50): 'Хорошее',
                (51, 100): 'Умеренное',
                (101, 150): 'Вредно для чувствительных групп',
                (151, 200): 'Вредно',
                (201, 300): 'Очень вредно',
                (301, 500): 'Опасно'
            }
            
            found_category = False
            for (min_aqi, max_aqi), expected_category in expected_categories.items():
                if min_aqi <= aqi_value <= max_aqi:
                    assert category == expected_category, f"For AQI {aqi_value}, expected category '{expected_category}', got '{category}'"
                    found_category = True
                    break
            
            if not found_category and aqi_value > 500:
                assert category == "Критически опасно", f"For AQI {aqi_value} > 500, expected 'Критически опасно', got '{category}'"
    
    @given(
        concentration=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        pollutant=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3'])
    )
    def test_single_pollutant_aqi_monotonicity_property(self, concentration: float, pollutant: str):
        """
        **Property 5: AQI Calculation Consistency**
        **Validates: Requirements 3.1, 3.2**
        
        For any single pollutant, higher concentrations should result in 
        higher or equal AQI values (monotonicity property).
        """
        assume(concentration > 0)
        
        # Тестируем с текущей концентрацией
        pollutants1 = {pollutant: concentration}
        aqi1, _, _ = self.calculator.calculate_aqi(pollutants1)
        
        # Тестируем с увеличенной концентрацией
        higher_concentration = concentration * 1.5
        pollutants2 = {pollutant: higher_concentration}
        aqi2, _, _ = self.calculator.calculate_aqi(pollutants2)
        
        # AQI должен увеличиваться или оставаться тем же при увеличении концентрации
        assert aqi2 >= aqi1, f"AQI should be monotonic: {concentration} -> AQI {aqi1}, {higher_concentration} -> AQI {aqi2}"
    
    @given(
        base_concentration=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        multiplier=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    def test_aqi_scaling_property(self, base_concentration: float, multiplier: float):
        """
        **Property 5: AQI Calculation Consistency**
        **Validates: Requirements 3.1, 3.2**
        
        For any pollutant concentration, scaling all pollutants by the same factor
        should result in predictable AQI changes based on Russian ПДК standards.
        """
        # Базовые концентрации
        base_pollutants = {
            'pm2_5': base_concentration,
            'pm10': base_concentration * 2,
            'no2': base_concentration * 1.5
        }
        
        # Увеличенные концентрации
        scaled_pollutants = {
            pollutant: concentration * multiplier
            for pollutant, concentration in base_pollutants.items()
        }
        
        aqi_base, _, _ = self.calculator.calculate_aqi(base_pollutants)
        aqi_scaled, _, _ = self.calculator.calculate_aqi(scaled_pollutants)
        
        # При увеличении всех концентраций AQI должен увеличиваться
        if multiplier > 1.0:
            assert aqi_scaled >= aqi_base, f"Scaled AQI {aqi_scaled} should be >= base AQI {aqi_base} when multiplier={multiplier}"
    
    @given(
        pollutants_dict=st.dictionaries(
            keys=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3']),
            values=st.floats(min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5
        )
    )
    def test_aqi_recommendations_consistency_property(self, pollutants_dict: Dict[str, float]):
        """
        **Property 5: AQI Calculation Consistency**
        **Validates: Requirements 3.1, 3.2**
        
        For any calculated AQI value, the recommendations should be consistent
        with the AQI level and in Russian language.
        """
        aqi_value, category, color = self.calculator.calculate_aqi(pollutants_dict)
        recommendations = self.calculator.get_recommendations(aqi_value, category)
        
        # Рекомендации должны быть строкой
        assert isinstance(recommendations, str), f"Recommendations should be string, got {type(recommendations)}"
        
        # Рекомендации не должны быть пустыми
        assert len(recommendations) > 0, "Recommendations should not be empty"
        
        # Рекомендации должны содержать русские слова (проверяем наличие кириллицы)
        assert any(ord(char) >= 1040 and ord(char) <= 1103 for char in recommendations), \
            f"Recommendations should contain Russian text: '{recommendations}'"
        
        # Проверяем соответствие рекомендаций уровню AQI
        if aqi_value <= 50:
            assert any(word in recommendations.lower() for word in ['отличное', 'идеальные', 'любых']), \
                f"For good AQI {aqi_value}, recommendations should mention good conditions: '{recommendations}'"
        elif aqi_value <= 100:
            assert any(word in recommendations.lower() for word in ['хорошее', 'можно', 'любыми']), \
                f"For moderate AQI {aqi_value}, recommendations should mention acceptable conditions: '{recommendations}'"
        elif aqi_value <= 150:
            assert any(word in recommendations.lower() for word in ['чувствительные', 'ограничить', 'должны']), \
                f"For unhealthy for sensitive AQI {aqi_value}, recommendations should mention sensitive groups: '{recommendations}'"
        elif aqi_value <= 200:
            assert any(word in recommendations.lower() for word in ['всем', 'ограничить', 'рекомендуется']), \
                f"For unhealthy AQI {aqi_value}, recommendations should mention general restrictions: '{recommendations}'"
        elif aqi_value <= 300:
            assert any(word in recommendations.lower() for word in ['избегать', 'оставаться', 'помещении']), \
                f"For very unhealthy AQI {aqi_value}, recommendations should mention staying indoors: '{recommendations}'"
        else:
            assert any(word in recommendations.lower() for word in ['чрезвычайная', 'ситуация', 'оставаться']), \
                f"For hazardous AQI {aqi_value}, recommendations should mention emergency: '{recommendations}'"
    
    def test_empty_pollutants_property(self):
        """
        **Property 5: AQI Calculation Consistency**
        **Validates: Requirements 3.1, 3.2**
        
        For empty pollutant data, the AQI calculator should return
        appropriate default values indicating no data.
        """
        aqi_value, category, color = self.calculator.calculate_aqi({})
        
        assert aqi_value == 0, f"Empty pollutants should return AQI 0, got {aqi_value}"
        assert category == "Нет данных", f"Empty pollutants should return 'Нет данных', got '{category}'"
        assert color == "#808080", f"Empty pollutants should return gray color, got '{color}'"
    
    @given(
        pollutants_dict=st.dictionaries(
            keys=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3']),
            values=st.one_of(st.none(), st.floats(min_value=-100.0, max_value=-0.1)),
            min_size=1,
            max_size=3
        )
    )
    def test_negative_pollutants_property(self, pollutants_dict: Dict[str, float]):
        """
        **Property 5: AQI Calculation Consistency**
        **Validates: Requirements 3.1, 3.2**
        
        For negative or None pollutant values, the AQI calculator should
        ignore them and calculate AQI from valid positive values only.
        """
        # Добавляем один валидный загрязнитель для проверки
        pollutants_dict['pm2_5'] = 25.0
        
        aqi_value, category, color = self.calculator.calculate_aqi(pollutants_dict)
        
        # Результат должен быть основан только на валидных данных
        assert aqi_value > 0, f"Should calculate AQI from valid data, got {aqi_value}"
        assert category != "Нет данных", f"Should not return 'Нет данных' when valid data exists, got '{category}'"
        
        # Проверяем, что результат соответствует только PM2.5 = 25.0
        expected_aqi, expected_category, expected_color = self.calculator.calculate_aqi({'pm2_5': 25.0})
        assert aqi_value == expected_aqi, f"AQI should match calculation from valid data only"
        assert category == expected_category, f"Category should match calculation from valid data only"
        assert color == expected_color, f"Color should match calculation from valid data only"