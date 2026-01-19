"""
Property-based tests for Russian localization.

**Property 8: Russian Localization**
**Validates: Requirements 4.2, 4.3**

Тестирует русскую локализацию всех текстовых выходов системы с использованием
property-based testing для проверки корректности русского языка во всех
рекомендациях, предупреждениях и описаниях.
"""

import pytest
import re
from hypothesis import given, strategies as st, settings
from typing import Dict

from utils import AQICalculator, get_pollutant_name_russian, format_russian_timestamp, check_nmu_risk
from datetime import datetime


class TestRussianLocalizationProperty:
    """Property-based тесты для русской локализации"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.calculator = AQICalculator()
    
    def _contains_cyrillic(self, text: str) -> bool:
        """Проверяет, содержит ли текст кириллические символы"""
        return bool(re.search(r'[а-яё]', text.lower()))
    
    def _is_valid_russian_text(self, text: str) -> bool:
        """Проверяет, является ли текст валидным русским текстом"""
        if not text or len(text.strip()) == 0:
            return False
        
        # Должен содержать кириллицу
        if not self._contains_cyrillic(text):
            return False
        
        # Не должен содержать английские буквы в основном тексте (кроме аббревиатур)
        english_pattern = r'\b[a-zA-Z]{3,}\b'  # Английские слова длиной 3+ символа
        if re.search(english_pattern, text):
            return False
        
        return True
    
    @given(
        pollutants_dict=st.dictionaries(
            keys=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3']),
            values=st.floats(min_value=0.1, max_value=300.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5
        )
    )
    def test_aqi_categories_russian_property(self, pollutants_dict: Dict[str, float]):
        """
        **Property 8: Russian Localization**
        **Validates: Requirements 4.2, 4.3**
        
        For any calculated AQI value, all category names should be in Russian language.
        """
        aqi_value, category, color = self.calculator.calculate_aqi(pollutants_dict)
        
        # Категория должна быть на русском языке
        assert self._is_valid_russian_text(category), \
            f"AQI category should be in Russian: '{category}'"
        
        # Проверяем, что категория соответствует ожидаемым русским категориям
        expected_russian_categories = [
            'Хорошее', 'Умеренное', 'Вредно для чувствительных групп',
            'Вредно', 'Очень вредно', 'Опасно', 'Критически опасно', 'Нет данных'
        ]
        
        assert category in expected_russian_categories, \
            f"Category '{category}' should be one of expected Russian categories: {expected_russian_categories}"
    
    @given(
        pollutants_dict=st.dictionaries(
            keys=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3']),
            values=st.floats(min_value=0.1, max_value=300.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5
        )
    )
    def test_aqi_recommendations_russian_property(self, pollutants_dict: Dict[str, float]):
        """
        **Property 8: Russian Localization**
        **Validates: Requirements 4.2, 4.3**
        
        For any calculated AQI value, all recommendations should be in Russian language
        and contain appropriate Russian health advice terminology.
        """
        aqi_value, category, color = self.calculator.calculate_aqi(pollutants_dict)
        recommendations = self.calculator.get_recommendations(aqi_value, category)
        
        # Рекомендации должны быть на русском языке
        assert self._is_valid_russian_text(recommendations), \
            f"Recommendations should be in Russian: '{recommendations}'"
        
        # Проверяем наличие ключевых русских терминов в зависимости от уровня AQI
        if aqi_value <= 50:
            russian_terms = ['отличное', 'идеальные', 'любых', 'активностей']
            assert any(term in recommendations.lower() for term in russian_terms), \
                f"Good AQI recommendations should contain Russian positive terms: '{recommendations}'"
        
        elif aqi_value <= 100:
            russian_terms = ['хорошее', 'можно', 'любыми', 'деятельности']
            assert any(term in recommendations.lower() for term in russian_terms), \
                f"Moderate AQI recommendations should contain Russian moderate terms: '{recommendations}'"
        
        elif aqi_value <= 150:
            russian_terms = ['чувствительные', 'ограничить', 'должны', 'физические', 'нагрузки']
            assert any(term in recommendations.lower() for term in russian_terms), \
                f"Unhealthy for sensitive AQI recommendations should contain Russian warning terms: '{recommendations}'"
        
        elif aqi_value <= 200:
            russian_terms = ['всем', 'рекомендуется', 'ограничить', 'избегать']
            assert any(term in recommendations.lower() for term in russian_terms), \
                f"Unhealthy AQI recommendations should contain Russian restriction terms: '{recommendations}'"
        
        elif aqi_value <= 300:
            russian_terms = ['избегать', 'оставаться', 'помещении', 'активности']
            assert any(term in recommendations.lower() for term in russian_terms), \
                f"Very unhealthy AQI recommendations should contain Russian indoor terms: '{recommendations}'"
        
        else:
            russian_terms = ['чрезвычайная', 'ситуация', 'оставаться', 'помещении', 'избегать']
            assert any(term in recommendations.lower() for term in russian_terms), \
                f"Hazardous AQI recommendations should contain Russian emergency terms: '{recommendations}'"
    
    @given(
        pollutant_code=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3'])
    )
    def test_pollutant_names_russian_property(self, pollutant_code: str):
        """
        **Property 8: Russian Localization**
        **Validates: Requirements 4.2, 4.3**
        
        For any pollutant code, the Russian name should be properly localized
        and contain appropriate Russian chemical/environmental terminology.
        """
        russian_name = get_pollutant_name_russian(pollutant_code)
        
        # Название должно быть на русском языке
        assert self._is_valid_russian_text(russian_name), \
            f"Pollutant name should be in Russian: '{russian_name}'"
        
        # Проверяем соответствие конкретных названий
        expected_names = {
            'pm2_5': 'Мелкодисперсные частицы PM2.5',
            'pm10': 'Взвешенные частицы PM10',
            'no2': 'Диоксид азота',
            'so2': 'Диоксид серы',
            'o3': 'Озон'
        }
        
        assert russian_name == expected_names[pollutant_code], \
            f"Pollutant name for {pollutant_code} should be '{expected_names[pollutant_code]}', got '{russian_name}'"
    
    @given(
        category=st.sampled_from(['Хорошее', 'Умеренное', 'Вредно для чувствительных групп', 
                                'Вредно', 'Очень вредно', 'Опасно'])
    )
    def test_category_descriptions_russian_property(self, category: str):
        """
        **Property 8: Russian Localization**
        **Validates: Requirements 4.2, 4.3**
        
        For any AQI category, the description should be in Russian language
        and contain appropriate health-related terminology.
        """
        description = self.calculator.get_category_description(category)
        
        # Описание должно быть на русском языке
        assert self._is_valid_russian_text(description), \
            f"Category description should be in Russian: '{description}'"
        
        # Проверяем наличие ключевых русских терминов здравоохранения
        health_terms = [
            'качество', 'воздуха', 'здоровье', 'риск', 'загрязнение',
            'люди', 'население', 'проблемы', 'условия', 'чувствительные'
        ]
        
        assert any(term in description.lower() for term in health_terms), \
            f"Category description should contain Russian health terms: '{description}'"
    
    @given(
        pollutants_dict=st.dictionaries(
            keys=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3']),
            values=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5
        )
    )
    def test_nmu_risk_levels_russian_property(self, pollutants_dict: Dict[str, float]):
        """
        **Property 8: Russian Localization**
        **Validates: Requirements 4.2, 4.3**
        
        For any pollutant data, NMU (неблагоприятные метеорологические условия)
        risk levels should use appropriate Russian terminology.
        """
        # Пустые метеоусловия для базового теста
        weather_conditions = {}
        
        risk_level = check_nmu_risk(pollutants_dict, weather_conditions)
        
        # Уровень риска должен быть на английском (это технический код), 
        # но проверим, что он из ожидаемого набора
        expected_risk_levels = ['low', 'medium', 'high', 'critical', 'unknown']
        assert risk_level in expected_risk_levels, \
            f"NMU risk level should be one of {expected_risk_levels}, got '{risk_level}'"
        
        # В будущем, когда будут добавлены русские описания НМУ,
        # здесь можно будет проверить русскую локализацию описаний
    
    @given(
        timestamp=st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31)
        )
    )
    def test_timestamp_formatting_russian_property(self, timestamp: datetime):
        """
        **Property 8: Russian Localization**
        **Validates: Requirements 4.2, 4.3**
        
        For any timestamp, the Russian formatting should use appropriate
        Russian date/time format conventions.
        """
        formatted_time = format_russian_timestamp(timestamp)
        
        # Проверяем формат даты (DD.MM.YYYY HH:MM UTC)
        date_pattern = r'\d{2}\.\d{2}\.\d{4} \d{2}:\d{2} UTC'
        assert re.match(date_pattern, formatted_time), \
            f"Russian timestamp should match DD.MM.YYYY HH:MM UTC format, got '{formatted_time}'"
        
        # Проверяем, что используется точка как разделитель даты (русский стандарт)
        assert '.' in formatted_time[:10], \
            f"Russian date format should use dots as separators: '{formatted_time}'"
        
        # Проверяем, что используется двоеточие как разделитель времени
        time_part = formatted_time.split(' ')[1]
        assert ':' in time_part, \
            f"Russian time format should use colon as separator: '{formatted_time}'"
    
    def test_empty_data_russian_localization_property(self):
        """
        **Property 8: Russian Localization**
        **Validates: Requirements 4.2, 4.3**
        
        For empty or invalid data, error messages and default values
        should be properly localized in Russian.
        """
        # Тест с пустыми данными
        aqi_value, category, color = self.calculator.calculate_aqi({})
        
        # Категория "Нет данных" должна быть на русском
        assert category == "Нет данных", \
            f"Empty data category should be 'Нет данных', got '{category}'"
        
        assert self._is_valid_russian_text(category), \
            f"Empty data category should be in Russian: '{category}'"
    
    @given(
        aqi_value=st.integers(min_value=0, max_value=500)
    )
    def test_all_aqi_ranges_russian_property(self, aqi_value: int):
        """
        **Property 8: Russian Localization**
        **Validates: Requirements 4.2, 4.3**
        
        For any AQI value in the valid range (0-500), all associated text
        (categories, recommendations) should be consistently in Russian.
        """
        # Получаем категорию для данного AQI
        category, color = self.calculator._get_aqi_category_and_color(aqi_value)
        
        # Категория должна быть на русском
        assert self._is_valid_russian_text(category), \
            f"Category for AQI {aqi_value} should be in Russian: '{category}'"
        
        # Получаем рекомендации
        recommendations = self.calculator.get_recommendations(aqi_value, category)
        
        # Рекомендации должны быть на русском
        assert self._is_valid_russian_text(recommendations), \
            f"Recommendations for AQI {aqi_value} should be in Russian: '{recommendations}'"
        
        # Получаем описание категории
        description = self.calculator.get_category_description(category)
        
        # Описание должно быть на русском
        assert self._is_valid_russian_text(description), \
            f"Description for category '{category}' should be in Russian: '{description}'"