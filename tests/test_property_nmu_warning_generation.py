"""
Property-based tests for NMU warning generation.

**Property 9: NMU Warning Generation**
**Validates: Requirements 4.2, 4.4**

Тестирует генерацию предупреждений о неблагоприятных метеорологических условиях (НМУ)
с использованием property-based testing для проверки корректности определения
уровней риска и генерации соответствующих предупреждений на русском языке.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Any, List

from utils import NMUDetector, check_nmu_risk, is_blacksky_conditions, get_nmu_recommendations


class TestNMUWarningGenerationProperty:
    """Property-based тесты для генерации НМУ предупреждений"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.nmu_detector = NMUDetector()
    
    @given(
        pm2_5=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        pm10=st.one_of(st.none(), st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False)),
        no2=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1500.0, allow_nan=False, allow_infinity=False)),
        so2=st.one_of(st.none(), st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False)),
        o3=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1200.0, allow_nan=False, allow_infinity=False))
    )
    def test_nmu_risk_level_consistency_property(self, pm2_5: float, pm10: float, no2: float, so2: float, o3: float):
        """
        **Property 9: NMU Warning Generation**
        **Validates: Requirements 4.2, 4.4**
        
        For any set of pollutant concentrations, the NMU risk assessment should
        return consistent risk levels and appropriate warnings in Russian language.
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
        
        risk_level = check_nmu_risk(pollutants)
        
        # Проверяем, что уровень риска валиден
        valid_risk_levels = ['low', 'medium', 'high', 'critical', 'unknown']
        assert risk_level in valid_risk_levels, f"Risk level '{risk_level}' should be one of {valid_risk_levels}"
        
        # Для валидных данных не должно быть 'unknown'
        if any(v > 0 for v in pollutants.values() if v is not None):
            assert risk_level != 'unknown', f"Risk level should not be 'unknown' for valid pollutant data: {pollutants}"
    
    @given(
        pm2_5=st.floats(min_value=125.0, max_value=1000.0, allow_nan=False, allow_infinity=False),  # 5x ПДК для PM2.5
        pm10=st.floats(min_value=250.0, max_value=2000.0, allow_nan=False, allow_infinity=False),  # 5x ПДК для PM10
        no2=st.floats(min_value=400.0, max_value=1500.0, allow_nan=False, allow_infinity=False),   # 10x ПДК для NO2
        so2=st.floats(min_value=500.0, max_value=2000.0, allow_nan=False, allow_infinity=False),   # 10x ПДК для SO2
        o3=st.floats(min_value=300.0, max_value=1200.0, allow_nan=False, allow_infinity=False)     # 3x ПДК для O3
    )
    def test_blacksky_conditions_detection_property(self, pm2_5: float, pm10: float, no2: float, so2: float, o3: float):
        """
        **Property 9: NMU Warning Generation**
        **Validates: Requirements 4.2, 4.4**
        
        For any pollutant concentrations exceeding "Black Sky" thresholds,
        the system should detect critical conditions and generate appropriate warnings.
        """
        # Тестируем каждый загрязнитель отдельно для условий "Черное небо"
        test_cases = [
            {'pm2_5': pm2_5},
            {'pm10': pm10},
            {'no2': no2},
            {'so2': so2},
            {'o3': o3}
        ]
        
        for pollutants in test_cases:
            blacksky = is_blacksky_conditions(pollutants)
            risk_level = check_nmu_risk(pollutants)
            
            # При условиях "Черное небо" риск должен быть критическим
            if blacksky:
                assert risk_level == 'critical', f"Black sky conditions should result in critical risk, got '{risk_level}' for {pollutants}"
    
    @given(
        risk_level=st.sampled_from(['low', 'medium', 'high', 'critical']),
        blacksky=st.booleans()
    )
    def test_nmu_recommendations_format_property(self, risk_level: str, blacksky: bool):
        """
        **Property 9: NMU Warning Generation**
        **Validates: Requirements 4.2, 4.4**
        
        For any risk level and black sky condition, the NMU recommendations
        should be properly formatted and contain Russian text.
        """
        recommendations = get_nmu_recommendations(risk_level, blacksky)
        
        # Рекомендации должны быть списком строк
        assert isinstance(recommendations, list), f"Recommendations should be a list, got {type(recommendations)}"
        assert len(recommendations) > 0, "Recommendations should not be empty"
        
        for recommendation in recommendations:
            assert isinstance(recommendation, str), f"Each recommendation should be a string, got {type(recommendation)}"
            assert len(recommendation) > 0, "Recommendation should not be empty string"
            
            # Проверяем наличие русского текста (кириллица)
            has_cyrillic = any(ord(char) >= 1040 and ord(char) <= 1103 for char in recommendation)
            assert has_cyrillic, f"Recommendation should contain Russian text: '{recommendation}'"
        
        recommendations_text = ' '.join(recommendations).lower()
        
        # Проверяем специфические требования для условий "Черное небо"
        if blacksky:
            assert 'черное небо' in recommendations_text, "Black sky recommendations should mention 'черное небо'"
            assert any(word in recommendations_text for word in ['критическая', 'ситуация', 'немедленно']), \
                "Black sky recommendations should indicate urgency"
            # При условиях "Черное небо" приоритет имеют экстренные рекомендации
            return
        
        # Проверяем соответствие рекомендаций уровню риска (только если нет условий "Черное небо")
        if risk_level == 'critical':
            assert any(word in recommendations_text for word in ['критический', 'помещении', 'избегайте']), \
                f"Critical risk recommendations should mention staying indoors: {recommendations}"
        elif risk_level == 'high':
            assert any(word in recommendations_text for word in ['высокий', 'ограничьте', 'избегайте']), \
                f"High risk recommendations should mention limitations: {recommendations}"
        elif risk_level == 'medium':
            assert any(word in recommendations_text for word in ['умеренный', 'осторожны', 'ограничьте']), \
                f"Medium risk recommendations should mention caution: {recommendations}"
        elif risk_level == 'low':
            assert any(word in recommendations_text for word in ['низкий', 'обычные', 'предосторожности']), \
                f"Low risk recommendations should mention normal precautions: {recommendations}"
    
    @given(
        base_concentration=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        multiplier=st.floats(min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False)
    )
    def test_nmu_risk_escalation_property(self, base_concentration: float, multiplier: float):
        """
        **Property 9: NMU Warning Generation**
        **Validates: Requirements 4.2, 4.4**
        
        For any base pollutant concentration, increasing the concentration
        should result in the same or higher NMU risk level.
        """
        # Базовые концентрации
        base_pollutants = {'pm2_5': base_concentration}
        
        # Увеличенные концентрации
        scaled_pollutants = {'pm2_5': base_concentration * multiplier}
        
        base_risk = check_nmu_risk(base_pollutants)
        scaled_risk = check_nmu_risk(scaled_pollutants)
        
        # Определяем порядок уровней риска
        risk_order = {'unknown': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        
        base_level = risk_order.get(base_risk, 0)
        scaled_level = risk_order.get(scaled_risk, 0)
        
        # При увеличении концентрации риск должен увеличиваться или оставаться тем же
        if multiplier > 1.0:
            assert scaled_level >= base_level, \
                f"Risk should not decrease when concentration increases: {base_concentration} -> {base_risk}, " \
                f"{base_concentration * multiplier} -> {scaled_risk}"
    
    @given(
        pollutants_dict=st.dictionaries(
            keys=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3']),
            values=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5
        )
    )
    def test_nmu_weather_factor_integration_property(self, pollutants_dict: Dict[str, float]):
        """
        **Property 9: NMU Warning Generation**
        **Validates: Requirements 4.2, 4.4**
        
        For any pollutant concentrations, the NMU risk assessment should
        handle weather conditions integration properly (even when weather data is None).
        """
        # Тестируем без метеоданных
        risk_no_weather = check_nmu_risk(pollutants_dict, None)
        
        # Тестируем с пустыми метеоданными
        risk_empty_weather = check_nmu_risk(pollutants_dict, {})
        
        # Тестируем с некоторыми метеоданными
        weather_conditions = {'wind_speed': 5.0, 'temperature': 20.0}
        risk_with_weather = check_nmu_risk(pollutants_dict, weather_conditions)
        
        # Все варианты должны возвращать валидные уровни риска
        valid_levels = ['low', 'medium', 'high', 'critical', 'unknown']
        assert risk_no_weather in valid_levels, f"Risk without weather should be valid: {risk_no_weather}"
        assert risk_empty_weather in valid_levels, f"Risk with empty weather should be valid: {risk_empty_weather}"
        assert risk_with_weather in valid_levels, f"Risk with weather should be valid: {risk_with_weather}"
        
        # Без метеоданных и с пустыми метеоданными результат должен быть одинаковым
        assert risk_no_weather == risk_empty_weather, \
            f"Risk should be same for None and empty weather: {risk_no_weather} vs {risk_empty_weather}"
    
    @given(
        pm2_5=st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False),  # Значительно ниже ПДК (25)
        pm10=st.floats(min_value=0.1, max_value=40.0, allow_nan=False, allow_infinity=False),  # Значительно ниже ПДК (50)
        no2=st.floats(min_value=0.1, max_value=30.0, allow_nan=False, allow_infinity=False),   # Значительно ниже ПДК (40)
        so2=st.floats(min_value=0.1, max_value=40.0, allow_nan=False, allow_infinity=False),   # Значительно ниже ПДК (50)
        o3=st.floats(min_value=0.1, max_value=80.0, allow_nan=False, allow_infinity=False)     # Значительно ниже ПДК (100)
    )
    def test_low_pollution_nmu_risk_property(self, pm2_5: float, pm10: float, no2: float, so2: float, o3: float):
        """
        **Property 9: NMU Warning Generation**
        **Validates: Requirements 4.2, 4.4**
        
        For any pollutant concentrations below Russian ПДК standards,
        the NMU risk should be low and no black sky conditions should be detected.
        """
        pollutants = {
            'pm2_5': pm2_5,
            'pm10': pm10,
            'no2': no2,
            'so2': so2,
            'o3': o3
        }
        
        risk_level = check_nmu_risk(pollutants)
        blacksky = is_blacksky_conditions(pollutants)
        
        # При концентрациях ниже ПДК риск должен быть низким
        assert risk_level == 'low', f"Risk should be low for concentrations below ПДК, got '{risk_level}' for {pollutants}"
        
        # Условия "Черное небо" не должны определяться
        assert not blacksky, f"Black sky should not be detected for concentrations below ПДК: {pollutants}"
    
    def test_empty_pollutants_nmu_property(self):
        """
        **Property 9: NMU Warning Generation**
        **Validates: Requirements 4.2, 4.4**
        
        For empty pollutant data, the NMU risk assessment should return
        'unknown' and no black sky conditions should be detected.
        """
        risk_level = check_nmu_risk({})
        blacksky = is_blacksky_conditions({})
        recommendations = get_nmu_recommendations('unknown', False)
        
        assert risk_level == 'unknown', f"Empty pollutants should return 'unknown' risk, got '{risk_level}'"
        assert not blacksky, "Empty pollutants should not trigger black sky conditions"
        assert isinstance(recommendations, list), "Recommendations should be a list even for unknown risk"
    
    @given(
        pollutants_dict=st.dictionaries(
            keys=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3']),
            values=st.one_of(st.none(), st.floats(min_value=-100.0, max_value=-0.1)),
            min_size=1,
            max_size=3
        )
    )
    def test_invalid_pollutants_nmu_property(self, pollutants_dict: Dict[str, float]):
        """
        **Property 9: NMU Warning Generation**
        **Validates: Requirements 4.2, 4.4**
        
        For invalid (negative or None) pollutant values, the NMU risk assessment
        should handle them gracefully and return appropriate risk levels.
        """
        risk_level = check_nmu_risk(pollutants_dict)
        blacksky = is_blacksky_conditions(pollutants_dict)
        
        # Для невалидных данных риск должен быть 'unknown'
        assert risk_level == 'unknown', f"Invalid pollutants should return 'unknown' risk, got '{risk_level}'"
        
        # Условия "Черное небо" не должны определяться для невалидных данных
        assert not blacksky, f"Black sky should not be detected for invalid pollutants: {pollutants_dict}"
    
    @given(
        concentration=st.floats(min_value=1000.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        pollutant=st.sampled_from(['pm2_5', 'pm10', 'no2', 'so2', 'o3'])
    )
    def test_extreme_pollution_nmu_property(self, concentration: float, pollutant: str):
        """
        **Property 9: NMU Warning Generation**
        **Validates: Requirements 4.2, 4.4**
        
        For any extremely high pollutant concentrations, the NMU risk should be
        critical and appropriate emergency warnings should be generated.
        """
        pollutants = {pollutant: concentration}
        
        risk_level = check_nmu_risk(pollutants)
        recommendations = get_nmu_recommendations(risk_level)
        
        # При экстремально высоких концентрациях риск должен быть критическим
        assert risk_level == 'critical', f"Extreme pollution should result in critical risk, got '{risk_level}' for {pollutants}"
        
        # Рекомендации должны содержать экстренные меры
        recommendations_text = ' '.join(recommendations).lower()
        assert any(word in recommendations_text for word in ['критический', 'помещении', 'избегайте', 'чрезвычайная']), \
            f"Extreme pollution recommendations should mention emergency measures: {recommendations}"