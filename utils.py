"""
Утилиты для расчета AQI и определения НМУ в AirTrace RU Backend

Содержит калькулятор AQI на основе российских стандартов ПДК,
функции для определения неблагоприятных метеорологических условий
и вспомогательные функции.
"""

import math
import logging
from typing import Dict, Tuple, Optional, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AQICalculator:
    """
    Калькулятор индекса качества воздуха (AQI) на основе российских стандартов ПДК.
    
    Использует российские предельно допустимые концентрации (ПДК) для расчета
    индекса качества воздуха и генерации рекомендаций на русском языке.
    """
    
    # Российские стандарты ПДК (мкг/м³) - среднесуточные значения
    RU_STANDARDS = {
        'pm2_5': {
            'good': 25,           # ПДК сс для PM2.5
            'moderate': 50,       # 2x ПДК
            'unhealthy_sensitive': 75,  # 3x ПДК
            'unhealthy': 100,     # 4x ПДК
            'very_unhealthy': 150, # 6x ПДК
            'hazardous': 250      # 10x ПДК
        },
        'pm10': {
            'good': 50,           # ПДК сс для PM10
            'moderate': 100,      # 2x ПДК
            'unhealthy_sensitive': 150,  # 3x ПДК
            'unhealthy': 200,     # 4x ПДК
            'very_unhealthy': 300, # 6x ПДК
            'hazardous': 500      # 10x ПДК
        },
        'no2': {
            'good': 40,           # ПДК сс для NO2
            'moderate': 80,       # 2x ПДК
            'unhealthy_sensitive': 120,  # 3x ПДК
            'unhealthy': 160,     # 4x ПДК
            'very_unhealthy': 240, # 6x ПДК
            'hazardous': 400      # 10x ПДК
        },
        'so2': {
            'good': 50,           # ПДК сс для SO2
            'moderate': 100,      # 2x ПДК
            'unhealthy_sensitive': 150,  # 3x ПДК
            'unhealthy': 200,     # 4x ПДК
            'very_unhealthy': 300, # 6x ПДК
            'hazardous': 500      # 10x ПДК
        },
        'o3': {
            'good': 100,          # ПДК мр для O3 (максимальная разовая)
            'moderate': 160,      # 1.6x ПДК
            'unhealthy_sensitive': 200,  # 2x ПДК
            'unhealthy': 240,     # 2.4x ПДК
            'very_unhealthy': 300, # 3x ПДК
            'hazardous': 400      # 4x ПДК
        }
    }
    
    # Соответствие категорий AQI значениям и цветам
    AQI_CATEGORIES = {
        (0, 50): {
            'category': 'Хорошее',
            'color': '#00E400',
            'description': 'Качество воздуха считается удовлетворительным, загрязнение воздуха представляет незначительный риск или не представляет риска'
        },
        (51, 100): {
            'category': 'Умеренное',
            'color': '#FFFF00',
            'description': 'Качество воздуха приемлемо для большинства людей. Однако чувствительные люди могут испытывать незначительные проблемы'
        },
        (101, 150): {
            'category': 'Вредно для чувствительных групп',
            'color': '#FF7E00',
            'description': 'Представители чувствительных групп могут испытывать проблемы со здоровьем. Широкая общественность, как правило, не пострадает'
        },
        (151, 200): {
            'category': 'Вредно',
            'color': '#FF0000',
            'description': 'Каждый может начать испытывать проблемы со здоровьем; представители чувствительных групп могут испытывать более серьезные проблемы'
        },
        (201, 300): {
            'category': 'Очень вредно',
            'color': '#8F3F97',
            'description': 'Предупреждения о вреде для здоровья при чрезвычайных условиях. Вероятность воздействия на все население'
        },
        (301, 500): {
            'category': 'Опасно',
            'color': '#7E0023',
            'description': 'Чрезвычайная ситуация: все население подвержено риску серьезных проблем со здоровьем'
        }
    }
    
    def calculate_aqi(self, pollutants: Dict[str, float]) -> Tuple[int, str, str]:
        """
        Расчет AQI на основе российских стандартов ПДК.
        
        Args:
            pollutants: Словарь с концентрациями загрязнителей
            
        Returns:
            Tuple[int, str, str]: (AQI значение, категория, цвет)
        """
        if not pollutants:
            logger.warning("No pollutant data provided for AQI calculation")
            return 0, "Нет данных", "#808080"
        
        max_aqi = 0
        dominant_pollutant = None
        
        for pollutant, concentration in pollutants.items():
            if concentration is None or concentration < 0:
                continue
                
            if pollutant in self.RU_STANDARDS:
                aqi_value = self._calculate_pollutant_aqi(pollutant, concentration)
                if aqi_value > max_aqi:
                    max_aqi = aqi_value
                    dominant_pollutant = pollutant
        
        if max_aqi == 0:
            logger.warning("No valid pollutant data for AQI calculation")
            return 0, "Нет данных", "#808080"
        
        category, color = self._get_aqi_category_and_color(max_aqi)
        
        logger.info(f"Calculated AQI: {max_aqi}, Category: {category}, Dominant: {dominant_pollutant}")
        return max_aqi, category, color
    
    def _calculate_pollutant_aqi(self, pollutant: str, concentration: float) -> int:
        """Расчет AQI для отдельного загрязнителя"""
        standards = self.RU_STANDARDS[pollutant]
        
        # Определение диапазона концентрации
        if concentration <= standards['good']:
            return self._linear_interpolation(concentration, 0, standards['good'], 0, 50)
        elif concentration <= standards['moderate']:
            return self._linear_interpolation(concentration, standards['good'], standards['moderate'], 51, 100)
        elif concentration <= standards['unhealthy_sensitive']:
            return self._linear_interpolation(concentration, standards['moderate'], standards['unhealthy_sensitive'], 101, 150)
        elif concentration <= standards['unhealthy']:
            return self._linear_interpolation(concentration, standards['unhealthy_sensitive'], standards['unhealthy'], 151, 200)
        elif concentration <= standards['very_unhealthy']:
            return self._linear_interpolation(concentration, standards['unhealthy'], standards['very_unhealthy'], 201, 300)
        elif concentration <= standards['hazardous']:
            return self._linear_interpolation(concentration, standards['very_unhealthy'], standards['hazardous'], 301, 400)
        else:
            # Для значений выше hazardous используем экстраполяцию
            return min(500, int(400 + (concentration - standards['hazardous']) / standards['hazardous'] * 100))
    
    def _linear_interpolation(self, concentration: float, c_low: float, c_high: float, aqi_low: int, aqi_high: int) -> int:
        """Линейная интерполяция для расчета AQI"""
        if c_high == c_low:
            return aqi_low
        
        aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
        return int(round(aqi))
    
    def _get_aqi_category_and_color(self, aqi_value: int) -> Tuple[str, str]:
        """Определение категории и цвета по значению AQI"""
        for (min_aqi, max_aqi), info in self.AQI_CATEGORIES.items():
            if min_aqi <= aqi_value <= max_aqi:
                return info['category'], info['color']
        
        # Для значений выше 500
        return "Критически опасно", "#7E0023"
    
    def get_category_description(self, category: str) -> str:
        """Получение описания категории качества воздуха"""
        for info in self.AQI_CATEGORIES.values():
            if info['category'] == category:
                return info['description']
        return "Описание недоступно"
    
    def get_recommendations(self, aqi_value: int, category: str) -> str:
        """
        Генерация рекомендаций на русском языке на основе AQI.
        
        Args:
            aqi_value: Значение AQI
            category: Категория качества воздуха
            
        Returns:
            str: Рекомендации на русском языке
        """
        if aqi_value <= 50:
            return "Отличное качество воздуха. Идеальные условия для любых активностей на открытом воздухе."
        
        elif aqi_value <= 100:
            return "Хорошее качество воздуха. Можно заниматься любыми видами деятельности на открытом воздухе."
        
        elif aqi_value <= 150:
            return "Чувствительные люди (дети, пожилые, люди с заболеваниями сердца и легких) должны ограничить длительные или интенсивные физические нагрузки на открытом воздухе."
        
        elif aqi_value <= 200:
            return "Всем рекомендуется ограничить длительные или интенсивные физические нагрузки на открытом воздухе. Чувствительные люди должны избегать физических нагрузок на улице."
        
        elif aqi_value <= 300:
            return "Всем рекомендуется избегать физических нагрузок на открытом воздухе. Чувствительные люди должны оставаться в помещении и поддерживать низкий уровень активности."
        
        else:
            return "Чрезвычайная ситуация! Всем рекомендуется оставаться в помещении и избегать любых физических нагрузок. Закройте окна и используйте очистители воздуха."


class NMUDetector:
    """
    Детектор неблагоприятных метеорологических условий (НМУ).
    
    НМУ - это метеорологические условия, способствующие накоплению 
    вредных веществ в приземном слое атмосферы. Включает определение
    условий "Черное небо" при критическом уровне загрязнения.
    """
    
    def __init__(self):
        self.aqi_calculator = AQICalculator()
        
        # Весовые коэффициенты для разных загрязнителей (на основе их опасности)
        self.pollutant_weights = {
            'pm2_5': 2.0,  # Наиболее опасный для здоровья
            'pm10': 1.5,   # Высокая опасность
            'no2': 1.3,    # Средне-высокая опасность
            'so2': 1.2,    # Средняя опасность
            'o3': 1.1      # Умеренная опасность
        }
        
        # Пороговые значения для "Черное небо" (кратность превышения ПДК)
        self.blacksky_multipliers = {
            'pm2_5': 5.0,  # 5x ПДК для PM2.5
            'pm10': 5.0,   # 5x ПДК для PM10
            'no2': 10.0,   # 10x ПДК для NO2 (менее критичен)
            'so2': 10.0,   # 10x ПДК для SO2
            'o3': 3.0      # 3x ПДК для озона (более токсичен)
        }
    
    def check_nmu_risk(self, pollutants: Dict[str, float], weather_conditions: Optional[Dict[str, Any]] = None) -> str:
        """
        Определение риска неблагоприятных метеорологических условий (НМУ).
        
        Args:
            pollutants: Концентрации загрязнителей в мкг/м³
            weather_conditions: Метеорологические условия (опционально)
            
        Returns:
            str: Уровень риска НМУ ('low', 'medium', 'high', 'critical')
        """
        if not pollutants:
            logger.warning("No pollutant data provided for NMU risk assessment")
            return "unknown"
        
        # Проверка условий "Черное небо"
        if self.is_blacksky_conditions(pollutants):
            logger.warning("Black sky conditions detected - critical NMU risk")
            return "critical"
        
        # Расчет общего уровня загрязнения
        pollution_score = self._calculate_pollution_score(pollutants)
        
        if pollution_score is None:
            return "unknown"
        
        # Учет метеорологических условий (если доступны)
        weather_factor = self._calculate_weather_factor(weather_conditions)
        adjusted_score = pollution_score * weather_factor
        
        # Определение уровня риска НМУ
        risk_level = self._determine_risk_level(adjusted_score)
        
        logger.info(f"NMU risk assessment: pollution_score={pollution_score:.2f}, "
                   f"weather_factor={weather_factor:.2f}, adjusted_score={adjusted_score:.2f}, "
                   f"risk_level={risk_level}")
        
        return risk_level
    
    def is_blacksky_conditions(self, pollutants: Dict[str, float]) -> bool:
        """
        Определение условий "Черное небо" - критического уровня загрязнения воздуха.
        
        "Черное небо" объявляется при превышении концентраций загрязнителей
        в несколько раз от ПДК (зависит от типа загрязнителя).
        
        Args:
            pollutants: Концентрации загрязнителей в мкг/м³
            
        Returns:
            bool: True если условия "Черное небо"
        """
        if not pollutants:
            return False
        
        for pollutant, concentration in pollutants.items():
            if concentration is None or concentration <= 0:
                continue
                
            if pollutant in self.aqi_calculator.RU_STANDARDS:
                pdk = self.aqi_calculator.RU_STANDARDS[pollutant]['good']
                threshold_multiplier = self.blacksky_multipliers.get(pollutant, 5.0)
                threshold = pdk * threshold_multiplier
                
                if concentration >= threshold:
                    logger.warning(f"Black sky conditions detected: {pollutant}={concentration:.1f} мкг/м³ "
                                 f"(>{threshold_multiplier}x ПДК = {threshold:.1f} мкг/м³)")
                    return True
        
        return False
    
    def _calculate_pollution_score(self, pollutants: Dict[str, float]) -> Optional[float]:
        """Расчет общего балла загрязнения"""
        pollution_score = 0
        valid_pollutants = 0
        
        for pollutant, concentration in pollutants.items():
            if concentration is None or concentration <= 0:
                continue
                
            if pollutant in self.aqi_calculator.RU_STANDARDS:
                standards = self.aqi_calculator.RU_STANDARDS[pollutant]
                weight = self.pollutant_weights.get(pollutant, 1.0)
                
                # Нормализация относительно ПДК с более точной градацией
                normalized_score = self._normalize_concentration(concentration, standards)
                pollution_score += normalized_score * weight
                valid_pollutants += 1
        
        if valid_pollutants == 0:
            logger.warning("No valid pollutant data for NMU risk calculation")
            return None
        
        # Средневзвешенный балл загрязнения
        return pollution_score / valid_pollutants
    
    def _normalize_concentration(self, concentration: float, standards: Dict[str, float]) -> float:
        """Нормализация концентрации загрязнителя относительно ПДК"""
        if concentration <= standards['good']:
            return concentration / standards['good'] * 0.5
        elif concentration <= standards['moderate']:
            return 0.5 + (concentration - standards['good']) / (standards['moderate'] - standards['good']) * 0.5
        elif concentration <= standards['unhealthy_sensitive']:
            return 1.0 + (concentration - standards['moderate']) / (standards['unhealthy_sensitive'] - standards['moderate']) * 0.5
        elif concentration <= standards['unhealthy']:
            return 1.5 + (concentration - standards['unhealthy_sensitive']) / (standards['unhealthy'] - standards['unhealthy_sensitive']) * 0.5
        elif concentration <= standards['very_unhealthy']:
            return 2.0 + (concentration - standards['unhealthy']) / (standards['very_unhealthy'] - standards['unhealthy']) * 1.0
        else:
            # Для экстремально высоких концентраций
            return 3.0 + min(2.0, (concentration - standards['very_unhealthy']) / standards['very_unhealthy'])
    
    def _calculate_weather_factor(self, weather_conditions: Optional[Dict[str, Any]]) -> float:
        """
        Расчет метеорологического фактора для корректировки риска НМУ.
        
        В будущих версиях будет учитывать:
        - Скорость и направление ветра
        - Температурную инверсию
        - Влажность воздуха
        - Атмосферное давление
        """
        if not weather_conditions:
            return 1.0  # Нейтральный фактор при отсутствии данных
        
        factor = 1.0
        
        # Пример учета скорости ветра (будет расширено в будущих задачах)
        wind_speed = weather_conditions.get('wind_speed')
        if wind_speed is not None:
            if wind_speed < 1.0:  # Штиль способствует накоплению загрязнителей
                factor *= 1.5
            elif wind_speed < 3.0:  # Слабый ветер
                factor *= 1.2
            elif wind_speed > 10.0:  # Сильный ветер способствует рассеиванию
                factor *= 0.7
        
        return factor
    
    def _determine_risk_level(self, adjusted_score: float) -> str:
        """Определение уровня риска НМУ по скорректированному баллу"""
        if adjusted_score <= 0.5:
            return "low"
        elif adjusted_score <= 1.0:
            return "medium"
        elif adjusted_score <= 2.0:
            return "high"
        else:
            return "critical"
    
    def get_nmu_recommendations(self, risk_level: str, blacksky: bool = False) -> List[str]:
        """
        Получение рекомендаций при НМУ на русском языке.
        
        Args:
            risk_level: Уровень риска НМУ
            blacksky: Наличие условий "Черное небо"
            
        Returns:
            List[str]: Список рекомендаций
        """
        recommendations = []
        
        if blacksky:
            recommendations.extend([
                "РЕЖИМ 'ЧЕРНОЕ НЕБО' - КРИТИЧЕСКАЯ СИТУАЦИЯ!",
                "Немедленно покиньте улицу и зайдите в помещение",
                "Закройте все окна и двери",
                "Включите очистители воздуха на максимальную мощность",
                "Избегайте любых физических нагрузок",
                "При необходимости выхода используйте респиратор FFP2/N95"
            ])
        elif risk_level == "critical":
            recommendations.extend([
                "Критический уровень загрязнения воздуха",
                "Оставайтесь в помещении с закрытыми окнами",
                "Используйте очистители воздуха",
                "Полностью исключите физические нагрузки на улице",
                "Людям с хроническими заболеваниями обратиться к врачу"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "Высокий риск НМУ - ограничьте время на улице",
                "Избегайте физических нагрузок на открытом воздухе",
                "Чувствительным группам оставаться в помещении",
                "Используйте маски при выходе на улицу"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Умеренный риск НМУ - будьте осторожны",
                "Ограничьте интенсивные физические нагрузки на улице",
                "Чувствительные люди должны сократить время на улице"
            ])
        else:  # low
            recommendations.append("Низкий риск НМУ - обычные меры предосторожности")
        
        return recommendations


# Глобальный экземпляр детектора НМУ
_nmu_detector = NMUDetector()


def check_nmu_risk(pollutants: Dict[str, float], weather_conditions: Optional[Dict[str, Any]] = None) -> str:
    """
    Определение риска неблагоприятных метеорологических условий (НМУ).
    
    Обертка для совместимости с существующим кодом.
    
    Args:
        pollutants: Концентрации загрязнителей
        weather_conditions: Метеорологические условия (опционально)
        
    Returns:
        str: Уровень риска НМУ ('low', 'medium', 'high', 'critical')
    """
    return _nmu_detector.check_nmu_risk(pollutants, weather_conditions)


def is_blacksky_conditions(pollutants: Dict[str, float]) -> bool:
    """
    Определение условий "Черное небо" - критического уровня загрязнения воздуха.
    
    Обертка для совместимости с существующим кодом.
    
    Args:
        pollutants: Концентрации загрязнителей
        
    Returns:
        bool: True если условия "Черное небо"
    """
    return _nmu_detector.is_blacksky_conditions(pollutants)


def get_nmu_recommendations(risk_level: str, blacksky: bool = False) -> List[str]:
    """
    Получение рекомендаций при НМУ на русском языке.
    
    Args:
        risk_level: Уровень риска НМУ
        blacksky: Наличие условий "Черное небо"
        
    Returns:
        List[str]: Список рекомендаций
    """
    return _nmu_detector.get_nmu_recommendations(risk_level, blacksky)


def format_russian_timestamp(timestamp: datetime) -> str:
    """
    Форматирование временной метки для российского часового пояса.
    
    Args:
        timestamp: Временная метка в UTC
        
    Returns:
        str: Отформатированная строка времени
    """
    # Простое форматирование, в будущем можно добавить поддержку часовых поясов
    return timestamp.strftime("%d.%m.%Y %H:%M UTC")


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Валидация координат для российской территории.
    
    Args:
        lat: Широта
        lon: Долгота
        
    Returns:
        bool: True если координаты валидны для России
    """
    # Приблизительные границы России
    # Широта: от 41° до 82° с.ш.
    # Долгота: от 19° до 169° в.д.
    
    if not (41.0 <= lat <= 82.0):
        return False
    
    if not (19.0 <= lon <= 169.0):
        return False
    
    return True


def get_pollutant_name_russian(pollutant_code: str) -> str:
    """
    Получение русского названия загрязнителя по коду.
    
    Args:
        pollutant_code: Код загрязнителя (pm2_5, pm10, no2, so2, o3)
        
    Returns:
        str: Русское название загрязнителя
    """
    names = {
        'pm2_5': 'Мелкодисперсные частицы PM2.5',
        'pm10': 'Взвешенные частицы PM10',
        'no2': 'Диоксид азота',
        'so2': 'Диоксид серы',
        'o3': 'Озон'
    }
    
    return names.get(pollutant_code, pollutant_code.upper())