"""AQI calculation domain logic."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class AQICalculator:
    """Russian AQI calculator based on ПДК standards."""

    RU_STANDARDS = {
        "pm2_5": {
            "good": 25,
            "moderate": 50,
            "unhealthy_sensitive": 75,
            "unhealthy": 100,
            "very_unhealthy": 150,
            "hazardous": 250,
        },
        "pm10": {
            "good": 50,
            "moderate": 100,
            "unhealthy_sensitive": 150,
            "unhealthy": 200,
            "very_unhealthy": 300,
            "hazardous": 500,
        },
        "no2": {
            "good": 40,
            "moderate": 80,
            "unhealthy_sensitive": 120,
            "unhealthy": 160,
            "very_unhealthy": 240,
            "hazardous": 400,
        },
        "so2": {
            "good": 50,
            "moderate": 100,
            "unhealthy_sensitive": 150,
            "unhealthy": 200,
            "very_unhealthy": 300,
            "hazardous": 500,
        },
        "o3": {
            "good": 100,
            "moderate": 160,
            "unhealthy_sensitive": 200,
            "unhealthy": 240,
            "very_unhealthy": 300,
            "hazardous": 400,
        },
    }

    AQI_CATEGORIES = {
        (0, 50): {
            "category": "Хорошее",
            "color": "#00E400",
            "description": "Качество воздуха считается удовлетворительным, загрязнение воздуха представляет незначительный риск или не представляет риска",
        },
        (51, 100): {
            "category": "Умеренное",
            "color": "#FFFF00",
            "description": "Качество воздуха приемлемо для большинства людей. Однако чувствительные люди могут испытывать незначительные проблемы",
        },
        (101, 150): {
            "category": "Вредно для чувствительных групп",
            "color": "#FF7E00",
            "description": "Представители чувствительных групп могут испытывать проблемы со здоровьем. Широкая общественность, как правило, не пострадает",
        },
        (151, 200): {
            "category": "Вредно",
            "color": "#FF0000",
            "description": "Каждый может начать испытывать проблемы со здоровьем; представители чувствительных групп могут испытывать более серьезные проблемы",
        },
        (201, 300): {
            "category": "Очень вредно",
            "color": "#8F3F97",
            "description": "Предупреждения о вреде для здоровья при чрезвычайных условиях. Вероятность воздействия на все население",
        },
        (301, 500): {
            "category": "Опасно",
            "color": "#7E0023",
            "description": "Чрезвычайная ситуация: все население подвержено риску серьезных проблем со здоровьем",
        },
    }

    def calculate_aqi(self, pollutants: Dict[str, float]) -> Tuple[int, str, str]:
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
        logger.info("Calculated AQI: %s, Category: %s, Dominant: %s", max_aqi, category, dominant_pollutant)
        return max_aqi, category, color

    def _calculate_pollutant_aqi(self, pollutant: str, concentration: float) -> int:
        standards = self.RU_STANDARDS[pollutant]
        if concentration <= standards["good"]:
            return self._linear_interpolation(concentration, 0, standards["good"], 0, 50)
        if concentration <= standards["moderate"]:
            return self._linear_interpolation(concentration, standards["good"], standards["moderate"], 51, 100)
        if concentration <= standards["unhealthy_sensitive"]:
            return self._linear_interpolation(
                concentration,
                standards["moderate"],
                standards["unhealthy_sensitive"],
                101,
                150,
            )
        if concentration <= standards["unhealthy"]:
            return self._linear_interpolation(
                concentration,
                standards["unhealthy_sensitive"],
                standards["unhealthy"],
                151,
                200,
            )
        if concentration <= standards["very_unhealthy"]:
            return self._linear_interpolation(
                concentration,
                standards["unhealthy"],
                standards["very_unhealthy"],
                201,
                300,
            )
        if concentration <= standards["hazardous"]:
            return self._linear_interpolation(
                concentration,
                standards["very_unhealthy"],
                standards["hazardous"],
                301,
                400,
            )
        return min(500, int(400 + (concentration - standards["hazardous"]) / standards["hazardous"] * 100))

    def _linear_interpolation(
        self,
        concentration: float,
        c_low: float,
        c_high: float,
        aqi_low: int,
        aqi_high: int,
    ) -> int:
        if c_high == c_low:
            return aqi_low
        aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
        return int(round(aqi))

    def _get_aqi_category_and_color(self, aqi_value: int) -> Tuple[str, str]:
        for (min_aqi, max_aqi), info in self.AQI_CATEGORIES.items():
            if min_aqi <= aqi_value <= max_aqi:
                return info["category"], info["color"]
        return "Критически опасно", "#7E0023"

    def get_category_description(self, category: str) -> str:
        for info in self.AQI_CATEGORIES.values():
            if info["category"] == category:
                return info["description"]
        return "Описание недоступно"

    def get_recommendations(self, aqi_value: int, category: str) -> str:
        if aqi_value <= 50:
            return "Отличное качество воздуха. Идеальные условия для любых активностей на открытом воздухе."
        if aqi_value <= 100:
            return "Хорошее качество воздуха. Можно заниматься любыми видами деятельности на открытом воздухе."
        if aqi_value <= 150:
            return "Чувствительные люди (дети, пожилые, люди с заболеваниями сердца и легких) должны ограничить длительные или интенсивные физические нагрузки на открытом воздухе."
        if aqi_value <= 200:
            return "Всем рекомендуется ограничить длительные или интенсивные физические нагрузки на открытом воздухе. Чувствительные люди должны избегать физических нагрузок на улице."
        if aqi_value <= 300:
            return "Всем рекомендуется избегать физических нагрузок на открытом воздухе. Чувствительные люди должны оставаться в помещении и поддерживать низкий уровень активности."
        return "Чрезвычайная ситуация! Всем рекомендуется оставаться в помещении и избегать любых физических нагрузок. Закройте окна и используйте очистители воздуха."
