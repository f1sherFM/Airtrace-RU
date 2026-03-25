"""NMU detection domain logic."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from domain.aqi.calculator import AQICalculator

logger = logging.getLogger(__name__)


class NMUDetector:
    """Detector for adverse meteorological conditions (NMU)."""

    def __init__(self):
        self.aqi_calculator = AQICalculator()
        self.pollutant_weights = {
            "pm2_5": 2.0,
            "pm10": 1.5,
            "no2": 1.3,
            "so2": 1.2,
            "o3": 1.1,
        }
        self.blacksky_multipliers = {
            "pm2_5": 5.0,
            "pm10": 5.0,
            "no2": 10.0,
            "so2": 10.0,
            "o3": 3.0,
        }

    def check_nmu_risk(
        self,
        pollutants: Dict[str, float],
        weather_conditions: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not pollutants:
            logger.warning("No pollutant data provided for NMU risk assessment")
            return "unknown"

        if self.is_blacksky_conditions(pollutants):
            logger.warning("Black sky conditions detected - critical NMU risk")
            return "critical"

        pollution_score = self._calculate_pollution_score(pollutants)
        if pollution_score is None:
            return "unknown"

        weather_factor = self._calculate_weather_factor(weather_conditions)
        adjusted_score = pollution_score * weather_factor
        risk_level = self._determine_risk_level(adjusted_score)
        logger.info(
            "NMU risk assessment: pollution_score=%.2f, weather_factor=%.2f, adjusted_score=%.2f, risk_level=%s",
            pollution_score,
            weather_factor,
            adjusted_score,
            risk_level,
        )
        return risk_level

    def is_blacksky_conditions(self, pollutants: Dict[str, float]) -> bool:
        if not pollutants:
            return False

        for pollutant, concentration in pollutants.items():
            if concentration is None or concentration <= 0:
                continue
            if pollutant in self.aqi_calculator.RU_STANDARDS:
                pdk = self.aqi_calculator.RU_STANDARDS[pollutant]["good"]
                threshold_multiplier = self.blacksky_multipliers.get(pollutant, 5.0)
                threshold = pdk * threshold_multiplier
                if concentration >= threshold:
                    logger.warning(
                        "Black sky conditions detected: %s=%.1f мкг/м³ (>%.1fx ПДК = %.1f мкг/м³)",
                        pollutant,
                        concentration,
                        threshold_multiplier,
                        threshold,
                    )
                    return True
        return False

    def _calculate_pollution_score(self, pollutants: Dict[str, float]) -> Optional[float]:
        pollution_score = 0.0
        valid_pollutants = 0

        for pollutant, concentration in pollutants.items():
            if concentration is None or concentration <= 0:
                continue
            if pollutant in self.aqi_calculator.RU_STANDARDS:
                standards = self.aqi_calculator.RU_STANDARDS[pollutant]
                weight = self.pollutant_weights.get(pollutant, 1.0)
                normalized_score = self._normalize_concentration(concentration, standards)
                pollution_score += normalized_score * weight
                valid_pollutants += 1

        if valid_pollutants == 0:
            logger.warning("No valid pollutant data for NMU risk calculation")
            return None
        return pollution_score / valid_pollutants

    def _normalize_concentration(self, concentration: float, standards: Dict[str, float]) -> float:
        if concentration <= standards["good"]:
            return concentration / standards["good"] * 0.5
        if concentration <= standards["moderate"]:
            return 0.5 + (concentration - standards["good"]) / (standards["moderate"] - standards["good"]) * 0.5
        if concentration <= standards["unhealthy_sensitive"]:
            return 1.0 + (
                (concentration - standards["moderate"])
                / (standards["unhealthy_sensitive"] - standards["moderate"])
                * 0.5
            )
        if concentration <= standards["unhealthy"]:
            return 1.5 + (
                (concentration - standards["unhealthy_sensitive"])
                / (standards["unhealthy"] - standards["unhealthy_sensitive"])
                * 0.5
            )
        if concentration <= standards["very_unhealthy"]:
            return 2.0 + (concentration - standards["unhealthy"]) / (standards["very_unhealthy"] - standards["unhealthy"]) * 1.0
        return 3.0 + min(2.0, (concentration - standards["very_unhealthy"]) / standards["very_unhealthy"])

    def _calculate_weather_factor(self, weather_conditions: Optional[Dict[str, Any]]) -> float:
        if not weather_conditions:
            return 1.0
        factor = 1.0
        wind_speed = weather_conditions.get("wind_speed")
        if wind_speed is not None:
            if wind_speed < 1.0:
                factor *= 1.5
            elif wind_speed < 3.0:
                factor *= 1.2
            elif wind_speed > 10.0:
                factor *= 0.7
        return factor

    def _determine_risk_level(self, adjusted_score: float) -> str:
        if adjusted_score <= 0.5:
            return "low"
        if adjusted_score <= 1.0:
            return "medium"
        if adjusted_score <= 2.0:
            return "high"
        return "critical"

    def get_nmu_recommendations(self, risk_level: str, blacksky: bool = False) -> List[str]:
        recommendations: List[str] = []
        if blacksky:
            recommendations.extend(
                [
                    "РЕЖИМ 'ЧЕРНОЕ НЕБО' - КРИТИЧЕСКАЯ СИТУАЦИЯ!",
                    "Немедленно покиньте улицу и зайдите в помещение",
                    "Закройте все окна и двери",
                    "Включите очистители воздуха на максимальную мощность",
                    "Избегайте любых физических нагрузок",
                    "При необходимости выхода используйте респиратор FFP2/N95",
                ]
            )
        elif risk_level == "critical":
            recommendations.extend(
                [
                    "Критический уровень загрязнения воздуха",
                    "Оставайтесь в помещении с закрытыми окнами",
                    "Используйте очистители воздуха",
                    "Полностью исключите физические нагрузки на улице",
                    "Людям с хроническими заболеваниями обратиться к врачу",
                ]
            )
        elif risk_level == "high":
            recommendations.extend(
                [
                    "Высокий риск НМУ - ограничьте время на улице",
                    "Избегайте физических нагрузок на открытом воздухе",
                    "Чувствительным группам оставаться в помещении",
                    "Используйте маски при выходе на улицу",
                ]
            )
        elif risk_level == "medium":
            recommendations.extend(
                [
                    "Умеренный риск НМУ - будьте осторожны",
                    "Ограничьте интенсивные физические нагрузки на улице",
                    "Чувствительные люди должны сократить время на улице",
                ]
            )
        else:
            recommendations.append("Низкий риск НМУ - обычные меры предосторожности")
        return recommendations


_nmu_detector = NMUDetector()


def check_nmu_risk(
    pollutants: Dict[str, float],
    weather_conditions: Optional[Dict[str, Any]] = None,
) -> str:
    return _nmu_detector.check_nmu_risk(pollutants, weather_conditions)


def is_blacksky_conditions(pollutants: Dict[str, float]) -> bool:
    return _nmu_detector.is_blacksky_conditions(pollutants)


def get_nmu_recommendations(risk_level: str, blacksky: bool = False) -> List[str]:
    return _nmu_detector.get_nmu_recommendations(risk_level, blacksky)
