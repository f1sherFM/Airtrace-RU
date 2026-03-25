"""Pollutant naming and normalization helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Optional


POLLUTANT_NAMES_RU = {
    "pm2_5": "Мелкодисперсные частицы PM2.5",
    "pm10": "Взвешенные частицы PM10",
    "no2": "Диоксид азота",
    "so2": "Диоксид серы",
    "o3": "Озон",
}


def get_pollutant_name_russian(pollutant_code: str) -> str:
    return POLLUTANT_NAMES_RU.get(pollutant_code, pollutant_code.upper())


def normalize_pollutant_mapping(pollutants: Optional[Dict[str, Optional[float]]]) -> Dict[str, float]:
    if not pollutants:
        return {}
    normalized: Dict[str, float] = {}
    for code, value in pollutants.items():
        if code not in POLLUTANT_NAMES_RU:
            continue
        if value is None:
            continue
        normalized[code] = float(value)
    return normalized


def ordered_pollutant_codes(codes: Iterable[str]) -> list[str]:
    known = [code for code in POLLUTANT_NAMES_RU if code in set(codes)]
    unknown = sorted(code for code in set(codes) if code not in POLLUTANT_NAMES_RU)
    return known + unknown
