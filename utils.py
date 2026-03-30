"""Compatibility facade for extracted domain logic and generic helpers."""

from __future__ import annotations

from datetime import datetime

from domain.aqi.calculator import AQICalculator
from domain.nmu.detector import (
    NMUDetector,
    check_nmu_risk,
    get_nmu_recommendations,
    is_blacksky_conditions,
)
from domain.pollutants.aggregator import get_pollutant_name_russian
from core.validation import CoordinateValidator


def format_russian_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime("%d.%m.%Y %H:%M UTC")


def validate_coordinates(lat: float, lon: float) -> bool:
    is_valid, _ = CoordinateValidator.validate_russian_territory(lat, lon)
    return is_valid
