"""Compatibility facade for confidence scoring."""

from __future__ import annotations

from typing import Tuple

from domain.confidence.calculator import ConfidenceCalculator, ConfidenceInputs


_confidence_calculator = ConfidenceCalculator()


def calculate_confidence(inputs: ConfidenceInputs) -> Tuple[float, str]:
    return _confidence_calculator.calculate(inputs)
