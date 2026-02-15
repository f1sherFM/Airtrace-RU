"""
Confidence scoring engine for data trust and provenance.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ConfidenceInputs:
    data_source: str
    source_available: bool
    cache_age_seconds: int
    fallback_used: bool


def _source_base_score(data_source: str) -> float:
    source = (data_source or "fallback").lower()
    mapping = {
        "live": 0.90,
        "historical": 0.86,
        "forecast": 0.74,
        "fallback": 0.50,
    }
    return mapping.get(source, 0.50)


def calculate_confidence(inputs: ConfidenceInputs) -> Tuple[float, str]:
    """
    Calculate confidence score [0..1] and explanation.

    The score degrades monotonically with:
    - source availability loss
    - cache age growth
    - fallback usage
    """
    score = _source_base_score(inputs.data_source)
    reasons = [f"source={inputs.data_source.lower()}"]

    if not inputs.source_available:
        score -= 0.15
        reasons.append("source_unavailable")
    else:
        reasons.append("source_available")

    age = max(0, int(inputs.cache_age_seconds))
    # Degrade linearly up to -0.30 by 6 hours.
    age_penalty = min(0.30, (age / 21600.0) * 0.30)
    score -= age_penalty
    reasons.append(f"cache_age={age}s")

    if inputs.fallback_used:
        score -= 0.22
        reasons.append("fallback_used")
    else:
        reasons.append("no_fallback")

    bounded = max(0.0, min(1.0, round(score, 3)))
    explanation = "confidence derived from " + ", ".join(reasons)
    return bounded, explanation
