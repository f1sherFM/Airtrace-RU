"""
Property-based tests for confidence scoring monotonic behavior (Issue 2.2).
"""

from hypothesis import given, strategies as st

from confidence_scoring import ConfidenceInputs, calculate_confidence


@given(
    data_source=st.sampled_from(["live", "historical", "forecast", "fallback", "unknown"]),
    source_available=st.booleans(),
    cache_age_seconds=st.integers(min_value=0, max_value=200000),
    fallback_used=st.booleans(),
)
def test_confidence_score_in_valid_range(data_source, source_available, cache_age_seconds, fallback_used):
    score, explanation = calculate_confidence(
        ConfidenceInputs(
            data_source=data_source,
            source_available=source_available,
            cache_age_seconds=cache_age_seconds,
            fallback_used=fallback_used,
        )
    )
    assert 0.0 <= score <= 1.0
    assert isinstance(explanation, str) and len(explanation) > 0


@given(
    data_source=st.sampled_from(["live", "historical", "forecast", "fallback"]),
    source_available=st.booleans(),
    fallback_used=st.booleans(),
    cache_age_small=st.integers(min_value=0, max_value=50000),
    delta=st.integers(min_value=0, max_value=50000),
)
def test_confidence_monotonic_decreases_with_cache_age(
    data_source, source_available, fallback_used, cache_age_small, delta
):
    older = cache_age_small + delta
    score_small, _ = calculate_confidence(
        ConfidenceInputs(
            data_source=data_source,
            source_available=source_available,
            cache_age_seconds=cache_age_small,
            fallback_used=fallback_used,
        )
    )
    score_older, _ = calculate_confidence(
        ConfidenceInputs(
            data_source=data_source,
            source_available=source_available,
            cache_age_seconds=older,
            fallback_used=fallback_used,
        )
    )
    assert score_older <= score_small


@given(
    data_source=st.sampled_from(["live", "historical", "forecast", "fallback"]),
    source_available=st.booleans(),
    cache_age_seconds=st.integers(min_value=0, max_value=50000),
)
def test_confidence_fallback_penalty(data_source, source_available, cache_age_seconds):
    score_no_fallback, _ = calculate_confidence(
        ConfidenceInputs(
            data_source=data_source,
            source_available=source_available,
            cache_age_seconds=cache_age_seconds,
            fallback_used=False,
        )
    )
    score_fallback, _ = calculate_confidence(
        ConfidenceInputs(
            data_source=data_source,
            source_available=source_available,
            cache_age_seconds=cache_age_seconds,
            fallback_used=True,
        )
    )
    assert score_fallback <= score_no_fallback


@given(
    data_source=st.sampled_from(["live", "historical", "forecast", "fallback"]),
    cache_age_seconds=st.integers(min_value=0, max_value=50000),
    fallback_used=st.booleans(),
)
def test_confidence_source_unavailable_penalty(data_source, cache_age_seconds, fallback_used):
    score_available, _ = calculate_confidence(
        ConfidenceInputs(
            data_source=data_source,
            source_available=True,
            cache_age_seconds=cache_age_seconds,
            fallback_used=fallback_used,
        )
    )
    score_unavailable, _ = calculate_confidence(
        ConfidenceInputs(
            data_source=data_source,
            source_available=False,
            cache_age_seconds=cache_age_seconds,
            fallback_used=fallback_used,
        )
    )
    assert score_unavailable <= score_available
