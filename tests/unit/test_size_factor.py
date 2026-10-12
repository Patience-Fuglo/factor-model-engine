import pandas as pd
import pytest

from factor_model.factors import compute_size


def test_compute_size_matches_hand_computed_log():
    market_cap = pd.Series([100.0, 1000.0, 10000.0])
    result = compute_size(market_cap)
    assert result.iloc[1] - result.iloc[0] == pytest.approx(result.iloc[2] - result.iloc[1])
    assert result.iloc[0] == pytest.approx(4.605170186)  # ln(100)


def test_compute_size_is_nan_for_non_positive_market_cap():
    market_cap = pd.Series([100.0, 0.0, -50.0])
    result = compute_size(market_cap)
    assert pd.notna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])


def test_compute_size_never_silently_zero_fills_missing_input():
    market_cap = pd.Series([100.0, float("nan")])
    result = compute_size(market_cap)
    assert pd.isna(result.iloc[1])
