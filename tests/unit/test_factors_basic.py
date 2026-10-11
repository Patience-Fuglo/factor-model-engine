import pandas as pd
import pytest

from factor_model.config import MomentumConfig
from factor_model.factors import compute_book_to_price, compute_momentum


def test_compute_momentum_matches_hand_computed_return():
    # 13 real monthly prices: index 0..12. 12-1 momentum at t=12 looks from
    # t-12=0 to t-1=11.
    prices = pd.Series([100.0 * (1.03 ** i) for i in range(14)])
    cfg = MomentumConfig(lookback_months=12, skip_recent_months=1)
    result = compute_momentum(prices, cfg)
    expected_at_12 = prices.iloc[11] / prices.iloc[0] - 1.0
    assert result.iloc[12] == pytest.approx(expected_at_12)


def test_compute_momentum_is_nan_before_enough_real_history():
    prices = pd.Series([100.0, 101.0, 102.0])
    cfg = MomentumConfig(lookback_months=12, skip_recent_months=1)
    result = compute_momentum(prices, cfg)
    assert result.isna().all()


def test_compute_momentum_skip_window_excludes_the_most_recent_period():
    # construct prices where the final month has a large real move -- a
    # skip_recent_months=1 momentum reading must NOT reflect that move
    prices = pd.Series([100.0] * 13 + [1000.0])
    cfg = MomentumConfig(lookback_months=12, skip_recent_months=1)
    result = compute_momentum(prices, cfg)
    assert result.iloc[13] == pytest.approx(0.0)  # excludes the final +900% move


def test_compute_momentum_is_configurable_different_windows_give_different_results():
    prices = pd.Series([100.0 * (1.01 ** i) for i in range(20)])
    result_12_1 = compute_momentum(prices, MomentumConfig(lookback_months=12, skip_recent_months=1))
    result_6_0 = compute_momentum(prices, MomentumConfig(lookback_months=6, skip_recent_months=0))
    assert not result_12_1.equals(result_6_0)


def test_compute_book_to_price_matches_hand_computed_ratio():
    book = pd.Series([10.0, 20.0])
    price = pd.Series([100.0, 50.0])
    result = compute_book_to_price(book, price)
    assert result.iloc[0] == pytest.approx(0.1)
    assert result.iloc[1] == pytest.approx(0.4)


def test_compute_book_to_price_is_nan_for_non_positive_price():
    book = pd.Series([10.0, 10.0])
    price = pd.Series([0.0, -5.0])
    result = compute_book_to_price(book, price)
    assert result.isna().all()


def test_compute_book_to_price_never_silently_zero_fills_missing_book_value():
    book = pd.Series([10.0, float("nan")])
    price = pd.Series([100.0, 100.0])
    result = compute_book_to_price(book, price)
    assert pd.isna(result.iloc[1])
