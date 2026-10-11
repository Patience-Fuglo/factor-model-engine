import numpy as np
import pandas as pd
import pytest

from factor_model.config import StandardizationConfig, WinsorizationConfig
from factor_model.factors import standardize, winsorize


def test_winsorize_clips_to_real_quantile_bounds():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    cfg = WinsorizationConfig(lower_pct=0.0, upper_pct=0.8)
    result = winsorize(values, cfg)
    assert result.max() < 100.0
    assert result.iloc[0] == 1.0  # unaffected lower value stays as-is


def test_winsorize_leaves_real_nan_untouched():
    values = pd.Series([1.0, 2.0, float("nan"), 100.0])
    cfg = WinsorizationConfig(lower_pct=0.0, upper_pct=0.9)
    result = winsorize(values, cfg)
    assert pd.isna(result.iloc[2])


def test_winsorize_handles_all_nan_gracefully():
    values = pd.Series([float("nan"), float("nan")])
    result = winsorize(values, WinsorizationConfig())
    assert result.isna().all()


def test_standardize_zscore_has_zero_mean_and_unit_std():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = standardize(values, StandardizationConfig(method="zscore"))
    assert result.mean() == pytest.approx(0.0, abs=1e-9)
    assert result.std(ddof=1) == pytest.approx(1.0, abs=1e-9)


def test_standardize_zscore_never_treats_nan_as_zero():
    values = pd.Series([1.0, 2.0, float("nan"), 4.0, 5.0])
    result = standardize(values, StandardizationConfig(method="zscore"))
    assert pd.isna(result.iloc[2])
    assert not result.dropna().eq(0.0).all()


def test_standardize_zscore_returns_nan_for_constant_series():
    values = pd.Series([5.0, 5.0, 5.0])
    result = standardize(values, StandardizationConfig(method="zscore"))
    assert result.isna().all()


def test_standardize_rank_is_between_zero_and_one():
    values = pd.Series([3.0, 1.0, 2.0])
    result = standardize(values, StandardizationConfig(method="rank"))
    assert (result.dropna() > 0).all()
    assert (result.dropna() <= 1).all()


def test_standardize_minmax_scales_to_unit_interval():
    values = pd.Series([10.0, 20.0, 30.0])
    result = standardize(values, StandardizationConfig(method="minmax"))
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_standardize_too_few_real_values_returns_all_nan():
    values = pd.Series([1.0, float("nan"), float("nan")])
    result = standardize(values, StandardizationConfig(method="zscore"))
    assert result.isna().all()
