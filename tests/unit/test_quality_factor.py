import pandas as pd
import pytest

from factor_model.config import QualityConfig
from factor_model.factors import compute_leverage, compute_profitability, compute_quality_composite


def test_compute_profitability_matches_hand_computed_roe_proxy():
    eps = pd.Series([2.0, 4.0])
    bvps = pd.Series([10.0, 8.0])
    result = compute_profitability(eps, bvps)
    assert result.iloc[0] == pytest.approx(0.2)
    assert result.iloc[1] == pytest.approx(0.5)


def test_compute_profitability_is_nan_for_non_positive_book_value():
    eps = pd.Series([2.0, 2.0])
    bvps = pd.Series([0.0, -5.0])
    result = compute_profitability(eps, bvps)
    assert result.isna().all()


def test_compute_leverage_matches_hand_computed_ratio():
    debt = pd.Series([40.0, 10.0])
    assets = pd.Series([100.0, 50.0])
    result = compute_leverage(debt, assets)
    assert result.iloc[0] == pytest.approx(0.4)
    assert result.iloc[1] == pytest.approx(0.2)


def test_compute_leverage_is_nan_for_invalid_inputs():
    debt = pd.Series([10.0, -5.0, 10.0])
    assets = pd.Series([0.0, 100.0, 100.0])
    result = compute_leverage(debt, assets)
    assert pd.isna(result.iloc[0])  # assets <= 0
    assert pd.isna(result.iloc[1])  # negative debt is invalid, never clipped to zero
    assert pd.notna(result.iloc[2])


def test_quality_composite_rewards_high_profitability_and_low_leverage():
    profitability_z = pd.Series({"A": 2.0, "B": -2.0})
    leverage_z = pd.Series({"A": -2.0, "B": 2.0})  # A is safe (low leverage), B is risky
    composite = compute_quality_composite(profitability_z, leverage_z, QualityConfig())
    assert composite["A"] > composite["B"]


def test_quality_composite_falls_back_to_the_one_real_pillar_when_the_other_is_missing():
    profitability_z = pd.Series({"A": 1.5, "B": float("nan")})
    leverage_z = pd.Series({"A": float("nan"), "B": -0.5})
    composite = compute_quality_composite(profitability_z, leverage_z, QualityConfig())
    assert composite["A"] == pytest.approx(1.5)  # only profitability pillar real for A
    assert composite["B"] == pytest.approx(0.5)  # only safety (=-leverage) pillar real for B


def test_quality_composite_is_nan_when_both_pillars_are_missing():
    profitability_z = pd.Series({"A": float("nan")})
    leverage_z = pd.Series({"A": float("nan")})
    composite = compute_quality_composite(profitability_z, leverage_z, QualityConfig())
    assert composite.isna().all()


def test_quality_config_rejects_negative_weights():
    with pytest.raises(ValueError):
        QualityConfig(profitability_weight=-0.1)


def test_quality_config_rejects_both_weights_zero():
    with pytest.raises(ValueError):
        QualityConfig(profitability_weight=0.0, leverage_weight=0.0)
