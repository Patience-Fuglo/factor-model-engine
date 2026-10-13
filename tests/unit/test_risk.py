import numpy as np
import pandas as pd
import pytest

from factor_model.config import RiskConfig
from factor_model.data import ExposureSubject, FactorExposureRow, FactorReturnRow, PortfolioHolding
from factor_model.risk import compute_factor_covariance, compute_portfolio_risk, compute_specific_variance


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _return_row(period_start: str, period_end: str, factor_name: str, value: float) -> FactorReturnRow:
    return FactorReturnRow(
        period_start=_ts(period_start), period_end=_ts(period_end), factor_name=factor_name,
        factor_return=value, r_squared=0.1, n_obs=30, known_as_of=_ts(period_end),
    )


def test_factor_covariance_matches_hand_computed_sample_covariance():
    rows = [
        _return_row("2020-01-31", "2020-02-29", "size", 0.01),
        _return_row("2020-01-31", "2020-02-29", "momentum", 0.02),
        _return_row("2020-02-29", "2020-03-31", "size", -0.01),
        _return_row("2020-02-29", "2020-03-31", "momentum", 0.03),
        _return_row("2020-03-31", "2020-04-30", "size", 0.02),
        _return_row("2020-03-31", "2020-04-30", "momentum", -0.01),
    ]
    result = compute_factor_covariance(rows, RiskConfig(min_periods_for_covariance=2))
    expected = pd.DataFrame({"momentum": [0.02, 0.03, -0.01], "size": [0.01, -0.01, 0.02]}).cov()
    expected.index.name = "factor_name"
    expected.columns.name = "factor_name"
    pd.testing.assert_frame_equal(result.covariance, expected)
    assert result.n_periods == 3
    assert result.warnings == []


def test_factor_covariance_warns_on_too_few_periods():
    rows = [
        _return_row("2020-01-31", "2020-02-29", "size", 0.01),
        _return_row("2020-02-29", "2020-03-31", "size", -0.01),
    ]
    result = compute_factor_covariance(rows, RiskConfig(min_periods_for_covariance=24))
    assert any("unstable" in w for w in result.warnings)


def test_factor_covariance_handles_empty_input():
    result = compute_factor_covariance([], RiskConfig())
    assert result.n_periods == 0
    assert result.warnings


def test_specific_variance_matches_hand_computed_variance():
    specific_returns = pd.DataFrame({"t1": [0.01, 0.02], "t2": [-0.01, 0.0], "t3": [0.02, -0.01]}, index=["A", "B"])
    result = compute_specific_variance(specific_returns)
    assert result["A"] == pytest.approx(specific_returns.loc["A"].var(ddof=1))
    assert result["B"] == pytest.approx(specific_returns.loc["B"].var(ddof=1))


def test_specific_variance_is_nan_with_too_few_real_observations():
    specific_returns = pd.DataFrame({"t1": [0.01, float("nan")], "t2": [float("nan"), float("nan")]}, index=["A", "B"])
    result = compute_specific_variance(specific_returns, min_periods=2)
    assert pd.isna(result["A"])  # only 1 real observation
    assert pd.isna(result["B"])  # 0 real observations


def test_portfolio_factor_variance_matches_the_real_quadratic_form():
    exposure_rows = [
        FactorExposureRow(subject_type=ExposureSubject.PORTFOLIO, subject_id="P", as_of=_ts("2020-06-30"),
                           factor_name="size", exposure_value=0.5, source_known_as_of=_ts("2020-06-30")),
        FactorExposureRow(subject_type=ExposureSubject.PORTFOLIO, subject_id="P", as_of=_ts("2020-06-30"),
                           factor_name="momentum", exposure_value=-0.3, source_known_as_of=_ts("2020-06-30")),
    ]
    covariance = pd.DataFrame({"size": [0.02, 0.005], "momentum": [0.005, 0.01]}, index=["size", "momentum"])
    holdings = [PortfolioHolding(portfolio_id="P", security_id="A", as_of=_ts("2020-06-30"), weight=1.0, known_as_of=_ts("2020-06-30"))]
    specific_var = pd.Series({"A": 0.04})

    result = compute_portfolio_risk(exposure_rows, covariance, holdings, specific_var, "P", RiskConfig())
    w = np.array([0.5, -0.3])
    expected_factor_var = float(w @ covariance.to_numpy() @ w)
    assert result.factor_variance == pytest.approx(expected_factor_var)
    assert sum(result.factor_risk_contribution.values()) == pytest.approx(result.factor_variance)
    assert result.specific_variance == pytest.approx(0.04)
    assert result.total_variance == pytest.approx(expected_factor_var + 0.04)


def test_portfolio_risk_flags_single_factor_concentration():
    exposure_rows = [
        FactorExposureRow(subject_type=ExposureSubject.PORTFOLIO, subject_id="P", as_of=_ts("2020-06-30"),
                           factor_name="size", exposure_value=5.0, source_known_as_of=_ts("2020-06-30")),
        FactorExposureRow(subject_type=ExposureSubject.PORTFOLIO, subject_id="P", as_of=_ts("2020-06-30"),
                           factor_name="momentum", exposure_value=0.01, source_known_as_of=_ts("2020-06-30")),
    ]
    covariance = pd.DataFrame({"size": [0.02, 0.0], "momentum": [0.0, 0.01]}, index=["size", "momentum"])
    holdings = [PortfolioHolding(portfolio_id="P", security_id="A", as_of=_ts("2020-06-30"), weight=1.0, known_as_of=_ts("2020-06-30"))]
    result = compute_portfolio_risk(exposure_rows, covariance, holdings, pd.Series({"A": 0.0}), "P", RiskConfig(max_single_factor_variance_share=0.5))
    assert any("size" in w and "concentration" in w for w in result.warnings)


def test_portfolio_risk_flags_low_effective_n_positions():
    exposure_rows = [
        FactorExposureRow(subject_type=ExposureSubject.PORTFOLIO, subject_id="P", as_of=_ts("2020-06-30"),
                           factor_name="size", exposure_value=0.1, source_known_as_of=_ts("2020-06-30")),
    ]
    covariance = pd.DataFrame({"size": [0.01]}, index=["size"])
    holdings = [PortfolioHolding(portfolio_id="P", security_id="A", as_of=_ts("2020-06-30"), weight=0.9, known_as_of=_ts("2020-06-30")),
                PortfolioHolding(portfolio_id="P", security_id="B", as_of=_ts("2020-06-30"), weight=0.1, known_as_of=_ts("2020-06-30"))]
    result = compute_portfolio_risk(exposure_rows, covariance, holdings, pd.Series({"A": 0.0, "B": 0.0}), "P", RiskConfig(min_effective_n_positions=5.0))
    assert result.effective_n_positions == pytest.approx(1.0 / (0.9**2 + 0.1**2))
    assert any("effective n positions" in w for w in result.warnings)


def test_portfolio_risk_excludes_missing_specific_variance_rather_than_zero_filling():
    exposure_rows = [
        FactorExposureRow(subject_type=ExposureSubject.PORTFOLIO, subject_id="P", as_of=_ts("2020-06-30"),
                           factor_name="size", exposure_value=0.1, source_known_as_of=_ts("2020-06-30")),
    ]
    covariance = pd.DataFrame({"size": [0.01]}, index=["size"])
    holdings = [PortfolioHolding(portfolio_id="P", security_id="A", as_of=_ts("2020-06-30"), weight=1.0, known_as_of=_ts("2020-06-30"))]
    result = compute_portfolio_risk(exposure_rows, covariance, holdings, pd.Series(dtype=float), "P", RiskConfig())
    assert result.specific_variance == 0.0
    assert any("no specific-variance estimate" in w for w in result.warnings)
