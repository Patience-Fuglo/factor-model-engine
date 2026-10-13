import pandas as pd
import pytest

from factor_model.attribution import compute_attribution
from factor_model.data import ExposureSubject, FactorExposureRow, FactorReturnRow


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _exposure(subject_id: str, factor_name: str, value: float, as_of: str = "2020-05-31") -> FactorExposureRow:
    return FactorExposureRow(
        subject_type=ExposureSubject.STOCK, subject_id=subject_id, as_of=_ts(as_of), factor_name=factor_name,
        exposure_value=value, source_known_as_of=_ts(as_of),
    )


def _factor_return(factor_name: str, value: float, period_start: str = "2020-05-31", period_end: str = "2020-06-30") -> FactorReturnRow:
    return FactorReturnRow(
        period_start=_ts(period_start), period_end=_ts(period_end), factor_name=factor_name,
        factor_return=value, r_squared=0.2, n_obs=30, known_as_of=_ts(period_end),
    )


def test_attribution_reconciles_exactly_to_realized_return():
    exposures = [_exposure("A", "momentum", 0.5), _exposure("A", "value", -0.2)]
    returns = [_factor_return("momentum", 0.02), _factor_return("value", 0.01)]
    realized_return = 0.5 * 0.02 + (-0.2) * 0.01 + 0.003  # a real, deliberate residual of 0.003

    report = compute_attribution(
        exposures, returns, realized_return, ExposureSubject.STOCK, "A", _ts("2020-05-31"), _ts("2020-06-30")
    )
    total = sum(row.contribution_to_return for row in report.rows)
    assert total == pytest.approx(realized_return)
    assert report.residual_return == pytest.approx(0.003)
    assert report.explained_return == pytest.approx(0.5 * 0.02 + (-0.2) * 0.01)


def test_attribution_contribution_values_match_hand_computation():
    exposures = [_exposure("A", "size", 1.5)]
    returns = [_factor_return("size", 0.04)]
    report = compute_attribution(exposures, returns, 0.06, ExposureSubject.STOCK, "A", _ts("2020-05-31"), _ts("2020-06-30"))
    size_row = next(r for r in report.rows if r.factor_name == "size")
    assert size_row.contribution_to_return == pytest.approx(1.5 * 0.04)
    assert size_row.exposure_at_period_start == pytest.approx(1.5)
    assert size_row.factor_return_over_period == pytest.approx(0.04)
    assert not size_row.is_residual


def test_attribution_folds_missing_exposure_into_residual_not_zero():
    # a real factor return exists this period, but this subject has no
    # real exposure reading for it -- must be folded into the residual,
    # never silently treated as a zero-exposure contribution
    exposures = [_exposure("A", "momentum", 0.5)]
    returns = [_factor_return("momentum", 0.02), _factor_return("value", 0.03)]
    report = compute_attribution(exposures, returns, 0.5 * 0.02, ExposureSubject.STOCK, "A", _ts("2020-05-31"), _ts("2020-06-30"))
    assert {r.factor_name for r in report.rows if not r.is_residual} == {"momentum"}
    assert any("value" in w for w in report.warnings)
    assert report.residual_return == pytest.approx(0.0)


def test_attribution_folds_nan_exposure_into_residual():
    exposures = [_exposure("A", "momentum", float("nan"))]
    returns = [_factor_return("momentum", 0.02)]
    report = compute_attribution(exposures, returns, 0.07, ExposureSubject.STOCK, "A", _ts("2020-05-31"), _ts("2020-06-30"))
    assert all(r.is_residual for r in report.rows)
    assert report.residual_return == pytest.approx(0.07)
    assert any("NaN" in w for w in report.warnings)


def test_attribution_with_no_real_factor_data_is_pure_residual():
    report = compute_attribution([], [], 0.05, ExposureSubject.STOCK, "A", _ts("2020-05-31"), _ts("2020-06-30"))
    assert len(report.rows) == 1
    assert report.rows[0].is_residual
    assert report.rows[0].contribution_to_return == pytest.approx(0.05)


def test_attribution_works_for_portfolio_subject_type_too():
    exposures = [
        FactorExposureRow(subject_type=ExposureSubject.PORTFOLIO, subject_id="P", as_of=_ts("2020-05-31"),
                           factor_name="momentum", exposure_value=0.2, source_known_as_of=_ts("2020-05-31"))
    ]
    returns = [_factor_return("momentum", 0.05)]
    report = compute_attribution(exposures, returns, 0.2 * 0.05, ExposureSubject.PORTFOLIO, "P", _ts("2020-05-31"), _ts("2020-06-30"))
    assert report.subject_type == ExposureSubject.PORTFOLIO
    assert report.residual_return == pytest.approx(0.0)
