import pandas as pd
import pytest

from factor_model.data import ExposureSubject, FactorExposureRow, PortfolioHolding
from factor_model.exposures import aggregate_exposure, equal_weight_benchmark_holdings


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _exposure(security_id: str, factor_name: str, value: float, known_as_of: str = "2020-06-30") -> FactorExposureRow:
    return FactorExposureRow(
        subject_type=ExposureSubject.STOCK,
        subject_id=security_id,
        as_of=_ts("2020-06-30"),
        factor_name=factor_name,
        exposure_value=value,
        source_known_as_of=_ts(known_as_of),
    )


def _holding(portfolio_id: str, security_id: str, weight: float, known_as_of: str = "2020-06-30") -> PortfolioHolding:
    return PortfolioHolding(
        portfolio_id=portfolio_id, security_id=security_id, as_of=_ts("2020-06-30"), weight=weight,
        known_as_of=_ts(known_as_of),
    )


def test_aggregate_exposure_is_a_real_weighted_average():
    rows = [_exposure("A", "momentum", 1.0), _exposure("B", "momentum", -1.0)]
    holdings = [_holding("P", "A", 0.75), _holding("P", "B", 0.25)]
    result = aggregate_exposure(rows, holdings, ExposureSubject.PORTFOLIO, "P")
    momentum = next(r for r in result if r.factor_name == "momentum")
    assert momentum.exposure_value == pytest.approx(0.75 * 1.0 + 0.25 * -1.0)
    assert momentum.subject_type == ExposureSubject.PORTFOLIO


def test_aggregate_exposure_renormalizes_over_real_available_constituents_only():
    # B's exposure is missing (NaN) -- must NOT be treated as zero;
    # the aggregate should renormalize over A's weight alone
    rows = [_exposure("A", "value", 2.0), _exposure("B", "value", float("nan"))]
    holdings = [_holding("P", "A", 0.5), _holding("P", "B", 0.5)]
    result = aggregate_exposure(rows, holdings, ExposureSubject.PORTFOLIO, "P")
    value = next(r for r in result if r.factor_name == "value")
    assert value.exposure_value == pytest.approx(2.0)  # NOT 1.0, which a zero-fill would give


def test_aggregate_exposure_flags_insufficient_coverage_when_nothing_is_real():
    rows = [_exposure("A", "quality", float("nan"))]
    holdings = [_holding("P", "A", 1.0)]
    result = aggregate_exposure(rows, holdings, ExposureSubject.PORTFOLIO, "P")
    quality = next(r for r in result if r.factor_name == "quality")
    assert pd.isna(quality.exposure_value)
    assert "insufficient_coverage" in quality.quality_flags


def test_aggregate_exposure_ignores_stocks_with_no_real_holding_weight():
    # C has an exposure row but is NOT in the portfolio -- must not
    # silently contribute to the aggregate
    rows = [_exposure("A", "size", 1.0), _exposure("C", "size", 100.0)]
    holdings = [_holding("P", "A", 1.0)]
    result = aggregate_exposure(rows, holdings, ExposureSubject.PORTFOLIO, "P")
    size = next(r for r in result if r.factor_name == "size")
    assert size.exposure_value == pytest.approx(1.0)


def test_aggregate_exposure_source_known_as_of_is_the_latest_real_contributing_input():
    rows = [_exposure("A", "momentum", 1.0, known_as_of="2020-07-01")]
    holdings = [_holding("P", "A", 1.0, known_as_of="2020-07-15")]
    result = aggregate_exposure(rows, holdings, ExposureSubject.PORTFOLIO, "P")
    momentum = next(r for r in result if r.factor_name == "momentum")
    assert momentum.source_known_as_of == _ts("2020-07-15")


def test_aggregate_exposure_rejects_mixed_as_of_dates():
    row_a = _exposure("A", "momentum", 1.0)
    row_b = FactorExposureRow(
        subject_type=ExposureSubject.STOCK, subject_id="B", as_of=_ts("2020-07-31"),
        factor_name="momentum", exposure_value=1.0, source_known_as_of=_ts("2020-07-31"),
    )
    with pytest.raises(ValueError):
        aggregate_exposure([row_a, row_b], [_holding("P", "A", 1.0)], ExposureSubject.PORTFOLIO, "P")


def test_benchmark_is_aggregated_through_the_same_function_as_a_portfolio():
    rows = [_exposure("A", "value", 1.0), _exposure("B", "value", 3.0)]
    benchmark_holdings = equal_weight_benchmark_holdings(["A", "B"], _ts("2020-06-30"))
    result = aggregate_exposure(rows, benchmark_holdings, ExposureSubject.PORTFOLIO, "benchmark_equal_weight")
    value = next(r for r in result if r.factor_name == "value")
    assert value.exposure_value == pytest.approx(2.0)  # equal-weight average


def test_equal_weight_benchmark_holdings_sums_to_one():
    holdings = equal_weight_benchmark_holdings(["A", "B", "C", "D"], _ts("2020-06-30"))
    assert sum(h.weight for h in holdings) == pytest.approx(1.0)


def test_aggregate_exposure_works_for_a_signal_weight_scheme_too():
    # a SIGNAL is just another real weight scheme (e.g. a scaled alpha
    # score per stock) -- same function, same mechanics, different
    # subject_type/weights
    rows = [_exposure("A", "momentum", 1.0), _exposure("B", "momentum", -1.0)]
    signal_weights = [_holding("alpha_signal_1", "A", 0.6), _holding("alpha_signal_1", "B", -0.4)]
    result = aggregate_exposure(rows, signal_weights, ExposureSubject.SIGNAL, "alpha_signal_1")
    momentum = next(r for r in result if r.factor_name == "momentum")
    assert momentum.subject_type == ExposureSubject.SIGNAL
    assert momentum.exposure_value == pytest.approx((0.6 * 1.0 + (-0.4) * -1.0) / (0.6 + -0.4))


def test_portfolio_holding_rejects_known_as_of_before_as_of():
    with pytest.raises(ValueError):
        PortfolioHolding(
            portfolio_id="P", security_id="A", as_of=_ts("2020-06-30"), weight=1.0, known_as_of=_ts("2020-06-29")
        )
