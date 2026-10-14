import pandas as pd
import pytest

from factor_model.config import RegimeConfig
from factor_model.data import FactorReturnRow
from factor_model.regimes import classify_volatility_regime


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _row(period_end: str, value: float) -> FactorReturnRow:
    start = pd.Timestamp(period_end) - pd.DateOffset(months=1)
    return FactorReturnRow(
        period_start=start, period_end=_ts(period_end), factor_name="size",
        factor_return=value, r_squared=0.1, n_obs=30, known_as_of=_ts(period_end),
    )


def test_classify_volatility_regime_identifies_a_real_high_vol_period():
    # a quiet run, then one real, sharply more volatile stretch at the end
    quiet = [0.001, -0.001, 0.0015, -0.0012, 0.0008, -0.0009, 0.0011, -0.0013, 0.0009, -0.001, 0.0012, -0.0011]
    loud = [0.05, -0.06, 0.055, -0.052]
    values = quiet + loud
    dates = pd.date_range("2016-01-31", periods=len(values), freq="ME")
    rows = [_row(str(d.date()), v) for d, v in zip(dates, values)]

    result = classify_volatility_regime(rows, "size", RegimeConfig(volatility_window=12))
    # the loud stretch's own rolling-window volatility should be classified high
    final_regime = result["regime"].iloc[-1]
    assert final_regime == "high"


def test_classify_volatility_regime_never_uses_future_volatility_to_rank_the_past():
    # a real point-in-time check: an EARLY period's regime label must be
    # identical whether or not later, more volatile periods exist in the
    # dataset -- an expanding (not full-sample) percentile rank guarantees this
    quiet = [0.001, -0.001, 0.0015, -0.0012, 0.0008, -0.0009, 0.0011, -0.0013, 0.0009, -0.001, 0.0012, -0.0011, 0.001, -0.0009]
    dates_short = pd.date_range("2016-01-31", periods=len(quiet), freq="ME")
    rows_short = [_row(str(d.date()), v) for d, v in zip(dates_short, quiet)]

    extended = quiet + [0.08, -0.09, 0.07]
    dates_long = pd.date_range("2016-01-31", periods=len(extended), freq="ME")
    rows_long = [_row(str(d.date()), v) for d, v in zip(dates_long, extended)]

    result_short = classify_volatility_regime(rows_short, "size", RegimeConfig(volatility_window=12))
    result_long = classify_volatility_regime(rows_long, "size", RegimeConfig(volatility_window=12))

    shared_dates = result_short.index
    pd.testing.assert_series_equal(result_short.loc[shared_dates, "regime"], result_long.loc[shared_dates, "regime"])


def test_classify_volatility_regime_returns_none_before_enough_real_history():
    rows = [_row("2016-01-31", 0.01), _row("2016-02-29", -0.01)]
    result = classify_volatility_regime(rows, "size", RegimeConfig(volatility_window=12))
    assert result["regime"].isna().all()


def test_classify_volatility_regime_handles_an_unknown_factor_name():
    rows = [_row("2016-01-31", 0.01)]
    result = classify_volatility_regime(rows, "nonexistent_factor", RegimeConfig())
    assert result.empty
