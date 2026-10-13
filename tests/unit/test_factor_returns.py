import numpy as np
import pandas as pd
import pytest

from factor_model.config import EngineConfig, UniverseConfig
from factor_model.data import FactorReturnRow, MarketObservation, Sector
from factor_model.regression import compute_factor_returns


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _market(security_id: str, as_of: str, price: float, market_cap: float) -> MarketObservation:
    return MarketObservation(
        security_id=security_id, ticker=security_id, as_of=_ts(as_of), known_as_of=_ts(as_of),
        sector=Sector.TECHNOLOGY, is_active=True, price=price, shares_outstanding=1.0e6,
        market_cap=market_cap, adv_20d=1.0e6,
    )


def test_factor_return_row_rejects_non_positive_period():
    with pytest.raises(ValueError):
        FactorReturnRow(
            period_start=_ts("2020-06-30"), period_end=_ts("2020-06-30"), factor_name="size",
            factor_return=0.01, r_squared=0.1, n_obs=20, known_as_of=_ts("2020-06-30"),
        )


def test_factor_return_row_rejects_known_as_of_before_period_end():
    with pytest.raises(ValueError):
        FactorReturnRow(
            period_start=_ts("2020-05-31"), period_end=_ts("2020-06-30"), factor_name="size",
            factor_return=0.01, r_squared=0.1, n_obs=20, known_as_of=_ts("2020-05-31"),
        )


def test_factor_returns_use_lagged_exposure_not_the_period_end_exposure():
    # a real, surgical look-ahead check: realized return is constructed
    # to depend EXACTLY on each security's size exposure AT t_prev, via
    # a known coefficient k. Size is DELIBERATELY reversed at t_curr --
    # if the regression used t_curr's exposure instead (a real
    # look-ahead bug), it would recover a coefficient with the WRONG
    # sign, not the planted +k.
    market_caps_prev = {"A": 100.0, "B": 200.0, "C": 50.0, "D": 400.0, "E": 10.0}
    log_mc = {sid: np.log(v) for sid, v in market_caps_prev.items()}
    vals = np.array(list(log_mc.values()))
    z = (vals - vals.mean()) / vals.std(ddof=1)
    size_std_at_t_prev = dict(zip(log_mc.keys(), z))

    k = 0.05
    price_prev = {sid: 100.0 for sid in market_caps_prev}
    realized_return = {sid: k * size_std_at_t_prev[sid] for sid in market_caps_prev}
    price_curr = {sid: price_prev[sid] * (1 + realized_return[sid]) for sid in market_caps_prev}

    # size reversed/different at t_curr -- must NOT be what drives the result
    market_caps_curr = {"A": 10.0, "B": 50.0, "C": 400.0, "D": 200.0, "E": 100.0}

    market_observations = [
        _market(sid, "2020-05-31", price_prev[sid], market_caps_prev[sid]) for sid in market_caps_prev
    ] + [
        _market(sid, "2020-06-30", price_curr[sid], market_caps_curr[sid]) for sid in market_caps_prev
    ]

    engine_cfg = EngineConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    rows, warnings = compute_factor_returns(
        market_observations, [], [_ts("2020-05-31"), _ts("2020-06-30")], engine_cfg
    )
    # momentum (insufficient price history) and value/quality (no real
    # fundamentals supplied at all) have no usable reading this period
    # and are correctly excluded, with a warning, not silently guessed at
    assert any("momentum" in w for w in warnings)
    assert {r.factor_name for r in rows} == {"size"}
    size_row = next(r for r in rows if r.factor_name == "size")
    assert size_row.factor_return == pytest.approx(k, abs=1e-4)
    assert size_row.period_start == _ts("2020-05-31")
    assert size_row.period_end == _ts("2020-06-30")
    assert size_row.known_as_of == _ts("2020-06-30")


def test_compute_factor_returns_warns_rather_than_crashes_on_a_too_small_period():
    market_observations = [
        _market("A", "2020-05-31", 100.0, 1.0e8),
        _market("A", "2020-06-30", 110.0, 1.1e8),
    ]
    engine_cfg = EngineConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    rows, warnings = compute_factor_returns(
        market_observations, [], [_ts("2020-05-31"), _ts("2020-06-30")], engine_cfg
    )
    assert rows == []
    # one real security alone can't support standardization (needs >= 2)
    # for ANY factor, so every factor is dropped, then the period itself
    # is skipped -- both warned about, never silently swallowed
    assert len(warnings) == 2
