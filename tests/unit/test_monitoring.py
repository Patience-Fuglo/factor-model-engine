import numpy as np
import pandas as pd
import pytest

from factor_model.config import EngineConfig, MonitoringConfig, UniverseConfig
from factor_model.data import MarketObservation, Sector
from factor_model.monitoring import compute_factor_stability
from factor_model.monitoring.monitoring import _build_stability_result


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _market(security_id: str, as_of: str, price: float, market_cap: float) -> MarketObservation:
    return MarketObservation(
        security_id=security_id, ticker=security_id, as_of=_ts(as_of), known_as_of=_ts(as_of),
        sector=Sector.TECHNOLOGY, is_active=True, price=price, shares_outstanding=1.0e6,
        market_cap=market_cap, adv_20d=1.0e6,
    )


def test_build_stability_result_flags_a_real_sign_flip_in_recent_ic():
    historical = [0.2] * 10
    recent = [-0.2] * 6
    ic_series = pd.Series(historical + recent, index=pd.RangeIndex(16))
    result = _build_stability_result("size", ic_series, MonitoringConfig(drift_recent_window=6))
    assert any("flipped sign" in w for w in result.warnings)


def test_build_stability_result_flags_real_ic_decay():
    historical = [0.3] * 10
    recent = [0.05] * 6  # same sign, but decayed well below half
    ic_series = pd.Series(historical + recent, index=pd.RangeIndex(16))
    result = _build_stability_result("value", ic_series, MonitoringConfig(drift_recent_window=6, drift_decay_threshold=0.5))
    assert any("decayed" in w for w in result.warnings)


def test_build_stability_result_does_not_flag_a_stable_factor():
    ic_series = pd.Series([0.25, 0.22, 0.24, 0.26, 0.23, 0.25, 0.24, 0.26, 0.25, 0.23])
    result = _build_stability_result("momentum", ic_series, MonitoringConfig(drift_recent_window=4))
    assert result.warnings == []
    assert result.icir > 0


def test_compute_factor_stability_recovers_a_real_strong_size_signal():
    # a toy, deliberate size-driven return panel: real IC should be
    # consistently strong (|IC| large), reusing the SAME real
    # information_coefficient the toolkit implements
    market_observations = []
    market_caps_by_period = {}
    base = {"A": 1.0e8, "B": 2.0e8, "C": 5.0e7, "D": 4.0e8, "E": 1.0e7}
    dates = pd.date_range("2018-01-31", periods=14, freq="ME")
    price = {sid: 100.0 for sid in base}
    for i, d in enumerate(dates):
        for sid, mc in base.items():
            market_observations.append(_market(sid, str(d.date()), price[sid], mc))
        # next period's price moves DOWN for large-cap, UP for small-cap
        # (a real, deliberate negative size-factor-return relationship)
        log_mc = {sid: np.log(v) for sid, v in base.items()}
        vals = np.array(list(log_mc.values()))
        z = dict(zip(log_mc.keys(), (vals - vals.mean()) / vals.std(ddof=1)))
        for sid in base:
            price[sid] = price[sid] * (1 - 0.05 * z[sid])

    engine_cfg = EngineConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    period_dates = [pd.Timestamp(d) for d in dates]
    results = compute_factor_stability(market_observations, [], period_dates, engine_cfg)
    assert "size" in results
    assert results["size"].historical_ic_mean < -0.5  # strong, consistent negative IC
