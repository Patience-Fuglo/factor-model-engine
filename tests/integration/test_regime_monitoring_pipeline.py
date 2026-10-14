"""End-to-end: regime detection and factor-stability monitoring on the
real synthetic dataset's real factor-return history.
"""

import pandas as pd

from factor_model.config import EngineConfig, UniverseConfig
from factor_model.data import (
    SyntheticDataConfig,
    fundamental_observations_from_frame,
    generate_synthetic_dataset,
    market_observations_from_frame,
)
from factor_model.monitoring import compute_factor_stability
from factor_model.regimes import classify_volatility_regime
from factor_model.regression import compute_factor_returns


def test_regime_and_stability_pipeline_runs_end_to_end_on_real_synthetic_data():
    cfg = SyntheticDataConfig(n_securities=30, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates[12:]]

    engine_cfg = EngineConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    return_rows, _ = compute_factor_returns(market_obs, fundamental_obs, period_dates, engine_cfg)
    assert len(return_rows) > 0

    regime_frame = classify_volatility_regime(return_rows, "momentum", engine_cfg.regimes)
    assert not regime_frame.empty
    assert set(regime_frame["regime"].dropna().unique()) <= {"low", "normal", "high"}

    stability = compute_factor_stability(market_obs, fundamental_obs, period_dates, engine_cfg)
    assert "momentum" in stability
    # momentum's planted relationship is positive and recovered reliably
    # (Stage 5's own finding) -- its real IC should trend positive too
    assert stability["momentum"].historical_ic_mean > 0
