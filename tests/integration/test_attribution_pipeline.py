"""End-to-end: Stage 2's exposures + Stage 5's factor returns feed real
per-stock attribution on the synthetic dataset, verifying the
reconciliation identity holds exactly for every real security tested.
"""

import pandas as pd
import pytest

from factor_model.attribution import compute_attribution
from factor_model.config import DataConfig, EngineConfig, UniverseConfig
from factor_model.data import (
    ExposureSubject,
    SyntheticDataConfig,
    build_point_in_time_universe,
    fundamental_observations_from_frame,
    generate_synthetic_dataset,
    market_observations_from_frame,
)
from factor_model.factors import compute_factor_exposure_table
from factor_model.regression import _period_realized_returns, compute_factor_returns


def test_per_stock_attribution_reconciles_exactly_on_real_synthetic_data():
    cfg = SyntheticDataConfig(n_securities=30, n_sectors=5, n_periods_months=24)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)

    dates = sorted(dataset.market["as_of"].unique())
    t_prev, t_curr = pd.Timestamp(dates[12]), pd.Timestamp(dates[13])

    engine_cfg = EngineConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    universe = build_point_in_time_universe(
        market_obs, t_prev, UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0), DataConfig(min_price_history_days=1)
    )
    exposure_rows, _ = compute_factor_exposure_table(market_obs, fundamental_obs, t_prev, universe, engine_cfg)
    return_rows, _ = compute_factor_returns(market_obs, fundamental_obs, [t_prev, t_curr], engine_cfg)
    realized_returns = _period_realized_returns(market_obs, universe.included_ids(), t_prev, t_curr)

    assert len(return_rows) > 0
    tested = 0
    for security_id in universe.included_ids()[:10]:
        if security_id not in realized_returns.index:
            continue
        report = compute_attribution(
            exposure_rows, return_rows, float(realized_returns[security_id]),
            ExposureSubject.STOCK, security_id, t_prev, t_curr,
        )
        total = sum(row.contribution_to_return for row in report.rows)
        assert total == pytest.approx(realized_returns[security_id])
        tested += 1
    assert tested > 0
