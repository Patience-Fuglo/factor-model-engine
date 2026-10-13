"""End-to-end: Stage 2's stock-level factor table feeds Stage 3's
portfolio/benchmark aggregation, on the real synthetic dataset -- not
just hand-built fixtures.
"""

import pandas as pd
import pytest

from factor_model.config import DataConfig, EngineConfig, UniverseConfig
from factor_model.data import (
    ExposureSubject,
    SyntheticDataConfig,
    build_point_in_time_universe,
    fundamental_observations_from_frame,
    generate_synthetic_dataset,
    market_observations_from_frame,
)
from factor_model.exposures import aggregate_exposure, equal_weight_benchmark_holdings
from factor_model.factors import compute_factor_exposure_table


def test_equal_weight_portfolio_momentum_exposure_is_the_real_cross_sectional_average():
    cfg = SyntheticDataConfig(n_securities=20, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    as_of = pd.Timestamp(dataset.metadata["end_date"])

    universe = build_point_in_time_universe(
        market_obs, as_of, UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0), DataConfig(min_price_history_days=1)
    )
    stock_rows, _ = compute_factor_exposure_table(market_obs, fundamental_obs, as_of, universe, EngineConfig())

    benchmark_holdings = equal_weight_benchmark_holdings(universe.included_ids(), as_of)
    portfolio_rows = aggregate_exposure(stock_rows, benchmark_holdings, ExposureSubject.PORTFOLIO, "benchmark_equal_weight")

    momentum_row = next(r for r in portfolio_rows if r.factor_name == "momentum")
    stock_momentum = [r.exposure_value for r in stock_rows if r.factor_name == "momentum" and pd.notna(r.exposure_value)]
    assert momentum_row.exposure_value == pytest.approx(sum(stock_momentum) / len(stock_momentum))
    assert momentum_row.subject_type == ExposureSubject.PORTFOLIO
