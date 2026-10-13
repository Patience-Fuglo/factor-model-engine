"""End-to-end: Stage 5's factor returns + Stage 6's specific returns
feed a real portfolio factor-risk decomposition on the real synthetic
dataset.
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
from factor_model.regression import compute_factor_returns
from factor_model.risk import compute_factor_covariance, compute_portfolio_risk, compute_specific_returns, compute_specific_variance


def test_equal_weight_portfolio_risk_decomposes_correctly_end_to_end():
    cfg = SyntheticDataConfig(n_securities=30, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates[12:]]
    as_of = period_dates[-1]

    engine_cfg = EngineConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    return_rows, _ = compute_factor_returns(market_obs, fundamental_obs, period_dates, engine_cfg)
    covariance_result = compute_factor_covariance(return_rows, engine_cfg.risk)
    assert not covariance_result.covariance.empty

    specific_returns = compute_specific_returns(market_obs, fundamental_obs, period_dates, engine_cfg)
    specific_variance = compute_specific_variance(specific_returns)

    universe = build_point_in_time_universe(
        market_obs, as_of, UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0), DataConfig(min_price_history_days=1)
    )
    stock_rows, _ = compute_factor_exposure_table(market_obs, fundamental_obs, as_of, universe, engine_cfg)
    benchmark_holdings = equal_weight_benchmark_holdings(universe.included_ids(), as_of)
    portfolio_rows = aggregate_exposure(stock_rows, benchmark_holdings, ExposureSubject.PORTFOLIO, "benchmark_equal_weight")

    result = compute_portfolio_risk(
        portfolio_rows, covariance_result.covariance, benchmark_holdings, specific_variance,
        "benchmark_equal_weight", engine_cfg.risk,
    )
    assert result.total_variance >= 0
    assert result.factor_variance >= 0
    assert result.specific_variance >= 0
    assert sum(result.factor_risk_contribution.values()) == pytest.approx(result.factor_variance)
    assert result.effective_n_positions == pytest.approx(len(universe.included_ids()))  # equal-weight
