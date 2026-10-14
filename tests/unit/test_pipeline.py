import pandas as pd
import pytest

from factor_model.config import EngineConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame
from factor_model.pipeline import run_full_pipeline


def _real_inputs():
    cfg = SyntheticDataConfig(n_securities=20, n_sectors=5, n_periods_months=30)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates[12:]]
    return market_obs, fundamental_obs, period_dates


def test_run_full_pipeline_rejects_as_of_not_matching_the_last_period_date():
    market_obs, fundamental_obs, period_dates = _real_inputs()
    engine_cfg = EngineConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    with pytest.raises(ValueError):
        run_full_pipeline(market_obs, fundamental_obs, period_dates[-2], period_dates, engine_cfg)


def test_run_full_pipeline_produces_a_fully_wired_real_result():
    market_obs, fundamental_obs, period_dates = _real_inputs()
    engine_cfg = EngineConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    result = run_full_pipeline(market_obs, fundamental_obs, period_dates[-1], period_dates, engine_cfg)

    assert result.as_of == period_dates[-1]
    assert len(result.universe.included_ids()) > 0
    assert len(result.stock_exposure_rows) > 0
    assert len(result.portfolio_exposure_rows) == 4  # one per factor
    assert not result.factor_covariance.covariance.empty
    assert result.portfolio_risk.total_variance >= 0
    total_attribution = sum(row.contribution_to_return for row in result.attribution.rows)
    assert total_attribution == pytest.approx(result.attribution.realized_return)
    assert set(result.regimes_by_factor.keys()) == {"momentum", "value", "size", "quality"}
    assert set(result.stability_by_factor.keys()) == {"momentum", "value", "size", "quality"}


def test_run_full_pipeline_is_reproducible():
    market_obs, fundamental_obs, period_dates = _real_inputs()
    engine_cfg = EngineConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    result_a = run_full_pipeline(market_obs, fundamental_obs, period_dates[-1], period_dates, engine_cfg)
    result_b = run_full_pipeline(market_obs, fundamental_obs, period_dates[-1], period_dates, engine_cfg)
    assert result_a.attribution.realized_return == result_b.attribution.realized_return
    assert result_a.portfolio_risk.total_variance == result_b.portfolio_risk.total_variance
    pd.testing.assert_frame_equal(result_a.factor_covariance.covariance, result_b.factor_covariance.covariance)
