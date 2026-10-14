"""Stage 10 -- Hardening: the single most comprehensive point-in-time
check in this repo. Every earlier stage's own tests verify PIT
discipline locally (one function, one schema, one query); this test
verifies it HOLISTICALLY, across the entire wired-together system at
once: does adding real market/fundamental data dated AFTER a query's
``as_of`` change ANYTHING about that query's full pipeline output?

It must not. Every stage, from universe construction through the
final report, filters on ``known_as_of <= as_of`` -- if even one stage
quietly failed to, this is the test that would catch it, because it
runs everything together rather than one function at a time.
"""

import pandas as pd

from factor_model.config import EngineConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame
from factor_model.pipeline import run_full_pipeline
from factor_model.reporting import report_to_dict


def test_extending_the_dataset_into_the_future_never_changes_an_earlier_full_pipeline_run():
    short_cfg = SyntheticDataConfig(n_securities=20, n_sectors=5, n_periods_months=36)
    long_cfg = SyntheticDataConfig(n_securities=20, n_sectors=5, n_periods_months=48)  # same seed, 12 extra REAL future months
    short_dataset = generate_synthetic_dataset(short_cfg)
    long_dataset = generate_synthetic_dataset(long_cfg)

    short_dates = sorted(short_dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in short_dates[12:]]
    as_of = period_dates[-1]
    engine_cfg = EngineConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))

    short_market = market_observations_from_frame(short_dataset.market)
    short_fundamentals = fundamental_observations_from_frame(short_dataset.fundamentals)
    long_market = market_observations_from_frame(long_dataset.market)  # includes 12 months AFTER as_of
    long_fundamentals = fundamental_observations_from_frame(long_dataset.fundamentals)

    result_short = run_full_pipeline(short_market, short_fundamentals, as_of, period_dates, engine_cfg)
    result_long = run_full_pipeline(long_market, long_fundamentals, as_of, period_dates, engine_cfg)

    # the real, decisive check: identical reports despite result_long
    # having seen a full extra real year of future data
    assert report_to_dict(result_short) == report_to_dict(result_long)


def test_full_pipeline_reconciles_and_reproduces_on_the_default_scale_dataset():
    # the default-scale (50 security, 60 month) dataset, as a real
    # security/liquidity-filtered universe would actually see it --
    # not the min_market_cap=0 relaxation used by most other tests
    cfg = SyntheticDataConfig()
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates[12:]]
    engine_cfg = EngineConfig()  # real default universe/data config, liquidity filters included

    result_a = run_full_pipeline(market_obs, fundamental_obs, period_dates[-1], period_dates, engine_cfg)
    result_b = run_full_pipeline(market_obs, fundamental_obs, period_dates[-1], period_dates, engine_cfg)
    assert report_to_dict(result_a) == report_to_dict(result_b)

    total_attribution = sum(row.contribution_to_return for row in result_a.attribution.rows)
    assert total_attribution == result_a.attribution.realized_return
    assert len(result_a.universe.included_ids()) > 0
