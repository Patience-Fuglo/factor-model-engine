import json

import pandas as pd
import pytest

from factor_model.config import EngineConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame
from factor_model.pipeline import run_full_pipeline
from factor_model.reporting import render_markdown_report, report_to_dict


@pytest.fixture(scope="module")
def pipeline_result():
    cfg = SyntheticDataConfig(n_securities=20, n_sectors=5, n_periods_months=30)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates[12:]]
    engine_cfg = EngineConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    return run_full_pipeline(market_obs, fundamental_obs, period_dates[-1], period_dates, engine_cfg)


def test_report_to_dict_is_json_serializable(pipeline_result):
    payload = report_to_dict(pipeline_result)
    json.dumps(payload)  # raises if anything (e.g. a Timestamp) leaked through un-serialized


def test_report_to_dict_matches_the_real_underlying_pipeline_result(pipeline_result):
    payload = report_to_dict(pipeline_result)
    assert payload["as_of"] == str(pipeline_result.as_of.date())
    assert payload["portfolio_id"] == pipeline_result.portfolio_id
    assert payload["universe"]["n_included"] == len(pipeline_result.universe.included_ids())
    assert payload["attribution"]["realized_return"] == pytest.approx(pipeline_result.attribution.realized_return)
    assert payload["portfolio_risk"]["total_variance"] == pytest.approx(pipeline_result.portfolio_risk.total_variance)


def test_report_to_dict_attribution_contributions_reconcile_to_realized_return(pipeline_result):
    payload = report_to_dict(pipeline_result)
    total = sum(payload["attribution"]["contributions"].values())
    assert total == pytest.approx(payload["attribution"]["realized_return"])


def test_render_markdown_report_contains_every_real_section(pipeline_result):
    markdown = render_markdown_report(pipeline_result)
    for heading in ["## Universe", "## Factor Library", "## Portfolio Factor Exposure", "## Portfolio Risk", "## Attribution", "## Regimes and Factor Stability"]:
        assert heading in markdown
    assert pipeline_result.portfolio_id in markdown
    assert str(pipeline_result.as_of.date()) in markdown
    assert "synthetic" in markdown.lower()


def test_report_rendering_is_deterministic(pipeline_result):
    first = render_markdown_report(pipeline_result)
    second = render_markdown_report(pipeline_result)
    assert first == second
    assert report_to_dict(pipeline_result) == report_to_dict(pipeline_result)
