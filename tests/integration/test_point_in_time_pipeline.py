"""End-to-end: the synthetic dataset's deliberate restatement is caught
correctly through the real pipeline -- generator -> typed observations ->
point-in-time query / universe / health report -- not just at the level
of one isolated unit test.
"""

import pandas as pd

from factor_model.config import DataConfig, DiagnosticsConfig, UniverseConfig
from factor_model.data import (
    SyntheticDataConfig,
    build_point_in_time_universe,
    fundamental_observations_from_frame,
    generate_synthetic_dataset,
    latest_known_fundamental,
    market_observations_from_frame,
)
from factor_model.diagnostics import run_data_health_report


def _cfg() -> SyntheticDataConfig:
    return SyntheticDataConfig(
        n_securities=10,
        n_sectors=2,
        n_periods_months=30,
        restatement_enabled=True,
        restatement_security_index=3,
        # a real period index landing on the fundamentals-reporting cadence
        # (t % 3 == 2), well inside this 30-period run
        restatement_period_index=23,
    )


def test_restatement_is_invisible_before_it_is_knowable_then_appears():
    dataset = generate_synthetic_dataset(_cfg())
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)

    restated_security_id = dataset.metadata["restatement_security_id"]
    period_end = pd.Timestamp(dataset.metadata["restatement_period_end"])
    assert restated_security_id == "SYN003"

    # before the restated vintage is knowable (original lag 45d, restated
    # lag 200d): a real researcher querying at period_end + 100d must see
    # only the original value
    before = latest_known_fundamental(fundamental_obs, restated_security_id, period_end, period_end + pd.Timedelta(days=100))
    after = latest_known_fundamental(fundamental_obs, restated_security_id, period_end, period_end + pd.Timedelta(days=250))

    assert before is not None and after is not None
    assert before.book_value_per_share != after.book_value_per_share
    assert after.book_value_per_share == before.book_value_per_share * 1.4
    assert before.quality_flags == ()
    assert after.quality_flags == ("restated",)

    # the real data-health report, run at each of those two as_of dates,
    # must reflect the same point-in-time truth -- not the full dataset
    report_before = run_data_health_report(
        market_obs, fundamental_obs, period_end + pd.Timedelta(days=100), DataConfig(), DiagnosticsConfig(min_cross_section_size=1)
    )
    report_after = run_data_health_report(
        market_obs, fundamental_obs, period_end + pd.Timedelta(days=250), DataConfig(), DiagnosticsConfig(min_cross_section_size=1)
    )
    assert restated_security_id not in report_before.restated_fundamental_security_ids
    assert restated_security_id in report_after.restated_fundamental_security_ids


def test_universe_construction_uses_the_same_point_in_time_market_observations():
    dataset = generate_synthetic_dataset(_cfg())
    market_obs = market_observations_from_frame(dataset.market)
    as_of = pd.Timestamp(dataset.metadata["start_date"]) + pd.DateOffset(months=12)

    snapshot = build_point_in_time_universe(
        market_obs, as_of, UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0), DataConfig(min_price_history_days=1)
    )
    # every one of the 10 synthetic securities has a real observation
    # knowable by month 12 -- all should be real universe members, none
    # excluded as not-yet-listed
    assert len(snapshot.included_ids()) == dataset.metadata["n_securities"]
    assert snapshot.excluded_ids() == []
