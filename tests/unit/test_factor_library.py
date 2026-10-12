import numpy as np
import pandas as pd
import pytest

from factor_model.config import EngineConfig
from factor_model.data import (
    ExclusionReason,
    FundamentalObservation,
    MarketObservation,
    Sector,
    SyntheticDataConfig,
    UniverseInclusion,
    UniversePITSnapshot,
    build_point_in_time_universe,
    fundamental_observations_from_frame,
    generate_synthetic_dataset,
    market_observations_from_frame,
)
from factor_model.config import DataConfig, UniverseConfig
from factor_model.factors import compute_factor_exposure_table, factor_exposure_rows_to_frame


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _market(security_id: str, as_of: str, price: float, **overrides) -> MarketObservation:
    defaults = dict(
        ticker=security_id, sector=Sector.TECHNOLOGY, is_active=True, shares_outstanding=1.0e6,
        market_cap=price * 1.0e6, adv_20d=1.0e6,
    )
    defaults.update(overrides)
    known_as_of = defaults.pop("known_as_of", as_of)
    return MarketObservation(security_id=security_id, as_of=_ts(as_of), known_as_of=_ts(known_as_of), price=price, **defaults)


def _fundamental(security_id: str, period_end: str, known_as_of: str, **overrides) -> FundamentalObservation:
    defaults = dict(book_value_per_share=10.0, earnings_per_share=1.0, total_assets=100.0, total_debt=30.0, quality_flags=())
    defaults.update(overrides)
    return FundamentalObservation(
        security_id=security_id, fundamental_period_end=_ts(period_end), known_as_of=_ts(known_as_of), **defaults
    )


def _snapshot(security_ids: list[str], as_of: pd.Timestamp) -> UniversePITSnapshot:
    return UniversePITSnapshot(
        as_of=as_of,
        members=tuple(UniverseInclusion(security_id=sid, included=True) for sid in security_ids),
        config_hash="test",
    )


def test_factor_table_computes_raw_and_standardized_values_for_a_simple_fixture():
    market = [
        _market("A", "2020-06-30", price=100.0, market_cap=5.0e9),
        _market("B", "2020-06-30", price=50.0, market_cap=1.0e9),
        _market("C", "2020-06-30", price=20.0, market_cap=2.0e8),
    ]
    fundamentals = [
        _fundamental("A", "2020-03-31", "2020-05-15", book_value_per_share=40.0, total_assets=1000.0, total_debt=200.0),
        _fundamental("B", "2020-03-31", "2020-05-15", book_value_per_share=60.0, total_assets=500.0, total_debt=400.0),
        _fundamental("C", "2020-03-31", "2020-05-15", book_value_per_share=10.0, total_assets=300.0, total_debt=100.0),
    ]
    universe = _snapshot(["A", "B", "C"], _ts("2020-06-30"))
    rows, report = compute_factor_exposure_table(market, fundamentals, _ts("2020-06-30"), universe, EngineConfig())

    frame = factor_exposure_rows_to_frame(rows)
    assert set(frame["factor_name"]) == {"momentum", "value", "size", "quality"}
    value_row_a = frame[(frame["security_id"] == "A") & (frame["factor_name"] == "value")].iloc[0]
    assert value_row_a["raw_value"] == pytest.approx(40.0 / 100.0)
    size_row_a = frame[(frame["security_id"] == "A") & (frame["factor_name"] == "size")].iloc[0]
    assert size_row_a["raw_value"] == pytest.approx(np.log(5.0e9))
    assert report.n_eligible == 3


def test_factor_table_flags_missing_raw_value_without_dropping_the_row():
    market = [_market("A", "2020-06-30", price=100.0)]
    universe = _snapshot(["A"], _ts("2020-06-30"))
    rows, report = compute_factor_exposure_table(market, [], _ts("2020-06-30"), universe, EngineConfig())
    value_row = next(r for r in rows if r.factor_name == "value")
    assert value_row.raw_value is None
    assert "missing_raw_value" in value_row.quality_flags
    assert report.missing_security_ids["value"] == ["A"]
    assert report.n_eligible == 1


def test_momentum_uses_only_point_in_time_knowable_price_history():
    # a price dated AFTER as_of must never leak into the momentum reading
    market = [
        _market("A", "2019-01-31", price=100.0),
        _market("A", "2019-02-28", price=110.0),
        _market("A", "2025-01-31", price=99999.0),  # future -- must be excluded
    ]
    universe = _snapshot(["A"], _ts("2019-02-28"))
    rows, _ = compute_factor_exposure_table(market, [], _ts("2019-02-28"), universe, EngineConfig())
    momentum_row = next(r for r in rows if r.factor_name == "momentum")
    # insufficient real history for a 12-1 momentum reading yet (NaN), but
    # critically: no error and no use of the future 2025 observation
    assert momentum_row.raw_value is None


def test_factor_table_is_reproducible():
    market = [_market("A", "2020-06-30", price=100.0), _market("B", "2020-06-30", price=50.0)]
    fundamentals = [_fundamental("A", "2020-03-31", "2020-05-15"), _fundamental("B", "2020-03-31", "2020-05-15")]
    universe = _snapshot(["A", "B"], _ts("2020-06-30"))
    rows_a, _ = compute_factor_exposure_table(market, fundamentals, _ts("2020-06-30"), universe, EngineConfig())
    rows_b, _ = compute_factor_exposure_table(market, fundamentals, _ts("2020-06-30"), universe, EngineConfig())
    assert factor_exposure_rows_to_frame(rows_a).equals(factor_exposure_rows_to_frame(rows_b))


def test_quality_factor_recovers_the_planted_ground_truth_relationship():
    # higher planted quality_score -> lower leverage -> higher Quality
    # composite: a real, testable recoverability check, same spirit as
    # Stage 1's momentum/value OLS recoverability test.
    #
    # Must compare the computed factor against ground truth AT THE SAME
    # FISCAL PERIOD the fundamentals vintage it was built from covers --
    # not at the query as_of, which (correctly, realistically) uses an
    # OLDER, already-knowable vintage once publication lag is accounted
    # for. Comparing against quality_score at as_of itself would compare
    # a stale vintage's quality to a LATER date's ground truth and could
    # legitimately decorrelate, since quality_score is a real AR(1)
    # process that keeps moving between vintages.
    # n_securities=50 (not a smaller universe): the composite mixes a real
    # signal pillar (leverage, |corr| ~0.5-0.7 with truth) with an
    # UNRELATED pillar (profitability, by construction -- see
    # quality_based.py). Profitability's correlation with quality_score
    # at small n is spurious finite-sample noise that shrinks toward 0 as
    # n grows (verified: -0.31 at n=20, -0.13 at n=100); a small universe
    # lets that noise meaningfully dilute or even partially cancel the
    # real safety signal in the equal-weighted composite, which is
    # expected statistical behavior, not a defect in the composite itself.
    cfg = SyntheticDataConfig(n_securities=50, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)

    # the second-to-last real reporting period, with a buffer before
    # end_date so there's real market data knowable comfortably after it
    period_end = pd.Timestamp(sorted(dataset.fundamentals["fundamental_period_end"].unique())[-2])
    as_of = period_end + pd.Timedelta(days=60)

    universe = build_point_in_time_universe(
        market_obs, as_of, UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0), DataConfig(min_price_history_days=1)
    )
    rows, _ = compute_factor_exposure_table(market_obs, fundamental_obs, as_of, universe, EngineConfig())
    frame = factor_exposure_rows_to_frame(rows)
    quality_scores = frame[frame["factor_name"] == "quality"].set_index("security_id")["standardized_score"]

    truth = dataset.ground_truth[dataset.ground_truth["as_of"] == period_end].set_index("security_id")["quality_score"]
    aligned = pd.concat([quality_scores, truth], axis=1, keys=["computed", "truth"]).dropna()
    assert len(aligned) >= cfg.n_securities // 2
    correlation = aligned["computed"].corr(aligned["truth"])
    assert correlation > 0.3


def test_leverage_pillar_alone_strongly_recovers_the_planted_relationship():
    # the clean, by-design half of the story: leverage was built directly
    # from quality_z, so its correlation with ground truth is strong and
    # stable even at a small n (unlike the full composite above, which is
    # diluted by the unrelated profitability pillar)
    from factor_model.factors import compute_leverage

    cfg = SyntheticDataConfig(n_securities=20, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    period_end = pd.Timestamp(sorted(dataset.fundamentals["fundamental_period_end"].unique())[-2])
    truth = dataset.ground_truth[dataset.ground_truth["as_of"] == period_end].set_index("security_id")["quality_score"]
    fund = dataset.fundamentals[dataset.fundamentals["fundamental_period_end"] == period_end].set_index("security_id")
    leverage = compute_leverage(fund["total_debt"], fund["total_assets"])
    correlation = leverage.corr(truth)
    assert correlation < -0.4  # higher quality -> lower leverage, by construction
