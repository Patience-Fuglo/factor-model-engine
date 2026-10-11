import pandas as pd

from factor_model.config import DataConfig, DiagnosticsConfig
from factor_model.data import FundamentalObservation, MarketObservation, Sector
from factor_model.diagnostics import run_data_health_report


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _market(security_id: str, as_of: str, **overrides) -> MarketObservation:
    defaults = dict(
        ticker=security_id, sector=Sector.TECHNOLOGY, is_active=True, price=100.0,
        shares_outstanding=1e6, market_cap=2.0e8, adv_20d=2.0e6,
    )
    defaults.update(overrides)
    known_as_of = defaults.pop("known_as_of", as_of)
    return MarketObservation(security_id=security_id, as_of=_ts(as_of), known_as_of=_ts(known_as_of), **defaults)


def _fundamental(security_id: str, period_end: str, known_as_of: str, **overrides) -> FundamentalObservation:
    defaults = dict(book_value_per_share=10.0, earnings_per_share=1.0, quality_flags=())
    defaults.update(overrides)
    return FundamentalObservation(
        security_id=security_id, fundamental_period_end=_ts(period_end), known_as_of=_ts(known_as_of), **defaults
    )


def _diag_cfg(**overrides) -> DiagnosticsConfig:
    defaults = dict(max_allowed_missingness_pct=0.10, min_cross_section_size=1)
    defaults.update(overrides)
    return DiagnosticsConfig(**defaults)


def test_report_warns_but_never_fixes_excess_missingness():
    market = [
        _market("A", "2020-06-30", adv_20d=1.0e6),
        _market("B", "2020-06-30", adv_20d=None),
        _market("C", "2020-06-30", adv_20d=None),
    ]
    report = run_data_health_report(market, [], _ts("2020-06-30"), DataConfig(), _diag_cfg(max_allowed_missingness_pct=0.10))
    assert report.missingness_pct["adv_20d"] == 2 / 3
    assert any("adv_20d missingness" in w for w in report.warnings)
    # the report only ever warns -- it never mutates or drops the real observations
    assert report.n_securities_observed == 3


def test_report_flags_real_duplicate_market_rows():
    market = [_market("A", "2020-06-30"), _market("A", "2020-06-30")]
    report = run_data_health_report(market, [], _ts("2020-06-30"), DataConfig(), _diag_cfg())
    assert report.n_duplicate_market_rows == 1
    assert any("duplicate market rows" in w for w in report.warnings)


def test_report_flags_true_fundamental_duplicate_not_a_restatement():
    # identical security/period_end/known_as_of twice -- a real duplicate,
    # not a legitimate restatement (which requires a DIFFERENT known_as_of)
    fundamentals = [
        _fundamental("A", "2020-03-31", "2020-05-15"),
        _fundamental("A", "2020-03-31", "2020-05-15"),
    ]
    report = run_data_health_report([], fundamentals, _ts("2020-06-01"), DataConfig(), _diag_cfg())
    assert report.n_duplicate_fundamental_vintages == 1
    assert report.restated_fundamental_security_ids == []


def test_report_flags_restatement_as_a_warning_not_an_error():
    fundamentals = [
        _fundamental("A", "2020-03-31", "2020-05-15", book_value_per_share=10.0),
        _fundamental("A", "2020-03-31", "2020-10-17", book_value_per_share=14.0, quality_flags=("restated",)),
    ]
    report = run_data_health_report([], fundamentals, _ts("2020-11-01"), DataConfig(), _diag_cfg())
    assert report.restated_fundamental_security_ids == ["A"]
    assert report.n_duplicate_fundamental_vintages == 0  # a real restatement is not a duplicate
    assert any("restatements" in w for w in report.warnings)


def test_report_flags_stale_fundamentals():
    fundamentals = [_fundamental("A", "2019-03-31", "2019-05-15")]
    data_cfg = DataConfig(stale_fundamentals_after_days=190)
    report = run_data_health_report([], fundamentals, _ts("2020-06-01"), data_cfg, _diag_cfg())
    assert report.stale_fundamental_security_ids == ["A"]
    assert any("stale fundamentals" in w for w in report.warnings)


def test_report_flags_market_cap_outliers_without_removing_them():
    # z-score outlier detection has a real breakdown point: a single
    # extreme value inflates its own mean/std, so a tiny cross-section
    # can never produce |z| > 4 for one outlier (max possible |z| is
    # sqrt(n-1), ~2.0 at n=5) -- this needs a realistically sized
    # cross-section to actually demonstrate real detection power
    market = [_market(f"S{i:02d}", "2020-06-30", market_cap=2.0e8 + (i % 3) * 1.0e6) for i in range(24)]
    market.append(_market("OUT", "2020-06-30", market_cap=5.0e12))  # a real, extreme outlier
    report = run_data_health_report(market, [], _ts("2020-06-30"), DataConfig(), _diag_cfg())
    assert "OUT" in report.outlier_security_ids["market_cap"]
    # still counted as an observed security -- diagnostics warn, they don't drop
    assert report.n_securities_observed == 25


def test_zscore_outlier_detection_has_a_real_breakdown_point_at_small_n():
    # documents, rather than hides, a genuine statistical limitation: at
    # n=5 a single outlier can never clear |z|=4 no matter how extreme,
    # because it drags its own mean and std along with it
    market = [
        _market("A", "2020-06-30", market_cap=2.0e8),
        _market("B", "2020-06-30", market_cap=2.1e8),
        _market("C", "2020-06-30", market_cap=1.9e8),
        _market("D", "2020-06-30", market_cap=2.0e8),
        _market("E", "2020-06-30", market_cap=5.0e12),
    ]
    report = run_data_health_report(market, [], _ts("2020-06-30"), DataConfig(), _diag_cfg())
    assert report.outlier_security_ids["market_cap"] == []


def test_report_warns_when_cross_section_too_small_to_trust():
    market = [_market("A", "2020-06-30")]
    report = run_data_health_report(market, [], _ts("2020-06-30"), DataConfig(), _diag_cfg(min_cross_section_size=20))
    assert any("too small to trust" in w for w in report.warnings)


def test_report_only_uses_knowable_fundamentals_not_the_full_dataset():
    # a vintage known only in the future must not count toward this
    # as_of's stale/restated bookkeeping at all
    fundamentals = [_fundamental("A", "2020-03-31", "2025-01-01")]
    report = run_data_health_report([], fundamentals, _ts("2020-06-01"), DataConfig(), _diag_cfg())
    assert report.stale_fundamental_security_ids == []
    assert report.restated_fundamental_security_ids == []


def test_report_does_not_leak_a_not_yet_knowable_restatement():
    # the original vintage is knowable at as_of; the restated vintage is
    # NOT knowable yet -- the report must reflect only what's real and
    # knowable at that as_of, never flag a restatement that hasn't
    # happened yet from this researcher's point of view
    fundamentals = [
        _fundamental("A", "2020-03-31", "2020-05-15", book_value_per_share=10.0),
        _fundamental("A", "2020-03-31", "2020-10-17", book_value_per_share=14.0, quality_flags=("restated",)),
    ]
    report = run_data_health_report([], fundamentals, _ts("2020-08-08"), DataConfig(), _diag_cfg())
    assert report.restated_fundamental_security_ids == []
    assert report.n_duplicate_fundamental_vintages == 0
