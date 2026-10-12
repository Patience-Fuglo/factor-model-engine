"""Golden-value regression test: pins the real synthetic generator's
summary statistics at a fixed config, so a future refactor that silently
changes its numerical behavior (not just its API) gets caught here,
rather than drifting unnoticed. These exact numbers were computed once
by running ``generate_synthetic_dataset`` at this exact config and are
re-asserted, not re-derived -- a deliberate "if this changes, stop and
explain why" tripwire.
"""

import pytest

from factor_model.data import SyntheticDataConfig, generate_synthetic_dataset

_GOLDEN_CFG = SyntheticDataConfig(n_securities=10, n_sectors=2, n_periods_months=24)


def _dataset():
    return generate_synthetic_dataset(_GOLDEN_CFG)


def test_golden_shapes_are_pinned():
    ds = _dataset()
    assert ds.market.shape == (240, 11)
    # Stage 2 added total_assets/total_debt (fundamentals) and
    # quality_score (ground_truth) -- all other Stage 1 shapes/values
    # (asserted below and in the other golden tests) stayed byte-identical
    assert ds.fundamentals.shape == (81, 8)
    assert ds.ground_truth.shape == (240, 5)


def test_golden_market_summary_statistics_are_pinned():
    ds = _dataset()
    assert ds.market["price"].mean() == pytest.approx(102.5677961, abs=1e-6)
    assert ds.market["price"].std() == pytest.approx(58.03019993, abs=1e-6)
    assert ds.market["market_cap"].sum() == pytest.approx(2439987202567.06, rel=1e-9)
    assert ds.market["adv_20d"].mean() == pytest.approx(62067090.257858, abs=1e-3)


def test_golden_fundamentals_summary_statistics_are_pinned():
    ds = _dataset()
    assert ds.fundamentals["book_value_per_share"].mean() == pytest.approx(27.5630918, abs=1e-6)
    assert ds.fundamentals["earnings_per_share"].mean() == pytest.approx(4.83021619, abs=1e-6)
    # Stage 2 additions
    assert ds.fundamentals["total_assets"].mean() == pytest.approx(1474869643.939799, rel=1e-9)
    assert ds.fundamentals["total_debt"].mean() == pytest.approx(452711997.243794, rel=1e-9)


def test_golden_ground_truth_cross_sectional_scores_are_pinned_zero_mean():
    # the cross-sectional z-score construction itself guarantees this --
    # pinning it catches a change to that construction, not just noise
    ds = _dataset()
    assert ds.ground_truth["momentum_score"].mean() == pytest.approx(0.0, abs=1e-9)
    assert ds.ground_truth["value_score"].mean() == pytest.approx(0.0, abs=1e-9)
    assert ds.ground_truth["quality_score"].mean() == pytest.approx(0.0, abs=1e-9)


def test_golden_first_and_last_market_row_are_pinned():
    ds = _dataset()
    first = ds.market.iloc[0]
    last = ds.market.iloc[-1]
    assert first["security_id"] == "SYN000"
    assert first["price"] == pytest.approx(94.60999931829896, abs=1e-9)
    assert first["market_cap"] == pytest.approx(7095451114.17769, rel=1e-9)
    assert last["security_id"] == "SYN009"
    assert last["price"] == pytest.approx(22.769919512123852, abs=1e-9)
    assert last["market_cap"] == pytest.approx(2971606073.9046974, rel=1e-9)


def test_golden_metadata_is_pinned():
    ds = _dataset()
    assert ds.metadata["restatement_security_id"] == "SYN007"
    assert ds.metadata["restatement_period_end"] == "2017-12-31"
    assert ds.metadata["end_date"] == "2017-12-31"
