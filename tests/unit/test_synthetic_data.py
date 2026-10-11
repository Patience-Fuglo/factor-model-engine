import numpy as np
import pytest

from factor_model.data import SyntheticDataConfig, generate_synthetic_dataset


def _small_cfg(**overrides) -> SyntheticDataConfig:
    defaults = dict(n_securities=10, n_sectors=2, n_periods_months=30)
    defaults.update(overrides)
    return SyntheticDataConfig(**defaults)


def test_generator_is_deterministic():
    cfg = _small_cfg()
    a = generate_synthetic_dataset(cfg)
    b = generate_synthetic_dataset(cfg)
    pd_assert = __import__("pandas").testing.assert_frame_equal
    pd_assert(a.market, b.market)
    pd_assert(a.fundamentals, b.fundamentals)
    pd_assert(a.ground_truth, b.ground_truth)


def test_generator_adding_a_security_does_not_perturb_earlier_ones_raw_draws():
    # the real, achievable guarantee: every per-security RAW draw (sector
    # assignment, and the independently-seeded shares/turnover/book/eps
    # scale parameters) is invariant to n_securities, since each is
    # spawned from its own (base_seed, security_index, stream) sub-sequence
    small = generate_synthetic_dataset(_small_cfg(n_securities=10, n_sectors=2))
    larger = generate_synthetic_dataset(_small_cfg(n_securities=20, n_sectors=2))
    common_ids = small.market["security_id"].unique()

    small_sectors = small.market.drop_duplicates("security_id").set_index("security_id")["sector"]
    larger_sectors = larger.market.drop_duplicates("security_id").set_index("security_id")["sector"].loc[common_ids]
    assert small_sectors.equals(larger_sectors)

    small_shares = small.market.drop_duplicates("security_id").set_index("security_id")["shares_outstanding"]
    larger_shares = larger.market.drop_duplicates("security_id").set_index("security_id")["shares_outstanding"].loc[common_ids]
    __import__("pandas").testing.assert_series_equal(small_shares, larger_shares)


def test_generator_cross_sectional_outputs_are_legitimately_not_invariant_to_universe_size():
    # the flip side, documented rather than assumed away: price/return/
    # market_cap/fundamentals are cross-sectionally standardized, so they
    # are EXPECTED to shift when the peer universe changes -- same real
    # property any cross-sectional factor model has
    small = generate_synthetic_dataset(_small_cfg(n_securities=10, n_sectors=2))
    larger = generate_synthetic_dataset(_small_cfg(n_securities=20, n_sectors=2))
    common_ids = small.market["security_id"].unique()
    small_price = small.market.sort_values(["security_id", "as_of"])["price"].reset_index(drop=True)
    larger_price = (
        larger.market[larger.market["security_id"].isin(common_ids)]
        .sort_values(["security_id", "as_of"])["price"]
        .reset_index(drop=True)
    )
    assert not small_price.equals(larger_price)


def test_ground_truth_structure_is_recoverable_by_real_ols():
    # the whole point of baking in a known structure: a real OLS fit of
    # realized return on the ground-truth factors should recover betas
    # close to the real configured loadings (0.5 momentum, 0.3 value)
    cfg = _small_cfg(n_securities=50, n_sectors=5, n_periods_months=60)
    dataset = generate_synthetic_dataset(cfg)

    market = dataset.market.copy()
    market["return"] = market.groupby("security_id")["price"].pct_change()
    merged = market.merge(dataset.ground_truth, on=["security_id", "as_of"])
    merged = merged.dropna(subset=["return"])

    X = np.column_stack([np.ones(len(merged)), merged["momentum_score"], merged["value_score"]])
    y = merged["return"].to_numpy()
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    _, momentum_beta, value_beta = coeffs

    # real monthly return scale is 0.08 * loading -- recovered betas should
    # land in that neighborhood, not exactly on it (real noise is present)
    assert 0.0 < momentum_beta < 0.08 * cfg.momentum_loading * 2
    assert 0.0 < value_beta < 0.08 * cfg.value_loading * 2


def test_restatement_scenario_produces_two_real_vintages():
    cfg = _small_cfg(n_securities=10, n_sectors=2, restatement_enabled=True, restatement_security_index=3)
    dataset = generate_synthetic_dataset(cfg)
    restated_security = "SYN003"
    vintages = dataset.fundamentals[dataset.fundamentals["security_id"] == restated_security]
    period_end = dataset.metadata["restatement_period_end"]
    matching = vintages[vintages["fundamental_period_end"].astype(str) == period_end]
    assert len(matching) == 2
    assert set(matching["quality_flags"]) == {"", "restated"}


def test_restatement_disabled_produces_one_vintage_per_period():
    cfg = _small_cfg(n_securities=10, n_sectors=2, restatement_enabled=False)
    dataset = generate_synthetic_dataset(cfg)
    counts = dataset.fundamentals.groupby(["security_id", "fundamental_period_end"]).size()
    assert (counts == 1).all()


def test_generator_rejects_securities_not_divisible_by_sectors():
    with pytest.raises(ValueError):
        SyntheticDataConfig(n_securities=10, n_sectors=3)


def test_generator_rejects_restatement_period_off_the_fundamental_cadence():
    with pytest.raises(ValueError):
        SyntheticDataConfig(fundamentals_update_every_n_months=3, restatement_period_index=1)
