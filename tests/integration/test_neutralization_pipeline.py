"""End-to-end: neutralize a toy composite signal against one real
Stage 2 factor on the real synthetic dataset, confirming the preserved
factor's exposure survives while the neutralized one collapses.
"""

import pandas as pd

from factor_model.config import DataConfig, EngineConfig, UniverseConfig
from factor_model.data import (
    SyntheticDataConfig,
    build_point_in_time_universe,
    fundamental_observations_from_frame,
    generate_synthetic_dataset,
    market_observations_from_frame,
)
from factor_model.factors import compute_factor_exposure_table, factor_exposure_rows_to_frame
from factor_model.neutralization import neutralize_signal


def test_neutralizing_size_preserves_the_momentum_exposure_on_real_synthetic_data():
    cfg = SyntheticDataConfig(n_securities=50, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    as_of = pd.Timestamp(dataset.metadata["end_date"])

    universe = build_point_in_time_universe(
        market_obs, as_of, UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0), DataConfig(min_price_history_days=1)
    )
    rows, _ = compute_factor_exposure_table(market_obs, fundamental_obs, as_of, universe, EngineConfig())
    frame = factor_exposure_rows_to_frame(rows)
    pivot = frame.pivot(index="security_id", columns="factor_name", values="standardized_score")

    # a toy composite signal deliberately built from BOTH momentum and
    # size, so neutralizing size alone has something real to remove
    # while momentum (preserved) should survive
    signal = 1.0 * pivot["momentum"] + 1.5 * pivot["size"]

    result = neutralize_signal(signal, pivot[["momentum", "size", "value", "quality"]], factors_to_neutralize=["size"])
    assert result.preserved_factors == ["momentum", "value", "quality"]
    assert abs(result.residual_exposure_corr["size"]) < 0.15
    assert result.residual_exposure_corr["momentum"] > 0.5
