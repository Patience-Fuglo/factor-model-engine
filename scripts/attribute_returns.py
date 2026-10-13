#!/usr/bin/env python3
"""Stage 7 CLI: real per-stock return attribution over one period,
reconciling each stock's realized return into factor contributions plus
a residual. Reads ``outputs/synthetic/{market,fundamentals_raw}.csv`` --
run ``generate_synthetic_dataset.py`` first.

Usage:
    python scripts/attribute_returns.py --period-start 2020-05-31 --period-end 2020-06-30 [--config configs/default.yaml] [--synthetic-dir outputs/synthetic] [--output-dir outputs/attribution]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_model.attribution import attribution_rows_to_frame, compute_attribution
from factor_model.config import load_config
from factor_model.data import ExposureSubject, build_point_in_time_universe, fundamental_observations_from_frame, market_observations_from_frame
from factor_model.factors import compute_factor_exposure_table
from factor_model.regression import _period_realized_returns, compute_factor_returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period-start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--period-end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic-dir", default="outputs/synthetic")
    parser.add_argument("--output-dir", default="outputs/attribution")
    args = parser.parse_args()

    engine_cfg = load_config(args.config)
    synthetic_dir = Path(args.synthetic_dir)
    market_df = pd.read_csv(synthetic_dir / "market.csv", parse_dates=["as_of", "known_as_of"])
    fundamentals_df = pd.read_csv(
        synthetic_dir / "fundamentals_raw.csv", parse_dates=["fundamental_period_end", "known_as_of"]
    )
    market_df["quality_flags"] = market_df["quality_flags"].fillna("")
    fundamentals_df["quality_flags"] = fundamentals_df["quality_flags"].fillna("")
    market_obs = market_observations_from_frame(market_df)
    fundamental_obs = fundamental_observations_from_frame(fundamentals_df)

    t_prev, t_curr = pd.Timestamp(args.period_start), pd.Timestamp(args.period_end)
    universe = build_point_in_time_universe(market_obs, t_prev, engine_cfg.universe, engine_cfg.data)
    exposure_rows, _ = compute_factor_exposure_table(market_obs, fundamental_obs, t_prev, universe, engine_cfg)
    return_rows, return_warnings = compute_factor_returns(market_obs, fundamental_obs, [t_prev, t_curr], engine_cfg)
    realized_returns = _period_realized_returns(market_obs, universe.included_ids(), t_prev, t_curr)

    all_rows = []
    for security_id in universe.included_ids():
        if security_id not in realized_returns.index:
            continue
        report = compute_attribution(
            exposure_rows, return_rows, float(realized_returns[security_id]),
            ExposureSubject.STOCK, security_id, t_prev, t_curr,
        )
        all_rows.extend(report.rows)

    frame = attribution_rows_to_frame(all_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"attribution_{t_prev.date()}_{t_curr.date()}.csv"
    frame.sort_values(["subject_id", "is_residual", "factor_name"]).to_csv(out_path, index=False)

    n_securities = frame["subject_id"].nunique()
    print(f"wrote {len(frame)} attribution rows for {n_securities} securities to {out_path}")
    if return_warnings:
        print(f"{len(return_warnings)} factor-return warning(s):")
        for w in return_warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
