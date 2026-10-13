#!/usr/bin/env python3
"""Stage 5 CLI: compute a real, lagged factor-return time series from
the synthetic dataset over a given monthly date range. Reads
``outputs/synthetic/{market,fundamentals_raw}.csv`` -- run
``generate_synthetic_dataset.py`` first.

Usage:
    python scripts/compute_factor_returns.py --start 2017-01-31 --end 2020-12-31 [--config configs/default.yaml] [--synthetic-dir outputs/synthetic] [--output-dir outputs/factor_returns]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_model.config import load_config
from factor_model.data import fundamental_observations_from_frame, market_observations_from_frame
from factor_model.regression import compute_factor_returns, factor_return_rows_to_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic-dir", default="outputs/synthetic")
    parser.add_argument("--output-dir", default="outputs/factor_returns")
    args = parser.parse_args()

    engine_cfg = load_config(args.config)
    synthetic_dir = Path(args.synthetic_dir)

    market_df = pd.read_csv(synthetic_dir / "market.csv", parse_dates=["as_of", "known_as_of"])
    fundamentals_df = pd.read_csv(
        synthetic_dir / "fundamentals_raw.csv", parse_dates=["fundamental_period_end", "known_as_of"]
    )
    market_df["quality_flags"] = market_df["quality_flags"].fillna("")
    fundamentals_df["quality_flags"] = fundamentals_df["quality_flags"].fillna("")

    market_observations = market_observations_from_frame(market_df)
    fundamental_observations = fundamental_observations_from_frame(fundamentals_df)

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    period_dates = sorted({pd.Timestamp(d) for d in market_df["as_of"].unique() if start <= pd.Timestamp(d) <= end})

    rows, warnings = compute_factor_returns(market_observations, fundamental_observations, period_dates, engine_cfg)
    frame = factor_return_rows_to_frame(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"factor_returns_{start.date()}_{end.date()}.csv"
    frame.sort_values(["factor_name", "period_start"]).to_csv(out_path, index=False)

    print(f"wrote {len(frame)} factor-return rows across {len(period_dates) - 1} periods to {out_path}")
    if not frame.empty:
        summary = frame.groupby("factor_name")["factor_return"].agg(["mean", "std", "count"])
        print(summary)
    if warnings:
        print(f"{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
