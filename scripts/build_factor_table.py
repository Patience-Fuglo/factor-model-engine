#!/usr/bin/env python3
"""Stage 2 CLI: build the real, point-in-time Factor Exposure Table
(Momentum, Value, Size, Quality) for the synthetic dataset as of a given
date, plus a concise factor-library data-quality report. Reads
``outputs/synthetic/{market,fundamentals_raw}.csv`` -- run
``generate_synthetic_dataset.py`` first.

Usage:
    python scripts/build_factor_table.py --as-of 2017-12-31 [--config configs/default.yaml] [--synthetic-dir outputs/synthetic] [--output-dir outputs/factors]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from factor_model.config import load_config
from factor_model.data import build_point_in_time_universe, fundamental_observations_from_frame, market_observations_from_frame
from factor_model.factors import compute_factor_exposure_table, factor_exposure_rows_to_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic-dir", default="outputs/synthetic")
    parser.add_argument("--output-dir", default="outputs/factors")
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

    as_of = pd.Timestamp(args.as_of)
    universe = build_point_in_time_universe(market_observations, as_of, engine_cfg.universe, engine_cfg.data)
    rows, report = compute_factor_exposure_table(
        market_observations, fundamental_observations, as_of, universe, engine_cfg
    )
    table = factor_exposure_rows_to_frame(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"factor_table_{as_of.date()}.csv"
    report_path = out_dir / f"factor_library_report_{as_of.date()}.json"

    table.sort_values(["factor_name", "security_id"]).to_csv(table_path, index=False)
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {len(table)} factor-exposure rows ({len(universe.included_ids())} eligible securities "
          f"x {len(set(table['factor_name']))} factors) to {table_path}")
    print(f"wrote {report_path}")
    if report.warnings:
        print(f"{len(report.warnings)} warning(s):")
        for w in report.warnings:
            print(f"  - {w}")
    else:
        print("no warnings")


if __name__ == "__main__":
    main()
