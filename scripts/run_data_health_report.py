#!/usr/bin/env python3
"""Run the real data-health report against the synthetic dataset as of a
given date, and write the result as JSON. Reads
``outputs/synthetic/{market,fundamentals_raw}.csv`` -- run
``generate_synthetic_dataset.py`` first.

Usage:
    python scripts/run_data_health_report.py --as-of 2017-12-31 [--config configs/default.yaml] [--synthetic-dir outputs/synthetic] [--output-dir outputs/reports]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from factor_model.config import load_config
from factor_model.data import fundamental_observations_from_frame, market_observations_from_frame
from factor_model.diagnostics import run_data_health_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic-dir", default="outputs/synthetic")
    parser.add_argument("--output-dir", default="outputs/reports")
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
    report = run_data_health_report(
        market_observations, fundamental_observations, as_of, engine_cfg.data, engine_cfg.diagnostics
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"data_health_{as_of.date()}.json"
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {out_path}")
    if report.warnings:
        print(f"{len(report.warnings)} warning(s):")
        for w in report.warnings:
            print(f"  - {w}")
    else:
        print("no warnings")


if __name__ == "__main__":
    main()
