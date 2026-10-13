#!/usr/bin/env python3
"""Stage 4 CLI: neutralize a signal's raw cross-sectional scores against
a chosen subset of Stage 2's factor exposures. Reads
``outputs/factors/factor_table_<as_of>.csv`` -- run ``build_factor_table.py``
first -- and a signal-scores CSV (columns: ``security_id,score``).

Usage:
    python scripts/neutralize_signal.py --as-of 2020-06-30 --signal-csv signal.csv --neutralize size,sector [--factor-dir outputs/factors] [--output-dir outputs/neutralization]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from factor_model.neutralization import neutralize_signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    parser.add_argument("--signal-csv", required=True, help="columns: security_id,score")
    parser.add_argument("--neutralize", required=True, help="comma-separated factor names to neutralize")
    parser.add_argument("--factor-dir", default="outputs/factors")
    parser.add_argument("--output-dir", default="outputs/neutralization")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of)
    factor_table_path = Path(args.factor_dir) / f"factor_table_{as_of.date()}.csv"
    factor_table = pd.read_csv(factor_table_path)
    exposures = factor_table.pivot(index="security_id", columns="factor_name", values="standardized_score")

    signal_df = pd.read_csv(args.signal_csv)
    signal_scores = signal_df.set_index("security_id")["score"]

    factors_to_neutralize = [f.strip() for f in args.neutralize.split(",") if f.strip()]
    result = neutralize_signal(signal_scores, exposures, factors_to_neutralize)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"neutralized_signal_{as_of.date()}.json"
    payload = {
        "as_of": str(as_of.date()),
        "neutralized_factors": result.neutralized_factors,
        "preserved_factors": result.preserved_factors,
        "ols": {
            "alpha": result.ols_result.alpha,
            "betas": result.ols_result.betas,
            "r_squared": result.ols_result.r_squared,
            "t_stats": result.ols_result.t_stats,
            "n_obs": result.ols_result.n_obs,
        },
        "original_exposure_corr": result.original_exposure_corr,
        "residual_exposure_corr": result.residual_exposure_corr,
        "warnings": result.warnings,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    result.residual_scores.rename("residual_score").to_csv(out_dir / f"residual_scores_{as_of.date()}.csv")

    print(f"wrote {out_path}")
    print(f"neutralized: {result.neutralized_factors}  preserved: {result.preserved_factors}")
    print(f"R-squared removed: {result.ols_result.r_squared:.4f}")
    if result.warnings:
        print(f"{len(result.warnings)} warning(s):")
        for w in result.warnings:
            print(f"  - {w}")
    else:
        print("no warnings")


if __name__ == "__main__":
    main()
