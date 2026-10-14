#!/usr/bin/env python3
"""Stage 8 CLI: regime classification + factor-stability monitoring
over a real date range on the synthetic dataset. Reads
``outputs/synthetic/{market,fundamentals_raw}.csv`` -- run
``generate_synthetic_dataset.py`` first.

Usage:
    python scripts/monitor_factors.py --start 2017-01-31 --end 2020-12-31 [--config configs/default.yaml] [--synthetic-dir outputs/synthetic] [--output-dir outputs/monitoring]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from factor_model.config import load_config
from factor_model.data import fundamental_observations_from_frame, market_observations_from_frame
from factor_model.monitoring import compute_factor_stability
from factor_model.regimes import classify_volatility_regime
from factor_model.regression import compute_factor_returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic-dir", default="outputs/synthetic")
    parser.add_argument("--output-dir", default="outputs/monitoring")
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

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    period_dates = sorted({pd.Timestamp(d) for d in market_df["as_of"].unique() if start <= pd.Timestamp(d) <= end})

    return_rows, return_warnings = compute_factor_returns(market_obs, fundamental_obs, period_dates, engine_cfg)
    stability = compute_factor_stability(market_obs, fundamental_obs, period_dates, engine_cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_names = sorted({row.factor_name for row in return_rows})
    regime_summary = {}
    for factor_name in factor_names:
        regime_frame = classify_volatility_regime(return_rows, factor_name, engine_cfg.regimes)
        regime_frame.to_csv(out_dir / f"regime_{factor_name}_{start.date()}_{end.date()}.csv")
        latest = regime_frame.dropna(subset=["regime"]).tail(1)
        regime_summary[factor_name] = None if latest.empty else latest["regime"].iloc[0]

    stability_payload = {name: result.to_dict() for name, result in stability.items()}
    out_path = out_dir / f"monitoring_report_{start.date()}_{end.date()}.json"
    with open(out_path, "w") as f:
        json.dump({"latest_regime": regime_summary, "factor_stability": stability_payload, "factor_return_warnings": return_warnings}, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {out_path}")
    print("latest regime per factor:", regime_summary)
    for name, result in stability.items():
        print(f"{name}: ICIR={result.icir:.3f}  historical IC mean={result.historical_ic_mean:.3f}  recent IC mean={result.recent_ic_mean:.3f}")
        for w in result.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
