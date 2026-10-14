#!/usr/bin/env python3
"""Stage 9 CLI: the full end-to-end pipeline (Stages 2-8) on the
synthetic dataset, rendered as one reproducible JSON + Markdown
research report. Reads
``outputs/synthetic/{market,fundamentals_raw}.csv`` -- run
``generate_synthetic_dataset.py`` first.

Usage:
    python scripts/generate_report.py --start 2017-01-31 --end 2020-12-31 [--portfolio-id benchmark_equal_weight] [--holdings-csv holdings.csv] [--config configs/default.yaml] [--synthetic-dir outputs/synthetic] [--output-dir outputs/reports_full]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from factor_model.config import load_config
from factor_model.data import PortfolioHolding, fundamental_observations_from_frame, market_observations_from_frame
from factor_model.pipeline import run_full_pipeline
from factor_model.reporting import render_markdown_report, report_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD -- estimation window start")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD -- as_of / estimation window end")
    parser.add_argument("--portfolio-id", default="benchmark_equal_weight")
    parser.add_argument("--holdings-csv", default=None, help="columns: security_id,weight")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic-dir", default="outputs/synthetic")
    parser.add_argument("--output-dir", default="outputs/reports_full")
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

    holdings = None
    if args.holdings_csv:
        holdings_df = pd.read_csv(args.holdings_csv)
        holdings = [
            PortfolioHolding(portfolio_id=args.portfolio_id, security_id=str(r.security_id), as_of=end, weight=float(r.weight), known_as_of=end)
            for r in holdings_df.itertuples(index=False)
        ]

    result = run_full_pipeline(market_obs, fundamental_obs, end, period_dates, engine_cfg, args.portfolio_id, holdings)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"report_{args.portfolio_id}_{end.date()}.json"
    md_path = out_dir / f"report_{args.portfolio_id}_{end.date()}.md"

    with open(json_path, "w") as f:
        json.dump(report_to_dict(result), f, indent=2, sort_keys=True)
        f.write("\n")
    with open(md_path, "w") as f:
        f.write(render_markdown_report(result))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if result.warnings:
        print(f"{len(result.warnings)} warning(s) across the full pipeline:")
        for w in result.warnings:
            print(f"  - {w}")
    else:
        print("no warnings")


if __name__ == "__main__":
    main()
