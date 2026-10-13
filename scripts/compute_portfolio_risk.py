#!/usr/bin/env python3
"""Stage 6 CLI: decompose a portfolio's real factor risk into factor and
specific variance, with concentration/risk-limit checks. Reads
``outputs/synthetic/{market,fundamentals_raw}.csv`` -- run
``generate_synthetic_dataset.py`` first.

Without ``--holdings-csv``, defaults to an equal-weight benchmark over
the real eligible universe at ``--as-of``.

Usage:
    python scripts/compute_portfolio_risk.py --as-of 2020-06-30 --start 2017-06-30 --end 2020-06-30 [--portfolio-id benchmark_equal_weight] [--holdings-csv holdings.csv] [--config configs/default.yaml] [--synthetic-dir outputs/synthetic] [--output-dir outputs/risk]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from factor_model.config import load_config
from factor_model.data import (
    ExposureSubject,
    PortfolioHolding,
    build_point_in_time_universe,
    fundamental_observations_from_frame,
    market_observations_from_frame,
)
from factor_model.exposures import aggregate_exposure, equal_weight_benchmark_holdings
from factor_model.factors import compute_factor_exposure_table
from factor_model.regression import compute_factor_returns
from factor_model.risk import compute_factor_covariance, compute_portfolio_risk, compute_specific_returns, compute_specific_variance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD -- the portfolio snapshot date")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD -- factor-return/covariance estimation window start")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD -- factor-return/covariance estimation window end")
    parser.add_argument("--portfolio-id", default="benchmark_equal_weight")
    parser.add_argument("--holdings-csv", default=None, help="columns: security_id,weight")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic-dir", default="outputs/synthetic")
    parser.add_argument("--output-dir", default="outputs/risk")
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

    as_of = pd.Timestamp(args.as_of)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    period_dates = sorted({pd.Timestamp(d) for d in market_df["as_of"].unique() if start <= pd.Timestamp(d) <= end})

    return_rows, return_warnings = compute_factor_returns(market_obs, fundamental_obs, period_dates, engine_cfg)
    covariance_result = compute_factor_covariance(return_rows, engine_cfg.risk)

    specific_returns = compute_specific_returns(market_obs, fundamental_obs, period_dates, engine_cfg)
    specific_variance = compute_specific_variance(specific_returns)

    universe = build_point_in_time_universe(market_obs, as_of, engine_cfg.universe, engine_cfg.data)
    stock_rows, _ = compute_factor_exposure_table(market_obs, fundamental_obs, as_of, universe, engine_cfg)

    if args.holdings_csv:
        holdings_df = pd.read_csv(args.holdings_csv)
        holdings = [
            PortfolioHolding(portfolio_id=args.portfolio_id, security_id=str(r.security_id), as_of=as_of, weight=float(r.weight), known_as_of=as_of)
            for r in holdings_df.itertuples(index=False)
        ]
    else:
        holdings = equal_weight_benchmark_holdings(universe.included_ids(), as_of, portfolio_id=args.portfolio_id)

    portfolio_rows = aggregate_exposure(stock_rows, holdings, ExposureSubject.PORTFOLIO, args.portfolio_id)
    result = compute_portfolio_risk(
        portfolio_rows, covariance_result.covariance, holdings, specific_variance, args.portfolio_id, engine_cfg.risk
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"portfolio_risk_{args.portfolio_id}_{as_of.date()}.json"
    payload = result.to_dict()
    payload["covariance_warnings"] = covariance_result.warnings
    payload["n_periods_used"] = covariance_result.n_periods
    payload["factor_return_warnings"] = return_warnings
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {out_path}")
    print(f"factor variance: {result.factor_variance:.6f}  specific variance: {result.specific_variance:.6f}  total: {result.total_variance:.6f}")
    print(f"effective n positions: {result.effective_n_positions:.1f}")
    all_warnings = result.warnings + covariance_result.warnings
    if all_warnings:
        print(f"{len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(f"  - {w}")
    else:
        print("no warnings")


if __name__ == "__main__":
    main()
