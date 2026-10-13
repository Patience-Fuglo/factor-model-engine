#!/usr/bin/env python3
"""Stage 3 CLI: aggregate a Stage 2 stock-level Factor Exposure Table
into one PORTFOLIO- (or SIGNAL-) level exposure per factor. Reads
``outputs/factors/factor_table_<as_of>.csv`` -- run
``build_factor_table.py`` first.

Without ``--holdings-csv``, defaults to an equal-weight benchmark over
every security present in the factor table. With it, expects columns
``security_id,weight`` for the named ``--portfolio-id``.

Usage:
    python scripts/build_portfolio_exposure.py --as-of 2020-06-30 --portfolio-id benchmark_equal_weight [--holdings-csv holdings.csv] [--factor-dir outputs/factors] [--output-dir outputs/exposures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from factor_model.data import ExposureSubject, FactorExposureRow, PortfolioHolding
from factor_model.exposures import aggregate_exposure, equal_weight_benchmark_holdings


def _load_stock_exposure_rows(factor_table_path: Path, as_of: pd.Timestamp) -> list[FactorExposureRow]:
    df = pd.read_csv(factor_table_path, parse_dates=["as_of", "source_known_as_of"])
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            FactorExposureRow(
                subject_type=ExposureSubject.STOCK,
                subject_id=row.security_id,
                as_of=as_of,
                factor_name=row.factor_name,
                exposure_value=float(row.standardized_score) if pd.notna(row.standardized_score) else float("nan"),
                source_known_as_of=pd.Timestamp(row.source_known_as_of),
                raw_value=None if pd.isna(row.raw_value) else float(row.raw_value),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    parser.add_argument("--portfolio-id", default="benchmark_equal_weight")
    parser.add_argument("--holdings-csv", default=None, help="columns: security_id,weight")
    parser.add_argument("--subject-type", default="portfolio", choices=["portfolio", "signal"])
    parser.add_argument("--factor-dir", default="outputs/factors")
    parser.add_argument("--output-dir", default="outputs/exposures")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of)
    factor_table_path = Path(args.factor_dir) / f"factor_table_{as_of.date()}.csv"
    stock_rows = _load_stock_exposure_rows(factor_table_path, as_of)

    if args.holdings_csv:
        holdings_df = pd.read_csv(args.holdings_csv)
        holdings = [
            PortfolioHolding(
                portfolio_id=args.portfolio_id, security_id=str(r.security_id), as_of=as_of,
                weight=float(r.weight), known_as_of=as_of,
            )
            for r in holdings_df.itertuples(index=False)
        ]
    else:
        security_ids = sorted({row.subject_id for row in stock_rows})
        holdings = equal_weight_benchmark_holdings(security_ids, as_of, portfolio_id=args.portfolio_id)

    subject_type = ExposureSubject.SIGNAL if args.subject_type == "signal" else ExposureSubject.PORTFOLIO
    result_rows = aggregate_exposure(stock_rows, holdings, subject_type, args.portfolio_id)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.portfolio_id}_exposure_{as_of.date()}.json"
    payload = [
        {
            "subject_type": row.subject_type.value,
            "subject_id": row.subject_id,
            "as_of": str(row.as_of.date()),
            "factor_name": row.factor_name,
            "exposure_value": row.exposure_value,
            "source_known_as_of": str(row.source_known_as_of.date()),
            "quality_flags": list(row.quality_flags),
        }
        for row in result_rows
    ]
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {len(result_rows)} {args.subject_type} exposure rows for '{args.portfolio_id}' to {out_path}")
    for row in result_rows:
        flag_note = f" [{','.join(row.quality_flags)}]" if row.quality_flags else ""
        print(f"  {row.factor_name}: {row.exposure_value:.4f}{flag_note}")


if __name__ == "__main__":
    main()
