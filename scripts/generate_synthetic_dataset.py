#!/usr/bin/env python3
"""Generate the deterministic synthetic demonstration dataset and write
it to ``outputs/synthetic/``. Re-running with the same config/seed must
produce byte-identical CSV output -- this is the real reproducibility
check this repo's Definition of Done requires, not just an in-memory
DataFrame comparison.

Usage:
    python scripts/generate_synthetic_dataset.py [--config configs/synthetic_data.yaml] [--output-dir outputs/synthetic]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from factor_model.data import generate_synthetic_dataset, load_synthetic_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/synthetic_data.yaml")
    parser.add_argument("--output-dir", default="outputs/synthetic")
    args = parser.parse_args()

    cfg = load_synthetic_config(args.config)
    dataset = generate_synthetic_dataset(cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sorted, fixed column order -> deterministic CSV bytes across runs
    dataset.market.sort_values(["security_id", "as_of"]).to_csv(out_dir / "market.csv", index=False)
    dataset.fundamentals.sort_values(["security_id", "fundamental_period_end", "known_as_of"]).to_csv(
        out_dir / "fundamentals_raw.csv", index=False
    )
    dataset.ground_truth.sort_values(["security_id", "as_of"]).to_csv(out_dir / "ground_truth.csv", index=False)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(dataset.metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {len(dataset.market)} market rows, {len(dataset.fundamentals)} fundamental rows, "
          f"{len(dataset.ground_truth)} ground-truth rows to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
