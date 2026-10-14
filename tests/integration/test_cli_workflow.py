"""Stage 10 -- runs every real CLI script in this repo, in the order a
researcher actually would, against one real synthetic dataset, and
confirms each produces its real expected output file. A unit test on
a library function proves the function works; this proves the
documented command-line workflow in the README actually works too, as
subprocesses, exactly as a user would run them.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"{args}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result


@pytest.mark.slow
def test_full_cli_workflow_runs_in_sequence_and_produces_every_real_output_file(tmp_path):
    synthetic_dir = tmp_path / "synthetic"
    factors_dir = tmp_path / "factors"
    exposures_dir = tmp_path / "exposures"
    neutralization_dir = tmp_path / "neutralization"
    returns_dir = tmp_path / "factor_returns"
    risk_dir = tmp_path / "risk"
    attribution_dir = tmp_path / "attribution"
    monitoring_dir = tmp_path / "monitoring"
    report_dir = tmp_path / "reports_full"

    as_of = "2020-06-30"
    start, end = "2017-06-30", "2020-06-30"

    _run(["scripts/generate_synthetic_dataset.py", "--output-dir", str(synthetic_dir)], REPO_ROOT)
    assert (synthetic_dir / "market.csv").exists()
    assert (synthetic_dir / "fundamentals_raw.csv").exists()
    assert (synthetic_dir / "metadata.json").exists()

    _run(["scripts/run_data_health_report.py", "--as-of", as_of, "--synthetic-dir", str(synthetic_dir),
          "--output-dir", str(tmp_path / "health")], REPO_ROOT)

    _run(["scripts/build_factor_table.py", "--as-of", as_of, "--synthetic-dir", str(synthetic_dir),
          "--output-dir", str(factors_dir)], REPO_ROOT)
    factor_table_path = factors_dir / f"factor_table_{as_of}.csv"
    assert factor_table_path.exists()

    _run(["scripts/build_portfolio_exposure.py", "--as-of", as_of, "--factor-dir", str(factors_dir),
          "--output-dir", str(exposures_dir)], REPO_ROOT)
    assert (exposures_dir / f"benchmark_equal_weight_exposure_{as_of}.json").exists()

    signal_csv = tmp_path / "signal.csv"
    table = pd.read_csv(factor_table_path)
    pivot = table.pivot(index="security_id", columns="factor_name", values="standardized_score")
    (pivot["momentum"] + pivot["size"]).rename("score").to_csv(signal_csv)
    _run(["scripts/neutralize_signal.py", "--as-of", as_of, "--signal-csv", str(signal_csv), "--neutralize", "size",
          "--factor-dir", str(factors_dir), "--output-dir", str(neutralization_dir)], REPO_ROOT)
    assert (neutralization_dir / f"neutralized_signal_{as_of}.json").exists()

    _run(["scripts/compute_factor_returns.py", "--start", start, "--end", end, "--synthetic-dir", str(synthetic_dir),
          "--output-dir", str(returns_dir)], REPO_ROOT)
    assert any(returns_dir.glob("factor_returns_*.csv"))

    _run(["scripts/compute_portfolio_risk.py", "--as-of", as_of, "--start", start, "--end", end,
          "--synthetic-dir", str(synthetic_dir), "--output-dir", str(risk_dir)], REPO_ROOT)
    assert any(risk_dir.glob("portfolio_risk_*.json"))

    _run(["scripts/attribute_returns.py", "--period-start", "2020-05-31", "--period-end", as_of,
          "--synthetic-dir", str(synthetic_dir), "--output-dir", str(attribution_dir)], REPO_ROOT)
    assert any(attribution_dir.glob("attribution_*.csv"))

    _run(["scripts/monitor_factors.py", "--start", start, "--end", end, "--synthetic-dir", str(synthetic_dir),
          "--output-dir", str(monitoring_dir)], REPO_ROOT)
    assert any(monitoring_dir.glob("monitoring_report_*.json"))

    _run(["scripts/generate_report.py", "--start", start, "--end", end, "--synthetic-dir", str(synthetic_dir),
          "--output-dir", str(report_dir)], REPO_ROOT)
    assert any(report_dir.glob("report_*.json"))
    assert any(report_dir.glob("report_*.md"))
