# factor-model-engine

A systematic equity factor-model research system: point-in-time universe
construction, a factor library, exposure measurement, neutralization,
factor returns, portfolio risk decomposition, return attribution, regime
analysis, monitoring, and reporting.

**Stages 1–9 of 10 are complete**: the repository's data foundation
(schemas, point-in-time semantics, universe construction, cross-sectional
transforms, a deterministic synthetic dataset with a known ground-truth
factor structure), a four-factor library (Momentum, Value, Size, Quality)
that builds the full point-in-time Factor Exposure Table, an exposure
engine that aggregates stock-level exposures into signal/portfolio/
benchmark-level exposures, a neutralization engine that removes a chosen
subset of factor exposures from a signal via real cross-sectional OLS
while preserving every other factor's exposure, a factor-return engine
that estimates a real, lagged factor-return time series via
cross-sectional regression, a portfolio risk engine that decomposes
total variance into factor and specific risk with concentration checks,
an attribution engine that reconciles each subject's realized return
exactly into factor contributions plus a residual, a regime/monitoring
engine that classifies volatility regimes and flags real factor-stability
drift, and an end-to-end pipeline + reporting layer that ties every
stage together into one reproducible JSON/Markdown research report. See
[`docs/STATUS.md`](docs/STATUS.md) for stage-by-stage build status,
including a documented, investigated limitation of the synthetic Value
factor over long backtests (independently confirmed by Stages 5, 8, and
9). Stage 10 (hardening) is intentionally stubbed (`__init__.py` only)
and out of scope so far.

## Data

All data shipped and used by this repo's own tests and scripts is
**synthetic and non-proprietary** — a deterministic generator
(`src/factor_model/data/synthetic.py`) with a known, baked-in ground-truth
factor structure, built specifically so this repo is runnable and verifiable
without any real vendor data. Real-vendor adapters are explicitly out of
scope for Stage 1 (`src/factor_model/data/real_adapters/` is a stub).

## Point-in-time design

Two independently timestamped observation types —`MarketObservation` and
`FundamentalObservation` — each carry their *own* `known_as_of`, since real
price availability and real fundamental-publication lag are genuinely
different processes and must never be collapsed into one shared timestamp.
Factor **exposure** (a point-in-time characteristic snapshot) and factor
**return attribution** (a derived P&L contribution over a period) are
likewise two structurally separate schemas, never merged into one table.
See [`docs/data_dictionary.md`](docs/data_dictionary.md) for the full
schema and [`docs/research_assumptions.md`](docs/research_assumptions.md)
for every configurable research judgment call.

## Relationship to this portfolio's other repos

Depends on
[`alpha-validation-toolkit`](https://github.com/Patience-Fuglo/alpha-validation-toolkit)
(pinned git dependency) for shared, domain-agnostic statistical tooling —
OLS, IC/Rank IC/ICIR, PSR/DSR, purged walk-forward with embargo — and never
reimplements any of it. This repo does not replace, depend on, or modify
[`systematic-equity-research`](https://github.com/Patience-Fuglo/systematic-equity-research),
which holds strategy-specific signal code; this repo is generic factor
research infrastructure.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m pytest tests/ -v

python scripts/generate_synthetic_dataset.py
python scripts/run_data_health_report.py --as-of 2020-12-31
python scripts/build_factor_table.py --as-of 2020-06-30
python scripts/build_portfolio_exposure.py --as-of 2020-06-30
python scripts/neutralize_signal.py --as-of 2020-06-30 --signal-csv my_signal.csv --neutralize size
python scripts/compute_factor_returns.py --start 2017-01-31 --end 2020-12-31
python scripts/compute_portfolio_risk.py --as-of 2020-12-31 --start 2017-01-31 --end 2020-12-31
python scripts/attribute_returns.py --period-start 2020-05-31 --period-end 2020-06-30
python scripts/monitor_factors.py --start 2017-01-31 --end 2020-12-31
python scripts/generate_report.py --start 2017-01-31 --end 2020-12-31
```

## Repository layout

```
src/factor_model/
  config.py              # EngineConfig + load_config(): every ambiguous
                          # research decision lives in configs/, never hardcoded
  data/
    schemas.py            # MarketObservation, FundamentalObservation,
                           # UniverseInclusion/UniversePITSnapshot,
                           # FactorExposureRow, FactorAttributionRow
    point_in_time.py       # as_of_view(), latest_known_fundamental()
    universe.py             # build_point_in_time_universe()
    synthetic.py             # generate_synthetic_dataset() -- deterministic,
                              # known ground truth, labeled synthetic throughout
    real_adapters/            # stub -- later stage
  factors/
    transforms.py           # winsorize(), standardize()
    price_based.py           # compute_momentum()
    fundamental_based.py      # compute_book_to_price()
    size_based.py              # compute_size() -- log(market_cap)
    quality_based.py            # compute_profitability(), compute_leverage(),
                                 # compute_quality_composite()
    factor_library.py            # compute_factor_exposure_table() -- the
                                  # Stage 2 engine; builds the full Stock x
                                  # Date x Factor table
  diagnostics/
    data_health.py           # coverage/missingness/staleness/duplicate/
                              # outlier checks -- warns, never silently fixes
  exposures/
    portfolio_exposure.py    # aggregate_exposure() -- stock exposures ->
                              # signal/portfolio/benchmark exposures
  neutralization/
    neutralize.py             # neutralize_signal() -- real cross-sectional
                               # OLS (reuses alpha-validation-toolkit)
  regression/
    factor_returns.py          # compute_factor_returns() -- lagged
                                # Fama-MacBeth-style cross-sectional
                                # regression (reuses alpha-validation-toolkit)
  risk/
    factor_covariance.py       # compute_factor_covariance() -- real
                                # sample covariance over factor returns
    specific_risk.py             # compute_specific_returns()/
                                  # compute_specific_variance()
    portfolio_risk.py              # compute_portfolio_risk() -- factor
                                    # vs. specific variance decomposition
                                    # + concentration/risk-limit checks
  attribution/
    attribution.py              # compute_attribution() -- populates
                                 # FactorAttributionRow; exact
                                 # reconciliation to realized return
  regimes/
    regime_detection.py          # classify_volatility_regime() --
                                  # expanding-percentile vol regime
  monitoring/
    monitoring.py                  # compute_factor_stability() -- real
                                    # IC/ICIR drift detection (reuses
                                    # alpha-validation-toolkit)
  pipeline/
    pipeline.py                   # run_full_pipeline() -- orchestrates
                                   # Stages 2-8 end to end
  reporting/
    report.py                      # report_to_dict()/render_markdown_report()
                                    # -- both derived from one PipelineResult

tests/
  unit/           # schemas, config, synthetic data, point-in-time,
                  # universe, transforms, factors, data health
  integration/     # end-to-end point-in-time pipeline (restatement caught
                    # correctly through generator -> query -> universe/health)
  regression/       # golden-value test pinning the synthetic generator's
                     # summary statistics against silent drift

scripts/
  generate_synthetic_dataset.py   # writes outputs/synthetic/*.csv + metadata.json
  run_data_health_report.py        # writes outputs/reports/data_health_<as_of>.json
  build_factor_table.py             # writes outputs/factors/factor_table_<as_of>.csv
                                     # + factor_library_report_<as_of>.json
  build_portfolio_exposure.py        # writes outputs/exposures/<portfolio_id>_exposure_<as_of>.json
  neutralize_signal.py                # writes outputs/neutralization/neutralized_signal_<as_of>.json
  compute_factor_returns.py            # writes outputs/factor_returns/factor_returns_<start>_<end>.csv
  compute_portfolio_risk.py             # writes outputs/risk/portfolio_risk_<portfolio_id>_<as_of>.json
  attribute_returns.py                   # writes outputs/attribution/attribution_<start>_<end>.csv
  monitor_factors.py                      # writes outputs/monitoring/monitoring_report_<start>_<end>.json
  generate_report.py                       # writes outputs/reports_full/report_<portfolio_id>_<end>.{json,md}

docs/
  research_assumptions.md   # every configurable research judgment call
  data_dictionary.md         # every schema field, with PIT semantics
```
