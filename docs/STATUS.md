# Build status

Tracks stage-by-stage completion for the coordinated Stages 2-10 build, so
work is resumable across sessions without repeating anything already
verified. Updated at each stage's real completion, not before.

## Stage 1 -- Foundation: COMPLETE

Schemas (point-in-time market/fundamental observations, universe,
exposure/attribution split), point-in-time query helpers, universe
construction, cross-sectional transforms, Momentum/Value factors,
data-health diagnostics, deterministic synthetic data generator.
78 tests passing. Pushed.

## Stage 2 -- Factor Library: COMPLETE

Added Size (log market cap) and Quality (profitability + leverage
composite) factors; extended `FundamentalObservation` with
`total_assets`/`total_debt` (same vintage, same `known_as_of` as the rest
of that filing); extended `FactorExposureRow` with `raw_value` +
`quality_flags` (additive, backward-compatible). New orchestration module
`factors/factor_library.py` (`compute_factor_exposure_table`) builds the
full Stock x Date x Factor table, reusing Stage 1's momentum/book-to-price/
winsorize/standardize rather than duplicating them. Synthetic generator
extended with an independent latent `quality_score` (own RNG stream, never
touches Stage 1's `realized_return`/price formulas) driving `total_assets`/
`total_debt`, enabling a real planted-relationship recoverability test.

A real bug was caught and fixed during this build: reusing the existing
per-security `fundamentals` RNG stream (which is drawn from at every
reporting period) for the new asset/debt noise draws shifted every LATER
period's `book_noise`/`eps_noise` draw, silently changing Stage 1's
already-pinned `book_value_per_share`/`earnings_per_share` means. Fixed by
giving the new draws their own independent `quality_noise` stream. Caught
by the Stage 1 golden regression test, which is exactly what it's for.

96 tests passing (was 78; +18 Stage 2). CLI:
`scripts/build_factor_table.py`. Verified: all Stage 1 golden-pinned
values unchanged; new synthetic fields' own values newly pinned;
factor-table CLI output byte-identical across two independent runs;
momentum point-in-time tested against a planted future observation; the
Quality factor's leverage pillar recovers its planted ground-truth
relationship at |corr| > 0.4, and the full composite at > 0.3 (diluted,
as expected, by the independent profitability pillar -- documented, not
hidden, in the test itself). Pushed.

## Stage 3 -- Exposure Engine: COMPLETE

Added `PortfolioHolding` schema (portfolio_id, security_id, as_of,
weight, known_as_of) and `exposures/portfolio_exposure.py`
(`aggregate_exposure`): a real weighted-average aggregation of Stage 2
stock-level exposures into one SIGNAL- or PORTFOLIO-level
`FactorExposureRow` per factor. Re-normalizes over only the real
available constituent exposures when some are missing (never zero-fills
a missing stock exposure); flags `"insufficient_coverage"` when nothing
real is available; propagates `source_known_as_of` as the LATEST real
input among every contributing constituent. A benchmark is just another
`PortfolioHolding` weight scheme aggregated through the exact same
function (`equal_weight_benchmark_holdings` provided as a convenience) --
no separate code path.

11 new tests (10 unit + 1 integration using the real synthetic dataset;
107 total, was 96). New CLI: `scripts/build_portfolio_exposure.py`.
Verified end-to-end: an equal-weight benchmark over the FULL eligible
universe correctly averages to ~0.0 for every z-scored factor (the real,
expected consequence of standardization being mean-zero across that same
universe) -- a strong sanity check that Stage 2 -> Stage 3 wiring is
correct, not just that each stage passes its own isolated tests. Pushed.

## Stage 4 -- Neutralization: COMPLETE

Added `neutralization/neutralize.py` (`neutralize_signal`): regresses a
signal's raw cross-sectional scores on an explicit, researcher-chosen
subset of factor exposures via `alpha-validation-toolkit`'s real
`ols_regression`/`OLSResult` (reused, never reimplemented). The residual
is the neutralized signal. Every factor NOT named is a `preserved_factor`
-- reported with its own before/after correlation to the signal, never
regressed out, so a researcher can confirm an intentional, profitable
exposure genuinely survived neutralization.

A real bug was caught and fixed: the residual-exposure warning check
computed a correlation against the residual directly, which is
numerically unstable when the residual has near-zero variance (e.g. an
exact, noiseless relationship leaves a residual of pure float noise at
~1e-16) -- that noise's correlation with anything can spuriously land
anywhere, including near 1.0, producing a false "still correlated"
warning on a PERFECT neutralization. Fixed with a variance-floor guard
(`_safe_corr`) that reports 0.0 rather than an unstable number when
either series' real standard deviation is below a small floor. Caught
by a test that constructed exactly that exact/noiseless case, not
hypothetically.

7 new tests (6 unit + 1 integration on the real synthetic dataset; 114
total, was 107). New CLI: `scripts/neutralize_signal.py`. Pushed.

## Next up: Stage 5 -- Factor Returns

Cross-sectional regression of realized returns on LAGGED (period-start)
factor exposures, per period -> a new `FactorReturnRow` time series +
regression diagnostics. Feeds Stages 6 and 7.

## Stage 6 -- Portfolio + Risk: NOT STARTED

Factor covariance from Stage 5's return history; portfolio factor
exposure vector (from Stage 3); risk decomposition (factor vs. specific
risk); concentration/risk-limit checks.

## Stage 7 -- Attribution: NOT STARTED

Populate `FactorAttributionRow` (already schema'd in Stage 1) using
Stage 2/3 period-start exposures x Stage 5 realized factor returns.
Separate factor-driven return from security selection/residual.

## Stage 8 -- Regimes + Monitoring: NOT STARTED

Rolling factor-return stability/correlation regime detection; factor
drift diagnostics; alerting thresholds, building on Stage 5's history.

## Stage 9 -- Reporting: NOT STARTED

Reproducible report (markdown/JSON) bundling factor tables, risk summary,
attribution, and regime diagnostics from all prior stages.

## Stage 10 -- Hardening: NOT STARTED

Full cross-stage integration/leak/reproducibility sweep, consolidated
documentation and CLI workflows, final end-to-end demonstration.
