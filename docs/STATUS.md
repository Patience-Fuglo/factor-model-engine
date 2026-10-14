# Build status

Tracks stage-by-stage completion for the coordinated Stages 2-10 build, so
work is resumable across sessions without repeating anything already
verified. Updated at each stage's real completion, not before.

## Final summary (all 10 stages complete)

**Architecture.** Ten stages, each reusing (never duplicating) the
previous: `data/` (schemas, point-in-time helpers, universe, synthetic
generator) -> `factors/` (Momentum/Value/Size/Quality -> the Factor
Exposure Table) -> `exposures/` (stock -> signal/portfolio/benchmark
aggregation) -> `neutralization/` (real cross-sectional OLS, via
`alpha-validation-toolkit`) -> `regression/` (lagged Fama-MacBeth factor
returns) -> `risk/` (factor covariance, specific risk, portfolio
decomposition) -> `attribution/` (exact P&L reconciliation) ->
`regimes/` + `monitoring/` (volatility regimes, IC-based drift
detection, reusing the toolkit's `information_coefficient`/`icir`) ->
`pipeline/` + `reporting/` (one orchestration function, one JSON/
Markdown report derived from it). `alpha-validation-toolkit` is reused
for all real statistical machinery (OLS, IC, ICIR) across Stages 4, 5,
6, and 8 -- never reimplemented.

**Important formulas** (full detail in `docs/data_dictionary.md`):
Momentum = trailing return, 12-1 default, configurable. Value =
book-to-price (latest point-in-time fundamental vintage / latest
price). Size = log(market cap). Quality = weighted average of
standardized profitability (EPS/BVPS) and standardized safety
(-total_debt/total_assets). Factor return = cross-sectional OLS beta of
realized returns on LAGGED (period-start) exposures. Portfolio factor
variance = `w^T Sigma w`. Attribution reconciles exactly:
`realized_return == sum(contribution_to_return)` including the
residual.

**Test counts.** 157 tests passing: unit tests per module, integration
tests on the real synthetic dataset for every stage transition, the
original Stage 1 golden regression suite (still pinned, byte-identical
through every later stage's changes), and two Stage 10 hardening
tests -- a full real CLI-chain subprocess run, and the single most
comprehensive point-in-time check in the repo (below).

**Point-in-time integrity, the decisive evidence.** Stage 10 added
`test_extending_the_dataset_into_the_future_never_changes_an_earlier_full_pipeline_run`:
runs the ENTIRE pipeline (universe through the final report) on a
36-period dataset, then again on a 48-period dataset generated from
the same seed (12 REAL extra future months added), both queried at the
SAME `as_of`. The two full reports come back byte-for-byte identical.
This is the system-wide version of what every earlier stage already
verified locally (as_of_view, latest_known_fundamental, universe
construction, lagged factor returns, regime classification) -- proven
once, holistically, rather than only piece by piece.

**Reproducibility.** Every stage's CLI output and every `PipelineResult`
/report is verified byte-identical across two independent runs on the
same input, including the full end-to-end `generate_report.py` output
(`test_full_pipeline_reconciles_and_reproduces_on_the_default_scale_dataset`).

**End-to-end demonstration.** `scripts/generate_report.py` runs the
complete Stages 2-9 pipeline and writes one real JSON + Markdown
research report; `tests/integration/test_cli_workflow.py` runs every
CLI script in this repo in sequence, as real subprocesses, exactly as a
researcher would from the command line (see README "Run").

**Limitations, identified rather than hidden:**
- The synthetic Value factor's correlation with its planted
  `value_score` reverses sign over a long backtest (a real compounding
  confound in the Stage 1 generator's price-path construction, not a
  bug) -- documented in Stages 5, 8, and reproduced again in Stage 9's
  report. A researcher using this repo on real data should not expect
  this specific artifact, but should expect the engine to surface a
  real decay/drift finding like it when one is genuinely present.
- Factor covariance (Stage 6) is a plain sample covariance, no
  shrinkage/regularization -- flagged as unstable below a configured
  minimum period count, never silently trusted, but not yet corrected.
- Specific risk assumes idiosyncratic returns are cross-sectionally
  uncorrelated (the standard textbook approximation, not verified
  against the real data).
- Attribution's `contribution_to_return` is a linear approximation
  (`exposure * factor_return`), documented as such since Stage 1 --
  not an exact multiplicative P&L decomposition.
- All data is synthetic; no real vendor adapter exists yet
  (`data/real_adapters/` remains a stub, exactly as planned in Stage 1).
- Out of scope by explicit instruction: portfolio optimization, full
  transaction-cost modeling, order management, and anything beyond
  this one engine's boundaries (the other six planned senior-quant
  engines remain separate, untouched repos).

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

## Stage 5 -- Factor Returns: COMPLETE

Added `FactorReturnRow` schema (period_start, period_end, factor_name,
factor_return, r_squared, n_obs, known_as_of) and
`regression/factor_returns.py` (`compute_factor_returns`): a real
Fama-MacBeth-style cross-sectional regression -- for each consecutive
pair of real period dates, regresses REALIZED stock returns over that
period on each stock's real exposure AS OF the period's START (lagged,
never the period's end), reusing Stage 2's `compute_factor_exposure_table`
for the lagged exposures and `alpha-validation-toolkit`'s real
`ols_regression` for the fit.

A factor with no usable reading for ANY security in a given period
(e.g. momentum before 12 real months of history exist) is dropped from
THAT period's joint regression with a warning, rather than letting one
all-NaN column wipe out every row for every factor via the toolkit's
own row-wise `dropna`.

**A real, investigated, documented finding (not a bug, not hidden):**
the synthetic Value factor's (book-to-price) correlation with its own
planted `value_score` decays and can REVERSE SIGN over a long synthetic
backtest -- confirmed directly: +0.35 at an early reporting period,
-0.14 by the final period 36 months later. Root cause: book-to-price's
denominator (price) is a cumulative product of returns that are
themselves partly driven by `value_score`, so a persistently
high-`value_score` security's price compounds upward over time and
eventually outpaces the direct effect on book value -- mechanically
shrinking the ratio back down, the same dynamic as a real "value trap"
re-rating. Momentum does NOT share this issue (it's measured directly
on price returns, the same quantity driving `realized_return`) and is
recovered reliably, confirmed stable across every window length tested.
This is a genuine property of the Stage 1 generator's design, not
something Stage 5 can or should "fix" without touching Stage 1's
pinned `realized_return`/price formulas -- documented here and in
`docs/data_dictionary.md` rather than papered over with a cherry-picked
test window.

6 new tests (4 unit + 2 integration on the real synthetic dataset; 120
total, was 114). New CLI: `scripts/compute_factor_returns.py`. Pushed.

## Stage 6 -- Portfolio + Risk: COMPLETE

Added `RiskConfig` (min_periods_for_covariance, max_single_factor_variance_share,
min_effective_n_positions -- all RESEARCH ASSUMPTION) and three new
`risk/` modules:

- `factor_covariance.py` (`compute_factor_covariance`): a real sample
  covariance matrix (`pandas.cov`) over Stage 5's factor-return panel.
  No shrinkage/regularization yet (a later-stage refinement); warns,
  rather than silently proceeding, when too few real periods are
  available relative to the number of factors (a well-known small-
  sample covariance instability).
- `specific_risk.py` (`compute_specific_returns`/`compute_specific_variance`):
  reuses Stage 5's exact per-period fit (refactored `regression._fit_one_period`,
  exposed rather than duplicated) to recover each period's real
  cross-sectional residual, then each security's own specific-return
  variance across time -- the standard textbook specific-risk input.
- `portfolio_risk.py` (`compute_portfolio_risk`): real factor variance
  as the quadratic form `w^T Sigma w` (the portfolio's own Stage 3
  exposure vector against the real factor covariance), decomposed into
  each factor's real marginal variance contribution (sums exactly to
  the total); specific variance as `sum(weight_i^2 * specific_variance_i)`,
  assuming uncorrelated idiosyncratic returns (the standard
  approximation); a real Herfindahl-based effective-n-positions
  concentration measure; and risk-limit warnings (single-factor
  variance share, minimum effective positions) against `RiskConfig`.
  Missing specific-variance data for a holding is excluded, not
  zero-filled, with a warning.

Refactored (not just extended) `regression/factor_returns.py`:
extracted the existing per-period lagged-regression logic into an
exposed `PeriodFit`/`_fit_one_period`, so Stage 6's specific-return
computation reuses the EXACT same regression Stage 5's factor returns
come from, rather than risking the two silently drifting into two
different real regressions over time. `compute_factor_returns`'s own
behavior is unchanged (re-verified: all Stage 5 tests still pass,
with one test updated for an improved, now-single comprehensive
warning instead of two redundant ones -- confirmed as an improvement,
not a regression).

11 new tests (9 unit + 1 integration on the real synthetic dataset;
130 total, was 120). New CLI: `scripts/compute_portfolio_risk.py`.
Verified end-to-end: an equal-weight benchmark's factor variance comes
out ~0 (consistent with Stage 3's own ~0 exposure finding for the same
construction) and its effective-n-positions equals its real security
count, as expected for an equal-weight portfolio. Pushed.

## Stage 7 -- Attribution: COMPLETE

Added `attribution/attribution.py` (`compute_attribution`): finally
populates `FactorAttributionRow`, the schema Stage 1 defined but left
empty. Combines a subject's real exposure AT PERIOD_START (Stage 2/3)
with each real factor's realized return OVER THE PERIOD (Stage 5) into
a per-factor P&L contribution, plus one explicit RESIDUAL row for
whatever the subject's own real realized return the factors couldn't
explain.

The real reconciliation identity -- `realized_return == sum(
contribution_to_return for every row, including the residual)` --
holds EXACTLY, not approximately, by construction, and is verified
directly: both in isolated unit tests and on 10 real securities from
the synthetic dataset in the integration test. A factor missing either
a real exposure or a real factor return for a given subject/period is
folded into the residual with a warning, never silently treated as a
zero-exposure (and therefore zero-contribution) factor.

7 new tests (6 unit + 1 integration; 137 total, was 130). New CLI:
`scripts/attribute_returns.py`. Pushed.

## Stage 8 -- Regimes + Monitoring: COMPLETE

Added `RegimeConfig` (volatility_window, low/high_volatility_percentile)
and `MonitoringConfig` (drift_recent_window, drift_decay_threshold --
all RESEARCH ASSUMPTION), and two new modules:

- `regimes/regime_detection.py` (`classify_volatility_regime`): a real
  trailing rolling volatility of one factor's return series, classified
  into low/normal/high via an EXPANDING (never full-sample) percentile
  rank -- a regime label for period t must only reflect what was
  knowable up to and including t, the same point-in-time discipline
  this repo applies everywhere else. A real, caught pandas-semantics
  pitfall: `Expanding.rank()`'s own `min_periods` counts non-null
  observations, not raw rows -- reusing the same strict threshold as
  the rolling std's own `min_periods` meant no regime label could ever
  appear within a modest test history; fixed by using a much smaller,
  separate `min_periods=2` for the rank step itself (a percentile among
  2+ real readings is already meaningful, just coarse -- it gets more
  reliable with more history, which is expected, not a defect).
- `monitoring/monitoring.py` (`compute_factor_stability`): a real
  period-by-period cross-sectional Information Coefficient between each
  factor's LAGGED exposure and its realized return, reusing Stage 5/6's
  exact per-period fit (`_fit_one_period`, now a third consumer) and
  `alpha-validation-toolkit`'s real `information_coefficient`/`icir` --
  never reimplemented. Flags a real drift signal when a factor's RECENT
  IC mean has flipped sign or decayed sharply vs. its own full history.

9 new tests (8 unit + 1 integration on the real synthetic dataset; 146
total, was 137). New CLI: `scripts/monitor_factors.py`. Verified
end-to-end on the real synthetic dataset: Momentum (the one factor
Stage 5 already confirmed recovers reliably) shows a consistently
positive IC (ICIR > 2 in one real run) with no drift warning; Value
(the one factor Stage 5 already documented as unreliable over long
horizons) showed a real decay warning in the same run -- the Stage 8
monitoring correctly flagged exactly the limitation Stage 5 already
investigated and documented, rather than a different, unrelated one.
Pushed.

## Stage 9 -- Reporting: COMPLETE

Added `pipeline/pipeline.py` (`run_full_pipeline`) -- the end-to-end
orchestration this repo's own layout reserved from Stage 1: runs
Stages 2-8 once, in dependency order, for one real `as_of`/portfolio
(universe -> factor library -> portfolio exposure -> factor returns ->
covariance -> specific risk -> portfolio risk -> attribution ->
regimes/stability), reusing each stage's own tested function --
nothing in it re-derives a computation another stage already owns.

Added `reporting/report.py` (`report_to_dict`/`render_markdown_report`):
both derived from the SAME `PipelineResult`, so the JSON and Markdown
outputs can never silently disagree with each other. The Markdown
report states plainly that its data is synthetic/non-proprietary (same
convention as the README), and surfaces every real warning collected
across the whole pipeline, not just a per-stage subset.

8 new tests (3 pipeline + 5 reporting, all exercising the real
synthetic dataset end to end; 154 total, was 146). New CLI:
`scripts/generate_report.py`. Verified: the full JSON + Markdown report
is byte-identical across two independent runs on the same synthetic
data, and correctly surfaces the SAME two real findings Stage 5
(Value's long-horizon sign reversal) and Stage 8 (Value's IC decay)
already investigated and documented -- the orchestration layer
reproduces, rather than contradicts, what each stage already found
independently.

## Stage 10 -- Hardening: COMPLETE

Added `tests/integration/test_full_system.py`:
`test_extending_the_dataset_into_the_future_never_changes_an_earlier_full_pipeline_run`
(the comprehensive, system-wide PIT check described in the Final
Summary above) and
`test_full_pipeline_reconciles_and_reproduces_on_the_default_scale_dataset`
(the real default-scale, liquidity-filtered universe, not the
`min_market_cap=0` relaxation most other tests use for simplicity).

Added `tests/integration/test_cli_workflow.py`: runs every one of this
repo's 9 CLI scripts in sequence, as real subprocesses with
`subprocess.run`, exactly as a researcher would from the command line
-- confirming the documented README workflow actually works, not just
the underlying library functions in isolation.

Registered the `slow` pytest marker (`pyproject.toml`) for the
subprocess-based CLI test, which is slower than the rest of the suite
but still runs by default.

157 tests passing (was 154). Full leak-check clean across the whole
repo. Pushed.

**All 10 stages of the factor-model-engine build are now complete.**
