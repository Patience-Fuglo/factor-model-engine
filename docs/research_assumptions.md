# Research assumptions

Every field below is a real modeling judgment call, not a safe engineering
default, and needs the researcher's explicit sign-off before being relied on
for any real analysis. Each lives in `configs/default.yaml`, tagged
`# RESEARCH ASSUMPTION` at the point of definition, so the config file and
this document never drift apart silently.

## Data (`data:` in `configs/default.yaml`)

**`fundamental_publication_lag_days: 45`**
The assumed delay between a fiscal period ending and its fundamentals
becoming knowable, used only as a fallback when a real adapter has no
genuine filing timestamp (see `src/factor_model/data/schemas.py`,
`FundamentalObservation`). 45 days approximates a typical US 10-Q filing
lag. For a real adapter with an actual publication timestamp, that
timestamp should always be used instead — this value should never be
presented as if it were an observed fact; a fallback use is expected to
carry an `"estimated_known_as_of"` quality flag.

**`min_price_history_days: 252`**
Minimum trading history (~1 calendar year) required before a security is
eligible for the point-in-time universe. Chosen to guarantee a full real
12-month momentum lookback is actually available, not just a smaller
arbitrary floor.

**`stale_fundamentals_after_days: 190`**
Fundamentals older than this (relative to the query `as_of`) are flagged
stale by the data-health report. ~6.3 months allows one real missed
quarterly filing cycle (a quarter is nominally ~91 days, plus the
publication lag) before flagging, rather than flagging on the very day a
quarter turns over.

## Universe (`universe:`)

**`min_market_cap: 1.0e+8`**
A $100M floor, a common small-cap liquidity/size cutoff. Excludes micro-caps
where real market-impact and data-quality risk are both elevated.

**`min_adv_usd: 1.0e+6`**
A $1M/day average-dollar-volume floor. Excludes names where a real
institutional-size position could not be built or unwound without
materially moving price.

**`excluded_sectors: []`**
No sector exclusions by default — any sector constraint is a real portfolio
decision that belongs to the researcher running a specific study, not a
baked-in default.

## Factors (`factors:`)

**`winsorization.lower_pct: 0.01`, `upper_pct: 0.99`**
A standard 1%/99% tail trim for cross-sectional factor inputs, aggressive
enough to blunt real data errors and extreme single-name outliers without
discarding genuine tail observations outright.

**`standardization.method: "zscore"`**
Z-score is the default cross-sectional standardization. `rank` and
`minmax` are both implemented and swappable via config — z-score is
chosen as the default because it preserves relative distance information
that rank-based standardization discards, which matters for anything
downstream that estimates factor-return magnitudes (not just factor
rankings).

**`momentum.lookback_months: 12`, `skip_recent_months: 1`**
The classic "12-1" momentum convention: measure the return from 12 months
ago to 1 month ago, skipping the most recent month specifically because
short-term reversal is a real, separate, well-documented effect that would
otherwise contaminate a momentum signal. Deliberately kept configurable in
`MomentumConfig` rather than hardcoded, since a researcher may legitimately
want a different window for a different universe or horizon.

**`quality.profitability_weight: 0.5`, `quality.leverage_weight: 0.5`**
Equal-weighted by default between the profitability pillar (`EPS/BVPS`, a
real ROE proxy) and the leverage/safety pillar (`-total_debt/total_assets`).
Both pillars are standardized before combining (see
`docs/data_dictionary.md`), so these weights are a genuine mixing
decision — not a unit-conversion fudge. A researcher who believes
profitability is the more reliable quality signal for a given universe
could reasonably raise `profitability_weight` above 0.5.

## Diagnostics (`diagnostics:`)

**`max_allowed_missingness_pct: 0.10`**
Above 10% missingness in any real input field, the data-health report
warns. Chosen as a round, conservative threshold — missingness above this
level in a liquid-universe cross-section usually indicates a real feed
problem worth investigating, not normal sparse coverage.

**`min_cross_section_size: 20`**
Below 20 real securities, the data-health report warns the cross-section is
"too small to trust." This also sets the practical floor for the z-score
outlier check's own real statistical power (see
`src/factor_model/diagnostics/data_health.py`): with a single extreme
outlier among `n` points, the outlier's own `|z|` can never exceed
`sqrt(n-1)` — at `n=20` that caps at ~4.36, only barely above the
detector's 4.0 threshold. Below this floor, outlier detection is close to
powerless by construction, which is itself a reason to also distrust
anything else computed from that cross-section.

## Not a research assumption (synthetic test-fixture parameters)

`configs/synthetic_data.yaml`'s sizing, seeding, and the planted
restatement scenario (`n_securities`, `n_sectors`, `base_seed`, factor
loadings, persistence, `restatement_scenario.*`) are test-fixture
mechanics, not claims about real markets, and are not listed here — see
the module docstring in `src/factor_model/data/synthetic.py`.
