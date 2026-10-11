# Data dictionary

Every schema in `src/factor_model/data/schemas.py`, field by field, with its
point-in-time (PIT) semantics. Five structurally separate concepts — never
merged into one table, by design (see the module docstring for why).

## `MarketObservation`

A security's real price/volume state on one trading date.

| Field | Type | PIT semantics |
|---|---|---|
| `security_id` | `str` | Stable surrogate key — never the ticker, which can be reused across different real companies over time. |
| `ticker` | `str` | Display-only; never used as a join key. |
| `as_of` | `Timestamp` | The real trading date this observation describes. |
| `known_as_of` | `Timestamp` | When *this* observation became knowable. For real market data this is almost always `== as_of` (same-day/near-instant). Must be `>= as_of` — the constructor raises otherwise (look-ahead). |
| `sector` | `Sector` | Point-in-time sector classification as of this observation — not today's classification applied backward. |
| `is_active` | `bool` | Whether the security was actively listed/tradeable as of this date. |
| `price` | `float` | Real close price. |
| `shares_outstanding` | `float` | Real shares outstanding as of this date. |
| `market_cap` | `float` | `price * shares_outstanding`, as of this date. |
| `adv_20d` | `float \| None` | 20-day average dollar volume. `None` when there isn't enough real history to compute it yet — never silently `0.0`, which would make an illiquid name indistinguishable from an untradeable one. |
| `quality_flags` | `tuple[str, ...]` | Carried on the observation itself (e.g. `"estimated_known_as_of"`), not inferred later by an external diagnostics pass. |

## `FundamentalObservation`

One real vintage of a security's fundamentals for one fiscal period. A
`(security_id, fundamental_period_end)` pair can legitimately have multiple
vintages over time (restatements) — this is expected, not an error.

| Field | Type | PIT semantics |
|---|---|---|
| `security_id` | `str` | Same surrogate key as `MarketObservation`. |
| `fundamental_period_end` | `Timestamp` | The fiscal period this vintage describes. |
| `known_as_of` | `Timestamp` | When *this specific vintage* became knowable. Its own, independent field — genuinely distinct from, and never to be confused with, `MarketObservation.known_as_of` on the same security and date, since fundamental publication lag and market-data availability are different real processes. Must be `>= fundamental_period_end` — the constructor raises otherwise. Ideally a real vendor filing timestamp; `data.fundamental_publication_lag_days` is a fallback only (see `docs/research_assumptions.md`). |
| `book_value_per_share` | `float \| None` | `None` on real missing data — never zero-filled. |
| `earnings_per_share` | `float \| None` | Same. |
| `quality_flags` | `tuple[str, ...]` | e.g. `"restated"`, `"estimated_known_as_of"`. |

To resolve "which vintage would a researcher at a given `as_of` have
actually seen," use `point_in_time.latest_known_fundamental` — never
`max(observations, key=lambda o: o.known_as_of)`, which leaks a future
restatement into a query made before it was knowable.

## `UniverseInclusion` / `UniversePITSnapshot`

Point-in-time universe membership, with the reason for every real inclusion
or exclusion recorded explicitly — never built by reconstructing history
from today's constituents.

| Field | Type | Notes |
|---|---|---|
| `UniverseInclusion.security_id` | `str` | |
| `UniverseInclusion.included` | `bool` | |
| `UniverseInclusion.reasons` | `tuple[ExclusionReason, ...]` | Empty iff `included`; non-empty iff excluded — enforced by `__post_init__`. |
| `UniversePITSnapshot.as_of` | `Timestamp` | |
| `UniversePITSnapshot.members` | `tuple[UniverseInclusion, ...]` | |
| `UniversePITSnapshot.config_hash` | `str` | Audit trail: a hash of the `UniverseConfig`/`DataConfig` used to build this snapshot, so two snapshots with different filter settings are distinguishable even if their membership happens to match. |

`ExclusionReason`: `NOT_YET_LISTED`, `DELISTED`, `INSUFFICIENT_PRICE_HISTORY`,
`BELOW_MIN_MARKET_CAP`, `BELOW_MIN_ADV`, `STALE_FUNDAMENTALS`,
`MISSING_REQUIRED_FIELD`, `MANUALLY_EXCLUDED`.

## `FactorExposureRow`

A point-in-time *characteristic* snapshot: how much of a factor a subject
has, as of a date. **Never a return or P&L figure** — see
`FactorAttributionRow` for that.

| Field | Type | Notes |
|---|---|---|
| `subject_type` | `ExposureSubject` | `STOCK`, `SIGNAL`, or `PORTFOLIO` — the same schema serves all three since they're the same *kind* of measurement; only *who* it's measured for differs. |
| `subject_id` | `str` | A `security_id`, `signal_name`, or `portfolio_id`, depending on `subject_type`. |
| `as_of` | `Timestamp` | |
| `factor_name` | `str` | |
| `exposure_value` | `float` | |
| `source_known_as_of` | `Timestamp` | When the underlying inputs this exposure was computed from were themselves knowable — propagates the PIT chain, rather than letting it end at this row. |

## `FactorAttributionRow`

A derived P&L *contribution* over a period — structurally separate from
`FactorExposureRow`, and never produced without both a real exposure and a
real realized factor return as inputs. Answers a different question than
exposure does: exposure is "how much tilt does this subject have right
now"; attribution is "how much did that tilt earn over this period."

| Field | Type | Notes |
|---|---|---|
| `subject_type` | `ExposureSubject` | |
| `subject_id` | `str` | |
| `period_start` / `period_end` | `Timestamp` | `period_end` must be strictly after `period_start` — enforced by `__post_init__`. |
| `factor_name` | `str` | |
| `exposure_at_period_start` | `float` | Pulled from a real `FactorExposureRow` at `period_start`. |
| `factor_return_over_period` | `float` | A real realized factor return (a later-stage factor-returns series — Stage 1 only defines the schema; computing this series is out of scope here). |
| `contribution_to_return` | `float` | `exposure_at_period_start * factor_return_over_period` — a **linear approximation**, documented explicitly as such, not an exact P&L decomposition. Later stages may refine this. |
| `is_residual` | `bool` | `True` for the stock-specific/unexplained row, when included. |

## Synthetic dataset outputs (`data/synthetic.py`)

Three tables produced by `generate_synthetic_dataset`, always labeled
synthetic/non-proprietary, never consumed as if they were real market data:

- `market` — flattens to `MarketObservation` rows via `market_observations_from_frame`.
- `fundamentals` — flattens to `FundamentalObservation` rows via `fundamental_observations_from_frame`.
- `ground_truth` — `momentum_score`/`value_score`: the cross-sectionally
  z-scored latent factors actually used to construct `realized_return`.
  Consumed only by tests/notebooks checking recoverability — **never** by
  any production pipeline code, keeping the generator's own "answer key"
  isolated from anything resembling a real-data adapter.
