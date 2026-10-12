"""Core data schemas.

Five structurally separate concepts live here, and are never merged into
one table:

1. ``MarketObservation`` -- a security's price/volume state, known
   same-day (or near it).
2. ``FundamentalObservation`` -- a security's fundamental state for one
   fiscal period, known only after a real publication lag -- a
   genuinely different, much longer availability delay than market data,
   so it carries its own ``known_as_of``, never the market side's.
3. ``UniverseInclusion`` / ``UniversePITSnapshot`` -- point-in-time
   universe membership, with the reason for every inclusion/exclusion
   recorded explicitly.
4. ``FactorExposureRow`` -- a point-in-time *characteristic* snapshot:
   how much of a factor a subject (a stock, a signal, or a portfolio)
   has, as of a date. Never a return or P&L figure.
5. ``FactorAttributionRow`` -- a derived P&L *contribution* over a
   period: how much a factor's realized return, combined with a
   subject's exposure at the start of that period, contributed to what
   the subject actually earned. Structurally separate from #4 -- these
   answer different questions ("how much exposure do we have" vs. "how
   much did that exposure earn us") and must never be confused, stored
   in the same table, or produced without both a real exposure and a
   real realized factor return as inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Sector(str, Enum):
    TECHNOLOGY = "technology"
    FINANCIALS = "financials"
    HEALTHCARE = "healthcare"
    ENERGY = "energy"
    INDUSTRIALS = "industrials"


class ExclusionReason(str, Enum):
    NOT_YET_LISTED = "not_yet_listed"
    DELISTED = "delisted"
    INSUFFICIENT_PRICE_HISTORY = "insufficient_price_history"
    BELOW_MIN_MARKET_CAP = "below_min_market_cap"
    BELOW_MIN_ADV = "below_min_adv"
    STALE_FUNDAMENTALS = "stale_fundamentals"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    MANUALLY_EXCLUDED = "manually_excluded"


class FactorName(str, Enum):
    """Canonical factor names for Stage 2's factor library. Kept as a
    plain string on ``FactorExposureRow.factor_name`` (not this enum
    directly), so later stages can introduce composite/custom factor
    names without a schema change -- this enum is the recommended,
    type-checked vocabulary for Stage 2's own four factors only."""

    MOMENTUM = "momentum"
    VALUE = "value"
    SIZE = "size"
    QUALITY = "quality"


class ExposureSubject(str, Enum):
    """Who/what a FactorExposureRow or FactorAttributionRow is measured
    for -- the same three subject kinds apply to both, but exposure and
    attribution themselves remain two separate schemas regardless."""

    STOCK = "stock"
    SIGNAL = "signal"
    PORTFOLIO = "portfolio"


@dataclass(frozen=True)
class MarketObservation:
    """A security's real price/volume state on one trading date.

    ``known_as_of`` is this record's OWN availability timestamp --
    distinct from, and never to be confused with, any
    ``FundamentalObservation.known_as_of`` on the same security and
    date. For real market data this is almost always ``== as_of``
    (same-day/near-instant); it is kept as an explicit field anyway
    rather than silently assumed, so a feed with real reporting delay
    (e.g. an illiquid OTC name) can still be modeled honestly.
    """

    security_id: str
    ticker: str
    as_of: pd.Timestamp
    known_as_of: pd.Timestamp
    sector: Sector
    is_active: bool
    price: float
    shares_outstanding: float
    market_cap: float
    adv_20d: float | None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.known_as_of < self.as_of:
            raise ValueError(
                f"{self.security_id}: known_as_of ({self.known_as_of}) cannot "
                f"precede as_of ({self.as_of}) -- look-ahead"
            )


@dataclass(frozen=True)
class FundamentalObservation:
    """One real vintage of a security's fundamentals for one fiscal
    period.

    ``known_as_of`` is when THIS vintage became knowable -- a real
    publication/filing timestamp when a real adapter has one. The
    45-day default used by the synthetic generator (see
    ``synthetic.py``) is a fallback only, never a universal rule for
    real data; a real adapter lacking a genuine timestamp should mark
    the estimate via ``quality_flags`` (e.g. ``"estimated_known_as_of"``)
    rather than present a guess as an observed fact.

    A single ``(security_id, fundamental_period_end)`` pair can have
    MULTIPLE real vintages over time (restatements) -- that is expected,
    not an error; ``point_in_time.latest_known_fundamental`` is what
    resolves "which vintage would a researcher at a given as_of have
    actually seen."
    """

    security_id: str
    fundamental_period_end: pd.Timestamp
    known_as_of: pd.Timestamp
    book_value_per_share: float | None
    earnings_per_share: float | None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    # Stage 2 addition, both optional and defaulted (None) for backward
    # compatibility -- real balance-sheet figures from the SAME filing/
    # vintage as book_value_per_share/earnings_per_share above, so they
    # share this vintage's one known_as_of rather than needing their own:
    # a single real 10-Q/10-K reports its whole balance sheet and income
    # statement together, at one real publication timestamp. A genuinely
    # different fundamentals feed with its own cadence (e.g. a separate
    # analyst-estimate source) would need its own FundamentalObservation-
    # like type with its own known_as_of -- never bolted on here.
    total_assets: float | None = None
    total_debt: float | None = None

    def __post_init__(self) -> None:
        if self.known_as_of < self.fundamental_period_end:
            raise ValueError(
                f"{self.security_id}: known_as_of ({self.known_as_of}) cannot "
                f"precede fundamental_period_end ({self.fundamental_period_end}) -- look-ahead"
            )


@dataclass(frozen=True)
class UniverseInclusion:
    security_id: str
    included: bool
    reasons: tuple[ExclusionReason, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.included and self.reasons:
            raise ValueError(f"{self.security_id}: included securities cannot carry exclusion reasons")
        if not self.included and not self.reasons:
            raise ValueError(f"{self.security_id}: excluded securities must carry at least one reason")


@dataclass(frozen=True)
class UniversePITSnapshot:
    as_of: pd.Timestamp
    members: tuple[UniverseInclusion, ...]
    config_hash: str

    def included_ids(self) -> list[str]:
        return [m.security_id for m in self.members if m.included]

    def excluded_ids(self) -> list[str]:
        return [m.security_id for m in self.members if not m.included]


@dataclass(frozen=True)
class FactorExposureRow:
    """A point-in-time characteristic: how much of ``factor_name`` does
    ``subject_id`` (a stock, signal, or portfolio, per ``subject_type``)
    have, as of ``as_of``. Never a return or P&L figure -- see
    ``FactorAttributionRow`` for that.
    """

    subject_type: ExposureSubject
    subject_id: str
    as_of: pd.Timestamp
    factor_name: str
    exposure_value: float
    source_known_as_of: pd.Timestamp
    # Stage 2 addition, both defaulted for backward compatibility.
    # ``raw_value`` is the factor's real pre-standardization value (e.g. a
    # real book-to-price ratio); ``exposure_value`` above remains the
    # cross-sectionally STANDARDIZED score actually used downstream.
    # ``quality_flags`` carries this specific row's own data-quality
    # state (e.g. "missing_raw_value") -- distinct from, and not a
    # substitute for, the flags already carried on the raw
    # MarketObservation/FundamentalObservation this row was derived from.
    raw_value: float | None = None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FactorAttributionRow:
    """A derived P&L contribution: how much did ``factor_name`` contribute
    to what ``subject_id`` actually earned over [period_start,
    period_end], given its real exposure at period_start and the
    factor's real realized return over that period.

    ``contribution_to_return`` is computed as a LINEAR approximation
    (``exposure_at_period_start * factor_return_over_period``), documented
    explicitly as such -- not an exact decomposition of realized P&L.
    Later stages may refine this; Stage 1 only defines the schema, since
    computing a real factor-return time series is out of scope here.
    """

    subject_type: ExposureSubject
    subject_id: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    factor_name: str
    exposure_at_period_start: float
    factor_return_over_period: float
    contribution_to_return: float
    is_residual: bool = False

    def __post_init__(self) -> None:
        if self.period_end <= self.period_start:
            raise ValueError(
                f"{self.subject_id}/{self.factor_name}: period_end must be after period_start"
            )
