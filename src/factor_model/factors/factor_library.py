"""Stage 2 -- Factor Library: the engine that turns point-in-time market
and fundamental observations into a real Factor Exposure Table (one row
per stock x date x factor), covering Momentum, Value, Size, and Quality.

Pipeline, explicit and auditable at every step: RAW (point-in-time, from
``data/``) -> VALIDATED (missing/invalid handled, never silently zeroed)
-> WINSORIZED -> STANDARDIZED (cross-sectional, over the real eligible
universe only) -> ``FactorExposureRow``.

Reuses Stage 1's ``compute_momentum``, ``compute_book_to_price``,
``winsorize``, and ``standardize`` rather than duplicating any of them --
Size and Quality are the only new raw-factor computations this module
introduces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from ..config import EngineConfig
from ..data.point_in_time import latest_known_fundamental_any_period, latest_known_market
from ..data.schemas import ExposureSubject, FactorExposureRow, FactorName, FundamentalObservation, MarketObservation, UniversePITSnapshot
from .fundamental_based import compute_book_to_price
from .price_based import compute_momentum
from .quality_based import compute_leverage, compute_profitability, compute_quality_composite
from .size_based import compute_size
from .transforms import standardize, winsorize


@dataclass
class FactorLibraryReport:
    """A concise data-quality report for one factor-table build -- how
    many of the real eligible universe got a real reading for each
    factor, and which securities didn't. Warns, same as
    ``diagnostics.DataHealthReport``; never silently drops a security
    from the output table because one factor was unavailable for it.
    """

    as_of: pd.Timestamp
    n_eligible: int
    n_with_reading: dict[str, int] = field(default_factory=dict)
    missing_security_ids: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "as_of": str(self.as_of.date()),
            "n_eligible": self.n_eligible,
            "n_with_reading": self.n_with_reading,
            "missing_security_ids": self.missing_security_ids,
            "warnings": self.warnings,
        }


def _price_series(market_observations: Sequence[MarketObservation], security_id: str, as_of: pd.Timestamp) -> pd.Series:
    """This security's own real, point-in-time price history up to
    ``as_of`` -- only observations that are both dated on/before
    ``as_of`` and already knowable by ``as_of``. O(n) per security per
    call; fine at this repo's research scale, not written for a
    production-size universe.
    """
    rows = [
        (obs.as_of, obs.price)
        for obs in market_observations
        if obs.security_id == security_id and obs.as_of <= as_of and obs.known_as_of <= as_of
    ]
    if not rows:
        return pd.Series(dtype=float)
    rows.sort(key=lambda r: r[0])
    dates, prices = zip(*rows)
    return pd.Series(prices, index=pd.DatetimeIndex(dates))


def compute_factor_exposure_table(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    as_of: pd.Timestamp,
    universe: UniversePITSnapshot,
    engine_cfg: EngineConfig,
) -> tuple[list[FactorExposureRow], FactorLibraryReport]:
    """Build the real, point-in-time Factor Exposure Table for every
    security in ``universe``'s real eligible membership, as of
    ``as_of``. Every raw/standardized value is computed using ONLY
    observations knowable by ``as_of`` -- no future information is ever
    read, for any factor.
    """
    security_ids = universe.included_ids()
    factor_cfg = engine_cfg.factors

    raw_momentum: dict[str, float] = {}
    raw_size: dict[str, float] = {}
    price_by_security: dict[str, float] = {}
    bvps_by_security: dict[str, float] = {}
    eps_by_security: dict[str, float] = {}
    assets_by_security: dict[str, float] = {}
    debt_by_security: dict[str, float] = {}
    momentum_source_known_as_of: dict[str, pd.Timestamp] = {}
    size_source_known_as_of: dict[str, pd.Timestamp] = {}
    fundamental_source_known_as_of: dict[str, pd.Timestamp] = {}

    for security_id in security_ids:
        price_series = _price_series(market_observations, security_id, as_of)
        if len(price_series) > 0:
            momentum_series = compute_momentum(price_series, factor_cfg.momentum)
            raw_momentum[security_id] = float(momentum_series.iloc[-1])
        else:
            raw_momentum[security_id] = float("nan")

        latest_market = latest_known_market(market_observations, security_id, as_of)
        if latest_market is not None:
            price_by_security[security_id] = latest_market.price
            raw_size[security_id] = float("nan") if latest_market.market_cap <= 0 else latest_market.market_cap
            momentum_source_known_as_of[security_id] = latest_market.known_as_of
            size_source_known_as_of[security_id] = latest_market.known_as_of
        else:
            price_by_security[security_id] = float("nan")
            raw_size[security_id] = float("nan")

        vintage = latest_known_fundamental_any_period(fundamental_observations, security_id, as_of)
        if vintage is not None:
            bvps_by_security[security_id] = vintage.book_value_per_share if vintage.book_value_per_share is not None else float("nan")
            eps_by_security[security_id] = vintage.earnings_per_share if vintage.earnings_per_share is not None else float("nan")
            assets_by_security[security_id] = vintage.total_assets if vintage.total_assets is not None else float("nan")
            debt_by_security[security_id] = vintage.total_debt if vintage.total_debt is not None else float("nan")
            fundamental_source_known_as_of[security_id] = vintage.known_as_of
        else:
            bvps_by_security[security_id] = float("nan")
            eps_by_security[security_id] = float("nan")
            assets_by_security[security_id] = float("nan")
            debt_by_security[security_id] = float("nan")

    index = pd.Index(security_ids, name="security_id")
    momentum_raw = pd.Series(raw_momentum).reindex(index)
    # Size's raw value IS log(market_cap) -- the log transform is the
    # factor's own definition, not a preprocessing step applied later
    market_cap_s = pd.Series(raw_size).reindex(index)
    size_raw = compute_size(market_cap_s)
    price_s = pd.Series(price_by_security).reindex(index)
    bvps_s = pd.Series(bvps_by_security).reindex(index)
    eps_s = pd.Series(eps_by_security).reindex(index)
    assets_s = pd.Series(assets_by_security).reindex(index)
    debt_s = pd.Series(debt_by_security).reindex(index)

    value_raw = compute_book_to_price(bvps_s, price_s)
    profitability_raw = compute_profitability(eps_s, bvps_s)
    leverage_raw = compute_leverage(debt_s, assets_s)

    def _winsorize_and_standardize(raw: pd.Series) -> pd.Series:
        return standardize(winsorize(raw, factor_cfg.winsorization), factor_cfg.standardization)

    momentum_std = _winsorize_and_standardize(momentum_raw)
    value_std = _winsorize_and_standardize(value_raw)
    size_std = _winsorize_and_standardize(size_raw)
    profitability_std = _winsorize_and_standardize(profitability_raw)
    leverage_std = _winsorize_and_standardize(leverage_raw)
    quality_std = compute_quality_composite(profitability_std, leverage_std, factor_cfg.quality)

    rows: list[FactorExposureRow] = []
    n_with_reading: dict[str, int] = {}
    missing_security_ids: dict[str, list[str]] = {}
    warnings: list[str] = []

    factor_specs = [
        (FactorName.MOMENTUM.value, momentum_raw, momentum_std, momentum_source_known_as_of),
        (FactorName.VALUE.value, value_raw, value_std, fundamental_source_known_as_of),
        (FactorName.SIZE.value, size_raw, size_std, size_source_known_as_of),
        # Quality is a composite with no single pre-standardization raw
        # figure -- its two real pillars (profitability, leverage) are
        # each already fully auditable via their own raw/standardized
        # series above; Stage 2 emits exactly the four factors named in
        # scope, not six, so those pillars are not separately exposed as
        # their own FactorExposureRow entries here.
        (FactorName.QUALITY.value, pd.Series(float("nan"), index=index), quality_std, fundamental_source_known_as_of),
    ]

    for factor_name, raw_series, std_series, known_as_of_by_security in factor_specs:
        n_present = 0
        missing: list[str] = []
        for security_id in security_ids:
            raw_value = raw_series.get(security_id, float("nan"))
            exposure_value = std_series.get(security_id, float("nan"))
            flags: list[str] = []
            if pd.isna(exposure_value):
                flags.append("missing_raw_value" if pd.isna(raw_value) else "standardization_unavailable")
                missing.append(security_id)
            else:
                n_present += 1
            source_known_as_of = known_as_of_by_security.get(security_id, as_of)
            rows.append(
                FactorExposureRow(
                    subject_type=ExposureSubject.STOCK,
                    subject_id=security_id,
                    as_of=as_of,
                    factor_name=factor_name,
                    exposure_value=float(exposure_value) if pd.notna(exposure_value) else float("nan"),
                    source_known_as_of=source_known_as_of,
                    raw_value=float(raw_value) if pd.notna(raw_value) else None,
                    quality_flags=tuple(flags),
                )
            )
        n_with_reading[factor_name] = n_present
        missing_security_ids[factor_name] = sorted(missing)
        if missing:
            warnings.append(f"{factor_name}: {len(missing)}/{len(security_ids)} eligible securities have no real reading")

    report = FactorLibraryReport(
        as_of=as_of,
        n_eligible=len(security_ids),
        n_with_reading=n_with_reading,
        missing_security_ids=missing_security_ids,
        warnings=warnings,
    )
    return rows, report


def factor_exposure_rows_to_frame(rows: Sequence[FactorExposureRow]) -> pd.DataFrame:
    """The Factor Exposure Table as a flat DataFrame: Stock, Date,
    Factor, Raw value, Standardized score, Quality flags -- exactly the
    shape the Stage 2 spec calls for.
    """
    return pd.DataFrame(
        [
            {
                "security_id": row.subject_id,
                "as_of": row.as_of,
                "factor_name": row.factor_name,
                "raw_value": row.raw_value,
                "standardized_score": row.exposure_value,
                "source_known_as_of": row.source_known_as_of,
                "quality_flags": ";".join(row.quality_flags),
            }
            for row in rows
        ]
    )
