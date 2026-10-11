"""Data-health diagnostics: coverage, missingness, staleness,
duplicates, and outliers, computed BEFORE any factor result is trusted.
Every check here warns -- it never silently fixes or drops real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from ..config import DataConfig, DiagnosticsConfig
from ..data.point_in_time import as_of_view, latest_known_fundamental
from ..data.schemas import FundamentalObservation, MarketObservation


@dataclass
class DataHealthReport:
    as_of: pd.Timestamp
    n_securities_observed: int
    n_duplicate_market_rows: int
    n_duplicate_fundamental_vintages: int
    missingness_pct: dict[str, float] = field(default_factory=dict)
    stale_fundamental_security_ids: list[str] = field(default_factory=list)
    restated_fundamental_security_ids: list[str] = field(default_factory=list)
    outlier_security_ids: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "as_of": str(self.as_of.date()),
            "n_securities_observed": self.n_securities_observed,
            "n_duplicate_market_rows": self.n_duplicate_market_rows,
            "n_duplicate_fundamental_vintages": self.n_duplicate_fundamental_vintages,
            "missingness_pct": self.missingness_pct,
            "stale_fundamental_security_ids": self.stale_fundamental_security_ids,
            "restated_fundamental_security_ids": self.restated_fundamental_security_ids,
            "outlier_security_ids": self.outlier_security_ids,
            "warnings": self.warnings,
        }


def _zscore_outliers(values: dict[str, float], threshold: float = 4.0) -> list[str]:
    """Real, documented limitation: z-score outlier detection has a
    breakdown point at small n -- a single extreme value inflates its
    own mean and std, so one outlier among n points can never clear
    ``|z| > sqrt(n-1)``. At n=5 that cap is 2.0, well under this
    function's default threshold of 4.0, so a single wild value will
    NOT be flagged in a small cross-section no matter how extreme it
    is. This is expected behavior, not a bug -- it needs a realistically
    sized cross-section (the same real population diagnostics.
    min_cross_section_size already gates on) to have any real power.
    """
    clean = {k: v for k, v in values.items() if v is not None}
    if len(clean) < 3:
        return []
    series = pd.Series(clean)
    std = series.std(ddof=1)
    if std == 0:
        return []
    z = (series - series.mean()) / std
    return sorted(z[z.abs() > threshold].index.tolist())


def run_data_health_report(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    as_of: pd.Timestamp,
    data_cfg: DataConfig,
    diagnostics_cfg: DiagnosticsConfig,
) -> DataHealthReport:
    """Run every real diagnostic check against what's knowable at
    ``as_of`` -- never against the full dataset as if it were all
    already known.
    """
    knowable_market = as_of_view(market_observations, as_of)
    knowable_fundamentals = as_of_view(fundamental_observations, as_of)

    latest_market: dict[str, MarketObservation] = {}
    market_rows_per_security: dict[str, int] = {}
    for obs in knowable_market:
        if obs.as_of > as_of:
            continue
        market_rows_per_security[obs.security_id] = market_rows_per_security.get(obs.security_id, 0) + 1
        current = latest_market.get(obs.security_id)
        if current is None or obs.as_of > current.as_of:
            latest_market[obs.security_id] = obs

    # real duplicate check: more than one real market row for the same (security_id, as_of)
    market_key_counts: dict[tuple[str, pd.Timestamp], int] = {}
    for obs in knowable_market:
        key = (obs.security_id, obs.as_of)
        market_key_counts[key] = market_key_counts.get(key, 0) + 1
    n_duplicate_market_rows = sum(count - 1 for count in market_key_counts.values() if count > 1)

    # real duplicate check for fundamentals: identical (security_id, period_end, known_as_of)
    # is a true duplicate; different known_as_of values for the same period_end are a real,
    # legitimate restatement, never a duplicate. Uses only knowable_fundamentals -- a vintage
    # not yet knowable at as_of must never surface here, or this report would leak a future
    # restatement into a point-in-time diagnostic, exactly the bug this report exists to catch.
    fundamental_key_counts: dict[tuple[str, pd.Timestamp, pd.Timestamp], int] = {}
    period_end_vintage_count: dict[tuple[str, pd.Timestamp], set[pd.Timestamp]] = {}
    for obs in knowable_fundamentals:
        key = (obs.security_id, obs.fundamental_period_end, obs.known_as_of)
        fundamental_key_counts[key] = fundamental_key_counts.get(key, 0) + 1
        period_end_vintage_count.setdefault((obs.security_id, obs.fundamental_period_end), set()).add(obs.known_as_of)
    n_duplicate_fundamental_vintages = sum(count - 1 for count in fundamental_key_counts.values() if count > 1)

    restated_security_ids = sorted(
        {sid for (sid, _period_end), vintages in period_end_vintage_count.items() if len(vintages) > 1}
    )

    market_fields = ["price", "shares_outstanding", "market_cap", "adv_20d"]
    missingness_pct: dict[str, float] = {}
    n_securities = len(latest_market)
    for field_name in market_fields:
        if n_securities == 0:
            missingness_pct[field_name] = 0.0
            continue
        n_missing = sum(1 for obs in latest_market.values() if getattr(obs, field_name) is None)
        missingness_pct[field_name] = n_missing / n_securities

    fundamental_security_ids = {obs.security_id for obs in knowable_fundamentals}
    stale_ids: list[str] = []
    for security_id in fundamental_security_ids:
        period_ends = sorted({obs.fundamental_period_end for obs in knowable_fundamentals if obs.security_id == security_id})
        if not period_ends:
            continue
        latest_period_end = period_ends[-1]
        latest_vintage = latest_known_fundamental(fundamental_observations, security_id, latest_period_end, as_of)
        if latest_vintage is None:
            continue
        age_days = (as_of - latest_vintage.fundamental_period_end).days
        if age_days > data_cfg.stale_fundamentals_after_days:
            stale_ids.append(security_id)
    stale_ids = sorted(stale_ids)

    market_cap_by_security = {sid: obs.market_cap for sid, obs in latest_market.items()}
    adv_by_security = {sid: obs.adv_20d for sid, obs in latest_market.items()}
    outlier_security_ids = {
        "market_cap": _zscore_outliers(market_cap_by_security),
        "adv_20d": _zscore_outliers(adv_by_security),
    }

    warnings: list[str] = []
    if n_securities < diagnostics_cfg.min_cross_section_size:
        warnings.append(
            f"cross-section too small to trust: {n_securities} real securities knowable at "
            f"{as_of.date()}, below the configured minimum of {diagnostics_cfg.min_cross_section_size}"
        )
    for field_name, pct in missingness_pct.items():
        if pct > diagnostics_cfg.max_allowed_missingness_pct:
            warnings.append(
                f"{field_name} missingness ({pct:.1%}) exceeds the configured max of "
                f"{diagnostics_cfg.max_allowed_missingness_pct:.1%}"
            )
    if n_duplicate_market_rows > 0:
        warnings.append(f"{n_duplicate_market_rows} real duplicate market rows found for the same (security, date)")
    if n_duplicate_fundamental_vintages > 0:
        warnings.append(
            f"{n_duplicate_fundamental_vintages} real duplicate fundamental vintages found "
            "(identical security/period/known_as_of) -- distinct from legitimate restatements"
        )
    if stale_ids:
        warnings.append(f"{len(stale_ids)} real securities have stale fundamentals as of {as_of.date()}: {stale_ids}")
    if restated_security_ids:
        warnings.append(
            f"{len(restated_security_ids)} real securities have multiple fundamental vintages for the "
            f"same period (restatements) -- resolve via latest_known_fundamental, never 'most recent row': {restated_security_ids}"
        )
    for field_name, ids in outlier_security_ids.items():
        if ids:
            warnings.append(f"{field_name} outliers (|z|>4) flagged for review, not auto-corrected: {ids}")

    return DataHealthReport(
        as_of=as_of,
        n_securities_observed=n_securities,
        n_duplicate_market_rows=n_duplicate_market_rows,
        n_duplicate_fundamental_vintages=n_duplicate_fundamental_vintages,
        missingness_pct=missingness_pct,
        stale_fundamental_security_ids=stale_ids,
        restated_fundamental_security_ids=restated_security_ids,
        outlier_security_ids=outlier_security_ids,
        warnings=warnings,
    )
