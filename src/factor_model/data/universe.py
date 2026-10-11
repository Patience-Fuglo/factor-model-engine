"""Point-in-time universe construction: which real securities are
eligible as of a given date, with the reason for every real inclusion
or exclusion recorded explicitly -- never silently assumed, and never
built from today's constituents applied backward in time.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import pandas as pd

from ..config import DataConfig, UniverseConfig
from .point_in_time import as_of_view
from .schemas import ExclusionReason, MarketObservation, UniverseInclusion, UniversePITSnapshot


def _config_hash(universe_cfg: UniverseConfig, data_cfg: DataConfig) -> str:
    payload = f"{universe_cfg}|{data_cfg}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_point_in_time_universe(
    market_observations: Sequence[MarketObservation],
    as_of: pd.Timestamp,
    universe_cfg: UniverseConfig,
    data_cfg: DataConfig,
    price_history_days_by_security: dict[str, int] | None = None,
) -> UniversePITSnapshot:
    """Build the real universe snapshot as of ``as_of``, using only real
    observations knowable by then (``as_of_view``) -- never today's
    universe reconstructed backward. ``price_history_days_by_security``
    is supplied by the caller (a later stage's data layer) rather than
    computed here, so this function stays a pure, testable filter.
    """
    knowable = as_of_view(market_observations, as_of)

    latest_by_security: dict[str, MarketObservation] = {}
    for obs in knowable:
        if obs.as_of > as_of:
            continue
        current = latest_by_security.get(obs.security_id)
        if current is None or obs.as_of > current.as_of:
            latest_by_security[obs.security_id] = obs

    price_history_days_by_security = price_history_days_by_security or {}
    all_security_ids = {obs.security_id for obs in market_observations}

    members: list[UniverseInclusion] = []
    for security_id, obs in sorted(latest_by_security.items()):
        reasons: list[ExclusionReason] = []
        if not obs.is_active:
            reasons.append(ExclusionReason.DELISTED)
        if universe_cfg.excluded_sectors and obs.sector.value in universe_cfg.excluded_sectors:
            reasons.append(ExclusionReason.MANUALLY_EXCLUDED)
        if obs.market_cap < universe_cfg.min_market_cap:
            reasons.append(ExclusionReason.BELOW_MIN_MARKET_CAP)
        if obs.adv_20d is None or obs.adv_20d < universe_cfg.min_adv_usd:
            reasons.append(ExclusionReason.BELOW_MIN_ADV)
        history_days = price_history_days_by_security.get(security_id, data_cfg.min_price_history_days)
        if history_days < data_cfg.min_price_history_days:
            reasons.append(ExclusionReason.INSUFFICIENT_PRICE_HISTORY)

        members.append(UniverseInclusion(security_id=security_id, included=not reasons, reasons=tuple(reasons)))

    never_yet_knowable = all_security_ids - set(latest_by_security.keys())
    for security_id in never_yet_knowable:
        members.append(
            UniverseInclusion(security_id=security_id, included=False, reasons=(ExclusionReason.NOT_YET_LISTED,))
        )

    return UniversePITSnapshot(
        as_of=as_of,
        members=tuple(sorted(members, key=lambda m: m.security_id)),
        config_hash=_config_hash(universe_cfg, data_cfg),
    )
