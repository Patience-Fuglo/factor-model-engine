"""Quality: a two-pillar composite of real profitability and real
financial-strength (leverage/safety) measures, built from the fields
``FundamentalObservation`` actually carries. Each pillar is its own
simple, auditable ratio -- no single "quality score" formula hides how
either half was computed.
"""

from __future__ import annotations

import pandas as pd

from ..config import QualityConfig


def compute_profitability(earnings_per_share: pd.Series, book_value_per_share: pd.Series) -> pd.Series:
    """A real ROE proxy: EPS / BVPS. The real share count cancels out of
    this ratio as long as both figures come from the same filing (the
    same real share count was used to compute both), so this needs no
    separate shares-outstanding input. NaN when BVPS <= 0 (a negative or
    zero book equity makes ROE economically meaningless, not just
    numerically awkward) or either real input is missing.
    """
    aligned = pd.concat([earnings_per_share.rename("eps"), book_value_per_share.rename("bvps")], axis=1)
    ratio = aligned["eps"] / aligned["bvps"].where(aligned["bvps"] > 0)
    return ratio.rename("profitability")


def compute_leverage(total_debt: pd.Series, total_assets: pd.Series) -> pd.Series:
    """Real leverage = total_debt / total_assets -- higher is LESS safe,
    combined with a sign flip in ``compute_quality_composite``. NaN when
    total_assets <= 0 (meaningless denominator) or total_debt < 0 (an
    invalid real observation, never silently clipped to zero).
    """
    aligned = pd.concat([total_debt.rename("debt"), total_assets.rename("assets")], axis=1)
    valid = (aligned["assets"] > 0) & (aligned["debt"] >= 0)
    ratio = aligned["debt"] / aligned["assets"].where(valid)
    return ratio.rename("leverage")


def compute_quality_composite(
    profitability_standardized: pd.Series, leverage_standardized: pd.Series, cfg: QualityConfig
) -> pd.Series:
    """Combine two ALREADY cross-sectionally standardized pillars into one
    composite Quality score. Both inputs must be standardized before
    reaching this function -- combining raw, differently-scaled ratios
    directly would let whichever has the larger raw scale dominate the
    composite, which is exactly the kind of silent, unintentional
    weighting this two-step design (standardize pillars, then combine)
    is meant to prevent.

    Safety = -leverage_standardized (lower leverage is higher quality).
    When one pillar is real but missing for a given security, the
    composite falls back to the other pillar alone rather than becoming
    NaN outright -- a security with real profitability data but no real
    balance-sheet data still gets a real (partial) Quality reading,
    weighted by the OTHER pillar's weight only.
    """
    safety_standardized = -leverage_standardized
    weighted = pd.concat(
        [
            profitability_standardized * cfg.profitability_weight,
            safety_standardized * cfg.leverage_weight,
        ],
        axis=1,
        keys=["profitability", "safety"],
    )
    weight_present = pd.concat(
        [
            profitability_standardized.notna() * cfg.profitability_weight,
            leverage_standardized.notna() * cfg.leverage_weight,
        ],
        axis=1,
        keys=["profitability", "safety"],
    )
    total_weight = weight_present.sum(axis=1)
    composite = weighted.sum(axis=1, skipna=True) / total_weight.where(total_weight > 0)
    return composite.rename("quality")
