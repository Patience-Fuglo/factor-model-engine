"""Momentum: a real, configurable price-based factor. Lookback and skip
window come from config, never hardcoded, per the standing
research-owner-control principle -- 12-1 is this build's recommended
default, not a universal definition baked into the function.
"""

from __future__ import annotations

import pandas as pd

from ..config import MomentumConfig


def compute_momentum(prices: pd.Series, cfg: MomentumConfig, periods_per_month: int = 1) -> pd.Series:
    """Real trailing momentum: the return from ``t - lookback_months`` to
    ``t - skip_recent_months``, skipping the most recent
    ``skip_recent_months`` specifically to avoid short-term reversal
    contaminating the signal (the real, documented reason behind the
    classic 12-1 convention, not an arbitrary choice).

    ``prices`` is a single real security's price series, indexed by
    date, at ``periods_per_month`` rows per month (``1`` for monthly
    data). Returns NaN wherever there isn't enough real history yet --
    never a fabricated early value.
    """
    lookback_periods = cfg.lookback_months * periods_per_month
    skip_periods = cfg.skip_recent_months * periods_per_month
    far_price = prices.shift(lookback_periods)
    near_price = prices.shift(skip_periods)
    momentum = near_price / far_price - 1.0
    return momentum.rename(f"momentum_{cfg.lookback_months}_{cfg.skip_recent_months}")
