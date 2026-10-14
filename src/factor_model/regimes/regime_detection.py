"""Stage 8 -- regime detection: classifies each real period into a
LOW/NORMAL/HIGH volatility regime for one factor's return series, using
a TRAILING rolling volatility and an EXPANDING (never full-sample)
percentile rank -- a regime label for period t must only reflect what
was knowable up to and including t, the same point-in-time discipline
this whole repo applies everywhere else. Ranking against the full
dataset's distribution (including future periods) would be a real
look-ahead bug in a regime classifier, not just a factor computation.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..config import RegimeConfig
from ..data.schemas import FactorReturnRow


def classify_volatility_regime(rows: Sequence[FactorReturnRow], factor_name: str, cfg: RegimeConfig) -> pd.DataFrame:
    """Real trailing rolling volatility of ``factor_name``'s realized
    return series, with each period's regime determined by its
    EXPANDING-window percentile rank among every real volatility reading
    knowable up to and including that period.

    Returns a DataFrame indexed by ``period_end`` with columns
    ``rolling_volatility``, ``percentile``, ``regime`` (one of "low",
    "normal", "high", or ``None`` before enough real history exists).
    """
    own_rows = [r for r in rows if r.factor_name == factor_name]
    if not own_rows:
        return pd.DataFrame(columns=["rolling_volatility", "percentile", "regime"])

    series = pd.Series(
        {r.period_end: r.factor_return for r in own_rows}
    ).sort_index()

    rolling_volatility = series.rolling(cfg.volatility_window, min_periods=cfg.volatility_window).std(ddof=1)
    # expanding rank: period t's percentile uses only volatility readings
    # known by t (itself included), never a later period's. min_periods=2
    # here (NOT cfg.volatility_window) -- Expanding.rank's min_periods
    # counts real NON-NULL volatility readings, which only start
    # appearing once the rolling std itself has enough history; reusing
    # the same strict threshold here would needlessly delay the first
    # real regime label by another full window's worth of periods for
    # no real statistical reason (a percentile among 2+ real readings is
    # already a meaningful, if coarse, relative rank -- it just gets more
    # reliable with more history, which is expected, not a defect)
    percentile = rolling_volatility.expanding(min_periods=2).rank(pct=True)

    def _label(p: float) -> str | None:
        if pd.isna(p):
            return None
        if p <= cfg.low_volatility_percentile:
            return "low"
        if p >= cfg.high_volatility_percentile:
            return "high"
        return "normal"

    regime = percentile.apply(_label)
    return pd.DataFrame({"rolling_volatility": rolling_volatility, "percentile": percentile, "regime": regime})
