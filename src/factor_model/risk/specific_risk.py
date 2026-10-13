"""Stage 6 -- specific (idiosyncratic) risk: the real per-security
variance of each period's cross-sectional regression RESIDUAL -- the
part of a stock's own realized return its factor exposures couldn't
explain that period. Reuses Stage 5's exact per-period fit
(``regression._fit_one_period``) rather than re-deriving a parallel
version of the same regression.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..config import EngineConfig
from ..data.schemas import FundamentalObservation, MarketObservation
from ..regression.factor_returns import _fit_one_period


def compute_specific_returns(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    period_dates: Sequence[pd.Timestamp],
    engine_cfg: EngineConfig,
) -> pd.DataFrame:
    """A real security_id x period_end panel of residual (specific)
    returns: ``realized_return - fitted_factor_return``, using the SAME
    lagged exposures and OLS fit Stage 5's ``compute_factor_returns``
    produces for that period -- so the two are never allowed to
    silently drift apart into two different real regressions.
    """
    sorted_dates = sorted(period_dates)
    columns: dict[pd.Timestamp, pd.Series] = {}
    for t_prev, t_curr in zip(sorted_dates[:-1], sorted_dates[1:]):
        fit = _fit_one_period(market_observations, fundamental_observations, t_prev, t_curr, engine_cfg)
        if fit.ols_result is None:
            continue
        aligned = pd.concat([fit.realized_returns.rename("y"), fit.exposure_pivot], axis=1, join="inner").dropna()
        fitted = pd.Series(fit.ols_result.alpha, index=aligned.index)
        for factor_name in fit.exposure_pivot.columns:
            fitted = fitted + fit.ols_result.betas[factor_name] * aligned[factor_name]
        columns[t_curr] = aligned["y"] - fitted
    return pd.DataFrame(columns)


def compute_specific_variance(specific_returns: pd.DataFrame, min_periods: int = 2) -> pd.Series:
    """Each real security's own specific-return variance across time --
    the standard textbook "specific risk" input to a factor risk model.
    NaN (never a fabricated zero) for a security with fewer than
    ``min_periods`` real residual observations.
    """
    counts = specific_returns.notna().sum(axis=1)
    variance = specific_returns.var(axis=1, ddof=1)
    return variance.where(counts >= min_periods)
