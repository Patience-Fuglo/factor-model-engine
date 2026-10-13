"""Stage 5 -- Factor Returns: a real Fama-MacBeth-style cross-sectional
regression. For each consecutive pair of real period dates, regresses
REALIZED stock returns over that period on each stock's real factor
exposure AS OF the period's START (never its end) -- the lag is the
whole point: a factor return answers "did last period's exposure
predict this period's return," not "does this period's exposure
correlate with this period's own return" (which the exposure itself
would trivially satisfy for Momentum/Value by construction).

Reuses Stage 2's ``compute_factor_exposure_table`` (for the lagged
exposures) and ``alpha-validation-toolkit``'s real ``ols_regression``
(for the cross-sectional fit) -- no regression mechanics are
reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from quant_toolkit.metrics import OLSResult, ols_regression

from ..config import EngineConfig
from ..data.point_in_time import latest_known_market
from ..data.schemas import FactorReturnRow, FundamentalObservation, MarketObservation
from ..data.universe import build_point_in_time_universe
from ..factors.factor_library import compute_factor_exposure_table, factor_exposure_rows_to_frame


def _period_realized_returns(
    market_observations: Sequence[MarketObservation], security_ids: Sequence[str], t_prev: pd.Timestamp, t_curr: pd.Timestamp
) -> pd.Series:
    """Real realized return from ``t_prev`` to ``t_curr`` per security,
    using the real, actually-observed price at each date (via
    ``latest_known_market``, which also guards against a missing
    observation at either end). Securities with no real price at either
    date, or a non-positive starting price, are excluded -- never
    silently zero-filled.
    """
    returns: dict[str, float] = {}
    for security_id in security_ids:
        obs_prev = latest_known_market(market_observations, security_id, t_prev)
        obs_curr = latest_known_market(market_observations, security_id, t_curr)
        if obs_prev is None or obs_curr is None or obs_prev.price <= 0:
            continue
        returns[security_id] = obs_curr.price / obs_prev.price - 1.0
    return pd.Series(returns)


@dataclass
class PeriodFit:
    """One period's real lagged cross-sectional fit -- exposed (not
    private) so Stage 6's specific-risk computation can reuse the exact
    same lagged exposures/realized returns/OLS fit that produced each
    period's factor returns, rather than re-deriving a parallel version
    of the same real regression and risking the two silently drifting
    apart.
    """

    period_start: pd.Timestamp
    period_end: pd.Timestamp
    exposure_pivot: pd.DataFrame  # usable (non-all-NaN) columns only
    realized_returns: pd.Series
    ols_result: OLSResult | None
    warning: str | None


def _fit_one_period(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    t_prev: pd.Timestamp,
    t_curr: pd.Timestamp,
    engine_cfg: EngineConfig,
) -> PeriodFit:
    universe = build_point_in_time_universe(market_observations, t_prev, engine_cfg.universe, engine_cfg.data)
    exposure_rows, _ = compute_factor_exposure_table(
        market_observations, fundamental_observations, t_prev, universe, engine_cfg
    )
    exposure_pivot = factor_exposure_rows_to_frame(exposure_rows).pivot(
        index="security_id", columns="factor_name", values="standardized_score"
    )
    # a factor with NO real reading for ANY security this period (e.g.
    # momentum, before 12 real months of history exist yet) can't be
    # estimated this period at all -- drop it from THIS period's joint
    # regression rather than letting one all-NaN column wipe out every
    # row via ols_regression's own dropna, which would silently kill
    # the whole period's factor-return estimate for every factor, not
    # just the genuinely unestimable one
    usable_columns = [c for c in exposure_pivot.columns if exposure_pivot[c].notna().any()]
    dropped_columns = [c for c in exposure_pivot.columns if c not in usable_columns]
    exposure_pivot = exposure_pivot[usable_columns]
    realized_returns = _period_realized_returns(market_observations, universe.included_ids(), t_prev, t_curr)

    warning = None
    if dropped_columns:
        warning = (
            f"{t_prev.date()} -> {t_curr.date()}: no real exposure reading for {dropped_columns} "
            "this period -- excluded from this period's regression, not estimated"
        )
    if exposure_pivot.empty:
        return PeriodFit(t_prev, t_curr, exposure_pivot, realized_returns, None,
                          warning or f"{t_prev.date()} -> {t_curr.date()}: skipped -- no real factor had any usable exposure")
    try:
        result = ols_regression(realized_returns, exposure_pivot)
    except ValueError as exc:
        return PeriodFit(t_prev, t_curr, exposure_pivot, realized_returns, None, f"{t_prev.date()} -> {t_curr.date()}: skipped -- {exc}")
    return PeriodFit(t_prev, t_curr, exposure_pivot, realized_returns, result, warning)


def compute_factor_returns(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    period_dates: Sequence[pd.Timestamp],
    engine_cfg: EngineConfig,
) -> tuple[list[FactorReturnRow], list[str]]:
    """Build the real factor-return time series across every consecutive
    pair in ``period_dates`` (sorted first). For each pair
    ``(t_prev, t_curr)``:

    1. Build the real point-in-time universe and factor exposure table
       AS OF ``t_prev`` (Stage 2/Stage 1, unchanged) -- these are the
       LAGGED exposures.
    2. Compute each eligible security's real realized return from
       ``t_prev`` to ``t_curr``.
    3. Regress realized returns on the lagged, standardized exposures
       (real cross-sectional OLS, aligned by security_id).

    A period with too few real aligned observations to fit the
    regression is skipped with a real warning, never silently dropped
    without explanation.
    """
    sorted_dates = sorted(period_dates)
    rows: list[FactorReturnRow] = []
    warnings: list[str] = []

    for t_prev, t_curr in zip(sorted_dates[:-1], sorted_dates[1:]):
        fit = _fit_one_period(market_observations, fundamental_observations, t_prev, t_curr, engine_cfg)
        if fit.warning:
            warnings.append(fit.warning)
        if fit.ols_result is None:
            continue
        for factor_name in fit.exposure_pivot.columns:
            rows.append(
                FactorReturnRow(
                    period_start=t_prev,
                    period_end=t_curr,
                    factor_name=factor_name,
                    factor_return=fit.ols_result.betas[factor_name],
                    r_squared=fit.ols_result.r_squared,
                    n_obs=fit.ols_result.n_obs,
                    known_as_of=t_curr,
                )
            )
    return rows, warnings


def factor_return_rows_to_frame(rows: Sequence[FactorReturnRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period_start": row.period_start,
                "period_end": row.period_end,
                "factor_name": row.factor_name,
                "factor_return": row.factor_return,
                "r_squared": row.r_squared,
                "n_obs": row.n_obs,
                "known_as_of": row.known_as_of,
            }
            for row in rows
        ]
    )
