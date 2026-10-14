"""Stage 9 -- the end-to-end orchestration this repo's own layout
reserved from Stage 1: runs Stages 2-8 once, in dependency order, for
one real ``as_of``/portfolio, and returns everything
``reporting/report.py`` needs to render a single cohesive report. Each
step reuses the exact library function that stage already built and
tested -- nothing here re-derives any real computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from ..attribution import AttributionReport, compute_attribution
from ..config import EngineConfig
from ..data.schemas import (
    ExposureSubject,
    FactorExposureRow,
    FundamentalObservation,
    MarketObservation,
    PortfolioHolding,
    UniversePITSnapshot,
)
from ..data.universe import build_point_in_time_universe
from ..exposures import aggregate_exposure, equal_weight_benchmark_holdings
from ..factors import FactorLibraryReport, compute_factor_exposure_table
from ..monitoring import FactorStabilityResult, compute_factor_stability
from ..regimes import classify_volatility_regime
from ..regression import _period_realized_returns, compute_factor_returns
from ..risk import FactorCovarianceResult, PortfolioRiskResult, compute_factor_covariance, compute_portfolio_risk, compute_specific_returns, compute_specific_variance


@dataclass
class PipelineResult:
    as_of: pd.Timestamp
    period_start: pd.Timestamp
    portfolio_id: str
    universe: UniversePITSnapshot
    stock_exposure_rows: list[FactorExposureRow]
    factor_library_report: FactorLibraryReport
    portfolio_exposure_rows: list[FactorExposureRow]
    factor_covariance: FactorCovarianceResult
    portfolio_risk: PortfolioRiskResult
    attribution: AttributionReport
    regimes_by_factor: dict[str, pd.DataFrame] = field(default_factory=dict)
    stability_by_factor: dict[str, FactorStabilityResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def run_full_pipeline(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    as_of: pd.Timestamp,
    period_dates: Sequence[pd.Timestamp],
    engine_cfg: EngineConfig,
    portfolio_id: str = "benchmark_equal_weight",
    holdings: Sequence[PortfolioHolding] | None = None,
) -> PipelineResult:
    """``period_dates`` is the full real monthly grid used to estimate
    factor returns/covariance/stability (Stages 5, 6, 8); ``as_of`` must
    be its last element -- the portfolio snapshot and attribution's
    final period both anchor there.
    """
    sorted_dates = sorted(period_dates)
    if sorted_dates[-1] != as_of:
        raise ValueError("as_of must be the last date in period_dates")

    warnings: list[str] = []

    universe = build_point_in_time_universe(market_observations, as_of, engine_cfg.universe, engine_cfg.data)
    stock_exposure_rows, factor_library_report = compute_factor_exposure_table(
        market_observations, fundamental_observations, as_of, universe, engine_cfg
    )

    if holdings is None:
        holdings = equal_weight_benchmark_holdings(universe.included_ids(), as_of, portfolio_id=portfolio_id)
    portfolio_exposure_rows = aggregate_exposure(stock_exposure_rows, holdings, ExposureSubject.PORTFOLIO, portfolio_id)

    return_rows, return_warnings = compute_factor_returns(market_observations, fundamental_observations, sorted_dates, engine_cfg)
    warnings.extend(return_warnings)
    factor_covariance = compute_factor_covariance(return_rows, engine_cfg.risk)
    warnings.extend(factor_covariance.warnings)

    specific_returns = compute_specific_returns(market_observations, fundamental_observations, sorted_dates, engine_cfg)
    specific_variance = compute_specific_variance(specific_returns)
    portfolio_risk = compute_portfolio_risk(
        portfolio_exposure_rows, factor_covariance.covariance, holdings, specific_variance, portfolio_id, engine_cfg.risk
    )
    warnings.extend(portfolio_risk.warnings)

    # the portfolio's own real realized return over the final period,
    # for attribution -- the same weighted-average construction
    # aggregate_exposure already uses for exposures, applied here to returns
    t_prev = sorted_dates[-2]
    weights_by_security = {h.security_id: h.weight for h in holdings if h.portfolio_id == portfolio_id}
    stock_returns = _period_realized_returns(market_observations, list(weights_by_security), t_prev, as_of)
    total_weight = sum(weights_by_security[sid] for sid in stock_returns.index)
    portfolio_realized_return = (
        float(sum(weights_by_security[sid] * stock_returns[sid] for sid in stock_returns.index) / total_weight)
        if total_weight
        else float("nan")
    )

    exposure_at_period_start, _ = compute_factor_exposure_table(
        market_observations, fundamental_observations, t_prev, universe, engine_cfg
    )
    portfolio_exposure_at_t_prev = aggregate_exposure(exposure_at_period_start, holdings, ExposureSubject.PORTFOLIO, portfolio_id)
    attribution = compute_attribution(
        portfolio_exposure_at_t_prev, return_rows, portfolio_realized_return, ExposureSubject.PORTFOLIO, portfolio_id, t_prev, as_of
    )
    warnings.extend(attribution.warnings)

    factor_names = sorted({row.factor_name for row in return_rows})
    regimes_by_factor = {name: classify_volatility_regime(return_rows, name, engine_cfg.regimes) for name in factor_names}
    stability_by_factor = compute_factor_stability(market_observations, fundamental_observations, sorted_dates, engine_cfg)
    for result in stability_by_factor.values():
        warnings.extend(result.warnings)

    return PipelineResult(
        as_of=as_of,
        period_start=sorted_dates[0],
        portfolio_id=portfolio_id,
        universe=universe,
        stock_exposure_rows=stock_exposure_rows,
        factor_library_report=factor_library_report,
        portfolio_exposure_rows=portfolio_exposure_rows,
        factor_covariance=factor_covariance,
        portfolio_risk=portfolio_risk,
        attribution=attribution,
        regimes_by_factor=regimes_by_factor,
        stability_by_factor=stability_by_factor,
        warnings=warnings,
    )
