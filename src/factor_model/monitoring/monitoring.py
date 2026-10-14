"""Stage 8 -- factor stability monitoring: a real period-by-period
cross-sectional Information Coefficient between each factor's LAGGED
exposure and the realized return it was meant to predict, reusing Stage
5/6's exact per-period fit (``regression._fit_one_period``) and
``alpha-validation-toolkit``'s real ``information_coefficient``/``icir``
-- IC math is never reimplemented here.

A real drift alert fires when a factor's RECENT IC has decayed sharply
(or flipped sign) relative to its own full history -- a concrete,
honest "is this factor's predictive power degrading" check, not a
vague dashboard metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
from quant_toolkit.metrics import icir, information_coefficient

from ..config import EngineConfig, MonitoringConfig
from ..data.schemas import FundamentalObservation, MarketObservation
from ..regression.factor_returns import _fit_one_period


@dataclass
class FactorStabilityResult:
    factor_name: str
    ic_series: pd.Series
    icir: float
    recent_ic_mean: float
    historical_ic_mean: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "icir": self.icir,
            "recent_ic_mean": self.recent_ic_mean,
            "historical_ic_mean": self.historical_ic_mean,
            "n_periods": int(self.ic_series.notna().sum()),
            "warnings": self.warnings,
        }


def compute_factor_stability(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    period_dates: Sequence[pd.Timestamp],
    engine_cfg: EngineConfig,
) -> dict[str, FactorStabilityResult]:
    """Per real factor present in the panel, a period-by-period
    cross-sectional IC (lagged exposure vs. realized return that
    period), using the SAME lagged exposures/realized returns Stage 5's
    factor-return regression is built from -- never a separately
    re-derived parallel computation.
    """
    sorted_dates = sorted(period_dates)
    ic_by_factor: dict[str, dict[pd.Timestamp, float]] = {}

    for t_prev, t_curr in zip(sorted_dates[:-1], sorted_dates[1:]):
        fit = _fit_one_period(market_observations, fundamental_observations, t_prev, t_curr, engine_cfg)
        if fit.ols_result is None:
            continue
        for factor_name in fit.exposure_pivot.columns:
            ic = information_coefficient(fit.exposure_pivot[factor_name], fit.realized_returns)
            ic_by_factor.setdefault(factor_name, {})[t_curr] = ic

    results: dict[str, FactorStabilityResult] = {}
    for factor_name, values in ic_by_factor.items():
        results[factor_name] = _build_stability_result(factor_name, pd.Series(values).sort_index(), engine_cfg.monitoring)
    return results


def _build_stability_result(factor_name: str, ic_series: pd.Series, monitoring_cfg: MonitoringConfig) -> FactorStabilityResult:
    clean = ic_series.dropna()
    historical_mean = float(clean.mean()) if len(clean) else float("nan")
    recent = clean.iloc[-monitoring_cfg.drift_recent_window :]
    recent_mean = float(recent.mean()) if len(recent) else float("nan")

    warnings: list[str] = []
    if len(clean) >= monitoring_cfg.drift_recent_window and pd.notna(historical_mean) and pd.notna(recent_mean):
        if historical_mean * recent_mean < 0:
            warnings.append(
                f"{factor_name}: recent IC mean ({recent_mean:.3f}) has flipped sign vs. "
                f"its historical mean ({historical_mean:.3f}) -- a real, material drift signal"
            )
        elif abs(historical_mean) > 1e-6 and abs(recent_mean) < monitoring_cfg.drift_decay_threshold * abs(historical_mean):
            warnings.append(
                f"{factor_name}: recent IC mean ({recent_mean:.3f}) has decayed below "
                f"{monitoring_cfg.drift_decay_threshold:.0%} of its historical mean ({historical_mean:.3f})"
            )

    return FactorStabilityResult(
        factor_name=factor_name,
        ic_series=ic_series,
        icir=icir(ic_series),
        recent_ic_mean=recent_mean,
        historical_ic_mean=historical_mean,
        warnings=warnings,
    )
