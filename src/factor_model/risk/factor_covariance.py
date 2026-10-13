"""Stage 6 -- real factor covariance estimation from Stage 5's realized
factor-return time series. A plain sample covariance (``pandas.cov``)
over the real factor-return panel -- no shrinkage/regularization yet
(a later-stage refinement); honestly flagged as unstable when too few
real periods are available relative to the number of factors, rather
than silently trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from ..config import RiskConfig
from ..data.schemas import FactorReturnRow


@dataclass
class FactorCovarianceResult:
    covariance: pd.DataFrame
    n_periods: int
    warnings: list[str] = field(default_factory=list)


def compute_factor_covariance(rows: Sequence[FactorReturnRow], risk_cfg: RiskConfig) -> FactorCovarianceResult:
    """A real sample covariance matrix over Stage 5's factor-return
    panel (period_end x factor_name -> factor_return, pivoted first).
    Warns, rather than silently proceeding, when ``n_periods`` is below
    ``risk_cfg.min_periods_for_covariance`` -- a real, well-known small-
    sample instability in covariance estimation (too few periods
    relative to the number of factors produces a noisy, sometimes
    near-singular matrix).
    """
    if not rows:
        return FactorCovarianceResult(covariance=pd.DataFrame(), n_periods=0, warnings=["no real factor-return rows supplied"])

    frame = pd.DataFrame(
        [{"period_end": row.period_end, "factor_name": row.factor_name, "factor_return": row.factor_return} for row in rows]
    )
    panel = frame.pivot(index="period_end", columns="factor_name", values="factor_return")
    n_periods = len(panel)
    covariance = panel.cov()

    warnings: list[str] = []
    if n_periods < risk_cfg.min_periods_for_covariance:
        warnings.append(
            f"only {n_periods} real periods available (< {risk_cfg.min_periods_for_covariance} configured minimum) "
            "-- this factor covariance matrix is a real but statistically unstable small-sample estimate"
        )
    return FactorCovarianceResult(covariance=covariance, n_periods=n_periods, warnings=warnings)
