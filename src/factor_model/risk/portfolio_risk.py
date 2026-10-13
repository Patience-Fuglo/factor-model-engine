"""Stage 6 -- real portfolio factor-risk decomposition: total variance
split into factor risk (via the real factor covariance matrix) and
specific risk (via each holding's own specific variance), plus real
concentration/risk-limit checks against ``RiskConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from ..config import RiskConfig
from ..data.schemas import ExposureSubject, FactorExposureRow, PortfolioHolding


@dataclass
class PortfolioRiskResult:
    portfolio_id: str
    factor_variance: float
    specific_variance: float
    total_variance: float
    factor_risk_contribution: dict[str, float] = field(default_factory=dict)
    effective_n_positions: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "factor_variance": self.factor_variance,
            "specific_variance": self.specific_variance,
            "total_variance": self.total_variance,
            "factor_risk_contribution": self.factor_risk_contribution,
            "effective_n_positions": self.effective_n_positions,
            "warnings": self.warnings,
        }


def _effective_n_positions(weights: Sequence[float]) -> float:
    """A real Herfindahl-based concentration measure: the number of
    EQUALLY-weighted positions that would produce the same real
    concentration as the actual weight vector. An equal-weighted
    10-position portfolio scores 10; a portfolio with one 90% position
    and nine tiny ones scores close to 1 -- a real, standard concentration
    read, not just a position count.
    """
    sum_sq = sum(w**2 for w in weights)
    if sum_sq <= 0:
        return float("nan")
    return 1.0 / sum_sq


def compute_portfolio_risk(
    portfolio_exposure_rows: Sequence[FactorExposureRow],
    factor_covariance: pd.DataFrame,
    holdings: Sequence[PortfolioHolding],
    specific_variance_by_security: pd.Series,
    portfolio_id: str,
    risk_cfg: RiskConfig,
) -> PortfolioRiskResult:
    """Real factor variance = w^T Sigma w, where w is this portfolio's
    own real exposure vector (from Stage 3's ``aggregate_exposure``,
    subject_type=PORTFOLIO) and Sigma is the real factor covariance
    matrix (Stage 6's ``compute_factor_covariance``). Each factor's real
    marginal contribution (``w_i * (Sigma w)_i``) sums exactly to the
    total factor variance -- the same real risk-contribution identity
    used elsewhere in this portfolio's risk-parity work, not a new,
    unverified formula.

    Specific variance assumes real idiosyncratic returns are
    uncorrelated across securities (the standard textbook
    approximation): ``sum(weight_i^2 * specific_variance_i)``. A holding
    with no real specific-variance estimate contributes 0 with a
    warning -- never a silently fabricated number.
    """
    own_rows = [r for r in portfolio_exposure_rows if r.subject_id == portfolio_id and r.subject_type == ExposureSubject.PORTFOLIO]
    w = pd.Series({r.factor_name: r.exposure_value for r in own_rows}).reindex(factor_covariance.columns)

    warnings: list[str] = []
    missing_factor_exposure = w[w.isna()].index.tolist()
    if missing_factor_exposure:
        warnings.append(f"no real portfolio exposure for {missing_factor_exposure} -- treated as 0 contribution")
    w = w.fillna(0.0)

    cov_w = factor_covariance.to_numpy() @ w.to_numpy()
    factor_variance = float(w.to_numpy() @ cov_w)
    factor_variance = max(factor_variance, 0.0)  # a real sample covariance is PSD; clip only float noise
    factor_risk_contribution = dict(zip(factor_covariance.columns, (w.to_numpy() * cov_w).tolist()))

    own_holdings = [h for h in holdings if h.portfolio_id == portfolio_id]
    weights_by_security = {h.security_id: h.weight for h in own_holdings}

    specific_variance = 0.0
    missing_specific = []
    for security_id, weight in weights_by_security.items():
        sv = specific_variance_by_security.get(security_id, float("nan"))
        if pd.isna(sv):
            missing_specific.append(security_id)
            continue
        specific_variance += (weight**2) * sv
    if missing_specific:
        warnings.append(f"{len(missing_specific)} real holdings have no specific-variance estimate -- excluded, not zero-filled by assumption")

    total_variance = factor_variance + specific_variance
    effective_n = _effective_n_positions(list(weights_by_security.values()))

    if total_variance > 0:
        for factor_name, contribution in factor_risk_contribution.items():
            share = contribution / total_variance
            if share > risk_cfg.max_single_factor_variance_share:
                warnings.append(
                    f"{factor_name}: {share:.1%} of total portfolio variance exceeds the configured "
                    f"{risk_cfg.max_single_factor_variance_share:.0%} concentration limit"
                )
    if pd.notna(effective_n) and effective_n < risk_cfg.min_effective_n_positions:
        warnings.append(
            f"effective n positions ({effective_n:.1f}) is below the configured minimum of "
            f"{risk_cfg.min_effective_n_positions}"
        )

    return PortfolioRiskResult(
        portfolio_id=portfolio_id,
        factor_variance=factor_variance,
        specific_variance=specific_variance,
        total_variance=total_variance,
        factor_risk_contribution=factor_risk_contribution,
        effective_n_positions=effective_n,
        warnings=warnings,
    )
