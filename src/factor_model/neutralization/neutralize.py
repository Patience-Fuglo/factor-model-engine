"""Stage 4 -- Neutralization: remove a chosen, EXPLICIT subset of factor
exposures from a signal's raw cross-sectional scores via real
cross-sectional OLS, leaving every other factor's exposure deliberately
untouched.

Reuses ``quant_toolkit.metrics.ols_regression``/``OLSResult`` from
``alpha-validation-toolkit`` -- the regression mechanics (design matrix,
residuals, R-squared, t-stats) are never reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
from quant_toolkit.metrics import OLSResult, ols_regression


def _safe_corr(a: pd.Series, b: pd.Series, std_floor: float = 1e-9) -> float:
    """A real correlation, guarded against a known numerical instability:
    when a series' real standard deviation is near zero (e.g. a residual
    from an exact, noiseless fit -- pure floating-point noise at
    ~1e-16), its correlation against anything is numerically meaningless
    and can spuriously land anywhere, including near 1.0, even though
    there is no real remaining relationship to speak of. Reports 0.0 in
    that case rather than a misleading, unstable number.
    """
    aligned = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    if aligned.iloc[:, 0].std() < std_floor or aligned.iloc[:, 1].std() < std_floor:
        return 0.0
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


@dataclass
class NeutralizationResult:
    """``residual_scores`` is the neutralized signal: what's left of the
    original score after removing the real, linear effect of
    ``neutralized_factors``. ``preserved_factors`` is every OTHER factor
    present in the real exposure panel supplied -- those are
    deliberately NOT touched, and their correlation with the signal is
    reported both before and after so a researcher can confirm an
    intentional, profitable exposure genuinely survived.
    """

    residual_scores: pd.Series
    ols_result: OLSResult
    neutralized_factors: list[str]
    preserved_factors: list[str]
    original_exposure_corr: dict[str, float] = field(default_factory=dict)
    residual_exposure_corr: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def neutralize_signal(
    signal_scores: pd.Series,
    factor_exposures: pd.DataFrame,
    factors_to_neutralize: Sequence[str],
    residual_exposure_threshold: float = 0.10,
) -> NeutralizationResult:
    """Regress ``signal_scores`` on ``factor_exposures[factors_to_neutralize]``
    (real cross-sectional OLS, aligned by security_id, NaN rows dropped by
    ``ols_regression`` itself) and return the residual as the neutralized
    signal.

    Every column of ``factor_exposures`` NOT named in
    ``factors_to_neutralize`` is a ``preserved_factor`` -- reported, with
    its own before/after correlation to the signal, but never regressed
    out. A real, honest diagnostic: if a neutralized factor's residual
    correlation still exceeds ``residual_exposure_threshold``, a warning
    is raised rather than silently reporting a "successful" neutralization
    that didn't actually work (e.g. a real nonlinear relationship a
    linear OLS fit cannot fully remove).
    """
    factors_to_neutralize = list(factors_to_neutralize)
    preserved_factors = [c for c in factor_exposures.columns if c not in factors_to_neutralize]

    X = factor_exposures[factors_to_neutralize]
    ols_result = ols_regression(signal_scores, X)

    aligned = pd.concat([signal_scores.rename("signal"), X], axis=1, join="inner").dropna()
    fitted = pd.Series(ols_result.alpha, index=aligned.index)
    for factor_name in factors_to_neutralize:
        fitted = fitted + ols_result.betas[factor_name] * aligned[factor_name]
    residual = (aligned["signal"] - fitted).reindex(signal_scores.index)

    original_corr = {col: _safe_corr(signal_scores, factor_exposures[col]) for col in factor_exposures.columns}
    residual_corr = {col: _safe_corr(residual, factor_exposures[col]) for col in factor_exposures.columns}

    warnings: list[str] = []
    for factor_name in factors_to_neutralize:
        value = residual_corr.get(factor_name, float("nan"))
        if pd.notna(value) and abs(value) > residual_exposure_threshold:
            warnings.append(
                f"{factor_name}: residual correlation {value:.3f} still exceeds the real "
                f"{residual_exposure_threshold} threshold after neutralization -- check for a "
                "real nonlinearity a linear OLS fit cannot remove"
            )

    return NeutralizationResult(
        residual_scores=residual,
        ols_result=ols_result,
        neutralized_factors=factors_to_neutralize,
        preserved_factors=preserved_factors,
        original_exposure_corr=original_corr,
        residual_exposure_corr=residual_corr,
        warnings=warnings,
    )
