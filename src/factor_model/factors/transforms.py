"""Cross-sectional preprocessing: ``winsorize`` and ``standardize`` are
pure, real transforms applied to one real cross-section at a time (one
date, many securities) -- never hidden inside a factor calculation
itself, so RAW -> CLEANED -> STANDARDIZED stays auditable as separate
steps.
"""

from __future__ import annotations

import pandas as pd

from ..config import StandardizationConfig, WinsorizationConfig


def winsorize(values: pd.Series, cfg: WinsorizationConfig) -> pd.Series:
    """Clip real cross-sectional values to the ``[lower_pct, upper_pct]``
    quantiles, computed from the real non-null values only. Real NaNs
    pass through unchanged -- winsorizing never manufactures a value
    for real missing data.
    """
    clean = values.dropna()
    if len(clean) == 0:
        return values.copy()
    lower = clean.quantile(cfg.lower_pct)
    upper = clean.quantile(cfg.upper_pct)
    return values.clip(lower=lower, upper=upper)


def standardize(values: pd.Series, cfg: StandardizationConfig) -> pd.Series:
    """Cross-sectionally standardize real values per ``cfg.method``
    (zscore / rank / minmax). Real NaN stays NaN -- never silently
    treated as 0.
    """
    clean = values.dropna()
    if len(clean) < 2:
        return pd.Series(float("nan"), index=values.index)

    if cfg.method == "zscore":
        mean = clean.mean()
        std = clean.std(ddof=1)
        if std == 0:
            return pd.Series(float("nan"), index=values.index)
        return (values - mean) / std
    if cfg.method == "rank":
        return values.rank(pct=True)
    if cfg.method == "minmax":
        lo, hi = clean.min(), clean.max()
        if hi == lo:
            return pd.Series(float("nan"), index=values.index)
        return (values - lo) / (hi - lo)
    raise ValueError(f"unknown standardization method: {cfg.method!r}")
