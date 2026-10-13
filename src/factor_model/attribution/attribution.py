"""Stage 7 -- Attribution: the whole reason ``FactorAttributionRow`` was
schema'd separately from ``FactorExposureRow`` back in Stage 1. Combines
a subject's real exposure AT PERIOD_START (Stage 2/3) with each real
factor's realized return OVER THE PERIOD (Stage 5) into a per-factor P&L
contribution, plus one explicit RESIDUAL row -- whatever the subject's
own real realized return the factors couldn't explain (security
selection / idiosyncratic skill, or just noise).

The real reconciliation identity, verified by every test here:
``realized_return == sum(contribution_to_return for every row, including
the residual)`` -- exactly, not approximately, by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from ..data.schemas import ExposureSubject, FactorAttributionRow, FactorExposureRow, FactorReturnRow

RESIDUAL_FACTOR_NAME = "residual"


@dataclass
class AttributionReport:
    subject_type: ExposureSubject
    subject_id: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    rows: list[FactorAttributionRow]
    realized_return: float
    explained_return: float
    residual_return: float
    warnings: list[str] = field(default_factory=list)


def compute_attribution(
    exposure_rows: Sequence[FactorExposureRow],
    factor_return_rows: Sequence[FactorReturnRow],
    realized_return: float,
    subject_type: ExposureSubject,
    subject_id: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> AttributionReport:
    """A factor only contributes a real attribution row when BOTH a real
    exposure (this subject, at ``period_start``) AND a real factor
    return (this exact ``[period_start, period_end]``) are available --
    one without the other is excluded with a warning, never silently
    assumed to be zero. The RESIDUAL row absorbs everything else, so the
    real reconciliation identity always holds regardless of coverage
    gaps: a factor this subject has no real exposure reading for simply
    has its real contribution folded into the residual, honestly, rather
    than invented.
    """
    own_exposures = {
        r.factor_name: r
        for r in exposure_rows
        if r.subject_id == subject_id and r.subject_type == subject_type and r.as_of == period_start
    }
    own_factor_returns = {
        r.factor_name: r for r in factor_return_rows if r.period_start == period_start and r.period_end == period_end
    }

    warnings: list[str] = []
    missing_exposure = sorted(set(own_factor_returns) - set(own_exposures))
    missing_return = sorted(set(own_exposures) - set(own_factor_returns))
    if missing_exposure:
        warnings.append(f"no real exposure for {missing_exposure} at period_start -- folded into residual")
    if missing_return:
        warnings.append(f"no real factor return for {missing_return} over this period -- folded into residual")

    rows: list[FactorAttributionRow] = []
    explained = 0.0
    for factor_name in sorted(set(own_exposures) & set(own_factor_returns)):
        exposure_row = own_exposures[factor_name]
        return_row = own_factor_returns[factor_name]
        if pd.isna(exposure_row.exposure_value):
            warnings.append(f"{factor_name}: exposure is NaN at period_start -- folded into residual")
            continue
        contribution = exposure_row.exposure_value * return_row.factor_return
        explained += contribution
        rows.append(
            FactorAttributionRow(
                subject_type=subject_type,
                subject_id=subject_id,
                period_start=period_start,
                period_end=period_end,
                factor_name=factor_name,
                exposure_at_period_start=exposure_row.exposure_value,
                factor_return_over_period=return_row.factor_return,
                contribution_to_return=contribution,
                is_residual=False,
            )
        )

    residual = realized_return - explained
    rows.append(
        FactorAttributionRow(
            subject_type=subject_type,
            subject_id=subject_id,
            period_start=period_start,
            period_end=period_end,
            factor_name=RESIDUAL_FACTOR_NAME,
            exposure_at_period_start=1.0,
            factor_return_over_period=residual,
            contribution_to_return=residual,
            is_residual=True,
        )
    )

    return AttributionReport(
        subject_type=subject_type,
        subject_id=subject_id,
        period_start=period_start,
        period_end=period_end,
        rows=rows,
        realized_return=realized_return,
        explained_return=explained,
        residual_return=residual,
        warnings=warnings,
    )


def attribution_rows_to_frame(rows: Sequence[FactorAttributionRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_type": row.subject_type.value,
                "subject_id": row.subject_id,
                "period_start": row.period_start,
                "period_end": row.period_end,
                "factor_name": row.factor_name,
                "exposure_at_period_start": row.exposure_at_period_start,
                "factor_return_over_period": row.factor_return_over_period,
                "contribution_to_return": row.contribution_to_return,
                "is_residual": row.is_residual,
            }
            for row in rows
        ]
    )
