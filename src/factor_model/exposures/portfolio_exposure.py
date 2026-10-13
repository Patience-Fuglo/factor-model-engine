"""Stage 3 -- Exposure Engine: aggregate real stock-level
``FactorExposureRow``s (``subject_type=STOCK``, from Stage 2) into one
SIGNAL- or PORTFOLIO-level exposure per factor, weighted by a real
``PortfolioHolding`` weight scheme.

A benchmark is just another ``PortfolioHolding`` weight scheme (e.g.
equal-weight or cap-weight over the eligible universe) -- it is
aggregated through the exact same function, with ``subject_type=
PORTFOLIO`` and its own ``subject_id``, never a separate code path.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..data.schemas import ExposureSubject, FactorExposureRow, PortfolioHolding


def aggregate_exposure(
    stock_exposure_rows: Sequence[FactorExposureRow],
    holdings: Sequence[PortfolioHolding],
    subject_type: ExposureSubject,
    subject_id: str,
) -> list[FactorExposureRow]:
    """Weighted-average aggregation of real stock-level standardized
    exposures into one ``subject_id`` exposure per real factor present
    in ``stock_exposure_rows``.

    Every input row must share the same real ``as_of`` -- aggregating
    across dates would silently mix point-in-time snapshots from
    different real moments, which is exactly the kind of conflation this
    whole engine exists to prevent.

    A stock with a real holding weight but a missing (NaN) exposure for
    a given factor is excluded from BOTH the weighted sum and the total
    weight for that factor -- i.e. the aggregate re-normalizes over only
    the real, available constituent exposures, rather than silently
    treating a missing stock exposure as zero (which would understate
    the aggregate whenever coverage is incomplete). If NO real
    constituent has a usable exposure for a factor, the aggregate is NaN
    with an explicit ``"insufficient_coverage"`` flag -- never a
    fabricated zero.

    The aggregate's own ``source_known_as_of`` is the LATEST real
    ``known_as_of`` among every constituent that actually contributed to
    it (both its exposure row and its holding weight) -- an aggregate is
    only as current as its stalest real contributing input.
    """
    if not stock_exposure_rows:
        return []

    as_of_values = {row.as_of for row in stock_exposure_rows}
    if len(as_of_values) > 1:
        raise ValueError(f"aggregate_exposure requires one shared as_of, got {sorted(as_of_values)}")
    as_of = next(iter(as_of_values))

    holdings_by_security = {h.security_id: h for h in holdings if h.portfolio_id == subject_id}
    factor_names = sorted({row.factor_name for row in stock_exposure_rows})
    rows_by_factor: dict[str, list[FactorExposureRow]] = {name: [] for name in factor_names}
    for row in stock_exposure_rows:
        rows_by_factor[row.factor_name].append(row)

    output: list[FactorExposureRow] = []
    for factor_name in factor_names:
        weighted_sum = 0.0
        total_weight = 0.0
        latest_known_as_of = as_of
        n_contributing = 0
        for row in rows_by_factor[factor_name]:
            holding = holdings_by_security.get(row.subject_id)
            if holding is None or pd.isna(row.exposure_value):
                continue
            weighted_sum += holding.weight * row.exposure_value
            total_weight += holding.weight
            latest_known_as_of = max(latest_known_as_of, row.source_known_as_of, holding.known_as_of)
            n_contributing += 1

        if n_contributing == 0 or total_weight == 0:
            output.append(
                FactorExposureRow(
                    subject_type=subject_type,
                    subject_id=subject_id,
                    as_of=as_of,
                    factor_name=factor_name,
                    exposure_value=float("nan"),
                    source_known_as_of=as_of,
                    raw_value=None,
                    quality_flags=("insufficient_coverage",),
                )
            )
            continue

        output.append(
            FactorExposureRow(
                subject_type=subject_type,
                subject_id=subject_id,
                as_of=as_of,
                factor_name=factor_name,
                exposure_value=weighted_sum / total_weight,
                source_known_as_of=latest_known_as_of,
                raw_value=None,
                quality_flags=(),
            )
        )
    return output


def equal_weight_benchmark_holdings(
    security_ids: Sequence[str], as_of: pd.Timestamp, portfolio_id: str = "benchmark_equal_weight"
) -> list[PortfolioHolding]:
    """A real, simple equal-weight benchmark over a given real eligible
    universe -- one of the most common real benchmark constructions, and
    a convenient default when a researcher has no specific cap-weighted
    or custom benchmark to supply. Not the only valid benchmark: any
    other real weight scheme is just another ``PortfolioHolding`` set.
    """
    if not security_ids:
        return []
    weight = 1.0 / len(security_ids)
    return [
        PortfolioHolding(portfolio_id=portfolio_id, security_id=sid, as_of=as_of, weight=weight, known_as_of=as_of)
        for sid in security_ids
    ]
