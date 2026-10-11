"""Point-in-time accessors: the real functions a look-ahead bug or a
restatement leak would actually break.

Both ``MarketObservation`` and ``FundamentalObservation`` expose their
own ``known_as_of`` (deliberately never shared between them -- see
``schemas.py``), so ``as_of_view`` works generically over either real
observation type via duck typing.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

import pandas as pd

from .schemas import FundamentalObservation

_Observation = TypeVar("_Observation")


def as_of_view(observations: Sequence[_Observation], as_of: pd.Timestamp) -> list[_Observation]:
    """Only the real observations knowable at ``as_of``: those with
    ``known_as_of <= as_of``. Works for a sequence of either
    ``MarketObservation`` or ``FundamentalObservation`` -- both expose
    ``known_as_of``, nothing else is assumed.
    """
    return [obs for obs in observations if obs.known_as_of <= as_of]


def latest_known_fundamental(
    observations: Sequence[FundamentalObservation],
    security_id: str,
    period_end: pd.Timestamp,
    as_of: pd.Timestamp,
) -> FundamentalObservation | None:
    """Among every real vintage recorded for ``(security_id, period_end)``,
    return the one with the latest real ``known_as_of`` that is still
    ``<= as_of`` -- i.e. what a researcher actually querying at ``as_of``
    would have seen.

    A naive "most recent vintage, full stop" implementation (ignoring
    ``known_as_of <= as_of``) would leak a real restatement into queries
    made before that restatement was actually knowable -- exactly the
    real bug this function exists to prevent. Returns ``None`` if no
    real vintage for this period was knowable yet at ``as_of``.
    """
    candidates = [
        obs
        for obs in observations
        if obs.security_id == security_id
        and obs.fundamental_period_end == period_end
        and obs.known_as_of <= as_of
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda obs: obs.known_as_of)
