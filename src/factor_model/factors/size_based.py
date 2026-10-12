"""Size: log market capitalization, a real, simple size factor."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_size(market_cap: pd.Series) -> pd.Series:
    """Real Size = log(market_cap). NaN for non-positive market cap --
    never log of zero/a negative number, and never silently clipped to
    some floor value that would fabricate a real reading.
    """
    positive = market_cap.where(market_cap > 0)
    return np.log(positive).rename("size")
