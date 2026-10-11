"""Value: book-to-price, a real, simple fundamental-based factor."""

from __future__ import annotations

import pandas as pd


def compute_book_to_price(book_value_per_share: pd.Series, price: pd.Series) -> pd.Series:
    """Real book-to-price = book_value_per_share / price, aligned by
    index. NaN wherever either real input is missing or price is
    non-positive -- never silently zero-filled or divided by zero.
    """
    aligned = pd.concat([book_value_per_share.rename("book"), price.rename("price")], axis=1)
    ratio = aligned["book"] / aligned["price"].where(aligned["price"] > 0)
    return ratio.rename("book_to_price")
