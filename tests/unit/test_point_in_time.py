import pandas as pd
import pytest

from factor_model.data import FundamentalObservation, MarketObservation, Sector, as_of_view, latest_known_fundamental


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _market(as_of: str, known_as_of: str) -> MarketObservation:
    return MarketObservation(
        security_id="A", ticker="A", as_of=_ts(as_of), known_as_of=_ts(known_as_of),
        sector=Sector.TECHNOLOGY, is_active=True, price=100.0, shares_outstanding=1e6,
        market_cap=1e8, adv_20d=1e6,
    )


def _fundamental(period_end: str, known_as_of: str, book_value: float, flags: tuple = ()) -> FundamentalObservation:
    return FundamentalObservation(
        security_id="A", fundamental_period_end=_ts(period_end), known_as_of=_ts(known_as_of),
        book_value_per_share=book_value, earnings_per_share=1.0, quality_flags=flags,
    )


def test_as_of_view_excludes_not_yet_knowable_market_observations():
    observations = [_market("2020-01-31", "2020-01-31"), _market("2020-02-29", "2020-02-29")]
    view = as_of_view(observations, _ts("2020-01-31"))
    assert len(view) == 1
    assert view[0].as_of == _ts("2020-01-31")


def test_as_of_view_works_generically_for_fundamentals_too():
    observations = [
        _fundamental("2020-03-31", "2020-05-15", 10.0),
        _fundamental("2020-06-30", "2020-08-15", 11.0),
    ]
    view = as_of_view(observations, _ts("2020-06-01"))
    assert len(view) == 1
    assert view[0].fundamental_period_end == _ts("2020-03-31")


def test_latest_known_fundamental_before_restatement_is_knowable_returns_original():
    observations = [
        _fundamental("2020-03-31", "2020-05-15", 10.0),
        _fundamental("2020-03-31", "2020-10-17", 14.0, flags=("restated",)),
    ]
    result = latest_known_fundamental(observations, "A", _ts("2020-03-31"), as_of=_ts("2020-08-08"))
    assert result is not None
    assert result.book_value_per_share == 10.0


def test_latest_known_fundamental_after_restatement_is_knowable_returns_restated():
    observations = [
        _fundamental("2020-03-31", "2020-05-15", 10.0),
        _fundamental("2020-03-31", "2020-10-17", 14.0, flags=("restated",)),
    ]
    result = latest_known_fundamental(observations, "A", _ts("2020-03-31"), as_of=_ts("2020-11-01"))
    assert result is not None
    assert result.book_value_per_share == 14.0


def test_latest_known_fundamental_returns_none_before_any_vintage_is_knowable():
    observations = [_fundamental("2020-03-31", "2020-05-15", 10.0)]
    result = latest_known_fundamental(observations, "A", _ts("2020-03-31"), as_of=_ts("2020-04-01"))
    assert result is None


def test_a_naive_latest_row_implementation_would_have_leaked_the_restatement():
    # demonstrates exactly the real bug latest_known_fundamental prevents:
    # sorting by period_end then taking the last REGARDLESS of known_as_of
    # would return the restated value even when queried before it existed
    observations = [
        _fundamental("2020-03-31", "2020-05-15", 10.0),
        _fundamental("2020-03-31", "2020-10-17", 14.0, flags=("restated",)),
    ]
    naive_latest = max(observations, key=lambda o: o.known_as_of)
    correct = latest_known_fundamental(observations, "A", _ts("2020-03-31"), as_of=_ts("2020-08-08"))
    assert naive_latest.book_value_per_share == 14.0   # the naive (wrong) answer
    assert correct.book_value_per_share == 10.0         # the real, point-in-time-correct answer
    assert naive_latest.book_value_per_share != correct.book_value_per_share
