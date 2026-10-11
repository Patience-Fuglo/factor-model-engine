import pandas as pd
import pytest

from factor_model.data import (
    ExclusionReason,
    ExposureSubject,
    FactorAttributionRow,
    FactorExposureRow,
    FundamentalObservation,
    MarketObservation,
    Sector,
    UniverseInclusion,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_market_observation_rejects_known_as_of_before_as_of():
    with pytest.raises(ValueError, match="look-ahead"):
        MarketObservation(
            security_id="A", ticker="A", as_of=_ts("2020-02-01"), known_as_of=_ts("2020-01-01"),
            sector=Sector.TECHNOLOGY, is_active=True, price=100.0, shares_outstanding=1e6,
            market_cap=1e8, adv_20d=1e6,
        )


def test_market_observation_allows_known_as_of_equal_to_as_of():
    obs = MarketObservation(
        security_id="A", ticker="A", as_of=_ts("2020-02-01"), known_as_of=_ts("2020-02-01"),
        sector=Sector.TECHNOLOGY, is_active=True, price=100.0, shares_outstanding=1e6,
        market_cap=1e8, adv_20d=1e6,
    )
    assert obs.known_as_of == obs.as_of


def test_market_observation_adv_defaults_to_none_not_zero():
    obs = MarketObservation(
        security_id="A", ticker="A", as_of=_ts("2020-02-01"), known_as_of=_ts("2020-02-01"),
        sector=Sector.TECHNOLOGY, is_active=True, price=100.0, shares_outstanding=1e6,
        market_cap=1e8, adv_20d=None,
    )
    assert obs.adv_20d is None


def test_fundamental_observation_rejects_known_as_of_before_period_end():
    with pytest.raises(ValueError, match="look-ahead"):
        FundamentalObservation(
            security_id="A", fundamental_period_end=_ts("2020-03-31"), known_as_of=_ts("2020-03-01"),
            book_value_per_share=10.0, earnings_per_share=1.0,
        )


def test_fundamental_observation_allows_a_real_publication_lag():
    obs = FundamentalObservation(
        security_id="A", fundamental_period_end=_ts("2020-03-31"),
        known_as_of=_ts("2020-03-31") + pd.Timedelta(days=45),
        book_value_per_share=10.0, earnings_per_share=1.0,
    )
    assert (obs.known_as_of - obs.fundamental_period_end).days == 45


def test_market_and_fundamental_known_as_of_are_genuinely_independent_fields():
    # the same security, same real date, can have very different known_as_of
    # values for its market vs. fundamental data -- that's the whole point
    # of splitting these into two observation types
    market = MarketObservation(
        security_id="A", ticker="A", as_of=_ts("2020-03-31"), known_as_of=_ts("2020-03-31"),
        sector=Sector.TECHNOLOGY, is_active=True, price=100.0, shares_outstanding=1e6,
        market_cap=1e8, adv_20d=1e6,
    )
    fundamental = FundamentalObservation(
        security_id="A", fundamental_period_end=_ts("2020-03-31"),
        known_as_of=_ts("2020-03-31") + pd.Timedelta(days=45),
        book_value_per_share=10.0, earnings_per_share=1.0,
    )
    assert market.known_as_of != fundamental.known_as_of
    assert fundamental.known_as_of > market.known_as_of


def test_universe_inclusion_requires_reasons_when_excluded():
    with pytest.raises(ValueError):
        UniverseInclusion(security_id="A", included=False, reasons=())


def test_universe_inclusion_rejects_reasons_when_included():
    with pytest.raises(ValueError):
        UniverseInclusion(security_id="A", included=True, reasons=(ExclusionReason.DELISTED,))


def test_universe_inclusion_included_with_no_reasons_is_valid():
    inclusion = UniverseInclusion(security_id="A", included=True)
    assert inclusion.reasons == ()


def test_factor_exposure_row_is_a_point_in_time_snapshot_not_a_return():
    row = FactorExposureRow(
        subject_type=ExposureSubject.STOCK, subject_id="A", as_of=_ts("2020-03-31"),
        factor_name="momentum", exposure_value=1.3, source_known_as_of=_ts("2020-03-31"),
    )
    assert row.exposure_value == 1.3
    assert not hasattr(row, "contribution_to_return")


def test_factor_attribution_row_is_structurally_separate_from_exposure():
    # FactorAttributionRow has fields FactorExposureRow does NOT have
    # (period_start/period_end/contribution_to_return) -- enforcing the
    # user's correction that attribution is never just another exposure row
    row = FactorAttributionRow(
        subject_type=ExposureSubject.PORTFOLIO, subject_id="PORT1",
        period_start=_ts("2020-03-31"), period_end=_ts("2020-04-30"),
        factor_name="momentum", exposure_at_period_start=1.3,
        factor_return_over_period=-0.02, contribution_to_return=1.3 * -0.02,
    )
    assert row.contribution_to_return == pytest.approx(-0.026)
    assert not hasattr(row, "exposure_value")


def test_factor_attribution_row_rejects_non_positive_period():
    with pytest.raises(ValueError):
        FactorAttributionRow(
            subject_type=ExposureSubject.PORTFOLIO, subject_id="PORT1",
            period_start=_ts("2020-04-30"), period_end=_ts("2020-03-31"),
            factor_name="momentum", exposure_at_period_start=1.3,
            factor_return_over_period=-0.02, contribution_to_return=-0.026,
        )
