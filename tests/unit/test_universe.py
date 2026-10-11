import pandas as pd
import pytest

from factor_model.config import DataConfig, UniverseConfig
from factor_model.data import ExclusionReason, MarketObservation, Sector, build_point_in_time_universe


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _obs(security_id: str, as_of: str, **overrides) -> MarketObservation:
    defaults = dict(
        ticker=security_id, known_as_of=as_of, sector=Sector.TECHNOLOGY, is_active=True,
        price=100.0, shares_outstanding=1e6, market_cap=2.0e8, adv_20d=2.0e6,
    )
    defaults.update(overrides)
    return MarketObservation(security_id=security_id, as_of=_ts(as_of), known_as_of=_ts(defaults.pop("known_as_of")), **defaults)


def test_universe_includes_a_security_meeting_every_real_filter():
    observations = [_obs("A", "2020-06-30")]
    snapshot = build_point_in_time_universe(observations, _ts("2020-06-30"), UniverseConfig(), DataConfig())
    assert snapshot.included_ids() == ["A"]


def test_universe_excludes_delisted_security_with_reason():
    observations = [_obs("A", "2020-06-30", is_active=False)]
    snapshot = build_point_in_time_universe(observations, _ts("2020-06-30"), UniverseConfig(), DataConfig())
    assert "A" in snapshot.excluded_ids()
    member = next(m for m in snapshot.members if m.security_id == "A")
    assert ExclusionReason.DELISTED in member.reasons


def test_universe_excludes_below_min_market_cap_with_reason():
    observations = [_obs("A", "2020-06-30", market_cap=1.0e6)]
    cfg = UniverseConfig(min_market_cap=1.0e8)
    snapshot = build_point_in_time_universe(observations, _ts("2020-06-30"), cfg, DataConfig())
    member = next(m for m in snapshot.members if m.security_id == "A")
    assert ExclusionReason.BELOW_MIN_MARKET_CAP in member.reasons


def test_universe_excludes_below_min_adv_with_reason():
    observations = [_obs("A", "2020-06-30", adv_20d=100.0)]
    cfg = UniverseConfig(min_adv_usd=1.0e6)
    snapshot = build_point_in_time_universe(observations, _ts("2020-06-30"), cfg, DataConfig())
    member = next(m for m in snapshot.members if m.security_id == "A")
    assert ExclusionReason.BELOW_MIN_ADV in member.reasons


def test_universe_excludes_null_adv_as_below_min_not_zero():
    observations = [_obs("A", "2020-06-30", adv_20d=None)]
    snapshot = build_point_in_time_universe(observations, _ts("2020-06-30"), UniverseConfig(), DataConfig())
    member = next(m for m in snapshot.members if m.security_id == "A")
    assert ExclusionReason.BELOW_MIN_ADV in member.reasons


def test_universe_excludes_manually_excluded_sector():
    observations = [_obs("A", "2020-06-30", sector=Sector.ENERGY)]
    cfg = UniverseConfig(excluded_sectors=("energy",))
    snapshot = build_point_in_time_universe(observations, _ts("2020-06-30"), cfg, DataConfig())
    member = next(m for m in snapshot.members if m.security_id == "A")
    assert ExclusionReason.MANUALLY_EXCLUDED in member.reasons


def test_universe_excludes_insufficient_price_history():
    observations = [_obs("A", "2020-06-30")]
    snapshot = build_point_in_time_universe(
        observations, _ts("2020-06-30"), UniverseConfig(), DataConfig(min_price_history_days=252),
        price_history_days_by_security={"A": 30},
    )
    member = next(m for m in snapshot.members if m.security_id == "A")
    assert ExclusionReason.INSUFFICIENT_PRICE_HISTORY in member.reasons


def test_universe_never_uses_a_future_observation_as_current_constituent():
    # a security that only has a real observation AFTER as_of must be
    # excluded as not-yet-listed -- never silently included via a future row
    observations = [_obs("A", "2025-01-31")]
    snapshot = build_point_in_time_universe(observations, _ts("2020-06-30"), UniverseConfig(), DataConfig())
    member = next(m for m in snapshot.members if m.security_id == "A")
    assert member.included is False
    assert ExclusionReason.NOT_YET_LISTED in member.reasons


def test_universe_snapshot_carries_a_real_config_audit_trail():
    observations = [_obs("A", "2020-06-30")]
    cfg_a = UniverseConfig(min_market_cap=1.0e8)
    cfg_b = UniverseConfig(min_market_cap=5.0e8)
    snap_a = build_point_in_time_universe(observations, _ts("2020-06-30"), cfg_a, DataConfig())
    snap_b = build_point_in_time_universe(observations, _ts("2020-06-30"), cfg_b, DataConfig())
    assert snap_a.config_hash != snap_b.config_hash
