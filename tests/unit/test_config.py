import pytest

from factor_model.config import (
    DataConfig,
    DiagnosticsConfig,
    EngineConfig,
    MomentumConfig,
    StandardizationConfig,
    UniverseConfig,
    WinsorizationConfig,
    load_config,
)


def test_engine_config_defaults_are_valid():
    cfg = EngineConfig()
    assert cfg.data.fundamental_publication_lag_days == 45
    assert cfg.factors.momentum.lookback_months == 12
    assert cfg.factors.momentum.skip_recent_months == 1


def test_load_config_reads_real_default_yaml():
    cfg = load_config("configs/default.yaml")
    assert cfg.universe.min_market_cap == pytest.approx(1.0e8)
    assert cfg.factors.standardization.method == "zscore"
    assert cfg.diagnostics.min_cross_section_size == 20


def test_winsorization_config_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        WinsorizationConfig(lower_pct=0.5, upper_pct=0.5)
    with pytest.raises(ValueError):
        WinsorizationConfig(lower_pct=0.9, upper_pct=0.1)


def test_standardization_config_rejects_unknown_method():
    with pytest.raises(ValueError):
        StandardizationConfig(method="not_a_real_method")


def test_momentum_config_rejects_skip_greater_than_lookback():
    with pytest.raises(ValueError):
        MomentumConfig(lookback_months=1, skip_recent_months=1)


def test_momentum_config_is_configurable_not_hardcoded():
    cfg = MomentumConfig(lookback_months=6, skip_recent_months=0)
    assert cfg.lookback_months == 6
    assert cfg.skip_recent_months == 0


def test_data_config_rejects_negative_lag():
    with pytest.raises(ValueError):
        DataConfig(fundamental_publication_lag_days=-1)


def test_universe_config_rejects_negative_filters():
    with pytest.raises(ValueError):
        UniverseConfig(min_market_cap=-1.0)


def test_diagnostics_config_rejects_out_of_range_missingness():
    with pytest.raises(ValueError):
        DiagnosticsConfig(max_allowed_missingness_pct=1.5)
