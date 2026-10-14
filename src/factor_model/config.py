"""Engine configuration: every ambiguous research decision lives here,
in YAML, with a recommended default and reasoning in
``docs/research_assumptions.md`` -- never hardcoded in source.

Plain ``@dataclass`` throughout, matching the house style already used
across this portfolio's other repos (``alpha-validation-toolkit``'s
``OLSResult``/``PurgedWalkForward``), rather than introducing pydantic
as a new dependency this stage doesn't strictly need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class RunConfig:
    output_dir: str = "outputs"
    random_seed: int = 1337


@dataclass(frozen=True)
class DataConfig:
    # RESEARCH ASSUMPTION -- typical US 10-Q filing lag; see docs/research_assumptions.md
    fundamental_publication_lag_days: int = 45
    # RESEARCH ASSUMPTION -- minimum real trading history for universe eligibility
    min_price_history_days: int = 252
    # RESEARCH ASSUMPTION -- fundamentals older than this are flagged stale
    stale_fundamentals_after_days: int = 190

    def __post_init__(self) -> None:
        if self.fundamental_publication_lag_days < 0:
            raise ValueError("fundamental_publication_lag_days cannot be negative")
        if self.min_price_history_days < 1:
            raise ValueError("min_price_history_days must be >= 1")
        if self.stale_fundamentals_after_days < 1:
            raise ValueError("stale_fundamentals_after_days must be >= 1")


@dataclass(frozen=True)
class UniverseConfig:
    # RESEARCH ASSUMPTION -- liquidity/size filter
    min_market_cap: float = 1.0e8
    # RESEARCH ASSUMPTION -- liquidity filter
    min_adv_usd: float = 1.0e6
    # RESEARCH ASSUMPTION -- sector exclusions, none by default
    excluded_sectors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.min_market_cap < 0:
            raise ValueError("min_market_cap cannot be negative")
        if self.min_adv_usd < 0:
            raise ValueError("min_adv_usd cannot be negative")


@dataclass(frozen=True)
class WinsorizationConfig:
    # RESEARCH ASSUMPTION -- tail-trim aggressiveness
    lower_pct: float = 0.01
    upper_pct: float = 0.99

    def __post_init__(self) -> None:
        if not 0 <= self.lower_pct < self.upper_pct <= 1:
            raise ValueError("require 0 <= lower_pct < upper_pct <= 1")


@dataclass(frozen=True)
class StandardizationConfig:
    # RESEARCH ASSUMPTION -- zscore vs. rank vs. min-max
    method: str = "zscore"

    def __post_init__(self) -> None:
        if self.method not in ("zscore", "rank", "minmax"):
            raise ValueError(f"unknown standardization method: {self.method!r}")


@dataclass(frozen=True)
class MomentumConfig:
    # RESEARCH ASSUMPTION -- classic 12-1 convention; configurable, never hardcoded elsewhere
    lookback_months: int = 12
    skip_recent_months: int = 1

    def __post_init__(self) -> None:
        if self.lookback_months < 1:
            raise ValueError("lookback_months must be >= 1")
        if self.skip_recent_months < 0:
            raise ValueError("skip_recent_months cannot be negative")
        if self.skip_recent_months >= self.lookback_months:
            raise ValueError("skip_recent_months must be smaller than lookback_months")


@dataclass(frozen=True)
class QualityConfig:
    # RESEARCH ASSUMPTION -- equal-weighted profitability/safety composite by
    # default; both sub-measures are already cross-sectionally standardized
    # before combining, so these weights are a real mixing choice, not a
    # unit-conversion fudge
    profitability_weight: float = 0.5
    leverage_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.profitability_weight < 0 or self.leverage_weight < 0:
            raise ValueError("quality component weights cannot be negative")
        if self.profitability_weight + self.leverage_weight == 0:
            raise ValueError("at least one quality component weight must be positive")


@dataclass(frozen=True)
class FactorConfig:
    winsorization: WinsorizationConfig = field(default_factory=WinsorizationConfig)
    standardization: StandardizationConfig = field(default_factory=StandardizationConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)


@dataclass(frozen=True)
class DiagnosticsConfig:
    # RESEARCH ASSUMPTION
    max_allowed_missingness_pct: float = 0.10
    # RESEARCH ASSUMPTION -- below this, warn "too small to trust"
    min_cross_section_size: int = 20

    def __post_init__(self) -> None:
        if not 0 <= self.max_allowed_missingness_pct <= 1:
            raise ValueError("max_allowed_missingness_pct must be in [0, 1]")
        if self.min_cross_section_size < 1:
            raise ValueError("min_cross_section_size must be >= 1")


@dataclass(frozen=True)
class RiskConfig:
    # RESEARCH ASSUMPTION -- below this many real periods, a factor
    # covariance matrix is flagged unstable, not trusted silently
    min_periods_for_covariance: int = 24
    # RESEARCH ASSUMPTION -- any one factor contributing more than this
    # share of total factor variance is flagged as concentrated
    max_single_factor_variance_share: float = 0.60
    # RESEARCH ASSUMPTION -- below this many real effective positions
    # (1 / sum(weight^2)), a portfolio is flagged as concentrated
    min_effective_n_positions: float = 5.0

    def __post_init__(self) -> None:
        if self.min_periods_for_covariance < 2:
            raise ValueError("min_periods_for_covariance must be >= 2")
        if not 0 < self.max_single_factor_variance_share <= 1:
            raise ValueError("max_single_factor_variance_share must be in (0, 1]")
        if self.min_effective_n_positions <= 0:
            raise ValueError("min_effective_n_positions must be positive")


@dataclass(frozen=True)
class RegimeConfig:
    # RESEARCH ASSUMPTION -- trailing window for rolling factor-return volatility
    volatility_window: int = 12
    # RESEARCH ASSUMPTION -- expanding-window percentile cutoffs for low/high vol regimes
    low_volatility_percentile: float = 0.33
    high_volatility_percentile: float = 0.67

    def __post_init__(self) -> None:
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be >= 2")
        if not 0 < self.low_volatility_percentile < self.high_volatility_percentile < 1:
            raise ValueError("require 0 < low_volatility_percentile < high_volatility_percentile < 1")


@dataclass(frozen=True)
class MonitoringConfig:
    # RESEARCH ASSUMPTION -- how many of the most recent real periods count as "recent" for drift detection
    drift_recent_window: int = 6
    # RESEARCH ASSUMPTION -- flag drift if |recent IC mean| falls below this fraction of |historical IC mean|
    drift_decay_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.drift_recent_window < 2:
            raise ValueError("drift_recent_window must be >= 2")
        if not 0 < self.drift_decay_threshold <= 1:
            raise ValueError("drift_decay_threshold must be in (0, 1]")


@dataclass(frozen=True)
class EngineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    regimes: RegimeConfig = field(default_factory=RegimeConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


def load_config(path: str) -> EngineConfig:
    """Load a real YAML config file into a validated EngineConfig.
    Missing sections fall back to each dataclass's own documented
    default -- never a silent empty/zero value."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    run = RunConfig(**raw.get("run", {}))
    data = DataConfig(**raw.get("data", {}))

    # PyYAML's float resolver requires a SIGNED exponent ("1.0e+8"); an
    # unsigned exponent ("1.0e8") is silently parsed as a plain string,
    # not a float -- coerce explicitly here so a future config edit that
    # drops the sign fails with a clear ValueError, not a confusing
    # TypeError deep inside a dataclass comparison.
    universe_raw = dict(raw.get("universe", {}))
    for numeric_field in ("min_market_cap", "min_adv_usd"):
        if numeric_field in universe_raw:
            universe_raw[numeric_field] = float(universe_raw[numeric_field])
    if "excluded_sectors" in universe_raw:
        universe_raw["excluded_sectors"] = tuple(universe_raw["excluded_sectors"])
    universe = UniverseConfig(**universe_raw)

    factors_raw = raw.get("factors", {})
    winsorization = WinsorizationConfig(**factors_raw.get("winsorization", {}))
    standardization = StandardizationConfig(**factors_raw.get("standardization", {}))
    momentum = MomentumConfig(**factors_raw.get("momentum", {}))
    quality = QualityConfig(**factors_raw.get("quality", {}))
    factors = FactorConfig(
        winsorization=winsorization, standardization=standardization, momentum=momentum, quality=quality
    )

    diagnostics = DiagnosticsConfig(**raw.get("diagnostics", {}))
    risk = RiskConfig(**raw.get("risk", {}))
    regimes = RegimeConfig(**raw.get("regimes", {}))
    monitoring = MonitoringConfig(**raw.get("monitoring", {}))

    return EngineConfig(
        run=run, data=data, universe=universe, factors=factors, diagnostics=diagnostics,
        risk=risk, regimes=regimes, monitoring=monitoring,
    )
