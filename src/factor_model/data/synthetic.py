"""A deterministic synthetic dataset with a KNOWN ground-truth factor
structure, so later stages' factor construction/regression can be
checked against it directly, without any proprietary data.

Ground truth (``SyntheticDataset.ground_truth``) is a separate table,
never consumed by production pipeline code -- only by tests/notebooks.
This keeps the synthetic generator's own "answer key" isolated from
anything resembling a real-data adapter.

Labeled explicitly, everywhere this data is used: this is synthetic,
not real market data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schemas import FundamentalObservation, MarketObservation, Sector

_STREAMS = {
    "momentum": 0,
    "value": 1,
    "noise": 2,
    "fundamentals": 3,
    "cross_section": 4,
    "scale": 5,
    # Stage 2 additions -- each its own independent stream, so adding
    # them can never perturb any Stage 1 stream's draws (each
    # (security_index, stream_index) pair is its own independent
    # SeedSequence spawn). "quality_noise" is deliberately NOT drawn
    # from the shared "fundamentals" stream: that stream is reused
    # across every reporting period per security, so inserting extra
    # draws mid-loop there would shift every LATER period's
    # book_noise/eps_noise draw -- a real bug caught by the golden
    # regression test (see git history), not a hypothetical one.
    "quality": 6,
    "quality_noise": 7,
}


def _rng_for(base_seed: int, security_index: int, stream: str) -> np.random.Generator:
    """A real, deterministic, per-security/per-stream generator. Adding a
    51st security later never perturbs securities 0..49's own draws,
    since each (security_index, stream) pair spawns its own independent
    sub-sequence from the same base_seed."""
    seed_seq = np.random.SeedSequence(base_seed, spawn_key=(security_index, _STREAMS[stream]))
    return np.random.default_rng(seed_seq)


def _ar1_series(rng: np.random.Generator, n_periods: int, persistence: float) -> np.ndarray:
    """A real AR(1) process with approximately unit unconditional variance
    (innovation variance = 1 - persistence^2), used for both latent
    momentum_score and value_score."""
    innovation_scale = float(np.sqrt(max(1e-8, 1.0 - persistence**2)))
    series = np.zeros(n_periods)
    series[0] = rng.normal(0.0, 1.0)
    for t in range(1, n_periods):
        series[t] = persistence * series[t - 1] + rng.normal(0.0, innovation_scale)
    return series


def _cross_sectional_zscore(matrix: np.ndarray) -> np.ndarray:
    """z-score each column (one real cross-section per date) across rows
    (securities) -- the same real transform ``factors/transforms.py``
    implements, applied here just to build the ground-truth factors."""
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, ddof=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std


@dataclass(frozen=True)
class SyntheticDataConfig:
    n_securities: int = 50
    n_sectors: int = 5
    n_periods_months: int = 60
    start_date: str = "2016-01-31"
    base_seed: int = 1337
    momentum_loading: float = 0.5
    value_loading: float = 0.3
    momentum_persistence: float = 0.85
    value_persistence: float = 0.95
    return_scale_monthly: float = 0.08
    target_r_squared: float = 0.30
    fundamentals_update_every_n_months: int = 3
    fundamental_publication_lag_days: int = 45
    restatement_enabled: bool = True
    restatement_security_index: int = 7
    restatement_period_index: int = 23
    restatement_multiplier: float = 1.4
    restatement_original_lag_days: int = 45
    restatement_restated_lag_days: int = 200
    # Stage 2 addition -- an independent latent factor driving
    # total_assets/total_debt only, NEVER realized_return (Stage 1's
    # realized_return formula is untouched, so every Stage 1 golden-
    # pinned price/return value stays byte-identical)
    quality_persistence: float = 0.90

    def __post_init__(self) -> None:
        if self.n_securities % self.n_sectors != 0:
            raise ValueError("n_securities must divide evenly across n_sectors for this generator")
        if not 0 < self.target_r_squared < 1:
            raise ValueError("target_r_squared must be in (0, 1)")
        if self.restatement_period_index % self.fundamentals_update_every_n_months != (
            self.fundamentals_update_every_n_months - 1
        ):
            raise ValueError("restatement_period_index must land on a real fundamental-reporting period")


def load_synthetic_config(path: str) -> SyntheticDataConfig:
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    restatement = raw.pop("restatement_scenario", {}) or {}
    factor_loadings = raw.pop("factor_loadings", {}) or {}
    return SyntheticDataConfig(
        n_securities=raw.get("n_securities", 50),
        n_sectors=raw.get("n_sectors", 5),
        n_periods_months=raw.get("n_periods_months", 60),
        start_date=raw.get("start_date", "2016-01-31"),
        base_seed=raw.get("base_seed", 1337),
        momentum_loading=factor_loadings.get("momentum", 0.5),
        value_loading=factor_loadings.get("value", 0.3),
        momentum_persistence=raw.get("momentum_persistence", 0.85),
        value_persistence=raw.get("value_persistence", 0.95),
        return_scale_monthly=raw.get("return_scale_monthly", 0.08),
        target_r_squared=raw.get("target_r_squared", 0.30),
        fundamentals_update_every_n_months=raw.get("fundamentals_update_every_n_months", 3),
        fundamental_publication_lag_days=restatement.get("original_lag_days", 45),
        restatement_enabled=restatement.get("enabled", True),
        restatement_security_index=restatement.get("security_index", 7),
        restatement_period_index=restatement.get("period_index", 23),
        restatement_multiplier=restatement.get("restated_multiplier", 1.4),
        restatement_original_lag_days=restatement.get("original_lag_days", 45),
        restatement_restated_lag_days=restatement.get("restated_lag_days", 200),
        quality_persistence=raw.get("quality_persistence", 0.90),
    )


@dataclass
class SyntheticDataset:
    market: pd.DataFrame
    fundamentals: pd.DataFrame
    ground_truth: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def generate_synthetic_dataset(cfg: SyntheticDataConfig | None = None) -> SyntheticDataset:
    """Build the real, deterministic synthetic dataset.

    ``realized_return = return_scale_monthly * (momentum_loading *
    momentum_score + value_loading * value_score + noise)`` -- both
    latent scores are independent AR(1) processes, cross-sectionally
    z-scored every period, with noise calibrated so the two-factor
    model explains ``target_r_squared`` of real return variance (a
    realistic ~30%, not a toy 100%). Price is a derived cumulative-
    return path; book value and EPS are their own series, correlated
    with (not identical to) the value score, on a quarterly cadence.

    One deliberate point-in-time violation: the configured
    (security_index, period_index) gets two real fundamental vintages
    for the same period -- an original and a later restated value --
    so ``point_in_time.latest_known_fundamental`` has something real to
    get right (or a naive implementation, wrong).

    Reproducibility scope -- what "adding a security never perturbs
    earlier ones" actually covers: every per-security RAW draw (the
    AR(1) momentum/value latent series, noise, and the shares/turnover/
    book/eps scale parameters) is independent of ``n_securities``, since
    each is spawned from its own ``(base_seed, security_index, stream)``
    sub-sequence. Cross-sectionally STANDARDIZED quantities --
    ``momentum_score``/``value_score`` in ``ground_truth`` (z-scored
    across the whole universe each period), and everything derived from
    them (``realized_return``, ``price``, ``market_cap``, ``adv_20d``,
    and the value-correlated ``book_value_per_share``/
    ``earnings_per_share``) -- are NOT invariant to ``n_securities``, and
    structurally cannot be: a cross-sectional z-score is relative to its
    peer universe by definition, exactly as in real cross-sectional
    factor research. This is expected, not a bug.
    """
    cfg = cfg or SyntheticDataConfig()
    dates = pd.date_range(start=cfg.start_date, periods=cfg.n_periods_months, freq="ME")
    sectors_cycle = list(Sector)[: cfg.n_sectors]

    momentum_matrix = np.vstack(
        [_ar1_series(_rng_for(cfg.base_seed, i, "momentum"), cfg.n_periods_months, cfg.momentum_persistence) for i in range(cfg.n_securities)]
    )
    value_matrix = np.vstack(
        [_ar1_series(_rng_for(cfg.base_seed, i, "value"), cfg.n_periods_months, cfg.value_persistence) for i in range(cfg.n_securities)]
    )
    momentum_z = _cross_sectional_zscore(momentum_matrix)
    value_z = _cross_sectional_zscore(value_matrix)

    # Stage 2 addition: an independent latent Quality factor, deliberately
    # NOT wired into realized_return (unlike momentum/value above) --
    # its only real role is to drive total_assets/total_debt below, so the
    # Quality factor's leverage pillar has a genuine, planted, recoverable
    # ground-truth relationship for testing, without touching Stage 1's
    # realized_return/price formula at all
    quality_matrix = np.vstack(
        [_ar1_series(_rng_for(cfg.base_seed, i, "quality"), cfg.n_periods_months, cfg.quality_persistence) for i in range(cfg.n_securities)]
    )
    quality_z = _cross_sectional_zscore(quality_matrix)

    explained_var = cfg.momentum_loading**2 + cfg.value_loading**2
    noise_var = explained_var * (1.0 - cfg.target_r_squared) / cfg.target_r_squared
    noise_std = float(np.sqrt(noise_var))
    noise_matrix = np.vstack(
        [_rng_for(cfg.base_seed, i, "noise").normal(0.0, noise_std, size=cfg.n_periods_months) for i in range(cfg.n_securities)]
    )

    realized_return = cfg.return_scale_monthly * (
        cfg.momentum_loading * momentum_z + cfg.value_loading * value_z + noise_matrix
    )

    price = np.empty((cfg.n_securities, cfg.n_periods_months))
    price[:, 0] = 100.0 * (1.0 + realized_return[:, 0])
    for t in range(1, cfg.n_periods_months):
        price[:, t] = price[:, t - 1] * (1.0 + realized_return[:, t])

    # drawn per-security (own SeedSequence spawn, like momentum/value/noise)
    # rather than as one shared size-N array -- a shared array would make
    # security i's own shares_outstanding/turnover/book/eps scale depend on
    # the TOTAL number of securities requested (via sequential consumption
    # of one rng), silently breaking the "adding a security never perturbs
    # earlier ones" guarantee the same way the old sector-assignment bug did
    shares_outstanding = np.empty(cfg.n_securities)
    turnover_rate = np.empty(cfg.n_securities)
    book_scale = np.empty(cfg.n_securities)
    eps_scale = np.empty(cfg.n_securities)
    asset_scale = np.empty(cfg.n_securities)
    leverage_base = np.empty(cfg.n_securities)
    for i in range(cfg.n_securities):
        scale_rng = _rng_for(cfg.base_seed, i, "scale")
        shares_outstanding[i] = scale_rng.uniform(1.0e6, 2.0e8)
        turnover_rate[i] = scale_rng.uniform(0.003, 0.02)
        book_scale[i] = scale_rng.uniform(5.0, 50.0)
        eps_scale[i] = scale_rng.uniform(0.5, 8.0)
        # Stage 2 additions, drawn AFTER the four Stage 1 draws above from
        # this SAME per-security generator -- appending more draws never
        # changes values already drawn earlier in sequence, so every
        # Stage 1 golden-pinned value stays byte-identical
        asset_scale[i] = scale_rng.uniform(1.0e8, 5.0e9)
        leverage_base[i] = scale_rng.uniform(0.2, 0.6)

    market_rows = []
    for i in range(cfg.n_securities):
        # round-robin by security_index alone (never by total n_securities /
        # per_sector), so security i's sector never changes when the total
        # universe size changes -- a real requirement, not cosmetic, for
        # "adding a security never perturbs earlier ones" to actually hold
        sector = sectors_cycle[i % cfg.n_sectors]
        for t in range(cfg.n_periods_months):
            as_of = dates[t]
            market_cap = float(price[i, t] * shares_outstanding[i])
            market_rows.append(
                {
                    "security_id": f"SYN{i:03d}",
                    "ticker": f"SYN{i:03d}",
                    "as_of": as_of,
                    "known_as_of": as_of,
                    "sector": sector.value,
                    "is_active": True,
                    "price": float(price[i, t]),
                    "shares_outstanding": float(shares_outstanding[i]),
                    "market_cap": market_cap,
                    "adv_20d": float(market_cap * turnover_rate[i]),
                    "quality_flags": "",
                }
            )
    market_df = pd.DataFrame(market_rows)

    fundamental_rows = []
    for i in range(cfg.n_securities):
        fundamental_rng = _rng_for(cfg.base_seed, i, "fundamentals")
        quality_noise_rng = _rng_for(cfg.base_seed, i, "quality_noise")
        for t in range(cfg.n_periods_months):
            if t % cfg.fundamentals_update_every_n_months != (cfg.fundamentals_update_every_n_months - 1):
                continue
            period_end = dates[t]
            book_noise, eps_noise = fundamental_rng.normal(0.0, 0.05, size=2)
            book_value_per_share = float(book_scale[i] * (1 + 0.25 * value_z[i, t] + book_noise))
            earnings_per_share = float(eps_scale[i] * (1 + 0.25 * value_z[i, t] + eps_noise))

            # Stage 2 addition: drawn from quality_noise_rng, its OWN
            # independent per-security stream -- never from
            # fundamental_rng, which is reused across every reporting
            # period, so inserting draws into ITS sequence would shift
            # every later period's book_noise/eps_noise and silently
            # change Stage 1's already-pinned values. total_assets/
            # total_debt carry a real, planted relationship to the
            # latent quality_z -- higher quality is lower leverage --
            # for the Quality factor's recoverability test; they are
            # NOT part of the restatement scenario (both vintages below
            # share the same real total_assets/total_debt).
            asset_noise, debt_noise = quality_noise_rng.normal(0.0, 0.05, size=2)
            total_assets = max(1.0, float(asset_scale[i] * (1 + 0.3 * quality_z[i, t] + asset_noise)))
            leverage_ratio_true = float(np.clip(leverage_base[i] * (1 - 0.3 * quality_z[i, t]), 0.05, 0.95))
            total_debt = max(0.0, float(total_assets * leverage_ratio_true * (1 + debt_noise)))

            is_restated_period = (
                cfg.restatement_enabled and i == cfg.restatement_security_index and t == cfg.restatement_period_index
            )
            if is_restated_period:
                fundamental_rows.append(
                    {
                        "security_id": f"SYN{i:03d}",
                        "fundamental_period_end": period_end,
                        "known_as_of": period_end + pd.Timedelta(days=cfg.restatement_original_lag_days),
                        "book_value_per_share": book_value_per_share,
                        "earnings_per_share": earnings_per_share,
                        "total_assets": total_assets,
                        "total_debt": total_debt,
                        "quality_flags": "",
                    }
                )
                fundamental_rows.append(
                    {
                        "security_id": f"SYN{i:03d}",
                        "fundamental_period_end": period_end,
                        "known_as_of": period_end + pd.Timedelta(days=cfg.restatement_restated_lag_days),
                        "book_value_per_share": book_value_per_share * cfg.restatement_multiplier,
                        "earnings_per_share": earnings_per_share,
                        "total_assets": total_assets,
                        "total_debt": total_debt,
                        "quality_flags": "restated",
                    }
                )
            else:
                fundamental_rows.append(
                    {
                        "security_id": f"SYN{i:03d}",
                        "fundamental_period_end": period_end,
                        "known_as_of": period_end + pd.Timedelta(days=cfg.fundamental_publication_lag_days),
                        "book_value_per_share": book_value_per_share,
                        "earnings_per_share": earnings_per_share,
                        "total_assets": total_assets,
                        "total_debt": total_debt,
                        "quality_flags": "",
                    }
                )
    fundamentals_df = pd.DataFrame(fundamental_rows)

    ground_truth_rows = [
        {
            "security_id": f"SYN{i:03d}",
            "as_of": dates[t],
            "momentum_score": float(momentum_z[i, t]),
            "value_score": float(value_z[i, t]),
            "quality_score": float(quality_z[i, t]),
        }
        for i in range(cfg.n_securities)
        for t in range(cfg.n_periods_months)
    ]
    ground_truth_df = pd.DataFrame(ground_truth_rows)

    metadata = {
        "base_seed": cfg.base_seed,
        "n_securities": cfg.n_securities,
        "n_sectors": cfg.n_sectors,
        "n_periods_months": cfg.n_periods_months,
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
        "momentum_loading": cfg.momentum_loading,
        "value_loading": cfg.value_loading,
        "target_r_squared": cfg.target_r_squared,
        "restatement_security_id": f"SYN{cfg.restatement_security_index:03d}" if cfg.restatement_enabled else None,
        "restatement_period_end": str(dates[cfg.restatement_period_index].date()) if cfg.restatement_enabled else None,
    }

    return SyntheticDataset(market=market_df, fundamentals=fundamentals_df, ground_truth=ground_truth_df, metadata=metadata)


def market_observations_from_frame(df: pd.DataFrame) -> list[MarketObservation]:
    observations = []
    for row in df.itertuples(index=False):
        flags = tuple(row.quality_flags.split(";")) if row.quality_flags else ()
        observations.append(
            MarketObservation(
                security_id=row.security_id,
                ticker=row.ticker,
                as_of=pd.Timestamp(row.as_of),
                known_as_of=pd.Timestamp(row.known_as_of),
                sector=Sector(row.sector),
                is_active=bool(row.is_active),
                price=float(row.price),
                shares_outstanding=float(row.shares_outstanding),
                market_cap=float(row.market_cap),
                adv_20d=None if pd.isna(row.adv_20d) else float(row.adv_20d),
                quality_flags=flags,
            )
        )
    return observations


def fundamental_observations_from_frame(df: pd.DataFrame) -> list[FundamentalObservation]:
    observations = []
    for row in df.itertuples(index=False):
        flags = tuple(row.quality_flags.split(";")) if row.quality_flags else ()
        observations.append(
            FundamentalObservation(
                security_id=row.security_id,
                fundamental_period_end=pd.Timestamp(row.fundamental_period_end),
                known_as_of=pd.Timestamp(row.known_as_of),
                book_value_per_share=None if pd.isna(row.book_value_per_share) else float(row.book_value_per_share),
                earnings_per_share=None if pd.isna(row.earnings_per_share) else float(row.earnings_per_share),
                quality_flags=flags,
                total_assets=None if pd.isna(getattr(row, "total_assets", None)) else float(row.total_assets),
                total_debt=None if pd.isna(getattr(row, "total_debt", None)) else float(row.total_debt),
            )
        )
    return observations
