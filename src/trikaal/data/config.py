"""FeatureConfig — every knob of the causal feature transform, with spec defaults.

The transform code is parameterised by this config; the synthetic-slice tests run the
*same* code with a small-fixture config (shorter half-lives/windows) so a few-thousand-bar
fixture has plenty of post-warm-up bars. The lookahead invariant is independent of these
constants — it holds for any half-life — so testing with small windows is faithful.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from trikaal.constants import CLIP_HI, CLIP_LO, FSQ_LEVELS

BAR_MS: int = 60_000  # 1-minute bar width in milliseconds


@dataclass(frozen=True)
class FeatureConfig:
    """Causal feature-transform configuration (feature-spec §1/§3/§4/§5 + vol §3)."""

    # --- normalization (§3.2) ---
    norm_estimator: str = "ewma"  # "ewma" | "rolling"
    half_life_fast: int = 1440  # ret_close, range, trade_count, mean_trade_size, dispersion
    half_life_slow: int = 5760  # log_volume, log_amount, log_oi, d_oi
    rolling_window: int = 1440  # for norm_estimator == "rolling"
    n_warm: int = 64  # z-score warm-up floor; effective = max(n_warm, H/8)
    clip: tuple[float, float] = (CLIP_LO, CLIP_HI)

    # --- aggTrades / tau (§2.3) ---
    w_tau: int = 1440  # rolling-past window for the large-trade percentile
    q_tau: float = 90.0  # percentile defining a "large" trade

    # --- data-quality gates (§5) ---
    badtick_k: float = 12.0  # MAD multiple flagging a bad-tick wick
    badtick_mad_window: int = 1440  # strictly-trailing MAD window
    stale_run: int = 30  # S: trailing bars of flat+zero-vol -> is_stale (QA only)
    structural_break_pct: float = 0.50  # P: single-bar |return| splitting a segment
    gap_ms: int = BAR_MS  # a bar_open gap > this many ms is a price gap (segment boundary)
    min_segment_len: int = 128  # L_min: segments shorter than this are unusable for training

    # --- vol-scaling (§3) ---
    vol_lambda: float = 0.97  # EWMA realized-variance decay (RiskMetrics)
    vol_window: int = 60  # rolling realized-vol reference window
    n_warm_vol: int = 30  # vol warm-up floor; effective = max(n_warm_vol, half-life)
    vol_scale_inputs: bool = False  # §3.3: optional; default off (minimal-coupling)
    vol_scale_targets: bool = True  # project contract: always on

    # --- regularizers (train-only; off in the causal path) ---
    vol_amount_dropout_p: float = 0.05  # Kronos §4.4 (not applied in the slice transforms)

    # --- tokenizer FSQ (referenced; owned by Tokenizer §c) ---
    fsq_levels: tuple[int, ...] = FSQ_LEVELS

    # --- a deliberately-planted lookahead, for proving the harness has teeth ---
    # When set, the named transform peeks at future bars. Used ONLY by the causal-safety
    # test to demonstrate the invariant catches a leak; never enabled in any real run.
    planted_leak: str | None = None  # None | "centered_zscore" | "sigma_includes_next"
    #                                  | "forward_revert_badtick" | "global_zscore"

    def effective_n_warm(self, half_life: int) -> int:
        return max(self.n_warm, half_life // 8)

    def effective_n_warm_vol(self) -> int:
        return max(self.n_warm_vol, self.half_life_fast)

    def with_leak(self, leak: str | None) -> FeatureConfig:
        return replace(self, planted_leak=leak)


def synthetic_test_config(**overrides: object) -> FeatureConfig:
    """Small-window config for the synthetic slice (keeps fixtures small, exercises live state)."""
    base = FeatureConfig(
        half_life_fast=64,
        half_life_slow=128,
        rolling_window=64,
        n_warm=16,
        w_tau=64,
        q_tau=90.0,
        badtick_mad_window=64,
        stale_run=10,
        min_segment_len=32,
        vol_lambda=0.94,
        vol_window=32,
        n_warm_vol=16,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


__all__ = ["BAR_MS", "FeatureConfig", "synthetic_test_config"]
