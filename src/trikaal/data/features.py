"""The per-bar feature transform (feature-spec §1/§3/§4/§5 + vol §3) — the *real transforms*.

``compute_features`` maps a raw stream to the model-ready output: the normalized 16-vector
``x``, the strictly-causal model-visible mask ``m``, ``segment_id``, the raw vol-hook channels,
the causal divisor ``sigma_t``, and the vol-scaled next-bar target. Every output for bar ``t``
is a pure function of raw data with effective timestamp ``<= t+1`` (the inputs/mask/segment/σ
read bars ``<= t``; the target *label* legitimately reads bar ``t+1`` but nothing beyond) — the
property the causal-safety harness enforces.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from trikaal.constants import (
    BOUNDED_IDX,
    EPS_CNT,
    EPS_OI,
    EPS_PX,
    EPS_SZ,
    EPS_VOL,
    FAST_ZSCORE_IDX,
    MASK_BITS_AGGTRADES,
    MASK_BITS_LIQUIDITY,
    N_FEATURES,
    Z_SCORED_IDX,
)
from trikaal.data.config import BAR_MS, FeatureConfig
from trikaal.data.normalize import causal_zscore_segment
from trikaal.data.quality import badtick_clip, stale_flag, vol_recon_err
from trikaal.data.segments import assign_segments, segment_bounds
from trikaal.data.synthetic import RawStream
from trikaal.data.targets import build_vol_scaled_targets
from trikaal.data.volatility import causal_vol_segment


@dataclass
class PerBarOutput:
    """Output of the per-bar transform. ``x_f64`` is the pre-cast value for bit-exact checks."""

    x: np.ndarray  # [T,16] float32 (model-visible)
    x_f64: np.ndarray  # [T,16] float64 (pre-cast)
    m: np.ndarray  # [T,16] uint8 (model-visible mask)
    segment_id: np.ndarray  # [T] int64
    ts: np.ndarray  # [T] int64 (bar_open_ms → temporal embeddings)
    raw: np.ndarray  # [T,2] float64 (raw_ret_close, raw_range) — vol hook
    sigma: np.ndarray  # [T] float64 (causal divisor at decision bar t)
    target: np.ndarray  # [T] float64 (y_{t→t+1}; NaN where invalid)
    target_valid: np.ndarray  # [T] bool
    is_stale: np.ndarray  # [T] uint8 (QA only)
    dq_flags: np.ndarray  # [T] uint16 (QA only)

    def __len__(self) -> int:
        return int(self.x.shape[0])


def _forward_fill(event_ts: np.ndarray, event_val: np.ndarray, close_ms: np.ndarray):
    """Causal forward-fill: for each bar return (value, missing) using events with ts ≤ close."""
    if event_ts.size == 0:
        miss = np.ones(close_ms.shape[0], dtype=bool)
        return np.zeros(close_ms.shape[0], dtype=np.float64), miss
    idx = np.searchsorted(event_ts, close_ms, side="right") - 1
    miss = idx < 0
    safe = np.clip(idx, 0, event_val.shape[0] - 1)
    val = np.where(miss, 0.0, event_val[safe]).astype(np.float64)
    return val, miss


def compute_features(stream: RawStream, cfg: FeatureConfig) -> PerBarOutput:
    """Run the full causal per-bar transform over a single-symbol raw stream."""
    T = len(stream)
    o = stream.open.astype(np.float64)
    c = stream.close.astype(np.float64)
    segment_id = assign_segments(stream.bar_open_ms, c, cfg)
    high_c, low_c, clip_bit = badtick_clip(
        o, stream.high, stream.low, c, segment_id, cfg, leak=cfg.planted_leak
    )

    close_ms = stream.bar_open_ms.astype(np.int64) + BAR_MS
    funding_ff, funding_miss = _forward_fill(stream.funding_ts, stream.funding_val, close_ms)
    oi_ff, oi_miss = _forward_fill(stream.oi_ts, stream.oi_val, close_ms)
    if not stream.is_perp:
        funding_ff[:] = 0.0
        oi_ff[:] = 0.0
        funding_miss[:] = True
        oi_miss[:] = True

    v_kline = stream.v_kline.astype(np.float64)
    quote_volume = stream.quote_volume.astype(np.float64)
    v_buy = stream.v_buy.astype(np.float64)
    v_sell = stream.v_sell.astype(np.float64)
    v_agg = v_buy + v_sell
    n_buy = stream.n_buy.astype(np.float64)
    n_sell = stream.n_sell.astype(np.float64)
    n_trades = n_buy + n_sell
    zero_vol = v_kline == 0.0

    # ---- raw (pre-normalization) feature columns ------------------------------------------
    raw_feat = np.zeros((T, N_FEATURES), dtype=np.float64)
    raw_ret = np.zeros(T, dtype=np.float64)
    raw_range = np.log(high_c / low_c)

    bounds = segment_bounds(segment_id)
    seg_start = np.zeros(T, dtype=bool)
    for s0, s1 in bounds:
        seg_start[s0] = True
        raw_ret[s0 + 1 : s1] = np.log(c[s0 + 1 : s1] / c[s0 : s1 - 1])

    rng_denom = (high_c - low_c) + EPS_PX
    raw_feat[:, 0] = raw_ret
    raw_feat[:, 1] = raw_range
    raw_feat[:, 2] = (c - o) / rng_denom
    raw_feat[:, 3] = (high_c - np.maximum(o, c)) / rng_denom
    raw_feat[:, 4] = (np.minimum(o, c) - low_c) / rng_denom
    raw_feat[:, 5] = np.log(v_kline + 1.0)
    raw_feat[:, 6] = np.log(quote_volume + 1.0)
    raw_feat[:, 7] = (v_buy - v_sell) / (v_agg + EPS_CNT)
    raw_feat[:, 8] = (n_buy - n_sell) / (n_buy + n_sell + EPS_CNT)
    raw_feat[:, 9] = np.log(n_trades + 1.0)
    raw_feat[:, 10] = np.log(v_agg / (n_trades + 1.0) + EPS_SZ)
    raw_feat[:, 11] = _dispersion(stream.sizes)
    raw_feat[:, 12] = _large_trade_share(stream.sizes, v_agg, segment_id, bounds, cfg)
    raw_feat[:, 13] = funding_ff
    raw_feat[:, 14] = np.log(oi_ff + 1.0)
    raw_feat[:, 15] = _d_oi(oi_ff, bounds)

    # ---- causal vol sigma_t (per segment), built from RAW ret_close -----------------------
    sigma = np.zeros(T, dtype=np.float64)
    vol_warm = np.ones(T, dtype=np.uint8)
    for s0, s1 in bounds:
        sig, warm = causal_vol_segment(
            raw_ret[s0:s1],
            lam=cfg.vol_lambda,
            n_warm_vol=cfg.effective_n_warm_vol(),
            estimator="rolling" if cfg.norm_estimator == "rolling" else "ewma",
            window=cfg.vol_window,
            leak=cfg.planted_leak,
        )
        sigma[s0:s1] = sig
        vol_warm[s0:s1] = warm

    # ---- planted SINGLE-BAR lookahead (test-only): sigma[leak_bar] reads the future label bar.
    # Guarded so it vanishes under truncation to leak_bar, which is exactly how the exhaustive
    # sweep catches it (a sparse anchor sampler that never draws leak_bar would not). ------------
    if cfg.planted_leak == "localized_sigma_next" and cfg.leak_bar is not None:
        lb = cfg.leak_bar
        if 0 <= lb < T - 1 and lb + 1 < T:
            sigma[lb] = sigma[lb] + abs(float(raw_ret[lb + 1]))

    # ---- optional input-path vol-scaling of ret_close / range (§3.3) ----------------------
    if cfg.vol_scale_inputs:
        raw_feat[:, 0] = raw_feat[:, 0] / (sigma + EPS_VOL)
        raw_feat[:, 1] = raw_feat[:, 1] / (sigma + EPS_VOL)

    # ---- assemble x + mask ----------------------------------------------------------------
    x_f64 = np.zeros((T, N_FEATURES), dtype=np.float64)
    m = np.zeros((T, N_FEATURES), dtype=np.uint8)

    # bounded features (§3.1): clip only — but an INFINITY is not a large value to be clamped.
    # ★ THE OTHER HALF OF THE _clip DEFECT. `np.clip` propagates NaN (which is why the z-scored
    # path's laundering was the visible half) but CLAMPS +/-inf to the bounds, so an infinity on
    # any of these seven dims arrived at the guard below as a finite +/-5 and the message
    # "NaN/Inf in emitted features" could only ever fire on the NaN half. `normalize._clip` now
    # propagates every non-finite input; this makes the bounded path agree, so the tripwire can
    # see an infinity on all sixteen dims rather than nine.
    for idx in BOUNDED_IDX:
        col = raw_feat[:, idx]
        x_f64[:, idx] = np.where(np.isfinite(col), np.clip(col, cfg.clip[0], cfg.clip[1]), np.nan)

    # z-scored features (§3.2): causal EWMA / rolling, per segment
    for idx in Z_SCORED_IDX:
        half = cfg.half_life_fast if idx in FAST_ZSCORE_IDX else cfg.half_life_slow
        nwarm = cfg.effective_n_warm(half)
        for s0, s1 in bounds:
            z, warm = causal_zscore_segment(
                raw_feat[s0:s1, idx],
                half_life=half,
                n_warm=nwarm,
                estimator=cfg.norm_estimator,
                rolling_window=cfg.rolling_window,
                leak=cfg.planted_leak if idx in (0, 1) else None,
            )
            x_f64[s0:s1, idx] = z
            m[s0:s1, idx] = np.maximum(m[s0:s1, idx], warm)

    # mask: segment-start (no t-1) → ret_close(0) & d_oi(15)
    m[seg_start, 0] = 1
    m[seg_start, 15] = 1
    # tau warm-up → large_trade_share(12); recomputed inside _large_trade_share via sentinel
    m[:, 12] = np.maximum(m[:, 12], _tau_warm(stream.sizes, segment_id, bounds, cfg))
    # zero-volume bar → liquidity bits 5..12
    for idx in MASK_BITS_LIQUIDITY:
        m[zero_vol, idx] = 1
    # vol_recon_err > 0.01 → aggTrades bits 7..12
    vre = vol_recon_err(v_agg, v_kline) > 0.01
    for idx in MASK_BITS_AGGTRADES:
        m[vre, idx] = 1
    # funding / OI missing → bits 13 / 14,15
    m[funding_miss, 13] = 1
    m[oi_miss, 14] = 1
    m[oi_miss, 15] = 1
    # input-path vol warm-up also masks ret_close/range if vol_scale_inputs
    if cfg.vol_scale_inputs:
        m[:, 0] = np.maximum(m[:, 0], vol_warm)
        m[:, 1] = np.maximum(m[:, 1], vol_warm)

    if not np.all(np.isfinite(x_f64)):
        raise ValueError("NaN/Inf in emitted features — an eps/segment rule was violated (§5.5)")

    x = x_f64.astype(np.float32)

    # ---- targets: y_{t→t+1} = raw_ret[t+1] / sigma_t (label reads bar t+1; divisor is causal)
    target, target_valid = build_vol_scaled_targets(raw_ret, sigma, vol_warm, bounds)

    # ---- planted SINGLE-BAR survivorship leak (test-only): the loss-inclusion flag at leak_bar
    # is dropped based on the EXISTENCE of a future bar (lb+2, beyond the legitimate t+1 label
    # horizon) — a textbook survivorship leak. Under truncation to lb+1 that bar is gone, so the
    # flag reverts to its causal value: the divergence is what the exhaustive sweep catches. ----
    if cfg.planted_leak == "localized_validity_next" and cfg.leak_bar is not None:
        lb = cfg.leak_bar
        if 0 <= lb and lb + 2 < T:
            target_valid[lb] = False

    # ---- QA-only columns (never fed to the model) -----------------------------------------
    is_stale = stale_flag(o, stream.high, stream.low, c, v_kline, cfg)
    dq_flags = np.zeros(T, dtype=np.uint16)
    dq_flags |= clip_bit.astype(np.uint16) << 0
    dq_flags |= is_stale.astype(np.uint16) << 1
    struct_start = seg_start & ~np.concatenate(([True], np.diff(stream.bar_open_ms) > cfg.gap_ms))
    dq_flags |= struct_start.astype(np.uint16) << 2
    dq_flags |= zero_vol.astype(np.uint16) << 4

    return PerBarOutput(
        x=x,
        x_f64=x_f64,
        m=m,
        segment_id=segment_id,
        ts=stream.bar_open_ms.astype(np.int64),
        raw=np.stack([raw_ret, raw_range], axis=1),
        sigma=sigma,
        target=target,
        target_valid=target_valid,
        is_stale=is_stale,
        dq_flags=dq_flags,
    )


def _dispersion(sizes: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(len(sizes), dtype=np.float64)
    for i, pool in enumerate(sizes):
        if pool.size >= 2:
            out[i] = float(np.std(pool, ddof=0)) / (float(np.mean(pool)) + EPS_SZ)
    return out


def _d_oi(oi_ff: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(oi_ff.shape[0], dtype=np.float64)
    for s0, s1 in bounds:
        if s1 - s0 >= 2:
            num = oi_ff[s0 + 1 : s1] + EPS_OI
            den = oi_ff[s0 : s1 - 1] + EPS_OI
            out[s0 + 1 : s1] = np.log(num / den)
    return out


def _subsample_sorted(arr: np.ndarray, cap: int) -> np.ndarray:
    """Deterministic distribution-preserving subsample: sorted, evenly-spaced order statistics.

    Recompute-identical (no RNG), so the causal-safety sweep still passes; preserves the empirical
    quantiles that ``tau`` needs (the spec's reservoir/sketch, §2.3)."""
    if arr.size <= cap:
        return arr
    idx = np.linspace(0, arr.size - 1, cap).round().astype(np.int64)
    return np.sort(arr)[idx]


def _large_trade_share(
    sizes: list[np.ndarray],
    v_agg: np.ndarray,
    segment_id: np.ndarray,
    bounds: list[tuple[int, int]],
    cfg: FeatureConfig,
) -> np.ndarray:
    out = np.zeros(len(sizes), dtype=np.float64)
    cap = cfg.tau_pool_cap
    for s0, s1 in bounds:
        window: deque[np.ndarray] = deque(maxlen=cfg.w_tau)  # past bars' (capped) size samples
        for p in range(s1 - s0):
            i = s0 + p
            if p >= cfg.w_tau and window:
                pool = np.concatenate(window) if len(window) > 1 else window[0]
                bar = sizes[i]  # numerator uses the bar's FULL sizes (not the capped pool)
                if pool.size and bar.size:  # all-empty window (e.g. a stale run) → large_share 0
                    tau = float(np.percentile(pool, cfg.q_tau))
                    out[i] = float(np.sum(bar >= tau)) / (float(v_agg[i]) + EPS_SZ)
            # fold bar i's sizes into the rolling window AFTER emitting (strictly causal)
            window.append(_subsample_sorted(sizes[i], cap) if sizes[i].size else sizes[i])
    return out


def _tau_warm(
    sizes: list[np.ndarray],
    segment_id: np.ndarray,
    bounds: list[tuple[int, int]],
    cfg: FeatureConfig,
) -> np.ndarray:
    warm = np.zeros(len(sizes), dtype=np.uint8)
    for s0, s1 in bounds:
        for p in range(s1 - s0):
            if p < cfg.w_tau:
                warm[s0 + p] = 1
    return warm
