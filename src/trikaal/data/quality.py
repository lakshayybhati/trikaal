"""Data-quality gates (feature-spec §5) — strictly causal, batch == live.

Each gate's verdict and its effect on bar ``t`` are decided from data with effective
timestamp ``<= t+1`` only; no gate performs a forward-revert rewrite of an already-emitted
bar. The bad-tick clip touches only the wicks (``H``/``L``); ``O``/``C`` are never rewritten,
so the ``ret_close`` reference for ``t+1`` is untouched. ``is_stale`` is a strictly-trailing
QA-only flag and is never OR-ed into the model-visible mask ``m_t`` (§4.5).

The ``forward_revert_badtick`` planted leak reintroduces the removed "clip only if it reverts
within R bars" rule (reads ``C_{t+R}``) so the harness can prove it is caught.
"""

from __future__ import annotations

import numpy as np

from trikaal.constants import EPS_SZ
from trikaal.data.config import FeatureConfig
from trikaal.data.segments import segment_bounds


def badtick_clip(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    segment_id: np.ndarray,
    cfg: FeatureConfig,
    *,
    leak: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(high_clipped, low_clipped, clip_bit)``. Clips implausible wicks only.

    A wick is implausible iff its fractional size exceeds ``k * trailing_return_MAD`` where the
    MAD is taken over the strictly-trailing in-segment window ``[i-W, i-1]``. ``O``/``C`` are
    never modified.
    """
    n = open_.shape[0]
    high_c = high.astype(np.float64).copy()
    low_c = low.astype(np.float64).copy()
    clip_bit = np.zeros(n, dtype=np.uint8)
    min_ref = 8  # need at least this many trailing returns for a stable MAD

    for s0, s1 in segment_bounds(segment_id):
        c = close[s0:s1].astype(np.float64)
        o = open_[s0:s1].astype(np.float64)
        m = c.shape[0]
        # in-segment close-to-close returns; r[0] undefined -> 0
        r = np.zeros(m, dtype=np.float64)
        r[1:] = np.log(c[1:] / c[:-1])
        for p in range(m):
            lo = max(0, p - cfg.badtick_mad_window)
            past = r[lo:p]  # strictly past within segment
            if past.size < min_ref:
                continue
            med = np.median(past)
            mad = np.median(np.abs(past - med))
            if mad <= 0.0:
                continue
            thr = cfg.badtick_k * mad
            i = s0 + p
            up_ref = max(o[p], c[p])
            dn_ref = min(o[p], c[p])
            up_exc = (float(high[i]) - up_ref) / up_ref if up_ref > 0 else 0.0
            dn_exc = (dn_ref - float(low[i])) / dn_ref if dn_ref > 0 else 0.0

            do_clip_up = up_exc > thr
            do_clip_dn = dn_exc > thr
            if leak == "forward_revert_badtick":
                # LEAK: only clip if the move reverts within R bars (reads C_{i+R}).
                r_fwd = 3
                reverted = False
                if i + r_fwd < n:
                    reverted = abs(float(close[i + r_fwd]) / float(close[i]) - 1.0) < thr
                do_clip_up = do_clip_up and reverted
                do_clip_dn = do_clip_dn and reverted

            if do_clip_up:
                high_c[i] = up_ref * (1.0 + thr)
                clip_bit[i] = 1
            if do_clip_dn:
                low_c[i] = dn_ref * (1.0 - thr)
                clip_bit[i] = 1
    return high_c, low_c, clip_bit


def stale_flag(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    v_kline: np.ndarray,
    cfg: FeatureConfig,
) -> np.ndarray:
    """``is_stale`` [T] uint8 — strictly trailing constant-price-run membership (§5.3). QA-only."""
    n = open_.shape[0]
    is_stale = np.zeros(n, dtype=np.uint8)
    s = cfg.stale_run
    for i in range(s - 1, n):
        w = slice(i - s + 1, i + 1)
        flat = (
            np.all(open_[w] == high[w])
            and np.all(high[w] == low[w])
            and np.all(low[w] == close[w])
            and np.all(close[w] == close[i])
        )
        zero_vol = np.all(v_kline[w] == 0.0)
        if flat and zero_vol:
            is_stale[i] = 1
    return is_stale


def vol_recon_err(v_agg: np.ndarray, v_kline: np.ndarray) -> np.ndarray:
    """QA scalar ``|V_agg - V_kline| / (V_kline + eps)`` (§2.2). Not a model input."""
    return np.abs(v_agg - v_kline) / (v_kline + EPS_SZ)
