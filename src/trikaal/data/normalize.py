"""Causal per-feature normalization (feature-spec §3.2).

EWMA (West incremental variance) is the default: bar ``t`` is normalized with
``(mu_{t-1}, var_{t-1})`` — stats accumulated only over bars ``< t`` within the segment —
and *then* the running stats fold in ``f_t``. This is strictly causal by construction.

Two planted-leak variants (``centered``/``global``) deliberately read future bars; they
exist ONLY so the causal-safety harness can prove it catches a leak. They are never used in
any real run.
"""

from __future__ import annotations

import math

import numpy as np

from trikaal.constants import CLIP_HI, CLIP_LO, EPS_VAR


def causal_zscore_segment(
    f: np.ndarray,
    *,
    half_life: int,
    n_warm: int,
    estimator: str = "ewma",
    rolling_window: int = 1440,
    leak: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one feature's per-segment series. Returns ``(z, warm_mask)`` (both [n]).

    ``warm_mask[i] == 1`` means bar ``i`` is in z-score warm-up (emit ``z = 0`` + mask bit).
    All arithmetic in float64; the final ``[-5, 5]`` clip is applied to live bars.
    """
    n = f.shape[0]
    z = np.zeros(n, dtype=np.float64)
    warm = np.ones(n, dtype=np.uint8)

    if leak == "global_zscore":
        # LEAK: uses whole-segment mean/std (reads the future) — must fail the invariant.
        mu = float(np.mean(f))
        sd = float(np.std(f))
        for i in range(n):
            warm[i] = 1 if i < n_warm else 0
            if not warm[i]:
                z[i] = _clip((f[i] - mu) / np.sqrt(sd * sd + EPS_VAR))
        return z, warm

    if leak == "centered_zscore":
        # LEAK: two-sided window centered on bar i (reads future bars) — must fail.
        half = max(1, rolling_window // 2)
        for i in range(n):
            warm[i] = 1 if i < n_warm else 0
            if not warm[i]:
                lo, hi = max(0, i - half), min(n, i + half + 1)
                w = f[lo:hi]
                z[i] = _clip((f[i] - w.mean()) / np.sqrt(w.var() + EPS_VAR))
        return z, warm

    if estimator == "rolling":
        for i in range(n):
            warm[i] = 1 if i < max(n_warm, 1) else 0
            if not warm[i]:
                lo = max(0, i - rolling_window)
                w = f[lo:i]  # strictly past (excludes i)
                if w.size == 0:
                    warm[i] = 1
                    continue
                z[i] = _clip((f[i] - w.mean()) / np.sqrt(w.var() + EPS_VAR))
        return z, warm

    # default: EWMA (West recursion), strictly causal
    alpha = 1.0 - 2.0 ** (-1.0 / half_life)
    mu = 0.0
    var = 0.0
    for i in range(n):
        if i == 0:
            mu = float(f[0])  # seed on the first in-segment bar (depends only on bar 0)
            var = 0.0
            warm[0] = 1
            continue
        warm[i] = 1 if i < n_warm else 0
        if not warm[i]:
            z[i] = _clip((f[i] - mu) / np.sqrt(var + EPS_VAR))
        # advance state with f[i] AFTER emitting (so f[i] never normalizes itself)
        delta = float(f[i]) - mu
        mu = mu + alpha * delta
        var = (1.0 - alpha) * (var + alpha * delta * delta)
        var = max(var, 0.0)  # West recursion is non-negative analytically; clamp fp drift
    return z, warm


def _clip(x: float) -> float:
    """Clamp to ``[CLIP_LO, CLIP_HI]``, but PROPAGATE non-finite input instead of clamping it.

    ★ THE DEFECT THIS FIXES, AND IT DISABLED A TRIPWIRE RATHER THAN MERELY MISCOMPUTING. Every
    comparison against NaN is False, so ``max(CLIP_LO, nan)`` returns ``CLIP_LO`` and ``min`` then
    keeps it: a NaN became **-5.0**, the most extreme negative value the feature can take. That is
    already wrong, but the worse half is what it did to ``features.compute_features``, which ends
    with ``if not np.all(np.isfinite(x_f64)): raise ValueError("NaN/Inf in emitted features")``.
    That guard exists precisely to catch this, and it CANNOT FIRE against a laundered value —
    -5.0 is finite. A violated eps/segment rule would have been absorbed into the data path,
    silently, on nine of sixteen dims including the return channel.

    ``np.clip`` — which the BOUNDED dims use — already propagates NaN, so this makes the z-scored
    path agree with the path beside it rather than inventing a new rule.
    """
    if not math.isfinite(x):
        return math.nan
    return min(CLIP_HI, max(CLIP_LO, x))
