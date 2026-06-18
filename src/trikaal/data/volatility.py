"""Causal current-regime volatility ``sigma_t`` (vol-scaling §1).

``sigma_t`` is the divisor of the prediction target. It is made at decision time ``t`` and
uses ONLY bars with effective timestamp ``<= t+1`` — i.e. it may read ``r_t`` (the return
into the close of bar ``t``) but never ``r_{t+1}``. The EWMA realized-variance recursion
folds ``r_t`` in *before* emitting ``sigma_t``; that is correct (it normalizes a target about
bar ``t+1``, so conditioning on ``r_t`` is legitimate, not leakage).

The ``sigma_includes_next`` planted leak shifts the recursion to read ``r_{t+1}`` — the
classic off-by-one that includes the predicted bar in its own scaler — so the harness can
prove it is caught at boundary ``t``.
"""

from __future__ import annotations

import numpy as np

from trikaal.constants import EPS_VOL


def causal_vol_segment(
    r: np.ndarray,
    *,
    lam: float,
    n_warm_vol: int,
    estimator: str = "ewma",
    window: int = 60,
    leak: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RiskMetrics-style realized vol per segment. Returns ``(sigma, warm_mask)`` ([n] each).

    ``r`` is the raw (un-z-scored) ``ret_close`` series of one segment (``r[0] == 0`` at the
    segment start, no ``t-1``). ``warm_mask[i] == 1`` ⇒ ``sigma_i`` is unreliable and any
    target whose decision bar is ``i`` is excluded from the loss (§1 warm-up).
    """
    n = r.shape[0]
    sigma = np.zeros(n, dtype=np.float64)
    warm = np.ones(n, dtype=np.uint8)

    if estimator == "rolling":
        for i in range(n):
            warm[i] = 1 if i < max(n_warm_vol, 1) else 0
            lo = max(0, i - window + 1)
            w = r[lo : i + 1]  # past-inclusive (uses r_i, never r_{i+1})
            sigma[i] = np.sqrt(np.mean(w * w) + EPS_VOL)
        return sigma, warm

    sigma_sq = 0.0
    for i in range(n):
        if leak == "sigma_includes_next":
            r_used = float(r[i + 1]) if i + 1 < n else float(r[i])  # LEAK: reads bar i+1
        else:
            r_used = float(r[i])  # causal: r_i known at close of bar i
        sigma_sq = lam * sigma_sq + (1.0 - lam) * r_used * r_used
        sigma[i] = np.sqrt(sigma_sq + EPS_VOL)
        warm[i] = 1 if i < n_warm_vol else 0
    return sigma, warm
