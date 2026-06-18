"""Vol-scaled, regime-relative target builder (vol-scaling §4).

The single-horizon target for decision bar ``t`` is the vol-scaled next-bar return
``y_{t→t+1} = ret_close_{t+1} / (sigma_t + eps_vol)``. The numerator is the *label* (bar
``t+1``, the bar being predicted); the divisor ``sigma_t`` is the strictly-causal vol known at
decision time ``t``. This is the one place a future-bar quantity legitimately appears — as a
label, never as an input or divisor. Targets whose decision bar is in vol warm-up are excluded.
"""

from __future__ import annotations

import numpy as np

from trikaal.constants import EPS_VOL


def build_vol_scaled_targets(
    raw_ret: np.ndarray,
    sigma: np.ndarray,
    vol_warm: np.ndarray,
    bounds: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(target, target_valid)`` ([T] float64 with NaN where invalid, [T] bool).

    ``target[t]`` is valid iff ``t+1`` is in the same segment and ``sigma_t`` is past warm-up.
    """
    T = raw_ret.shape[0]
    target = np.full(T, np.nan, dtype=np.float64)
    valid = np.zeros(T, dtype=bool)
    for s0, s1 in bounds:
        for t in range(s0, s1 - 1):
            if vol_warm[t]:
                continue
            target[t] = float(raw_ret[t + 1]) / (float(sigma[t]) + EPS_VOL)
            valid[t] = True
    return target, valid
