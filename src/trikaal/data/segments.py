"""Contiguous-segment assignment (feature-spec §4.1 / §5.4).

A segment boundary is a **price gap** (a missing 1m grid timestamp) OR a strictly-causal
**structural break** (a ``>= P`` single-bar price move known at close ``t``). Both are decided
from bar ``t`` and ``t-1`` alone — no forward-reversion confirmation — so whether bar ``t``
starts a new segment depends only on data ``<= t+1``. Windows never straddle a boundary; all
stateful estimators reset at one.
"""

from __future__ import annotations

import numpy as np

from trikaal.data.config import FeatureConfig


def assign_segments(
    bar_open_ms: np.ndarray, close: np.ndarray, cfg: FeatureConfig
) -> np.ndarray:
    """Return ``segment_id`` [T] int64. Strictly causal: bar ``i`` reads only ``i`` and ``i-1``."""
    n = bar_open_ms.shape[0]
    seg = np.zeros(n, dtype=np.int64)
    cur = 0
    for i in range(1, n):
        gap = (int(bar_open_ms[i]) - int(bar_open_ms[i - 1])) > cfg.gap_ms
        # structural break: single-bar simple-return magnitude >= P (no forward reversion check)
        prev = float(close[i - 1])
        move = abs(float(close[i]) / prev - 1.0) if prev != 0.0 else 0.0
        struct = move >= cfg.structural_break_pct
        if gap or struct:
            cur += 1
        seg[i] = cur
    return seg


def segment_bounds(segment_id: np.ndarray) -> list[tuple[int, int]]:
    """List of ``(start, end_exclusive)`` index ranges, one per contiguous segment."""
    bounds: list[tuple[int, int]] = []
    n = segment_id.shape[0]
    if n == 0:
        return bounds
    start = 0
    for i in range(1, n):
        if segment_id[i] != segment_id[i - 1]:
            bounds.append((start, i))
            start = i
    bounds.append((start, n))
    return bounds
