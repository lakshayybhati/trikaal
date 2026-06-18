"""Shared fixtures: the tiny seeded synthetic stream and feature windows.

All tests run on synthetic data (no real Parquet), so the suite finishes in CI minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.data.config import synthetic_test_config
from trikaal.data.features import compute_features
from trikaal.data.synthetic import make_synthetic_stream


@pytest.fixture(scope="session")
def causal_cfg():
    """Small-window feature config for the fast causal-safety gate."""
    return synthetic_test_config(
        half_life_fast=48,
        half_life_slow=96,
        n_warm=12,
        w_tau=20,
        vol_window=24,
        n_warm_vol=12,
    )


@pytest.fixture(scope="session")
def causal_data(causal_cfg):
    """(stream, anom, cfg) — a small fixture sized for the truncation/perturbation harness."""
    stream, anom = make_synthetic_stream(seed=7, n_bars=1000)
    return stream, anom, causal_cfg


@pytest.fixture(scope="session")
def feature_window():
    """A dense [L, 16] feature tensor + mask drawn from one clean segment (for model gates).

    Returns ``(x, mask)`` as float32 / uint8 numpy arrays of shape ``[L, 16]`` / ``[L, 16]``,
    taken from the longest contiguous segment past warm-up so the vectors carry real signal.
    """
    cfg = synthetic_test_config(half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20)
    stream, _ = make_synthetic_stream(seed=11, n_bars=2400, inject_anomalies=True)
    out = compute_features(stream, cfg)
    # longest segment
    seg_ids, counts = np.unique(out.segment_id, return_counts=True)
    best = seg_ids[int(np.argmax(counts))]
    idx = np.nonzero(out.segment_id == best)[0]
    warm = 100  # skip z-score warm-up
    idx = idx[warm:]
    L = 256
    assert idx.size >= L, f"need >= {L} post-warm-up bars, got {idx.size}"
    sel = idx[:L]
    return out.x[sel].copy(), out.m[sel].copy()
