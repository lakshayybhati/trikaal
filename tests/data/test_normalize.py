"""Causal normalization unit tests (feature-spec §3.2)."""

from __future__ import annotations

import numpy as np

from trikaal.data.normalize import causal_zscore_segment


def test_ewma_is_strictly_causal():
    """z-scoring bar t must use only bars < t: truncating the series at t leaves z_t unchanged."""
    rng = np.random.default_rng(0)
    f = rng.standard_normal(400)
    z_full, _ = causal_zscore_segment(f, half_life=32, n_warm=8)
    for t in (50, 120, 250, 399):
        z_trunc, _ = causal_zscore_segment(f[: t + 1], half_life=32, n_warm=8)
        assert z_full[t] == z_trunc[t], f"z[{t}] changed when future was truncated — lookahead"


def test_warmup_emits_zero_and_flags():
    f = np.linspace(-2, 2, 100)
    z, warm = causal_zscore_segment(f, half_life=16, n_warm=20)
    assert np.all(warm[:20] == 1)
    assert np.all(z[:20] == 0.0)
    assert np.any(warm[20:] == 0)


def test_clip_bounds_are_respected():
    rng = np.random.default_rng(1)
    f = rng.standard_normal(500) * 50  # huge dispersion → many would exceed ±5
    z, _ = causal_zscore_segment(f, half_life=16, n_warm=8)
    assert z.min() >= -5.0 and z.max() <= 5.0


def test_rolling_matches_manual_reference():
    rng = np.random.default_rng(2)
    f = rng.standard_normal(120)
    z, _warm = causal_zscore_segment(
        f, half_life=0, n_warm=8, estimator="rolling", rolling_window=20
    )
    t = 60
    w = f[t - 20 : t]
    expect = (f[t] - w.mean()) / np.sqrt(w.var() + 1e-8)
    expect = min(5.0, max(-5.0, expect))
    assert abs(z[t] - expect) < 1e-9
