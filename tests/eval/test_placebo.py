"""Known-answer tests for the Cell-5 placebo surrogate machinery (§8.C.4)."""

from __future__ import annotations

import numpy as np

from trikaal.eval.placebo import (
    MICRO_DIMS,
    block_time_permute,
    phase_randomize,
    placebo_verdict,
)


def _toy(n=400):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 16))
    seg = np.zeros(n, dtype=np.int64)
    seg[200:] = 1  # two segments
    return x, seg


def test_block_permute_preserves_marginal_destroys_alignment():
    x, seg = _toy()
    surr = block_time_permute(x, seg, seed=0)
    cols = list(MICRO_DIMS)
    # OHLCV (and funding/OI) dims untouched
    others = [d for d in range(16) if d not in MICRO_DIMS]
    assert np.array_equal(surr[:, others], x[:, others])
    # micro block marginal preserved per segment (same multiset of rows), alignment destroyed
    for s0, s1 in ((0, 200), (200, 400)):
        a = np.sort(x[s0:s1][:, cols], axis=0)
        b = np.sort(surr[s0:s1][:, cols], axis=0)
        assert np.allclose(a, b)  # same values, permuted in time
        assert not np.allclose(surr[s0:s1][:, cols], x[s0:s1][:, cols])  # moved


def test_phase_randomize_preserves_power_spectrum():
    x, seg = _toy()
    surr = phase_randomize(x, seg, seed=0)
    for s0, s1 in ((0, 200), (200, 400)):
        for d in MICRO_DIMS:
            mag_x = np.abs(np.fft.rfft(x[s0:s1, d]))
            mag_s = np.abs(np.fft.rfft(surr[s0:s1, d]))
            assert np.allclose(mag_x, mag_s, atol=1e-8)  # power spectrum (autocorr) preserved
    others = [d for d in range(16) if d not in MICRO_DIMS]
    assert np.array_equal(surr[:, others], x[:, others])


def test_placebo_verdict_decision_rule():
    info = placebo_verdict(ir_cell2=0.5, ir_cell4=1.0, ir_cell5=0.6)  # micro gain >> capacity gain
    assert info["micro_exceeds_placebo"] and np.isclose(info["delta_micro"], 0.5)
    capacity = placebo_verdict(
        ir_cell2=0.5, ir_cell4=1.0, ir_cell5=1.1
    )  # placebo matches → capacity
    assert not capacity["micro_exceeds_placebo"]
