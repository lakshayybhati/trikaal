"""Known-answer tests for the Cell-5 placebo surrogate machinery (§8.C.4).

2026-07-06 audit fix coverage: the block permutation must move each bar's (value, mask) pair
TOGETHER — values-only shuffling strands fill values on mask-active bars (and real values on
masked bars), a one-directional Cell-5 handicap that could fake a positive ΔIR(Cell4−Cell5).
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.eval.placebo import (
    MICRO_DIMS,
    block_time_permute,
    phase_randomize,
    placebo_verdict,
)

FILL = 999.0  # sentinel fill value for masked micro entries in the toy fixtures


def _toy(n=400, holes=False, hole_seed=1):
    """x[n,16] + mask[n,16] (funding/OI masked per the build invariant) + 2-segment ids.

    ``holes=True`` sprinkles ~25% masked micro entries whose x is the FILL sentinel, so a
    (value, mask) misalignment is directly observable."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 16))
    m = np.zeros((n, 16), dtype=np.uint8)
    m[:, 13:16] = 1  # funding/OI masked — the real-build invariant
    seg = np.zeros(n, dtype=np.int64)
    seg[n // 2 :] = 1  # two segments
    if holes:
        hrng = np.random.default_rng(hole_seed)
        for d in MICRO_DIMS:
            hole = hrng.random(n) < 0.25
            m[hole, d] = 1
            x[hole, d] = FILL
    return x, m, seg


def test_block_permute_preserves_marginal_destroys_alignment():
    x, m, seg = _toy()
    surr, m_surr = block_time_permute(x, m, seg, seed=0)
    cols = list(MICRO_DIMS)
    # OHLCV (and funding/OI) dims untouched — values AND mask
    others = [d for d in range(16) if d not in MICRO_DIMS]
    assert np.array_equal(surr[:, others], x[:, others])
    assert np.array_equal(m_surr[:, others], m[:, others])
    # micro block marginal preserved per segment (same multiset of rows), alignment destroyed
    for s0, s1 in ((0, 200), (200, 400)):
        a = np.sort(x[s0:s1][:, cols], axis=0)
        b = np.sort(surr[s0:s1][:, cols], axis=0)
        assert np.allclose(a, b)  # same values, permuted in time
        assert not np.allclose(surr[s0:s1][:, cols], x[s0:s1][:, cols])  # moved


def test_block_permute_moves_value_mask_pairs_together():
    """KAT (a): one per-segment permutation moves the whole micro block — values and mask bits
    travel together, the permutation never crosses a segment, mask marginals are preserved."""
    n = 300
    seg = np.zeros(n, dtype=np.int64)
    seg[150:] = 1
    x = np.zeros((n, 16))
    for d in range(16):
        x[:, d] = np.arange(n) * 100.0 + d  # unique + decodable: source row = value // 100
    m = (np.random.default_rng(3).random((n, 16)) < 0.3).astype(np.uint8)
    m[:, 13:16] = 1
    xs, ms = block_time_permute(x, m, seg, seed=7)
    cols = list(MICRO_DIMS)
    for s0, s1 in ((0, 150), (150, 300)):
        src = (xs[s0:s1][:, cols] // 100).astype(int)  # source row per (t, micro-dim)
        # the micro block moves as ONE unit: every micro dim of row t came from the same row
        assert (src == src[:, :1]).all()
        # the permutation stays inside its segment
        assert src.min() >= s0 and src.max() < s1
        # the mask bit traveled with its value
        for j, d in enumerate(cols):
            np.testing.assert_array_equal(ms[s0:s1, d], m[src[:, j], d])
        # mask marginals preserved per segment/dim
        np.testing.assert_array_equal(ms[s0:s1][:, cols].sum(axis=0), m[s0:s1][:, cols].sum(axis=0))


def test_block_permute_fill_value_never_lands_on_active_bar():
    """KAT (b): after the joint shuffle, (value == FILL) ⟺ (mask == masked) everywhere — and
    the pre-fix behavior (values moved, mask left in place) demonstrably violates it."""
    x, m, seg = _toy(holes=True)
    xs, ms = block_time_permute(x, m, seg, seed=0)
    cols = list(MICRO_DIMS)
    is_fill = xs[:, cols] == FILL
    is_masked = ms[:, cols] == 1
    np.testing.assert_array_equal(is_fill, is_masked)  # pairs intact everywhere
    # regression pin: against the UNMOVED mask (the old bug), fills land on "active" bars
    assert not np.array_equal(is_fill, m[:, cols] == 1)


def test_block_permute_rejects_mismatched_mask():
    x, m, seg = _toy()
    with pytest.raises(ValueError, match="mask shape"):
        block_time_permute(x, m[:, :7], seg, seed=0)


def test_phase_randomize_preserves_power_spectrum():
    x, _m, seg = _toy()
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
