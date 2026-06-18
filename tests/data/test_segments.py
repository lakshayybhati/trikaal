"""Segment-assignment unit tests (feature-spec §4.1 / §5.4)."""

from __future__ import annotations

import numpy as np

from trikaal.data.config import BAR_MS, synthetic_test_config
from trikaal.data.segments import assign_segments, segment_bounds


def test_gap_starts_new_segment():
    cfg = synthetic_test_config()
    bar_open = np.array([0, 1, 2, 3, 10, 11, 12], dtype=np.int64) * BAR_MS  # gap after index 3
    close = np.full(7, 100.0)
    seg = assign_segments(bar_open, close, cfg)
    assert seg.tolist() == [0, 0, 0, 0, 1, 1, 1]


def test_structural_break_splits_on_single_bar_jump():
    cfg = synthetic_test_config()
    bar_open = np.arange(6, dtype=np.int64) * BAR_MS
    close = np.array([100.0, 101.0, 102.0, 180.0, 181.0, 182.0])  # +76% jump at index 3
    seg = assign_segments(bar_open, close, cfg)
    assert seg.tolist() == [0, 0, 0, 1, 1, 1]


def test_small_move_does_not_split():
    cfg = synthetic_test_config()
    bar_open = np.arange(5, dtype=np.int64) * BAR_MS
    close = np.array([100.0, 105.0, 110.0, 115.0, 120.0])  # all < 50% moves
    seg = assign_segments(bar_open, close, cfg)
    assert seg.tolist() == [0, 0, 0, 0, 0]


def test_segment_bounds():
    seg = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    assert segment_bounds(seg) == [(0, 2), (2, 5), (5, 6)]


def test_segmentation_is_causal_bar_t_only():
    """Whether bar t starts a segment depends only on bars t and t-1, never the future."""
    cfg = synthetic_test_config()
    bar_open = np.arange(10, dtype=np.int64) * BAR_MS
    close = np.array([100.0, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    seg_full = assign_segments(bar_open, close, cfg)
    # mutate a future bar; segment ids at/before t must not change
    for t in range(1, 9):
        close2 = close.copy()
        close2[t + 1 :] = close2[t + 1 :] * 3.0
        seg2 = assign_segments(bar_open, close2, cfg)
        assert seg_full[: t + 1].tolist() == seg2[: t + 1].tolist()
