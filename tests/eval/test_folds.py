"""Known-answer tests for purged walk-forward fold construction (§8.A)."""

from __future__ import annotations

import numpy as np

from trikaal.eval.folds import (
    EMBARGO_DEFAULT,
    H_MAX_DEFAULT,
    L_CORR_DEFAULT,
    make_fold_plan,
    quadrant,
)


def test_forward_blocks_partition():
    fp = make_fold_plan(1000, train_frac=0.7, k=6, seq_len=512)
    assert fp.train_end == 700
    assert fp.blocks == [(700, 750), (750, 800), (800, 850), (850, 900), (900, 950), (950, 1000)]


def test_purge_and_embargo_boundaries():
    fp = make_fold_plan(1000, train_frac=0.7, k=6, seq_len=512)  # embargo 120, h_max 60
    anchors = np.arange(700)
    valid = fp.valid_train_anchors(anchors)
    assert valid.max() == 579 and valid.size == 580  # t < 700 - 120
    purged_only = fp.purged_only_anchors(anchors)
    assert purged_only.max() == 639  # t + 60 < 700
    # the embargo strictly removes more than the purge alone (it binds)
    assert valid.size < purged_only.size


def test_block_anchors_selection():
    fp = make_fold_plan(1000, train_frac=0.7, k=6, seq_len=512)
    b0 = fp.block_anchors(np.arange(1000), 0)
    assert b0.min() == 700 and b0.max() == 749


def test_quadrant_labels():
    assert quadrant(True, True) == "Q1"
    assert quadrant(True, False) == "Q2"
    assert quadrant(False, True) == "Q3"
    assert quadrant(False, False) == "Q4"  # held-out symbols × future block = headline


def test_the_embargo_is_enforced_STRUCTURALLY_not_by_a_flatness_gate():
    """§7 v1.6.16 — REPLACES ``test_embargo_flatness_gate``, which was the ONLY caller of a
    function that never ran on real data. The leakage control that actually binds is
    ``fold_valid_starts``: a window is fold-legal only if its last bar opens strictly before
    ``boundary - E`` bars. That is asserted here, on every window, rather than on a hand-written
    dict of IRs that no run ever produced."""
    import numpy as np

    from trikaal.data.universe_loader import EMBARGO_BARS_DEFAULT, fold_valid_starts

    assert EMBARGO_BARS_DEFAULT == EMBARGO_DEFAULT == H_MAX_DEFAULT + L_CORR_DEFAULT == 120
    n, seq = 4000, 128
    ts = np.arange(n, dtype=np.int64) * 60_000
    seg = np.zeros(n, dtype=np.int64)
    boundary = int(ts[3000])
    starts = fold_valid_starts(ts, seg, seq_len=seq, boundary_ms=boundary)
    # DISCRIMINATION: the fixture must yield SOME legal windows, or the bound below is vacuous.
    assert starts.size > 0
    cutoff = boundary - EMBARGO_BARS_DEFAULT * 60_000
    assert ts[starts + seq - 1].max() < cutoff, "a window crosses into the embargo"
    # and a tighter embargo must admit STRICTLY MORE windows — proving E is actually read
    wider = fold_valid_starts(ts, seg, seq_len=seq, boundary_ms=boundary, embargo_bars=10)
    assert wider.size > starts.size
