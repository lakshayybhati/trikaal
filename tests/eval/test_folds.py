"""Known-answer tests for purged walk-forward fold construction (§8.A)."""

from __future__ import annotations

import numpy as np

from trikaal.eval.folds import embargo_flatness, make_fold_plan, quadrant


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


def test_embargo_flatness_gate():
    leaky_at_60 = embargo_flatness({60: 1.0, 120: 0.5, 240: 0.48})
    assert leaky_at_60["leak_suspected_at_60"] and leaky_at_60["flat_past_120"]
    still_leaking = embargo_flatness({60: 0.5, 120: 0.4, 240: 0.2})
    assert not still_leaking["flat_past_120"]  # IR keeps dropping past 120 → still leaks
