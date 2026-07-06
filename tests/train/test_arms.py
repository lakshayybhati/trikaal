"""KATs for the feature-arm switch + the Cell-5 shuffled-micro training arm (§6 items 2-3).

Item 2: the OHLCV-only arm REMOVES the micro dims (F=7 input; never zero-fill) and the tokenizer
trains cleanly at n_features=7; the arm's output is INVARIANT to micro-channel perturbations
(the leak-check). Item 3: the Cell-5 shuffle is deterministic under (seed, symbol), preserves
each channel's marginal within a segment, destroys temporal alignment, and carries the
funding/OI tripwire in the TRAINING path too."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trikaal.constants import OHLCV_ONLY_IDX
from trikaal.eval.placebo import MICRO_DIMS
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import (
    ARM_MICRO,
    ARM_MICRO_SHUFFLED,
    ARM_OHLCV,
    arm_feature_idx,
    arm_n_features,
    cell5_seed,
    select_arm,
    shuffle_micro,
)
from trikaal.utils.seeding import set_determinism


def _xm(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 16)).astype(np.float32)
    m = np.zeros((n, 16), dtype=np.uint8)
    m[:, 13:16] = 1  # funding/OI masked — the real-build invariant
    return x, m


# --------------------------------------------------------- item 2: the arm switch
def test_arm_shapes_are_removals_not_zero_fill():
    x, m = _xm()
    xa, ma = select_arm(x, m, ARM_OHLCV)
    assert xa.shape == (240, 7) and ma.shape == (240, 7)  # REMOVED, not zero-filled
    np.testing.assert_array_equal(xa, x[:, list(OHLCV_ONLY_IDX)])
    xb, _ = select_arm(x, m, ARM_MICRO)
    assert xb.shape == (240, 16)
    assert arm_n_features(ARM_OHLCV) == 7 and arm_n_features(ARM_MICRO) == 16
    with pytest.raises(ValueError, match="unknown arm"):
        arm_feature_idx("bogus")


def test_ohlcv_arm_output_invariant_to_micro_perturbation():
    """The leak-check: nothing the OHLCV-only cell computes may depend on micro channels."""
    set_determinism(0)
    tok = TokenizerAE(n_features=7, d_model=32, n_layers=1, n_heads=2, d_ff=64).eval()
    x, m = _xm()
    x_pert = x.copy()
    x_pert[:, list(MICRO_DIMS)] += np.random.default_rng(1).standard_normal((240, 6)) * 100.0
    xa, ma = select_arm(x, m, ARM_OHLCV)
    xp, mp = select_arm(x_pert, m, ARM_OHLCV)
    np.testing.assert_array_equal(xa, xp)  # the arm never sees micro
    with torch.no_grad():
        out_a = tok(torch.from_numpy(xa[None]), torch.from_numpy(ma[None]))
        out_p = tok(torch.from_numpy(xp[None]), torch.from_numpy(mp[None]))
    assert torch.equal(out_a["x_hat"], out_p["x_hat"])
    b1 = tok.encode_tokens(torch.from_numpy(xa[None]), torch.from_numpy(ma[None]))
    b2 = tok.encode_tokens(torch.from_numpy(xp[None]), torch.from_numpy(mp[None]))
    assert torch.equal(b1[0], b2[0]) and torch.equal(b1[1], b2[1])


def test_f7_tokenizer_trains_cleanly():
    """TokenizerAE(n_features=7): forward + backward run, loss finite, grads flow, loss drops."""
    set_determinism(0)
    tok = TokenizerAE(n_features=7, d_model=32, n_layers=1, n_heads=2, d_ff=64)
    x, m = _xm(480)
    xa, ma = select_arm(x, m, ARM_OHLCV)
    xt = torch.from_numpy(xa[None])
    mt = torch.from_numpy(ma[None])
    opt = torch.optim.AdamW(tok.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        out = tok(xt, mt)
        loss = out["loss"]
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]  # trains, not just runs


# --------------------------------------------------------- item 3: the Cell-5 training arm
def _punch_micro_holes(x, m, frac=0.25, seed=5):
    """Mask ~frac of micro entries so the (value, mask) pairing is observable in the shuffle."""
    rng = np.random.default_rng(seed)
    for d in MICRO_DIMS:
        hole = rng.random(x.shape[0]) < frac
        m[hole, d] = 1
    return x, m


def test_cell5_shuffle_deterministic_and_symbol_seeded():
    """KAT (c): determinism under (seed, symbol) — for the values AND the co-shuffled mask."""
    x, m = _punch_micro_holes(*_xm())
    seg = np.zeros(240, dtype=np.int64)
    seg[120:] = 1  # two segments
    x1, m1 = shuffle_micro(x, m, seg, symbol="BTCUSDT", seed=0)
    x2, m2 = shuffle_micro(x, m, seg, symbol="BTCUSDT", seed=0)
    np.testing.assert_array_equal(x1, x2)  # deterministic under (seed, symbol)
    np.testing.assert_array_equal(m1, m2)  # ... including the mask
    x3, m3 = shuffle_micro(x, m, seg, symbol="ETHUSDT", seed=0)
    x4, m4 = shuffle_micro(x, m, seg, symbol="BTCUSDT", seed=1)
    assert not np.array_equal(x1, x3) and not np.array_equal(x1, x4)
    assert not np.array_equal(m1, m3) and not np.array_equal(m1, m4)
    assert cell5_seed(0, "BTCUSDT") != cell5_seed(0, "ETHUSDT") != cell5_seed(1, "ETHUSDT")


def test_cell5_shuffle_preserves_marginals_destroys_alignment_leaves_ohlcv():
    x, m = _punch_micro_holes(*_xm())
    seg = np.zeros(240, dtype=np.int64)
    s, ms = shuffle_micro(x, m, seg, symbol="BTCUSDT", seed=0)
    micro = list(MICRO_DIMS)
    ohlcv = list(range(7))
    np.testing.assert_array_equal(s[:, ohlcv], x[:, ohlcv].astype(np.float64))  # OHLCV untouched
    np.testing.assert_array_equal(ms[:, ohlcv], m[:, ohlcv])
    for d in micro:  # marginal preserved per channel (a permutation) — values and mask bits
        np.testing.assert_allclose(np.sort(s[:, d]), np.sort(x[:, d].astype(np.float64)))
        assert ms[:, d].sum() == m[:, d].sum()
    assert not np.array_equal(s[:, micro], x[:, micro])  # alignment destroyed
    assert not np.array_equal(ms[:, micro], m[:, micro])  # the mask moved with it


def test_cell5_training_path_carries_the_perp_tripwire():
    x, m = _xm()
    seg = np.zeros(240, dtype=np.int64)
    m_bad = m.copy()
    m_bad[7, 14] = 0  # one ACTIVE log_oi entry
    with pytest.raises(AssertionError, match="under-shuffle"):
        shuffle_micro(x, m_bad, seg, symbol="BTCUSDT", seed=0)
    # arm constant sanity: Cell 5 sees the full 16 dims (shuffle before selection)
    assert arm_n_features(ARM_MICRO_SHUFFLED) == 16
