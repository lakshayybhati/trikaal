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
def test_cell5_shuffle_deterministic_and_symbol_seeded():
    x, m = _xm()
    seg = np.zeros(240, dtype=np.int64)
    seg[120:] = 1  # two segments
    s1 = shuffle_micro(x, m, seg, symbol="BTCUSDT", seed=0)
    s2 = shuffle_micro(x, m, seg, symbol="BTCUSDT", seed=0)
    np.testing.assert_array_equal(s1, s2)  # deterministic under (seed, symbol)
    s3 = shuffle_micro(x, m, seg, symbol="ETHUSDT", seed=0)
    s4 = shuffle_micro(x, m, seg, symbol="BTCUSDT", seed=1)
    assert not np.array_equal(s1, s3) and not np.array_equal(s1, s4)
    assert cell5_seed(0, "BTCUSDT") != cell5_seed(0, "ETHUSDT") != cell5_seed(1, "ETHUSDT")


def test_cell5_shuffle_preserves_marginals_destroys_alignment_leaves_ohlcv():
    x, m = _xm()
    seg = np.zeros(240, dtype=np.int64)
    s = shuffle_micro(x, m, seg, symbol="BTCUSDT", seed=0)
    micro = list(MICRO_DIMS)
    ohlcv = list(range(7))
    np.testing.assert_array_equal(s[:, ohlcv], x[:, ohlcv].astype(np.float64))  # OHLCV untouched
    for d in micro:  # marginal preserved per channel (a permutation)
        np.testing.assert_allclose(np.sort(s[:, d]), np.sort(x[:, d].astype(np.float64)))
    assert not np.array_equal(s[:, micro], x[:, micro])  # alignment destroyed


def test_cell5_training_path_carries_the_perp_tripwire():
    x, m = _xm()
    seg = np.zeros(240, dtype=np.int64)
    m_bad = m.copy()
    m_bad[7, 14] = 0  # one ACTIVE log_oi entry
    with pytest.raises(AssertionError, match="under-shuffle"):
        shuffle_micro(x, m_bad, seg, symbol="BTCUSDT", seed=0)
    # arm constant sanity: Cell 5 sees the full 16 dims (shuffle before selection)
    assert arm_n_features(ARM_MICRO_SHUFFLED) == 16
