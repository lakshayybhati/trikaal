"""Stage-2 trainer: D=0 ablation smoke, baseline math, and a learns-and-beats-baselines smoke."""

from __future__ import annotations

import numpy as np
import torch

from trikaal.data.config import BAR_MS
from trikaal.model.predictor import TrikaalAR
from trikaal.train.train_predictor import (
    marginal_ce_and_acc,
    persistence_accuracy,
    train_stage2,
)
from trikaal.utils.seeding import set_determinism


def test_d0_mode_runs_without_mtp_keys():
    """D=0 (bit-exact Kronos objective for the M6 ablation switch) must run and carry no MTP loss."""
    set_determinism(0)
    model = TrikaalAR(mtp_depths=0, n_layers=1, d_model=64, d_ff=128, n_heads=4, max_len=32)
    b_c = torch.randint(0, model.v_c, (2, 16))
    b_f = torch.randint(0, model.v_f, (2, 16))
    ts = (1_600_000_000_000 + torch.arange(16).repeat(2, 1) * BAR_MS).long()
    out = model.compute_loss(b_c, b_f, ts)
    assert "loss" in out and "nll_main" in out
    assert not any(k.startswith("mtp_depth") for k in out)
    assert torch.isclose(out["loss"], out["nll_main"])  # no MTP term added


def test_marginal_and_persistence_baselines():
    # skewed marginal: token 3 dominates → mode-accuracy high, CE below log(vocab)
    train = np.full(1000, 3, dtype=np.int64)
    train[:100] = np.arange(100) % 5
    val = np.full((4, 10), 3, dtype=np.int64)
    ce, acc = marginal_ce_and_acc(train, val[:, 1:], vocab=8)
    assert acc > 0.9 and ce < np.log(8)
    # a strictly repeating sequence → persistence is perfect
    rep = np.tile(np.array([7]), (3, 12))
    assert persistence_accuracy(rep) == 1.0
    # a never-repeating cycle → persistence is zero
    cyc = np.tile(np.arange(12) % 6, (3, 1))
    assert persistence_accuracy(cyc) < 0.01


def test_stage2_learns_and_beats_baselines_on_structured_stream():
    """On a deterministic cyclic stream the model must learn (val NLL drops) and beat both
    trivial baselines on coarse: CE below the marginal, accuracy above last-token-persistence."""
    set_determinism(0)
    n = 4000
    b_c = (np.arange(n) % 8).astype(np.int64)  # period-8 cycle (persistence never right)
    b_f = (np.arange(n) % 5).astype(np.int64)  # period-5 cycle
    ts = (1_600_000_000_000 + np.arange(n) * BAR_MS).astype(np.int64)
    seg = np.zeros(n, dtype=np.int64)

    res = train_stage2(
        b_c,
        b_f,
        ts,
        seg,
        seq_len=64,
        stride=8,
        batch_size=8,
        val_frac=0.2,
        peak_lr=3e-3,
        max_steps=400,
        eval_every=50,
        n_eval_windows=16,
        mtp_depths=2,
        model_kwargs=dict(n_layers=2, d_model=64, d_ff=128, n_heads=4),
        device="cpu",
        seed=0,
    )
    assert res.history[-1]["val_nll"] < res.history[0]["val_nll"], "val NLL did not drop"
    # beats marginal CE (context helps) and persistence accuracy (b_t != b_{t-1})
    assert res.final["val_nll_coarse"] < res.baselines["marginal_ce_coarse"]
    assert res.final["val_acc_coarse"] > res.baselines["persistence_acc_coarse"]
    # MTP depths are tracked
    assert any(k.startswith("val_mtp_depth") for k in res.final)
