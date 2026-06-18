"""G2 — determinism smoke: same seed + config → bit-identical loss curves (§7.7, §2.7).

Runs a short micro-train of both the tokenizer and the predictor twice with identical seed and
asserts byte-identical per-step loss curves. This guards the "reproduce any prediction on any
date" deliverable. Runs on CPU with ``torch.use_deterministic_algorithms(True)``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trikaal.data.config import BAR_MS
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.gates import determinism_smoke
from trikaal.utils.seeding import set_determinism

_STEPS = 40


@pytest.mark.gate
def test_tokenizer_determinism():
    # fixed data, identical across both runs
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((2, 32, 16)).astype(np.float32))
    m = torch.zeros(2, 32, 16, dtype=torch.float32)

    def build_and_run() -> list[float]:
        model = TokenizerAE()  # init RNG already pinned by the helper
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        model.train()
        for _ in range(_STEPS):
            opt.zero_grad(set_to_none=True)
            out = model(x, m)
            out["loss"].backward()
            opt.step()
            losses.append(float(out["loss"].detach()))
        return losses

    passed, max_div = determinism_smoke(build_and_run, seed=1234, steps=_STEPS)
    assert passed, f"tokenizer loss curves diverged (max |Δ| = {max_div:.3e})"


@pytest.mark.gate
def test_predictor_determinism():
    b, length = 2, 24
    rng = np.random.default_rng(1)
    b_c = torch.from_numpy(rng.integers(0, 891, (b, length)))
    b_f = torch.from_numpy(rng.integers(0, 1225, (b, length)))
    ts = (1_600_000_000_000 + torch.arange(length).repeat(b, 1) * BAR_MS).long()

    def build_and_run() -> list[float]:
        model = TrikaalAR(mtp_depths=2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        model.train()
        for _ in range(_STEPS):
            opt.zero_grad(set_to_none=True)
            out = model.compute_loss(b_c, b_f, ts)
            out["loss"].backward()
            opt.step()
            losses.append(float(out["loss"].detach()))
        return losses

    passed, max_div = determinism_smoke(build_and_run, seed=2024, steps=_STEPS)
    assert passed, f"predictor loss curves diverged (max |Δ| = {max_div:.3e})"


def test_set_determinism_is_idempotent_across_runs():
    set_determinism(7)
    a = torch.randn(100)
    set_determinism(7)
    b = torch.randn(100)
    assert torch.equal(a, b)
