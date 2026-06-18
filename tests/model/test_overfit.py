"""G1 stage-2 — overfit a single token batch to coarse+fine CE < 0.05 nats/token (§7.0).

Proves the AR backbone + MTP can drive next-token NLL to ~0 on a fixed batch — i.e. the causal
mask, the hierarchical coarse→fine factorization, the MTP causal chain, and the loss reductions
are correctly wired. Inability to overfit one batch would indicate a masking/causality/loss bug,
not a capacity issue.
"""

from __future__ import annotations

import pytest
import torch

from trikaal.data.config import BAR_MS
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.gates import overfit_single_batch
from trikaal.utils.seeding import set_determinism


@pytest.mark.gate
def test_stage2_overfit_to_ce_below_0_05():
    set_determinism(0)
    # one fixed batch of pre-tokenized sequences (untrained tokenizer → arbitrary but valid tokens)
    tok = TokenizerAE().eval()
    x = torch.randn(2, 24, 16)
    mask = torch.zeros(2, 24, 16)
    with torch.no_grad():
        b_c, b_f = tok.encode_tokens(x, mask)
    ts = (1_600_000_000_000 + torch.arange(24).repeat(2, 1) * BAR_MS).long()

    set_determinism(0)
    model = TrikaalAR(mtp_depths=4)  # MTP enabled; depth-1 head trains alongside the main head

    def step_fn(mdl: TrikaalAR) -> tuple[torch.Tensor, float]:
        out = mdl.compute_loss(b_c, b_f, ts)
        return out["loss"], float(out["nll_main"])

    res = overfit_single_batch(model, step_fn, target=0.05, max_steps=2000, lr=1e-3)
    assert res.passed, (
        f"coarse+fine CE floored at {res.metric:.3e} after {res.steps} steps — "
        "masking/causality/loss bug, not capacity"
    )


@pytest.mark.gate
def test_stage2_mtp_depths_also_learn():
    """The MTP causal-chain depths must train (their per-depth CE drops)."""
    set_determinism(0)
    tok = TokenizerAE().eval()
    x = torch.randn(2, 24, 16)
    with torch.no_grad():
        b_c, b_f = tok.encode_tokens(x, torch.zeros(2, 24, 16))
    ts = (1_600_000_000_000 + torch.arange(24).repeat(2, 1) * BAR_MS).long()

    set_determinism(0)
    model = TrikaalAR(mtp_depths=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    model.train()
    first = model.compute_loss(b_c, b_f, ts)
    d1_start = float(first["mtp_depth1"])
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        out = model.compute_loss(b_c, b_f, ts)
        out["loss"].backward()
        opt.step()
    d1_end = float(model.compute_loss(b_c, b_f, ts)["mtp_depth1"])
    assert d1_end < d1_start, "MTP depth-1 did not learn"
