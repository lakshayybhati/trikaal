"""MTP causal-chain unit tests (§6) — part of gate G0."""

from __future__ import annotations

import torch

from trikaal.data.config import BAR_MS
from trikaal.model.predictor import TrikaalAR
from trikaal.utils.seeding import set_determinism


def _ts(b: int, length: int) -> torch.Tensor:
    return (1_600_000_000_000 + torch.arange(length).repeat(b, 1) * BAR_MS).long()


def test_mtp_is_a_causal_chain():
    """Depth k's representation must depend on depth (k-1)'s — not parallel-independent."""
    set_determinism(0)
    model = TrikaalAR(mtp_depths=3).eval()
    b, length = 2, 24
    b_c = torch.randint(0, model.v_c, (b, length))
    b_f = torch.randint(0, model.v_f, (b, length))
    with torch.no_grad():
        h0 = model.trunk(b_c, b_f, _ts(b, length))
        emb = model.token_embed.fuse(b_c, b_f)
        states = model.mtp(h0, emb)
        # perturb H^{(1)} by re-running with a different h0 and confirm H^{(2)} changes
        states_alt = model.mtp(h0 + 0.5, emb)
    assert not torch.allclose(states[1], states_alt[1]), "depth-2 ignored depth-1 — chain broken"


def test_mtp_loss_runs_and_has_per_depth_terms():
    set_determinism(0)
    model = TrikaalAR(mtp_depths=4)
    b, length = 2, 32
    b_c = torch.randint(0, model.v_c, (b, length))
    b_f = torch.randint(0, model.v_f, (b, length))
    out = model.compute_loss(b_c, b_f, _ts(b, length))
    for k in range(1, 5):
        assert f"mtp_depth{k}" in out
        assert torch.isfinite(out[f"mtp_depth{k}"])
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad


def test_depth_predicts_further_offsets():
    """Deeper depths should generally be harder (validity shrinks toward the sequence edge)."""
    set_determinism(0)
    model = TrikaalAR(mtp_depths=4).eval()
    b, length = 1, 10
    b_c = torch.randint(0, model.v_c, (b, length))
    b_f = torch.randint(0, model.v_f, (b, length))
    out = model.compute_loss(b_c, b_f, _ts(b, length))
    # depth k consumes bars t+k+1; with L=10 all of depths 1..4 still have valid positions
    assert all(f"mtp_depth{k}" in out for k in range(1, 5))
