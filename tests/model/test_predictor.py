"""AR backbone unit tests (Backbone §2/§5/§6) — part of gate G0."""

from __future__ import annotations

import torch

from trikaal.data.config import BAR_MS
from trikaal.model.predictor import TrikaalAR
from trikaal.utils.seeding import set_determinism


def _ts(b: int, length: int) -> torch.Tensor:
    return (1_600_000_000_000 + torch.arange(length).repeat(b, 1) * BAR_MS).long()


def test_param_count_matches_kronos_small():
    set_determinism(0)
    model = TrikaalAR(mtp_depths=0)
    n = model.num_params()
    assert n == 21_301_248, f"expected 21,301,248 (Kronos_small class), got {n:,}"
    assert 20_000_000 <= n <= 24_000_000


def test_mtp_param_overhead():
    set_determinism(0)
    model = TrikaalAR(mtp_depths=4)
    assert model.num_backbone_params() == 21_301_248
    mtp = sum(p.numel() for p in model.mtp.parameters())
    assert mtp % 4 == 0
    per_depth = mtp // 4
    assert abs(per_depth - 2_621_440) < 5_000, (
        per_depth
    )  # ~2.62M/depth (norm gains differ slightly)


def test_forward_shapes():
    set_determinism(0)
    model = TrikaalAR(mtp_depths=4).eval()
    b, length = 2, 16
    b_c = torch.randint(0, model.v_c, (b, length))
    b_f = torch.randint(0, model.v_f, (b, length))
    out = model(b_c, b_f, _ts(b, length))
    assert out.h_final.shape == (b, length, 512)
    assert out.logits_c.shape == (b, length, model.v_c)
    assert out.logits_f.shape == (b, length, model.v_f)


def test_fine_head_conditions_on_coarse():
    """G0: swapping the conditioning coarse token changes the fine logits."""
    set_determinism(0)
    model = TrikaalAR(mtp_depths=0).eval()
    b, length = 2, 12
    b_c = torch.randint(0, model.v_c, (b, length))
    b_f = torch.randint(0, model.v_f, (b, length))
    h = model.trunk(b_c, b_f, _ts(b, length))
    lf1 = model.fine_logits(h, b_c)
    lf2 = model.fine_logits(h, (b_c + 7) % model.v_c)
    assert (lf1 - lf2).abs().max() > 1e-5


def test_backbone_is_causal():
    """Coarse logits at positions <= t must not change when future tokens are altered."""
    set_determinism(0)
    model = TrikaalAR(mtp_depths=0).eval()
    b, length, t = 1, 16, 8
    b_c = torch.randint(0, model.v_c, (b, length))
    b_f = torch.randint(0, model.v_f, (b, length))
    ts = _ts(b, length)
    with torch.no_grad():
        h1 = model.trunk(b_c, b_f, ts)
        b_c2, b_f2 = b_c.clone(), b_f.clone()
        b_c2[:, t + 1 :] = (b_c2[:, t + 1 :] + 13) % model.v_c
        b_f2[:, t + 1 :] = (b_f2[:, t + 1 :] + 13) % model.v_f
        h2 = model.trunk(b_c2, b_f2, ts)
    assert torch.allclose(h1[:, : t + 1], h2[:, : t + 1], atol=1e-5), (
        "future tokens leaked into past"
    )
    assert not torch.allclose(h1[:, t + 1 :], h2[:, t + 1 :]), (
        "altering future should change future"
    )
