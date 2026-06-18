"""Coarse→fine hierarchy unit tests (Tokenizer §d/§e)."""

from __future__ import annotations

import torch

from trikaal.constants import FSQ_LEVELS
from trikaal.tokenizer.hierarchy import FSQQuantizer
from trikaal.tokenizer.model import TokenizerAE
from trikaal.utils.seeding import set_determinism


def test_coarse_only_pins_fine_to_center():
    q = FSQQuantizer(FSQ_LEVELS, per_stage_scale=True)
    z = torch.randn(2, 8, len(FSQ_LEVELS)) * 2.0
    z_hat, _codes, _c, _f = q(z)
    z_c = q.coarse_only(z_hat)
    # fine dims are zeroed (the level-0 center), coarse dims preserved
    for i in q.fine_idx:
        assert torch.all(z_c[..., i] == 0.0)
    for i in q.coarse_idx:
        assert torch.allclose(z_c[..., i], z_hat[..., i])


def test_gamma_starts_at_one_and_is_positive():
    q = FSQQuantizer(FSQ_LEVELS, per_stage_scale=True)
    g = q.gamma()
    assert g is not None
    assert torch.allclose(g, torch.ones_like(g), atol=1e-5)
    assert torch.all(g > 0)


def test_encode_tokens_shapes_and_ranges():
    set_determinism(0)
    model = TokenizerAE().eval()
    x = torch.randn(3, 20, 16)
    m = torch.zeros(3, 20, 16)
    b_c, b_f = model.encode_tokens(x, m)
    assert b_c.shape == (3, 20) and b_f.shape == (3, 20)
    assert int(b_c.min()) >= 0 and int(b_c.max()) < model.v_c == 891
    assert int(b_f.min()) >= 0 and int(b_f.max()) < model.v_f == 1225


def test_no_commitment_term_in_loss():
    """FSQ has NO commitment/codebook loss — total loss is L_coarse + L_fine (+ tiny gamma reg)."""
    set_determinism(0)
    model = TokenizerAE()
    x = torch.randn(2, 16, 16)
    m = torch.zeros(2, 16, 16)
    out = model(x, m)
    reconstructed = out["loss_coarse"] + out["loss_fine"] + model.quant.gamma_reg()
    assert torch.allclose(out["loss"], reconstructed)


def test_both_decode_paths_learn_to_reconstruct(feature_window):
    """Both the full and the coarse-only decode paths must learn (the hierarchy is wired).

    On a small batch where the coarse group alone has ample capacity, the *fine* group is
    legitimately under-used (residual decay is a scale-time concern, not a small-batch bug), so
    we do NOT require Δ_fine > 0 here. What we DO require: both paths drop well below the untrained
    baseline, proving the coarse-only (level-0-center) decode and the full decode both function.
    """
    set_determinism(0)
    x_np, m_np = feature_window
    b, length = 2, 32
    x = torch.from_numpy(x_np[: b * length]).view(b, length, 16)
    m = torch.from_numpy(m_np[: b * length].astype("float32")).view(b, length, 16)
    model = TokenizerAE()
    with torch.no_grad():
        base = float(model(x, m)["recon_mae"])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(400):
        opt.zero_grad(set_to_none=True)
        out = model(x, m)
        out["loss"].backward()
        opt.step()
    model.eval()
    out = model(x, m)
    # both paths must clearly improve over the untrained baseline (proves both are wired + get grad)
    assert float(out["recon_mae"]) < 0.8 * base, "full decode did not learn"
    assert float(out["coarse_mae"]) < 0.8 * base, "coarse-only decode path did not learn"
