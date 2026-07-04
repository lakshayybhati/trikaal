"""KATs for the BSQ baseline bottleneck (m6_design §6 item 1) + the G-parity gate.

BSQ is the Cells-1/3 baseline: k_c=k_f=10 bits, V_c=V_f=1024, bpt exactly 20.00, commitment term
λ·L_quant (the FSQ arm stays commitment-free — that asymmetry is the headline simplification
under test). Self-written (invariant 8)."""

from __future__ import annotations

import pytest
import torch

from trikaal.constants import FSQ_BPT_BAND
from trikaal.tokenizer.bsq import BSQQuantizer, bsq_quantize
from trikaal.tokenizer.model import TokenizerAE
from trikaal.tokenizer.parity import GParityError, check_arm_parity
from trikaal.utils.seeding import set_determinism


def test_bsq_bpt_vocab_and_band():
    q = BSQQuantizer()
    assert q.bpt == 20.0  # exactly k bits
    assert q.v_c == q.v_f == 1024
    assert FSQ_BPT_BAND[0] <= q.bpt <= FSQ_BPT_BAND[1]
    assert q.dim == 20 and q.coarse_idx == list(range(10)) and q.fine_idx == list(range(10, 20))


def test_bsq_quantize_determinism_and_geometry():
    set_determinism(0)
    z = torch.randn(4, 32, 20, dtype=torch.float64).float()
    z_hat1, codes1, _u1 = bsq_quantize(z)
    z_hat2, codes2, _u2 = bsq_quantize(z)
    assert torch.equal(codes1, codes2) and torch.equal(z_hat1, z_hat2)  # deterministic
    assert codes1.min() >= 0 and codes1.max() <= 1  # bits
    # every forward value is a hypercube corner on the unit sphere: |z_hat_i| = 1/√k, ‖z_hat‖ = 1
    assert torch.allclose(z_hat1.abs(), torch.full_like(z_hat1, 1 / 20**0.5), atol=1e-6)
    assert torch.allclose(z_hat1.norm(dim=-1), torch.ones(4, 32), atol=1e-5)
    # STE: gradient flows back to z through the binarization
    z_g = z.clone().requires_grad_(True)
    out, _, _ = bsq_quantize(z_g)
    out.sum().backward()
    assert z_g.grad is not None and torch.isfinite(z_g.grad).all()


def test_bsq_tokenizer_roundtrip_and_commitment():
    """decode_tokens(*encode_tokens(x,m)) reproduces forward()['x_hat'] (the AR rollout path);
    the BSQ arm carries λ·L_quant while the FSQ arm's aux_loss is z-independent (NO commitment)."""
    set_determinism(0)
    tok = TokenizerAE(quantizer="bsq", d_model=32, n_layers=1, n_heads=2, d_ff=64).eval()
    x = torch.randn(2, 64, 16)
    m = torch.zeros(2, 64, dtype=torch.uint8).unsqueeze(-1).expand(2, 64, 16).contiguous()
    with torch.no_grad():
        out = tok(x, m)
        b_c, b_f = tok.encode_tokens(x, m)
        x_rt = tok.decode_tokens(b_c, b_f)
    assert b_c.max() < 1024 and b_f.max() < 1024
    assert torch.allclose(x_rt, out["x_hat"], atol=1e-6)
    # commitment: BSQ aux_loss depends on z's DIRECTION (the sphere projection makes it
    # scale-invariant, so perturb additively) and is > 0 for generic z; FSQ's is z-independent
    z = torch.randn(2, 64, 20)
    bsq_aux1, bsq_aux2 = tok.quant.aux_loss(z), tok.quant.aux_loss(z + 1.0)
    assert bsq_aux1.item() > 0.0 and not torch.isclose(bsq_aux1, bsq_aux2)
    assert torch.isclose(tok.quant.aux_loss(z), tok.quant.aux_loss(z * 3.0))  # scale-invariant
    fsq = TokenizerAE(d_model=32, n_layers=1, n_heads=2, d_ff=64)
    zf = torch.randn(2, 64, fsq.quant.dim)
    assert torch.isclose(fsq.quant.aux_loss(zf), fsq.quant.aux_loss(zf + 1.0))  # z-independent


def test_g_parity_gate_passes_canonical_pair_and_has_teeth():
    # the CANONICAL pair — the dims the cells actually construct (d_model=256 shells dominate,
    # so the D=7-vs-k=20 projection delta is ~0.3%; at toy d_model=32 it is >2% and would fail,
    # which is the gate doing its job — parity is a property of the real cells, not toys)
    fsq = TokenizerAE()
    bsq = TokenizerAE(quantizer="bsq")
    rep = check_arm_parity(fsq, bsq)
    assert rep["d_bpt"] == pytest.approx(abs(fsq.quant.bpt - 20.0), abs=1e-9)  # 20.06 vs 20.00
    assert rep["param_frac"] <= 0.02
    # teeth 1: a bpt-mismatched arm (10-bit BSQ → bpt 10, outside the band) must raise
    with pytest.raises(GParityError):
        check_arm_parity(fsq, TokenizerAE(quantizer="bsq", bsq_bits=(5, 5)))
    # teeth 2: a param-mismatched arm (double-width shells) must raise
    with pytest.raises(GParityError):
        check_arm_parity(fsq, TokenizerAE(quantizer="bsq", d_model=512, d_ff=1024))
