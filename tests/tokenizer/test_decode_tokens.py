"""decode_tokens is the exact inverse of encode_tokens into the decoder (the AR rollout→features path)."""

from __future__ import annotations

import torch

from trikaal.tokenizer.model import TokenizerAE
from trikaal.utils.seeding import set_determinism


def test_decode_tokens_matches_forward_x_hat():
    set_determinism(0)
    tok = TokenizerAE(d_model=32, n_layers=1, n_heads=2, d_ff=64).eval()
    x = torch.randn(3, 20, 16)
    mask = torch.zeros(3, 20, 16)
    with torch.no_grad():
        out = tok(x, mask)
        x_hat = tok.decode_tokens(out["cidx"], out["fidx"])
    # decode_tokens(encode_tokens(x)) reconstructs the same z_hat the forward pass decoded
    assert torch.allclose(x_hat, out["x_hat"], atol=1e-5)


def test_decode_tokens_id_ranges_roundtrip():
    set_determinism(0)
    tok = TokenizerAE(d_model=32, n_layers=1, n_heads=2, d_ff=64).eval()
    cidx = torch.randint(0, tok.v_c, (2, 8))
    fidx = torch.randint(0, tok.v_f, (2, 8))
    with torch.no_grad():
        x_hat = tok.decode_tokens(cidx, fidx)  # arbitrary valid ids decode without error
    assert x_hat.shape == (2, 8, 16) and torch.isfinite(x_hat).all()
