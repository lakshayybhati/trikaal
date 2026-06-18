"""Attention primitive unit tests (Backbone §5) — RoPE, causal mask, KV-cache equivalence."""

from __future__ import annotations

import torch

from trikaal.model.attention import (
    MultiHeadSelfAttention,
    RMSNorm,
    apply_rope,
    build_rope_cache,
)
from trikaal.utils.seeding import set_determinism


def test_kv_cache_step_matches_full_forward():
    """Incremental KV-cached generation must equal the parallel causal forward."""
    set_determinism(0)
    attn = MultiHeadSelfAttention(64, 4, causal=True, max_len=32).eval()
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        full = attn(x)
        cache: dict = {"k": None, "v": None}
        inc = torch.cat([attn.step(x[:, t : t + 1], cache, t) for t in range(10)], dim=1)
    assert torch.allclose(full, inc, atol=1e-5), (
        f"KV-cache diverged (max|Δ|={(full - inc).abs().max():.2e})"
    )


def test_causal_attention_ignores_future():
    """Output at position t must not change when positions > t are altered."""
    set_determinism(0)
    attn = MultiHeadSelfAttention(32, 4, causal=True, max_len=16).eval()
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        a1 = attn(x)
        x2 = x.clone()
        x2[:, 5:] = torch.randn_like(x2[:, 5:])
        a2 = attn(x2)
    assert torch.allclose(a1[:, :5], a2[:, :5], atol=1e-6)
    assert not torch.allclose(a1[:, 5:], a2[:, 5:])


def test_rope_preserves_norm_and_is_relative():
    cos, sin = build_rope_cache(16, 8)
    x = torch.randn(1, 2, 16, 8)
    rx = apply_rope(x, cos, sin)
    # rotation preserves per-vector norm
    assert torch.allclose(x.norm(dim=-1), rx.norm(dim=-1), atol=1e-5)


def test_rmsnorm_unit_scale():
    norm = RMSNorm(64)
    x = torch.randn(4, 10, 64) * 7.0
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)
