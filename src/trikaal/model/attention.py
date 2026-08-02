"""From-scratch attention primitives: RMSNorm, RoPE, and multi-head self-attention.

RoPE is applied to Q/K per head (θ_base = 10000); V is untouched. Attention uses
``torch.scaled_dot_product_attention`` (the deterministic SDPA math path on CPU; the
FlashAttention-2 kernel is the production substitute on CUDA). A standard KV-cache supports
O(1)-per-bar incremental generation. ``MultiHeadSelfAttention`` serves both the bidirectional
tokenizer AE (``causal=False``) and the causal AR backbone (``causal=True``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """RMSNorm with a learnable gain (no mean-centering), computed in fp32 (precision §1)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    """Return ``(cos, sin)`` each ``[seq_len, head_dim]`` for rotary position embedding."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: Tensor) -> Tensor:
    """ATTRIBUTION (§7 v1.6.17, audit S-4): the split-half rotation and the ``x*cos + rot(x)*sin``
    form below are the STANDARD LLaMA/HuggingFace formulation of RoPE (Su et al., "RoFormer",
    arXiv:2104.09864). The convention is effectively forced — a different pairing would not
    interoperate with any published RoPE — so this is unavoidable and is NOT an invariant-8
    violation; it is recorded here because invariant 8 is stated strongly and a reader is
    entitled to know which lines are ours and which are the field's."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding. ``x``: [B, H, L, Dh]; ``cos``/``sin``: [L, Dh]."""
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE, an optional causal mask, and a KV-cache."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        causal: bool,
        attn_dropout: float = 0.0,
        rope_base: float = 10000.0,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        cos, sin = build_rope_cache(max_len, self.d_head, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, length, _ = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, Dh]
        return qkv[0], qkv[1], qkv[2]

    def forward(self, x: Tensor) -> Tensor:
        b, length, _ = x.shape
        q, k, v = self._project(x)
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        dropout_p = self.attn_dropout if self.training else 0.0
        # backend selected per run via attention_mode.set_attention_backend (inv. 7: the mode is
        # RECORDED; flash2 is the fast non-deterministic CUDA path, SDPA the bit-exact fallback)
        if getattr(self, "attention_backend", None) == "flash2":  # pragma: no cover — CUDA only
            from flash_attn import flash_attn_func

            a = flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=dropout_p,
                causal=self.causal,
            ).transpose(1, 2)
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=dropout_p)
        a = a.transpose(1, 2).reshape(b, length, self.d_model)
        return self.out(a)

    @torch.no_grad()
    def step(self, x_t: Tensor, cache: dict, pos: int) -> Tensor:
        """Incremental one-bar step for KV-cached generation. ``x_t``: [B, 1, d_model]."""
        b = x_t.shape[0]
        q, k, v = self._project(x_t)  # each [B, H, 1, Dh]
        cos = self.rope_cos[pos : pos + 1]
        sin = self.rope_sin[pos : pos + 1]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache.get("k") is None:
            cache["k"], cache["v"] = k, v
        else:
            cache["k"] = torch.cat([cache["k"], k], dim=2)
            cache["v"] = torch.cat([cache["v"], v], dim=2)
        # current query attends over all cached keys (which already include the new bar)
        a = F.scaled_dot_product_attention(q, cache["k"], cache["v"], is_causal=False)
        a = a.transpose(1, 2).reshape(b, 1, self.d_model)
        return self.out(a)
