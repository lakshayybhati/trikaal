"""Pre-LN transformer block + FFN (feed-forward).

The AR backbone uses a plain 2-matrix SiLU FFN for exact Kronos parity (Backbone §1.1); the
tokenizer AE uses SwiGLU (Tokenizer §a). Both are selectable via ``ffn_kind``.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from trikaal.model.attention import MultiHeadSelfAttention, RMSNorm


class FeedForward(nn.Module):
    """SiLU 2-matrix (Kronos parity) or SwiGLU (tokenizer) FFN."""

    def __init__(
        self, d_model: int, d_ff: int, *, kind: str = "silu", dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "silu":
            self.w_in = nn.Linear(d_model, d_ff, bias=False)
            self.w_out = nn.Linear(d_ff, d_model, bias=False)
        elif kind == "swiglu":
            self.w_gate = nn.Linear(d_model, d_ff, bias=False)
            self.w_in = nn.Linear(d_model, d_ff, bias=False)
            self.w_out = nn.Linear(d_ff, d_model, bias=False)
        else:
            raise ValueError(f"unknown ffn kind: {kind}")
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.kind == "silu":
            h = F.silu(self.w_in(x))
        else:
            h = F.silu(self.w_gate(x)) * self.w_in(x)
        return self.w_out(self.drop(h))


class TransformerBlock(nn.Module):
    """Pre-LN + RMSNorm block: ``x + attn(norm(x))`` then ``x + ffn(norm(x))``."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        causal: bool,
        ffn_kind: str = "silu",
        resid_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, n_heads, causal=causal, attn_dropout=attn_dropout, max_len=max_len
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, kind=ffn_kind, dropout=ffn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.resid_drop(self.attn(self.norm1(x)))
        x = x + self.resid_drop(self.ffn(self.norm2(x)))
        return x
