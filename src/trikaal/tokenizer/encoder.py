"""Tokenizer encoder — a shallow Transformer AE encoder (Tokenizer §a).

Embeds the 16-dim bar plus its 16-bit mask (summed), then a 3-layer Transformer (d_model=256,
RoPE, RMSNorm, Pre-LN, SwiGLU). Bidirectional within the window by default (``causal=False``);
``encoder_causal=True`` is exposed so the FSQ ablation can run under a causal AE too.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from trikaal.model.block import TransformerBlock


class PointwiseEncoder(nn.Module):
    """Per-bar encoder — ZERO cross-bar operations (prereg §7 v1.4 fine path).

    Embeds bar t's own (features, mask) exactly like :class:`TokenizerEncoder`'s input stage,
    then residual per-bar MLP blocks (Linear→GELU→Linear). No attention, no convolution, no
    normalization across time: token content at t is a pure function of bar t. The extended
    flip-KAT (fine digits invariant to every other bar) holds by construction and is verified
    empirically in CI."""

    def __init__(
        self,
        n_features: int = 16,
        d_model: int = 256,
        d_ff: int = 512,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_x = nn.Linear(n_features, d_model, bias=False)
        self.w_m = nn.Linear(n_features, d_model, bias=False)
        self.bias_in = nn.Parameter(torch.zeros(d_model))
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(n_layers)
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        e = self.w_x(x) + self.w_m(mask.to(x.dtype)) + self.bias_in
        for blk in self.blocks:
            e = e + blk(e)
        return e


class TokenizerEncoder(nn.Module):
    def __init__(
        self,
        n_features: int = 16,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.0,
        causal: bool = False,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.w_x = nn.Linear(n_features, d_model, bias=False)
        self.w_m = nn.Linear(n_features, d_model, bias=False)
        self.bias_in = nn.Parameter(torch.zeros(d_model))
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model,
                n_heads,
                d_ff,
                causal=causal,
                ffn_kind="swiglu",
                resid_dropout=dropout,
                attn_dropout=dropout,
                ffn_dropout=dropout,
                max_len=max_len,
            )
            for _ in range(n_layers)
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        e = self.w_x(x) + self.w_m(mask.to(x.dtype)) + self.bias_in
        for blk in self.blocks:
            e = blk(e)
        return e
