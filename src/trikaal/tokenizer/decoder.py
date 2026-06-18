"""Tokenizer decoder — structural mirror of the encoder (Tokenizer §h).

A 3-layer Transformer (same d_model=256, RoPE, RMSNorm, Pre-LN, SwiGLU) over the
``W_out·z_hat`` latent, then a linear head ``256 → 16``. Bidirectional within the window (this is
reconstruction, not generation). No output activation: the linear head predicts the normalized
``[-5, 5]`` features directly and the Huber loss handles the scale.
"""

from __future__ import annotations

from torch import Tensor, nn

from trikaal.model.block import TransformerBlock


class TokenizerDecoder(nn.Module):
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
        self.w_dec = nn.Linear(d_model, n_features, bias=True)

    def forward(self, g: Tensor) -> Tensor:
        for blk in self.blocks:
            g = blk(g)
        return self.w_dec(g)
