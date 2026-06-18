"""Multi-Token Prediction causal-chain heads (§6) — SECONDARY contribution.

``D`` sequential depths on top of the backbone trunk state ``h_t``. Depth ``k`` predicts bar
``t+k+1`` via a causal chain: ``H^{(k)}`` is computed from ``H^{(k-1)}`` and the (teacher-forced)
embedding of bar ``t+k`` — so ``H^{(k)}`` literally cannot be computed without ``H^{(k-1)}``. The
per-bar embedding ``Emb`` and the output ``Head`` (coarse + cross-attn + fine) are SHARED with
the backbone by reference, so MTP carries zero marginal embedding/head params.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from trikaal.model.attention import RMSNorm
from trikaal.model.block import TransformerBlock


class MTPHeads(nn.Module):
    """``D`` MTP depths: each a fusion projection + one causal Transformer block."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        depths: int,
        *,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.depths = depths
        self.fuse = nn.ModuleList(
            nn.Linear(2 * d_model, d_model, bias=False) for _ in range(depths)
        )
        self.norm_h = nn.ModuleList(RMSNorm(d_model) for _ in range(depths))
        self.norm_e = nn.ModuleList(RMSNorm(d_model) for _ in range(depths))
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_ff, causal=True, ffn_kind="silu", max_len=max_len)
            for _ in range(depths)
        )

    def forward(self, h0: Tensor, emb: Tensor) -> list[Tensor]:
        """Return ``[H^{(1)}, …, H^{(D)}]``. ``h0``: trunk [B,L,d]; ``emb``: fused embed [B,L,d]."""
        length = h0.shape[1]
        states: list[Tensor] = []
        h_prev = h0
        for k in range(1, self.depths + 1):
            # e^{(k)}_t = Emb(b_{t+k}) — shift the fused embedding left by k (teacher-forced)
            e_k = torch.zeros_like(emb)
            if k < length:
                e_k[:, : length - k] = emb[:, k:]
            z = self.fuse[k - 1](
                torch.cat([self.norm_h[k - 1](h_prev), self.norm_e[k - 1](e_k)], dim=-1)
            )
            h_prev = self.blocks[k - 1](z)
            states.append(h_prev)
        return states
