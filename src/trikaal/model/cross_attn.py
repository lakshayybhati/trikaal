"""Intra-block coarse→fine cross-attention (Backbone §6.2).

Realizes ``p(b^f_t | b_{<t}, b^c_t)``: the query is the embedding of the conditioning coarse
token; the key/value set is the per-position pair ``{h_t, coarse_embed_t}``. The attention is
**per-position** (no cross-bar mixing — every position attends only within its own bar), so the
fine prediction depends on which coarse cell was chosen without leaking other timesteps.

Note on the spec text: §6.2's literal ``K = V = h_t`` (a single key) degenerates to an output
independent of the query, which would not condition the fine head on the coarse token at all
(contradicting §6.3 and the G0 swap test). We therefore let the coarse-query attend over the
2-element set ``{h_t, coarse_embed}`` — a faithful, parameter-identical (``4·d² + RMSNorm``)
realization in which the coarse token genuinely modulates the fine logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from trikaal.model.attention import RMSNorm


class CoarseToFineCrossAttn(nn.Module):
    """Per-position cross-attention fusing the conditioning coarse embedding into ``h_t``."""

    def __init__(self, d_model: int, n_heads: int = 1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, h_t: Tensor, coarse_embed: Tensor) -> Tensor:
        """``h_t``, ``coarse_embed``: [B, L, d]. Returns the fused fine state [B, L, d]."""
        b, length, d = h_t.shape
        kv = torch.stack([h_t, coarse_embed], dim=2)  # [B, L, 2, d] — per-position kv set
        q = self.wq(coarse_embed).view(b * length, self.n_heads, 1, self.d_head)
        k = self.wk(kv).view(b * length, 2, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(kv).view(b * length, 2, self.n_heads, self.d_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)  # [B*L, n_heads, 1, d_head]
        a = a.reshape(b, length, d)
        return self.norm(h_t + self.wo(a))
