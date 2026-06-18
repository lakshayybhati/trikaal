"""Token embedding (coarse ⊕ fine → W_fuse) + learnable temporal embeddings (Backbone §3, §4).

The fused token path ``CoarseEmbed ⊕ FineEmbed → W_fuse`` IS the per-bar embedding ``Emb(b)``
that MTP consumes by reference (zero marginal embedding params). Five temporal tables are keyed
off the bar-open timestamp (deterministic, trivially causal). Token dropout zeros whole fused bar
embeddings (train only) — Kronos's input regularizer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class TokenDropout(nn.Module):
    """Zero entire fused bar embeddings at random (train only), inverted-dropout scaled."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.empty((*x.shape[:-1], 1), device=x.device).bernoulli_(keep)
        return x * mask / keep


class TokenEmbedding(nn.Module):
    """Coarse + fine subtoken embeddings fused into one per-bar embedding."""

    def __init__(self, v_c: int, v_f: int, d_model: int, token_dropout: float = 0.1) -> None:
        super().__init__()
        self.coarse = nn.Embedding(v_c, d_model)
        self.fine = nn.Embedding(v_f, d_model)
        self.w_fuse = nn.Linear(2 * d_model, d_model, bias=False)
        self.token_drop = TokenDropout(token_dropout)

    def fuse(self, b_c: Tensor, b_f: Tensor) -> Tensor:
        e = self.w_fuse(torch.cat([self.coarse(b_c), self.fine(b_f)], dim=-1))
        return e

    def forward(self, b_c: Tensor, b_f: Tensor) -> Tensor:
        return self.token_drop(self.fuse(b_c, b_f))


def temporal_keys(ts: Tensor) -> Tensor:
    """Derive the 5 calendar keys ``[B, L, 5]`` (minute, hour, dow, dom, moy) from bar-open ms."""
    t = ts.detach().cpu().numpy().astype("datetime64[ms]")
    minute_of_day = (t.astype("datetime64[m]").astype("int64")) % 1440
    hour_of_day = (t.astype("datetime64[h]").astype("int64")) % 24
    days = t.astype("datetime64[D]")
    dow = (days.astype("int64") + 3) % 7  # 1970-01-01 was Thursday; Monday=0
    months = t.astype("datetime64[M]")
    dom = (days - months).astype("timedelta64[D]").astype("int64")  # 0-indexed day of month
    years = t.astype("datetime64[Y]")
    moy = (months - years).astype("timedelta64[M]").astype("int64")  # 0-indexed month
    keys = np.stack([minute_of_day, hour_of_day, dow, dom, moy], axis=-1)
    return torch.from_numpy(keys).to(ts.device).long()


class TemporalEmbedding(nn.Module):
    """Five learnable lookup tables summed into the bar embedding (Kronos parity)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.minute = nn.Embedding(1440, d_model)
        self.hour = nn.Embedding(24, d_model)
        self.dow = nn.Embedding(7, d_model)
        self.dom = nn.Embedding(31, d_model)
        self.moy = nn.Embedding(12, d_model)

    def forward(self, ts: Tensor) -> Tensor:
        k = temporal_keys(ts)
        return (
            self.minute(k[..., 0])
            + self.hour(k[..., 1])
            + self.dow(k[..., 2])
            + self.dom(k[..., 3])
            + self.moy(k[..., 4])
        )
