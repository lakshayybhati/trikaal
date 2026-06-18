"""Finite Scalar Quantization — the novel leg (Tokenizer §b).

Each latent coordinate is quantized independently to a small fixed grid of ``L_i`` levels via a
bounded ``tanh`` squash + per-dim rounding with a straight-through estimator. There is no learned
codebook (hence no commitment loss and no codebook-collapse mode); the "codebook" is the implicit
Cartesian product of the per-dimension level sets. All levels are odd so the coarse-only decode
can pin each fine dim to its level-0 **center** cell.
"""

from __future__ import annotations

import torch
from torch import Tensor


def fsq_quantize(z: Tensor, levels: Tensor, scale: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Quantize ``z`` (``[..., D]``) to the per-dim FSQ grid. Returns ``(z_hat, codes)``.

    ``levels``: float tensor ``[D]`` of (odd) level counts. ``scale``: optional learnable per-dim
    pre-squash scale ``gamma`` (residual-decay guard, §e). STE: exact forward, identity backward
    through ``round``; the ``tanh`` is differentiated normally.
    """
    if scale is not None:
        z = scale * z
    half = (levels - 1.0) / 2.0  # [D]
    z_sq = half * torch.tanh(z)  # bounded in (-half, half)
    z_round = torch.round(z_sq)
    z_hat = z_sq + (z_round - z_sq).detach()  # STE
    codes = (z_round + half).to(torch.long)  # shift to {0 .. L_i-1}
    return z_hat, codes


def mixed_radix_flatten(codes: Tensor, levels: list[int]) -> Tensor:
    """Flatten per-dim codes ``[..., k]`` to a single index via mixed-radix (big-endian)."""
    out = torch.zeros(codes.shape[:-1], dtype=torch.long, device=codes.device)
    for i, lvl in enumerate(levels):
        out = out * lvl + codes[..., i]
    return out


def split_codes(
    codes: Tensor, coarse_idx: list[int], fine_idx: list[int], levels: list[int]
) -> tuple[Tensor, Tensor]:
    """Factor ``codes`` into ``(cidx, fidx)`` coarse/fine sub-token indices."""
    coarse_levels = [levels[i] for i in coarse_idx]
    fine_levels = [levels[i] for i in fine_idx]
    cidx = mixed_radix_flatten(codes[..., coarse_idx], coarse_levels)
    fidx = mixed_radix_flatten(codes[..., fine_idx], fine_levels)
    return cidx, fidx
