"""Hierarchical reconstruction loss — Huber, masked, per-feature weighted (Tokenizer §d).

``L = L_coarse + L_fine`` with NO commitment/codebook term (the whole point of FSQ over BSQ).
Both terms use Huber (smooth-L1, δ=1.0) for heavy-tailed robustness. The per-feature loss is
multiplied by ``(1 - mask)`` so structurally-absent channels contribute zero gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from trikaal.constants import FINANCE_WEIGHTED_IDX, N_FEATURES


def weighted_masked_huber(
    x_hat: Tensor, x: Tensor, keep: Tensor, w_feat: Tensor, delta: float = 1.0
) -> Tensor:
    """Masked, per-feature-weighted Huber over ``[B, L, F]`` tensors."""
    err = F.smooth_l1_loss(x_hat, x, reduction="none", beta=delta)  # [B,L,F]
    w = keep * w_feat  # w_feat [F] broadcasts over (B,L)
    return (err * w).sum() / w.sum().clamp_min(1e-6)


def masked_mae(x_hat: Tensor, x: Tensor, keep: Tensor) -> Tensor:
    """Reconstruction MAE over kept features (z-scored units) — the G1 stage-1 gate metric."""
    num = ((x_hat - x).abs() * keep).sum()
    return num / keep.sum().clamp_min(1e-6)


def default_feature_weights(finance_weighted: bool = False, n_features: int = N_FEATURES) -> Tensor:
    """``w_feat`` [n_features]. Uniform by default (Training §2); optional 2× on the finance block.

    ``n_features < 16`` is the OHLCV-only ablation arm (the ``OHLCV_ONLY_IDX`` prefix 0-6 — the
    subset is contiguous, so arm feature *i* is original feature *i*). Finance weights apply only
    to indices that exist in the arm (in-range members of ``FINANCE_WEIGHTED_IDX``)."""
    w = torch.ones(n_features)
    if finance_weighted:
        for i in FINANCE_WEIGHTED_IDX:
            if i < n_features:
                w[i] = 2.0
    return w
