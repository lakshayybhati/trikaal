"""Coarse→fine hierarchy + per-stage learnable scaling (Tokenizer §d / §e).

Partitions the ``D`` FSQ dims into a coarse group (high-``L`` dims) and a fine group via the
derived grouping rule. The coarse-only decode pins fine dims to their level-0 center (well-defined
because all levels are odd). A learnable per-dim scale ``gamma_i = softplus(rho_i)`` (init 1.0)
counters residual magnitude decay by stretching a small fine residual across the squash range.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from trikaal.constants import bits_per_token, derive_coarse_fine, fsq_vocab_sizes
from trikaal.tokenizer.fsq import fsq_quantize, split_codes

_INV_SOFTPLUS_1 = math.log(math.e - 1.0)  # rho such that softplus(rho) == 1.0


class FSQQuantizer(nn.Module):
    """FSQ bottleneck with the derived coarse/fine split and optional per-stage scaling."""

    def __init__(self, levels: tuple[int, ...] | list[int], per_stage_scale: bool = True) -> None:
        super().__init__()
        self.levels: tuple[int, ...] = tuple(levels)
        self.dim = len(self.levels)
        self.coarse_idx, self.fine_idx = derive_coarse_fine(self.levels)
        self.v_c, self.v_f = fsq_vocab_sizes(self.levels)
        self.register_buffer("levels_t", torch.tensor(self.levels, dtype=torch.float32))
        if per_stage_scale:
            self.rho = nn.Parameter(torch.full((self.dim,), _INV_SOFTPLUS_1))
        else:
            self.register_parameter("rho", None)

    @property
    def bpt(self) -> float:
        """Bits per token = Σ log2(L_i) — the G-parity control variable (canonical: 20.06)."""
        return bits_per_token(self.levels)

    def gamma(self) -> Tensor | None:
        return None if self.rho is None else F.softplus(self.rho)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        z_hat, codes = fsq_quantize(z, self.levels_t, self.gamma())
        cidx, fidx = split_codes(codes, self.coarse_idx, self.fine_idx, list(self.levels))
        return z_hat, codes, cidx, fidx

    def coarse_only(self, z_hat: Tensor) -> Tensor:
        """Pin fine dims to their level-0 center (z_hat == 0) for the coarse-only decode."""
        z_c = z_hat.clone()
        z_c[..., self.fine_idx] = 0.0
        return z_c

    def gamma_reg(self) -> Tensor:
        """Light regularizer ``1e-4 · Σ (log gamma)^2`` discouraging runaway scaling (§e)."""
        g = self.gamma()
        if g is None:
            return torch.zeros((), device=self.levels_t.device)
        return 1e-4 * (torch.log(g) ** 2).sum()

    def aux_loss(self, z: Tensor) -> Tensor:
        """The arm-specific loss extra. For FSQ this is ONLY ``gamma_reg`` — z-independent, with
        **NO commitment term** (spec §984/§1833: the code is a deterministic function of ``z``, so
        no λ·L_quant; its absence vs the BSQ arm is the headline simplification under test)."""
        return self.gamma_reg()
