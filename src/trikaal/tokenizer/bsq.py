"""Binary Spherical Quantization — the Cells-1/3 baseline bottleneck (m6_design §1, spec §1190).

Self-written (invariant 8 — no imported BSQ). Kronos-style k-bit BSQ: the latent is projected to
the unit hypersphere and each dimension is binarized to ``±1/√k`` with a straight-through
estimator — the implicit codebook is the ``2^k`` hypercube corners on the sphere. Unlike FSQ, the
sphere-projected latent and its binarized code are distinct objects with nothing coupling them, so
BSQ carries the classic commitment-style term ``λ·L_quant = λ·‖u − sg(û)‖²`` (spec §1190: the loss
is ``L_coarse + L_fine + λ·L_quant``; the FSQ arm stays commitment-free — that asymmetry IS the
headline simplification under test).

Parity with the FSQ arm (G-parity, invariant 6): ``k = 20`` bits → bpt exactly **20.00** (FSQ
canonical: 20.06 — |Δ| = 0.06 ≤ 0.5); coarse = first 10 bits (V_c = 1024 ≈ 891), fine = last 10
bits (V_f = 1024 ≈ 1225). Same encoder/decoder shells; only the bottleneck + loss term differ.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from trikaal.tokenizer.fsq import mixed_radix_flatten


def bsq_quantize(z: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize ``z`` (``[..., k]``) to the BSQ hypercube-on-sphere. Returns ``(z_hat, codes, u)``.

    ``u = z/‖z‖`` (unit sphere), ``û = sign(u)/√k`` (every corner has unit norm). STE: exact
    forward, identity backward through the binarization; the normalization is differentiated
    normally. ``codes`` are the per-dim bits in {0,1}; ``u`` is returned for the commitment term.
    """
    k = z.shape[-1]
    u = z / z.norm(dim=-1, keepdim=True).clamp_min(eps)
    hard = torch.where(u >= 0, 1.0, -1.0) / math.sqrt(k)
    z_hat = u + (hard - u).detach()  # STE
    codes = (u >= 0).to(torch.long)  # bit b_i = [u_i >= 0]
    return z_hat, codes, u


class BSQQuantizer(nn.Module):
    """BSQ bottleneck with the Kronos 10/10 coarse/fine bit split — FSQQuantizer-interface parity.

    ``forward(z) -> (z_hat, codes, cidx, fidx)`` and ``coarse_only`` mirror
    :class:`~trikaal.tokenizer.hierarchy.FSQQuantizer` so :class:`TokenizerAE` swaps bottlenecks
    with zero shell changes (spec §1190's symmetry guarantee). ``aux_loss(z)`` is the arm-specific
    loss extra: ``λ·L_quant`` here, ``gamma_reg`` (NO commitment) on the FSQ side.
    """

    def __init__(self, k_coarse: int = 10, k_fine: int = 10, quant_lambda: float = 0.25) -> None:
        super().__init__()
        self.k_coarse = int(k_coarse)
        self.k_fine = int(k_fine)
        self.dim = self.k_coarse + self.k_fine
        self.quant_lambda = float(quant_lambda)
        self.coarse_idx = list(range(self.k_coarse))
        self.fine_idx = list(range(self.k_coarse, self.dim))
        self.v_c = 2**self.k_coarse
        self.v_f = 2**self.k_fine

    @property
    def bpt(self) -> float:
        """Bits per token — exactly k (each dim is one bit). The G-parity control variable."""
        return float(self.dim)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        z_hat, codes, _u = bsq_quantize(z)
        cidx = mixed_radix_flatten(codes[..., self.coarse_idx], [2] * self.k_coarse)
        fidx = mixed_radix_flatten(codes[..., self.fine_idx], [2] * self.k_fine)
        return z_hat, codes, cidx, fidx

    def coarse_only(self, z_hat: Tensor) -> Tensor:
        """Pin the fine bits to 0 (the sphere midpoint) for the coarse-only decode — the BSQ
        analog of FSQ's odd-level center pin: 'no fine information', not a valid fine corner."""
        z_c = z_hat.clone()
        z_c[..., self.fine_idx] = 0.0
        return z_c

    def aux_loss(self, z: Tensor) -> Tensor:
        """``λ·L_quant`` — the commitment term dragging the sphere-projected latent toward its
        binarized code (spec §984: in BSQ, ``u`` and ``û`` are distinct objects with nothing else
        coupling them). ``L_quant = ‖u − sg(û)‖²`` (mean over elements)."""
        z_hat, _codes, u = bsq_quantize(z)
        return self.quant_lambda * F.mse_loss(u, z_hat.detach())

    def gamma_reg(self) -> Tensor:  # interface parity with FSQQuantizer (no scale params here)
        return torch.zeros(())


__all__ = ["BSQQuantizer", "bsq_quantize"]
