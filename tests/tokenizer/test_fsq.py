"""FSQ mechanism unit tests (Tokenizer §b/§c/§d) — part of gate G0."""

from __future__ import annotations

import math

import torch

from trikaal.constants import (
    FSQ_LEVELS,
    FSQ_V_C,
    FSQ_V_F,
    bits_per_token,
    derive_coarse_fine,
    fsq_vocab_sizes,
)
from trikaal.tokenizer.fsq import fsq_quantize, mixed_radix_flatten, split_codes


def test_canonical_config_bits_and_oddness():
    assert all(level % 2 == 1 for level in FSQ_LEVELS), "all FSQ levels must be odd"
    bpt = bits_per_token(FSQ_LEVELS)
    assert abs(bpt - 20.06) < 0.01
    assert 19.5 <= bpt <= 20.5, "bpt must sit in the BSQ parity band"


def test_derived_coarse_fine_split():
    coarse_idx, fine_idx = derive_coarse_fine(FSQ_LEVELS)
    coarse_levels = sorted(FSQ_LEVELS[i] for i in coarse_idx)
    fine_levels = sorted(FSQ_LEVELS[i] for i in fine_idx)
    assert coarse_levels == [9, 9, 11], coarse_levels  # {11,9,9}
    assert fine_levels == [5, 5, 7, 7], fine_levels  # {7,7,5,5}
    v_c, v_f = fsq_vocab_sizes(FSQ_LEVELS)
    assert (v_c, v_f) == (FSQ_V_C, FSQ_V_F) == (891, 1225)


def test_quantize_index_range_and_bounded_error():
    torch.manual_seed(0)
    levels = torch.tensor(FSQ_LEVELS, dtype=torch.float32)
    z = torch.randn(4, 32, len(FSQ_LEVELS)) * 3.0  # wide spread
    z_hat, codes = fsq_quantize(z, levels)
    # codes are valid per-dim indices
    for i, lvl in enumerate(FSQ_LEVELS):
        assert int(codes[..., i].min()) >= 0
        assert int(codes[..., i].max()) <= lvl - 1
    # bounded quantization error: |z_sq - round(z_sq)| <= 0.5 per dim
    half = (levels - 1) / 2
    z_sq = half * torch.tanh(z)
    assert (z_sq - z_hat.detach()).abs().max() <= 0.5 + 1e-6


def test_straight_through_gradient():
    levels = torch.tensor(FSQ_LEVELS, dtype=torch.float32)
    z = torch.randn(2, 8, len(FSQ_LEVELS), requires_grad=True)
    z_hat, _ = fsq_quantize(z, levels)
    z_hat.sum().backward()
    assert z.grad is not None
    # gradient is non-zero where tanh is not saturated
    assert z.grad.abs().sum() > 0
    # forward is exact (z_hat equals the rounded grid value, integer + center offset)
    half = (levels - 1) / 2
    assert torch.allclose(z_hat + half, (z_hat + half).round())


def test_mixed_radix_roundtrip():
    levels = [11, 9, 9]
    codes = torch.tensor([[10, 8, 8], [0, 0, 0], [5, 4, 3]])
    flat = mixed_radix_flatten(codes, levels)
    assert flat.tolist() == [10 * 81 + 8 * 9 + 8, 0, 5 * 81 + 4 * 9 + 3]
    assert int(flat.max()) < math.prod(levels)


def test_split_codes_ranges():
    coarse_idx, fine_idx = derive_coarse_fine(FSQ_LEVELS)
    levels = torch.tensor(FSQ_LEVELS, dtype=torch.float32)
    z = torch.randn(3, 16, len(FSQ_LEVELS)) * 2.0
    _z_hat, codes = fsq_quantize(z, levels)
    cidx, fidx = split_codes(codes, coarse_idx, fine_idx, list(FSQ_LEVELS))
    assert int(cidx.min()) >= 0 and int(cidx.max()) < FSQ_V_C
    assert int(fidx.min()) >= 0 and int(fidx.max()) < FSQ_V_F
