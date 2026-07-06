"""Input-arm selection + the Cell-5 shuffled-micro transform (m6_design §1, §6 items 2-3).

The 2×2 input axis: the **OHLCV-only arm** feeds the tokenizer the ``OHLCV_ONLY_IDX`` subset
(F=7) — the micro dims are **REMOVED, never zero-filled** (zero-fill leaks "micro absent" and
mismatches capacity; m6_design §1). The **+micro arm** feeds all 16 (funding/OI carried but
masked — the tripwire enforces that). **Cell 5** is the +micro arm with the microstructure
channels temporally permuted within each symbol-segment.

The Cell-5 shuffle is ONE function (:func:`shuffle_micro`) with a per-``(seed, symbol)`` seed
derivation, called by BOTH the training path and the eval driver — "applied identically in train
and eval" (m6_design §1) is enforced structurally, not by convention. It reuses
``placebo.block_time_permute`` (the KAT'd surrogate) and wires ``assert_perp_dims_masked`` so a
future funding/OI activation fails loudly instead of silently under-shuffling.
"""

from __future__ import annotations

import zlib

import numpy as np

from trikaal.constants import N_FEATURES, OHLCV_ONLY_IDX
from trikaal.eval.placebo import assert_perp_dims_masked, block_time_permute

ARM_OHLCV = "ohlcv"
ARM_MICRO = "micro"
ARM_MICRO_SHUFFLED = "micro_shuffled"
ARMS = (ARM_OHLCV, ARM_MICRO, ARM_MICRO_SHUFFLED)


def arm_feature_idx(arm: str) -> tuple[int, ...]:
    """The feature columns an arm's tokenizer SEES (its input dimensionality = len of this)."""
    if arm == ARM_OHLCV:
        return OHLCV_ONLY_IDX  # F=7: price-shape + log-volume/amount; micro dims REMOVED
    if arm in (ARM_MICRO, ARM_MICRO_SHUFFLED):
        return tuple(range(N_FEATURES))  # F=16 (funding/OI carried-but-masked)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def arm_n_features(arm: str) -> int:
    return len(arm_feature_idx(arm))


def cell5_seed(seed: int, symbol: str) -> int:
    """Deterministic per-(run-seed, symbol) shuffle seed — identical wherever it is derived.

    crc32 is stable across processes/platforms (unlike ``hash``), so train and eval agree."""
    return (int(seed) << 16) ^ zlib.crc32(symbol.encode())


def shuffle_micro(
    x: np.ndarray, mask: np.ndarray, segment_id: np.ndarray, *, symbol: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """The Cell-5 transform: per-segment temporal block-permute of the micro channels.

    Returns the shuffled ``(x, mask)`` — the (value, mask) pair moves together under one
    per-segment permutation (2026-07-06 audit fix; values-only shuffling strands fill values
    on mask-active bars, a one-directional Cell-5 handicap). Marginal distribution preserved,
    temporal information destroyed (the placebo). Deterministic under ``(seed, symbol)``; the
    SAME call sits in the training loader and the eval driver. The funding/OI tripwire runs
    first — if those dims ever activate, the shuffle-dim set under-shuffles and MUST fail
    loudly (m6_design §6 item 10f)."""
    assert_perp_dims_masked(mask)
    return block_time_permute(x, mask, segment_id, seed=cell5_seed(seed, symbol))


def select_arm(x: np.ndarray, mask: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Column-select the arm's feature subset from ``(x[T,16], mask[T,16])`` — a REMOVAL.

    Returns contiguous copies with F = ``arm_n_features(arm)`` columns. For Cell 5 apply
    :func:`shuffle_micro` BEFORE selection (the shuffle needs the full 16-dim layout)."""
    idx = list(arm_feature_idx(arm))
    return np.ascontiguousarray(x[:, idx]), np.ascontiguousarray(mask[:, idx])


__all__ = [
    "ARMS",
    "ARM_MICRO",
    "ARM_MICRO_SHUFFLED",
    "ARM_OHLCV",
    "arm_feature_idx",
    "arm_n_features",
    "cell5_seed",
    "select_arm",
    "shuffle_micro",
]
