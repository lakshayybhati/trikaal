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
# §7 v1.6.27 (C-12 bracket): the CONSTANT-micro arm. Same 16-dim width as cells 4/5, micro slots
# 7-12 pinned to a constant. Constant compresses to ~nothing, so this arm pays ~NO capacity for
# its micro channels where Cell 5 pays the maximum (incompressible noise) — it is the ADVANTAGED
# end of the bracket, not a second placebo.
ARM_MICRO_CONSTANT = "micro_constant"
ARMS = (ARM_OHLCV, ARM_MICRO, ARM_MICRO_SHUFFLED, ARM_MICRO_CONSTANT)

# ★ §7 v1.6.27 — WHICH ARMS THE MICRO-LEGIBILITY GATE APPLIES TO, EXHAUSTIVELY AND BY NAME.
#
# The gate previously fired on `spec.arm in ("micro", "micro_shuffled")` at
# `orchestrator.py:328`. That is an INCLUSION TEST, so **a newly added arm falls through it
# silently** — the gate would not run on the constant arm and nothing would say so. That is
# exactly the C-10 shape ("skipped is a third state and it HALTS"), one level up: not a skipped
# DIM but a skipped CELL. So membership is a REGISTRY with a reason per arm, an unknown arm is a
# hard error rather than a default, and an exemption is RECORDED in the run manifest.
#
# The constant arm's exemption is not leniency — the gate is UNDEFINED on it. It measures
# `sign(x_t[dim])` recovery, i.e. above-vs-below the causal rolling mean; **a constant channel
# has no sign to recover** and its "accuracy" would be whatever the degenerate class balance
# happened to be. Reporting that number as a pass would be worse than not running it.
MICRO_LEGIBILITY_APPLIES: dict[str, bool] = {
    ARM_OHLCV: False,  # dims 7-12 are REMOVED, not present-but-uninformative
    ARM_MICRO: True,
    ARM_MICRO_SHUFFLED: True,  # the shuffle preserves each dim's marginal, so sign is defined
    ARM_MICRO_CONSTANT: False,  # UNDEFINED: a constant channel has no sign
}

# An exemption must SAY WHY, in the run manifest, per arm. "Absent" is not "considered".
MICRO_LEGIBILITY_EXEMPTION_REASON: dict[str, str] = {
    ARM_OHLCV: "dims 7-12 are REMOVED from this arm's input (F=7), so there is no channel whose "
    "per-bar legibility could be measured",
    ARM_MICRO_CONSTANT: "the micro channels are CONSTANT, and the gate measures recovery of "
    "sign(x_t[dim]) = above-vs-below the causal rolling mean. A constant channel HAS NO SIGN, so "
    "the statistic is UNDEFINED rather than lenient — reporting it as a pass would be worse than "
    "not running it. This exemption is keyed on the ARM, never on 'the dim looked degenerate'",
}


def arm_feature_idx(arm: str) -> tuple[int, ...]:
    """The feature columns an arm's tokenizer SEES (its input dimensionality = len of this)."""
    if arm == ARM_OHLCV:
        return OHLCV_ONLY_IDX  # F=7: price-shape + log-volume/amount; micro dims REMOVED
    if arm in (ARM_MICRO, ARM_MICRO_SHUFFLED, ARM_MICRO_CONSTANT):
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


def micro_legibility_applies(arm: str) -> bool:
    """Does the per-bar micro-legibility gate apply to this arm? Reads the registry, never infers.

    An unregistered arm is a HARD ERROR. The predecessor was `arm in ("micro", "micro_shuffled")`
    inline at the call site, so adding an arm silently disabled the gate for it — the C-10 defect
    at cell scope instead of dim scope."""
    if arm not in MICRO_LEGIBILITY_APPLIES:
        raise ValueError(
            f"arm {arm!r} is not in MICRO_LEGIBILITY_APPLIES — a new arm must be CLASSIFIED "
            "deliberately (gate applies / gate undefined + why), never left to fall through an "
            "inclusion test and disable the gate in silence (§7 v1.6.27)"
        )
    return MICRO_LEGIBILITY_APPLIES[arm]


def constant_micro(
    x: np.ndarray, mask: np.ndarray, *, micro_dims: tuple[int, ...] = (7, 8, 9, 10, 11, 12)
) -> tuple[np.ndarray, np.ndarray]:
    """The Cell-6 transform: pin the micro channels to a CONSTANT, keeping the 16-dim width.

    **Zero, not the per-dim mean.** Every micro feature enters Z-SCORED (§3.2), so 0.0 IS the
    causal rolling mean — the least surprising constant, and one that requires no statistic
    computed over the series (which would itself leak). The (value, mask) pair is left mask-aligned
    the way ``shuffle_micro`` keeps them together.

    The funding/OI tripwire runs first for the same reason it does in the shuffle: dims 13-15 are
    masked everywhere in the v1 lake, and this transform must not be the thing that hides an
    activation. Note the tripwire guards 13-15 while this touches 7-12 — DISJOINT, verified, so it
    can never mask the very dims it is asserting about.
    """
    assert_perp_dims_masked(mask)
    out = np.array(x, copy=True)
    out[:, list(micro_dims)] = 0.0
    return out, np.array(mask, copy=True)


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
    "MICRO_LEGIBILITY_APPLIES",
    "MICRO_LEGIBILITY_EXEMPTION_REASON",
    "arm_feature_idx",
    "arm_n_features",
    "cell5_seed",
    "constant_micro",
    "micro_legibility_applies",
    "select_arm",
    "shuffle_micro",
]
