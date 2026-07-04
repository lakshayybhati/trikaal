"""The 5-cell matrix {quantizer × input arm} (m6_design §1) + per-cell construction gates.

| Cell | Quantizer | Arm             | Role                                     |
|------|-----------|-----------------|------------------------------------------|
| 1    | BSQ       | ohlcv           | baseline (externally validated §8.C.3)   |
| 2    | FSQ       | ohlcv           | FSQ-vs-BSQ tokenizer effect              |
| 3    | BSQ       | micro           | micro under BSQ                          |
| 4    | FSQ       | micro           | the hero                                 |
| 5    | FSQ       | micro_shuffled  | placebo (capacity kept, info destroyed)  |

Construction gates (asserted before any training step): the AR backbone at Kronos_small dims must
realize EXACTLY 21,301,248 base params per cell (invariant 4's matched-backbone requirement), and
the BSQ/FSQ tokenizer arms must pass G-parity (bpt + param matching, ``parity.check_arm_parity``).
"""

from __future__ import annotations

from dataclasses import dataclass

from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.tokenizer.parity import check_arm_parity
from trikaal.train.arms import ARM_MICRO, ARM_MICRO_SHUFFLED, ARM_OHLCV, arm_n_features

KRONOS_SMALL_BASE_PARAMS = 21_301_248


@dataclass(frozen=True)
class CellSpec:
    cell_id: int
    quantizer: str  # "bsq" | "fsq"
    arm: str  # "ohlcv" | "micro" | "micro_shuffled"

    @property
    def name(self) -> str:
        return f"cell{self.cell_id}_{self.quantizer}_{self.arm}"


CELLS: tuple[CellSpec, ...] = (
    CellSpec(1, "bsq", ARM_OHLCV),
    CellSpec(2, "fsq", ARM_OHLCV),
    CellSpec(3, "bsq", ARM_MICRO),
    CellSpec(4, "fsq", ARM_MICRO),
    CellSpec(5, "fsq", ARM_MICRO_SHUFFLED),
)


def build_cell_tokenizer(spec: CellSpec, **tok_kwargs) -> TokenizerAE:
    """The cell's tokenizer: quantizer from the spec, input width from the arm (F=7 or 16)."""
    return TokenizerAE(quantizer=spec.quantizer, n_features=arm_n_features(spec.arm), **tok_kwargs)


def build_cell_backbone(
    v_c: int,
    v_f: int,
    *,
    expect_base_params: int | None = KRONOS_SMALL_BASE_PARAMS,
    **backbone_kwargs,
) -> TrikaalAR:
    """The cell's AR backbone over the cell tokenizer's (V_c, V_f).

    ``expect_base_params`` asserts the realized base-parameter count at construction (default:
    the pinned Kronos_small 21,301,248 — invariant 4's matched-backbone requirement). Pass None
    ONLY for deliberately-tiny smoke shells (the canonical count stays pinned by the KATs).

    NOTE: BSQ cells have V_c=V_f=1024 vs FSQ's 891/1225 → embedding/head rows differ slightly
    (2048 vs 2116 rows); the 21.3M pin applies at the FSQ canonical vocab, so BSQ cells assert
    the ±2% band around it instead of exact equality (the bpt-parity gate is what matches the
    arms' token budgets — invariant 6)."""
    model = TrikaalAR(v_c=v_c, v_f=v_f, **backbone_kwargs)
    if expect_base_params is not None:
        n = model.num_backbone_params()
        if (v_c, v_f) == (891, 1225):
            assert n == expect_base_params, (
                f"backbone base params {n:,} != pinned {expect_base_params:,} (Kronos_small)"
            )
        else:
            lo, hi = 0.98 * expect_base_params, 1.02 * expect_base_params
            assert lo <= n <= hi, (
                f"backbone base params {n:,} outside ±2% of {expect_base_params:,} "
                f"(vocab ({v_c},{v_f}))"
            )
    return model


def assert_cells_parity(**tok_kwargs) -> dict[str, float]:
    """G-parity across the two quantizer arms at the given shell dims (m6_design §3).

    Constructs a canonical FSQ-vs-BSQ pair with the run's shell kwargs and asserts
    bpt + param parity. Call at orchestration start, before any cell trains."""
    fsq = TokenizerAE(quantizer="fsq", **tok_kwargs)
    bsq = TokenizerAE(quantizer="bsq", **tok_kwargs)
    return check_arm_parity(fsq, bsq)


__all__ = [
    "CELLS",
    "KRONOS_SMALL_BASE_PARAMS",
    "CellSpec",
    "assert_cells_parity",
    "build_cell_backbone",
    "build_cell_tokenizer",
]
