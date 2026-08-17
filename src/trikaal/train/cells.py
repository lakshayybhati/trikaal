"""The 5-cell matrix {quantizer × input arm} (m6_design §1) + per-cell construction gates.

| Cell | Quantizer | Arm             | Role                                     |
|------|-----------|-----------------|------------------------------------------|
| 1    | BSQ       | ohlcv           | our BSQ baseline — NOT externally valid. |
| 2    | FSQ       | ohlcv           | FSQ-vs-BSQ tokenizer effect              |
| 3    | BSQ       | micro           | micro under BSQ                          |
| 4    | FSQ       | micro           | the hero                                 |
| 5    | FSQ       | micro_shuffled  | placebo (capacity kept, info destroyed)  |

★ CELL 1 IS NOT EXTERNALLY VALIDATED, and this table said it was. The §8.C.3 Kronos gate was found
unexecutable and DROPPED AS BINDING (2026-08-03, prereg §7 v1.6.22); no Kronos weights were ever
pulled and invariant 8 forbids it. CLAUDE.md has called the old wording "a false statement of what
validation we performed" since then — and this docstring, which defines the cell matrix every arm
is built from, kept asserting it. Found by grepping the SUPERSEDED string rather than re-reading
the corrected files (``tests/test_superseded_strings.py``); two hand-sweeps and two audits had
missed it. The compensating control is a required disclosure in every verdict manifest.

Construction gates (asserted before any training step): the AR backbone at Kronos_small dims must
realize EXACTLY 21,301,248 base params per cell (invariant 4's matched-backbone requirement —
note this is the backbone EXCLUDING the MTP heads, a named secondary; the realized checkpoint
total is 31,795,200 FSQ / 31,725,568 BSQ), and the BSQ/FSQ tokenizer arms must pass G-parity
(bpt + param matching, ``parity.check_arm_parity``).
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
    """The cell's tokenizer: quantizer from the spec, input width from the arm (F=7 or 16).

    ``encoder_causal=True`` is FORCED for every M6 cell (prereg §7 v1.3, supervisor
    adjudication of the token-causality finding): bidirectional encoding puts future-bar
    content into the pre-tokenized eval context (measured 41.8 %/28.3 % past-token flips on
    the real lake), and invariant 2 binds eval INPUTS. Symmetric across all five cells, so
    bpt fairness and the paired design are unaffected; the flag changes no tensor. A caller
    passing ``encoder_causal=False`` is a hard error, never a silent override.

    ``fine_pointwise=True`` is likewise FORCED (prereg §7 v1.4, supervisor adjudication of
    the token-control programme): the fine subtoken is a PER-BAR encoding of bar t's own
    features — the causal contextual encoder was measured to smear per-bar feature state
    forward across later tokens' ids (per-bar id visibility ~= chance), so feature-space
    conditionals arrived per-bar-illegible to the AR. Identical for both quantizers and all
    arms; bpt parity and the cell design unchanged. ``fine_pointwise=False`` is a hard
    error, never a silent override."""
    if tok_kwargs.get("encoder_causal") is False:
        raise ValueError(
            "M6 cells are causal-encoder ONLY (prereg §7 v1.3) — a bidirectional cell "
            "tokenizer is not constructible through build_cell_tokenizer"
        )
    if tok_kwargs.get("fine_pointwise") is False:
        raise ValueError(
            "M6 cells are pointwise-fine ONLY (prereg §7 v1.4) — a contextual-fine cell "
            "tokenizer is not constructible through build_cell_tokenizer"
        )
    lam = tok_kwargs.get("micro_point_weight")
    if lam is not None and lam != 3.0:
        raise ValueError(
            "M6 cells pin micro_point_weight = 3.0 (prereg §7 v1.4.1, calibrated not "
            f"chosen) — got {lam!r}; a different lambda is not constructible through "
            "build_cell_tokenizer"
        )
    tok_kwargs["encoder_causal"] = True
    tok_kwargs["fine_pointwise"] = True
    tok_kwargs["micro_point_weight"] = 3.0
    return TokenizerAE(quantizer=spec.quantizer, n_features=arm_n_features(spec.arm), **tok_kwargs)


def build_calibration_tokenizer(spec: CellSpec, *, micro_point_weight: float, **tok_kwargs):
    """§7 v1.5 item E — a tokenizer at a NON-PINNED lambda, for the ONCE-ONLY re-derivation.

    ★ THIS EXISTS SO THAT `build_cell_tokenizer` DOES NOT HAVE TO CHANGE. The money path's pin is
    the thing that guards the experiment; it does not move to let a diagnostic run. This is a
    SEPARATE, EXPLICITLY-NAMED constructor, and the only caller is a run that has declared
    ``money_run=False`` (enforced in ``orchestrator.run_cell``, not merely documented here).

    Everything except lambda is IDENTICAL to `build_cell_tokenizer`: same quantizer, same feature
    count for the arm, `encoder_causal=True`, `fine_pointwise=True`. Item E permits ONE axis to
    move and this moves exactly that one — no level reallocation, no bit re-split, no change to
    total bits.

    A tokenizer built here CANNOT pass `conformance.cell_tokenizer_failures`, which compares
    `micro_point_weight` against `PINNED_MICRO_POINT_WEIGHT` — so an artifact produced from it is
    refused by the money surface by construction, not by anyone remembering to check."""
    if micro_point_weight == 3.0:
        raise ValueError(
            "build_calibration_tokenizer is for NON-PINNED lambda only; lambda=3.0 is the pinned "
            "money value and must be built through build_cell_tokenizer"
        )
    tok_kwargs["encoder_causal"] = True
    tok_kwargs["fine_pointwise"] = True
    tok_kwargs["micro_point_weight"] = float(micro_point_weight)
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
    bpt + param parity. Call at orchestration start, before any cell trains. The pair is
    built under the same forced ``encoder_causal=True`` (§7 v1.3) and ``fine_pointwise=True``
    (§7 v1.4) as the cells themselves — the pointwise branch adds identical parameters to
    both quantizer arms, so parity is asserted on the configuration the run actually uses."""
    tok_kwargs = {
        **tok_kwargs,
        "encoder_causal": True,
        "fine_pointwise": True,
        "micro_point_weight": 3.0,
    }
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
