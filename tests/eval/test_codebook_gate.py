"""§7 v1.6.25 (RE-AUDIT R1) — THE CODEBOOK GATE, THROUGH THE REAL WRITE → LOAD → ASSEMBLE PATH.

**WHY THIS FILE EXISTS, AND WHY IT COULD NOT HAVE EXISTED BEFORE.** v1.6.22 declared the
"specified but not enforced" class closed for the fourth time by gating codebook utilization at the
spec's own ≥ 95 %. The gate shipped in `591ec44` **below `_validate_artifact`'s `return bad`**. It
was unreachable from the moment it was written, so the compensating control for dropping the C-4
external-validation gate never executed once — and 601 tests stayed green, because not one of them
fed the gate an artifact it was supposed to reject. A gate is not closed by the existence of its
code; it is closed by a test that fails when the code is removed.

The three rejections below are the ones the fourth instance was declared closed on:
  1. a COLLAPSED codebook (utilization 0.01) — the failure mode the threshold names;
  2. an EMPTY codebook payload — "the diagnostic ran and measured nothing";
  3. an artifact with NO `codebook` key at all — the pre-v1.6.22 schema, tampered in on disk and
     re-indexed so the content hash cannot be what refuses it.

Discrimination first: the healthy set must assemble, or all three prove nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trikaal.eval.verdict import (
    DSR_HORIZONS,
    DSR_KAPPAS,
    DSR_SEEDS,
    PRIMARY_H,
    VerdictInputError,
    assemble_verdict,
    load_cell_evals,
    write_cell_eval_artifact,
    write_eval_index,
)

T = 400
HEALTHY = {"coarse": {"utilization": 0.99, "vocab": 891}, "fine": {"utilization": 0.98}}
COLLAPSED = {"coarse": {"utilization": 0.01, "vocab": 891}, "fine": {"utilization": 0.98}}
MU_OK = {"frac_negative": 0.5, "activity_decisions": 0.5}
RECON_OK = {"ohlcv_recon_mae": 0.10, "ohlcv_recon_mae_by_dim": [0.1] * 7, "n_ohlcv_dims": 7}
DECODE_OK = {"sign_agreement_dim0": 0.95, "variance_ratio_dim0": 1.0}


def _build(out: Path, *, sick_unit: tuple[int, int] | None = None, sick_codebook=COLLAPSED) -> Path:
    """25 artifacts; ``sick_unit`` gets ``sick_codebook`` and everyone else gets HEALTHY."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    common = rng.standard_normal(T) * 0.02
    entries: dict[str, str] = {}
    for cell in (1, 2, 3, 4, 5):
        for seed in DSR_SEEDS:
            cb = sick_codebook if sick_unit == (cell, seed) else HEALTHY
            name, sha = write_cell_eval_artifact(
                out,
                cell_id=cell,
                seed=seed,
                grid={"h": PRIMARY_H, "start_ms": 1_704_132_000_000, "n_periods": T},
                headline_series=common + rng.standard_normal(T) * 0.01,
                kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
                val_ir_by_kappa_by_h={
                    h: {k: float(rng.normal(0.5, 0.3)) for k in DSR_KAPPAS} for h in DSR_HORIZONS
                },
                mu_diag=MU_OK,
                ohlcv_recon=RECON_OK,
                decode_agreement=DECODE_OK,
                codebook=cb,
                meta={"fixture": True},
            )
            entries[name] = sha
    write_eval_index(out, entries)
    return out


def _reindex(out: Path) -> None:
    """Rebuild the content index over whatever is on disk — so a tampered artifact is refused by
    the CODEBOOK check and not merely by its sha256 (fixture-discrimination rule)."""
    write_eval_index(
        out,
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.glob("cell*_eval.json"))
        },
    )


# ------------------------------------------------------------------ DISCRIMINATION FIRST
def test_a_healthy_codebook_set_assembles(tmp_path):
    """If this failed, every refusal below would be vacuous."""
    evals, shas = load_cell_evals(_build(tmp_path / "ok"))
    assert len(evals) == 25
    man = assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)
    assert man["verdict"]["primary"] in ("SURVIVES", "NULL")


# ------------------------------------------------------------------ the three rejections
@pytest.mark.parametrize("unit", [(1, 0), (4, 3), (5, 4)])
def test_ONE_collapsed_codebook_refuses_the_whole_verdict(tmp_path, unit):
    """utilization 0.01 on ONE of 25 units. The threshold exists for dead-code collapse; a
    collapsed quantizer is not a measurement of anything, in ANY cell."""
    with pytest.raises(VerdictInputError, match="utilization"):
        load_cell_evals(_build(tmp_path / f"sick{unit[0]}{unit[1]}", sick_unit=unit))


def test_an_EMPTY_codebook_payload_refuses(tmp_path):
    """ "The diagnostic ran and measured nothing" must not read as "the diagnostic passed"."""
    with pytest.raises(VerdictInputError, match="codebook"):
        load_cell_evals(_build(tmp_path / "empty", sick_unit=(2, 1), sick_codebook={}))


def test_a_MISSING_codebook_key_refuses(tmp_path):
    """The pre-v1.6.22 schema. Tampered on disk and RE-INDEXED, so the content hash agrees and the
    codebook check is the only thing left that can refuse it."""
    out = _build(tmp_path / "nokey")
    p = out / "cell3_seed2_eval.json"
    doc = json.loads(p.read_text())
    doc.pop("codebook")
    p.write_text(json.dumps(doc, sort_keys=True))
    _reindex(out)
    with pytest.raises(VerdictInputError, match="codebook"):
        load_cell_evals(out)


def test_a_codebook_missing_the_fine_leg_refuses(tmp_path):
    """Half a diagnostic. The FSQ hierarchy is coarse + fine; a payload carrying one of them
    cannot certify the other did not collapse."""
    with pytest.raises(VerdictInputError, match="fine"):
        load_cell_evals(
            _build(tmp_path / "half", sick_unit=(4, 0), sick_codebook={"coarse": HEALTHY["coarse"]})
        )


def test_a_non_finite_utilization_refuses(tmp_path):
    """NaN compares False against every threshold — the exact shape that disarmed the degeneracy
    guard in C-2. It must be caught as MISSING, never read as passing."""
    with pytest.raises(VerdictInputError, match="utilization"):
        load_cell_evals(
            _build(
                tmp_path / "nan",
                sick_unit=(5, 1),
                sick_codebook={
                    "coarse": {"utilization": float("nan")},
                    "fine": {"utilization": 0.98},
                },
            )
        )


def test_the_gate_reads_the_PINNED_constant_not_a_literal(tmp_path):
    """The threshold is `conformance.PINNED_CODEBOOK_MIN_UTILIZATION`, so moving the pin moves the
    gate — one copy of one truth (the C-6 rule)."""
    from trikaal.eval.conformance import PINNED_CODEBOOK_MIN_UTILIZATION

    assert PINNED_CODEBOOK_MIN_UTILIZATION == 0.95
    just_under = {
        "coarse": {"utilization": PINNED_CODEBOOK_MIN_UTILIZATION - 1e-6},
        "fine": {"utilization": 0.98},
    }
    just_over = {
        "coarse": {"utilization": PINNED_CODEBOOK_MIN_UTILIZATION},
        "fine": {"utilization": 0.98},
    }
    with pytest.raises(VerdictInputError, match="utilization"):
        load_cell_evals(_build(tmp_path / "under", sick_unit=(1, 1), sick_codebook=just_under))
    # and the boundary itself is INCLUSIVE — the comparison is `<`, not `<=`
    evals, _ = load_cell_evals(_build(tmp_path / "over", sick_unit=(1, 1), sick_codebook=just_over))
    assert len(evals) == 25
