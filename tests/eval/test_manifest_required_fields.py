"""§7 v1.6.25 (RE-AUDIT R4) — THE G-§8.C.3 VERDICT AND ITS DISCLOSURE REACH THE MANIFEST.

C-4 ended with Lakshay DROPPING the Kronos external-validation gate as binding. That was
conditional: the paper ships a required disclosure in its place — *"We therefore cannot exclude
that our BSQ baseline is weaker than a reference BSQ implementation, which would inflate the
FSQ-vs-BSQ comparison reported in §5."* The gate's verdict carries that sentence.

**It was computed and thrown away.** ``assemble_verdict`` called ``evaluate(...)``, handed the
result straight to ``assert_clear_to_compute_deltas``, and kept no reference. Nothing reached the
manifest, so the one compensating record for dropping a binding entry gate existed for the
duration of one expression. A missing disclosure is the single defect disclosure cannot neutralize.

The record is REQUIRED now, on both sides: ``assemble_verdict`` validates its own output, and
``load_verdict_manifest`` re-validates from the FILE — because a manifest written by an older
revision is exactly the case emission-side validation cannot see.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from trikaal.eval.external_validation import BSQ_DISCLOSURE, DROPPED
from trikaal.eval.verdict import (
    DSR_HORIZONS,
    DSR_KAPPAS,
    DSR_SEEDS,
    PRIMARY_H,
    REQUIRED_MANIFEST_FIELDS,
    VerdictInputError,
    assemble_verdict,
    load_cell_evals,
    load_verdict_manifest,
    verdict_manifest_failures,
    write_cell_eval_artifact,
    write_eval_index,
)

T = 300
CODEBOOK = {"coarse": {"utilization": 0.99}, "fine": {"utilization": 0.98}}
MU = {"frac_negative": 0.5, "activity_decisions": 0.5}
RECON = {"ohlcv_recon_mae": 0.10, "ohlcv_recon_mae_by_dim": [0.1] * 7, "n_ohlcv_dims": 7}
DECODE = {"sign_agreement_dim0": 0.95, "variance_ratio_dim0": 1.0}


def _manifest(tmp_path, *, money_verdict: bool = False) -> dict:
    out = tmp_path / "art"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    common = rng.standard_normal(T) * 0.02
    entries = {}
    for cell in (1, 2, 3, 4, 5):
        for seed in DSR_SEEDS:
            name, sha = write_cell_eval_artifact(
                out,
                cell_id=cell,
                seed=seed,
                grid={"h": PRIMARY_H, "start_ms": 1_704_132_000_000, "n_periods": T},
                headline_series=common + rng.standard_normal(T) * 0.01,
                kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
                val_ir_by_kappa_by_h={
                    h: {k: float(rng.normal(0.5, 0.2)) for k in DSR_KAPPAS} for h in DSR_HORIZONS
                },
                mu_diag=MU,
                ohlcv_recon=RECON,
                decode_agreement=DECODE,
                codebook=CODEBOOK,
                meta={"fixture": True},
            )
            entries[name] = sha
    write_eval_index(out, entries)
    evals, shas = load_cell_evals(out)
    return assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=money_verdict)


# ------------------------------------------------------------------ the record exists
def test_the_gate_verdict_and_the_BSQ_DISCLOSURE_are_IN_the_manifest(tmp_path):
    ev = _manifest(tmp_path)["external_validation"]
    assert ev["gate"]["status"] == DROPPED
    assert ev["gate_is_binding"] is False
    assert ev["required_disclosure"] == BSQ_DISCLOSURE
    assert "cannot exclude" in ev["required_disclosure"]
    # the three reasons the gate is unexecutable travel with it, not in a doc somewhere
    assert len(ev["gate"]["why"]) == 3


def test_enforcement_is_recorded_separately_from_evaluation(tmp_path):
    """A fixture run does not ENFORCE the gate, but it still RECORDS it — so "was this checked?"
    is answerable from the artifact rather than from which flags someone passed."""
    assert _manifest(tmp_path / "a")["external_validation"]["enforced_this_run"] is False
    assert (
        _manifest(tmp_path / "b", money_verdict=True)["external_validation"]["enforced_this_run"]
        is True
    )


# ------------------------------------------------------------------ the loader refuses absence
@pytest.mark.parametrize("field", REQUIRED_MANIFEST_FIELDS)
def test_removing_ANY_required_field_makes_the_manifest_unquotable(tmp_path, field):
    man = _manifest(tmp_path)
    man["grid_pinned"] = True
    assert verdict_manifest_failures(man) == []  # DISCRIMINATION FIRST
    man.pop(field)
    fails = verdict_manifest_failures(man)
    assert fails and any(field in f for f in fails)


def test_an_EMPTY_required_field_is_as_bad_as_a_missing_one(tmp_path):
    man = _manifest(tmp_path)
    man["grid_pinned"] = True
    man["external_validation"] = {}
    assert verdict_manifest_failures(man)


def test_the_disclosure_specifically_cannot_be_blanked(tmp_path):
    """The field can be present and still say nothing — the C-2 shape. An empty string is a
    missing disclosure wearing the right key."""
    man = _manifest(tmp_path)
    man["grid_pinned"] = True
    for blank in ("", "   "):
        man["external_validation"]["required_disclosure"] = blank
        fails = verdict_manifest_failures(man)
        assert fails and any("required_disclosure" in f for f in fails)


def test_load_verdict_manifest_refuses_from_the_FILE(tmp_path):
    """Emission-side validation cannot see a manifest written by an older revision; this can."""
    man = _manifest(tmp_path)
    man["grid_pinned"] = True
    p = tmp_path / "verdict.json"
    p.write_text(json.dumps(man, sort_keys=True))
    assert load_verdict_manifest(p)["content_hash"] == man["content_hash"]

    man.pop("external_validation")
    p.write_text(json.dumps(man, sort_keys=True))
    with pytest.raises(VerdictInputError, match="external_validation"):
        load_verdict_manifest(p)


def test_a_toy_grid_manifest_is_refused_by_the_loader(tmp_path):
    """`grid_pinned` was already prose-level policy ("can never be quoted as the M6 outcome").
    The loader enforces it."""
    man = _manifest(tmp_path)
    man["grid_pinned"] = False
    p = tmp_path / "toy.json"
    p.write_text(json.dumps(man, sort_keys=True))
    with pytest.raises(VerdictInputError, match="grid_pinned"):
        load_verdict_manifest(p)
    assert load_verdict_manifest(p, require_grid_pinned=False)["grid_pinned"] is False
