"""§7 v1.6 C-12 M1 (supervisor-adopted) — the placebo capacity disclosure.

WHAT IT BOUNDS. The Cell-5 permutation destroys the contemporaneous micro↔OHLCV dependence as well
as the intended temporal alignment (**measured 0.2037 → 0.0005**), so Cell 5's micro channels are
six dims of INDEPENDENT NOISE rather than "Cell 4 minus information". Noise with a preserved
marginal is incompressible and shares no structure with OHLCV, so bits spent on it are bits taken
from OHLCV — and ``micro_point_weight = 3.0`` aims triple gradient pressure at fitting it. ΔIR(4−5)
therefore carries (information) + (capacity handicap), inseparable from the contrast alone.

The OHLCV dims are byte-identical across arms — ``block_time_permute`` never touches them — so
OHLCV reconstruction is like-for-like, and Cell 5 doing it worse is capacity diverted. That turns
an unbounded confound into a measured magnitude.

TWO PROPERTIES ARE PINNED HERE AND THEY PULL IN OPPOSITE DIRECTIONS, which is the point:
  * it is **REQUIRED** — an optional disclosure is decoration (the C-2 lesson);
  * it is **NON-GATING** — it must never be able to move a verdict word.
A test suite that only proved the first would be pinning a gate by accident.
"""

from __future__ import annotations

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
    placebo_capacity_disclosure,
    write_cell_eval_artifact,
    write_eval_index,
)

T = 600
BENIGN_MU = {"frac_negative": 0.5, "activity_decisions": 0.5}
OK = {"ohlcv_recon_mae": 0.10, "ohlcv_recon_mae_by_dim": [0.1] * 7, "n_ohlcv_dims": 7}


def _recon(mae: float) -> dict:
    return {"ohlcv_recon_mae": mae, "ohlcv_recon_mae_by_dim": [mae] * 7, "n_ohlcv_dims": 7}


def _fixture(tmp: Path, recon_for) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    common = rng.standard_normal(T) * 1e-3
    entries = {}
    for c in (1, 2, 3, 4, 5):
        for s in DSR_SEEDS:
            name, sha = write_cell_eval_artifact(
                tmp,
                cell_id=c,
                seed=s,
                grid={"h": PRIMARY_H, "start_ms": 0, "n_periods": T},
                headline_series=common + rng.standard_normal(T) * 1e-3,
                kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
                val_ir_by_kappa_by_h={
                    h: {k: float(rng.normal(0.4, 0.2)) for k in DSR_KAPPAS} for h in DSR_HORIZONS
                },
                mu_diag=BENIGN_MU,
                ohlcv_recon=recon_for(c, s),
                meta={},
            )
            entries[name] = sha
    write_eval_index(tmp, entries)
    return tmp


# ------------------------------------------------------------------ REQUIRED
def test_the_disclosure_is_a_required_keyword(tmp_path):
    with pytest.raises(TypeError):
        write_cell_eval_artifact(
            tmp_path,
            cell_id=1,
            seed=0,
            grid={"h": PRIMARY_H, "start_ms": 0, "n_periods": 8},
            headline_series=np.zeros(8),
            kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
            val_ir_by_kappa_by_h={h: {k: 0.5 for k in DSR_KAPPAS} for h in DSR_HORIZONS},
            mu_diag=BENIGN_MU,
        )


@pytest.mark.parametrize("payload", [{}, None, {"wrong_key": 1.0}], ids=["empty", "none", "wrong"])
def test_the_writer_refuses_an_unusable_disclosure(tmp_path, payload):
    with pytest.raises(VerdictInputError, match="REQUIRED beside the headline"):
        write_cell_eval_artifact(
            tmp_path,
            cell_id=1,
            seed=0,
            grid={"h": PRIMARY_H, "start_ms": 0, "n_periods": 8},
            headline_series=np.zeros(8),
            kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
            val_ir_by_kappa_by_h={h: {k: 0.5 for k in DSR_KAPPAS} for h in DSR_HORIZONS},
            mu_diag=BENIGN_MU,
            ohlcv_recon=payload,
        )
    assert not list(tmp_path.glob("*.json"))


def test_the_loader_refuses_an_on_disk_artifact_without_it(tmp_path):
    """The writer can be bypassed by a hand-edited or pre-v1.6 file; the loader must still refuse."""
    import hashlib

    d = _fixture(tmp_path / "f", lambda c, s: OK)
    victim = d / "cell5_seed0_eval.json"
    doc = json.loads(victim.read_text())
    doc.pop("ohlcv_recon")
    victim.write_text(json.dumps(doc, sort_keys=True))
    ent = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.glob("cell*_eval.json")}
    write_eval_index(d, ent)
    with pytest.raises(VerdictInputError, match="ohlcv_recon"):
        load_cell_evals(d)


# ------------------------------------------------------------------ it MEASURES something
def test_the_disclosure_is_computed_not_restated(tmp_path):
    """MUTATION: move cell 5's OHLCV recon and the reported ratio must move with it."""
    clean = load_cell_evals(_fixture(tmp_path / "a", lambda c, s: OK))[0]
    d0 = placebo_capacity_disclosure(clean)
    assert d0["cell5_over_cell4_ratio"] == pytest.approx(1.0)
    assert d0["excess_ohlcv_recon_error_of_the_placebo"] == pytest.approx(0.0)

    handicapped = load_cell_evals(
        _fixture(tmp_path / "b", lambda c, s: _recon(0.30 if c == 5 else 0.10))
    )[0]
    d1 = placebo_capacity_disclosure(handicapped)
    assert d1["cell5_over_cell4_ratio"] == pytest.approx(3.0), (
        "the disclosure must be COMPUTED from the artifacts, not restated"
    )
    assert d1["excess_ohlcv_recon_error_of_the_placebo"] == pytest.approx(0.20)


def test_it_appears_in_the_shipped_manifest(tmp_path):
    """REQUIRED IN THE RESULTS, not an appendix — asserted on the emitted manifest."""
    evals, shas = load_cell_evals(_fixture(tmp_path / "m", lambda c, s: OK))
    man = assemble_verdict(evals, shas, tabled_mde_h15=3.518)
    assert "placebo_capacity_disclosure" in man, "the disclosure is not in the results"
    d = man["placebo_capacity_disclosure"]
    assert d["non_gating"] is True
    assert "ohlcv_recon_mae_cell5_placebo" in d and "cell5_over_cell4_ratio" in d


# ------------------------------------------------------------------ NON-GATING
def test_the_disclosure_can_never_move_the_verdict(tmp_path):
    """The other half. A disclosure that could flip a word would be a clause in disguise.

    Same series, same mu_diag, same everything the clauses read — only the OHLCV-recon numbers
    differ, benign versus a 10x placebo handicap. Every clause, the primary word, and the
    fallback must be IDENTICAL."""
    a_ev, a_sh = load_cell_evals(_fixture(tmp_path / "x", lambda c, s: OK))
    b_ev, b_sh = load_cell_evals(
        _fixture(tmp_path / "y", lambda c, s: _recon(1.0 if c == 5 else 0.10))
    )
    ma = assemble_verdict(a_ev, a_sh, tabled_mde_h15=3.518)
    mb = assemble_verdict(b_ev, b_sh, tabled_mde_h15=3.518)

    assert ma["verdict"]["emitted"] == mb["verdict"]["emitted"]
    assert ma["verdict"]["primary"] == mb["verdict"]["primary"]
    assert ma["verdict"]["failing_clauses"] == mb["verdict"]["failing_clauses"]
    for name in ma["clauses"]:
        assert ma["clauses"][name]["pass"] == mb["clauses"][name]["pass"], name
    # and the disclosure itself DID differ, or this test proved nothing
    assert (
        ma["placebo_capacity_disclosure"]["cell5_over_cell4_ratio"]
        != mb["placebo_capacity_disclosure"]["cell5_over_cell4_ratio"]
    ), "FIXTURE DISCRIMINATION: the two fixtures must actually differ in the disclosure"


def test_no_threshold_is_attached_to_it():
    """A disclosure with a bar is a clause. There must be no comparison to any constant."""
    src = (Path(__file__).resolve().parents[2] / "src/trikaal/eval/verdict.py").read_text()
    body = src[src.index("def placebo_capacity_disclosure") : src.index("def provenance_failures")]
    for token in ("pass", ">=", "<=", "ECON_FLOOR", "THRESHOLD"):
        assert token not in body.replace("passes", ""), (
            f"placebo_capacity_disclosure contains {token!r} — it must report a magnitude, "
            "never adjudicate one"
        )
