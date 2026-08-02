"""§7 v1.6 C-2 — the KATs that were missing when the degeneracy guard could be silently disarmed.

THE DEFECT THESE PIN. ``write_cell_eval_artifact`` accepted ``mu_diag=None`` and persisted ``{}``;
``_validate_artifact`` never looked at it; ``degeneracy_guard`` read the two scalars it needs via
``.get(..., nan)``, every NaN comparison evaluated False, and the guard returned a HARDCODED
``"armed": True`` with ``halted: False`` having examined nothing. ``m6_toy_rehearsal.py`` — the
driver the end-to-end validation run goes through — never passed ``mu_diag``. That run would have
produced exactly its stated deliverable, a verdict word with ``dual_specification``, while the
degeneracy HALT gate was dead. A check that cannot fail is not a check.

WHAT IS ASSERTED HERE, in the order the failure would have to pass through:
  1. the WRITER refuses (required kwarg; empty / missing-key / non-finite payloads raise)
  2. the READER refuses (an on-disk artifact without mu_diag is rejected by ``load_cell_evals``)
  3. the GUARD refuses (unreadable inputs → ``armed: False`` AND ``halted: True``, fail-closed)
  4. the GUARD STILL DISCRIMINATES (benign → no halt, degenerate → halt), so 3 is not passing by
     halting on everything
  5. every CALL SITE forwards it, checked on the AST — the "flag accepted but never forwarded"
     class, which is what actually bit here

Item 5's checker is itself mutation-tested against a synthetic call that omits ``mu_diag``: an
assertion that cannot fail would reproduce the very defect this file exists to close.
"""

from __future__ import annotations

import ast
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
    degeneracy_guard,
    load_cell_evals,
    write_cell_eval_artifact,
    write_eval_index,
)

# §7 v1.6.22: codebook is a REQUIRED per-(cell, seed) field, gated at 95% utilization.
CODEBOOK_OK = {
    "coarse": {"utilization": 0.99, "vocab": 891},
    "fine": {"utilization": 0.98, "vocab": 1225},
}

# §7 v1.6.11 C-1: the decode-agreement disclosure is REQUIRED on every artifact.
DECODE_AGREEMENT_OK = {
    "sign_agreement_dim0": 0.95,
    "variance_ratio_dim0": 1.0,
    "single_bar_degenerate": False,
}


# §7 v1.6 C-12 M1: the capacity disclosure is REQUIRED on every artifact.
OHLCV_RECON_OK = {"ohlcv_recon_mae": 0.10, "ohlcv_recon_mae_by_dim": [0.1] * 7, "n_ohlcv_dims": 7}


REPO = Path(__file__).resolve().parents[2]
BENIGN = {"frac_negative": 0.5, "activity_decisions": 0.5}
DEGENERATE = {"frac_negative": 0.999, "activity_decisions": 0.5}
T = 128


def _write(out_dir: Path, *, cell_id: int, seed: int, mu_diag: dict, mu: float = 0.0):
    rng = np.random.default_rng(1000 * cell_id + seed)
    return write_cell_eval_artifact(
        out_dir,
        cell_id=cell_id,
        seed=seed,
        grid={"h": PRIMARY_H, "start_ms": 0, "n_periods": T},
        headline_series=rng.standard_normal(T) * 1e-3 + mu,
        kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
        val_ir_by_kappa_by_h={h: {k: 0.5 for k in DSR_KAPPAS} for h in DSR_HORIZONS},
        mu_diag=mu_diag,
        ohlcv_recon=OHLCV_RECON_OK,
        decode_agreement=DECODE_AGREEMENT_OK,
        meta={},
        codebook=CODEBOOK_OK,
    )


def _fixture(out_dir: Path, mu_diag_for) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for c in (1, 2, 3, 4, 5):
        for s in DSR_SEEDS:
            name, sha = _write(out_dir, cell_id=c, seed=s, mu_diag=mu_diag_for(c, s))
            entries[name] = sha
    write_eval_index(out_dir, entries)
    return load_cell_evals(out_dir)[0]


# --------------------------------------------------------------- 1. the WRITER refuses
def test_mu_diag_is_a_required_keyword(tmp_path):
    """Omitting it is a TypeError, not a silent ``{}`` — the defect's entry point."""
    with pytest.raises(TypeError):
        write_cell_eval_artifact(
            tmp_path,
            cell_id=1,
            seed=0,
            grid={"h": PRIMARY_H, "start_ms": 0, "n_periods": T},
            headline_series=np.zeros(T),
            kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
            val_ir_by_kappa_by_h={h: {k: 0.5 for k in DSR_KAPPAS} for h in DSR_HORIZONS},
            codebook=CODEBOOK_OK,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        None,
        {"frac_negative": 0.5},  # activity_decisions missing → the filter leg would no-op
        {"activity_decisions": 0.5},  # frac_negative missing → the sign leg would no-op
        {"frac_negative": float("nan"), "activity_decisions": 0.5},  # NaN is what disarmed it
        {"frac_negative": 0.5, "activity_decisions": float("nan")},
    ],
    ids=["empty", "none", "no_activity", "no_frac_neg", "nan_frac_neg", "nan_activity"],
)
def test_writer_rejects_unreadable_mu_diag(tmp_path, payload):
    with pytest.raises(VerdictInputError, match="degeneracy guard could not read"):
        _write(tmp_path, cell_id=1, seed=0, mu_diag=payload)
    assert not list(tmp_path.glob("*.json")), "a rejected artifact must not reach disk"


# --------------------------------------------------------------- 2. the READER refuses
def test_loader_rejects_an_on_disk_artifact_without_mu_diag(tmp_path):
    """The writer can be bypassed (hand-edited or pre-v1.6 file); the loader must still refuse."""
    entries = {}
    for c in (1, 2, 3, 4, 5):
        for s in DSR_SEEDS:
            name, sha = _write(tmp_path, cell_id=c, seed=s, mu_diag=BENIGN)
            entries[name] = sha
    victim = tmp_path / "cell4_seed0_eval.json"
    doc = json.loads(victim.read_text())
    doc.pop("mu_diag")
    victim.write_text(json.dumps(doc, sort_keys=True))
    import hashlib

    entries[victim.name] = hashlib.sha256(victim.read_bytes()).hexdigest()  # re-index: not a hash
    write_eval_index(tmp_path, entries)  # failure, so the mu_diag leg is what must catch it
    with pytest.raises(VerdictInputError, match="mu_diag"):
        load_cell_evals(tmp_path)


# --------------------------------------------------------------- 3. the GUARD fails closed
def test_guard_never_reports_armed_having_read_nothing():
    """``armed`` is a measurement now. Unreadable inputs must HALT, never pass silently."""
    evals = {
        (c, s): {"mu_diag": {}, "headline_series": [0.0] * T}
        for c in (1, 2, 3, 4, 5)
        for s in DSR_SEEDS
    }
    g = degeneracy_guard(evals)
    assert g["armed"] is False, "a guard that examined nothing must not call itself armed"
    assert g["halted"] is True, "fail-closed: unreadable inputs are a HALT, not a pass"
    assert g["unreadable_inputs"], "the guard must name what it could not read"
    # the exact historical signature: armed AND not halted AND nothing examined
    assert not (
        g["armed"] and not g["halted"] and not any(pc["reasons"] for pc in g["per_cell"].values())
    ), "reproduces the v1.5 defect: armed:true, halted:false, nothing examined"


def test_guard_halts_when_only_one_seed_is_unreadable(tmp_path):
    """Partial coverage is still blindness — the per-seed leg is what the pin exists for."""
    evals = _fixture(tmp_path, lambda c, s: BENIGN)
    assert degeneracy_guard(evals)["halted"] is False
    evals[(4, DSR_SEEDS[1])]["mu_diag"] = {"activity_decisions": 0.5}  # drop frac_negative
    g = degeneracy_guard(evals)
    assert g["armed"] is False and g["halted"] is True
    assert any("cell4_seed" in u and "frac_negative" in u for u in g["unreadable_inputs"])


# --------------------------------------------------------------- 4. it STILL discriminates
def test_guard_still_separates_benign_from_degenerate(tmp_path):
    """Fail-closed must not become halt-on-everything, or leg 3 passes vacuously."""
    benign = _fixture(tmp_path / "a", lambda c, s: BENIGN)
    degen = _fixture(tmp_path / "b", lambda c, s: DEGENERATE if c == 4 else BENIGN)
    gb, gd = degeneracy_guard(benign), degeneracy_guard(degen)
    assert (gb["armed"], gb["halted"], gb["degenerate_cells"]) == (True, False, [])
    assert (gd["armed"], gd["halted"], gd["degenerate_cells"]) == (True, True, [4])


# --------------------------------------------------------------- 5. every CALL SITE forwards it
def _call_sites_missing_mu_diag(source: str) -> list[int]:
    """Line numbers of ``write_cell_eval_artifact(...)`` calls that do not pass ``mu_diag``."""
    missing = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name != "write_cell_eval_artifact":
            continue
        kws = {k.arg for k in node.keywords}
        if "mu_diag" not in kws and not any(k.arg is None for k in node.keywords):
            missing.append(node.lineno)
    return missing


def test_the_call_site_checker_can_actually_fail():
    """FIXTURE DISCRIMINATION, first: a checker that never fires would reproduce the defect."""
    omits = "write_cell_eval_artifact(d, cell_id=1, seed=0, headline_series=s)\n"
    passes = "write_cell_eval_artifact(d, cell_id=1, seed=0, mu_diag=m)\n"
    assert _call_sites_missing_mu_diag(omits) == [1], "checker failed to flag a real omission"
    assert _call_sites_missing_mu_diag(passes) == [], "checker flags a compliant call"


@pytest.mark.parametrize(
    "rel",
    ["scripts/m6_toy_rehearsal.py", "scripts/m6_verdict_integration_check.py"],
)
def test_every_driver_call_site_forwards_mu_diag(rel):
    """The actual defect: the driver had the value on ``head_score`` and never passed it."""
    src = (REPO / rel).read_text()
    assert "write_cell_eval_artifact" in src, f"{rel}: guard against a vacuous pass"
    bad = _call_sites_missing_mu_diag(src)
    assert not bad, f"{rel}: write_cell_eval_artifact without mu_diag at line(s) {bad}"
