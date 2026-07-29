"""KATs for the §7 v1.4.4 HALT-only degeneracy guard (verdict.py).

Proves: (1) HALT on sign-lock (frac_negative outside the band); (2) HALT on filter-never-binds
(decision-activity at κ* == 0 or 1); (3) NO HALT just inside both boundaries; (4) the guard is
HALT-ONLY — it can NEVER alter a clause result or flip SURVIVES↔NULL, only supersede the emitted
verdict with HALT_ADJUDICATE. The fixtures are written through the SAME emission path
(``write_cell_eval_artifact``) the real driver uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from trikaal.eval.verdict import (  # noqa: E402
    DSR_HORIZONS,
    DSR_KAPPAS,
    DSR_SEEDS,
    HALT_ADJUDICATE,
    NULL,
    PRIMARY_H,
    SURVIVES,
    assemble_verdict,
    load_cell_evals,
    write_cell_eval_artifact,
    write_eval_index,
)

T = 1500
SIGMA_COMMON, SIGMA_IDIO = 0.02, 0.01
MU_PLANTED = {1: 0.0005, 2: 0.0005, 3: 0.0010, 4: 0.0020, 5: 0.0000}  # → SURVIVES (test_verdict)
MU_NULL = {1: 0.0002, 2: 0.0002, 3: 0.0000, 4: -0.0002, 5: 0.0002}  # → NULL
BENIGN = {
    "mean": 0.0,
    "std": 1.0,
    "frac_negative": 0.5,
    "n": T,
    "estimator": "expectation",
    "activity_decisions": 0.5,
}


def _md(frac_negative=0.5, activity=0.5):
    return {**BENIGN, "frac_negative": frac_negative, "activity_decisions": activity}


def _fixture(out_dir: Path, mu_by_cell, mudiag_by_cell, *, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(T) * SIGMA_COMMON
    entries = {}
    for cell in (1, 2, 3, 4, 5):
        for sd in DSR_SEEDS:
            series = common + rng.standard_normal(T) * SIGMA_IDIO + mu_by_cell[cell]
            val = {h: {k: float(rng.normal(0.5, 0.3)) for k in DSR_KAPPAS} for h in DSR_HORIZONS}
            name, sha = write_cell_eval_artifact(
                out_dir,
                cell_id=cell,
                seed=sd,
                grid={"h": PRIMARY_H, "start_ms": 1_704_132_000_000, "n_periods": T},
                headline_series=series,
                kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
                val_ir_by_kappa_by_h=val,
                mu_diag=mudiag_by_cell[cell],
                meta={"fixture": True},
            )
            entries[name] = sha
    write_eval_index(out_dir, entries)
    return out_dir


def _assemble(d):
    evals, shas = load_cell_evals(d)
    return assemble_verdict(evals, shas, tabled_mde_h15=3.518)


def test_halt_on_sign_lock(tmp_path):
    """A cell with frac_negative outside [0.05, 0.95] → HALT (primary preserved un-altered)."""
    md = {c: _md() for c in (1, 2, 3, 4, 5)}
    md[5] = _md(frac_negative=0.99, activity=0.5)  # sign-locked short
    m = _assemble(_fixture(tmp_path / "sl", MU_PLANTED, md, seed=101))
    assert m["verdict"]["primary"] == SURVIVES  # the CLAUSES still say SURVIVES
    assert m["verdict"]["emitted"] == HALT_ADJUDICATE  # but the guard supersedes with HALT
    assert m["verdict"]["halted_for_degeneracy"] is True
    assert m["degeneracy_guard"]["degenerate_cells"] == [5]
    assert "sign-locked" in m["degeneracy_guard"]["per_cell"]["5"]["reasons"][0]


def test_halt_on_filter_never_binds(tmp_path):
    """A cell trading every decision (activity_decisions==1.0) with a BALANCED sign → HALT via the
    binding-filter leg alone (sign is in-band, so this isolates the second leg)."""
    md = {c: _md() for c in (1, 2, 3, 4, 5)}
    md[2] = _md(frac_negative=0.5, activity=1.0)  # balanced sign, but the θ=κ·c filter never binds
    m = _assemble(_fixture(tmp_path / "fb", MU_PLANTED, md, seed=101))
    assert m["verdict"]["emitted"] == HALT_ADJUDICATE
    assert m["degeneracy_guard"]["degenerate_cells"] == [2]
    assert "filter never binds" in m["degeneracy_guard"]["per_cell"]["2"]["reasons"][0]


def test_no_halt_just_inside_both_boundaries(tmp_path):
    """frac_negative 0.94 (inside 0.95) AND activity 0.90 (inside (0,1)) → NO HALT; the emitted
    verdict is the clause-derived SURVIVES, guard armed but not firing."""
    md = {c: _md(frac_negative=0.94, activity=0.90) for c in (1, 2, 3, 4, 5)}
    m = _assemble(_fixture(tmp_path / "in", MU_PLANTED, md, seed=101))
    assert m["degeneracy_guard"]["halted"] is False
    assert m["degeneracy_guard"]["degenerate_cells"] == []
    assert m["verdict"]["emitted"] == SURVIVES == m["verdict"]["primary"]


def test_guard_cannot_alter_a_clause_outcome(tmp_path):
    """HALT-ONLY proof: the SAME series (identical clauses) with benign vs degenerate mu_diag give
    BYTE-IDENTICAL clauses and the SAME primary; only `emitted` differs (SURVIVES vs HALT). The
    guard can never flip SURVIVES↔NULL, only refuse to emit."""
    benign = {c: _md() for c in (1, 2, 3, 4, 5)}
    degen = {c: _md() for c in (1, 2, 3, 4, 5)}
    degen[4] = _md(frac_negative=0.999, activity=1.0)  # the signal cell degenerate
    m_ok = _assemble(_fixture(tmp_path / "ok", MU_PLANTED, benign, seed=101))
    m_bad = _assemble(_fixture(tmp_path / "bad", MU_PLANTED, degen, seed=101))
    # clause results are IDENTICAL — the guard never touched them
    assert m_ok["clauses"] == m_bad["clauses"]
    assert m_ok["verdict"]["primary"] == m_bad["verdict"]["primary"] == SURVIVES
    # only the emitted verdict changed, and only toward HALT (never to NULL)
    assert m_ok["verdict"]["emitted"] == SURVIVES
    assert m_bad["verdict"]["emitted"] == HALT_ADJUDICATE
    assert m_bad["verdict"]["emitted"] != NULL


def test_guard_holds_on_a_null_fixture_too(tmp_path):
    """Symmetry: a NULL fixture with a degenerate cell also HALTs — the guard never converts NULL
    into SURVIVES either. primary stays NULL, emitted becomes HALT."""
    degen = {c: _md() for c in (1, 2, 3, 4, 5)}
    degen[2] = _md(frac_negative=0.0, activity=1.0)
    m = _assemble(_fixture(tmp_path / "n", MU_NULL, degen, seed=202))
    assert m["verdict"]["primary"] == NULL
    assert m["verdict"]["emitted"] == HALT_ADJUDICATE
    assert m["verdict"]["emitted"] != SURVIVES
