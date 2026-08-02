"""KATs for the M6 verdict assembler — the ONLY SURVIVES/NULL path (pre-rehearsal close-out #2).

Three decision fixtures (the third audit's demand), one per §3 failure geometry:

* **planted effect** → SURVIVES with every clause green;
* **null** → NULL with the failing clauses named + §5 fallback DESCRIPTIVE_ONLY (double-NULL);
* **placebo-harmed** (Cell 5 crushed, Cell 4 ≈ Cell 2) → NULL **via clause 3 alone** — the
  false-positive channel clause 3 exists to block: ΔIR(4−5) is huge and clauses 1/2/4/5 all
  pass, yet no micro-information claim survives, and the (5−2) health diagnostic discloses
  "shuffle harmed";

plus a §5 fallback-CLAIMED fixture, content-hash/loader integrity KATs, determinism, and the
driver script end-to-end (conformance gate + grid-pin refusal included).

Every fixture is written through ``write_cell_eval_artifact`` — the same emission path the real
eval driver uses — so writer/loader drift is structurally impossible. All fixtures are
deterministic (fixed RNG seeds) and the bootstrap recipe is pinned, so every asserted outcome
is bit-reproducible.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from trikaal.eval.verdict import (
    DSR_HORIZONS,
    DSR_KAPPAS,
    DSR_N_TRIALS,
    DSR_SEEDS,
    FALLBACK_CLAIMED,
    FALLBACK_DESCRIPTIVE,
    NULL,
    PRIMARY_H,
    SURVIVES,
    VerdictInputError,
    assemble_verdict,
    enumerate_dsr_trials,
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
T = 1500  # toy grid length — the script's grid pin is exercised separately
SIGMA_COMMON = 0.02  # the shared market component (cancels in every paired Δ)
SIGMA_IDIO = 0.01  # per-(cell, seed) idiosyncratic vol

# per-cell per-period mean returns (μ). Annualization at h=15 is ≈ ×187, series std ≈ 0.021,
# and the paired SE_boot at T=1500 lands ≈ 1.9 annualized-IR units (MDE_paired ≈ 4.8) — each
# fixture's planted gaps are sized ≥ 3 SE away from every asserted threshold.
MU_PLANTED = {1: 0.0005, 2: 0.0005, 3: 0.0010, 4: 0.0020, 5: 0.0000}
MU_NULL = {1: 0.0002, 2: 0.0002, 3: 0.0000, 4: -0.0002, 5: 0.0002}
MU_PLACEBO_HARMED = {1: 0.0017, 2: 0.0020, 3: 0.0010, 4: 0.0020, 5: -0.0040}
MU_FALLBACK_CLAIMED = {1: 0.0000, 2: 0.0030, 3: 0.0010, 4: 0.0030, 5: 0.0030}

# §7 v1.6 C-2: a mu_diag comfortably inside FRAC_NEG_BAND with a binding filter, so these clause
# fixtures test the CLAUSES and never trip the degeneracy HALT for an unrelated reason.
BENIGN_MU_DIAG = {"frac_negative": 0.5, "activity_decisions": 0.5}


def _build_fixture(out_dir: Path, mu_by_cell: dict[int, float], *, fixture_seed: int) -> Path:
    """15 deterministic eval artifacts + index: series = common + idio + μ(cell)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(fixture_seed)
    common = rng.standard_normal(T) * SIGMA_COMMON
    entries: dict[str, str] = {}
    for cell in (1, 2, 3, 4, 5):
        for seed in DSR_SEEDS:
            series = common + rng.standard_normal(T) * SIGMA_IDIO + mu_by_cell[cell]
            val = {h: {k: float(rng.normal(0.5, 0.3)) for k in DSR_KAPPAS} for h in DSR_HORIZONS}
            name, sha = write_cell_eval_artifact(
                out_dir,
                cell_id=cell,
                seed=seed,
                grid={"h": PRIMARY_H, "start_ms": 1_704_132_000_000, "n_periods": T},
                headline_series=series,
                kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
                val_ir_by_kappa_by_h=val,
                # §7 v1.6 C-2: BENIGN, and now REQUIRED. Until the requirement landed this
                # fixture omitted mu_diag entirely, so every clause KAT in this file was
                # exercising the blind-guard path — the same hole the real driver had.
                mu_diag=BENIGN_MU_DIAG,
                ohlcv_recon=OHLCV_RECON_OK,
                decode_agreement=DECODE_AGREEMENT_OK,
                meta={"fixture": True},
                codebook=CODEBOOK_OK,
            )
            entries[name] = sha
    write_eval_index(out_dir, entries)
    return out_dir


def _assemble(art_dir: Path) -> dict:
    evals, shas = load_cell_evals(art_dir)
    return assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)


@pytest.fixture(scope="module")
def planted_dir(tmp_path_factory) -> Path:
    return _build_fixture(tmp_path_factory.mktemp("planted"), MU_PLANTED, fixture_seed=101)


# ------------------------------------------------------------------ the three decision KATs
def test_planted_effect_survives_every_clause(planted_dir):
    m = _assemble(planted_dir)
    assert m["verdict"]["primary"] == SURVIVES
    assert m["verdict"]["failing_clauses"] == []
    assert all(c["pass"] for c in m["clauses"].values())
    assert m["verdict"]["fallback"] is None  # §5 is NULL-conditional — never evaluated here
    assert m["verdict"]["double_null"] is False
    # the numbers behind the word are recorded, not just the word
    c2 = m["clauses"]["2_mde_paired"]
    assert c2["delta_ir"] >= c2["mde_paired"] and c2["tabled_mde_h15"] == 3.518
    assert m["clauses"]["5_dsr"]["n_trials"] == DSR_N_TRIALS
    assert m["content_hash"].startswith("sha256:")


def test_null_fixture_is_null_with_failing_clauses_named(tmp_path):
    m = _assemble(_build_fixture(tmp_path / "null", MU_NULL, fixture_seed=202))
    v = m["verdict"]
    assert v["primary"] == NULL
    # no effect anywhere: the CI, MDE, floor, and placebo-validity clauses must all be named
    for must_fail in ("1_paired_ci", "2_mde_paired", "3_placebo_validity", "4_economic_floor"):
        assert must_fail in v["failing_clauses"]
    # §5 fallback WAS evaluated (its precondition holds) and claims nothing → double-NULL
    assert v["fallback"] is not None
    assert v["fallback"]["word"] == FALLBACK_DESCRIPTIVE
    assert v["double_null"] is True


def test_placebo_harmed_fixture_is_null_via_clause_3_alone(tmp_path):
    """THE false-positive channel: the shuffle crushed Cell 5, so ΔIR(4−5) is huge — every
    other clause passes — but micro adds nothing over OHLCV (Cell 4 ≈ Cell 2) and clause 3
    alone must block survival, with the (5−2) health diagnostic disclosing the harm."""
    m = _assemble(_build_fixture(tmp_path / "ph", MU_PLACEBO_HARMED, fixture_seed=303))
    v = m["verdict"]
    assert v["primary"] == NULL
    assert v["failing_clauses"] == ["3_placebo_validity"]  # 1, 2, 4, 5 all green
    for passing in ("1_paired_ci", "2_mde_paired", "4_economic_floor", "5_dsr"):
        assert m["clauses"][passing]["pass"] is True
    health = m["clauses"]["3_placebo_validity"]["health_diagnostic_5_minus_2"]
    assert health["shuffle_harmed"] is True and health["ci_upper"] < 0
    assert v["fallback"] is not None and v["double_null"] is True


def test_fallback_claimed_when_fsq_leg_passes_the_identical_rule(tmp_path):
    m = _assemble(_build_fixture(tmp_path / "fb", MU_FALLBACK_CLAIMED, fixture_seed=404))
    v = m["verdict"]
    assert v["primary"] == NULL  # micro adds nothing (4 ≈ 5 ≈ 2)
    fb = v["fallback"]
    assert fb["word"] == FALLBACK_CLAIMED and all(c["pass"] for c in fb["clauses"].values())
    assert v["double_null"] is False
    assert (
        fb["bootstrap_2_minus_1"]["b"] == 10_000 and fb["bootstrap_2_minus_1"]["seed"] == 20260704
    )


# ------------------------------------------------------------------ loader integrity + hashes
def test_tampered_artifact_is_refused_by_content_hash(planted_dir, tmp_path):
    import shutil

    d = tmp_path / "tampered"
    shutil.copytree(planted_dir, d)
    victim = d / "cell4_seed1_eval.json"
    doc = json.loads(victim.read_text())
    doc["headline_series"][0] += 1.0  # a result-changing edit, index NOT updated
    victim.write_text(json.dumps(doc, sort_keys=True))
    with pytest.raises(VerdictInputError, match="sha256"):
        load_cell_evals(d)


def test_missing_artifact_is_refused(planted_dir, tmp_path):
    import shutil

    d = tmp_path / "missing"
    shutil.copytree(planted_dir, d)
    (d / "cell2_seed0_eval.json").unlink()
    with pytest.raises(VerdictInputError, match="cell2_seed0"):
        load_cell_evals(d)


def test_doctored_horizon_budget_is_refused(tmp_path):
    """An artifact carrying a 4-horizon VAL table (the old '240 trials' bookkeeping error)
    must be rejected — h=1 is not evaluated anywhere in M6 (§3 clause 5)."""
    d = _build_fixture(tmp_path / "doct", MU_PLANTED, fixture_seed=505)
    victim = d / "cell1_seed0_eval.json"
    doc = json.loads(victim.read_text())
    doc["val_ir_by_kappa_by_h"]["1"] = doc["val_ir_by_kappa_by_h"]["5"]  # 4th horizon
    victim.write_text(json.dumps(doc, sort_keys=True))
    idx = json.loads((d / "index.json").read_text())
    import hashlib

    idx["artifacts"]["cell1_seed0_eval.json"] = hashlib.sha256(victim.read_bytes()).hexdigest()
    (d / "index.json").write_text(json.dumps(idx, sort_keys=True))
    with pytest.raises(VerdictInputError, match="horizons"):
        load_cell_evals(d)


def test_trial_enumeration_and_the_v1_5_three_way_split(planted_dir):
    """§7 v1.5 A.5 splits three things that used to be one number, and each is asserted here:

    (1) the ENUMERATION is still the FULL cells x SEEDS x horizons x kappas cross-product — the
        audit trail, so nothing can go missing;
    (2) ``DSR_N_TRIALS`` is the MULTIPLICITY count and EXCLUDES seeds (replicates are not
        configurations), so it is cells x horizons x kappas and is INDEPENDENT of how many seeds
        we run;
    (3) ``var_sr`` is the DISPERSION and is computed over the CELL-5 placebo subset only.
    """
    evals, _ = load_cell_evals(planted_dir)
    trials = enumerate_dsr_trials(evals)
    # (1) full enumeration, seeds INCLUDED
    assert set(trials) == {
        (c, s, h, k)
        for c in (1, 2, 3, 4, 5)
        for s in DSR_SEEDS
        for h in DSR_HORIZONS
        for k in DSR_KAPPAS
    }
    assert len(trials) == 5 * len(DSR_SEEDS) * len(DSR_HORIZONS) * len(DSR_KAPPAS)
    # (2) multiplicity EXCLUDES seeds and does not move with S
    assert DSR_N_TRIALS == 5 * len(DSR_HORIZONS) * len(DSR_KAPPAS) == 60
    assert len(trials) != DSR_N_TRIALS  # the enumeration and the count are DIFFERENT sets
    # de-annualized: every trial is a per-period Sharpe, |sr| ≪ 1
    assert max(abs(v) for v in trials.values()) < 0.1


def test_verdict_is_deterministic(planted_dir):
    m1, m2 = _assemble(planted_dir), _assemble(planted_dir)
    assert m1["content_hash"] == m2["content_hash"]
    assert m1 == m2


# ------------------------------------------------------------------ the driver script itself
def _run_script(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": "src"}
    return subprocess.run(
        [sys.executable, "scripts/m6_verdict.py", *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_script_end_to_end_on_the_planted_fixture(planted_dir, tmp_path):
    out = tmp_path / "verdict.json"
    r = _run_script("--artifacts", str(planted_dir), "--out", str(out), "--allow-toy-grid")
    assert r.returncode == 0, r.stderr
    assert "§3a conformance PASS" in r.stdout  # the gate ran FIRST
    m = json.loads(out.read_text())
    assert m["verdict"]["primary"] == SURVIVES
    assert m["grid_pinned"] is False  # toy fixture — loudly recorded, never quotable as M6
    assert m["conformance"] == "PASS" and len(m["prereg_sha256"]) == 64
    assert len(m["artifact_sha256"]) == 5 * len(DSR_SEEDS)


def test_script_refuses_a_toy_grid_without_the_flag(planted_dir, tmp_path):
    r = _run_script("--artifacts", str(planted_dir), "--out", str(tmp_path / "v.json"))
    assert r.returncode == 1
    assert "not the pinned §3a money grid" in r.stderr
    assert not (tmp_path / "v.json").exists()  # refused runs write nothing


def test_doctored_verdict_n_trials_is_caught_by_the_conformance_pin(planted_dir, monkeypatch):
    """CO2 item 3: the verdict module's own recipe literals are asserted against
    conformance.PINNED_DSR (an independent statement) — editing either side to 240 is caught
    before any DSR is computed."""
    import trikaal.eval.verdict as vd
    from trikaal.eval.conformance import ConformanceError

    monkeypatch.setattr(vd, "DSR_N_TRIALS", 240)
    evals, shas = load_cell_evals(planted_dir)
    with pytest.raises(ConformanceError, match="n_trials 240"):
        vd.assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)


# ------------------------- §7 v1.4.7: the guards must work END-TO-END through the driver script
@pytest.mark.slow
def test_verdict_integration_check_script_passes():
    """CI protection for the v1.4.7 integration item.

    Both guards live in the ONLY path that may emit SURVIVES/NULL, and the pre-flight verdict dry
    run predates both. ``scripts/m6_verdict_integration_check.py`` drives the REAL driver as a
    subprocess over healthy / degenerate / low-power toy fixtures and exits non-zero on any failed
    assertion, so asserting its exit code here keeps the whole integration surface covered."""
    env = {**os.environ, "PYTHONPATH": "src"}
    r = subprocess.run(
        [sys.executable, "scripts/m6_verdict_integration_check.py"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert r.returncode == 0, f"integration check failed:\n{r.stdout}\n{r.stderr}"
    assert "all_passed=True" in r.stdout
