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
    FRAC_NEG_BAND,
    HALT_ADJUDICATE,
    NULL,
    PRIMARY_H,
    SURVIVES,
    assemble_verdict,
    load_cell_evals,
    write_cell_eval_artifact,
    write_eval_index,
)

# §7 v1.6.11 C-1: the decode-agreement disclosure is REQUIRED on every artifact.
DECODE_AGREEMENT_OK = {
    "sign_agreement_dim0": 0.95,
    "variance_ratio_dim0": 1.0,
    "single_bar_degenerate": False,
}


# §7 v1.6 C-12 M1: the capacity disclosure is REQUIRED on every artifact.
OHLCV_RECON_OK = {"ohlcv_recon_mae": 0.10, "ohlcv_recon_mae_by_dim": [0.1] * 7, "n_ohlcv_dims": 7}

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


def _wide_spread(centre: float, half_width: float) -> dict:
    """Per-seed values spread symmetrically about ``centre`` — S-agnostic.

    The seed MEAN is preserved exactly, so a fixture built with this has the SAME seed-mean series
    as its flat counterpart and differs ONLY in seed SPREAD — which is what the power-guard
    HALT-only proof requires."""
    n = len(DSR_SEEDS)
    if n == 1:
        return {DSR_SEEDS[0]: centre}
    offs = [half_width * (2.0 * i / (n - 1) - 1.0) for i in range(n)]
    return {sd: centre + o for sd, o in zip(DSR_SEEDS, offs, strict=True)}


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
                ohlcv_recon=OHLCV_RECON_OK,
                decode_agreement=DECODE_AGREEMENT_OK,
                meta={"fixture": True},
            )
            entries[name] = sha
    write_eval_index(out_dir, entries)
    return out_dir


def _fixture_per_seed(out_dir: Path, mu_by_cell, mudiag_by_cell_seed, *, seed):
    """Like ``_fixture`` but with a PER-(cell, seed) mu_diag — needed to build a cell whose seeds
    disagree (the §7 v1.4.6 one-locked-seed pattern), which a single per-cell dict cannot express."""
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
                mu_diag=mudiag_by_cell_seed[cell][sd],
                ohlcv_recon=OHLCV_RECON_OK,
                decode_agreement=DECODE_AGREEMENT_OK,
                meta={"fixture": True},
            )
            entries[name] = sha
    write_eval_index(out_dir, entries)
    return out_dir


def _assemble(d):
    evals, shas = load_cell_evals(d)
    return assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)


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


# ------------------------------------------------- §7 v1.4.6: the ONE-LOCKED-SEED hole (supervisor)
def test_one_locked_seed_halts_where_the_seed_mean_rule_passed(tmp_path):
    """THE v1.4.6 REGRESSION KAT — the exact (0.999, 0.55, 0.55) pattern from the ruling.

    v1.4.4 tested only the seed-MEAN: (0.999 + 0.55 + 0.55)/3 = 0.6997, comfortably inside
    [0.05, 0.95], so it did NOT halt — while seed 0's constant-direction book still entered the
    seed-mean headline series the §3 clauses are computed on. This test proves BOTH halves: the
    old rule passes the pattern, and the new per-seed union halts on it.
    """
    locked, healthy = 0.999, 0.55
    md = {c: dict.fromkeys(DSR_SEEDS) for c in (1, 2, 3, 4, 5)}
    for c in (1, 2, 3, 4, 5):
        for sd in DSR_SEEDS:
            md[c][sd] = _md()  # benign everywhere else
    md[4][DSR_SEEDS[0]] = _md(
        frac_negative=locked, activity=0.5
    )  # ONE locked seed on the signal cell
    for sd in DSR_SEEDS[1:]:  # every OTHER seed healthy, whatever S is
        md[4][sd] = _md(frac_negative=healthy, activity=0.5)

    m = _assemble(_fixture_per_seed(tmp_path / "onelock", MU_PLANTED, md, seed=101))
    g, cell4 = m["degeneracy_guard"], m["degeneracy_guard"]["per_cell"]["4"]

    # (1) the OLD v1.4.4 rule — seed-mean only — would NOT have halted on this pattern
    lo, hi = FRAC_NEG_BAND
    seed_mean = cell4["frac_negative_seed_mean"]
    n = len(DSR_SEEDS)
    assert abs(seed_mean - (locked + (n - 1) * healthy) / n) < 1e-12
    old_rule_would_halt = not (lo <= seed_mean <= hi)
    assert old_rule_would_halt is False, "the pattern must be one the seed-mean rule PASSED"

    # (2) the NEW v1.4.6 per-seed union DOES halt, and names the offending seed
    assert g["halted"] is True
    assert g["degenerate_cells"] == [4]
    assert cell4["degenerate_seeds_frac_negative"] == [DSR_SEEDS[0]]
    assert any("per-seed frac_negative" in r for r in cell4["reasons"])
    assert not any(
        "seed-mean frac_negative" in r for r in cell4["reasons"]
    )  # only the per-seed leg

    # (3) the per-seed DISTRIBUTION is persisted for adjudication, not just the aggregate
    assert cell4["frac_negative_by_seed"] == {
        str(sd): (locked if sd == DSR_SEEDS[0] else healthy) for sd in DSR_SEEDS
    }
    assert cell4["activity_decisions_by_seed"] == {str(s): 0.5 for s in DSR_SEEDS}

    # (4) HALT-ONLY is unchanged: the clauses still say SURVIVES, only `emitted` is superseded
    assert m["verdict"]["primary"] == SURVIVES
    assert m["verdict"]["emitted"] == HALT_ADJUDICATE
    assert m["verdict"]["emitted"] != NULL


def test_one_seed_never_binding_also_halts(tmp_path):
    """The activity leg gets the same per-seed treatment: one seed at activity 1.0 with a balanced
    sign and an in-band seed-mean activity must still HALT."""
    md = {c: {sd: _md() for sd in DSR_SEEDS} for c in (1, 2, 3, 4, 5)}
    md[3][DSR_SEEDS[0]] = _md(frac_negative=0.5, activity=1.0)  # filter never binds on ONE seed
    m = _assemble(_fixture_per_seed(tmp_path / "onebind", MU_PLANTED, md, seed=101))
    cell3 = m["degeneracy_guard"]["per_cell"]["3"]
    assert 0.0 < cell3["activity_decisions_seed_mean"] < 1.0  # seed-mean leg cannot fire
    assert m["degeneracy_guard"]["degenerate_cells"] == [3]
    assert cell3["degenerate_seeds_activity"] == [DSR_SEEDS[0]]
    assert m["verdict"]["emitted"] == HALT_ADJUDICATE


def test_per_seed_leg_does_not_fire_on_healthy_seeds(tmp_path):
    """No false positives: three in-band seeds with differing values must NOT halt."""
    n = len(DSR_SEEDS)
    fracs = [0.45 + 0.27 * i / max(1, n - 1) for i in range(n)]  # 0.45..0.72, in band, S-agnostic
    acts = [0.60 + 0.30 * i / max(1, n - 1) for i in range(n)]  # 0.60..0.90, strictly inside (0,1)
    md = {
        c: {
            sd: _md(frac_negative=f, activity=a)
            for sd, f, a in zip(DSR_SEEDS, fracs, acts, strict=True)
        }
        for c in (1, 2, 3, 4, 5)
    }
    m = _assemble(_fixture_per_seed(tmp_path / "healthy", MU_PLANTED, md, seed=101))
    assert m["degeneracy_guard"]["halted"] is False
    assert m["degeneracy_guard"]["degenerate_cells"] == []
    assert m["verdict"]["emitted"] == SURVIVES


# ----------------------------------- §7 v1.4.7: the HALT-only POWER guard (across-seed IR spread)
def _fixture_per_seed_mu(out_dir: Path, mu_by_cell_seed, mudiag, *, seed):
    """Fixture with a PER-(cell, seed) mu offset, so a cell's seeds can be made to disagree on IR."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(T) * SIGMA_COMMON
    entries = {}
    for cell in (1, 2, 3, 4, 5):
        for sd in DSR_SEEDS:
            series = common + rng.standard_normal(T) * SIGMA_IDIO + mu_by_cell_seed[cell][sd]
            val = {h: {k: float(rng.normal(0.5, 0.3)) for k in DSR_KAPPAS} for h in DSR_HORIZONS}
            name, sha = write_cell_eval_artifact(
                out_dir,
                cell_id=cell,
                seed=sd,
                grid={"h": PRIMARY_H, "start_ms": 1_704_132_000_000, "n_periods": T},
                headline_series=series,
                kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
                val_ir_by_kappa_by_h=val,
                mu_diag=mudiag,
                ohlcv_recon=OHLCV_RECON_OK,
                decode_agreement=DECODE_AGREEMENT_OK,
                meta={"fixture": True},
            )
            entries[name] = sha
    write_eval_index(out_dir, entries)
    return out_dir


def test_power_guard_halts_when_seed_spread_swamps_the_claim(tmp_path):
    """Direction 1: cell4's seeds are driven far apart, so its within-cell across-seed IR range
    exceeds the between-cell ΔIR being claimed → HALT, primary preserved."""
    mu = {c: dict.fromkeys(DSR_SEEDS, MU_PLANTED[c]) for c in (1, 2, 3, 4, 5)}
    # a wide seed spread on the signal cell — the v1.4.5 basin-hopping scenario
    mu[4] = _wide_spread(MU_PLANTED[4], 0.020)
    m = _assemble(_fixture_per_seed_mu(tmp_path / "wide", mu, _md(), seed=303))
    p = m["power_guard"]
    assert p["halted"] is True
    assert p["underpowered_claims"], p["checks"]
    c4 = p["per_cell"]["4"]
    assert c4["ir_range_across_seeds"] > 0
    assert set(c4["ir_by_seed"]) == {str(s) for s in DSR_SEEDS}
    # the trip condition really is range >= |delta| for at least one claimed delta
    trip = next(v for v in p["checks"].values() if v["seed_spread_exceeds_claim"])
    assert trip["worst_within_cell_ir_range"] >= abs(trip["delta_ir_claimed"])
    # HALT-ONLY: the clause-derived primary is untouched, only `emitted` is superseded
    assert m["verdict"]["emitted"] == HALT_ADJUDICATE
    assert m["verdict"]["halted_for_power"] is True
    assert m["verdict"]["primary"] in (SURVIVES, NULL)


def test_power_guard_silent_when_seeds_agree(tmp_path):
    """Direction 2: identical per-seed mu → a tight across-seed IR range → the guard does NOT fire,
    and the emitted verdict is the clause-derived one."""
    mu = {c: dict.fromkeys(DSR_SEEDS, MU_PLANTED[c]) for c in (1, 2, 3, 4, 5)}
    m = _assemble(_fixture_per_seed_mu(tmp_path / "tight", mu, _md(), seed=303))
    p = m["power_guard"]
    assert p["halted"] is False
    assert p["underpowered_claims"] == []
    assert m["verdict"]["halted_for_power"] is False
    assert m["verdict"]["emitted"] == m["verdict"]["primary"]


def test_power_guard_cannot_alter_a_clause_outcome(tmp_path):
    """HALT-ONLY proof for the power guard, as a STRUCTURAL invariant.

    The v1.4.7 version of this test compared a tight-spread against a wide-spread fixture with the
    same seed-MEAN and asserted identical clauses. That premise DIED with §7 v1.5 item B: the MDE
    now carries an across-seed training-variance term, so clause 2 legitimately DOES depend on seed
    spread. Comparing two spreads can no longer isolate the guard.

    The invariant that actually matters is asserted instead, and it is stronger because it holds on
    EVERY fixture: `primary` is a pure function of the clause results (the guard is nowhere in its
    computation), and `emitted` is either `primary` or HALT_ADJUDICATE and never anything else — so
    the guard can only ever REFUSE to emit, never flip SURVIVES<->NULL.
    """
    for name, mu in (
        ("tight", {c: dict.fromkeys(DSR_SEEDS, MU_PLANTED[c]) for c in (1, 2, 3, 4, 5)}),
        ("wide", None),
    ):
        if mu is None:
            mu = {c: dict.fromkeys(DSR_SEEDS, MU_PLANTED[c]) for c in (1, 2, 3, 4, 5)}
            mu[4] = _wide_spread(MU_PLANTED[4], 0.02)
        m = _assemble(_fixture_per_seed_mu(tmp_path / name, mu, _md(), seed=404))
        v, clauses = m["verdict"], m["clauses"]
        # 1) primary is EXACTLY the clause conjunction — the guard never enters it
        expected = SURVIVES if all(c["pass"] for c in clauses.values()) else NULL
        assert v["primary"] == expected, name
        # 2) emitted is primary, or HALT — never a third thing, never the opposite word
        assert v["emitted"] in (v["primary"], HALT_ADJUDICATE), name
        if v["emitted"] == HALT_ADJUDICATE:
            assert v["halted_for_degeneracy"] or v["halted_for_power"], name
        else:
            assert not (v["halted_for_degeneracy"] or v["halted_for_power"]), name


def test_power_guard_records_the_seed_spread_it_judged_on(tmp_path):
    """The guard must persist the distribution it ruled on, not just its verdict."""
    mu = {c: dict.fromkeys(DSR_SEEDS, MU_PLANTED[c]) for c in (1, 2, 3, 4, 5)}
    mu[4] = _wide_spread(MU_PLANTED[4], 0.02)
    m = _assemble(_fixture_per_seed_mu(tmp_path / "rec", mu, _md(), seed=404))
    for c in ("1", "2", "3", "4", "5"):
        pc = m["power_guard"]["per_cell"][c]
        assert set(pc["ir_by_seed"]) == {str(s) for s in DSR_SEEDS}
        assert pc["ir_range_across_seeds"] >= 0.0
    # cell4 was deliberately spread; it must show the widest range
    ranges = {c: m["power_guard"]["per_cell"][c]["ir_range_across_seeds"] for c in "12345"}
    assert ranges["4"] == max(ranges.values())
