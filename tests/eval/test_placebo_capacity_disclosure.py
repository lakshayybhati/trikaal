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

# §7 v1.6.11 C-1: the decode-agreement disclosure is REQUIRED on every artifact.
DECODE_AGREEMENT_OK = {
    "sign_agreement_dim0": 0.95,
    "variance_ratio_dim0": 1.0,
    "single_bar_degenerate": False,
}


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
                decode_agreement=DECODE_AGREEMENT_OK,
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
            decode_agreement=DECODE_AGREEMENT_OK,
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
    man = assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)
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
    ma = assemble_verdict(a_ev, a_sh, tabled_mde_h15=3.518, money_verdict=False)
    mb = assemble_verdict(b_ev, b_sh, tabled_mde_h15=3.518, money_verdict=False)

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
    # SLICE TO THE NEXT top-level def, not to a NAMED one. The first version ended the slice at
    # "def provenance_failures" and silently scanned a NEIGHBOURING function once
    # decode_agreement_disclosure was inserted between them — a fixture whose boundaries move
    # when unrelated code moves is not measuring what it claims to.
    import ast

    path = Path(__file__).resolve().parents[2] / "src/trikaal/eval/verdict.py"
    src = path.read_text()
    tree = ast.parse(src)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "placebo_capacity_disclosure"
    )
    lines = src.split("\n")
    body = "\n".join(lines[fn.lineno - 1 : fn.end_lineno])
    assert "placebo_capacity_disclosure" in body and "decode_agreement_disclosure" not in body
    for token in ("pass", ">=", "<=", "ECON_FLOOR", "THRESHOLD"):
        assert token not in body.replace("passes", ""), (
            f"placebo_capacity_disclosure contains {token!r} — it must report a magnitude, "
            "never adjudicate one"
        )


# ============================================================ §7 v1.6.11 C-1: the SECOND channel
def _da(agree: float, degenerate: bool = False) -> dict:
    return {
        "sign_agreement_dim0": agree,
        "variance_ratio_dim0": 0.0 if degenerate else 1.0,
        "single_bar_degenerate": degenerate,
    }


def _fixture2(tmp: Path, da_for) -> Path:
    """Same shape as _fixture, varying decode_agreement instead of ohlcv_recon."""
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
                ohlcv_recon=OK,
                decode_agreement=da_for(c, s),
                meta={},
            )
            entries[name] = sha
    write_eval_index(tmp, entries)
    return tmp


def test_the_withdrawn_statistic_and_the_correct_one_disagree_at_n3():
    """THE REGRESSION TEST FOR THE ACTUAL DEFECT, on the recorded values.

    The withdrawn form divided |diff| by ONE ARM'S sd and read >1 as "asymmetric". The correct
    two-sample form divides by the SE of the DIFFERENCE, which combines both arms and is sqrt(n)
    smaller. On the recorded C-1 numbers (micro sd 0.0303, micro_shuffled sd 0.0973, |diff|
    0.1071, n=3) these give OPPOSITE answers, which is why "ASYMMETRIC" was withdrawn.

    Note the disagreement is n-DEPENDENT: at n=5 the same spread WOULD be significant. That is
    the point — a ratio with no n in it cannot answer an inference question."""
    import math

    from trikaal.eval.tdist import student_t_ppf, welch_satterthwaite_df

    sa, sb, diff, n = 0.0303, 0.0973, 0.1071, 3
    wrong = diff / max(sa, sb)
    se = math.sqrt(sa**2 / n + sb**2 / n)
    t = diff / se
    df = welch_satterthwaite_df(sa / math.sqrt(n), n - 1, sb / math.sqrt(n), n - 1)
    crit = student_t_ppf(0.95, df)

    assert wrong > 1.0, f"the withdrawn form must read 'asymmetric' here, got {wrong:.3f}"
    assert t < crit, f"the correct form must read indeterminate, got t={t:.3f} crit={crit:.3f}"
    assert abs(t - 1.821) < 0.01 and abs(crit - 2.6226) < 0.01, (t, crit)


def test_the_disclosure_reports_the_se_of_the_difference(tmp_path):
    """Whatever the outcome, the SE of the DIFFERENCE must be what is reported."""
    from trikaal.eval.verdict import decode_agreement_disclosure

    vals = {4: [0.96, 0.94, 0.95, 0.96, 0.94], 5: [0.95, 0.77, 0.81, 0.95, 0.77]}
    ev, _ = load_cell_evals(
        _fixture2(
            tmp_path / "t",
            lambda c, s: _da(vals.get(c, [0.95] * 5)[DSR_SEEDS.index(s)]),
        )
    )
    t = decode_agreement_disclosure(ev)["primary_pair_test_cell4_vs_cell5"]
    assert "se_of_diff" in t and "welch_df" in t and "t_crit_0p95" in t
    assert t["se_of_diff"] > 0
    # and the reported t is |diff| / SE(diff), NOT |diff| / one arm's sd
    assert abs(t["t"] - t["abs_diff_cell4_minus_cell5"] / t["se_of_diff"]) < 1e-9


def test_a_degenerate_decode_makes_the_disclosure_refuse(tmp_path):
    """A collapsed single-bar decode scores a PERFECT agreement for free (§7 v1.6.10)."""
    from trikaal.eval.verdict import decode_agreement_disclosure

    ev, _ = load_cell_evals(
        _fixture2(tmp_path / "d", lambda c, s: _da(1.0 if c == 5 else 0.90, degenerate=(c == 5)))
    )
    t = decode_agreement_disclosure(ev)["primary_pair_test_cell4_vs_cell5"]
    assert t["any_degenerate"] is True
    assert t["significant"] is False
    assert "artifact" in t["reading"]


def test_the_decode_disclosure_can_never_move_the_verdict(tmp_path):
    """Non-gating, proven the same way as M1: identical series, only decode_agreement differs."""
    from trikaal.eval.verdict import decode_agreement_disclosure

    a_ev, a_sh = load_cell_evals(_fixture2(tmp_path / "x", lambda c, s: _da(0.98)))
    b_ev, b_sh = load_cell_evals(
        _fixture2(tmp_path / "y", lambda c, s: _da(0.40 if c == 5 else 0.98))
    )
    ma = assemble_verdict(a_ev, a_sh, tabled_mde_h15=3.518, money_verdict=False)
    mb = assemble_verdict(b_ev, b_sh, tabled_mde_h15=3.518, money_verdict=False)
    assert ma["verdict"]["emitted"] == mb["verdict"]["emitted"]
    assert ma["verdict"]["failing_clauses"] == mb["verdict"]["failing_clauses"]
    for name in ma["clauses"]:
        assert ma["clauses"][name]["pass"] == mb["clauses"][name]["pass"], name
    # FIXTURE DISCRIMINATION: the disclosures must actually differ, or this proves nothing
    assert (
        decode_agreement_disclosure(a_ev)["per_cell"]["5"]["sign_agreement_mean"]
        != decode_agreement_disclosure(b_ev)["per_cell"]["5"]["sign_agreement_mean"]
    )


def test_both_handicap_channels_are_present_in_the_manifest(tmp_path):
    """They compound: both inflate dIR(4-5) in the same direction if real."""
    ev, sh = load_cell_evals(_fixture2(tmp_path / "m", lambda c, s: _da(0.95)))
    man = assemble_verdict(ev, sh, tabled_mde_h15=3.518, money_verdict=False)
    assert man["placebo_capacity_disclosure"]["non_gating"] is True
    assert man["decode_agreement_disclosure"]["non_gating"] is True
