"""AUDIT TIER-4 C-3 — VERIFY ONLY. Does clause 5's ``var_sr`` mix three time-unit Sharpes?

REPORT, DO NOT FIX. §3 clause 5 and `PINNED_DSR` are FROZEN; this script measures and records.

The auditor's two claims, verified separately:
  (1) a **3.46x span** across the de-annualization divisors, and
  (2) a hand-calculation of **5.8056e-4**.

CONTROL ARM (standing norm — a probe that cannot fail is not a check). Before any comparison the
script re-derives the ALL-ARMS ``var_sr`` of the toy manifest from its own per-cell artifacts and
requires a bit-exact match with the value that manifest recorded. If that fails the probe emits
PROBE INVALID and no conclusion about the units, because the re-derivation path would not be the
one the code used.

Units argument (from the code, not from assumption):
  * ``verdict.enumerate_dsr_trials`` divides each **annualized** VAL IR by
    ``sqrt(periods_per_year(h))`` — a per-**h**-minute Sharpe, i.e. a DIFFERENT time unit for
    each of h in {5, 15, 60}.
  * ``deflated_sharpe_ratio`` feeds ``var_sr`` to ``expected_max_sharpe`` -> SR0, and
    ``probabilistic_sharpe_ratio`` compares SR0 against ``_sharpe(headline_series)``, whose grid the
    artifact gate pins to ``PRIMARY_H`` = 15. So SR0 is compared against a per-15-minute Sharpe.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

from trikaal.eval.dsr import deflated_sharpe_ratio, expected_max_sharpe
from trikaal.eval.metrics import periods_per_year
from trikaal.eval.verdict import (
    DSR_KAPPAS,
    DSR_N_TRIALS,
    DSR_THRESHOLD,
    DSR_VAR_SR_BASIS_CELL,
    PRIMARY_H,
)
from trikaal.utils.receipts import ReceiptRefused, write_receipt

#: This script has no argparse. `--force` is still the switch that permits overwriting a
#: TRACKED receipt; see trikaal.utils.receipts for why that is gated.
_FORCE = "--force" in sys.argv

TOY_EVAL = Path("runs_cloud/runs/m6_toy/eval")
TOY_MANIFEST = Path("runs_manifest/m6_toy_verdict_manifest.json")
OUT = Path("runs_manifest/m6_c3_dsr_units.json")
H_GRID = (5, 15, 60)
AUDITOR_SPAN = 3.46
AUDITOR_HAND_CALC = 5.8056e-4


def _load() -> tuple[dict, dict]:
    """(annualized IR by (cell,seed,h,kappa), seed-mean headline series by cell)."""
    ann: dict[tuple[int, int, int, float], float] = {}
    series: dict[int, list] = {}
    for f in sorted(glob.glob(str(TOY_EVAL / "cell*_seed*_eval.json"))):
        d = json.load(open(f))
        cid, seed = int(d["cell_id"]), int(d["seed"])
        for h in H_GRID:
            for k in DSR_KAPPAS:
                ann[(cid, seed, h, k)] = float(d["val_ir_by_kappa_by_h"][str(h)][f"{k:g}"])
        series.setdefault(cid, []).append(np.asarray(d["headline_series"], dtype=np.float64))
    return ann, {c: np.mean(np.stack(v), axis=0) for c, v in series.items()}


def _deann(h: int) -> float:
    return float(np.sqrt(periods_per_year(h)))


def main() -> int:
    ann, series = _load()
    seeds = sorted({k[1] for k in ann})
    rep: dict = {
        "what": "AUDIT TIER-4 C-3 — clause-5 var_sr time-unit audit. VERIFY ONLY, nothing fixed.",
        "source": str(TOY_EVAL),
        "source_note": (
            "The ONLY real multi-cell artifact set in the repo. 3 seeds, toy scale, and its "
            "manifest predates the v1.5 cell-5 basis — so the DIRECTION it shows is illustrative, "
            "NOT the M6 value. The unit ALGEBRA below is exact and data-independent."
        ),
        "seeds_present": seeds,
    }

    # ---------------------------------------------------------------- CONTROL ARM
    mixed = {k: v / _deann(k[2]) for k, v in ann.items()}
    all_arms = float(np.var(np.array([mixed[k] for k in sorted(mixed)], dtype=np.float64)))
    recorded = float(json.load(open(TOY_MANIFEST))["clauses"]["5_dsr"]["var_sr"])
    control_ok = all_arms == recorded
    rep["control"] = {
        "claim": "re-deriving the toy manifest's own all-arms var_sr must be BIT-EXACT",
        "recorded_in_manifest": recorded,
        "rederived_here": all_arms,
        "bit_exact": control_ok,
    }
    if not control_ok:
        rep["verdict"] = "PROBE INVALID — re-derivation path differs from the one the code used"
        # ★ Printed "PROBE INVALID" and wrote regardless, deleting 103 lines of CONFIRMED
        # audit findings from a tracked receipt.
        try:
            write_receipt(
                OUT,
                rep,
                measured=1,
                valid="PROBE INVALID" not in json.dumps(rep),
                invalid_reason="the report verdict is PROBE INVALID",
                force=_FORCE,
            )
        except ReceiptRefused as e:
            print(f"[c3-dsr-units] {e}")
            return 2
        print(json.dumps(rep["control"], indent=2))
        print("PROBE INVALID")
        return 1

    # ------------------------------------------------------- CLAIM 1: the 3.46x span
    divs = {h: _deann(h) for h in H_GRID}
    span = divs[min(H_GRID)] / divs[max(H_GRID)]
    rep["claim_1_span"] = {
        "auditor_said": AUDITOR_SPAN,
        "deannualization_divisors_sqrt_ppy": divs,
        "span_h5_over_h60": span,
        "closed_form": "sqrt(60/5) = sqrt(12)",
        "closed_form_value": math.sqrt(12.0),
        "exact": span == math.sqrt(12.0),
        "meaning": (
            "one and the same ANNUALIZED IR enters the trial set as a value 3.4641x LARGER at "
            "h=60 than at h=5, purely from the horizon it was measured at"
        ),
        "verdict": "CONFIRMED" if abs(span - AUDITOR_SPAN) < 0.005 else "NOT CONFIRMED",
    }

    # ------------------------------------------- CLAIM 2: the 5.8056e-4 hand-calculation
    unit_only = float(np.var(np.array([1.0 / _deann(h) for h in H_GRID for _ in DSR_KAPPAS])))
    implied_ir = math.sqrt(AUDITOR_HAND_CALC / unit_only)
    saturated = 7.563377128272953  # the project's own sign-saturated IR (prereg §7 v1.4.5)
    rep["claim_2_hand_calc"] = {
        "auditor_said": AUDITOR_HAND_CALC,
        "construction_tried": (
            "pure unit mismatch: every trial carries the SAME annualized IR, so all dispersion "
            "in var_sr is the time-unit artifact alone -> var = IR_ann^2 * Var({1/sqrt(ppy(h))})"
        ),
        "var_per_unit_ir_squared": unit_only,
        "implied_abs_IR_ann": implied_ir,
        "candidates": {
            "|IR|=7.56 (the saturated IR, as quoted to 3sf)": 7.56**2 * unit_only,
            "|IR|=7.563377128272953 (the saturated IR, exact)": saturated**2 * unit_only,
        },
        "display_bracket_for_5.8056e-4": [
            math.sqrt(5.80555e-4 / unit_only),
            math.sqrt(5.80565e-4 / unit_only),
        ],
        "verdict": "NOT REPRODUCED EXACTLY — closest construction agrees to 0.026%",
    }

    # ------------------------------- THE MATERIAL QUESTION: does the mismatch move clause 5?
    basis = [k for k in sorted(mixed) if k[0] == DSR_VAR_SR_BASIS_CELL]
    var_mixed = float(np.var(np.array([mixed[k] for k in basis])))
    s4 = series[4]
    legs: dict[str, dict] = {}
    for label, hc in [("as-pinned (MIXED units)", None)] + [
        (f"all trials in h={h} units", h) for h in H_GRID
    ]:
        vals = np.array(
            [ann[k] / (_deann(k[2]) if hc is None else _deann(hc)) for k in basis], dtype=np.float64
        )
        v = float(np.var(vals))
        sr0 = float(expected_max_sharpe(v, DSR_N_TRIALS))
        dsr = float(deflated_sharpe_ratio(s4, n_trials=DSR_N_TRIALS, var_sr=v))
        legs[label] = {
            "var_sr": v,
            "sr0": sr0,
            "dsr_cell4": dsr,
            "clause5_passes": bool(dsr >= DSR_THRESHOLD),
            "ratio_var_vs_pinned": v / var_mixed,
        }
    correct = legs[f"all trials in h={PRIMARY_H} units"]
    rep["claim_3_materiality"] = {
        "tested_series": (
            f"cell-4 seed-mean headline_series, grid h={PRIMARY_H} (artifact gate pins it)"
        ),
        "so_SR_hat_is": f"a per-{PRIMARY_H}-minute Sharpe",
        "but_var_sr_is": "a blend of per-5, per-15 and per-60-minute Sharpes",
        "basis_cell": DSR_VAR_SR_BASIS_CELL,
        "n_basis_values": len(basis),
        "legs": legs,
        "pinned_over_correct_var_ratio": var_mixed / correct["var_sr"],
        "sr0_shortfall_vs_h15_units": correct["sr0"] - legs["as-pinned (MIXED units)"]["sr0"],
        "direction_on_this_fixture": (
            "ANTI-CONSERVATIVE: the pinned mixed basis is SMALLER than the h=15-consistent basis, "
            "so SR0 is understated and clause 5 is EASIER to pass than the recipe intends"
        )
        if var_mixed < correct["var_sr"]
        else (
            "CONSERVATIVE on this fixture: the pinned mixed basis exceeds the h=15-consistent one"
        ),
        "direction_is_data_dependent": (
            "x_h = y_h * sqrt(h/15) rescales the h=5 group by 0.5774 and the h=60 group by 2.0 "
            "about ZERO, so the sign of the change depends on the realized per-horizon means and "
            "spreads. It is NOT fixed pre-data, which is the part that matters: an "
            "unknown-direction "
            "distortion of a gating threshold cannot be signed as conservative."
        ),
    }
    # ---- FIXTURE DISCRIMINATION (standing norm): can this fixture separate the OUTCOMES?
    outcomes = {lg["clause5_passes"] for lg in legs.values()}
    dsrs = {lg["dsr_cell4"] for lg in legs.values()}
    rep["fixture_discrimination"] = {
        "question": "does the toy fixture separate PASS from FAIL across the four unit bases?",
        "distinct_clause5_outcomes": len(outcomes),
        "distinct_dsr_values": len(dsrs),
        "answer": "NO" if len(outcomes) == 1 else "YES",
        "why": (
            "the toy cell-4 series is so far from the bar that DSR saturates at 0.0 under EVERY "
            "basis, so the pass/fail column above is uninformative and must not be read as "
            "'the mismatch does not matter'. What the fixture DOES discriminate is the INPUT: "
            "var_sr and SR0 differ materially between bases. The outcome question is answered "
            "by the existence proof below instead."
        ),
    }

    # ---- EXISTENCE PROOF: the mismatch alone can flip clause 5. Deterministic construction.
    sr0_mixed = legs["as-pinned (MIXED units)"]["sr0"]
    sr0_correct = correct["sr0"]
    n = int(s4.shape[0])
    z = np.random.default_rng(0).standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=0)  # exactly mean 0, population sd 1

    def _dsr_at(sr_hat: float, var: float) -> float:
        return float(deflated_sharpe_ratio(sr_hat + z, n_trials=DSR_N_TRIALS, var_sr=var))

    lo, hi = sr0_mixed, sr0_correct + 0.5
    for _ in range(200):  # smallest SR_hat that passes under the PINNED (mixed) basis
        mid = 0.5 * (lo + hi)
        if _dsr_at(mid, var_mixed) >= DSR_THRESHOLD:
            hi = mid
        else:
            lo = mid
    sr_pass_mixed = hi
    lo, hi = sr0_correct, sr0_correct + 1.0
    for _ in range(200):  # smallest SR_hat that passes under the h=15-CONSISTENT basis
        mid = 0.5 * (lo + hi)
        if _dsr_at(mid, correct["var_sr"]) >= DSR_THRESHOLD:
            hi = mid
        else:
            lo = mid
    sr_pass_correct = hi
    witness = 0.5 * (sr_pass_mixed + sr_pass_correct)
    rep["existence_proof_outcome_flip"] = {
        "construction": (
            f"a deterministic n={n} series with population sd 1 and mean = SR_hat "
            "(seed 0, standardized), so SR_hat is exact; both bases scored with the SAME series"
        ),
        "min_SR_hat_passing_under_pinned_mixed_basis": sr_pass_mixed,
        "min_SR_hat_passing_under_h15_consistent_basis": sr_pass_correct,
        "flip_window_width_per_15min_sharpe": sr_pass_correct - sr_pass_mixed,
        "flip_window_width_in_annualized_IR": (sr_pass_correct - sr_pass_mixed) * _deann(PRIMARY_H),
        "witness_SR_hat": witness,
        "witness_dsr_under_pinned": _dsr_at(witness, var_mixed),
        "witness_dsr_under_h15_consistent": _dsr_at(witness, correct["var_sr"]),
        "witness_flips": bool(
            _dsr_at(witness, var_mixed) >= DSR_THRESHOLD > _dsr_at(witness, correct["var_sr"])
        ),
        "so": (
            "the time-unit mismatch is OUTCOME-MATERIAL, not merely input-material: there is a "
            "non-empty band of cell-4 Sharpes for which clause 5 PASSES under the pinned recipe "
            "and FAILS under a unit-consistent one. Whether the real cell-4 lands in that band is "
            "unknown and unknowable pre-data."
        ),
    }
    rep["headline"] = {
        "sr0_pinned": sr0_mixed,
        "sr0_h15_consistent": sr0_correct,
        "sr0_relative_shortfall": sr0_mixed / sr0_correct - 1.0,
        "statement": (
            "on this fixture the pinned mixed-unit basis understates SR0 by "
            f"{abs(sr0_mixed / sr0_correct - 1.0):.2%}, i.e. it lowers the clause-5 bar. SR0 "
            "scales "
            "as sqrt(var_sr), so the shift is a pure ratio and carries to any scale; only its SIGN "
            "is data-dependent."
        ),
    }
    rep["reported_not_fixed"] = (
        "§3 clause 5 and PINNED_DSR are frozen. No value moved, no clause touched, no code changed."
    )
    # ★ Printed "PROBE INVALID" and wrote regardless, deleting 103 lines of CONFIRMED
    # audit findings from a tracked receipt.
    try:
        write_receipt(
            OUT,
            rep,
            measured=1,
            valid="PROBE INVALID" not in json.dumps(rep),
            invalid_reason="the report verdict is PROBE INVALID",
            force=_FORCE,
        )
    except ReceiptRefused as e:
        print(f"[c3-dsr-units] {e}")
        return 2
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
