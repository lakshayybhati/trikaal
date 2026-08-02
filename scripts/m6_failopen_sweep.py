"""THE FAIL-OPEN SWEEP — every guard, gate, disclosure and conformance check on the M6 path,
against every degenerate input, with what it actually returns.

WHY THIS EXISTS. C-2 was one guard that returned a hardcoded ``armed: True`` while reading nothing.
It was fixed at that site. **Nothing swept the siblings.** S-2 then found the same shape in
``power_guard``, which is exactly the ninth standing norm — *fix the class, not the instance* —
being violated by the pass that wrote it. So the class gets a tool.

DEFINITIONS, stated before any row is read:
  * **FAILS OPEN** — with a degenerate input the check returns a value a caller reads as *fine*:
    an empty failure list, ``halted: False``, ``armed: True``, ``pass: True``, or a quiet verdict.
    This is the defect class.
  * **FAILS CLOSED** — it raises, or it reports the problem (non-empty failures / halted / pass
    False). Both are acceptable; the row records which, because "raised TypeError" and "named the
    missing field" are very different qualities of closed.

THE SWEEP IS SUSPECT UNTIL SHOWN CAPABLE OF FAILING. ``--mutate`` wraps one known-CLOSED guard so
it returns a quiet pass on degenerate input, and requires the sweep to name it. If that mutation
does not flip the row, the sweep is decoration and the script says so and exits non-zero.

COVERAGE IS REPORTED, NOT IMPLIED. The registry below is the explicit list of what was swept.
"we swept and found one" and "our sweep only looked at one" produce the same table otherwise.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from eval.test_verdict import MU_PLANTED, _build_fixture
from trikaal.eval import conformance as C
from trikaal.eval import verdict as V
from trikaal.run import matrix as M

OUT = Path("runs_manifest/m6_failopen_sweep.json")

CASES = ("absent", "empty", "none", "nan", "inf", "wrongtype")
_SENTINEL: dict[str, Any] = {
    "empty": {},
    "none": None,
    "nan": float("nan"),
    "inf": float("inf"),
    "wrongtype": "not-a-number",
}


# --------------------------------------------------------------------------- classification
def _is_open(result: Any) -> bool:
    """True iff a caller would read this return as 'nothing wrong here'."""
    if isinstance(result, list):
        return len(result) == 0  # a failure-list check that found no failures
    if isinstance(result, dict):
        if result.get("halted") is True:
            return False
        if result.get("pass") is False:
            return False
        if result.get("degenerate"):
            return False
        # a disclosure that reported it could not measure is CLOSED
        if result.get("unmeasurable") or result.get("degenerate_flag"):
            return False
        return True
    if isinstance(result, dict) or result is None:
        return True
    return True


def _call(fn: Callable, *a, **kw) -> tuple[str, Any, bool]:
    try:
        r = fn(*a, **kw)
    except Exception as e:
        return ("CLOSED (raised)", f"{type(e).__name__}: {str(e)[:120]}", False)
    open_ = _is_open(r)
    return (("OPEN" if open_ else "CLOSED (reported)"), _brief(r), open_)


def _brief(r: Any) -> Any:
    if isinstance(r, list):
        return {"n_failures": len(r), "first": (r[0][:120] if r else None)}
    if isinstance(r, dict):
        keep = ("armed", "halted", "pass", "degenerate", "underpowered_claims", "unmeasurable")
        got = {k: r[k] for k in keep if k in r}
        return got or {"keys": sorted(r)[:6]}
    return repr(r)[:120]


# --------------------------------------------------------------------------- input mutation
def _mutate_evals(evals: dict, field: str, case: str, sub: str | None = None) -> dict:
    """Return a deep-ish copy of ``evals`` with ``field`` (or ``field[sub]``) made degenerate."""
    out = {}
    for k, doc in evals.items():
        d = dict(doc)
        if case == "absent":
            if sub is None:
                d.pop(field, None)
            else:
                inner = dict(d.get(field) or {})
                inner.pop(sub, None)
                d[field] = inner
        elif sub is None:
            d[field] = _SENTINEL[case]
        else:
            inner = dict(d.get(field) or {})
            inner[sub] = _SENTINEL[case]
            d[field] = inner
        out[k] = d
    return out


# --------------------------------------------------------------------------- the registry
def _registry(evals: dict, shas: dict) -> list[dict]:
    """(name, guarded thing, how each degenerate case is built) — EXPLICITLY the coverage."""
    claimed = {"dIR(4-5)": {"delta_ir": 1.1, "cells": [4, 5]}}
    healthy_trials = V.enumerate_dsr_trials(evals)
    basis = [k for k in sorted(healthy_trials) if k[0] == V.DSR_VAR_SR_BASIS_CELL]
    var_ok = float(np.var(np.array([healthy_trials[k] for k in basis])))

    return [
        {
            "name": "verdict.power_guard",
            "guards": "refuses a claim smaller than its own inputs' seed spread",
            "run": lambda case: _call(
                V.power_guard, _mutate_evals(evals, "headline_series", case), claimed
            ),
            "degenerate_input": "headline_series",
        },
        {
            "name": "verdict.power_guard [claimed delta]",
            "guards": "same guard, degenerate CLAIM rather than degenerate data",
            "run": lambda case: _call(
                V.power_guard,
                evals,
                {"dIR(4-5)": {"delta_ir": _SENTINEL.get(case, 1.1), "cells": [4, 5]}},
            ),
            "degenerate_input": "claimed[delta_ir]",
        },
        {
            "name": "verdict.degeneracy_guard",
            "guards": "HALTs on a constant-direction book (C-2's site, already fixed)",
            "run": lambda case: _call(
                V.degeneracy_guard, _mutate_evals(evals, "mu_diag", case, "frac_negative")
            ),
            "degenerate_input": "mu_diag.frac_negative",
        },
        {
            "name": "verdict.placebo_capacity_disclosure",
            "guards": "the C-12 capacity handicap channel (M1)",
            "run": lambda case: _call(
                V.placebo_capacity_disclosure, _mutate_evals(evals, "ohlcv_recon", case)
            ),
            "degenerate_input": "ohlcv_recon",
        },
        {
            "name": "verdict.decode_agreement_disclosure",
            "guards": "the C-1 decode handicap channel",
            "run": lambda case: _call(
                V.decode_agreement_disclosure, _mutate_evals(evals, "decode_agreement", case)
            ),
            "degenerate_input": "decode_agreement",
        },
        {
            # BUILDER DEFECT, CAUGHT BY THIS SWEEP AND DISCLOSED: the first version mutated a
            # TOP-LEVEL "provenance" field. The guard reads meta.provenance, so it never saw a
            # degenerate input and all six rows read FAIL OPEN — a false finding produced by the
            # harness, not the guard. Same class as the control-arm rule: a probe must be able to
            # tell the failure it is testing for from an unrelated fault.
            "name": "verdict.provenance_failures",
            "guards": "the A7 per-unit provenance stamp (fan-out instrument identity)",
            "run": lambda case: _call(
                V.provenance_failures, _mutate_evals(evals, "meta", case, "provenance")
            ),
            "degenerate_input": "meta.provenance",
        },
        {
            "name": "verdict._mu_diag_problems",
            "guards": "the mu_diag REQUIRED-field contract",
            "run": lambda case: _call(
                V._mu_diag_problems,
                _SENTINEL.get(case, {}) if case != "absent" else None,
                "sweep",
            ),
            "degenerate_input": "mu_diag (whole object)",
        },
        {
            "name": "verdict._validate_artifact",
            "guards": "the per-artifact schema/grid/field gate",
            "run": lambda case: _call(
                V._validate_artifact,
                _mutate_evals(evals, "mu_diag", case)[(4, 0)],
                "cell4_seed0_eval.json",
            ),
            "degenerate_input": "mu_diag",
        },
        {
            "name": "verdict.enumerate_dsr_trials",
            "guards": "the clause-5 trial enumeration",
            "run": lambda case: _call(
                V.enumerate_dsr_trials, _mutate_evals(evals, "val_ir_by_kappa_by_h", case)
            ),
            "degenerate_input": "val_ir_by_kappa_by_h",
        },
        {
            "name": "conformance.verdict_dsr_failures [var_sr]",
            "guards": "clause-5 recipe conformance",
            "run": lambda case: _call(
                C.verdict_dsr_failures,
                n_trials=C.PINNED_DSR["n_trials"],
                threshold=0.95,
                trials=healthy_trials,
                var_sr=_SENTINEL.get(case, var_ok) if case != "absent" else float("nan"),
            ),
            "degenerate_input": "var_sr",
        },
        {
            "name": "conformance.verdict_dsr_failures [trials]",
            "guards": "same gate, degenerate TRIAL SET",
            "run": lambda case: _call(
                C.verdict_dsr_failures,
                n_trials=C.PINNED_DSR["n_trials"],
                threshold=0.95,
                trials=_SENTINEL.get(case, {}) if case != "absent" else {},
                var_sr=var_ok,
            ),
            "degenerate_input": "trials",
        },
        {
            # This check takes NO input, so "degenerate input" means a degenerate PIN. Modelling it
            # with a no-op call produced six meaningless OPEN rows in the first version — a count
            # inflated by a mis-modelled entry is worse than a missing entry.
            "name": "conformance.pinned_threshold_failures",
            "guards": "the pinned surface holds its own registered values",
            "run": lambda case: _call(_patched_threshold_check, case),
            "degenerate_input": "PINNED_THRESHOLDS entry",
        },
        {
            "name": "matrix.artifact_reuse_failures",
            "guards": "C-14 resume binding (artifact <-> checkpoints)",
            "run": lambda case: _call(
                M.artifact_reuse_failures,
                _SENTINEL.get(case, {}) if case != "absent" else {},
                expected={
                    "checkpoints": "abc",
                    "symbols": ["A"],
                    "window": [0, 1],
                    "h": 15,
                    "n_periods": 10,
                    "start_ms": 0,
                },
            ),
            "degenerate_input": "the stored artifact doc",
        },
        {
            "name": "matrix.shard_partition_failures",
            "guards": "A5 fan-out shard disjointness/coverage",
            "run": lambda case: _call(
                M.shard_partition_failures,
                _SENTINEL.get(case, []) if case != "absent" else [],
                1,
            ),
            "degenerate_input": "the unit list",
        },
    ]


def _patched_threshold_check(case: str) -> list[str]:
    """pinned_threshold_failures with one pin made degenerate — the meaningful test for a check
    that has no arguments of its own."""
    from unittest import mock

    key = next(iter(C.PINNED_THRESHOLDS))
    if case == "absent":
        patched = {k: v for k, v in C.PINNED_THRESHOLDS.items() if k != key}
        with mock.patch.object(C, "PINNED_THRESHOLDS", patched):
            return C.pinned_threshold_failures()
    val = _SENTINEL[case]
    with mock.patch.dict(C.PINNED_THRESHOLDS, {key: val}):
        return C.pinned_threshold_failures()


# --------------------------------------------------------------------------- driver
def sweep(mutate: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as td:
        art = _build_fixture(Path(td), MU_PLANTED, fixture_seed=101)
        evals, shas = V.load_cell_evals(art)

    entries = _registry(evals, shas)
    if mutate:
        # Wrap a guard that is CLOSED on every case so it returns a quiet pass. The sweep must
        # name it. Chosen deliberately: degeneracy_guard is the C-2 site, i.e. the one we most
        # believe is fixed — a sweep that cannot catch a regression THERE is worthless.
        target = next(e for e in entries if e["name"] == "verdict.degeneracy_guard")
        target["run"] = lambda case: _call(
            lambda *_a, **_k: {"armed": True, "halted": False, "degenerate": []}
        )
        target["name"] += " [MUTATED]"

    rows, open_rows = [], []
    for e in entries:
        for case in CASES:
            verdict, value, is_open = e["run"](case)
            row = {
                "function": e["name"],
                "guards": e["guards"],
                "degenerate_input": e["degenerate_input"],
                "case": case,
                "returned": value,
                "verdict": verdict,
            }
            rows.append(row)
            if is_open:
                open_rows.append(row)
    return {
        "coverage": [e["name"] for e in entries],
        "n_functions": len(entries),
        "cases": list(CASES),
        "n_rows": len(rows),
        "rows": rows,
        "fail_open_rows": open_rows,
        "n_fail_open": len(open_rows),
    }


def main() -> int:
    rep: dict = {
        "what": "THE FAIL-OPEN SWEEP — every M6-path guard against every degenerate input",
        "definitions": {
            "OPEN": "returned a value a caller reads as fine (empty failures / halted False / "
            "armed True / pass True) despite a degenerate input — the defect class",
            "CLOSED (reported)": "named the problem in its return value",
            "CLOSED (raised)": "raised — acceptable, but a worse quality of closed than reporting",
        },
    }
    live = sweep(mutate=False)
    mutated = sweep(mutate=True)

    mutated_names = {r["function"] for r in mutated["fail_open_rows"]}
    caught = any("MUTATED" in n for n in mutated_names)
    rep["mutation_control"] = {
        "claim": "a known-CLOSED guard made to return a quiet pass must appear as FAIL OPEN",
        "target": "verdict.degeneracy_guard (the C-2 site — the one we most believe is fixed)",
        "mutant_named_by_the_sweep": caught,
        "mutant_open_rows": sum(1 for r in mutated["fail_open_rows"] if "MUTATED" in r["function"]),
    }
    if not caught:
        rep["verdict"] = "SWEEP INVALID — the mutation did not fire; this table proves nothing"
        OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
        print(json.dumps(rep["mutation_control"], indent=2))
        return 1

    rep.update(live)
    rep["verdict"] = f"{live['n_fail_open']} FAIL-OPEN rows across {live['n_functions']} functions"
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")

    print(f"mutation control: {'FIRED' if caught else 'DID NOT FIRE'}")
    print(f"swept {live['n_functions']} functions x {len(CASES)} cases = {live['n_rows']} rows\n")
    w = max(len(r["function"]) for r in live["rows"]) + 1
    for r in live["rows"]:
        mark = "  <<< FAIL OPEN" if r["verdict"] == "OPEN" else ""
        print(f"  {r['function']:<{w}} {r['case']:<10} {r['verdict']:<18} {r['returned']}{mark}")
    print(f"\nFAIL OPEN: {live['n_fail_open']} / {live['n_rows']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() if "--mutate-only" not in sys.argv else 0)
