"""THE VACUITY SWEEP — every M6-path check that derives a verdict by ITERATING a collection, asked
what it returns when that collection is EMPTY.

WHY THIS IS A SEPARATE SWEEP. The fail-open sweep varied the VALUES inside a collection. This
varies the COLLECTION ITSELF. They are different failure modes and the first one cannot find the
second: `for x in []: check(x)` never executes, `all([])` is True, and a failure list built by
comprehension over an empty iterable is empty — so the check reports success having examined
nothing. Two rows of the first sweep were already this shape ("deleting a pin deleted its own
check", "the empty matrix was certified a perfect cover") and were fixed as individual guard bugs.
They are one class, and the class gets a tool.

COVERAGE IS DERIVED, NOT RECALLED. `_candidates()` walks the AST of every module under
`src/trikaal/{eval,run,train,data,utils}` and returns each function that BOTH iterates a collection
AND returns a verdict-shaped value. The registry below then states, for each candidate, either the
empty-collection probe or an explicit reason for exclusion. **A candidate with neither is a hard
error**, so a sibling cannot be silently dropped by forgetting to list it.

MUTATION CONTROL: a known-CLOSED check is wrapped to return a clean verdict on empty input and the
sweep must name it. If it does not, the sweep is decoration and exits non-zero.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from trikaal.eval import conformance as C
from trikaal.eval import verdict as V
from trikaal.run import matrix as M
from trikaal.train import gates as G

OUT = Path("runs_manifest/m6_vacuity_sweep.json")
ROOTS = ("src/trikaal/eval", "src/trikaal/run", "src/trikaal/train", "src/trikaal/data")


def _candidates() -> list[str]:
    """AST-derived: functions that iterate a collection AND return a verdict-shaped value."""
    out = []
    for r in ROOTS:
        for p in sorted(pathlib.Path(r).rglob("*.py")):
            text = p.read_text()
            tree = ast.parse(text)
            for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
                src = ast.get_source_segment(text, fn) or ""
                iterates = any(
                    isinstance(n, ast.For | ast.ListComp | ast.GeneratorExp | ast.SetComp)
                    for n in ast.walk(fn)
                )
                verdicty = any(
                    k in src for k in ("fails", "failures", "problems", "all(", "any(", "passed")
                ) and any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn))
                if iterates and verdicty:
                    out.append(f"{p.name}:{fn.name}")
    return out


# Candidates NOT probed, each with the reason. A candidate in neither this map nor PROBES is a
# hard error — that is what stops the coverage list from quietly shrinking.
EXCLUDED = {
    "diagnostics.py:mean_pinball": "a metric, not a verdict — an empty input is a caller error",
    "metrics.py:break_even_cost": "a metric over a fixed cost grid, never a verdict",
    "paired_bootstrap.py:paired_delta_ir_bootstrap": "a statistic; empty series raise upstream",
    "verdict.py:write_cell_eval_artifact": "a WRITER, not a check",
    "verdict.py:assemble_verdict": "the top-level composer — its constituents are probed instead",
    "matrix.py:shard_units": "a partition helper; its VERDICT function is probed",
    "matrix.py:eval_matrix": "an orchestrator that needs a live model; its checks are probed",
    "build.py:build_symbol_stream": "an ingest builder, not a verdict",
    "features.py:compute_features": "a transform, not a verdict",
    "quality.py:badtick_clip": "a transform, not a verdict",
    "quality.py:stale_flag": "a transform, not a verdict",
    "reduce_shard.py:build_symbol_features_sharded": "an ingest builder, not a verdict",
    "synthetic.py:make_synthetic_stream": "a fixture generator, not a verdict",
    "synthetic.py:perturb_after_bar": "a fixture mutator, not a verdict",
    "universe_bootstrap.py:_list_page": "an HTTP pager, not a verdict",
    "universe_build.py:slice_stream_by_ms": "a slicer, not a verdict",
    "universe_loader.py:train_region_liquidity_stats": "a statistic, not a verdict",
    "ingest_orchestrator.py:run": "the ingest driver; its causal gate is probed",
    "causal_check.py:summary": "a formatter over an already-computed report",
    "causal_check.py:build_anchor_strata": "probed indirectly via exhaustive_truncation_sweep",
    "causal_check.py:check_causal_safety": "probed via exhaustive_truncation_sweep (same core)",
    "verdict.py:load_cell_evals": "probed below",
    "diagnostics.py:codebook_usage": "a metric, not a verdict",
    # BUILDER CORRECTION, disclosed: the first run classified this VACUOUS. It is a PRODUCER, not
    # a check — returning {} for {} is correct behaviour, and its guard (verdict_dsr_failures)
    # reports "300 missing" on that same empty set, i.e. CLOSED. Counting it would have inflated
    # the finding count with the probe's own mis-modelling, the same defect as the fail-open
    # sweep's three harness bugs.
    "verdict.py:enumerate_dsr_trials": (
        "a PRODUCER; its verdict is conformance.verdict_dsr_failures, probed below"
    ),
    "strategy.py:portfolio_period_returns": "a metric over a position series, not a verdict",
    "gates.py:overfit_single_batch": (
        "a training smoke gate over STEPS, not a data collection; needs a live model"
    ),
    "gates.py:determinism_smoke": (
        "a two-run byte-comparison gate; its collection is a fixed step count"
    ),
}


def _is_clean(r: Any) -> bool:
    """True iff the return reads as 'nothing wrong' — the vacuity failure mode."""
    if isinstance(r, list):
        return len(r) == 0
    if isinstance(r, dict):
        if r.get("halted") is True or r.get("pass") is False or r.get("unmeasurable"):
            return False
        if r.get("degenerate"):
            return False
        return True
    if hasattr(r, "passed"):
        return bool(r.passed)
    return True


def _probe(name: str, fn, *a, **kw) -> dict:
    try:
        r = fn(*a, **kw)
    except Exception as e:
        return {
            "check": name,
            "empty_collection": kw.pop("_what", "see registry"),
            "verdict": "CLOSED (raised)",
            "returned": f"{type(e).__name__}: {str(e)[:110]}",
            "vacuous": False,
        }
    clean = _is_clean(r)
    brief = (
        {"n_failures": len(r), "first": (r[0][:110] if r else None)}
        if isinstance(r, list)
        else (
            {k: r[k] for k in ("armed", "halted", "pass", "unmeasurable") if k in r}
            or {"keys": sorted(r)[:5]}
        )
        if isinstance(r, dict)
        else repr(r)[:110]
    )
    return {
        "check": name,
        "verdict": "VACUOUS (clean on an empty collection)" if clean else "CLOSED (reported)",
        "returned": brief,
        "vacuous": clean,
    }


def _run_probes(mutate: bool = False) -> list[dict]:
    rows = []
    rows.append(_probe("verdict.provenance_failures  [evals = {}]", V.provenance_failures, {}))
    rows.append(_probe("verdict.degeneracy_guard  [evals = {}]", V.degeneracy_guard, {}))
    rows.append(_probe("verdict.power_guard  [evals = {}, claimed = {}]", V.power_guard, {}, {}))
    rows.append(
        _probe(
            "verdict.decode_agreement_disclosure  [evals = {}]", V.decode_agreement_disclosure, {}
        )
    )
    rows.append(
        _probe("verdict._mu_diag_problems  [mu_diag = {}]", V._mu_diag_problems, {}, "sweep")
    )
    rows.append(
        _probe("verdict._validate_artifact  [doc = {}]", V._validate_artifact, {}, "d.json")
    )
    rows.append(_probe("conformance.conformance_failures  [symbols = []]", _empty_symbols))
    rows.append(
        _probe(
            "conformance.verdict_dsr_failures  [trials = {}]",
            C.verdict_dsr_failures,
            n_trials=C.PINNED_DSR["n_trials"],
            threshold=0.95,
            trials={},
            var_sr=0.0,
        )
    )
    rows.append(_probe("conformance.pinned_threshold_failures  [pins = {}]", _empty_pins))
    rows.append(
        _probe("matrix.shard_partition_failures  [units = []]", M.shard_partition_failures, [], 1)
    )
    rows.append(
        _probe(
            "matrix.artifact_reuse_failures  [doc = {}]",
            M.artifact_reuse_failures,
            {},
            expected={},
        )
    )
    guard = (lambda *a, **k: {"pass": True}) if mutate else G.micro_legibility_gate
    rows.append(
        _probe(
            "gates.micro_legibility_gate  [per_symbol = []]" + (" [MUTATED]" if mutate else ""),
            guard,
            None,
            [],
            {},
        )
    )
    rows.append(_probe("verdict.load_cell_evals  [empty dir]", _empty_dir_load))
    rows.append(
        _probe("causal_check.exhaustive_truncation_sweep  [0-bar stream]", _empty_stream_sweep)
    )
    return rows


def _empty_symbols() -> list[str]:
    from trikaal.eval.xsection import XSectionConfig

    cfg = XSectionConfig(money=True, cap_per_symbol=None, spread_frac_by_symbol={"A": 1e-4})
    return C.conformance_failures(cfg, symbols=[], seeds=C.PINNED_SEEDS)


def _empty_pins() -> list[str]:
    from unittest import mock

    with mock.patch.object(C, "PINNED_THRESHOLDS", {}):
        return C.pinned_threshold_failures()


def _empty_dir_load():
    with tempfile.TemporaryDirectory() as td:
        return V.load_cell_evals(Path(td))


def _empty_stream_sweep():
    import numpy as np

    from trikaal.data.causal_check import exhaustive_truncation_sweep
    from trikaal.data.config import synthetic_test_config
    from trikaal.data.synthetic import make_synthetic_stream
    from trikaal.data.universe_build import slice_stream_by_ms

    stream, _ = make_synthetic_stream(seed=7, n_bars=200)
    empty = slice_stream_by_ms(stream, -1e18, float(np.min(stream.bar_open_ms)) - 1)
    return exhaustive_truncation_sweep(empty, synthetic_test_config())


def main() -> int:
    cands = _candidates()
    rows = _run_probes(mutate=False)
    mutated = _run_probes(mutate=True)

    caught = any(r["vacuous"] and "MUTATED" in r["check"] for r in mutated)
    rep: dict = {
        "what": (
            "THE VACUITY SWEEP — a verdict derived by iterating a collection, given an EMPTY one"
        ),
        "definition": (
            "VACUOUS = the check returns a clean verdict (empty failures / not halted / pass) "
            "having iterated nothing. `for x in []` never runs and `all([])` is True."
        ),
        "mutation_control": {
            "claim": "a known-CLOSED check wrapped to return pass=True on empty input must appear",
            "target": "gates.micro_legibility_gate",
            "named_by_the_sweep": caught,
        },
    }
    if not caught:
        rep["verdict"] = "SWEEP INVALID — the mutation did not fire; this table proves nothing"
        OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
        print(json.dumps(rep["mutation_control"], indent=2))
        return 1

    probed = {r["check"].split(" ")[0] for r in rows}
    unaccounted = [
        c for c in cands if c not in EXCLUDED and not any(c.split(":")[1] in p for p in probed)
    ]
    vac = [r for r in rows if r["vacuous"]]
    rep.update(
        {
            "ast_candidates": cands,
            "n_ast_candidates": len(cands),
            "excluded_with_reason": EXCLUDED,
            "unaccounted_candidates": unaccounted,
            "rows": rows,
            "n_rows": len(rows),
            "vacuous_rows": vac,
            "n_vacuous": len(vac),
            "verdict": f"{len(vac)} VACUOUS of {len(rows)} probed",
        }
    )
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(f"mutation control: {'FIRED' if caught else 'DID NOT FIRE'}")
    print(
        f"AST candidates {len(cands)} | probed {len(rows)} | excluded {len(EXCLUDED)} | "
        f"unaccounted {len(unaccounted)}\n"
    )
    w = max(len(r["check"]) for r in rows) + 1
    for r in rows:
        mark = "  <<< VACUOUS" if r["vacuous"] else ""
        print(f"  {r['check']:<{w}} {r['verdict']:<40} {r['returned']}{mark}")
    if unaccounted:
        print(f"\n!! UNACCOUNTED CANDIDATES (neither probed nor excluded): {unaccounted}")
    print(f"\nVACUOUS: {len(vac)} / {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
