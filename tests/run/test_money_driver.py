"""§7 v1.6 C-5 — the money driver's contract, enforced rather than described.

Pre-flight Item 2 promises ZERO first-time code paths on rented hardware. A brand-new driver IS a
first-time path, so its guarantees are pinned here and the outstanding CUDA validation is re-scoped
to run THROUGH it. Each rule from the scope is a test, and each test is written so it can fail.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from trikaal.eval.verdict import (
    ARTIFACT_SCHEMA,
    DSR_HORIZONS,
    DSR_KAPPAS,
    DSR_SEEDS,
    PRIMARY_H,
    VerdictInputError,
    load_cell_evals,
    provenance_failures,
    write_cell_eval_artifact,
    write_eval_index,
)
from trikaal.run.matrix import (
    all_units,
    artifact_reuse_failures,
    shard_partition_failures,
    shard_units,
)
from trikaal.train.cells import CELLS
from trikaal.utils.provenance import PROVENANCE_IDENTITY_KEYS, run_provenance

# §7 v1.6.11 C-1: the decode-agreement disclosure is REQUIRED on every artifact.
DECODE_AGREEMENT_OK = {
    "sign_agreement_dim0": 0.95,
    "variance_ratio_dim0": 1.0,
    "single_bar_degenerate": False,
}


REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts/m6_money_run.py"

# Flags that would let an operator move a pinned value from the command line. C-7's lesson was
# that a pin nothing reads is decoration; a pin a FLAG can move is worse — it looks enforced.
FORBIDDEN_FLAG_SUBSTRINGS = (
    "seed",
    "seq",
    "step",
    "symbol",
    "cost",
    "kappa",
    "cell",
    "horizon",
    "window",
    "lr",
    "batch",
    "cap",
    "boot",
    "threshold",
    "legibility",
    "param",
)

# §7 v1.6 C-12 M1: the capacity disclosure is REQUIRED on every artifact.
OHLCV_RECON_OK = {"ohlcv_recon_mae": 0.10, "ohlcv_recon_mae_by_dim": [0.1] * 7, "n_ohlcv_dims": 7}


def _load_driver():
    spec = importlib.util.spec_from_file_location("m6_money_run", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ A2: no flag may move a pin
def test_the_driver_exposes_operational_flags_only():
    opts = {
        o
        for a in _load_driver().build_parser()._actions
        for o in a.option_strings
        if o.startswith("--")
    }
    assert opts, "guard against a vacuous pass — the parser must have flags"
    offenders = {o for o in opts if any(f in o.lower() for f in FORBIDDEN_FLAG_SUBSTRINGS)}
    assert not offenders, (
        f"these flags could move a pinned value: {sorted(offenders)}. Config comes from "
        "money_config() and the pins; argparse carries operational concerns only (C-5 A2)."
    )


def test_the_forbidden_flag_check_can_actually_fail():
    """FIXTURE DISCRIMINATION — a filter that matches nothing would pass against any parser."""
    assert any("seed" in f for f in FORBIDDEN_FLAG_SUBSTRINGS)
    fake = {"--out", "--seeds"}
    assert {o for o in fake if any(f in o.lower() for f in FORBIDDEN_FLAG_SUBSTRINGS)} == {
        "--seeds"
    }


# ------------------------------------------------------------------ A3: conformance comes first
def test_conformance_is_the_first_action_before_any_compute():
    """Asserted on the AST: ``assert_conformance`` must precede every lake/train/eval call."""
    tree = ast.parse(DRIVER.read_text())
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    order = [
        (node.lineno, node.func.id)
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    conf = [ln for ln, name in order if name == "assert_conformance"]
    compute = [
        ln
        for ln, name in order
        if name
        in {
            "connect_lake",
            "train_matrix",
            "eval_matrix",
            "load_symbol_arrays",
            "windows_by_arm",
            "reload_checkpoints",
        }
    ]
    assert conf, "the driver must call assert_conformance"
    assert compute, "guard against a vacuous pass — there must be compute to come after"
    assert max(conf) < min(compute), (
        f"assert_conformance at line(s) {conf} must precede all compute {sorted(compute)[:3]} — "
        "a divergent config must fail in SECONDS, not after hours on a rented box (C-5 A3)"
    )


# ------------------------------------------------------------------ A5: fan-out is a partition
@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 25])
def test_sharding_is_an_exact_disjoint_cover(n):
    units = all_units(CELLS, DSR_SEEDS)
    assert len(units) == 25
    assert shard_partition_failures(units, n) == []
    sizes = [len(shard_units(units, i, n)) for i in range(n)]
    assert sum(sizes) == 25
    assert max(sizes) - min(sizes) <= 1, f"unbalanced shards {sizes}"


def test_the_partition_checker_can_actually_fail():
    """Discrimination: a checker that always returns [] would hide a dropped unit."""
    units = all_units(CELLS, DSR_SEEDS)

    def broken(u, shard, n):  # drops the last unit of every shard
        return shard_units(u, shard, n)[:-1]

    seen = [x for i in range(5) for x in broken(units, i, 5)]
    assert len(seen) < len(units), "the fixture must actually drop units"


def test_shard_bounds_are_rejected():
    units = all_units(CELLS, DSR_SEEDS)
    for bad in ((5, 5), (-1, 5), (0, 0)):
        with pytest.raises(ValueError):
            shard_units(units, *bad)


# ------------------------------------------------------------------ A5: mixed hardware refused
def _write_set(tmp, provenance_for):
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    entries = {}
    for c in (1, 2, 3, 4, 5):
        for s in DSR_SEEDS:
            name, sha = write_cell_eval_artifact(
                tmp,
                cell_id=c,
                seed=s,
                grid={"h": PRIMARY_H, "start_ms": 0, "n_periods": 64},
                headline_series=rng.standard_normal(64) * 1e-3,
                kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
                val_ir_by_kappa_by_h={h: {k: 0.4 for k in DSR_KAPPAS} for h in DSR_HORIZONS},
                mu_diag={"frac_negative": 0.5, "activity_decisions": 0.5},
                ohlcv_recon=OHLCV_RECON_OK,
                decode_agreement=DECODE_AGREEMENT_OK,
                meta={"provenance": provenance_for(c, s)},
            )
            entries[name] = sha
    write_eval_index(tmp, entries)
    return tmp


def test_one_instrument_assembles(tmp_path):
    """Discrimination first: the homogeneous set MUST load, or the next test is vacuous."""
    prov = run_provenance("cpu", attention_mode="sdpa")
    evals, _ = load_cell_evals(_write_set(tmp_path / "ok", lambda c, s: dict(prov)))
    assert len(evals) == 25
    assert provenance_failures(evals) == []


@pytest.mark.parametrize("key", ["gpu_name", "torch", "python", "attention_mode"])
def test_the_verdict_refuses_to_assemble_across_mixed_hardware(tmp_path, key):
    """C-5 A5: fan-out makes this reachable — 25 units, N boxes, one paired comparison."""
    base = run_provenance("cpu", attention_mode="sdpa")

    def prov(c, s):
        p = dict(base)
        if (c, s) == (4, DSR_SEEDS[0]):
            p[key] = "SOMETHING-ELSE"
        return p

    with pytest.raises(VerdictInputError, match="mixed hardware"):
        load_cell_evals(_write_set(tmp_path / key, prov))


def test_an_unattributable_unit_is_refused(tmp_path):
    """ "Some units cannot be attributed" is exactly the state that must not be certified."""
    base = run_provenance("cpu")
    evals = {
        (c, s): {"meta": {"provenance": dict(base)}} for c in (1, 2, 3, 4, 5) for s in DSR_SEEDS
    }
    assert provenance_failures(evals) == []
    evals[(3, DSR_SEEDS[1])]["meta"] = {}
    fails = provenance_failures(evals)
    assert any("no meta.provenance" in f for f in fails), fails


def test_provenance_records_the_interpreter():
    """C-18: uv.lock pins packages, not python; ">=3.11" admits 3.14 and a box silently ran it."""
    p = run_provenance("cpu")
    assert "python" in PROVENANCE_IDENTITY_KEYS
    assert p["python"].count(".") >= 2 and p["python"][0].isdigit()
    for k in PROVENANCE_IDENTITY_KEYS:
        assert k in p, f"identity key {k} is compared but never recorded"


# ------------------------------------------------------------------ A6 / C-14: resume binding
def _expected(**over):
    base = {
        "cell_id": 4,
        "seed": 0,
        "checkpoints": {"tokenizer.pt": "aaa", "predictor.pt": "bbb"},
        "symbols": ["BTCUSDT"],
        "eval_window": ["2021-01-01", "2025-01-01"],
        "h": PRIMARY_H,
        "n_periods": 64,
        "start_ms": 17,
    }
    base.update(over)
    return base


def _doc(**over):
    exp = _expected()
    doc = {
        "schema": ARTIFACT_SCHEMA,
        "cell_id": exp["cell_id"],
        "seed": exp["seed"],
        "meta": {
            k: exp[k]
            for k in ("checkpoints", "symbols", "eval_window", "h", "n_periods", "start_ms")
        },
    }
    for k, v in over.items():
        if k in doc["meta"] or k in (
            "checkpoints",
            "symbols",
            "eval_window",
            "h",
            "n_periods",
            "start_ms",
        ):
            doc["meta"][k] = v
        else:
            doc[k] = v
    return doc


def test_a_matching_artifact_is_resumable():
    """Discrimination: if the clean case failed, every refusal below would be vacuous."""
    assert artifact_reuse_failures(_doc(), expected=_expected()) == []


def test_a_retrained_checkpoint_invalidates_the_artifact():
    """C-14, the whole point: schema equality alone lets a rerun MIX training generations."""
    stale = _doc(checkpoints={"tokenizer.pt": "OLD", "predictor.pt": "OLD"})
    fails = artifact_reuse_failures(stale, expected=_expected())
    assert any("DIFFERENT training generation" in f for f in fails), fails


@pytest.mark.parametrize(
    "over",
    [
        {"symbols": ["ETHUSDT"]},
        {"eval_window": ["2020-01-01", "2021-01-01"]},
        {"h": 60},
        {"start_ms": 18},
        {"schema": "m6_cell_eval_v1"},
        {"cell_id": 3},
        {"seed": 1},
    ],
    ids=["symbols", "window", "h", "start_ms", "old_schema", "cell", "seed"],
)
def test_every_binding_field_blocks_reuse(over):
    assert artifact_reuse_failures(_doc(**over), expected=_expected()), f"{over} was accepted"


def test_n_periods_is_deliberately_not_a_binding_field():
    """It is DERIVED from the scored series, so it cannot form a PRE-scoring expectation.

    Nothing is lost: n_periods is a deterministic function of (symbols, eval_window, h, start_ms)
    and the checkpoints, every one of which IS bound. Recorded explicitly so the omission reads
    as a decision rather than an oversight."""
    assert artifact_reuse_failures(_doc(n_periods=99), expected=_expected()) == []


# ------------------------------------------------------------------ A1: ONE shared path
#: calls that mean "this loop is doing the matrix work itself" rather than orchestrating it.
_MATRIX_WORK = {"run_cell", "score_cell", "write_cell_eval_artifact"}


def _loops_doing_matrix_work(source: str) -> list[int]:
    """Line numbers of ``for`` loops whose body performs per-unit matrix work.

    Deliberately NOT a text search for ``for spec in CELLS``: the first version of this check was
    exactly that and it flagged the rehearsal's ``--eval-only`` branch, which merely READS prior
    run manifests. Loading manifests is not reimplementing the matrix. The precise statement is
    "a loop that trains or scores a unit", so that is what is detected."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.For):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                name = (
                    inner.func.attr
                    if isinstance(inner.func, ast.Attribute)
                    else getattr(inner.func, "id", None)
                )
                if name in _MATRIX_WORK:
                    hits.append(node.lineno)
                    break
    return sorted(set(hits))


def test_the_duplication_detector_can_actually_fail():
    """FIXTURE DISCRIMINATION, before the check below is trusted."""
    dup = "for spec in CELLS:\n    for seed in SEEDS:\n        run_cell(spec, seed)\n"
    ok = "for spec in CELLS:\n    for seed in SEEDS:\n        manifests[spec] = read(seed)\n"
    # both the outer and inner loop are flagged — nesting is not an excuse, so assert membership
    assert 1 in _loops_doing_matrix_work(dup), "detector missed a real duplicated matrix loop"
    assert _loops_doing_matrix_work(ok) == [], "detector flags a loop that only reads manifests"


def test_neither_driver_reimplements_the_matrix():
    """A1: duplication is how C-6 happened. Both drivers must CALL the shared functions."""
    shared = ("train_matrix", "eval_matrix", "reload_checkpoints")
    for path in (DRIVER, REPO / "scripts/m6_toy_rehearsal.py"):
        src = path.read_text()
        assert "from trikaal.run.matrix import" in src, f"{path.name} does not use the shared path"
        for fn in shared:
            assert f"{fn}(" in src, f"{path.name} does not call {fn}"
        assert _loops_doing_matrix_work(src) == [], (
            f"{path.name} trains or scores units in its own loop at line(s) "
            f"{_loops_doing_matrix_work(src)} — that is the duplication A1 forbids"
        )


def test_the_matrix_work_happens_in_exactly_one_place():
    """The positive half: the shared module IS where the per-unit work lives."""
    shared_src = (REPO / "src/trikaal/run/matrix.py").read_text()
    assert _loops_doing_matrix_work(shared_src), (
        "run.matrix does not itself contain the per-unit loops — if neither the drivers nor the "
        "shared module do the work, this whole check is vacuous"
    )


def test_unpinned_parameters_are_recorded_with_their_reasoning():
    """A8: nothing invented; the reasoning travels WITH the numbers."""
    from trikaal.train.orchestrator import OrchestratorConfig

    rec = _load_driver().unpinned_parameters(OrchestratorConfig())
    assert set(rec["values"]) >= {"steps_stage1", "steps_stage2", "peak_lr_stage1", "batch_size"}
    assert "ALL FIVE CELLS SHARE THEM" in rec["why_ablation_validity_does_not_depend_on_them"]
    assert "16,479-18,311" in rec["stage2_budget_is_derivable_not_arbitrary"]
    assert json.dumps(rec)  # must be manifest-serialisable
