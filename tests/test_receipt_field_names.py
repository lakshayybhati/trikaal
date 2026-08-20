"""A CORRECT MEASUREMENT UNDER THE WRONG NAME PASSES EVERY CHECK, BECAUSE BOTH VALUES ARE REAL.

``m6_clip_nan_scan.json`` published **3,018** as ``longest_mid_segment_run_dim0``, and the prose
said "reaches 3,018 CONSECUTIVE bars". 3,018 is a real number — it is the COUNT of exactly-``-5.0``
bars in DOGEUSDT segment 0, dim 0 — but that segment's longest consecutive run is **9**. The island
query computed ``row_number()`` over the ALREADY-FILTERED set in both CTEs, so the difference
cancelled to zero, every segment collapsed into one group, and ``count(*)`` returned the total.

Nothing could have caught it. The value was real, the field was plausible, the receipt round-tripped
byte-identically, and no schema or type check was violated. The true lake-wide longest run is
**265** (RNDRUSDT, segment 0, dim 6) — verified two independent ways, in DuckDB and in plain Python
over the raw column — against a longest TRAILING run of 2, so the argument the number supports was
never in doubt. Only the number a reader would quote was wrong.

WHAT THIS FILE DOES ABOUT IT, AND WHAT IT CANNOT DO.

It asserts the one relationship that makes THIS class visible — a run cannot exceed the count of
matching items it is drawn from — and sweeps every tracked receipt for names that IMPLY a testable
property: a fraction that must lie in [0,1], a count that must be an integer, a mean that cannot
exceed its own max. That is the whole of the general handle available.

★ IT DOES NOT AND CANNOT VERIFY THAT A FIELD MEANS WHAT IT SAYS. A median stored as ``*_mean``, a
p95 stored as ``*_max``, a wall-clock stored as ``*_cpu_s`` — each is a plausible value in a
plausible field with no sibling to contradict it, and no structural check reaches them. The honest
scope is: names that imply arithmetic are checked; names that imply only semantics are not.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCAN = REPO / "runs_manifest/m6_clip_nan_scan.json"


@pytest.fixture(scope="module")
def scan() -> dict:
    return json.loads(SCAN.read_text())


# ── the specific defect ───────────────────────────────────────────────────────────────────────
def test_a_run_cannot_exceed_the_bars_it_is_drawn_from(scan) -> None:
    """THE INVARIANT THAT WOULD HAVE CAUGHT IT. 3,018 as a run inside a segment holding 3,018
    matching bars is only possible if every one of them is contiguous — which is the tell."""
    run = scan["longest_mid_segment_RUN_any_z_dim"]
    assert run["longest_run_bars"] <= run["matching_bars_in_that_segment"], run
    assert run["longest_run_bars"] < run["matching_bars_in_that_segment"], (
        "the longest run EQUALS the total matching count, which means either one contiguous block "
        "or — far more likely — a count published under a run's name again"
    )


def test_the_published_run_is_the_measured_one(scan) -> None:
    run = scan["longest_mid_segment_RUN_any_z_dim"]
    assert (run["symbol"], run["segment_id"], run["longest_run_bars"]) == ("RNDRUSDT", 0, 265), run
    assert run["dim"] == "x_6"


def test_the_run_is_the_max_over_the_per_dim_table(scan) -> None:
    """Anti-vacuity: the headline must be the maximum of the table beside it, not a stray value."""
    per_dim = scan["longest_mid_segment_run_by_dim"]
    assert len(per_dim) >= 5, per_dim
    best = max(v["longest_run_bars"] for v in per_dim.values())
    assert scan["longest_mid_segment_RUN_any_z_dim"]["longest_run_bars"] == best


def test_the_prose_quotes_the_run_not_the_count(scan) -> None:
    why = scan["WHY_THE_DISTINCTION_MATTERS"]
    assert "265" in why and "CONSECUTIVE" in why, why
    assert "3,018 CONSECUTIVE" not in why, "the count is being called a run again"


def test_the_correction_is_recorded_rather_than_quietly_dropped(scan) -> None:
    """3,018 was a real measurement. Deleting it would hide that the receipt once misnamed it."""
    note = scan["CORRECTION_2026_08_20"]
    assert "3,018" in note and "COUNT" in note, note


# ── the general sweep: names that imply arithmetic ────────────────────────────────────────────
def _leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def _receipts() -> list[Path]:
    return sorted(REPO.joinpath("runs_manifest").glob("*.json"))


def test_every_field_named_a_fraction_lies_in_the_unit_interval() -> None:
    bad = []
    checked = 0
    for f in _receipts():
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for path, v in _leaves(doc):
            leaf = path.rsplit(".", 1)[-1].lower()
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if not math.isfinite(v):
                continue
            if leaf.startswith("frac") or leaf.endswith(("_frac", "_share", "_fraction")):
                checked += 1
                if not 0.0 <= v <= 1.0:
                    bad.append(f"{f.name}{path} = {v}")
    assert checked >= 100, f"only {checked} fraction-named fields swept — is this looking at data?"
    assert not bad, bad


def test_no_mean_exceeds_its_own_sibling_max() -> None:
    bad = []
    checked = 0
    for f in _receipts():
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        stack = [(doc, f.name)]
        while stack:
            node, where = stack.pop()
            if isinstance(node, dict):
                for a, b in (("mean", "max_abs"), ("mean", "max"), ("min", "max")):
                    if a in node and b in node:
                        va, vb = node[a], node[b]
                        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                            checked += 1
                            cmp_a = abs(va) if b == "max_abs" else va
                            if cmp_a > vb + 1e-12:
                                bad.append(f"{where}: {a}={va} > {b}={vb}")
                stack.extend((v, where) for v in node.values() if isinstance(v, (dict, list)))
            elif isinstance(node, list):
                stack.extend((v, where) for v in node if isinstance(v, (dict, list)))
    assert checked >= 20, f"only {checked} mean/max sibling pairs found"
    assert not bad, bad


def test_the_only_non_integer_count_named_field_is_an_effective_sample_size() -> None:
    """``N_breadth_eff = 1.5`` trips an ``n_``-prefix heuristic and is CORRECT — an effective
    sample size is continuous by construction. Named here so the exemption is visible and so a
    genuinely fractional count elsewhere still fails."""
    offenders = []
    for f in _receipts():
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for path, v in _leaves(doc):
            leaf = path.rsplit(".", 1)[-1]
            low = leaf.lower()
            if isinstance(v, bool) or not isinstance(v, float) or not math.isfinite(v):
                continue
            if low.startswith("n_") or low.endswith(("_count", "_bars", "_rows", "_files")):
                if not v.is_integer() and not low.endswith("_eff"):
                    offenders.append(f"{f.name}{path} = {v}")
    assert not offenders, f"count-named fields holding a fractional value: {offenders}"
