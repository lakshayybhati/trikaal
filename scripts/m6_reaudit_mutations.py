"""§7 v1.6.25 — THE RE-AUDIT MUTATION RECEIPT. Every new gate proven capable of FAILING.

    PYTHONPATH=src python3 scripts/m6_reaudit_mutations.py

WHY THIS SCRIPT AND NOT EIGHT HAND-CHECKS. The re-audit's headline was a gate that shipped DEAD —
`_validate_artifact`'s codebook block sat below `return bad`, so the fourth "specified but not
enforced" instance was declared closed by a fix that never executed. The lesson is not "read more
carefully"; it is that **a gate is closed only when something has watched it fail.** So each fix in
this pass gets its pre-fix source restored in an isolated copy of the tree, and the tests written
for it must FAIL there. If they pass under the mutation, the test proves nothing and the finding is
NOT closed.

THE HARNESS MUST TELL ITS OWN BUGS FROM ITS FINDINGS — three of my probes have manufactured false
findings from their own defects. So every row runs THREE stages, in order, and any stage can only
produce the verdicts below:
  1. BASELINE — the unmutated copy must PASS the named tests. A failure here is HARNESS_BROKEN
     (a bad copy, a missing fixture), never evidence about the mutation.
  2. APPLY    — the pre-fix string must be found EXACTLY ONCE. Zero or many is NOT_APPLIED, which
     is a harness/drift problem, never a finding.
  3. MUTANT   — the named tests must FAIL. Passing is ★ NOT CLOSED ★: the fix is in the tree but
     nothing detects its removal.

The real repository is never touched: everything happens in a temporary copy.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Long anchor strings, split so each SOURCE line fits; the values are byte-exact.
_CB_TAIL = (
    '                    "catches COLLAPSE only \u2014 it does not certify the quantizer '
    'is competitive."\n                )\n'
)
_V12_HYBRID = (
    "var_sr_v12 = float(np.var(np.array([trials[k] for k in sorted(trials)], dtype=np.float64)))"
)
_STEPS_TRUE = (
    '                "SMOKE-TEST budget, and the money run would have inherited it. '
    '26,003 steps is "'
)
_STEPS_STALE = (
    '                "default 2000 is BELOW that band; the operative budget is not a '
    'spec constant. "'
)
OUT = REPO / "runs_manifest" / "m6_reaudit_mutations.json"
COPY = ("src", "tests", "scripts", "docs", "runs_manifest")
FILES = ("pyproject.toml", "uv.lock")

# (finding, why it matters, file, pre-fix string, post-fix string, tests that must fail)
MUTATIONS: list[dict] = [
    {
        "finding": "R1",
        "what": "the codebook >=95% gate sits below `return bad` — born dead in 591ec44",
        "file": "src/trikaal/eval/verdict.py",
        "post": _CB_TAIL + "    return bad\n",
        "pre": _CB_TAIL,
        "pre_prefix": "    return bad\n    # §7 v1.6.22: the codebook diagnostic is REQUIRED",
        "tests": ["tests/eval/test_codebook_gate.py"],
    },
    {
        "finding": "R2",
        "what": "the C-3 unit convention reverts to each trial's OWN horizon (the easier basis)",
        "file": "src/trikaal/eval/verdict.py",
        "post": "trials[(cid, seed, h, k)] = ir_ann / float(np.sqrt(periods_per_year(PRIMARY_H)))",
        "pre": "trials[(cid, seed, h, k)] = ir_ann / float(np.sqrt(periods_per_year(h)))",
        "tests": ["tests/eval/test_dsr_unit_convention.py"],
    },
    {
        "finding": "R3",
        "what": "the dual-specification v1.2 leg reads the AMENDED trial set (the hybrid)",
        "file": "src/trikaal/eval/verdict.py",
        "post": "var_sr_v12 = float(np.var(own_h_trial_values(evals, sorted(trials))))",
        "pre": _V12_HYBRID,
        "tests": ["tests/eval/test_dsr_unit_convention.py"],
    },
    {
        "finding": "R4",
        "what": "the G-§8.C.3 verdict + BSQ disclosure never reach the manifest",
        "file": "src/trikaal/eval/verdict.py",
        "post": '        "external_validation": external_validation_record,\n',
        "pre": "",
        "tests": ["tests/eval/test_manifest_required_fields.py"],
    },
    {
        "finding": "R6a",
        "what": "git_commit drops out of the identity surface (a stale-revision shard assembles)",
        "file": "src/trikaal/utils/provenance.py",
        "post": '    "image",\n    "git_commit",\n',
        "pre": '    "image",\n',
        "tests": ["tests/run/test_fanout_refusal.py"],
    },
    {
        "finding": "R6b",
        "what": "the step budgets drop out of the identity surface (a 2,000-step shard assembles)",
        "file": "src/trikaal/utils/provenance.py",
        "post": '    "steps_stage1",\n    "steps_stage2",\n',
        "pre": "",
        "tests": ["tests/run/test_fanout_refusal.py"],
    },
    {
        "finding": "R7",
        "what": "the money manifest calls the step budget 'the default 2000 … not a spec constant'",
        "file": "scripts/m6_money_run.py",
        "post": _STEPS_TRUE,
        "pre": _STEPS_STALE,
        "tests": ["tests/run/test_money_driver.py"],
    },
    {
        "finding": "R10a",
        "what": "the money label enters at C_t instead of C_{t+1} — same-bar-entry lookahead",
        "file": "src/trikaal/eval/harness.py",
        "post": "y[s0 + p] = clp[p + h] - clp[p + 1]  # log(C_{t+h}/C_{t+1})",
        "pre": "y[s0 + p] = clp[p + h] - clp[p]  # LOOKAHEAD MUTANT",
        "tests": ["tests/eval/test_forward_log_returns.py"],
    },
    {
        "finding": "R12e",
        "what": "cell 5's required disclosures are measured on UNSHUFFLED micro",
        "file": "src/trikaal/eval/xsection.py",
        # the `first_arm is None` guard above is left in place: it still holds, and leaving it
        # keeps the mutation to the ONE line that carried the defect.
        "post": "    _x0, _m0 = first_arm\n",
        "pre": '    _se0 = next(iter(prepared.values()))["se"]\n'
        "    _x0, _m0 = select_arm(_se0.x, _se0.mask, arm)\n",
        "tests": ["tests/eval/test_placebo_diagnostic_inputs.py"],
    },
    # ------------------------------------------------------------------ THE NEGATIVE CONTROL
    # "9/9 CLOSED" is itself a check that has never been seen to fail. This row mutates a COMMENT
    # — semantics untouched — and the harness MUST report NOT CLOSED for it. If it reports CLOSED,
    # the tests are failing for a reason unrelated to the defect (a broken copy, an import, a
    # stray edit) and every green row above is suspect.
    {
        "finding": "CONTROL",
        "what": "a comment-only edit — semantically a no-op; the tests MUST still pass",
        "file": "src/trikaal/eval/verdict.py",
        "post": "# ---- prereg pins (literals mirror docs/m6_prereg.md §3/§3a/§5)",
        "pre": "# ---- prereg pins (this comment is the negative control; nothing semantic moved)",
        "tests": ["tests/eval/test_codebook_gate.py"],
        "control": True,
    },
]


def _run(tests: list[str], root: Path) -> tuple[int, str]:
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return p.returncode, (p.stdout + p.stderr)[-1500:]


def _stage_tree(dst: Path) -> None:
    for d in COPY:
        shutil.copytree(
            REPO / d,
            dst / d,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    for f in FILES:
        shutil.copy2(REPO / f, dst / f)


def _apply(root: Path, m: dict) -> str | None:
    """Rewrite the fixed source back to its pre-fix form. Returns an error string, or None."""
    path = root / m["file"]
    src = path.read_text()
    edits = [(m["post"], m["pre"]), *m.get("pre_extra", [])]
    for post, pre in edits:
        n = src.count(post)
        if n != 1:
            return (
                f"NOT_APPLIED: the post-fix string occurs {n}x (expected exactly 1) in {m['file']}"
            )
        src = src.replace(post, pre, 1)
    if m.get("pre_prefix"):  # R1: the `return` moves back ABOVE the block it guards
        src = src.replace(
            "    # §7 v1.6.22: the codebook diagnostic is REQUIRED", m["pre_prefix"], 1
        )
    path.write_text(src)
    return None


def main() -> int:
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="trikaal_mut_") as td:
        base = Path(td) / "base"
        _stage_tree(base)
        for m in MUTATIONS:
            row = {
                "finding": m["finding"],
                "what": m["what"],
                "tests": m["tests"],
                "is_control": bool(m.get("control")),
            }
            # 1. BASELINE — the unmutated copy must pass, or nothing below is evidence
            rc0, tail0 = _run(m["tests"], base)
            row["baseline_exit"] = rc0
            if rc0 != 0:
                row["verdict"] = "HARNESS_BROKEN — the unmutated copy fails; NOT a finding"
                row["baseline_tail"] = tail0
                rows.append(row)
                continue
            # 2. APPLY into a throwaway clone of the copy
            work = Path(td) / f"mut_{m['finding']}"
            shutil.copytree(base, work, symlinks=True)
            err = _apply(work, m)
            if err:
                row["verdict"] = err
                rows.append(row)
                continue
            # 3. MUTANT — the tests MUST fail
            rc1, tail1 = _run(m["tests"], work)
            row["mutant_exit"] = rc1
            row["mutant_tail"] = tail1[-400:]
            if m.get("control"):
                row["verdict"] = (
                    "CONTROL OK — a semantic no-op does NOT fail the tests"
                    if rc1 == 0
                    else "★ CONTROL FAILED — the tests fail on a comment edit; every green row "
                    "above is suspect (the harness, not the code, is what is being measured)"
                )
            else:
                row["verdict"] = (
                    "CLOSED — the pre-fix code fails the new tests"
                    if rc1 != 0
                    else "★ NOT CLOSED — the mutation PASSES; the test cannot detect the defect"
                )
            rows.append(row)
            shutil.rmtree(work, ignore_errors=True)

    real = [r for r in rows if not r.get("is_control")]
    ok = all(
        r["verdict"].startswith("CLOSED") or r["verdict"].startswith("CONTROL OK") for r in rows
    )
    rep = {
        "receipt": "m6_reaudit_mutations",
        "what": "every §7 v1.6.25 fix, proven detectable by restoring its pre-fix source",
        "protocol": "baseline must PASS -> mutation applied exactly once -> mutant must FAIL",
        "n": len(real),
        "n_closed": sum(1 for r in real if r["verdict"].startswith("CLOSED")),
        "negative_control": next(
            (r["verdict"] for r in rows if r.get("is_control")), "ABSENT — no control was run"
        ),
        "rows": rows,
        "verdict": "GREEN — every fix has a test that fails without it, and a semantic no-op does "
        "not fail those tests"
        if ok
        else "RED — a fix is undetectable by its own tests, or the negative control failed",
    }
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    for r in rows:
        print(f"{r['finding']:<8} {r['verdict']}")
    print(f"\n{rep['n_closed']}/{rep['n']} closed → {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
