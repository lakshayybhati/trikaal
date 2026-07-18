"""The M6 verdict — the ONLY runnable path that may produce the SURVIVES/NULL decision.

    PYTHONPATH=src python3 scripts/m6_verdict.py --artifacts runs/m6/eval [--out ...]

Sequence (docs/m6_prereg.md §3/§3a/§5; the logic lives in ``trikaal.eval.verdict``):

1. **Conformance gate FIRST** — the money surface is rebuilt via the shared
   ``conformance.money_config`` and asserted against every §3a pin; ANY divergence prints the
   full failure list and exits non-zero before a single artifact is opened.
2. The 15 per-(cell, seed) eval artifacts are loaded **by content hash** against their index.
3. The artifact grid must BE the pinned §3a money grid (start + length of
   ``primary_region_grid_ms``). ``--allow-toy-grid`` relaxes ONLY this check for fixture/dry
   runs and is recorded loudly in the manifest (``grid_pinned: false``) — a manifest without
   ``grid_pinned: true`` can never be quoted as the M6 outcome.
4. ``assemble_verdict`` evaluates §3's five conjunctive clauses (paired bootstraps on (4−5),
   (4−2), (5−2); MDE_paired with the no-ceiling rule; the 0.5 floor; the pinned N=180 DSR) and
   §5's NULL-fallback, and the manifest — every clause's number + pass/fail + the final word —
   is written durably (content-hashed), never stdout-only.

``harness.run_harness`` (M5) and ``xsection.ablation_verdict`` (point diagnostics) are NOT
decision paths; the third pre-training audit exists because nothing wired the §3 clauses
together until this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from trikaal.eval.conformance import MDE_INPUTS, PINNED_SEEDS, conformance_failures, money_config
from trikaal.eval.verdict import (
    VerdictInputError,
    assemble_verdict,
    load_cell_evals,
)
from trikaal.eval.xsection import primary_region_grid_ms

PREREG = Path("docs/m6_prereg.md")
DEFAULT_OUT = Path("runs_manifest/m6_verdict.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", required=True, type=Path, help="dir of the 15 eval artifacts")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="verdict manifest path")
    ap.add_argument(
        "--allow-toy-grid",
        action="store_true",
        help="accept a non-pinned grid (KAT fixtures / toy dry runs); recorded in the manifest",
    )
    args = ap.parse_args()

    # 1) the §3a conformance gate — before any artifact is opened
    mde = json.loads(MDE_INPUTS.read_text())
    cfg = money_config()
    fails = conformance_failures(cfg, symbols=list(mde["symbols_sampled"]), seeds=PINNED_SEEDS)
    if fails:
        print("VERDICT REFUSED — config diverges from the §3a surface:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("[gate] §3a conformance PASS")

    # 2) content-addressed artifact load
    try:
        evals, shas = load_cell_evals(args.artifacts)
    except VerdictInputError as e:
        print(f"VERDICT REFUSED — {e}", file=sys.stderr)
        return 1
    print(f"[load] 15 artifacts content-verified from {args.artifacts}")

    # 3) the grid must be the pinned §3a money grid (unless explicitly toy)
    pinned = primary_region_grid_ms(cfg)
    g = next(iter(evals.values()))["grid"]
    grid_pinned = int(g["start_ms"]) == int(pinned[0]) and int(g["n_periods"]) == int(pinned.size)
    if not grid_pinned and not args.allow_toy_grid:
        print(
            f"VERDICT REFUSED — artifact grid (start {g['start_ms']}, T {g['n_periods']}) is "
            f"not the pinned §3a money grid (start {int(pinned[0])}, T {int(pinned.size)}); "
            "pass --allow-toy-grid ONLY for fixtures/dry runs",
            file=sys.stderr,
        )
        return 1

    # 4) the five clauses + §5 fallback → the durable, content-hashed manifest
    manifest = assemble_verdict(
        evals, shas, tabled_mde_h15=float(mde["h15_pooled"]["MDE_annualized_IR"])
    )
    manifest["grid_pinned"] = bool(grid_pinned)
    manifest["conformance"] = "PASS"
    manifest["prereg_sha256"] = hashlib.sha256(PREREG.read_bytes()).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    v = manifest["verdict"]
    print(f"[clauses] h={manifest['primary_h']} grid_pinned={grid_pinned}")
    for name, c in manifest["clauses"].items():
        print(f"  {name}: {'PASS' if c['pass'] else 'FAIL'}")
    word = v["primary"]
    if v["fallback"] is not None:
        word += f" (fallback §5: {v['fallback']['word']}; double_null={v['double_null']})"
    print(f"[verdict] {word}")
    print(f"[manifest] → {args.out} ({manifest['content_hash'][:23]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
