"""§7 v1.6.25 — RETAIN P1's COMPLETION EVIDENCE: the first end-to-end money-driver run.

    PYTHONPATH=src python3 scripts/m6_p1_retain.py <run_dir>

WHY. `m6_money_run.py --dry-run` completed for the first time in this project's history on
2026-08-04: 25 units trained, 25 checkpoints reloaded, 25 eval artifacts + index written,
`money_run_manifest.json` produced. That run directory is a SCRATCH path and will not survive; the
standing pre-teardown retention rule (§7 v1.4.4) exists for exactly this. This script lifts the
durable part — the manifest, its content hash, the identity surface, the measured timings and the
loader's refusal — into a committed receipt.

**WHAT THIS RECEIPT IS NOT.** The run declared `money_run=False` and `grid_pinned=False` and scored
a deliberately shortened grid (8 decisions/symbol, 4 symbols). Its NUMBERS ARE WIRING EVIDENCE AND
NOTHING ELSE, and `verdict.load_cell_evals` refuses the artifact set on exactly that ground — the
refusal is re-executed here and recorded, so the receipt carries the proof of its own
un-quotability rather than a promise of it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trikaal.eval.verdict import VerdictInputError, load_cell_evals
from trikaal.utils.provenance import PROVENANCE_IDENTITY_KEYS

OUT = Path("runs_manifest/m6_p1_dry_run_completion.json")


def main(run_dir: str) -> int:
    rd = Path(run_dir)
    man_path = rd / "money_run_manifest.json"
    if not man_path.exists():
        print(f"no money_run_manifest.json under {rd}", file=sys.stderr)
        return 1
    raw = man_path.read_bytes()
    man = json.loads(raw)

    arts = sorted((rd / "eval").glob("cell*_eval.json"))
    prov = json.loads(arts[0].read_text())["meta"]["provenance"]
    stamps = {
        json.dumps(
            {
                k: json.loads(p.read_text())["meta"]["provenance"].get(k)
                for k in PROVENANCE_IDENTITY_KEYS
            },
            sort_keys=True,
        )
        for p in arts
    }

    # the un-quotability proof, re-executed rather than asserted
    try:
        load_cell_evals(rd / "eval")
        refusal = "★ NOT REFUSED — a dry-run set assembled; this is a defect, not a receipt"
    except VerdictInputError as e:
        refusal = next(
            (ln.strip() for ln in str(e).splitlines() if "DRY-RUN" in ln), str(e).splitlines()[0]
        )

    ev = man["eval_seconds"]
    secs = sorted(ev.values())
    rep = {
        "receipt": "m6_p1_dry_run_completion",
        "what": (
            "P1 (fan-out runbook §1a) — `m6_money_run.py --dry-run` ran to completion for the "
            "FIRST TIME in this project's history, at commit a9743938 with the R5 fix applied"
        ),
        "why_retained": (
            "the run directory is a scratch path and does not survive. The standing pre-teardown "
            "retention rule covers less than this, and this is the completion evidence for the "
            "driver that produces every M6 artifact"
        ),
        "p1_exit_code": 0,
        "exit_code_read_from": (
            "the run's own status line appended to its log — NOT from a compound ending in "
            "`echo`, which reports echo's status (the pipefail rider, §7 v1.6.25)"
        ),
        "source_run_dir": str(rd),
        "money_run_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "money_run_manifest": man,
        "stages_completed": [
            "conformance PASS (full money surface, 40 symbols, seeds 0-4) — first action",
            "symbol load + memory-strategy check",
            "25 units trained (0 steps: --dry-run)",
            "25 checkpoints reloaded OK",
            "25 eval artifacts written",
            "eval index written",
            "money_run_manifest.json written",
        ],
        "identity_surface": {
            "n_keys": len(PROVENANCE_IDENTITY_KEYS),
            "keys": list(PROVENANCE_IDENTITY_KEYS),
            "all_stamped": [k for k in PROVENANCE_IDENTITY_KEYS if k not in prov] == [],
            "distinct_stamps_across_units": len(stamps),
            "git_commit": prov.get("git_commit"),
            "steps": [prov.get("steps_stage1"), prov.get("steps_stage2")],
            "deterministic_algorithms": prov.get("deterministic_algorithms"),
            "unavailable_on_this_host": [
                k for k in PROVENANCE_IDENTITY_KEYS if str(prov.get(k)) == "unavailable"
            ],
            "note": (
                "`image`, `gpu_name`, `cuda_build`, `driver_version` read `unavailable` on a "
                "laptop with no CUDA and no TRIKAAL_IMAGE. That is CORRECT here and is exactly "
                "what runbook §1a P3 exists to catch on the box, where they must be real"
            ),
        },
        "un_quotable_by_construction": {
            "money_run": man["money_run"],
            "grid_pinned": man["grid_pinned"],
            "dry_run_bounds": man["dry_run_bounds"],
            "loader_refusal_reexecuted": refusal,
        },
        "measured_timings_local_mps": {
            "wall_h": round(man["wall_s"] / 3600, 2),
            "eval_h": round(sum(ev.values()) / 3600, 2),
            "train_and_overhead_h": round((man["wall_s"] - sum(ev.values())) / 3600, 2),
            "eval_seconds_per_unit": {
                "mean": round(sum(secs) / len(secs), 1),
                "min": secs[0],
                "max": secs[-1],
            },
            "decisions_scored_per_unit": 32,
            "builder_estimate_error": (
                "I estimated the eval leg at 'single-digit minutes total'; it is 638 s = 10.6 min "
                "PER UNIT, ~25x out. I reasoned from decision count (32) and ignored that ~97% of "
                "rollout wall-clock is context prefill at seq_len 512 — item C6 in our own "
                "limitations register. I had the number and did not apply it"
            ),
            "does_not_transfer": (
                "MPS is not CUDA and local timings carry ~8x variance. This bounds NOTHING about "
                "the 4090; the money run is costed from the measured 4090 probes only"
            ),
        },
    }
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "money_run_manifest"}, indent=2)[:2200])
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
