"""§7 v1.4.7 — ONE-TIME CORRECTIVE RECEIPT for the `bit_exact_claim` sufficiency defect ($0).

    PYTHONPATH=src .venv/bin/python scripts/m6_bit_exact_correction.py

THE DEFECT. `attention_mode.determinism_record` derived `bit_exact_claim` from the attention mode
ALONE (`claim = attention_mode == MODE_SDPA`). That encodes invariant 7's sufficiency premise in
code, and the premise is false: deterministic attention is NECESSARY, not SUFFICIENT. Forced
deterministic algorithms are also required, and `orchestrator.py:159` — the PRODUCTION path —
passes `deterministic_algorithms=False`, overriding a default of `True` on a function whose own
docstring calls it "required for the G2 determinism gate and any published run".

WHY A RECEIPT AND NOT A REWRITE. The claim field is already recorded in artifacts. Silently
redefining it would rewrite history. So the historical field is KEPT with its historical meaning,
`bit_exact_preconditions_met` is added as the field any published claim must cite, and this receipt
enumerates every affected record so the correction is discoverable from the ARTIFACTS rather than
only from the prereg.

SCOPE — this is neither broader nor narrower than the following:
  * FALSE: the GPU-TRAINING bit-exactness clause of invariant 7, for the enumerated records.
  * INTACT: invariant 7's data-pipeline / frozen-stats / prediction-replay clause.
  * INTACT: the Gate-A anchor. It anchors the CPU eval-harness replay on a frozen checkpoint and is
    genuinely bit-identical; it does not depend on GPU training determinism.
  * NO prior result is invalidated. No number changes. What changes is that runs which stamped
    `bit_exact_claim: true` may not be quoted as bit-exact GPU training runs.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("runs_manifest/m6_bit_exact_claim_correction.json")

CAVEAT = (
    "bit_exact_claim in this record is TRUE on attention mode alone. The preconditions for "
    "run-to-run GPU bit-exactness were NOT met: torch.use_deterministic_algorithms was not "
    "forced, CUBLAS_WORKSPACE_CONFIG was unset and cudnn.deterministic was unset (the run passed "
    "set_determinism(..., deterministic_algorithms=False)). CUDA reduction order was therefore "
    "unconstrained. This run MUST NOT be quoted as a bit-exact GPU training run. See "
    "bit_exact_preconditions_met (added in prereg s7 v1.4.6) for the predicate a published claim "
    "must cite."
)

# Artifacts that are DERIVED (they quote records rather than being run records themselves) and so
# must not be double-counted as affected runs. This receipt's OWN output is skipped entirely —
# without that, each re-run scans the enumeration it wrote last time and the count inflates, which
# is precisely the kind of silent drift a corrective receipt must not have.
DERIVED = {"runs_manifest/m6_divergence_attribution.json"}
SKIP = {str(OUT)}


def _records() -> list[dict]:
    found: list[dict] = []
    for p in sorted(
        set(Path().glob("runs_manifest/**/*.json")) | set(Path().glob("runs_cloud/**/*.json"))
    ):
        if str(p) in SKIP:
            continue
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue

        def walk(o, path="", _p=p):
            if isinstance(o, dict):
                if "bit_exact_claim" in o and "attention_mode" in o:
                    found.append(
                        {
                            "artifact": str(_p),
                            "json_path": path or "<root>",
                            "run": (path.rsplit("/", 1)[-1] or "<root>"),
                            "bit_exact_claim": o.get("bit_exact_claim"),
                            "attention_mode": o.get("attention_mode"),
                            "deterministic_algorithms": o.get("deterministic_algorithms"),
                            "device": o.get("device"),
                            "seed": o.get("seed"),
                        }
                    )
                    return
                for k, v in o.items():
                    walk(v, f"{path}/{k}", _p)
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{path}[{i}]", _p)

        walk(doc)
    return found


def main() -> int:
    recs = _records()
    affected = [r for r in recs if r["bit_exact_claim"] and not r["deterministic_algorithms"]]
    run_records = [r for r in affected if r["artifact"] not in DERIVED]
    derived_copies = [r for r in affected if r["artifact"] in DERIVED]
    # the same training run is recorded in a per-run manifest AND a rollup, and the rollup exists
    # in both runs_cloud/ and runs_manifest/ — so record count != distinct-run count.
    distinct_runs = sorted({r["run"] for r in run_records})

    out = {
        "receipt": "m6_bit_exact_claim_correction",
        "amendment": "prereg s7 v1.4.7 (supervisor-ordered one-time corrective receipt)",
        "defect": (
            "attention_mode.determinism_record derived bit_exact_claim from the attention mode "
            "ALONE. Deterministic attention is NECESSARY, not SUFFICIENT: forced deterministic "
            "algorithms are also required. orchestrator.py:159 (the PRODUCTION path) passes "
            "deterministic_algorithms=False, overriding a default of True."
        ),
        "counts": {
            "determinism_records_found_total": len(recs),
            "records_claiming_bit_exact_without_forced_determinism": len(affected),
            "of_which_are_run_records": len(run_records),
            "of_which_are_derived_copies_in_this_sessions_own_receipts": len(derived_copies),
            "distinct_training_runs_affected": len(distinct_runs),
            "count_note": (
                "The record count and the run count differ and mean different things. Every "
                "determinism record found claims bit-exactness with forced determinism off — the "
                "rate is 100%, not a sample. But the same training run is recorded in a per-run "
                "run_manifest.json AND in a rollup manifest, and the rollup is duplicated between "
                "runs_cloud/ and runs_manifest/, so the record count over-counts runs by ~3x. "
                "Both numbers are stated so neither can be quoted misleadingly."
            ),
        },
        "distinct_runs_affected": distinct_runs,
        "caveat_applying_to_every_affected_record": CAVEAT,
        "scope": {
            "false": (
                "The GPU-TRAINING bit-exactness clause of invariant 7, for the enumerated records."
            ),
            "intact_pipeline_clause": (
                "Invariant 7's data-pipeline / frozen-stats / prediction-replay bit-exactness "
                "clause is INTACT and independently evidenced (M4a dataset_hash reproduction, "
                "token-stream content hashing, fixed-chunk rollout replay KATs)."
            ),
            "intact_gate_a": (
                "The Gate-A anchor STANDS. It anchors the CPU eval-harness replay on a frozen M3 "
                "checkpoint and is genuinely bit-identical (results_hash 3f86882a, re-proven with "
                "a byte-identical run manifest); it does not depend on GPU training determinism."
            ),
            "no_result_invalidated": (
                "No prior number changes. What changes is that a run stamping bit_exact_claim: "
                "true may not be quoted as a bit-exact GPU training run."
            ),
            "not_narrower_either": (
                "This is not a documentation nit. The same-seed model divergence measured in "
                "v1.4.5 (frac_negative 0.5662 vs 0.99883 on byte-identical recipes) is the "
                "observable consequence, and it is BLOCKING for the 15-day run until adjudicated."
            ),
        },
        "remediation_status": {
            "code": (
                "bit_exact_preconditions_met + bit_exact_caveat ADDED in v1.4.6; the historical "
                "bit_exact_claim field is deliberately NOT redefined."
            ),
            "posture": (
                "NOT changed by the builder. Forcing deterministic algorithms is a "
                "throughput/cost decision and CLAUDE.md invariant 7 embeds the false sufficiency "
                "premise in its own text; amending it is Lakshay's sign-off. Two candidate "
                "amendments are drafted in prereg s7 v1.4.7."
            ),
        },
        "affected_records": run_records,
        "derived_copies_excluded_from_the_run_count": derived_copies,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    c = out["counts"]
    print(f"determinism records found            : {c['determinism_records_found_total']}")
    n_bad = c["records_claiming_bit_exact_without_forced_determinism"]
    print(f"  claiming bit-exact, determinism off: {n_bad}")
    print(f"  run records                        : {c['of_which_are_run_records']}")
    n_dup = c["of_which_are_derived_copies_in_this_sessions_own_receipts"]
    print(f"  derived copies (own receipts)      : {n_dup}")
    print(f"  DISTINCT training runs affected    : {c['distinct_training_runs_affected']}")
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
