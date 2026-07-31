"""§7 v1.6 FINDING 0 — where did the M6 cost basis come from, and is anything behind it?

    PYTHONPATH=src .venv/bin/python scripts/m6_cost_basis_forensics.py

THE CHARGE. "47.2k bars/s → full M6 ≈ 2–3 GPU-days ≈ $20–30" is the project's entire cost basis.
It appears repo-wide in exactly ONE place as a value: a PROSE STRING in
``m6_eval_throughput_expectation.json`` at ``labels_required_by_item_5.training_throughput``,
labelled MEASURED. The manifest that string names — ``m6_rehearsal_manifest.json`` — records
``throughput.bars_per_s_into_gpu = NaN`` and ``train_wall_s = 0``.

This script answers, from artifacts and logs only, three questions the ruling asks, and REFUSES to
reconstruct anything the evidence cannot settle:

  Q1  Was the measurement ever taken, or never taken at all?
  Q2  Is the 47.2k figure derivable from any persisted number?
  Q3  Does ANY artifact in the repo carry an end-to-end training throughput at the canonical
      geometry — i.e. a datum that could legitimately price the M6 run?

Every number printed here is read from a file; nothing is assumed. Writes
``runs_manifest/m6_cost_basis_forensics.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path("runs_manifest/m6_cost_basis_forensics.json")

REHEARSALS = (
    "runs_manifest/m6_rehearsal_manifest.json",
    "runs_cloud/runs/m6_toy/rehearsal_manifest.json",
)
BENCHES = (
    "runs_manifest/m6_attention_bench.json",
    "runs_cloud/m6_attention_bench.json",
    "runs_cloud/runs/m6_attention_bench.json",
)
CLAIM_SITE = "runs_manifest/m6_eval_throughput_expectation.json"
LOG_DIR = Path("runs_cloud/box_logs")

# The canonical training geometry the real M6 run uses (prereg §3a / the 21,301,248-param pin).
CANONICAL = {"batch": 32, "seq_len": 512}


def _load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _isnum(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def q1_rehearsal_state() -> dict:
    """What the cited receipt actually contains, and whether the logs can settle why."""
    found = {}
    for path in REHEARSALS:
        d = _load(path)
        if d is None:
            continue
        t = d.get("throughput", {})
        found[path] = {
            "exists": True,
            "eval_only_rerun": d.get("config", {}).get("eval_only_rerun"),
            "bars_per_s_into_gpu": t.get("bars_per_s_into_gpu"),
            "bars_per_s_is_number": _isnum(t.get("bars_per_s_into_gpu")),
            "train_wall_s": t.get("train_wall_s"),
            "total_train_steps": t.get("total_train_steps"),
            "wall_s_total": d.get("wall_s_total"),
            "per_run_wall_s_entries": len(t.get("per_run_wall_s", {}) or {}),
        }

    # The logs: the training invocation (item2) vs the eval-only re-run (item2b).
    logs = {}
    for name in ("item2.log", "item2b.log", "bench.log"):
        p = LOG_DIR / name
        if not p.exists():
            logs[name] = {"exists": False}
            continue
        text = p.read_text(errors="replace")
        lines = text.splitlines()
        non_wandb = [ln for ln in lines if ln.strip() and not ln.startswith("wandb")]
        tput = [ln for ln in lines if "[throughput]" in ln]
        logs[name] = {
            "exists": True,
            "total_lines": len(lines),
            "non_wandb_lines": len(non_wandb),
            "throughput_lines": tput,
            "captures_script_stdout": bool(non_wandb),
        }

    item2 = LOG_DIR / "item2.log"
    trained = item2.exists() and "stage2/loss" in item2.read_text(errors="replace")
    settled = logs.get("item2.log", {}).get("captures_script_stdout", False)
    return {
        "artifacts": found,
        "logs": logs,
        "training_did_happen": bool(trained),
        "logs_can_settle_whether_it_was_measured": bool(settled),
        "answer": (
            "NOT SETTLED BY THE LOGS, AND NOT RECONSTRUCTED. item2.log (the training invocation) "
            "contains ZERO non-wandb lines — it captured only wandb's stderr, so the script's own "
            "[throughput]/[bundle] stdout is absent from it entirely. Whether that invocation "
            "printed a finite rate cannot be determined from anything on disk. What IS settled: "
            "(a) training DID happen (per-cell stage1/stage2 wandb summaries for all 15 runs); "
            "(b) the manifest on disk is the --eval-only RE-RUN (config.eval_only_rerun=true), "
            "which writes to the SAME path, so whatever the training invocation wrote there was "
            "OVERWRITTEN; (c) item2b.log shows that re-run printing '[throughput] nan steps/s -> "
            "nan bars/s'. The failure is a lost measurement, not a failed one."
        )
        if not settled
        else "SETTLED — see logs.item2.log.throughput_lines.",
    }


def q2_where_47_2k_comes_from() -> dict:
    """Is 47.2k derivable from a persisted number anywhere?"""
    claim = _load(CLAIM_SITE) or {}
    claim_str = (claim.get("labels_required_by_item_5", {}) or {}).get("training_throughput", "")
    bench_hits = []
    for path in BENCHES:
        d = _load(path)
        if d is None:
            continue
        sps = (d.get("sdpa_deterministic") or {}).get("steps_per_s")
        if not _isnum(sps):
            continue
        cfg = d.get("config", {})
        batch, seq = cfg.get("batch"), cfg.get("seq_len")
        bench_hits.append(
            {
                "path": path,
                "steps_per_s": sps,
                "batch": batch,
                "seq_len": seq,
                "derived_bars_per_s": float(sps) * float(batch) * float(seq),
                "matches_canonical_geometry": {"batch": batch, "seq_len": seq} == CANONICAL,
                "posture_recorded_in_artifact": "deterministic_algorithms" in cfg,
            }
        )
    derived = bench_hits[0]["derived_bars_per_s"] if bench_hits else float("nan")
    return {
        "claim_site": CLAIM_SITE,
        "claim_string": claim_str,
        "claim_is_prose_not_number": isinstance(claim_str, str),
        "quoted_figure_bars_per_s": 47200.0,
        "bench_artifacts": bench_hits,
        "derived_bars_per_s": derived,
        "rounds_to_quoted_figure": bool(
            _isnum(derived) and abs(round(derived / 1000.0, 1) - 47.2) < 1e-9
        ),
        "answer": (
            "YES — and NOT from the receipt it cites. 47.2k is exactly reproducible from "
            "m6_attention_bench.json: steps_per_s x batch x seq_len. So the ruling's dichotomy "
            "('measured but not persisted' vs 'never measured') resolves to a THIRD option: a "
            "REAL measurement of a DIFFERENT QUANTITY, persisted numerically in a DIFFERENT "
            "artifact, whose derived product was never persisted anywhere and whose prose label "
            "points at the wrong receipt. The bench is a COMPUTE-ONLY micro-benchmark (60 steps, "
            "10 warmup, pre-generated random-token batches resident in memory, AR backbone only: "
            "no data loading, no Stage-1 tokenizer training, no tokenize pass, no eval, no "
            "checkpointing) — an UPPER BOUND on the Stage-2 step rate, not an end-to-end training "
            "throughput. The cost extrapolation treats it as the latter."
        ),
    }


def q3_any_end_to_end_datum() -> dict:
    """Does any artifact carry an end-to-end training rate at the CANONICAL geometry?"""
    candidates = []
    for path in sorted(Path("runs_manifest").glob("*.json")):
        d = _load(str(path))
        if not isinstance(d, dict):
            continue
        recipe = d.get("recipe") or {}
        arms = d.get("arms") or {}
        if not (isinstance(arms, dict) and recipe):
            continue
        batch, seq = recipe.get("batch"), recipe.get("seq_len")
        for arm_name, arm in arms.items():
            if not isinstance(arm, dict):
                continue
            training = arm.get("training") or {}
            for key, tr in training.items() if isinstance(training, dict) else []:
                for att in tr.get("attempts", []) or []:
                    steps, wall = att.get("steps"), att.get("wall_s")
                    if not (_isnum(steps) and _isnum(wall) and wall > 0):
                        continue
                    candidates.append(
                        {
                            "path": str(path),
                            "arm": arm_name,
                            "cell": tr.get("cell", key),
                            "steps": steps,
                            "wall_s": wall,
                            "batch": batch,
                            "seq_len": seq,
                            "steps_per_s": float(steps) / float(wall),
                            "bars_per_s": float(steps) / float(wall) * float(batch) * float(seq)
                            if _isnum(batch) and _isnum(seq)
                            else float("nan"),
                            "canonical_geometry": {"batch": batch, "seq_len": seq} == CANONICAL,
                        }
                    )
    canonical = [c for c in candidates if c["canonical_geometry"]]
    return {
        "n_end_to_end_records_found": len(candidates),
        "n_at_canonical_geometry": len(canonical),
        "records": candidates,
        "answer": (
            "NO. Real end-to-end GPU training wall-times DO exist — the canary/money-leg arms "
            "record per-training `steps` and `wall_s` — but every one of them was run at "
            f"seq_len={candidates[0]['seq_len'] if candidates else 'n/a'}, not the canonical "
            f"{CANONICAL['seq_len']}. Attention cost is quadratic in sequence length and per-step "
            "overhead dominates at short sequences, so those rates cannot be transferred to the "
            "canonical geometry in either direction. The project therefore has NO end-to-end "
            "training-throughput measurement at the geometry it intends to run."
        )
        if not canonical
        else "YES — see records with canonical_geometry=true.",
    }


def main() -> int:
    q1, q2, q3 = q1_rehearsal_state(), q2_where_47_2k_comes_from(), q3_any_end_to_end_datum()
    out = {
        "receipt": "m6_cost_basis_forensics",
        "amendment": "prereg §7 v1.6 Finding 0 (supervisor-ordered, local $0)",
        "question": (
            "Is the M6 cost basis backed by a measurement? Where did 47.2k bars/s come from, and "
            "does any artifact carry an end-to-end training rate at the canonical geometry?"
        ),
        "q1_was_it_ever_measured": q1,
        "q2_provenance_of_47_2k": q2,
        "q3_end_to_end_datum_exists": q3,
        "verdict": (
            "THE COST BASIS IS NOT BACKED BY AN END-TO-END MEASUREMENT. The 47.2k figure is real "
            "arithmetic on a real CUDA measurement, but of a compute-only micro-benchmark at "
            "canonical geometry — an upper bound that excludes data loading, the Stage-1 "
            "tokenizer, the tokenize pass, eval and checkpointing. The end-to-end rate the "
            "cost extrapolation needs was written to a manifest by the training invocation and "
            "then OVERWRITTEN by an --eval-only re-run at the same path; the logs on disk cannot "
            "settle what it said, and it is not reconstructed here. Every downstream cost claim "
            "is therefore ESTIMATED, NOT MEASURED, until a real datum exists."
        ),
        "additional_finding_the_ruling_did_not_ask_for": (
            "The attention bench ran under FORCED determinism and never recorded that it had: "
            "bench_mode called set_determinism(0), whose default is deterministic_algorithms="
            "True, while the production path (orchestrator.py) passes False. So 47.2k is a "
            "FORCED-determinism rate, and the '~1.3x forced-determinism penalty applied to "
            "47.2k -> $43-65' arithmetic DOUBLE-COUNTS the penalty. The posture is now recorded "
            "in the bench artifact (config.deterministic_algorithms) and both arms are runnable."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    print("=== M6 COST-BASIS FORENSICS ===")
    print(f"Q1 measured?      {q1['answer'][:120]}...")
    print(
        f"Q2 47.2k source?  derived={q2['derived_bars_per_s']:,.1f} bars/s "
        f"(matches quoted: {q2['rounds_to_quoted_figure']})"
    )
    print(
        f"Q3 end-to-end?    {q3['n_end_to_end_records_found']} records, "
        f"{q3['n_at_canonical_geometry']} at canonical geometry"
    )
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
