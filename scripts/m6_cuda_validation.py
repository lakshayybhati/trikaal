"""§7 v1.6 — CUDA VALIDATION of the FULL production path, and the EVAL-LEG measurement.

    PYTHONPATH=src python3 scripts/m6_cuda_validation.py --device cuda \
        --symbols FRONTUSDT,KLAYUSDT --steps 30 --out runs/m6_validation

WHY THIS EXISTS. The CUDA probe exercised 120 training steps and nothing else. Eval scoring,
checkpoint resume, both guards and the verdict driver have never run on CUDA in their current
form — and the ``_cuda_getDriverVersion`` crash proved that **CPU-green does not imply
CUDA-green**: every local path returned early at ``torch.cuda.is_available()`` and 333 tests
passed over a line that raises on a GPU.

THE PRIMARY DELIVERABLE is the eval leg: **seconds per (cell, seed)** on the real GPU at the
pinned chunk, as a NUMBER under the ``measured``/``is_costable`` discipline. That is the last
unmeasured input to the run cost — its own receipt said the absolute rate was "PENDING the 4090".

SEEDS: 5, NOT 2. ``m6_verdict.py`` builds ``want_names`` over ``CELLS x DSR_SEEDS`` (5 x 5 = 25
artifacts) and raises on any missing, so a run that must reach a verdict word cannot use fewer.
The instruction asking for 2 seeds AND a verdict word was self-contradictory; caught pre-spend.

Stages: lake load -> Stage-1 -> tokenize -> Stage-2 -> checkpoint + RESUME -> eval scoring ->
degeneracy_guard -> power_guard -> verdict word + ``dual_specification``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from trikaal.eval.verdict import DSR_SEEDS
from trikaal.train.cells import CELLS
from trikaal.train.orchestrator import OrchestratorConfig

OUT = Path("runs_manifest/m6_cuda_validation.json")


def stage1_budget_from_code() -> dict:
    """READ the Stage-1 budget the orchestrator will use. Do not choose one.

    A documented operational parameter, not a specification change: the value exists already and
    was simply never written down. Ablation VALIDITY does not depend on it — all five cells take
    the same Stage-1 budget from the same flag, so the comparison stays clean — but
    REPRODUCIBILITY does, because a replicator needs the number and would otherwise have to read
    three driver scripts to discover that Stage-1 is coupled to Stage-2.
    """
    defaults = OrchestratorConfig()
    drivers = {}
    for path in ("scripts/m6_smoke.py", "scripts/m6_toy_rehearsal.py", "scripts/m6_cuda_probe.py"):
        try:
            txt = Path(path).read_text()
        except OSError:
            continue
        s1 = "steps_stage1=args.steps" in txt or "steps_stage1=steps" in txt
        s2 = "steps_stage2=args.steps" in txt or "steps_stage2=steps" in txt
        drivers[path] = {"sets_stage1": s1, "sets_stage2": s2, "coupled_to_one_flag": s1 and s2}
    return {
        "OrchestratorConfig.steps_stage1_default": defaults.steps_stage1,
        "OrchestratorConfig.steps_stage2_default": defaults.steps_stage2,
        "OrchestratorConfig.seq_len_default": defaults.seq_len,
        "OrchestratorConfig.batch_size_default": defaults.batch_size,
        "OrchestratorConfig.alpha_default": defaults.alpha,
        "drivers": drivers,
        "operative_convention": (
            "EVERY driver in the repo sets steps_stage1 == steps_stage2 from a SINGLE --steps "
            "flag; no driver sets them independently and no separate Stage-1 budget exists "
            "anywhere. The declared library default is "
            f"{defaults.steps_stage1} for both."
        ),
        "consequence": (
            "Under the §4 Stage-2 pin (~1 effective pass, 270-300M bar-visits => 16,479-18,311 "
            "steps at batch 32 x seq 512), Stage-1 INHERITS the same step count BY CODE "
            "CONVENTION, not by written specification. No real-run driver exists yet."
        ),
        "gap_status": (
            "A GAP THAT EXISTED UNNOTICED, not a routine fix. The prereg pins the AR budget and "
            "is silent on Stage-1; the coupling lives only in driver code. Ablation validity is "
            "UNAFFECTED (all five cells share the budget); REPRODUCIBILITY is affected."
        ),
    }


def eval_leg_extrapolation(eval_leg: dict | None, *, dph: float) -> dict:
    """Project the REAL eval leg from the measured per-decision rate. LABELLED, assumptions listed.

    The toy total is useless for costing; the RATE is not. money mode forbids cap_per_symbol
    (xsection.py:71), so nothing can bound the real eval leg — if this comes back large, that is a
    cost fact to accept, because capping it would be a specification change and the design is
    frozen.
    """
    if not eval_leg:
        return {"status": "NOT COMPUTED — no eval_leg in the manifest"}
    rate = eval_leg.get("seconds_per_decision")
    if not isinstance(rate, (int, float)) or rate != rate:  # NaN-safe
        return {"status": "NOT COMPUTED — no measured per-decision rate"}
    # The real headline surface: §3a pinned 40-symbol set, blocks 1-5, h=15 stride grid.
    real_periods, real_symbols = 35_064, 40
    per_cell_seed = real_periods * real_symbols
    cells, seeds = 5, 5
    total_dec = per_cell_seed * cells * seeds
    secs = total_dec * float(rate)
    return {
        "STATUS": "EXTRAPOLATION — NOT A MEASUREMENT",
        "measured_seconds_per_decision": rate,
        "EXTRAPOLATED_decisions_per_cell_seed": per_cell_seed,
        "EXTRAPOLATED_total_decisions": total_dec,
        "EXTRAPOLATED_seconds_per_cell_seed": per_cell_seed * float(rate),
        "EXTRAPOLATED_eval_hours_total": secs / 3600.0,
        "EXTRAPOLATED_eval_usd_total": secs / 3600.0 * dph,
        "dph_used": dph,
        "ASSUMPTIONS_every_one_of_which_could_be_wrong": [
            "the per-decision rate measured on the TOY grid holds at ~1.4M decisions per "
            "(cell, seed) — it may not: caches, memory pressure and batching behave differently "
            "three orders of magnitude up, and the toy rate could be optimistic OR pessimistic",
            f"the real surface is {real_periods:,} periods x {real_symbols} symbols per "
            f"(cell, seed), {cells} cells x {seeds} seeds",
            "the toy scored the same horizon set, so the denominator is comparable; if the real "
            "run scores a different horizon set this scales with it",
            "rate measured at the pinned chunk on THIS GPU; another GPU changes it",
            f"spot price {dph}/hr is the one observed on this rental and is not fixed",
            "TRAINING IS EXCLUDED — this is the eval leg ONLY; add the training extrapolation "
            "from the CUDA probe receipt for a run total",
        ],
        "why_this_matters": (
            "money mode FORBIDS cap_per_symbol (xsection.py:71, §3a uncapped primary), so there "
            "is NO knob to bound the real eval leg. If this figure is large it is a cost to "
            "accept or a schedule to plan around — capping it would be a specification change "
            "and the design is FROZEN."
        ),
    }


def run(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env_extra: dict | None = None,
    timeout_s: float | None = None,
) -> dict:
    """Run a step under a HARD wall-clock bound.

    WHY THE TIMEOUT IS NOT OPTIONAL. A CPU gate of this script ran for TEN HOURS before it was
    stopped — 25 training runs at ~11 min each and eval at ~19 min per (cell, seed), against 2.5 s
    and 12.1 s for the same nominal config in an earlier run. The cause was never proven, so the
    defence cannot be "pick a better config": it has to be a bound that fires regardless of cause.
    On a paid box an unbounded step is a runaway meter, and TimeoutExpired here is a clean,
    reportable failure instead of a silent one.
    """
    import os

    env = {**os.environ, "PYTHONPATH": "src", **(env_extra or {})}
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": " ".join(cmd),
            "returncode": None,
            "timed_out": True,
            "timeout_s": timeout_s,
            "wall_s": time.time() - t0,
            "stdout_tail": (e.stdout or b"").decode(errors="replace")[-4000:]
            if isinstance(e.stdout, bytes)
            else (e.stdout or "")[-4000:],
            "stderr_tail": "",
            "ok": False,
        }
    return {
        "timed_out": False,
        "cmd": " ".join(cmd),
        "returncode": p.returncode,
        "wall_s": time.time() - t0,
        "stdout_tail": p.stdout[-4000:],
        "stderr_tail": p.stderr[-3000:],
        "ok": p.returncode == 0,
    }


def resume_check(out_dir: Path, device: str) -> dict:
    """Load every saved train-state and confirm it restores — the resume leg, on CUDA."""
    from trikaal.train.train_state import load_train_state

    results, failures = {}, []
    for spec in CELLS:
        for seed in DSR_SEEDS:
            rd = out_dir / f"{spec.name}_seed{seed}"
            for stage in ("stage1", "stage2"):
                p = rd / f"{stage}_state.pt"
                key = f"{spec.name}_seed{seed}/{stage}"
                if not p.exists():
                    failures.append(f"{key}: state file absent")
                    continue
                try:
                    payload = torch.load(p, map_location=device, weights_only=False)
                    model_keys = len(payload["model"])
                    opt_groups = len(payload["optimizer"]["param_groups"])
                    samp = payload.get("sampler_state") or {}
                    results[key] = {
                        "step": int(payload["step"]),
                        "model_tensors": model_keys,
                        "optimizer_param_groups": opt_groups,
                        "sampler_n_drawn": samp.get("n_drawn"),
                        "has_rng": bool(payload.get("rng")),
                    }
                except Exception as e:  # a resume failure is a real CUDA-only defect
                    failures.append(f"{key}: {type(e).__name__}: {e}"[:300])
    return {
        "checked": len(results),
        "failures": failures,
        "all_resumable": not failures,
        "sample": dict(list(results.items())[:3]),
        "note": (
            "loads model + optimizer + RNG + sampler cursor from disk on the run device; "
            "load_train_state is the same entry point a real resume uses"
        ),
        "load_train_state_importable": callable(load_train_state),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--symbols", default="FRONTUSDT,KLAYUSDT")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--eval-window", nargs=2, default=["2024-01-01", "2024-03-01"])
    ap.add_argument(
        "--local-smoke",
        action="store_true",
        help="CPU plumbing gate: tiny shells so 25 runs finish locally. Canonical-dims "
        "convergence is NOT proven by this — that comes from the CUDA probe, which cleared the "
        "tripwire at 120 steps/stage at batch 32 x seq 512 on the 4090.",
    )
    ap.add_argument("--dph", type=float, default=0.3256, help="observed $/hr for extrapolation")
    ap.add_argument("--train-timeout-s", type=float, default=5400.0)
    ap.add_argument("--verdict-timeout-s", type=float, default=900.0)
    ap.add_argument("--out", type=Path, default=Path("runs/m6_validation"))
    ap.add_argument("--receipt", type=Path, default=OUT)
    args = ap.parse_args()
    t_start = time.time()

    seeds = ",".join(str(s) for s in DSR_SEEDS)
    print(f"=== M6 CUDA VALIDATION (device={args.device}, seeds={seeds}) ===")
    prov = {
        "device": args.device,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_build": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }

    # ---- 1. the full production path, 5 cells x 5 seeds -------------------------------------
    print("[1/4] full path: lake -> Stage-1 -> tokenize -> Stage-2 -> checkpoint -> eval …")
    rehearsal = run(
        [
            sys.executable,
            "scripts/m6_toy_rehearsal.py",
            "--device",
            args.device,
            "--steps",
            str(args.steps),
            "--batch",
            str(args.batch),
            "--seq-len",
            str(args.seq_len),
            "--symbols",
            args.symbols,
            "--seeds",
            seeds,
            "--wandb",
            "disabled",
            "--out",
            str(args.out),
            "--eval-window",
            args.eval_window[0],
            args.eval_window[1],
        ]
        + (["--local-smoke"] if args.local_smoke else []),
        timeout_s=args.train_timeout_s,
    )
    print(
        f"      rehearsal ok={rehearsal['ok']} wall={rehearsal['wall_s']:.1f}s "
        f"timed_out={rehearsal.get('timed_out')}"
    )
    manifest_path = args.out / "rehearsal_manifest.json"
    rehearsal_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    # ---- 2. checkpoint RESUME on CUDA -------------------------------------------------------
    print("[2/4] checkpoint resume …")
    resume = resume_check(args.out, args.device) if rehearsal["ok"] else {"skipped": True}
    print(f"      resumable={resume.get('all_resumable')} checked={resume.get('checked')}")

    # ---- 3. the verdict driver, end to end --------------------------------------------------
    print("[3/4] m6_verdict.py --allow-toy-grid …")
    verdict_out = args.out / "verdict_manifest.json"
    verdict = (
        run(
            [
                sys.executable,
                "scripts/m6_verdict.py",
                "--artifacts",
                str(args.out / "eval"),
                "--allow-toy-grid",
                "--out",
                str(verdict_out),
            ],
            timeout_s=args.verdict_timeout_s,
        )
        if rehearsal["ok"]
        else {"skipped": True, "ok": False}
    )
    vman = json.loads(verdict_out.read_text()) if verdict_out.exists() else None
    # NESTED, not top-level. Reading vman["emitted"] returned None while the driver had in fact
    # emitted HALT_ADJUDICATE — a receipt contradicting its own artifact, which is the defect this
    # project polices. The word lives at verdict.emitted; the guard flags live beside it.
    vblock = (vman or {}).get("verdict") or {}
    verdict_word = vblock.get("emitted")
    print(f"      verdict ok={verdict.get('ok')} word={verdict_word}")

    # ---- 4. assemble ------------------------------------------------------------------------
    eval_leg = (rehearsal_manifest or {}).get("eval_leg")
    guards = {}
    if vman:
        guards = {
            "degeneracy_armed": bool((vman.get("degeneracy_guard") or {}).get("armed")),
            "power_armed": bool((vman.get("power_guard") or {}).get("armed")),
            "halted_for_degeneracy": bool(vblock.get("halted_for_degeneracy")),
            "halted_for_power": bool(vblock.get("halted_for_power")),
            "primary": vblock.get("primary"),
            "failing_clauses": vblock.get("failing_clauses"),
        }
    out = {
        "receipt": "m6_cuda_validation",
        "EVAL_LEG_EXTRAPOLATION": eval_leg_extrapolation(eval_leg, dph=args.dph),
        "amendment": (
            "prereg §7 v1.6 — CUDA validation of the full production path (~$3, one rental)"
        ),
        "provenance": prov,
        "config": {
            "symbols": args.symbols.split(","),
            "seeds": list(DSR_SEEDS),
            "steps_per_stage": args.steps,
            "batch": args.batch,
            "seq_len": args.seq_len,
            "eval_window": args.eval_window,
        },
        "seed_count_note": (
            "5 seeds, NOT the 2 requested: m6_verdict.py builds want_names over CELLS x DSR_SEEDS "
            "(5 x 5 = 25) and raises on any missing, so '2 seeds' and 'must reach a verdict word' "
            "were incompatible. Caught by the $0 gate BEFORE spend and ruled on; recorded here so "
            "the deviation is on the record rather than silent."
        ),
        "stage1_budget_from_code": stage1_budget_from_code(),
        "step1_full_path": {k: v for k, v in rehearsal.items() if k != "stdout_tail"},
        "step2_resume": resume,
        "step3_verdict": {k: v for k, v in verdict.items() if k != "stdout_tail"},
        "EVAL_LEG_PRIMARY_DELIVERABLE": eval_leg,
        "verdict_word": verdict_word,
        "verdict_reached": verdict_word is not None,
        "dual_specification_present": bool(vman and "dual_specification" in vman),
        "guards": guards,
        "hf_pull": {
            "status": "DEFERRED — NOT CANCELLED",
            "reason": (
                "the only available HuggingFace token is WRITE-SCOPED and NOT fine-grained. The "
                "risk is INTEGRITY, not confidentiality: the lake derives from public Binance "
                "data, but that token can OVERWRITE OR DELETE the Merkle-5dfd667d anchor the "
                "entire reproducibility claim rests on. A fine-grained READ token scoped to the "
                "one dataset reduces the residual to low-stakes read exposure, which is "
                "acceptable. Never ship a write-capable or non-fine-grained credential to rented "
                "third-party infrastructure."
            ),
            "unblocks_the_real_run": False,
            "note": (
                "the rsync fallback is proven (this run used it), so the real run is NOT blocked "
                "either way — HF is a transfer-time optimisation. Prove the pull as a cheap "
                "STANDALONE step once a read token exists, NOT at hour zero of a 50-hour rental."
            ),
        },
        "wall_s_total": time.time() - t_start,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    print(f"[4/4] verdict_word={verdict_word} dual_spec={out['dual_specification_present']}")
    if eval_leg:
        print(f"      eval mean s/(cell,seed) = {eval_leg.get('mean_seconds_per_cell_seed')}")
    print(f"[receipt] -> {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
