"""§7 v1.6 — THE STAGED CUDA PROBE. Two jobs, one rental (supervisor-authorized, ~$1-2).

    PYTHONPATH=src python3 scripts/m6_cuda_probe.py --device cuda [--steps 60] [--cell-steps 120]

JOB 1 — DETERMINISM PENALTY. The same canonical training config under two determinism postures,
each in its OWN SUBPROCESS (forcing is process-global and irreversible in practice; the CPU
rehearsal showed a sequential in-process pair is confounded by warm-up ordering, so the arms must
not share a process). Each arm's record carries ``determinism_record()``, which reads the LIVE
torch flags rather than our intent — so the arm is self-identifying and the double-counting
question is closed by the artifact, not by argument.

JOB 2 — REAL END-TO-END THROUGHPUT. One (cell, seed) through the REAL orchestrator path at
canonical geometry — lake load → Stage-1 tokenizer → tokenize pass → Stage-2 AR → checkpoint —
timed per stage, with the fixed overheads (lake load, window build, tokenize, checkpoint) reported
SEPARATELY. A short probe cannot complete a full cycle, so anything projected to full-run cost is
emitted under ``extrapolation`` with its assumptions listed, and is labelled an extrapolation in
the field name itself. An unlabelled projection is the exact defect Finding 0 was about.

CONTROL-ARM RULE. If the UNFORCED arm fails, this emits ``PROBE INVALID`` and no determinism
verdict. A forced-arm failure is only interpretable against a baseline that completes.
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

from trikaal.data.universe_loader import build_symbol_windows, calendar_boundary_ms, connect_lake
from trikaal.train.arms import ARMS
from trikaal.train.cells import CELLS
from trikaal.train.orchestrator import OrchestratorConfig, run_cell
from trikaal.utils.throughput import BASIS_END_TO_END, is_costable, throughput_record

OUT = Path("runs_manifest/m6_cuda_probe.json")
LAKE = Path("processed/universe_bars")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "KLAYUSDT", "FRONTUSDT")
TRAIN_WINDOW = ("2021-01-01", "2025-01-01")
TRAIN_FRAC = 0.7
BENCH_OUT = Path("runs/m6_attention_bench.json")


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"<unavailable: {e}>"


def provenance(device: str) -> dict:
    """GPU model, driver, CUDA build, cuDNN, torch/numpy — the gap closed at v1.6 item 3."""
    rec = {
        "device": device,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_build": getattr(torch.version, "cuda", None),
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "nvidia_smi_header": "\n".join(_sh(["nvidia-smi"]).splitlines()[:12]),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rec |= {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": f"{p.major}.{p.minor}",
            "gpu_total_memory_bytes": int(p.total_memory),
            "gpu_count": torch.cuda.device_count(),
        }
    return rec


# --------------------------------------------------------------------------------------------
# JOB 1


def bench_arm(
    *, forced: bool, device: str, steps: int, warmup: int, batch: int, seq_len: int
) -> dict:
    """One bench arm in a FRESH SUBPROCESS. Returns the parsed artifact, or an error record."""
    flag = "--forced-determinism" if forced else "--no-forced-determinism"
    cmd = [
        sys.executable,
        "scripts/m6_attention_bench.py",
        "--device",
        device,
        "--steps",
        str(steps),
        "--batch",
        str(batch),
        "--seq-len",
        str(seq_len),
        # FORWARDED. The $0 dry-run gate caught this missing: the probe accepted --warmup and
        # dropped it, so the bench used its own default (10) against 4 steps -> timed_steps = -6,
        # t_timed = 0, steps_per_s = NaN. It would have "worked" at the real settings only because
        # the defaults coincide -- a silent coupling, not a correct one.
        "--warmup",
        str(warmup),
        flag,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**_env(), "PYTHONPATH": "src"})
    wall = time.time() - t0
    if proc.returncode != 0 or not BENCH_OUT.exists():
        return {
            "arm": flag,
            "completed": False,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-1500:],
            "wall_s": wall,
        }
    art = json.loads(BENCH_OUT.read_text())
    sd = art["sdpa_deterministic"]
    return {
        "arm": flag,
        "completed": True,
        "requested_forced": forced,
        # THE FIELD THAT CLOSES THE DOUBLE-COUNTING QUESTION: read from live torch flags.
        "actual_deterministic_algorithms": sd["determinism_record"]["deterministic_algorithms"],
        "determinism_record": sd["determinism_record"],
        "steps_per_s": sd["steps_per_s"],
        "throughput": sd["throughput"],
        "final_loss": sd["final_loss"],
        "stable": sd["stable"],
        "subprocess_wall_s": wall,
    }


def _env() -> dict:
    import os

    return dict(os.environ)


# --------------------------------------------------------------------------------------------
# JOB 2


def end_to_end_cell(*, device: str, steps: int, batch: int, seq_len: int, out: Path) -> dict:
    """ONE (cell, seed) through the REAL orchestrator, with fixed overheads timed separately."""
    # sys.path[0] is scripts/ when this file is run directly, so the repo root is NOT importable
    # and a bare `from scripts.… import` fails. Reuse the rehearsal's own loader by adding the
    # repo root explicitly — caught by the $0 dry-run gate, which is what that gate is for.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.m6_toy_rehearsal import load_symbol  # the rehearsal's own loader

    # Use only the symbols actually present. The probe pushes a lake SLICE over the operator's
    # home uplink, so a partial slice is a real possibility — and silently training on whatever
    # happened to arrive, without saying so, would make lake_load_s (a REPORTED overhead) describe
    # an unstated configuration. The count is recorded and travels with the extrapolation.
    present = tuple(s for s in SYMBOLS if (LAKE / f"symbol={s}").is_dir())
    if not present:
        raise RuntimeError(f"no lake symbols present under {LAKE} — cannot measure end-to-end")

    overheads: dict[str, float] = {}
    t = time.time()
    boundary = calendar_boundary_ms(*TRAIN_WINDOW, TRAIN_FRAC)
    con = connect_lake(LAKE)
    raw = {s: load_symbol(con, s, boundary) for s in present}
    overheads["lake_load_s"] = time.time() - t
    bars_loaded = int(sum(d["x"].shape[0] for d in raw.values()))

    t = time.time()
    per_arm = {
        arm: [
            build_symbol_windows(
                s,
                d["x"],
                d["mask"],
                d["segment_id"],
                d["ts"],
                arm=arm,
                seq_len=seq_len,
                boundary_ms=boundary,
                seed=0,
            )
            for s, d in raw.items()
        ]
        for arm in ARMS
    }
    overheads["window_build_s"] = time.time() - t

    spec = CELLS[3]  # cell4 fsq_micro — the treatment cell, the most expensive arm
    cfg = OrchestratorConfig(
        money_run=False,  # §7 v1.6 C-6: a throughput probe, one seed, declared
        seeds=(0,),
        seq_len=seq_len,
        batch_size=batch,
        micro_legibility_min=None,  # probe: the standing gate binds the REAL run, not this
        steps_stage1=steps,
        steps_stage2=steps,
        tok_kwargs={"max_len": max(512, seq_len)},
        backbone_kwargs={"max_len": seq_len + 64},
        expect_backbone_params=21_301_248,
        enforce_parity=True,
        device=device,
        autocast_bf16=device.startswith("cuda"),
        out_dir=out,
        state_every=max(50, steps // 4),
        wandb_mode="disabled",
        wandb_group="m6_cuda_probe",
    )
    t = time.time()
    man = run_cell(spec, 0, per_arm, cfg)
    total_cell_wall = time.time() - t

    tp = man["throughput"]
    s1, s2 = tp["stage1"], tp["stage2"]
    train_wall_s = float(s1["wall_s"]) + float(s2["wall_s"])
    total_steps = int(s1["steps"]) + int(s2["steps"])
    pooled = throughput_record(
        steps=total_steps,
        batch=batch,
        seq_len=seq_len,
        wall_s=train_wall_s,
        basis=BASIS_END_TO_END,
        device=device,
        note="both stages pooled, wall-clock — the JOB-2 headline rate",
    )
    # everything in the cell that is NOT the two training loops (tokenize pass, checkpoints, …)
    overheads["tokenize_and_checkpoint_s"] = max(0.0, total_cell_wall - train_wall_s)
    return {
        "cell": spec.name,
        "seed": 0,
        "symbols_used": list(present),
        "n_symbols_used": len(present),
        "symbols_requested": list(SYMBOLS),
        "bars_loaded": bars_loaded,
        "per_stage": {"stage1": s1, "stage2": s2},
        "pooled_training": pooled,
        # the two field NAMES the ruling asked for, as NUMBERS
        "bars_per_s_into_gpu": pooled["bars_per_s"],
        "train_wall_s": train_wall_s,
        "measured": pooled["measured"],
        "is_costable": is_costable(pooled),
        "total_cell_wall_s": total_cell_wall,
        "fixed_overheads_s": overheads,
        "determinism_record": man["determinism"],
        "params_gate": "21,301,248 enforced by OrchestratorConfig.expect_backbone_params",
    }


def full_run_extrapolation(job2: dict, job1: dict, *, dph: float) -> dict:
    """Project the M6 run cost from the probe — LABELLED AS AN EXTRAPOLATION, assumptions listed.

    An honest labelled extrapolation is acceptable; an unlabelled one is Finding 0. Every field
    name here says ``EXTRAPOLATED``, the assumptions are enumerated beside the number, and the
    whole block refuses to produce a figure if the measurement it would rest on is absent.
    """
    if not job2.get("is_costable"):
        return {
            "status": "NOT COMPUTED — no costable end-to-end measurement",
            "why": "is_costable() was False, so there is nothing to extrapolate from",
        }
    rate = float(job2["bars_per_s_into_gpu"])  # end_to_end_training basis, both stages pooled
    # The v1.5 run: 5 cells x 5 seeds, both stages, at the pinned budget.
    cells, seeds = 5, 5
    steps_per_stage = 20_000  # §3a canonical training budget per stage
    batch, seq = (
        int(job2["per_stage"]["stage2"]["batch"]),
        int(job2["per_stage"]["stage2"]["seq_len"]),
    )
    bar_visits = cells * seeds * 2 * steps_per_stage * batch * seq
    train_h = bar_visits / rate / 3600.0
    overheads = job2.get("fixed_overheads_s", {})
    per_cell_overhead_s = sum(float(v) for v in overheads.values())
    overhead_h = per_cell_overhead_s * cells * seeds / 3600.0
    pen = job1.get("forced_over_unforced_ratio")
    return {
        "STATUS": "EXTRAPOLATION — NOT A MEASUREMENT",
        "EXTRAPOLATED_train_hours": train_h,
        "EXTRAPOLATED_overhead_hours": overhead_h,
        "EXTRAPOLATED_total_hours": train_h + overhead_h,
        "EXTRAPOLATED_usd_at_observed_dph": (train_h + overhead_h) * dph,
        "EXTRAPOLATED_usd_if_determinism_forced": (
            (train_h / pen + overhead_h) * dph if pen else None
        ),
        "dph_used": dph,
        "ASSUMPTIONS_every_one_of_which_could_be_wrong": [
            f"the measured end-to-end rate ({rate:,.0f} bars/s) holds for the WHOLE run — it was "
            f"measured over {job2['per_stage']['stage2']['steps']} steps/stage on ONE cell "
            f"({job2['cell']}) and {job2.get('n_symbols_used')} symbols, not 5 cells x 5 seeds "
            "over 40 symbols",
            f"{cells} cells x {seeds} seeds x 2 stages x {steps_per_stage:,} steps at batch "
            f"{batch} x seq {seq}; if the pinned step budget differs, this scales linearly",
            f"fixed overheads ({per_cell_overhead_s:.0f}s measured for this cell) recur ONCE PER "
            "(cell, seed) and do NOT scale with symbol count — the real run loads 40 symbols, not "
            f"{job2.get('n_symbols_used')}, so the lake-load term is UNDERSTATED here",
            "EVAL IS EXCLUDED ENTIRELY — this probe measured no scoring pass; the eval leg was "
            "previously observed at ~40 min/(cell,seed) on a 4090, which would DOMINATE this "
            "figure. Treat the number below as a TRAINING-ONLY floor, not a run cost",
            f"the spot price {dph}/hr is the one observed on this rental and is not fixed",
            "the determinism variant applies JOB 1's compute-only ratio to an end-to-end rate; "
            "the two need not share a penalty",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=60, help="JOB-1 bench steps per arm")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--cell-steps", type=int, default=120, help="JOB-2 steps per stage")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("runs/m6_cuda_probe"))
    # A CPU dry-run must NOT be able to leave a file that reads like the real CUDA receipt.
    ap.add_argument("--receipt", type=Path, default=OUT)
    ap.add_argument("--dph", type=float, default=0.3256, help="observed $/hr for the extrapolation")
    args = ap.parse_args()
    t_start = time.time()

    prov = provenance(args.device)
    print(f"=== M6 CUDA PROBE — {prov.get('gpu_name', args.device)} ===")

    # ---- JOB 1 -----------------------------------------------------------------------------
    print("[job1] arm A: UNFORCED (the control arm) …")
    unforced = bench_arm(
        forced=False,
        device=args.device,
        steps=args.steps,
        warmup=args.warmup,
        batch=args.batch,
        seq_len=args.seq_len,
    )
    print(f"       completed={unforced['completed']} steps/s={unforced.get('steps_per_s')}")
    print("[job1] arm B: FORCED …")
    forced = bench_arm(
        forced=True,
        device=args.device,
        steps=args.steps,
        warmup=args.warmup,
        batch=args.batch,
        seq_len=args.seq_len,
    )
    print(f"       completed={forced['completed']} steps/s={forced.get('steps_per_s')}")

    # "completed" is NOT enough: an arm can exit 0 and still produce no rate (see the warmup
    # defect above). A ratio computed from NaN is a check that cannot fail, so require a real
    # measurement from BOTH arms before any verdict is emitted.
    def _measured(a: dict) -> bool:
        return bool(a.get("completed")) and bool((a.get("throughput") or {}).get("measured"))

    probe_valid = _measured(unforced)
    job1: dict = {"unforced": unforced, "forced": forced, "probe_valid": probe_valid}
    if not probe_valid:
        job1["verdict"] = (
            "PROBE INVALID — the UNFORCED control arm did not complete, so nothing here is a "
            "determinism finding. A forced-arm result is only interpretable against a baseline "
            "that completes."
        )
    elif not _measured(forced):
        job1["verdict"] = (
            "FORCED ARM FAILED while the control completed — report and STOP. Do not read a "
            "throughput penalty from a failure; the error text names the op."
        )
    else:
        u, f = (
            float(unforced["throughput"]["bars_per_s"]),
            float(forced["throughput"]["bars_per_s"]),
        )
        job1 |= {
            "unforced_bars_per_s": u,
            "forced_bars_per_s": f,
            "penalty_pct": (u - f) / u * 100.0,
            "forced_over_unforced_ratio": f / u,
            "postures_actually_differed": (
                unforced["actual_deterministic_algorithms"]
                != forced["actual_deterministic_algorithms"]
            ),
        }
        job1["verdict"] = (
            f"forced/unforced = {f / u:.4f} ({(u - f) / u * 100.0:+.2f}% throughput change under "
            "forced determinism), both arms at canonical geometry, each in its own process, "
            "posture read from live torch flags."
        )
        if not job1["postures_actually_differed"]:
            job1["verdict"] = (
                "PROBE INVALID FOR JOB 1 — both arms recorded the SAME actual determinism state, "
                "so the flag did not take effect and no penalty can be attributed. "
                + job1["verdict"]
            )

    # ---- JOB 2 -----------------------------------------------------------------------------
    job2: dict
    try:
        print("[job2] end-to-end cell at canonical geometry …")
        job2 = end_to_end_cell(
            device=args.device,
            steps=args.cell_steps,
            batch=args.batch,
            seq_len=args.seq_len,
            out=args.out,
        )
        print(
            f"       bars_per_s_into_gpu={job2['bars_per_s_into_gpu']:,.1f} "
            f"train_wall_s={job2['train_wall_s']:.1f} costable={job2['is_costable']}"
        )
    except Exception as e:  # a JOB-2 failure must not silently become a JOB-1-only report
        import traceback

        job2 = {
            "completed": False,
            "error": f"{type(e).__name__}: {e}"[:1500],
            "traceback_tail": traceback.format_exc()[-1500:],
        }
        print(f"       JOB 2 FAILED: {job2['error'][:200]}")

    extrapolation = full_run_extrapolation(job2, job1, dph=args.dph)

    out = {
        "receipt": "m6_cuda_probe",
        "amendment": "prereg §7 v1.6 — the staged CUDA probe (supervisor-authorized, ~$1-2)",
        "EXTRAPOLATION_full_run_cost": extrapolation,
        "provenance": prov,
        "config": {
            "bench_steps": args.steps,
            "bench_warmup": args.warmup,
            "cell_steps_per_stage": args.cell_steps,
            "batch": args.batch,
            "seq_len": args.seq_len,
            "canonical_geometry": {"batch": args.batch, "seq_len": args.seq_len}
            == {"batch": 32, "seq_len": 512},
        },
        "job1_determinism_penalty": job1,
        "job2_end_to_end_throughput": job2,
        "wall_s_total": time.time() - t_start,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    print(f"[receipt] -> {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
