"""The §3a attention-mode decision — flash2 vs sdpa_deterministic, matched segment (Item 2).

    PYTHONPATH=src python3 scripts/m6_attention_bench.py --device cuda [--steps 60]

Prereg §3a: ONE attention mode produces all 15 real runs and the headline; it is fixed at the
toy-CUDA rehearsal, BEFORE any real cell trains, and logged in §7. This bench produces that
decision's evidence on the actual GPU type: the same canonical backbone (identical init — one
seed before construction), the same pre-generated token batches, trained S steps under each
mode with bf16 autocast (the real-run training config), measuring

* throughput (steps/s, warmup excluded) → the speed case for flash2;
* stability (every loss/grad finite) → the safety case;
* loss agreement (mean/max |Δloss| across matched steps) → the numerics case (bf16 kernels
  differ in rounding; agreement is expected to be approximate, never bit-exact).

Recommendation rule (mechanical, stated up front): flash2 iff available AND stable AND
≥ 15 % faster AND the final-loss gap is within 5 % of the loss scale — otherwise
sdpa_deterministic (whose CUDA runs additionally carry the bit-exact-resume claim, inv. 7).
EVAL is unaffected either way: ``predict_mu`` runs fp32 through the default SDPA path, so the
decision covers TRAINING kernels only. Writes ``runs/m6_attention_bench.json``; the §7 prereg
entry is appended from that artifact (pre-authorized in the rehearsal instruction).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from trikaal.model.attention_mode import (
    MODE_FLASH2,
    MODE_SDPA,
    determinism_record,
    flash2_available,
    set_attention_backend,
)
from trikaal.train.cells import build_cell_backbone
from trikaal.train.train_state import autocast_ctx
from trikaal.utils.seeding import set_determinism
from trikaal.utils.throughput import BASIS_COMPUTE_ONLY, throughput_record, unmeasured_record

OUT = Path("runs/m6_attention_bench.json")
V_C, V_F = 891, 1225  # canonical FSQ vocab (the 21,301,248-param pin)


def bench_mode(
    mode: str,
    batches: list,
    *,
    device: str,
    seq_len: int,
    warmup: int,
    forced_determinism: bool = True,
) -> dict:
    set_determinism(0, deterministic_algorithms=forced_determinism)
    batch = int(batches[0][0].shape[0]) if batches else 0
    model = build_cell_backbone(V_C, V_F, max_len=seq_len + 64).to(device)
    set_attention_backend(model, mode)
    # v1.6 JOB 1: each arm SELF-IDENTIFIES its determinism posture from the live torch flags
    # (determinism_record reads torch.are_deterministic_algorithms_enabled(), not our intent), and
    # carries GPU/driver/CUDA/cuDNN/torch/numpy provenance. This is what closes the
    # double-counting question by ARTIFACT rather than by argument: the 47.2k bench carried no
    # posture field at all, so nothing on disk said whether the penalty was already in the number.
    det = determinism_record(seed=0, device=device, attention_mode=mode)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    losses: list[float] = []
    t_timed = 0.0
    for i, (bc, bf, ts) in enumerate(batches):
        t0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, bf16=device.startswith("cuda")):
            loss = model.compute_loss(bc.to(device), bf.to(device), ts.to(device))["loss"]
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if i >= warmup:
            t_timed += time.time() - t0
        losses.append(float(loss.detach()))
        if not (np.isfinite(losses[-1]) and np.isfinite(float(gn))):
            return {
                "mode": mode,
                "stable": False,
                "losses": losses,
                "failed_at_step": i + 1,
                "throughput": unmeasured_record(
                    reason="bench aborted on non-finite loss/grad — no rate is claimed",
                    device=device,
                ),
                "determinism_record": det,
                "forced_determinism_requested": bool(forced_determinism),
            }
    timed_steps = len(batches) - warmup
    return {
        "mode": mode,
        "stable": True,
        "losses": losses,
        "steps_per_s": timed_steps / t_timed if t_timed > 0 else float("nan"),
        "final_loss": losses[-1],
        # Finding 0: this bench is where the 47.2k bars/s figure actually came from, but it was
        # never persisted as a number — only the steps/s was, and the product lived in prose. The
        # basis is COMPUTE-ONLY (batches are pre-generated and resident; no loader, no tokenizer
        # stage, no eval, no checkpointing), so it is an UPPER BOUND and may not price a run.
        "throughput": throughput_record(
            steps=timed_steps,
            batch=batch,
            seq_len=seq_len,
            wall_s=t_timed,
            basis=BASIS_COMPUTE_ONLY,
            device=device,
            note="AR training step on pre-resident random-token batches; warmup excluded",
        ),
        "determinism_record": det,
        "forced_determinism_requested": bool(forced_determinism),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    # The determinism POSTURE of this bench was never recorded in its own artifact, and the
    # default has always been FORCED (set_determinism's own default is deterministic_algorithms
    # =True) — while the production path, orchestrator.py, runs UNFORCED. Both flags exist so the
    # staged CUDA probe can measure the pair in one rental; the default is unchanged (forced).
    ap.add_argument("--forced-determinism", dest="forced", action="store_true", default=True)
    ap.add_argument("--no-forced-determinism", dest="forced", action="store_false")
    args = ap.parse_args()

    # one fixed, pre-generated batch stream — both modes consume the identical bytes
    rng = np.random.default_rng(20260704)
    batches = []
    for _ in range(args.steps):
        bc = torch.from_numpy(rng.integers(0, V_C, (args.batch, args.seq_len)).astype(np.int64))
        bf = torch.from_numpy(rng.integers(0, V_F, (args.batch, args.seq_len)).astype(np.int64))
        t0 = 1_600_000_000_000 + int(rng.integers(0, 10_000)) * 60_000
        ts = torch.from_numpy(
            (t0 + np.arange(args.seq_len) * 60_000)[None, :].repeat(args.batch, 0).astype(np.int64)
        )
        batches.append((bc, bf, ts))

    print(f"=== M6 attention-mode bench (device={args.device}, {args.steps} steps) ===")
    sdpa = bench_mode(
        MODE_SDPA,
        batches,
        device=args.device,
        seq_len=args.seq_len,
        warmup=args.warmup,
        forced_determinism=args.forced,
    )
    print(f"[sdpa]   stable={sdpa['stable']} steps/s={sdpa.get('steps_per_s', float('nan')):.3f}")

    flash = None
    if flash2_available():
        flash = bench_mode(
            MODE_FLASH2,
            batches,
            device=args.device,
            seq_len=args.seq_len,
            warmup=args.warmup,
            forced_determinism=args.forced,
        )
        print(
            f"[flash2] stable={flash['stable']} "
            f"steps/s={flash.get('steps_per_s', float('nan')):.3f}"
        )
    else:
        print("[flash2] UNAVAILABLE on this device — the decision requires the CUDA box")

    agreement = None
    recommend = MODE_SDPA
    reason = "flash2 unavailable — deterministic SDPA is the only candidate"
    if flash is not None and flash["stable"] and sdpa["stable"]:
        la = np.array(sdpa["losses"]), np.array(flash["losses"])
        gaps = np.abs(la[0] - la[1])
        scale = float(np.mean(np.abs(la[0]))) or 1.0
        agreement = {
            "mean_abs_gap": float(gaps.mean()),
            "max_abs_gap": float(gaps.max()),
            "final_gap_frac_of_scale": float(gaps[-1] / scale),
        }
        speedup = flash["steps_per_s"] / sdpa["steps_per_s"]
        if speedup >= 1.15 and agreement["final_gap_frac_of_scale"] <= 0.05:
            recommend = MODE_FLASH2
            reason = (
                f"stable, {speedup:.2f}x faster, final-loss gap "
                f"{agreement['final_gap_frac_of_scale']:.3%} of scale (<= 5%)"
            )
        else:
            reason = (
                f"speedup {speedup:.2f}x (< 1.15x) or loss gap too large — the bit-exact "
                "sdpa fallback wins by default"
            )
    elif flash is not None and not flash["stable"]:
        reason = "flash2 UNSTABLE on this box — deterministic SDPA mandatory"

    result = {
        "bench": "m6_attention_mode_s3a",
        "device": args.device,
        "config": {
            "steps": args.steps,
            "warmup": args.warmup,
            "batch": args.batch,
            "seq_len": args.seq_len,
            "bf16_autocast": args.device.startswith("cuda"),
            # RECORDED FROM v1.6 ON: the posture this bench ran under. It was absent from every
            # prior artifact, and the default has always been FORCED — so the 47.2k figure is a
            # FORCED-determinism rate, while production (orchestrator.py) runs UNFORCED.
            "deterministic_algorithms": bool(args.forced),
        },
        "sdpa_deterministic": {k: v for k, v in sdpa.items() if k != "losses"},
        "flash2": ({k: v for k, v in flash.items() if k != "losses"} if flash else None),
        "loss_agreement": agreement,
        "losses": {"sdpa": sdpa["losses"], "flash2": flash["losses"] if flash else None},
        "recommendation": recommend,
        "reason": reason,
        "note": "eval (predict_mu) always runs fp32 deterministic-SDPA — this decision covers "
        "TRAINING kernels only; the §7 prereg entry is appended from this artifact.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[decision] recommend={recommend} — {reason}")
    print(f"[artifact] → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
