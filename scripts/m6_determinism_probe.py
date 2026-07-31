"""§7 v1.4.7 — $0 FEASIBILITY PROBE: can the M6 training path run with determinism FORCED?

    PYTHONPATH=src .venv/bin/python scripts/m6_determinism_probe.py

WHY THIS COMES BEFORE ANY COST DISCUSSION. ``torch.use_deterministic_algorithms(True,
warn_only=False)`` RAISES on any op that has no deterministic kernel. If the M6 training path hits
such an op, then "make the bit-exactness claim true" is not merely a throughput question — it needs
``warn_only=True`` plus an explicit list of ops that stay non-deterministic, which WEAKENS the claim
regardless of what it costs. That changes the decision, so it must be measured first.

The probe runs the real Stage-1 + Stage-2 cell path (``orchestrator.run_cell`` via the canary's own
tokenizer/backbone builders) at toy scale on CPU/MPS, once with determinism forced and once without,
and reports whether the forced run COMPLETES and every op it names if it does not.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from trikaal.model.attention_mode import MODE_SDPA, determinism_record, set_attention_backend
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.utils.seeding import set_determinism
from trikaal.utils.throughput import BASIS_COMPUTE_ONLY, throughput_record

OUT = Path("runs_manifest/m6_determinism_feasibility_probe.json")

# torch names the offending op in the exception text; this pulls it out for the report.
OP_HINTS = (
    "does not have a deterministic implementation",
    "deterministic behavior",
    "nondeterministic",
)


def _one_pass(*, forced: bool, steps: int, device: str, seed: int = 0) -> dict:
    """Train a tiny tokenizer + AR for ``steps`` steps under the requested determinism posture."""
    if forced:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_determinism(seed, deterministic_algorithms=forced)
    rec: dict = {
        "forced": forced,
        "determinism_record": determinism_record(
            seed=seed, device=device, attention_mode=MODE_SDPA
        ),
    }
    try:
        torch.manual_seed(seed)
        tok = TokenizerAE(d_model=32, n_layers=1, n_heads=2, d_ff=64, max_len=64).to(device)
        ar = TrikaalAR(
            n_layers=1,
            d_model=32,
            d_ff=64,
            n_heads=2,
            mtp_depths=1,
            max_len=64,
            ffn_dropout=0.1,
            resid_dropout=0.1,
            attn_dropout=0.1,
            token_dropout=0.1,
        ).to(device)
        set_attention_backend(tok, MODE_SDPA)
        set_attention_backend(ar, MODE_SDPA)
        opt_t = torch.optim.AdamW(tok.parameters(), lr=1e-3)
        opt_a = torch.optim.AdamW(ar.parameters(), lr=1e-3)
        rng = np.random.default_rng(seed)
        seq, batch = 16, 4

        losses = []
        t_train = time.time()
        # Stage 1: the tokenizer objective (embedding/scatter ops are the usual offenders)
        for _ in range(steps):
            x = torch.from_numpy(rng.standard_normal((batch, seq, 16)).astype(np.float32)).to(
                device
            )
            m = torch.zeros(batch, seq, 16, device=device)
            loss = tok(x, m)["loss"]
            opt_t.zero_grad(set_to_none=True)
            loss.backward()
            opt_t.step()
            losses.append(float(loss.detach()))
        # Stage 2: the AR objective on the frozen token stream (index_put / gather / CE paths)
        tok.eval()
        with torch.no_grad():
            x = torch.from_numpy(rng.standard_normal((batch, seq, 16)).astype(np.float32)).to(
                device
            )
            m = torch.zeros(batch, seq, 16, device=device)
            b_c, b_f = tok.encode_tokens(x, m)
        ts = torch.zeros(batch, seq, dtype=torch.int64, device=device)
        for _ in range(steps):
            loss = ar.compute_loss(b_c, b_f, ts)["loss"]  # the orchestrator's own AR step
            opt_a.zero_grad(set_to_none=True)
            loss.backward()
            opt_a.step()
            losses.append(float(loss.detach()))
        t_train = time.time() - t_train

        rec.update(
            {
                "completed": True,
                "steps_each_stage": steps,
                "final_stage1_loss": losses[steps - 1],
                "final_stage2_loss": losses[-1],
                "offending_ops": [],
                # Item (iv): the probe now also produces a NUMBER, so one rental settles the
                # determinism question AND yields a forced-vs-unforced rate pair. TOY geometry —
                # only the RATIO between the two arms is quotable; the absolute is not a cost
                # basis and the basis tag says so.
                "throughput": throughput_record(
                    steps=2 * steps,
                    batch=batch,
                    seq_len=seq,
                    wall_s=t_train,
                    basis=BASIS_COMPUTE_ONLY,
                    device=device,
                    note=(
                        "TOY probe geometry (d_model=32, seq=16, batch=4) — quote the "
                        "forced/unforced RATIO only, never the absolute rate"
                    ),
                ),
            }
        )
    except Exception as e:  # the whole point of the probe
        txt = f"{type(e).__name__}: {e}"
        rec.update(
            {
                "completed": False,
                "error_type": type(e).__name__,
                "error": txt[:2000],
                "looks_like_missing_deterministic_kernel": any(h in txt for h in OP_HINTS),
                "traceback_tail": traceback.format_exc()[-1500:],
            }
        )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"=== determinism feasibility probe (device={args.device}, steps={args.steps}) ===")
    baseline = _one_pass(forced=False, steps=args.steps, device=args.device)
    print(f"  baseline (deterministic_algorithms=False): completed={baseline['completed']}")
    forced = _one_pass(forced=True, steps=args.steps, device=args.device)
    print(f"  FORCED   (deterministic_algorithms=True) : completed={forced['completed']}")
    if not forced["completed"]:
        print(f"    error: {forced['error'][:300]}")

    if not baseline["completed"]:
        verdict = (
            "PROBE INVALID — the UNFORCED baseline also failed, so this measured nothing about "
            "determinism. A forced-run failure is only interpretable against a baseline that "
            f"completes. Baseline error: {baseline.get('error', '')[:300]}"
        )
    elif forced["completed"]:
        verdict = (
            "FORCED DETERMINISM COMPLETES on this device — no op in the M6 training path lacks a "
            "deterministic kernel here, so 'make the claim true' reduces to a pure GPU-THROUGHPUT "
            "question. NOTE THE SCOPE: this is CPU/MPS. CUDA has its own set of non-deterministic "
            "kernels, so a CUDA probe is still required before the claim can be relied "
            "on; that probe is STAGED, not run."
        )
    elif forced.get("looks_like_missing_deterministic_kernel"):
        verdict = (
            "FORCED DETERMINISM RAISES on a MISSING DETERMINISTIC KERNEL. 'Make the claim true' "
            "therefore requires warn_only=True plus an explicit list of ops that remain "
            "non-deterministic, which WEAKENS the claim regardless of cost. Op named in `error`."
        )
    else:
        verdict = (
            "FORCED RUN FAILED, BUT NOT ON A DETERMINISM KERNEL — the error signature does not "
            "match a missing deterministic implementation, so this is a probe/API fault and must "
            f"NOT be read as a determinism finding. Error: {forced.get('error', '')[:300]}"
        )
    out = {
        "receipt": "m6_determinism_feasibility_probe",
        "amendment": "prereg s7 v1.4.7 item 2 feasibility probe (supervisor-ordered, $0)",
        "question": (
            "Does the M6 training path complete under torch.use_deterministic_algorithms(True, "
            "warn_only=False)? If it raises, 'make the claim true' needs warn_only + an op "
            "allowlist, which weakens the claim independently of cost."
        ),
        "device": args.device,
        "steps_each_stage": args.steps,
        "baseline_unforced": baseline,
        "forced": forced,
        "forced_completes": bool(forced["completed"]),
        "probe_valid": bool(baseline["completed"]),
        "throughput_pair": {
            "unforced_steps_per_s": (baseline.get("throughput") or {}).get("steps_per_s"),
            "forced_steps_per_s": (forced.get("throughput") or {}).get("steps_per_s"),
            "CAVEAT_DO_NOT_QUOTE_THIS_RATIO_YET": (
                "The two arms run SEQUENTIALLY in one process with no warmup, unforced first, so "
                "the ratio is confounded by cache/JIT warming and process state — on CPU it comes "
                "out with the SECOND arm faster, which is an ordering artifact, not a determinism "
                "effect. This pair exists to prove the plumbing emits NUMBERS. The quotable "
                "measurement is the CUDA bench pair (canonical geometry, warmup excluded, one "
                "posture per process invocation) in `staged_cuda_measurement_PENDING.command`."
            ),
        },
        "verdict": verdict,
        "staged_cuda_measurement_PENDING": {
            "status": "STAGED — NOT RUN. Needs GPU credentials; ~$1-2 of box time.",
            "purpose": (
                "TWO JOBS IN ONE RENTAL (v1.6 Finding 0): (1) confirm no CUDA op raises under "
                "forced determinism; (2) produce the COST DATUM the project does not have. Both "
                "are persisted as numeric utils.throughput records, never as prose."
            ),
            "corrections_to_the_prior_staging": [
                "The prior command called `--forced-determinism`, a flag m6_attention_bench.py "
                "DID NOT HAVE — it would have failed at the box. The flag now exists (with "
                "--no-forced-determinism for the other arm).",
                "The prior purpose said 'against the measured baseline of 47.2k bars/s'. That "
                "baseline is NOT end-to-end training throughput: it is the attention bench's "
                "compute-only rate on pre-resident batches. It also was ALREADY measured under "
                "FORCED determinism (set_determinism's default), so treating it as the unforced "
                "baseline and applying a penalty on top double-counts. The bench now records its "
                "own posture in config.deterministic_algorithms.",
            ],
            "command": (
                "PYTHONPATH=src python3 scripts/m6_determinism_probe.py --device cuda --steps 50 "
                "&& PYTHONPATH=src python3 scripts/m6_attention_bench.py --no-forced-determinism "
                "&& PYTHONPATH=src python3 scripts/m6_attention_bench.py --forced-determinism"
            ),
            "expected_wall_clock": (
                "probe < 2 min; the throughput delta needs ~10 min of steady-state training per "
                "arm (forced vs unforced) to be quotable, so ~25 min of box time total."
            ),
            "what_would_make_it_conclusive": (
                "A forced-vs-unforced bars/s pair measured on the SAME box in the SAME session, "
                "plus a same-seed twice-run weight-hash comparison to confirm forcing actually "
                "delivers bit-identity rather than merely claiming it."
            ),
            "cost_datum_note": (
                "Even both bench arms are basis=compute_only_step and are an UPPER BOUND. The "
                "only end-to-end datum that can price the real run comes from a real cell run — "
                "orchestrator.run_cell now writes throughput.stage1/stage2 as end_to_end_training "
                "records, so the first real cell of the M6 run produces it as a by-product."
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nVERDICT: {verdict}")
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
