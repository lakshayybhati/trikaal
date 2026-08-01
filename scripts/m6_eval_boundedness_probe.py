"""§7 v1.6 — IS THE EVAL LEG CPU-BOUND? A $0 local probe that decides the rental class.

    PYTHONPATH=src .venv/bin/python scripts/m6_eval_boundedness_probe.py --decisions 512

THE QUESTION. The 4090 measured 0.021604938 s per decision at CANONICAL dims (d512, 8 layers,
seq 512, chunk 512) with nvidia-smi reading 0% utilization. Two readings:

  (2) CPU-BOUND — the rollout is Python/launch-bound, the GPU contributes little, and renting a
      $0.3671/hr card that sits idle is waste. Then the eval leg wants a CPU box selected on
      SINGLE-CORE performance.
  (1)/(3) GPU IS DOING THE WORK — then the rate is simply what canonical dims cost, and a GPU is
      the right instrument.

THE TEST. Run the SAME canonical geometry locally on CPU (and MPS if present) with random
weights — timing does not need trained weights — and compare seconds-per-decision against the
4090's measured 0.021604938.

  local CPU ≈ 4090  ⇒  the GPU added ~nothing ⇒ CPU-BOUND (outcome 2)
  local CPU ≫ 4090  ⇒  the GPU carried the work ⇒ NOT CPU-bound (outcome 1/3)

A WITHDRAWN CLAIM THIS PROBE EXISTS TO REPLACE: the first comparison put the 4090's canonical-dims
rate against a CPU gate that ran ``--local-smoke`` — d_model 16, 1 layer, seq 32. That is a
different model by roughly three orders of magnitude in per-decision FLOPs, not a slower machine.
"17.4x slower on the rented box" was an artifact of comparing two different models and is
withdrawn; this probe holds the geometry FIXED and varies only the device.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trikaal.eval.predict import predict_mu
from trikaal.train.cells import CELLS, build_cell_backbone, build_cell_tokenizer
from trikaal.utils.seeding import set_determinism

OUT = Path("runs_manifest/m6_eval_boundedness_probe.json")
BOX_S_PER_DECISION = 0.021604938  # measured on the 4090, canonical dims, chunk 512
V_C, V_F = 891, 1225


def time_device(device: str, *, n_decisions: int, seq_len: int, h: int, chunk: int) -> dict:
    set_determinism(0, deterministic_algorithms=False)
    spec = CELLS[3]  # cell4 fsq_micro — the arm the box timed
    tok = build_cell_tokenizer(spec, max_len=max(512, seq_len)).to(device).eval()
    model = build_cell_backbone(V_C, V_F, max_len=seq_len + 64).to(device).eval()

    n_bars = n_decisions + seq_len + h + 8
    rng = np.random.default_rng(0)
    # predict_mu takes NUMPY arrays, not tensors — passing tensors raised .astype on both arms,
    # which the control-arm rule correctly reported as PROBE INVALID rather than a verdict.
    b_c = rng.integers(0, V_C, n_bars).astype(np.int64)
    b_f = rng.integers(0, V_F, n_bars).astype(np.int64)
    ts = (1_600_000_000_000 + np.arange(n_bars) * 60_000).astype(np.int64)
    sigma = np.full(n_bars, 0.01, dtype=np.float64)
    dec = np.arange(seq_len, seq_len + n_decisions, dtype=np.int64)

    t0 = time.time()
    with torch.no_grad():
        mu = predict_mu(
            model, tok, b_c, b_f, ts, sigma, dec, h=h, seq_len=seq_len, device=device, chunk=chunk
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.time() - t0
    return {
        "device": device,
        "n_decisions": int(n_decisions),
        "wall_s": wall,
        "seconds_per_decision": wall / n_decisions,
        "mu_finite_frac": float(np.isfinite(np.asarray(mu)).mean()),
        "vs_box_ratio": (wall / n_decisions) / BOX_S_PER_DECISION,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--h", type=int, default=15)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    results = {}
    for dev in devices:
        print(f"[probe] {dev} — canonical dims, {args.decisions} decisions …")
        try:
            results[dev] = time_device(
                dev, n_decisions=args.decisions, seq_len=args.seq_len, h=args.h, chunk=args.chunk
            )
            r = results[dev]
            print(
                f"        {r['seconds_per_decision']:.6f} s/dec ({r['vs_box_ratio']:.2f}x the 4090)"
            )
        except Exception as e:  # a probe that dies must not be read as a result
            results[dev] = {"device": dev, "error": f"{type(e).__name__}: {e}"[:400]}
            print(f"        FAILED: {results[dev]['error'][:160]}")

    ok = [r for r in results.values() if "seconds_per_decision" in r]
    best = min((r["seconds_per_decision"] for r in ok), default=float("nan"))
    ratio = best / BOX_S_PER_DECISION if ok else float("nan")
    if not ok:
        verdict = "PROBE INVALID — no local device produced a rate; nothing is concluded."
    elif ratio <= 2.0:
        verdict = (
            f"CPU-BOUND (outcome 2). The best local rate is {ratio:.2f}x the 4090's, i.e. a "
            "consumer CPU is within ~2x of the rented GPU. The card is contributing little, so "
            "renting a GPU for the eval leg pays for silicon that sits idle — select a CPU "
            "instance on SINGLE-CORE performance instead."
        )
    elif ratio >= 5.0:
        verdict = (
            f"NOT CPU-BOUND (outcome 1/3). The best local rate is {ratio:.2f}x SLOWER than the "
            "4090, so the GPU is carrying the work and the measured rate is simply what canonical "
            "dims cost. A GPU is the right instrument; the cost figure stands on its own merits."
        )
    else:
        verdict = (
            f"AMBIGUOUS — best local rate is {ratio:.2f}x the 4090, between the 2x and 5x "
            "thresholds. Neither reading is supported; report as inconclusive rather than "
            "choosing."
        )

    out = {
        "receipt": "m6_eval_boundedness_probe",
        "amendment": "prereg §7 v1.6 — $0 investigation of the eval-leg cost (supervisor-ordered)",
        "question": "Is the eval leg CPU-bound? It decides whether the eval rental is GPU or CPU.",
        "box_reference": {
            "seconds_per_decision": BOX_S_PER_DECISION,
            "device": "RTX 4090",
            "dims": "canonical d512 / 8 layers / seq 512 / chunk 512",
            "nvidia_smi_utilization": "0% at every sample",
        },
        "config": {
            "decisions": args.decisions,
            "seq_len": args.seq_len,
            "h": args.h,
            "chunk": args.chunk,
            "geometry": "CANONICAL — identical to the box; only the DEVICE varies",
        },
        "results": results,
        "best_local_seconds_per_decision": best,
        "best_local_over_box_ratio": ratio,
        "verdict": verdict,
        "WITHDRAWN_CLAIM": (
            "The earlier '17.4x slower on the rented box' compared the 4090 at CANONICAL dims "
            "(d512, 8 layers, seq 512) against a CPU gate running --local-smoke (d_model 16, "
            "1 layer, seq 32) — different models by orders of magnitude in per-decision FLOPs, "
            "not a slower machine. That comparison is WITHDRAWN. This probe fixes the geometry "
            "and varies only the device, which is what the question actually required."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nVERDICT: {verdict}")
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
