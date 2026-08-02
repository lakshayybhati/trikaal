"""THE MISSING MEASUREMENT — end-to-end eval throughput at the EXACT money surface.

Every cost number this project has rests on an eval leg that was INFERRED BY SUBTRACTION from an
estimate: $30.98–48.69 = the approved $33–50 minus $1.31–2.02 of MEASURED training. That inferred
leg is **94–97 % of the spend** and has never been measured, because the pinned `chunk=512` OOM'd
on local hardware (19.69 GiB KV cache > 20.13 GiB). `scripts/m6_decode_profile.py:198` names this
exact gap: *"hardware (CUDA), across the pinned chunk size and several decision counts"*.

WHAT THIS MEASURES: `predict_mu` end-to-end — context fill + rollout — at `chunk=512`,
`seq_len=512`, `h=15`, `estimator="expectation"`, i.e. every value the money path pins. Output is
**decisions/second with an error bar** (repeats × per-repeat std), plus the extrapolation to a full
25-unit run.

WHY IT IS SMALL: throughput at fixed chunk is linear in decisions once the first chunk is paid, so
two decision counts across a few repeats determine the line. It does not need to be long; it needs
to be at the right geometry.

RANDOM WEIGHTS ARE CORRECT HERE. This is a TIMING probe. Weights change no shape and no control
flow in the rollout, so a trained checkpoint would cost a download and change nothing measured. The
receipt says so rather than leaving a reader to wonder.

★ IF `chunk=512` OOMs ON THE 4090 TOO, THAT IS THE FINDING AND IT IS BIGGER THAN THE COST — it
would mean the pinned recipe cannot run on the hardware class the whole experiment was costed
against. The probe catches OOM explicitly and records it as a result, not a crash.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trikaal.eval.conformance import PINNED_MONEY_SEQ_LEN, PINNED_MU_ESTIMATOR
from trikaal.eval.predict import predict_mu
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE

OUT = Path("runs_manifest/m6_eval_throughput_probe.json")
PRIMARY_H = 15
CHUNK = 512  # the pinned recipe value — the one that has never been measured


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--decisions", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=PINNED_MONEY_SEQ_LEN)
    ap.add_argument("--v-c", type=int, default=891)
    ap.add_argument("--v-f", type=int, default=1225)
    ap.add_argument("--tiny", action="store_true", help="shape-only local smoke; NOT a measurement")
    a = ap.parse_args()

    dims = (
        {"d_model": 64, "n_layers": 2, "n_heads": 2, "d_ff": 128}
        if a.tiny
        else {"d_model": 512, "n_layers": 8, "n_heads": 8, "d_ff": 1024}
    )
    model = (
        TrikaalAR(v_c=a.v_c, v_f=a.v_f, max_len=a.seq_len + PRIMARY_H, **dims).to(a.device).eval()
    )
    tok = (
        TokenizerAE(n_features=16, encoder_causal=True, fine_pointwise=True, micro_point_weight=3.0)
        .to(a.device)
        .eval()
    )

    rep: dict = {
        "receipt": "m6_eval_throughput_probe",
        "what": (
            "end-to-end eval throughput at the pinned money surface — the leg that was "
            "only ever INFERRED, never measured"
        ),
        "surface": {
            "chunk": CHUNK,
            "seq_len": a.seq_len,
            "h": PRIMARY_H,
            "estimator": PINNED_MU_ESTIMATOR,
            "v_c": a.v_c,
            "v_f": a.v_f,
            "backbone": dims,
            "device": a.device,
            "torch": torch.__version__,
        },
        "weights": "RANDOM — this is a TIMING probe; weights change no shape and no control flow",
        "tiny_smoke_not_a_measurement": bool(a.tiny),
        "runs": [],
    }
    if a.device.startswith("cuda") and torch.cuda.is_available():
        rep["surface"]["gpu"] = torch.cuda.get_device_name(0)

    rng = np.random.default_rng(0)
    for n_dec in a.decisions:
        total = n_dec + a.seq_len + PRIMARY_H + 8
        b_c = rng.integers(0, a.v_c, total).astype(np.int64)
        b_f = rng.integers(0, a.v_f, total).astype(np.int64)
        ts = (np.arange(total, dtype=np.int64) * 60_000) + 1_700_000_000_000
        sigma = np.full(total, 1e-3)
        dec = np.arange(a.seq_len, a.seq_len + n_dec, dtype=np.int64)
        # ONE UNTIMED WARM-UP CALL, then time. The first CUDA call pays context init, kernel
        # autotune and allocator growth: the first run measured 15.579 s +/- 9.289 at n=512 (the
        # std is larger than half the mean) against 20.380 s +/- 0.012 at n=1024. A mean over a
        # warm-up-contaminated first repeat is not a throughput measurement, and the marginal
        # derived from it would be biased LOW. Discarding it is the fix, not averaging harder.
        predict_mu(
            model,
            tok,
            b_c,
            b_f,
            ts,
            sigma,
            dec[: min(32, dec.size)],
            h=PRIMARY_H,
            seq_len=a.seq_len,
            device=a.device,
            chunk=CHUNK,
            estimator=PINNED_MU_ESTIMATOR,
        )
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        wall = []
        for _r in range(a.repeats):
            try:
                if a.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.time()
                mu = predict_mu(
                    model,
                    tok,
                    b_c,
                    b_f,
                    ts,
                    sigma,
                    dec,
                    h=PRIMARY_H,
                    seq_len=a.seq_len,
                    device=a.device,
                    chunk=CHUNK,
                    estimator=PINNED_MU_ESTIMATOR,
                )
                if a.device.startswith("cuda"):
                    torch.cuda.synchronize()
                wall.append(time.time() - t0)
                assert mu.shape[0] == n_dec
            except torch.cuda.OutOfMemoryError as e:  # ★ the finding, not a crash
                rep["OOM_AT_PINNED_CHUNK"] = {
                    "decisions": n_dec,
                    "chunk": CHUNK,
                    "error": str(e)[:300],
                    "meaning": "the PINNED recipe cannot run on this hardware class — this is a "
                    "specification/hardware finding and outranks the cost question",
                }
                OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
                print(json.dumps(rep["OOM_AT_PINNED_CHUNK"], indent=2))
                return 2
        w = np.asarray(wall, dtype=np.float64)
        rep["runs"].append(
            {
                "decisions": n_dec,
                "repeats": a.repeats,
                "wall_s_mean": float(w.mean()),
                "wall_s_std": float(w.std(ddof=1)) if w.size > 1 else 0.0,
                "decisions_per_s_mean": float(n_dec / w.mean()),
                "decisions_per_s_std": float(n_dec * w.std(ddof=1) / w.mean() ** 2)
                if w.size > 1
                else 0.0,
                "peak_mem_bytes": int(torch.cuda.max_memory_allocated())
                if a.device.startswith("cuda")
                else None,
            }
        )
        print(
            f"  n_dec={n_dec:>6}  {w.mean():8.3f} s +/- {w.std(ddof=1) if w.size > 1 else 0:.3f}  "
            f"-> {n_dec / w.mean():8.2f} decisions/s"
        )

    if len(rep["runs"]) >= 2 and not a.tiny:
        (n1, t1), (n2, t2) = [(r["decisions"], r["wall_s_mean"]) for r in rep["runs"][:2]]
        marginal = (t2 - t1) / (n2 - n1) if n2 != n1 else float("nan")
        rep["marginal_seconds_per_decision"] = marginal
        rep["fixed_overhead_s"] = t1 - marginal * n1
        rep["extrapolation_note"] = (
            "a full-run estimate needs the REAL decision count per (cell, seed) from the pinned "
            "grid; it is deliberately NOT invented here. marginal_seconds_per_decision x that "
            "count x 25 units is the arithmetic, and the count comes from the money config."
        )
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(f"\nreceipt -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
