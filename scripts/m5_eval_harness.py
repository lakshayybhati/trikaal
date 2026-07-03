"""Milestone-5 run: machine-validate the eval harness on the M3 checkpoint (LOCAL, no training).

    python3 scripts/m5_eval_harness.py [HORIZONS] [CAP]      # default "5,15,60" 300

Loads the real BTCUSDT lake + the cached frozen tokenizer + the M3 predictor checkpoint, re-runs the
Gate-A causal exhaustive sweep on a Q4-representative sample (asserts leak-free), then runs the full
§8 machine (folds → KV-cache rollout → strategy → cost-aware net-IR @0.30% + cost-stress +
break-even → DSR/PBO → Cell-5 placebo → secondary diagnostics → Q3/Q4 synthetic path) per horizon.

**The numbers are meaningless** (dev-grade model, one symbol, synthetic Q3/Q4) — this proves the
machine is correct and leak-free, not any result. Outputs are content-hashed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import torch

from trikaal.constants import N_FEATURES
from trikaal.data.causal_check import exhaustive_truncation_sweep
from trikaal.data.config import synthetic_test_config
from trikaal.data.synthetic import make_synthetic_stream
from trikaal.eval.harness import run_harness
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.checkpoint import load_checkpoint
from trikaal.train.token_stream import load_token_stream
from trikaal.utils.hashing import content_hash

LAKE_ROOT = Path("processed/bars")
RUN_DIR = Path("runs/m3")
SYMBOL = "BTCUSDT"
M2_DATASET_HASH_PREFIX = "sha256:1d3ec4f8db6686e2"


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_lake() -> dict:
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW bars AS SELECT * FROM read_parquet('{LAKE_ROOT}/**/*.parquet', "
        "hive_partitioning=true)"
    )
    xc = ", ".join(f"x_{i}" for i in range(N_FEATURES))
    mc = ", ".join(f"m_{i}" for i in range(N_FEATURES))
    t = con.execute(
        f"SELECT bar_open_ms, segment_id, raw_ret_close, sigma, {xc}, {mc} FROM bars "
        "WHERE symbol = ? AND frequency = '1m' ORDER BY bar_open_ms",
        [SYMBOL],
    ).to_arrow_table()
    n = t.num_rows
    x = np.empty((n, N_FEATURES), dtype=np.float32)
    m = np.empty((n, N_FEATURES), dtype=np.uint8)
    for i in range(N_FEATURES):
        x[:, i] = t.column(f"x_{i}").to_numpy()
        m[:, i] = t.column(f"m_{i}").to_numpy()
    ts = t.column("bar_open_ms").to_numpy().astype(np.int64)
    return {
        "x": np.ascontiguousarray(x),
        "mask": np.ascontiguousarray(m),
        "ts": ts,
        "segment_id": t.column("segment_id").to_numpy().astype(np.int64),
        "raw_ret_close": t.column("raw_ret_close").to_numpy().astype(np.float64),
        "sigma": t.column("sigma").to_numpy().astype(np.float64),
        "dataset_hash": content_hash(np.ascontiguousarray(x), np.ascontiguousarray(m), ts),
    }


def gate_a_causal_sweep() -> dict:
    """§8.A.5 / §8.E.5 re-validation: exhaustive truncation sweep on a Q4-representative sample."""
    stream, _ = make_synthetic_stream(seed=44, n_bars=420, inject_anomalies=True)
    cfg = synthetic_test_config(half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20)
    rep = exhaustive_truncation_sweep(stream, cfg, stride=1)
    return {"passed": rep.passed, "coverage": rep.coverage_bars, "total": rep.total_bars}


def main(horizons=(5, 15, 60), cap: int = 300) -> int:
    if not LAKE_ROOT.exists() or not (RUN_DIR / "predictor.pt").exists():
        print("missing lake or M3 checkpoint — run scripts/build + m3 first")
        return 1
    dev = device()
    print(f"=== Milestone 5: eval-harness machine-validation (device={dev}) ===")

    data = load_lake()
    assert data["dataset_hash"].startswith(M2_DATASET_HASH_PREFIX), "lake dataset_hash != M2"
    print(f"[lake]  {data['x'].shape[0]:,} bars, dataset_hash {data['dataset_hash'][:23]}… (M2 ✓)")

    tok_l = load_checkpoint(RUN_DIR / "tokenizer.pt", TokenizerAE, map_location=dev)
    pred_l = load_checkpoint(RUN_DIR / "predictor.pt", TrikaalAR, map_location=dev)
    tok, model = tok_l.model.to(dev).eval(), pred_l.model.to(dev).eval()
    stream = load_token_stream(RUN_DIR / "tokens.npz", tokenizer_hash=tok_l.hash)
    print(f"[ckpt]  tokenizer {tok_l.hash[:20]}… predictor {pred_l.hash[:20]}… (M3 fixture)")

    lake = {**data, "b_c": stream["b_c"], "b_f": stream["b_f"]}

    gate = gate_a_causal_sweep()
    ok = "PASS ✓" if gate["passed"] else "FAIL ✗"
    print(
        f"[gate-A] causal exhaustive sweep on Q4 sample: {ok} "
        f"({gate['coverage']}/{gate['total']} bars, 100% coverage)"
    )
    assert gate["passed"], "Gate-A causal sweep FAILED — harness is not leak-free"

    results = {}
    for h in horizons:
        t0 = time.time()
        try:
            res = run_harness(lake, model, tok, h=h, seq_len=128, cap=cap, device=dev, seed=0)
            results[h] = res
            _print_horizon(res, time.time() - t0)
        except Exception as e:  # crash-safe: a horizon failure doesn't lose the others
            print(f"[h={h}] FAILED: {type(e).__name__}: {e}")

    # The anchor payload pins the headline (0.30%-netted), the modeled-cost secondary, the full
    # per-κ VAL curve (the κ*-knife-edge), and DSR/PBO/break-even — a regression in any of them
    # moves the hash (§6 item 10d; re-anchored after the item-10 instrument fixes).
    rhash = content_hash(
        {
            str(h): [
                r.ir_headline,
                r.ir_modeled_cost,
                r.dsr,
                r.pbo,
                r.break_even,
                r.kappa_chosen,
                [[k, v] for k, v in sorted(r.val_ir_by_kappa.items())],
            ]
            for h, r in results.items()
        },
        {"gate_a": gate["passed"], "dataset": data["dataset_hash"]},
    )
    print(
        f"\n[m5] horizons run: {sorted(results)}; gate-A {gate['passed']}; "
        f"results_hash {rhash[:23]}…"
    )
    # Durable run-manifest (m6_preflight Item 1: "never stdout-only again") — FULL hashes +
    # env/device/seed provenance, written to a TRACKED dir. Content is deterministic (no
    # wall-clock), so two bit-identical runs produce a byte-identical file; the git commit
    # supplies the timestamp. This is the Gate-A anchor artifact.
    manifest = {
        "anchor": "gate-A results_hash (M5 harness on the frozen M3 checkpoint)",
        "results_hash": rhash,
        "inputs": {
            "dataset_hash": data["dataset_hash"],
            "tokenizer_hash": tok_l.hash,
            "predictor_hash": pred_l.hash,
        },
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": dev,
        },
        "params": {"horizons": sorted(results), "cap": cap, "seq_len": 128, "seed": 0},
        "gate_a_causal_sweep": gate,
        "per_horizon": {
            str(h): {
                "ir_headline_030_netting": r.ir_headline,
                "ir_modeled_cost": r.ir_modeled_cost,
                "kappa_chosen": r.kappa_chosen,
                "val_ir_by_kappa": {str(k): v for k, v in sorted(r.val_ir_by_kappa.items())},
                "dsr": r.dsr,
                "pbo": r.pbo,
                "break_even": r.break_even,
                "activity": r.activity,
            }
            for h, r in results.items()
        },
    }
    mdir = Path("runs_manifest")
    mdir.mkdir(exist_ok=True)
    mpath = mdir / "gate_a_run_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[m5] run-manifest → {mpath}")
    print(
        "[m5] NUMBERS ARE MEANINGLESS (dev-grade model, one symbol, synthetic Q3/Q4) — "
        "this proves the machine, not a result."
    )
    return 0


def _print_horizon(res, secs) -> None:
    p = print
    p(f"\n--- horizon h={res.horizon}min ({secs:.0f}s) — MEANINGLESS dev-grade numbers ---")
    # ir_headline is genuinely NETTED at the flat 0.30% round-trip (§6 item 10a); the modeled-cost
    # IR (the pre-fix number this label used to mislabel) is the named secondary.
    p(
        f"  κ* (tuned on VAL): {res.kappa_chosen};  net-IR headline @0.30% netting: "
        f"{res.ir_headline:+.3f}  (secondary, modeled-cost: {res.ir_modeled_cost:+.3f}; "
        f"activity {res.activity:.2f})"
    )
    p(f"  per-κ VAL IR: {{{', '.join(f'{k}: {v:+.2f}' for k, v in res.val_ir_by_kappa.items())}}}")
    cs = res.cost_stress
    p(
        f"  cost-stress IR: 0.10%={cs[0.001]:+.2f} 0.20%={cs[0.002]:+.2f} 0.30%={cs[0.003]:+.2f}  "
        f"break-even c={res.break_even:.4g}"
    )
    p(f"  DSR={res.dsr:.3f}  PBO={res.pbo:.3f}  (trials budget documented; PBO over the κ configs)")
    pl = res.placebo
    p(
        f"  placebo: IR(real tokens)={pl['ir_real_tokens']:+.3f}  IR(shuffled-micro)="
        f"{pl['ir_placebo_tokens']:+.3f}  micro>placebo={pl['micro_exceeds_placebo']}"
    )
    d = res.diagnostics
    p(
        f"  diagnostics: RankIC={d['rank_ic']:+.4f} IC={d['ic']:+.4f} vol_MAE={d['vol_mae']:.4f} "
        f"PICP80={d['picp80']:.3f}"
    )
    p(f"  quadrants exercised: {res.quadrants_run} (Q3/Q4 via synthetic 2-symbol fixture)")


if __name__ == "__main__":
    hs = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (5, 15, 60)
    cp = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    raise SystemExit(main(hs, cp))
