"""M6 Phase-1 SMOKE gate (m6_design §6): the whole 5-cell arc, tiny, on REAL lake symbols.

    PYTHONPATH=src python3 scripts/m6_smoke.py [--steps 150] [--device cpu]

All 5 cells instantiate → train a few hundred steps on 3 real symbols (incl. ≥1 THIN coin) drawn
fold-legally from the compacted universe lake → checkpoint (content-hashed) → the cross-sectional
driver scores all 5 models and emits the 5-model + placebo verdict. **Numbers are MEANINGLESS by
design** (tiny shells, minutes of training) — the gate is "the machinery runs end-to-end".

Symbols: BTCUSDT (deep, live), KLAYUSDT (delisted → tail-truncated coverage), FRONTUSDT (one of
the 43 THIN coins — listed 2023-09, weeks before the train/eval boundary, stale 2.5%) — the thin
coin's presence in the training draw (with its down-weighted share) is part of the smoke evidence.

Writes runs/m6_smoke/smoke_manifest.json (durable, never stdout-only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from trikaal.constants import N_FEATURES
from trikaal.data.universe_loader import (
    build_symbol_windows,
    calendar_boundary_ms,
    connect_lake,
)
from trikaal.eval.xsection import SymbolEval, XSectionConfig, ablation_verdict, score_cell
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARMS
from trikaal.train.checkpoint import load_checkpoint
from trikaal.train.orchestrator import OrchestratorConfig, run_all_cells
from trikaal.train.tripwire import TripwireConfig, TripwireMonitor

LAKE = Path("processed/universe_bars")
SYMBOLS = ("BTCUSDT", "KLAYUSDT", "FRONTUSDT")  # deep + delisted-tail + THIN(43)
THIN = "FRONTUSDT"
WINDOW = ("2021-01-01", "2025-01-01")
TRAIN_FRAC = 0.7
SEQ_LEN = 32
TRAIN_TAIL_BARS = 200_000  # slice: enough fold-legal windows without loading 2.1M-bar histories

TOK_KW = dict(d_model=16, n_layers=1, n_heads=2, d_ff=32, max_len=64)
BB_KW = dict(
    n_layers=1,
    d_model=16,
    d_ff=32,
    n_heads=2,
    mtp_depths=0,
    max_len=64,
    ffn_dropout=0.0,
    resid_dropout=0.0,
    attn_dropout=0.0,
    token_dropout=0.0,
)


def load_symbol(con, symbol: str, boundary_ms: int) -> dict:
    """One symbol's raw arrays from the lake, sliced to [boundary − TRAIN_TAIL_BARS, end)."""
    xcols = ", ".join(f"x_{i}" for i in range(N_FEATURES))
    mcols = ", ".join(f"m_{i}" for i in range(N_FEATURES))
    t = con.execute(
        f"SELECT bar_open_ms, segment_id, sigma, raw_ret_close, {xcols}, {mcols} "
        "FROM bars WHERE symbol = ? ORDER BY bar_open_ms",
        [symbol],
    ).to_arrow_table()
    n = t.num_rows
    ts = t.column("bar_open_ms").to_numpy().astype(np.int64)
    lo = max(0, int(np.searchsorted(ts, boundary_ms)) - TRAIN_TAIL_BARS)
    x = np.empty((n, N_FEATURES), dtype=np.float32)
    m = np.empty((n, N_FEATURES), dtype=np.uint8)
    for i in range(N_FEATURES):
        x[:, i] = t.column(f"x_{i}").to_numpy()
        m[:, i] = t.column(f"m_{i}").to_numpy()
    return {
        "x": x[lo:],
        "mask": m[lo:],
        "ts": ts[lo:],
        "segment_id": t.column("segment_id").to_numpy().astype(np.int64)[lo:],
        "sigma": t.column("sigma").to_numpy().astype(np.float64)[lo:],
        "raw_ret_close": t.column("raw_ret_close").to_numpy().astype(np.float64)[lo:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    if not LAKE.exists():
        print("missing lake — run the M4b ingest first")
        return 1
    boundary = calendar_boundary_ms(*WINDOW, TRAIN_FRAC)
    con = connect_lake(LAKE)
    print(f"=== M6 SMOKE (device={args.device}) — NUMBERS ARE MEANINGLESS BY DESIGN ===")
    raw = {}
    for s in SYMBOLS:
        raw[s] = load_symbol(con, s, boundary)
        print(f"[lake] {s}: {raw[s]['x'].shape[0]:,} bars loaded (train-tail + eval region)")

    # per-arm training windows (fold-legal; the Cell-5 shuffle inside, per (seed, symbol))
    per_symbol_by_arm = {
        arm: [
            build_symbol_windows(
                s,
                d["x"],
                d["mask"],
                d["segment_id"],
                d["ts"],
                arm=arm,
                seq_len=SEQ_LEN,
                boundary_ms=boundary,
                seed=args.seed,
            )
            for s, d in raw.items()
        ]
        for arm in ARMS
    }
    for sw in per_symbol_by_arm["micro"]:
        print(f"[draw]  {sw.symbol}: {sw.n_windows:,} fold-legal train windows")

    cfg = OrchestratorConfig(
        seeds=(args.seed,),
        seq_len=SEQ_LEN,
        batch_size=16,
        steps_stage1=args.steps,
        steps_stage2=args.steps,
        tok_kwargs=TOK_KW,
        backbone_kwargs=BB_KW,
        expect_backbone_params=None,
        enforce_parity=False,  # parity is a canonical-dims gate (KAT'd); smoke shells are tiny
        device=args.device,
        out_dir=Path("runs/m6_smoke"),
        state_every=50,
        wandb_mode="disabled",
    )
    monitor = TripwireMonitor(TripwireConfig(k_first_steps=min(100, args.steps)))
    top = run_all_cells(per_symbol_by_arm, cfg, monitor)
    print(f"[train] {top['n_runs']} cell-runs complete ({time.time() - t0:.0f}s)")

    # cross-sectional scoring of all 5 smoke models + the placebo verdict
    xcfg = XSectionConfig(
        window_start=WINDOW[0],
        window_end=WINDOW[1],
        train_frac=TRAIN_FRAC,
        h=15,
        seq_len=SEQ_LEN,
        cap_per_symbol=60,
        device=args.device,
        seed=args.seed,
    )
    sym_evals = [SymbolEval(symbol=s, **raw[s]) for s in SYMBOLS]
    cell_ir: dict[int, float] = {}
    scores = {}
    from trikaal.train.cells import CELLS

    for spec in CELLS:
        run_name = f"{spec.name}_seed{args.seed}"
        rd = Path("runs/m6_smoke") / run_name
        tok = load_checkpoint(rd / "tokenizer.pt", TokenizerAE, map_location=args.device)
        pred = load_checkpoint(rd / "predictor.pt", TrikaalAR, map_location=args.device)
        sc = score_cell(
            run_name,
            pred.model.to(args.device).eval(),
            tok.model.to(args.device).eval(),
            spec.arm,
            sym_evals,
            xcfg,
        )
        cell_ir[spec.cell_id] = sc.ir_headline
        scores[run_name] = sc
        thin_n = sc.per_symbol_decisions.get(THIN, 0)
        print(
            f"[eval]  {run_name}: IR@0.30%={sc.ir_headline:+.2f} κ*={sc.kappa_chosen} "
            f"activity={sc.activity:.2f} decisions={sc.n_decisions} (thin {THIN}: {thin_n})"
        )
        cb = sc.codebook
        print(
            f"[codebook] {run_name}: coarse use {cb['coarse']['utilization']:.1%} "
            f"H={cb['coarse']['entropy_bits']:.2f}b, fine use {cb['fine']['utilization']:.1%} "
            f"H={cb['fine']['entropy_bits']:.2f}b, effective "
            f"{cb['effective_bits_per_token']:.2f}/{cb['capacity_bits_per_token']:.2f} bpt "
            "(non-gating diagnostic)"
        )

    verdict = ablation_verdict(cell_ir)
    print(
        f"[verdict] ΔIR_info (Cell4-Cell5) = {verdict['delta_info']:+.3f}  "
        f"micro>placebo={verdict['micro_exceeds_placebo']}  (MEANINGLESS — machinery proof only)"
    )

    manifest = {
        "smoke": "m6_phase1",
        "symbols": list(SYMBOLS),
        "thin_coin": THIN,
        "thin_coin_drawn_stage1": {
            r["run"]: r["draw"]["drawn_by_symbol_stage1"].get(THIN, 0)
            for r in [
                json.loads(
                    (
                        Path("runs/m6_smoke") / f"{s.name}_seed{args.seed}" / "run_manifest.json"
                    ).read_text()
                )
                for s in CELLS
            ]
        },
        "artifacts": top["artifacts"],
        "cell_ir_headline": {str(k): v for k, v in cell_ir.items()},
        "codebook_by_run": {run: sc.codebook for run, sc in scores.items()},
        "verdict": verdict,
        "wall_s": round(time.time() - t0, 1),
        "note": "NUMBERS MEANINGLESS BY DESIGN — tiny shells, minutes of training; the gate is "
        "that all 5 cells + the cross-sectional driver ran end-to-end on real lake symbols.",
    }
    out = Path("runs/m6_smoke/smoke_manifest.json")
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(f"[smoke] manifest → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
