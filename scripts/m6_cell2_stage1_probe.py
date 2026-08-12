"""CELL 2 (FSQ + ohlcv) STAGE-1 CODEBOOK PROBE — the controlled comparison for claim 1.

Cell 1 is BSQ+ohlcv; cell 2 is FSQ+ohlcv. SAME seed, SAME arm, SAME 40 symbols, SAME pinned
Stage-1 budget. **The quantizer is the only thing that differs**, which is exactly the contrast
claim 1 makes — so the two utilizations are directly comparable and nothing else is.

WHY STAGE 1 ONLY. `cell_codebook_diagnostic(b_c, b_f, *, v_c, v_f)` takes token streams and vocab
sizes. The AR predictor is NOT an input. The number is a Stage-1 product, so Stage 2 (2.46 h) and
the eval leg (14.65 h) buy nothing here. Measured from the banked r0 manifest: stage 1 = 1168.3 s,
tokenize pass = 215.4 s => ~23 min, ~$0.18, versus 17.51 h and $8.09 for a full unit.

★ ALL 40 SYMBOLS, NOT THE PROBE DEFAULT OF 8. Utilization is "fraction of codes seen at least
once", which rises MONOTONICALLY with sample size. Cell 1's 0.8672 was measured over 84,153,600
tokens from 40 symbols. Measuring FSQ over 8 symbols would bias it DOWNWARD and manufacture the
asymmetry this probe exists to test for. The token counts must match or the comparison is void.

THIS IS A PROBE. `money_run=False`, `steps_stage2=0`, its own out-dir, no eval artifact, no
verdict clause, HEAD unmoved. It cannot enter the 25 and is not trying to.

REPORTS TWO NUMBERS AND DOES NOT INTERPRET THEM (supervisor's ruling, item 1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake
from trikaal.eval.conformance import (
    MDE_INPUTS,
    PINNED_CODEBOOK_MIN_UTILIZATION,
    PINNED_MONEY_SEQ_LEN,
    PINNED_STEPS_STAGE1,
)
from trikaal.eval.diagnostics import cell_codebook_diagnostic
from trikaal.run.matrix import load_symbol_arrays, windows_by_arm
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARM_OHLCV
from trikaal.train.cells import CELLS
from trikaal.train.checkpoint import load_checkpoint
from trikaal.train.orchestrator import OrchestratorConfig, run_cell
from trikaal.train.token_stream import tokenize_features

CELL2 = next(c for c in CELLS if c.cell_id == 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", type=Path, default=Path("processed/universe_bars"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("/root/m6_cell2_probe"))
    ap.add_argument("--out", type=Path, default=Path("/root/m6_cell2_stage1_probe.json"))
    ap.add_argument("--n-symbols", type=int, default=40)
    a = ap.parse_args()

    if not a.lake.exists():
        print(f"LAKE MISSING at {a.lake} — refusing to invent data", file=sys.stderr)
        return 1
    assert CELL2.quantizer == "fsq" and CELL2.arm == ARM_OHLCV, f"cell2 is not FSQ+ohlcv: {CELL2}"

    pinned = json.loads(MDE_INPUTS.read_text())
    symbols = list(pinned["symbols_sampled"])[: a.n_symbols]
    window, train_frac = tuple(pinned["window"]), float(pinned["train_frac"])
    boundary = calendar_boundary_ms(window[0], window[1], train_frac)
    print(f"[probe] cell={CELL2.name} seed={a.seed} symbols={len(symbols)}", flush=True)

    t0 = time.time()
    con = connect_lake(a.lake)
    raw = {s: load_symbol_arrays(con, s, boundary_ms=boundary, tail_bars=None) for s in symbols}
    by_arm = windows_by_arm(
        raw, seq_len=PINNED_MONEY_SEQ_LEN, boundary_ms=boundary, seed=a.seed, arms=(ARM_OHLCV,)
    )
    per_symbol = by_arm[ARM_OHLCV]
    t_load = time.time() - t0
    print(f"[probe] load+windows {t_load:.1f}s", flush=True)

    cfg = OrchestratorConfig(
        seeds=(a.seed,),
        seq_len=PINNED_MONEY_SEQ_LEN,
        steps_stage1=PINNED_STEPS_STAGE1,
        steps_stage2=0,  # STAGE 1 ONLY — the number is a Stage-1 output
        device=a.device,
        out_dir=a.out_dir,
        money_run=False,  # a probe declares itself; it never assembles
        micro_legibility_min=None,  # ohlcv arm is EXEMPT (cell 1's own log: "EXEMPT: ohlcv")
    )
    t1 = time.time()
    manifest = run_cell(CELL2, a.seed, {ARM_OHLCV: per_symbol}, cfg)
    t_train = time.time() - t1
    print(f"[probe] stage1 train {t_train:.1f}s", flush=True)

    ckpt = a.out_dir / f"{CELL2.name}_seed{a.seed}" / "tokenizer.pt"
    tok = load_checkpoint(ckpt, TokenizerAE, map_location=a.device)
    t2 = time.time()
    bcs, bfs = [], []
    for sw in per_symbol:
        b_c, b_f = tokenize_features(
            tok.model, sw.x, sw.mask, sw.segment_id, window=PINNED_MONEY_SEQ_LEN, device=a.device
        )
        bcs.append(b_c)
        bfs.append(b_f)
    b_c, b_f = np.concatenate(bcs), np.concatenate(bfs)
    t_tok = time.time() - t2
    print(f"[probe] tokenize {t_tok:.1f}s  tokens={b_c.size:,}", flush=True)

    diag = cell_codebook_diagnostic(b_c, b_f, v_c=tok.model.v_c, v_f=tok.model.v_f)

    rec = {
        "receipt": "m6_cell2_stage1_probe",
        "what": "FSQ codebook utilization at the pinned Stage-1 budget, 40 symbols — the "
        "controlled counterpart to cell1_seed0's 0.8672 (same seed, same ohlcv arm, "
        "ONLY the quantizer differs).",
        "PROBE_NOT_A_UNIT": "money_run=False, steps_stage2=0, own out-dir, no eval artifact, "
        "enters no verdict clause, HEAD unmoved. It cannot join the 25.",
        "NO_INTERPRETATION": "reports the numbers; which hypothesis they support is not this "
        "script's call (supervisor ruling, item 1).",
        "cell": CELL2.name,
        "quantizer": CELL2.quantizer,
        "arm": CELL2.arm,
        "seed": a.seed,
        "n_symbols": len(symbols),
        "n_tokens_per_leg": int(b_c.size),
        "steps_stage1": PINNED_STEPS_STAGE1,
        "steps_stage2": 0,
        "threshold_under_test": PINNED_CODEBOOK_MIN_UTILIZATION,
        "codebook": diag,
        "final_loss": manifest.get("final_loss") if isinstance(manifest, dict) else None,
        "seconds": {"load": t_load, "stage1": t_train, "tokenize": t_tok},
    }
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")

    c, f = diag["coarse"], diag["fine"]
    print(
        f"\n=== CELL 2 (FSQ + ohlcv) seed {a.seed} — "
        f"{len(symbols)} symbols, {b_c.size} tokens/leg ===",
        flush=True,
    )
    for nm, leg in (("coarse", c), ("fine", f)):
        print(
            f"  {nm:<6} utilization={leg['utilization']:.4f}  "
            f"n_used={leg['n_used']}/{leg['vocab']}  "
            f"entropy={leg['entropy_bits']:.4f} bits  perplexity={leg['perplexity']:.2f}",
            flush=True,
        )
    print(
        f"  effective_bits_per_token={diag['effective_bits_per_token']:.4f}  "
        f"capacity={diag['capacity_bits_per_token']:.4f}",
        flush=True,
    )
    print(f"WROTE {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
