"""Milestone-2 run: BTCUSDT real-data slice → Parquet/DuckDB lake → Stage-1 tokenizer.

    python scripts/build_btcusdt_stage1.py [N_MONTHS]

Builds per-bar features for BTCUSDT 2023 (first N_MONTHS, default 12), writes the Hive Parquet
lake, round-trips through DuckDB, trains the FSQ tokenizer Stage-1, and prints the acceptance
report: reconstruction-MAE convergence, codebook usage/perplexity (collapse check), and the
per-stage coarse/fine contribution Δ_fine (residual-decay guard).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from trikaal.data.build import build_symbol_features, year_months
from trikaal.data.lake import PROCESSED_ROOT_DEFAULT, assemble_window, connect, write_lake
from trikaal.train.train_tokenizer import train_stage1
from trikaal.utils.hashing import content_hash

SYMBOL = "BTCUSDT"


def main(n_months: int = 12) -> int:
    months = year_months(2023)[:n_months]
    print(f"=== Milestone 2: {SYMBOL} 2023 ({len(months)} months: {months[0]}..{months[-1]}) ===")

    t0 = time.time()
    out, _stream = build_symbol_features(SYMBOL, months)
    n = len(out)
    dhash = content_hash(out.x, out.m, out.ts)
    print(f"[build] {n:,} bars, {out.segment_id.max() + 1} segment(s), {time.time() - t0:.0f}s")
    finite = bool(np.all(np.isfinite(out.x_f64)))
    print(f"        x∈[{out.x_f64.min():.2f},{out.x_f64.max():.2f}] finite={finite}")
    print(f"        target_valid={int(out.target_valid.sum()):,}  dataset_hash={dhash[:23]}…")

    t0 = time.time()
    write_lake(out, SYMBOL, root=PROCESSED_ROOT_DEFAULT)
    con = connect(PROCESSED_ROOT_DEFAULT)
    lo, hi = int(out.ts.min()), int(out.ts.max()) + 1
    data = assemble_window(con, SYMBOL, lo, hi)
    ok = data["x"].shape == out.x.shape and np.allclose(data["x"], out.x, atol=0)
    print(
        f"[lake]  Parquet+DuckDB round-trip {'OK' if ok else 'MISMATCH'} "
        f"({data['x'].shape[0]:,} bars, {time.time() - t0:.0f}s)"
    )

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[stage1] training FSQ tokenizer on real bars (device={dev}) …")
    t0 = time.time()
    res = train_stage1(
        data["x"],
        data["mask"],
        data["segment_id"],
        seq_len=128,
        batch_size=64,
        peak_lr=1e-3,
        max_steps=4000,
        eval_every=100,
        device=dev,
        seed=0,
    )
    print(f"[stage1] done ({time.time() - t0:.0f}s)")
    cols = ("step", "val_mae", "coarse_mae", "Δ_fine", "use_c", "use_f", "ppl_c", "ppl_f")
    widths = (6, 10, 12, 9, 7, 7, 8, 8)
    print("\n" + "".join(f"{c:>{w}}" for c, w in zip(cols, widths, strict=True)))
    for r in res.history:
        print(
            f"{r['step']:>6}{r['val_mae']:>10.4f}{r['coarse_mae']:>12.4f}{r['delta_fine']:>9.4f}"
            f"{r['usage_coarse']:>7.2f}{r['usage_fine']:>7.2f}{r['ppl_coarse']:>8.0f}{r['ppl_fine']:>8.0f}"
        )
    cs = res.converged_step
    conv = "<= 2k OK" if cs and cs <= 2000 else ("FLAG: > 2k" if cs else "no plateau in budget")
    last = res.history[-1]
    fine = "fine group ALIVE" if last["delta_fine"] > 0 else "FLAG: fine group DEAD"
    collapse = (
        "no collapse"
        if last["usage_coarse"] > 0.5 and last["usage_fine"] > 0.5
        else "CHECK collapse"
    )
    print("\n=== Stage-1 report ===")
    print(f"  converged_step : {cs}  ({conv})")
    print(f"  final val MAE  : {res.final_recon_mae:.4f}")
    print(f"  Δ_fine (final) : {last['delta_fine']:.4f}  ({fine})")
    print(
        f"  codebook usage : coarse {last['usage_coarse']:.2f} / fine {last['usage_fine']:.2f}  "
        f"(ppl {last['ppl_coarse']:.0f}/{last['ppl_fine']:.0f}) — {collapse}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 12))
