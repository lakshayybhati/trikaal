"""ITEM E DIAGNOSTIC (c) — WHAT λ=3 COST EACH ARM IN OHLCV FIDELITY. $0, local, CPU.

THE QUESTION. `micro_point_weight = λ = 3.0` aims triple gradient pressure at the six micro dims.
The C-12 disclosure exists because that pressure is paid for in OHLCV reconstruction: bits spent
on the micro channels are bits taken from OHLCV. Both arms carry BYTE-IDENTICAL OHLCV columns —
`block_time_permute` never touches them — so OHLCV recon is a like-for-like comparison, and the
ratio between the arms is what λ=3 already cost.

WHY IT MATTERS HERE AND NOW. It bears directly on whether a HIGHER λ is affordable at all: if
λ=3 has already measurably degraded OHLCV fidelity, the budget the micro dims would need is being
taken from somewhere that is also being measured.

Uses `eval.diagnostics.ohlcv_recon_diagnostic` — the C-12 instrument itself, not a
re-implementation. REPORTS ONLY. NON-GATING by construction: no threshold, no clause, no
comparison against a bar.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake
from trikaal.eval.conformance import MDE_INPUTS, PINNED_MONEY_SEQ_LEN
from trikaal.eval.diagnostics import ohlcv_recon_diagnostic
from trikaal.run.matrix import load_symbol_arrays, windows_by_arm
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARM_MICRO, ARM_MICRO_SHUFFLED
from trikaal.train.checkpoint import load_checkpoint

CKPT = {
    "cell4_micro": ("runs_cloud/legibility_stop/cell4_seed0/tokenizer.pt", ARM_MICRO),
    "cell5_micro_shuffled": (
        "runs_cloud/legibility_stop/cell5_seed0/tokenizer.pt",
        ARM_MICRO_SHUFFLED,
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", type=Path, default=Path("processed/universe_bars"))
    ap.add_argument("--n-symbols", type=int, default=10)
    ap.add_argument("--tail-bars", type=int, default=20_000)
    ap.add_argument("--n-windows", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("runs_manifest/m6_ohlcv_recon_ratio.json"))
    a = ap.parse_args()
    if not a.lake.exists():
        print(f"LAKE MISSING at {a.lake}", file=sys.stderr)
        return 1

    pinned = json.loads(MDE_INPUTS.read_text())
    symbols = list(pinned["symbols_sampled"])[: a.n_symbols]
    window, train_frac = tuple(pinned["window"]), float(pinned["train_frac"])
    boundary = calendar_boundary_ms(window[0], window[1], train_frac)
    con = connect_lake(a.lake)

    out: dict = {
        "receipt": "m6_ohlcv_recon_ratio",
        "what": "C-12 OHLCV reconstruction for the two banked λ=3.0 tokenizers. OHLCV columns are "
        "BYTE-IDENTICAL across arms (block_time_permute never touches them), so this is "
        "like-for-like: worse OHLCV recon = capacity diverted to the micro channels.",
        "NON_GATING": "disclosure only — no threshold, no clause, cannot flip any verdict.",
        "NO_INTERPRETATION": "reports MAE per arm and the ratio; what it implies for λ is not "
        "this script's call.",
        "lambda_in_force": 3.0,
        "sampling": {
            "n_symbols": len(symbols),
            "tail_bars_per_symbol": a.tail_bars,
            "n_windows_per_arm": a.n_windows,
            "seq_len": PINNED_MONEY_SEQ_LEN,
        },
        "arms": {},
    }

    for label, (ck, arm) in CKPT.items():
        t0 = time.time()
        tok = load_checkpoint(Path(ck), TokenizerAE, map_location="cpu").model.eval()
        raw = {
            s: load_symbol_arrays(con, s, boundary_ms=boundary, tail_bars=a.tail_bars)
            for s in symbols
        }
        per_symbol = windows_by_arm(
            raw, seq_len=PINNED_MONEY_SEQ_LEN, boundary_ms=boundary, seed=0, arms=(arm,)
        )[arm]
        # Deterministic window selection: the FIRST n windows of each symbol, evenly shared.
        per_sym = max(1, a.n_windows // max(1, len(per_symbol)))
        xs, ms = [], []
        for sw in per_symbol:
            nb = sw.x.shape[0]
            for k in range(per_sym):
                s0 = k * PINNED_MONEY_SEQ_LEN
                if s0 + PINNED_MONEY_SEQ_LEN > nb:
                    break
                xs.append(sw.x[s0 : s0 + PINNED_MONEY_SEQ_LEN])
                ms.append(sw.mask[s0 : s0 + PINNED_MONEY_SEQ_LEN])
        x = np.stack(xs).astype(np.float32)
        m = np.stack(ms).astype(np.float32)
        d = ohlcv_recon_diagnostic(tok, x, m, device="cpu")
        d["n_windows"] = int(x.shape[0])
        out["arms"][label] = d
        print(
            f"[{label}] windows={x.shape[0]} mae={d['ohlcv_recon_mae']:.6f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    c4 = out["arms"]["cell4_micro"]["ohlcv_recon_mae"]
    c5 = out["arms"]["cell5_micro_shuffled"]["ohlcv_recon_mae"]
    out["comparison"] = {
        "cell4_ohlcv_recon_mae": c4,
        "cell5_ohlcv_recon_mae": c5,
        "ratio_cell5_over_cell4": round(c5 / c4, 6) if c4 else None,
        "delta_cell5_minus_cell4": round(c5 - c4, 8),
        "read_with": "OHLCV columns are identical across arms, so a HIGHER MAE means that arm "
        "reconstructed the SAME targets worse — capacity diverted. Non-gating.",
    }
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["comparison"], indent=2, sort_keys=True), flush=True)
    print(f"WROTE {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
