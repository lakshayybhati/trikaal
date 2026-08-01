"""AUDIT TIER-2 C-1 — the single-bar-decode gap at CANONICAL dims, on the real lake.

    PYTHONPATH=src .venv/bin/python scripts/m6_c1_single_bar_decode.py

REPORT ONLY. This measures and prints; it proposes nothing and changes nothing.

THE STRUCTURE, WHICH IS NOT IN DOUBT (S-1). The tokenizer's decoder is a Transformer over the
SEQUENCE dimension and its reconstruction objective is optimised on FULL WINDOWS — every bar is
decoded with its neighbours visible. The M6 decision path does not do that: ``predict._rollout``
calls ``tok.decode_latent(z)`` with ``z`` of shape ``[n, 1, dim]`` — **one bar, length-1 sequence,
no context at all**. So the decoder is evaluated in a regime its objective never trained.

WHAT WAS AND WAS NOT ESTABLISHED BEFORE THIS. The auditor measured a **1.44x variance inflation**
and **16.2% sign flips** on a **d64 / 2-layer toy**. S-1 correctly recorded that the STRUCTURE is
certain and the MAGNITUDE is not — a toy an order of magnitude from canonical says little about
how large the effect is on the instrument that will actually be used. This probe holds the four
metrics identical and changes only the model and the data: **canonical d256 / 3 layers, trained,
on real lake bars**.

THE FOUR METRICS ARE THE AUDITOR'S FOUR, deliberately, so the numbers are directly comparable:
variance ratio, reconstruction MAE, sign agreement, mean |delta|.

A DISCLOSURE THAT BOUNDS WHAT THIS CAN CLAIM. No trained tokenizer in the repo carries the FULL
M6 pin set — both available canonical-dims checkpoints predate ``fine_pointwise=True`` and
``micro_point_weight=3.0``. Both are measured and both are reported rather than one being picked;
the decoder gap is a property of the DECODER (a sequence Transformer) and neither pin changes the
decoder's architecture, but the checkpoints are not the M6 instrument and no number here should be
quoted as if they were.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trikaal.constants import N_FEATURES
from trikaal.data.universe_loader import connect_lake
from trikaal.run.matrix import load_symbol_arrays
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.checkpoint import load_checkpoint
from trikaal.utils.seeding import set_determinism

OUT = Path("runs_manifest/m6_c1_single_bar_decode.json")
CANDIDATES = (
    ("runs_cloud/ckpt_cell4_seed0/tokenizer.pt", "cell4 cloud ckpt (encoder_causal=True)"),
    ("runs/m3/tokenizer.pt", "M3 ckpt (the Gate-A anchor's tokenizer)"),
)
# The auditor's d64/2-layer toy figures, for direct comparison — NOT a target.
TOY_VARIANCE_RATIO = 1.44
TOY_SIGN_FLIP_RATE = 0.162


def measure(
    tok: TokenizerAE,
    x: np.ndarray,
    mask: np.ndarray,
    *,
    seq_len: int,
    n_windows: int,
    seed: int,
) -> dict:
    """Per-bar reconstruction IN-WINDOW vs SINGLE-BAR, on the same quantized latents.

    The comparison is exact by construction: both regimes decode the SAME ``z_hat``. The only
    difference is the sequence length the decoder sees — L in training, 1 in the decision path.
    """
    set_determinism(seed, deterministic_algorithms=False)
    rng = np.random.default_rng(seed)
    usable = x.shape[0] - seq_len - 1
    starts = rng.choice(usable, size=min(n_windows, usable), replace=False)
    xb = np.stack([x[s : s + seq_len] for s in starts]).astype(np.float32)
    mb = np.stack([mask[s : s + seq_len] for s in starts])
    xt, mt = torch.from_numpy(xb), torch.from_numpy(mb.astype(np.float32))

    with torch.no_grad():
        # the SHIPPED path: latent -> quantize -> decode. Both regimes decode the SAME z_hat, so
        # the only difference is the sequence length the decoder sees.
        z_hat, _codes, _ci, _fi = tok.quant(tok.latent(xt, mt))
        recon_win = tok.decode_latent(z_hat)  # [B, L, F] — the trained regime
        # the DECISION-PATH regime: one bar at a time, length-1 sequence, no context
        singles = [tok.decode_latent(z_hat[:, t : t + 1, :]) for t in range(seq_len)]
        recon_one = torch.cat(singles, dim=1)  # [B, L, F]

    w = recon_win.numpy().astype(np.float64)
    o = recon_one.numpy().astype(np.float64)
    tgt = xb.astype(np.float64)
    d = o - w

    def per_dim(fn):
        return [float(fn(i)) for i in range(N_FEATURES)]

    var_w, var_o = w.var(axis=(0, 1)), o.var(axis=(0, 1))
    ratio = np.where(var_w > 0, var_o / np.maximum(var_w, 1e-300), np.nan)
    sign_agree = (np.sign(o) == np.sign(w)).mean(axis=(0, 1))

    return {
        "n_windows": int(xb.shape[0]),
        "seq_len": int(seq_len),
        "bars_compared": int(xb.shape[0] * seq_len),
        # (1) VARIANCE RATIO — the auditor's 1.44x on the toy
        "variance_ratio_dim0_return": float(ratio[0]),
        "variance_ratio_mean_over_dims": float(np.nanmean(ratio)),
        "variance_ratio_by_dim": per_dim(lambda i: ratio[i]),
        # (2) RECONSTRUCTION MAE against the true features, both regimes
        "recon_mae_in_window": float(np.abs(w - tgt).mean()),
        "recon_mae_single_bar": float(np.abs(o - tgt).mean()),
        "recon_mae_ratio": float(np.abs(o - tgt).mean() / max(np.abs(w - tgt).mean(), 1e-300)),
        "recon_mae_dim0_in_window": float(np.abs(w[..., 0] - tgt[..., 0]).mean()),
        "recon_mae_dim0_single_bar": float(np.abs(o[..., 0] - tgt[..., 0]).mean()),
        # (3) SIGN AGREEMENT — the auditor's 16.2% flips on the toy
        "sign_agreement_dim0_return": float(sign_agree[0]),
        "sign_flip_rate_dim0_return": float(1.0 - sign_agree[0]),
        "sign_agreement_mean_over_dims": float(sign_agree.mean()),
        # (4) MEAN |DELTA| between the two regimes
        "mean_abs_delta": float(np.abs(d).mean()),
        "mean_abs_delta_dim0": float(np.abs(d[..., 0]).mean()),
        "max_abs_delta": float(np.abs(d).max()),
        "mean_abs_delta_over_sd_of_in_window": float(
            np.abs(d).mean() / max(float(w.std()), 1e-300)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--lake", type=Path, default=Path("processed/universe_bars"))
    args = ap.parse_args()

    con = connect_lake(args.lake)
    symbols = [s for s in args.symbols.split(",") if s]
    results: dict = {}
    for path, label in CANDIDATES:
        p = Path(path)
        if not p.exists():
            results[path] = {"skipped": "checkpoint absent"}
            continue
        ck = load_checkpoint(p, TokenizerAE, map_location="cpu")
        tok = ck.model.eval()
        cfg = tok.get_config()
        per_symbol = {}
        for sym in symbols:
            raw = load_symbol_arrays(con, sym, boundary_ms=0, tail_bars=None)
            if raw["x"].shape[0] < args.seq_len + 2:
                per_symbol[sym] = {"skipped": "too few bars"}
                continue
            per_symbol[sym] = measure(
                tok,
                raw["x"],
                raw["mask"],
                seq_len=args.seq_len,
                n_windows=args.windows,
                seed=0,
            )
            r = per_symbol[sym]
            print(
                f"  {label[:24]:24s} {sym:9s} "
                f"var_ratio={r['variance_ratio_dim0_return']:.3f} "
                f"MAE {r['recon_mae_in_window']:.4f}->{r['recon_mae_single_bar']:.4f} "
                f"sign_agree={r['sign_agreement_dim0_return']:.4f} "
                f"mean|d|={r['mean_abs_delta']:.4f}",
                flush=True,
            )
        results[path] = {
            "label": label,
            "config": {
                k: cfg.get(k)
                for k in ("d_model", "n_layers", "quantizer", "encoder_causal", "fine_pointwise")
            },
            "carries_full_m6_pin_set": bool(
                cfg.get("encoder_causal") and cfg.get("fine_pointwise")
            ),
            "by_symbol": per_symbol,
        }

    doc = {
        "receipt": "m6_c1_single_bar_decode",
        "finding": "AUDIT TIER-2 C-1 — REPORT ONLY, nothing proposed",
        "question": (
            "the tokenizer decoder is a sequence Transformer trained on FULL WINDOWS, but the M6 "
            "decision path calls decode_latent on a LENGTH-1 sequence (predict._rollout). How "
            "large is the gap at CANONICAL dims on real data, versus the d64/2-layer toy the "
            "auditor measured?"
        ),
        "toy_reference": {
            "dims": "d64 / 2 layers",
            "variance_ratio": TOY_VARIANCE_RATIO,
            "sign_flip_rate": TOY_SIGN_FLIP_RATE,
            "status": "S-1: the STRUCTURE is certain, the MAGNITUDE was not established",
        },
        "config": {"symbols": symbols, "seq_len": args.seq_len, "windows": args.windows},
        "results": results,
        "DISCLOSURE": (
            "No trained tokenizer in the repo carries the FULL M6 pin set: both canonical-dims "
            "checkpoints predate fine_pointwise=True and micro_point_weight=3.0. Both are "
            "measured and both reported rather than one being chosen. The decoder gap is a "
            "property of the DECODER and neither pin alters the decoder's architecture, but "
            "these are not the M6 instrument and no number here should be quoted as if they were."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"\n[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
