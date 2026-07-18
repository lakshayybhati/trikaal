"""Token-causality probe — do PAST bars' tokens change when only FUTURE bars change?

    PYTHONPATH=src python3 scripts/m6_token_causality_probe.py [--out runs_manifest/...]

THE FINDING (2026-07-19, canary root-cause chase): ``tokenize_features`` pre-tokenizes each
series in fixed contiguous windows, and the tokenizer encoder is **bidirectional by default**
(``encoder_causal=False``, blueprint-spec default) — so the token for bar t is a function of
every bar in its tokenization chunk, INCLUDING bars after t. The blueprint's causal-safety
note (spec §"tokenizer ≠ forecast") argues this cannot leak future into a forecast because
"the token for bar t is a label, not a prediction" — which covers the TRAINING TARGET side.
It does not cover the CONDITIONING side: at eval, ``predict_mu`` conditions on the
pre-tokenized ``token_{≤t}`` as INPUTS, and any future-bar content inside them reaches the
forecast made "at t". Invariant 2 (causal safety, hard) binds every eval input.

This probe measures that dependence directly: for each chunk, mutate ONLY bars ≥ ``mutate_from``
and count how many tokens of the UNTOUCHED earlier bars change. Any nonzero rate = the eval
context carries future-bar information. It also runs the structural control: under
``encoder_causal=True`` the rate must be exactly 0 (attention masking), trained or not.

Measured at discovery (committed artifact): trained toy tokenizers flip 5–13 % of past-bar
coarse tokens (FSQ and BSQ alike; synthetic AND real-lake data); the causal flag flips 0.0 %.
Side effect explained: within-chunk token correlation is also why toy AR cells learn chunk
context instead of genuine bar dynamics (the canary's planted circuit was drowned by it).

Adjudication is the supervisor's (research surface + spec default + prereg instrument). The
probe itself is evidence-only and changes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from trikaal.constants import N_FEATURES
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.checkpoint import load_checkpoint

DEFAULT_OUT = Path("runs_manifest/m6_token_causality_probe.json")


def flip_rates(
    tok: TokenizerAE,
    x: np.ndarray,
    m: np.ndarray,
    *,
    window: int,
    mutate_from: int,
    n_chunks: int,
    seed: int = 7,
) -> dict:
    """Fraction of bar-<mutate_from tokens that change when bars ≥ mutate_from are replaced."""
    tok.eval()
    rng = np.random.default_rng(seed)
    flips_c = np.zeros(mutate_from)
    flips_f = np.zeros(mutate_from)
    n = 0
    for k in range(n_chunks):
        a = k * window
        if a + window > x.shape[0]:
            break
        xb = torch.from_numpy(x[a : a + window][None])
        mb = torch.from_numpy(m[a : a + window].astype(np.float32))[None]
        x2 = xb.clone()
        x2[0, mutate_from:, :] = torch.from_numpy(
            rng.standard_normal((window - mutate_from, x.shape[1])).astype(np.float32)
        )
        with torch.no_grad():
            c0, f0 = tok.encode_tokens(xb, mb)
            c1, f1 = tok.encode_tokens(x2, mb)
        flips_c += (c0[0, :mutate_from] != c1[0, :mutate_from]).numpy()
        flips_f += (f0[0, :mutate_from] != f1[0, :mutate_from]).numpy()
        n += 1
    total = n * mutate_from
    return {
        "n_chunks": n,
        "past_bars_per_chunk": mutate_from,
        "coarse_flip_rate": float(flips_c.sum() / total),
        "fine_flip_rate": float(flips_f.sum() / total),
        "coarse_flip_by_position": (flips_c / max(n, 1)).round(4).tolist(),
    }


def _synthetic(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    m = np.zeros((n, N_FEATURES), dtype=np.uint8)
    m[:, 13:16] = 1
    return x, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--chunks", type=int, default=60)
    args = ap.parse_args()
    results: dict[str, dict] = {}

    # 1) structural control: encoder_causal=True must show EXACTLY zero, trained or not
    x, m = _synthetic(0, 60 * 32)
    causal = TokenizerAE(
        d_model=64, n_layers=1, n_heads=2, d_ff=128, max_len=64, encoder_causal=True
    ).eval()
    results["untrained_causal_flag"] = flip_rates(
        causal, x, m, window=32, mutate_from=20, n_chunks=args.chunks
    )
    bidi = TokenizerAE(d_model=64, n_layers=1, n_heads=2, d_ff=128, max_len=64).eval()
    results["untrained_bidirectional"] = flip_rates(
        bidi, x, m, window=32, mutate_from=20, n_chunks=args.chunks
    )

    # 2) trained tokenizers from this session's local validation runs, where present
    trained = {
        "trained_fsq_synthetic_canary": (
            Path("runs/m6_canary/planted/cell4_fsq_micro_seed0/tokenizer.pt"),
            None,
        ),
        "trained_bsq_synthetic_canary": (
            Path("runs/m6_canary/planted/cell3_bsq_micro_seed0/tokenizer.pt"),
            None,
        ),
        "trained_fsq_real_lake_smoke": (
            Path("runs/m6_toy/cell4_fsq_micro_seed0/tokenizer.pt"),
            "lake",
        ),
    }
    for name, (path, data) in trained.items():
        if not path.exists():
            results[name] = {"skipped": f"{path} absent in this checkout"}
            continue
        tok = load_checkpoint(path, TokenizerAE).model.eval()
        win = int(tok.get_config().get("max_len", 64))
        win = min(win, 64)
        if data == "lake":
            lake = Path("processed/universe_bars")
            if not lake.exists():
                results[name] = {"skipped": "lake absent"}
                continue
            from trikaal.data.universe_loader import connect_lake

            con = connect_lake(lake)
            xcols = ", ".join(f"x_{i}" for i in range(N_FEATURES))
            mcols = ", ".join(f"m_{i}" for i in range(N_FEATURES))
            t = con.execute(
                f"SELECT {xcols}, {mcols} FROM bars WHERE symbol='BTCUSDT' "
                "ORDER BY bar_open_ms LIMIT 4096"
            ).to_arrow_table()
            xr = np.stack([t.column(f"x_{i}").to_numpy() for i in range(N_FEATURES)], 1).astype(
                np.float32
            )
            mr = np.stack([t.column(f"m_{i}").to_numpy() for i in range(N_FEATURES)], 1).astype(
                np.uint8
            )
            results[name] = flip_rates(
                tok, xr, mr, window=win, mutate_from=win * 5 // 8, n_chunks=args.chunks
            )
        else:
            results[name] = flip_rates(
                tok, x, m, window=win, mutate_from=win * 5 // 8, n_chunks=args.chunks
            )
        results[name]["checkpoint"] = str(path)

    doc = {
        "probe": "m6_token_causality",
        "question": "do past bars' tokens depend on future bars inside the tokenization chunk?",
        "reading": "any nonzero flip rate under the bidirectional default = future-bar content "
        "in the pre-tokenized eval context (predict_mu's INPUT); the causal flag must be 0.0",
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True))
    for name, r in results.items():
        if "skipped" in r:
            print(f"[probe] {name}: {r['skipped']}")
        else:
            print(
                f"[probe] {name}: coarse {r['coarse_flip_rate']:.1%}, "
                f"fine {r['fine_flip_rate']:.1%} over {r['n_chunks']} chunks"
            )
    print(f"[probe] artifact → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
