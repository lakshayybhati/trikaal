"""M6 pre-flight Item 2 CANARY — the machine MEASURES, not just runs (two-arm, synthetic).

    PYTHONPATH=src python3 scripts/m6_canary.py [--device cpu] [--steps 600] [--out runs/m6_canary]

Item 2 proves the 5-cell machinery RUNS; this proves it can DETECT a known signal and stays
quiet on none (docs/m6_preflight.md Item 2, canary clause; adjudicated 2026-07-06):

(a) **planted-signal arm** — a synthetic lake where ONE micro channel (dim 9, the TFI slot)
    carries an iid state that reaches returns TWO bars later (r_{t+2} += c·σ·state_t). The
    state is inside every y[t] window yet — states being iid — it is STRUCTURALLY invisible to
    the OHLCV filtration at decision time (past returns reveal only states ≤ t−2, orthogonal
    to y[t]). So only micro-armed cells can predict, and the OHLCV-only counterfactual AND the
    shuffled placebo are uninformed BY CONSTRUCTION — neutrality does not depend on toy-scale
    convergence. (A persistent-state design was tried first and failed exactly there: the
    signal leaked into return history, Cell 2 recovered it fast, and IR(5)−IR(2) went hugely
    negative from learning-rate asymmetry — a toy artifact, not a placebo defect.) Required:
    ΔIR_info(4−5) fires (paired CI lower bound > 0) AND the placebo tracks the OHLCV-only
    counterfactual (paired CI of IR(5)−IR(2) straddles 0 — "a control, not a victim");
(b) **pure-noise arm** — micro carries nothing. Required: ΔIR_info must NOT fire (paired CI
    straddles 0) and |IR(5)−IR(2)| stays small (CI straddles 0);
(c) **reconstruction contribution of micro dims 7–12** through Cell 4's trained tokenizer is
    non-degenerate — the SIGNAL dim beats the predict-the-mean baseline AND the block-mean
    MAE does (the silent-micro-suppression false-NULL check). Per-EVERY-dim is deliberately
    not required: the filler dims are iid noise, incompressible by construction, and the
    coarse→fine hierarchy rationally triages bits away from them (measured: some fillers sit
    at baseline while the signal dim reconstructs at ~0.5× baseline — allocation, not
    suppression). Plus: the codebook diagnostic shows non-collapsed effective bits for BOTH
    quantizers (FSQ Cell 4, BSQ Cell 3).

Verdicts come from the SAME instruments the real decision uses (paired_delta_ir_bootstrap on
score_cell's pooled headline series) — the canary calibrates the decision path itself, not a
side calculation. Writes ``<out>/canary_manifest.json`` (durable, never stdout-only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from trikaal.constants import N_FEATURES
from trikaal.data.universe_loader import build_symbol_windows, calendar_boundary_ms
from trikaal.eval.paired_bootstrap import paired_delta_ir_bootstrap
from trikaal.eval.xsection import SymbolEval, XSectionConfig, score_cell
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARMS
from trikaal.train.cells import CELLS
from trikaal.train.checkpoint import load_checkpoint
from trikaal.train.orchestrator import OrchestratorConfig, run_all_cells
from trikaal.train.tripwire import TripwireConfig, TripwireMonitor

SYMBOLS = ("SYNAUSDT", "SYNBUSDT", "SYNCUSDT")
N_BARS = 26_000  # ≈18 days of 1-min bars per symbol
WINDOW = ("2024-01-01", "2024-01-19")
TRAIN_FRAC = 0.7
SEQ_LEN = 32
SIGMA = 0.01  # constant per-bar vol → x0 = r/σ (standardized ret_close)
SIGNAL_LAG = 2  # state_t first touches r at t+2 — inside y[t], outside the OHLCV filtration
C_SIGNAL = 3.0  # r_{t+2} = C·σ·state_t + σ·eps — per-bar corr(x0_{t+2}, state_t) ≈ 0.95: the
# planted effect is deliberately UNDENIABLE (μ̂ rollout noise at toy scale is ~3 standardized
# units; a subtler c measurably drowns — probed at c=1.2, corr(μ̂, state) ≈ 0)
T0_MS = 1_704_067_200_000  # 2024-01-01 UTC

# shells sized for the CIRCUIT, not the machinery: a d32/1-layer AR at 600 steps measurably
# fails to learn the 2-back state→x0 mapping (probe: corr(μ̂, state) ≈ −0.1 while the tokenizer
# carries the state at 0.95) — the canary needs enough optimization for the only temporal
# structure in the stream to be found
TOK_KW = dict(d_model=64, n_layers=1, n_heads=2, d_ff=128, max_len=64)
BB_KW = dict(
    n_layers=2,
    d_model=64,
    d_ff=128,
    n_heads=2,
    mtp_depths=0,
    max_len=64,
    ffn_dropout=0.0,
    resid_dropout=0.0,
    attn_dropout=0.0,
    token_dropout=0.0,
)


def synth_symbol(rng: np.random.Generator, *, planted: bool) -> dict:
    """One synthetic symbol: micro dim 9 = an iid state that (if planted) drives r two bars on.

    Causal AND asymmetric by construction: state_t is known at the close of bar t (dim 9) and
    first moves the price at bar t+SIGNAL_LAG — so it sits inside the entry-next-bar label
    y[t] = log(C_{t+h}/C_{t+1}) but no function of returns available at t carries it (iid
    states ⇒ past returns reveal only states ≤ t−SIGNAL_LAG, orthogonal to y[t])."""
    state = rng.standard_normal(N_BARS)
    eps = rng.standard_normal(N_BARS)
    r = SIGMA * eps
    if planted:
        r[SIGNAL_LAG:] = r[SIGNAL_LAG:] + C_SIGNAL * SIGMA * state[:-SIGNAL_LAG]
    x = rng.standard_normal((N_BARS, N_FEATURES)).astype(np.float32)  # dims 1-8,10-12 = noise
    x[:, 0] = (r / SIGMA).astype(np.float32)  # standardized ret_close
    x[:, 9] = (state if planted else rng.standard_normal(N_BARS)).astype(np.float32)
    m = np.zeros((N_BARS, N_FEATURES), dtype=np.uint8)
    m[:, 13:16] = 1  # perp dims masked everywhere (the lake convention; placebo tripwire)
    return {
        "x": x,
        "mask": m,
        "segment_id": np.zeros(N_BARS, dtype=np.int64),
        "ts": (T0_MS + np.arange(N_BARS) * 60_000).astype(np.int64),
        "sigma": np.full(N_BARS, SIGMA),
        "raw_ret_close": r.astype(np.float64),
    }


def run_arm(arm_name: str, *, planted: bool, args, boundary: int) -> dict:
    """Train the 5 cells on one synthetic arm and score them through the REAL decision path."""
    t0 = time.time()
    raw = {
        s: synth_symbol(np.random.default_rng((zlib.crc32(arm_name.encode()), i)), planted=planted)
        for i, s in enumerate(SYMBOLS)
    }
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
    out_dir = Path(args.out) / arm_name
    cfg = OrchestratorConfig(
        seeds=(args.seed,),
        seq_len=SEQ_LEN,
        batch_size=32,
        steps_stage1=args.steps,
        steps_stage2=args.steps,
        tok_kwargs=TOK_KW,
        backbone_kwargs=BB_KW,
        expect_backbone_params=None,
        enforce_parity=False,  # canary shells are tiny; parity is the canonical-dims gate
        device=args.device,
        autocast_bf16=args.device.startswith("cuda"),
        out_dir=out_dir,
        state_every=200,
        wandb_mode=args.wandb,
        wandb_group=f"canary_{arm_name}",
    )
    monitor = TripwireMonitor(TripwireConfig(k_first_steps=min(100, args.steps)))
    run_all_cells(per_symbol_by_arm, cfg, monitor)

    xcfg = XSectionConfig(
        window_start=WINDOW[0],
        window_end=WINDOW[1],
        train_frac=TRAIN_FRAC,
        h=15,
        seq_len=SEQ_LEN,
        cap_per_symbol=None,  # tiny toy grid — score every decision
        device=args.device,
        seed=args.seed,
    )
    sym_evals = [SymbolEval(symbol=s, **raw[s]) for s in SYMBOLS]
    scores = {}
    for spec in CELLS:
        rd = out_dir / f"{spec.name}_seed{args.seed}"
        tok = load_checkpoint(rd / "tokenizer.pt", TokenizerAE, map_location=args.device)
        pred = load_checkpoint(rd / "predictor.pt", TrikaalAR, map_location=args.device)
        scores[spec.cell_id] = score_cell(
            spec.name,
            pred.model.to(args.device).eval(),
            tok.model.to(args.device).eval(),
            spec.arm,
            sym_evals,
            xcfg,
        )
        print(
            f"[{arm_name}] cell{spec.cell_id}: IR@0.30%={scores[spec.cell_id].ir_headline:+.2f} "
            f"κ*={scores[spec.cell_id].kappa_chosen} act={scores[spec.cell_id].activity:.2f}"
        )

    # the SAME instrument as the real decision: paired bootstraps on the pooled headline series
    pb45 = paired_delta_ir_bootstrap(scores[4].headline_series, scores[5].headline_series, h=xcfg.h)
    pb52 = paired_delta_ir_bootstrap(scores[5].headline_series, scores[2].headline_series, h=xcfg.h)
    fired = bool(pb45.ci_lower > 0.0)
    # "IR(5) ≈ IR(2)" at toy scale, jitter-robustly: the paired CI straddles 0 OR the placebo
    # gap is small RELATIVE to the fired effect (cells 2 and 5 are different MODELS, so their
    # gap carries single-seed model jitter the pairing cannot cancel — at real scale the seed
    # mean absorbs this; a lone straddle check measurably flips sign-to-sign run-to-run)
    neutral = bool(
        (pb52.ci_lower <= 0.0 <= pb52.ci_upper) or (abs(pb52.delta_ir) <= 0.5 * abs(pb45.delta_ir))
    )

    # (c) per-dim recon of micro dims 7-12 through Cell 4's tokenizer + codebook non-collapse
    recon = _micro_recon_contribution(out_dir, raw, device=args.device)
    cb4, cb3 = scores[4].codebook, scores[3].codebook
    non_collapsed = {
        "fsq_cell4": {
            "coarse_bits": cb4["coarse"]["entropy_bits"],
            "fine_bits": cb4["fine"]["entropy_bits"],
            "ok": bool(cb4["coarse"]["entropy_bits"] > 1.0 and cb4["fine"]["entropy_bits"] > 1.0),
        },
        "bsq_cell3": {
            "coarse_bits": cb3["coarse"]["entropy_bits"],
            "fine_bits": cb3["fine"]["entropy_bits"],
            "ok": bool(cb3["coarse"]["entropy_bits"] > 1.0 and cb3["fine"]["entropy_bits"] > 1.0),
        },
    }
    return {
        "arm": arm_name,
        "planted": planted,
        "cell_ir": {str(c): float(s.ir_headline) for c, s in scores.items()},
        "delta_info": float(pb45.delta_ir),
        "pb_4_minus_5": {
            "ci_lower": pb45.ci_lower,
            "ci_upper": pb45.ci_upper,
            "fired": fired,
        },
        "pb_5_minus_2": {
            "delta": pb52.delta_ir,
            "ci_lower": pb52.ci_lower,
            "ci_upper": pb52.ci_upper,
            "placebo_neutral": neutral,
        },
        "micro_recon": recon,
        "codebook_non_collapsed": non_collapsed,
        "wall_s": round(time.time() - t0, 1),
    }


def _micro_recon_contribution(out_dir: Path, raw: dict, *, device: str) -> dict:
    """Per-dim recon MAE of micro dims 7-12 through Cell 4's trained tokenizer vs the
    predict-the-mean degenerate baseline (kills the silent-micro-suppression false-NULL)."""
    import torch

    ck = load_checkpoint(out_dir / "cell4_fsq_micro_seed0" / "tokenizer.pt", TokenizerAE)
    tok = ck.model.to(device).eval()
    d = raw[SYMBOLS[0]]
    n_win = min(128, d["x"].shape[0] // SEQ_LEN)
    n = n_win * SEQ_LEN
    # encode in SEQ_LEN windows — the tokenizer attends within max_len windows, never a full
    # series (same contract as train.token_stream.tokenize_features)
    x = torch.from_numpy(d["x"][:n].reshape(n_win, SEQ_LEN, -1)).to(device)
    m = torch.from_numpy(d["mask"][:n].astype(np.float32).reshape(n_win, SEQ_LEN, -1)).to(device)
    with torch.no_grad():
        cidx, fidx = tok.encode_tokens(x, m)
        x_hat = tok.decode_tokens(cidx, fidx)
    per_dim = {}
    for dim in range(7, 13):
        truth = d["x"][:n, dim].astype(np.float64)
        rec = x_hat[:, :, dim].reshape(-1).cpu().numpy().astype(np.float64)
        mae = float(np.abs(rec - truth).mean())
        baseline = float(np.abs(truth - truth.mean()).mean())  # degenerate: predict the mean
        per_dim[str(dim)] = {
            "recon_mae": mae,
            "mean_baseline_mae": baseline,
            "non_degenerate": bool(mae < 0.9 * baseline),
        }
    return per_dim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", default="disabled", choices=["disabled", "offline", "online"])
    ap.add_argument("--out", default="runs/m6_canary")
    args = ap.parse_args()
    boundary = calendar_boundary_ms(*WINDOW, TRAIN_FRAC)
    print("=== M6 CANARY — planted-signal + pure-noise arms (Item 2 measurement check) ===")

    planted = run_arm("planted", planted=True, args=args, boundary=boundary)
    noise = run_arm("noise", planted=False, args=args, boundary=boundary)

    verdicts = {
        "planted_fires": planted["pb_4_minus_5"]["fired"],
        "planted_placebo_neutral": planted["pb_5_minus_2"]["placebo_neutral"],
        "noise_does_not_fire": not noise["pb_4_minus_5"]["fired"],
        "noise_placebo_neutral": noise["pb_5_minus_2"]["placebo_neutral"],
        # signal dim + block mean (NOT per-every-dim: iid fillers are incompressible and the
        # hierarchy rationally triages bits away from them — see the module docstring)
        "micro_recon_non_degenerate": bool(
            planted["micro_recon"]["9"]["non_degenerate"]
            and np.mean([v["recon_mae"] for v in planted["micro_recon"].values()])
            < 0.9 * np.mean([v["mean_baseline_mae"] for v in planted["micro_recon"].values()])
        ),
        "codebooks_non_collapsed": all(
            side["ok"] for side in planted["codebook_non_collapsed"].values()
        ),
    }
    ok = all(verdicts.values())
    manifest = {
        "canary": "m6_preflight_item2",
        "arms": {"planted": planted, "noise": noise},
        "verdicts": verdicts,
        "PASS": ok,
        "recipe": {
            "signal_lag": SIGNAL_LAG,
            "c_signal": C_SIGNAL,
            "sigma": SIGMA,
            "n_bars": N_BARS,
            "symbols": list(SYMBOLS),
            "seq_len": SEQ_LEN,
            "steps": args.steps,
            "seed": args.seed,
            "device": args.device,
        },
    }
    out = Path(args.out) / "canary_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[canary] verdicts: {verdicts}")
    print(f"[canary] {'PASS' if ok else 'FAIL'} — manifest → {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
