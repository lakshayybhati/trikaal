"""AUDIT TIER-2 C-1, LEG (i)+(ii) — the single-bar decode under the FULL M6 PIN SET, per arm.

    PYTHONPATH=src .venv/bin/python scripts/m6_c1_pinset_decode.py

REPORT ONLY. Measures and prints; proposes nothing, changes nothing.

WHY THE FIRST C-1 MEASUREMENT DOES NOT SETTLE IT. That run used the two trained checkpoints in the
repo, and **neither carries ``fine_pointwise=True``**. Under §7 v1.4 the fine subtoken becomes a
PER-BAR encoding of bar t's own features via a pointwise branch, with the causal contextual encoder
keeping only the coarse dims (``model.py:83-86``) — precisely the mechanism that could make a
single-bar decode meaningful, and it is ``PINNED_FINE_POINTWISE`` for every M6 cell. So the earlier
**0.507** measures the PRE-v1.4 architecture, i.e. the very problem the interface re-spec was built
to fix, not the cells that will run.

BUT IT DOES NOT FOLLOW THAT v1.4 CLOSES IT, and this probe assumes nothing either way:
``model.py:104-105`` constructs ``TokenizerDecoder(..., False, ...)`` — **the decoder is hardcoded
bidirectional and window-trained**, so decoding a length-1 sequence is still out of distribution.
``fine_pointwise`` gives the decoder bar-local CONTENT when its attention becomes a no-op; whether
that recovers the SIGN is empirical.

THE COMPARISON IS MATCHED, WHICH THE FIRST ONE WAS NOT. The earlier numbers came from checkpoints
that differ in training as well as architecture, so they could not isolate the pin. Here the data,
seed, steps, optimizer, dims and quantizer are identical and **only the pin set varies**:

    PRE-v1.4 : encoder_causal=True, fine_pointwise=False, micro_point_weight=1.0
    M6 PINS  : encoder_causal=True, fine_pointwise=True,  micro_point_weight=3.0

LEG (ii) — SYMMETRY ACROSS ARMS, which decides SEVERITY. Every arm is measured: OHLCV (7 dims) and
micro (16 dims). If the gap is symmetric it costs POWER and cannot bias the paired contrast, which
is survivable. If it is arm-dependent it does NOT cancel in ΔIR(4−5) and it is a confound.

A THIRD REGIME IS MEASURED BECAUSE IT EXISTS, not because anything is proposed. When
``fine_pointwise`` is on, the model also builds ``point_decoder`` (``model.py:95-102``) — a
genuinely per-bar, zero-cross-bar decode head — whose own comment says *"Train-time only; the AR
interface and the main decode path are unchanged."* The decision path calls ``decode_latent``, i.e.
the bidirectional window decoder. Reporting what that per-bar head would have produced is a
measurement of the gap's nature. **It is not a proposal to wire it in.**
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trikaal.data.universe_loader import connect_lake
from trikaal.run.matrix import load_symbol_arrays
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import (
    ARM_MICRO,
    ARM_MICRO_SHUFFLED,
    ARM_OHLCV,
    arm_feature_idx,
    arm_n_features,
    shuffle_micro,
)
from trikaal.utils.seeding import set_determinism

OUT = Path("runs_manifest/m6_c1_pinset_decode.json")
TOY_SIGN_AGREEMENT = 0.838  # auditor, d64/2L
PRIOR_CKPT_SIGN_AGREEMENT = 0.507  # C-1 leg 1, causal ckpt WITHOUT fine_pointwise

CONFIGS = {
    "pre_v1_4": {"encoder_causal": True, "fine_pointwise": False, "micro_point_weight": 1.0},
    "m6_pin_set": {"encoder_causal": True, "fine_pointwise": True, "micro_point_weight": 3.0},
}


def four_metrics(w: np.ndarray, o: np.ndarray, tgt: np.ndarray) -> dict:
    """The auditor's four, held identical so every row is directly comparable."""
    d = o - w
    vw, vo = w[..., 0].var(), o[..., 0].var()
    degenerate = bool(vo <= 1e-8 * max(vw, 1e-30) or float(np.std(o[..., 0])) <= 1e-8)
    return {
        # A CONSTANT single-bar output scores sign_agreement 1.0 whenever the window sign is also
        # constant — a perfect score from a collapsed decode. Measured live in this experiment
        # (micro_shuffled / pre_v1_4: var_ratio 0.000, sign_agree 1.0000, |d|/sd 1.246), so the
        # flag is not hypothetical. Sign agreement must never be read without it.
        "single_bar_degenerate": degenerate,
        "variance_ratio_dim0": float(vo / vw) if vw > 0 else float("nan"),
        "sign_agreement_dim0": float((np.sign(o[..., 0]) == np.sign(w[..., 0])).mean()),
        "recon_mae_in_window_dim0": float(np.abs(w[..., 0] - tgt[..., 0]).mean()),
        "recon_mae_single_bar_dim0": float(np.abs(o[..., 0] - tgt[..., 0]).mean()),
        "recon_mae_ratio_dim0": float(
            np.abs(o[..., 0] - tgt[..., 0]).mean()
            / max(np.abs(w[..., 0] - tgt[..., 0]).mean(), 1e-30)
        ),
        "mean_abs_delta_dim0": float(np.abs(d[..., 0]).mean()),
        "mean_abs_delta_over_sd_dim0": float(
            np.abs(d[..., 0]).mean() / max(float(w[..., 0].std()), 1e-30)
        ),
    }


def train_and_measure(*, arm: str, cfg_name: str, xw, mw, steps: int, seed: int, lr: float) -> dict:
    set_determinism(seed, deterministic_algorithms=False)
    f = arm_n_features(arm)
    tok = TokenizerAE(n_features=f, d_model=256, n_layers=3, **CONFIGS[cfg_name])
    opt = torch.optim.AdamW(tok.parameters(), lr=lr)
    tok.train()
    losses = []
    for step in range(steps):
        i = torch.randint(0, xw.shape[0], (8,))
        loss = tok(xw[i], mw[i])["loss"]  # THE SHIPPED OBJECTIVE, not a re-implementation
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            losses.append(round(float(loss.detach()), 5))

    tok.eval()
    with torch.no_grad():
        xb, mb = xw[:8], mw[:8]
        z_hat, _c, _ci, _fi = tok.quant(tok.latent(xb, mb))
        recon_win = tok.decode_latent(z_hat)
        L = xb.shape[1]
        recon_one = torch.cat([tok.decode_latent(z_hat[:, t : t + 1, :]) for t in range(L)], dim=1)
        w = recon_win.numpy().astype(np.float64)
        o = recon_one.numpy().astype(np.float64)
        tgt = xb.numpy().astype(np.float64)
        res = four_metrics(w, o, tgt)

        # THE THIRD REGIME — measured because it exists, not proposed. point_decoder is the
        # genuinely per-bar head; the DECISION PATH does not use it.
        res["point_decoder_regime"] = None
        if getattr(tok, "fine_pointwise", False):
            fine = list(tok.quant.fine_idx)
            p = tok.point_decoder(z_hat[..., fine]).numpy().astype(np.float64)
            res["point_decoder_regime"] = four_metrics(w, p, tgt)

    res.update(
        {
            "arm": arm,
            "n_features": f,
            "config": cfg_name,
            "pins": dict(CONFIGS[cfg_name]),
            "steps": steps,
            "loss_trace": losses,
            "bars_compared": int(xb.shape[0] * xb.shape[1]),
        }
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1,2", help="replicates — the spread is the yardstick")
    ap.add_argument("--lake", type=Path, default=Path("processed/universe_bars"))
    args = ap.parse_args()

    seeds = [int(v) for v in str(args.seeds).split(",") if v != ""]
    con = connect_lake(args.lake)
    raw = load_symbol_arrays(con, args.symbol, boundary_ms=0, tail_bars=400_000)
    rng = np.random.default_rng(seeds[0])
    starts = rng.choice(raw["x"].shape[0] - args.seq_len - 1, size=args.windows, replace=False)
    x_full = np.stack([raw["x"][s : s + args.seq_len] for s in starts]).astype(np.float32)
    m_full = np.stack([raw["mask"][s : s + args.seq_len] for s in starts]).astype(np.float32)

    # THE PLACEBO ARM IS INCLUDED, and it is the one that decides the PRIMARY. Cells 4 and 5 are
    # BOTH 16-dim micro-shaped (micro vs micro_shuffled), so an ohlcv-vs-micro asymmetry cannot
    # contaminate dIR(4-5) — it lands on clause 3's dIR(4-2) and the §5 fallback dIR(2-1), which
    # DO compare across widths. The question for the primary is whether the decode gap differs
    # between REAL micro and SHUFFLED micro at the same width.
    seg = raw["segment_id"]
    xs_full, ms_full = shuffle_micro(
        raw["x"].astype(np.float64), raw["mask"], seg, symbol=args.symbol, seed=seeds[0]
    )
    xs_w = np.stack([xs_full[s : s + args.seq_len] for s in starts]).astype(np.float32)
    ms_w = np.stack([ms_full[s : s + args.seq_len] for s in starts]).astype(np.float32)

    results = {}
    for arm in (ARM_OHLCV, ARM_MICRO, ARM_MICRO_SHUFFLED):
        idx = list(arm_feature_idx(arm))
        src_x, src_m = (xs_w, ms_w) if arm == ARM_MICRO_SHUFFLED else (x_full, m_full)
        xw = torch.from_numpy(np.ascontiguousarray(src_x[..., idx]))
        mw = torch.from_numpy(np.ascontiguousarray(src_m[..., idx]))
        for cfg_name in CONFIGS:
            reps = [
                train_and_measure(
                    arm=arm,
                    cfg_name=cfg_name,
                    xw=xw,
                    mw=mw,
                    steps=args.steps,
                    seed=sd,
                    lr=args.lr,
                )
                for sd in seeds
            ]
            r = dict(reps[0])
            sa = [x["sign_agreement_dim0"] for x in reps]
            r["sign_agreement_by_seed"] = sa
            r["sign_agreement_mean"] = float(np.mean(sa))
            r["sign_agreement_sd"] = float(np.std(sa, ddof=1)) if len(sa) > 1 else float("nan")
            r["seeds"] = seeds
            r["any_seed_degenerate"] = any(x["single_bar_degenerate"] for x in reps)
            r["sign_agreement_dim0"] = r["sign_agreement_mean"]
            for k in ("variance_ratio_dim0", "recon_mae_ratio_dim0", "mean_abs_delta_over_sd_dim0"):
                vals = [x[k] for x in reps if x[k] == x[k]]
                r[k] = float(np.mean(vals)) if vals else float("nan")
                r[f"{k}_by_seed"] = [x[k] for x in reps]
            r["single_bar_degenerate"] = r["any_seed_degenerate"]
            results[f"{arm}|{cfg_name}"] = r
            pdr = r["point_decoder_regime"]
            print(
                f"  {arm:14s} {cfg_name:11s} sign_agree={r['sign_agreement_dim0']:.4f}  "
                f"var_ratio={r['variance_ratio_dim0']:8.3f}  "
                f"MAEx{r['recon_mae_ratio_dim0']:6.2f}  "
                f"|d|/sd={r['mean_abs_delta_over_sd_dim0']:.3f}"
                + (f"   [point_decoder sign={pdr['sign_agreement_dim0']:.4f}]" if pdr else ""),
                flush=True,
            )

    # LEG (ii): is the gap symmetric across arms?
    symmetry = {}
    for cfg_name in CONFIGS:
        a = results[f"{ARM_OHLCV}|{cfg_name}"]["sign_agreement_dim0"]
        b = results[f"{ARM_MICRO}|{cfg_name}"]["sign_agreement_dim0"]
        c = results[f"{ARM_MICRO_SHUFFLED}|{cfg_name}"]["sign_agreement_dim0"]
        deg_a = results[f"{ARM_OHLCV}|{cfg_name}"]["single_bar_degenerate"]
        deg_b = results[f"{ARM_MICRO}|{cfg_name}"]["single_bar_degenerate"]
        deg_c = results[f"{ARM_MICRO_SHUFFLED}|{cfg_name}"]["single_bar_degenerate"]

        def _sd(arm: str, _cfg: str = cfg_name) -> float:  # bind the loop var explicitly (B023)
            return float(results[f"{arm}|{_cfg}"].get("sign_agreement_sd", float("nan")))

        sds_pair = [v for v in (_sd(ARM_MICRO), _sd(ARM_MICRO_SHUFFLED)) if v == v]
        sds_cross = [v for v in (_sd(ARM_MICRO), _sd(ARM_OHLCV)) if v == v]
        sd_pair = float(max(sds_pair)) if sds_pair else 0.0
        sd_cross = float(max(sds_cross)) if sds_cross else 0.0
        symmetry[cfg_name] = {
            "sign_agreement_ohlcv_7dim": a,
            "sign_agreement_micro_16dim": b,
            "sign_agreement_micro_shuffled_16dim": c,
            # THE PRIMARY: cells 4 and 5, same width, real vs shuffled micro
            "primary_4_vs_5_abs_difference": abs(b - c),
            "primary_any_arm_degenerate": deg_b or deg_c,
            "primary_within_arm_sd": sd_pair,
            "primary_diff_over_own_spread": (abs(b - c) / sd_pair) if sd_pair > 0 else float("inf"),
            "primary_reading": (
                "INCONCLUSIVE — an arm's single-bar decode collapsed to a constant, so its sign "
                "agreement is an artifact and no symmetry claim is made (control-arm rule)"
                if (deg_b or deg_c)
                # SELF-SCALING (§7 v1.6, supervisor ruling): the yardstick is the statistic's OWN
                # across-seed spread, not an invented band. Same shape as the power guard, which
                # refuses to report a dIR smaller than its inputs' spread. A between-arm gap
                # inside the within-arm noise is not a measured asymmetry.
                else "SYMMETRIC across the PRIMARY pair — the between-arm gap is INSIDE the "
                "within-arm across-seed spread, so it is not distinguishable from measurement "
                "noise; costs POWER, cannot bias dIR(4-5)"
                if sd_pair > 0 and abs(b - c) <= sd_pair
                else "ASYMMETRIC across the PRIMARY pair — the between-arm gap EXCEEDS the "
                "within-arm across-seed spread, so it is a measured difference and does NOT "
                "cancel in dIR(4-5)"
            ),
            # CLAUSE 3 / §5 FALLBACK: these DO compare across widths
            "cross_width_4_vs_2_abs_difference": abs(b - a),
            "cross_width_any_arm_degenerate": deg_a or deg_b,
            "cross_width_within_arm_sd": sd_cross,
            "cross_width_diff_over_own_spread": (
                (abs(b - a) / sd_cross) if sd_cross > 0 else float("inf")
            ),
            "cross_width_reading": (
                "INCONCLUSIVE — a degenerate arm (control-arm rule)"
                if (deg_a or deg_b)
                else "SYMMETRIC across widths — inside the within-arm spread"
                if sd_cross > 0 and abs(b - a) <= sd_cross
                else "WIDTH-DEPENDENT — exceeds the within-arm spread; lands on clause 3's "
                "dIR(4-2) and the §5 fallback dIR(2-1), NOT on the primary dIR(4-5)"
            ),
        }

    doc = {
        "receipt": "m6_c1_pinset_decode",
        "finding": "AUDIT TIER-2 C-1 legs (i) and (ii) — REPORT ONLY",
        "matched_comparison": (
            "data, seed, steps, optimizer, dims and quantizer identical across rows; ONLY the pin "
            "set varies. The first C-1 measurement compared two CHECKPOINTS that differed in "
            "training as well as architecture and so could not isolate the pin."
        ),
        "reference_points": {
            "auditor_toy_d64_2L_sign_agreement": TOY_SIGN_AGREEMENT,
            "c1_leg1_causal_ckpt_no_fine_pointwise": PRIOR_CKPT_SIGN_AGREEMENT,
        },
        "config": {
            "symbol": args.symbol,
            "seq_len": args.seq_len,
            "windows": args.windows,
            "steps": args.steps,
            "lr": args.lr,
            "seeds": seeds,
            "d_model": 256,
            "n_layers": 3,
        },
        "results": results,
        "LEG_II_ARM_SYMMETRY": symmetry,
        "STANDING_FACT": (
            "model.py:104-105 constructs TokenizerDecoder(..., False, ...) — the decoder is "
            "hardcoded BIDIRECTIONAL and window-trained regardless of fine_pointwise, so a "
            "length-1 decode is out of distribution in every row here. fine_pointwise changes "
            "what the ENCODER puts in the fine dims, not what the DECODER does with them."
        ),
        "POINT_DECODER_NOTE": (
            "With fine_pointwise on, model.py:95-102 builds point_decoder, a genuinely per-bar "
            "zero-cross-bar head, whose own comment says 'Train-time only; the AR interface and "
            "the main decode path are unchanged.' predict._rollout calls decode_latent, i.e. the "
            "window decoder. Its numbers are reported as a MEASUREMENT of the gap's nature and "
            "explicitly NOT as a proposal to wire it into the decision path."
        ),
        "SCOPE_LIMIT": (
            "These tokenizers are trained briefly at reduced seq_len for a $0 local comparison. "
            "The CONTRAST between rows is the result; the absolute level is not a claim about a "
            "fully-trained M6 cell."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print("\n=== LEG (ii) ARM SYMMETRY ===")
    for k, v in symmetry.items():
        print(
            f"  {k:11s} ohlcv={v['sign_agreement_ohlcv_7dim']:.4f} "
            f"micro={v['sign_agreement_micro_16dim']:.4f} "
            f"micro_shuf={v['sign_agreement_micro_shuffled_16dim']:.4f}"
        )
        print(
            f"    PRIMARY (4 vs 5, same width) |diff|="
            f"{v['primary_4_vs_5_abs_difference']:.4f}  degenerate="
            f"{v['primary_any_arm_degenerate']}\n      -> {v['primary_reading']}"
        )
        print(
            f"    cross-width (4 vs 2)         |diff|="
            f"{v['cross_width_4_vs_2_abs_difference']:.4f}  degenerate="
            f"{v['cross_width_any_arm_degenerate']}\n      -> {v['cross_width_reading']}"
        )
    print(f"\n[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
