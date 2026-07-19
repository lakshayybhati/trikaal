"""M6 pre-flight Item 2 CANARY — the machine MEASURES, at REAL dims, to CONVERGENCE.

    PYTHONPATH=src python3 scripts/m6_canary.py --device cuda            # the box run (canonical)
    PYTHONPATH=src python3 scripts/m6_canary.py --toy-shells             # cpu machinery validation

Supervisor ruling (2026-07-19, option (i) after the toy-scale STOP): the canary runs at REAL
canonical dims (d512 backbone, canonical vocabs) on the 4090, on a ≥200k-bar synthetic stream
(the 26k toy's median-1 joint-id sparsity was a measured confound), with CONVERGENCE-BASED
stopping — train until val-loss improvement < epsilon over K consecutive evals (defaults
epsilon=0.005 nats, K=5; hard cap --max-steps) — and every eval logs the committed probes
(teacher-forced corr(x̂0_next, state) + h=2 rollout corr) so the TRAJECTORY, not an endpoint,
discriminates under-convergence from inability. Toy lr is permitted here (TOY-ONLY: the real
run's schedule stays the orchestrator default, identical across cells).

Arms and pass conditions (unchanged):
(a) planted — one micro channel (dim 9) carries an iid state that reaches returns at lag 2
    (inside y[t]; structurally invisible to OHLCV/placebo filtrations): must DETECT
    (probe corr rises materially toward the 0.95 ceiling; ΔIR ordering; paired CI of
    ΔIR(4−5) clears 0) with placebo neutrality (IR(5) ≈ IR(2));
(b) noise — must NOT fire, placebo neutral;
(c) micro-dim recon through Cell 4's tokenizer non-degenerate (signal dim + block mean) and
    codebook diagnostics non-collapsed for both quantizers.

IF DETECTION STILL FAILS WITH A CONVERGED CURVE AT REAL DIMS: FULL STOP — that is a
pre-spend architectural finding; nothing further runs (supervisor ruling, binding).

Verdicts come from the SAME instruments as the real decision (paired_delta_ir_bootstrap on
score_cell's pooled headline series). Writes ``<out>/canary_manifest.json`` with the full
per-eval trajectories (durable, never stdout-only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from trikaal.constants import N_FEATURES
from trikaal.data.universe_loader import (
    MultiSymbolWindowSampler,
    build_symbol_windows,
    calendar_boundary_ms,
)
from trikaal.eval.paired_bootstrap import paired_delta_ir_bootstrap
from trikaal.eval.predict import predict_mu
from trikaal.eval.xsection import SymbolEval, XSectionConfig, score_cell
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARMS
from trikaal.train.cells import (
    CELLS,
    assert_cells_parity,
    build_cell_backbone,
    build_cell_tokenizer,
)
from trikaal.train.checkpoint import load_checkpoint, save_checkpoint
from trikaal.train.token_stream import tokenize_features
from trikaal.utils.seeding import set_determinism

SYMBOLS = ("SYNAUSDT", "SYNBUSDT", "SYNCUSDT")
# ≥200k bars (supervisor spec): kills the joint-id one-shot sparsity measured at 26k
# (24,695 distinct pairs / 26k tokens, median occurrence 1) and gives the eval grid T≈4,000
N_BARS = 200_000
WINDOW = ("2024-01-01", "2024-05-19")
TRAIN_FRAC = 0.7
SEQ_LEN = 32
SIGMA = 0.01  # constant per-bar vol → x0 = r/σ (standardized ret_close)
SIGNAL_LAG = 2  # state_t first touches r at t+2 — inside y[t], outside the OHLCV filtration
C_SIGNAL = 3.0  # r_{t+2} = C·σ·state_t + σ·eps — per-bar corr(x0_{t+2}, state_t) ≈ 0.95
T0_MS = 1_704_067_200_000  # 2024-01-01 UTC
TRAIN_HOLD_BAR = 130_000  # train draw < here; [here, true boundary) = the held-out VAL slice
TOY_LR = 1e-3  # TOY-ONLY (supervisor): the real run keeps the orchestrator default schedule

TOY_TOK_KW = dict(d_model=48, n_layers=1, n_heads=2, d_ff=96, max_len=64)
TOY_BB_KW = dict(
    n_layers=2,
    d_model=48,
    d_ff=96,
    n_heads=2,
    mtp_depths=0,
    ffn_dropout=0.0,
    resid_dropout=0.0,
    attn_dropout=0.0,
    token_dropout=0.0,
)


def synth_symbol(rng: np.random.Generator, *, planted: bool) -> dict:
    """One synthetic symbol: micro dim 9 = an iid state that (if planted) drives r two bars on.

    Causal AND asymmetric by construction: state_t is known at the close of bar t (dim 9) and
    first moves the price at bar t+SIGNAL_LAG — inside the entry-next-bar label
    y[t] = log(C_{t+h}/C_{t+1}) but invisible to any function of returns available at t
    (iid states ⇒ past returns reveal only states ≤ t−SIGNAL_LAG, orthogonal to y[t])."""
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


def _batches_from(sampler, sym_tokens, batch, seq_len, rng):
    """One stage-2 batch through the sampler's two-stage draw over pre-tokenized streams."""
    sym_ids = rng.choice(len(sampler.per_symbol), size=batch, p=sampler.weights)
    bc_l, bf_l, ts_l = [], [], []
    for si in sym_ids:
        sw = sampler.per_symbol[int(si)]
        b_c, b_f, ts, starts = sym_tokens[sw.symbol]
        s = int(starts[rng.integers(0, starts.size)])
        bc_l.append(b_c[s : s + seq_len])
        bf_l.append(b_f[s : s + seq_len])
        ts_l.append(ts[s : s + seq_len])
    return (
        torch.from_numpy(np.stack(bc_l).astype(np.int64)),
        torch.from_numpy(np.stack(bf_l).astype(np.int64)),
        torch.from_numpy(np.stack(ts_l).astype(np.int64)),
    )


def _probes(model, tok, d: dict, b_c, b_f, *, device: str) -> dict:
    """The committed probes on the held-out VAL slice: teacher-forced corr + h=2 rollout corr."""
    lo = TRAIN_HOLD_BAR + SEQ_LEN
    starts = np.arange(lo, lo + 120 * SEQ_LEN, SEQ_LEN)
    bc = torch.from_numpy(np.stack([b_c[s : s + SEQ_LEN] for s in starts])).to(device)
    bf = torch.from_numpy(np.stack([b_f[s : s + SEQ_LEN] for s in starts])).to(device)
    ts = torch.from_numpy(np.stack([d["ts"][s : s + SEQ_LEN] for s in starts])).to(device)
    model.eval()
    with torch.no_grad():
        o = model(bc, bf, ts)
        xh = tok.decode_tokens(o.logits_c.argmax(-1), o.logits_f.argmax(-1))
    px0 = xh[:, 1:-1, 0].float().cpu().numpy().ravel()
    st = np.stack([d["x"][s : s + SEQ_LEN, 9] for s in starts])[:, 0:-2].ravel()
    tf_corr = float(np.corrcoef(px0, st)[0, 1])
    dec = np.arange(lo, lo + 12_000, 25, dtype=np.int64)
    mu2 = predict_mu(
        model, tok, b_c, b_f, d["ts"], d["sigma"], dec, h=2, seq_len=SEQ_LEN, device=device
    )
    r2_corr = float(np.corrcoef(mu2, d["x"][dec, 9])[0, 1])
    return {"teacher_forced_corr": tf_corr, "rollout_h2_corr": r2_corr}


def train_cell_convergent(spec, per_symbol_by_arm, raw, args, out_dir: Path) -> dict:
    """Stage-1 (fixed steps) → stage-2 to CONVERGENCE (val-loss plateau) with per-eval probes.

    Plateau rule (stated per supervisor spec): stop when best-val-loss has not improved by
    > --epsilon nats for --k-plateau consecutive evals (evals every --eval-every steps), with
    the --max-steps hard cap. Constant TOY_LR with linear warmup — toy-only."""
    set_determinism(args.seed, deterministic_algorithms=False)
    dev = args.device
    run_dir = out_dir / f"{spec.name}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tok_kw = {} if args.canonical else dict(TOY_TOK_KW)
    tok = build_cell_tokenizer(spec, **tok_kw).to(dev)

    per_symbol = per_symbol_by_arm[spec.arm]
    sampler = MultiSymbolWindowSampler(per_symbol, alpha=0.5, seed=args.seed)

    # ---- stage 1: tokenizer, fixed steps (recon convergence is fast; fidelity is checked) --
    opt = torch.optim.AdamW(tok.parameters(), lr=TOY_LR)
    for step in range(1, args.stage1_steps + 1):
        xb, mb, _ = sampler.sample_batch(args.batch, SEQ_LEN)
        xt = torch.from_numpy(xb).to(dev)
        mt = torch.from_numpy(mb.astype(np.float32)).to(dev)
        tok.train()
        opt.zero_grad(set_to_none=True)
        loss = tok(xt, mt)["loss"]
        loss.backward()
        opt.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{spec.name}: stage-1 loss non-finite at step {step}")
    save_checkpoint(run_dir / "tokenizer.pt", tok, tok.get_config())
    tok.eval()

    # ---- tokenize each symbol ONCE with the frozen tokenizer (train + val + eval regions) --
    sym_tokens = {}
    for sw in per_symbol:
        b_c, b_f = tokenize_features(tok, sw.x, sw.mask, sw.segment_id, window=SEQ_LEN, device=dev)
        sym_tokens[sw.symbol] = (b_c, b_f, sw.ts, sw.starts)

    # ---- stage 2: AR to convergence -------------------------------------------------------
    bb_kw = {} if args.canonical else dict(TOY_BB_KW)
    model = build_cell_backbone(
        tok.v_c,
        tok.v_f,
        expect_base_params=21_301_248 if args.canonical else None,
        max_len=SEQ_LEN + 64,
        **bb_kw,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=TOY_LR, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = 200
    rng = sampler._rng
    d0 = raw[SYMBOLS[0]]
    b_c0, b_f0 = sym_tokens[SYMBOLS[0]][0], sym_tokens[SYMBOLS[0]][1]
    # val batches: windows from the HELD-OUT slice [TRAIN_HOLD_BAR, boundary) — never drawn
    val_starts = np.arange(TRAIN_HOLD_BAR, TRAIN_HOLD_BAR + 96 * SEQ_LEN, SEQ_LEN)
    val = [
        (
            torch.from_numpy(np.stack([sym_tokens[s][0][i : i + SEQ_LEN] for i in val_starts])),
            torch.from_numpy(np.stack([sym_tokens[s][1][i : i + SEQ_LEN] for i in val_starts])),
            torch.from_numpy(np.stack([raw[s]["ts"][i : i + SEQ_LEN] for i in val_starts])),
        )
        for s in SYMBOLS
    ]

    def val_loss() -> float:
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for bc, bf, ts in val:
                tot += float(model.compute_loss(bc.to(dev), bf.to(dev), ts.to(dev))["nll_main"])
        return tot / len(val)

    probe_cell = spec.arm in ("micro", "micro_shuffled")
    best, stall, step, trajectory = float("inf"), 0, 0, []
    t0 = time.time()
    while step < args.max_steps:
        for g in opt.param_groups:
            g["lr"] = TOY_LR * min(1.0, (step + 1) / warmup)
        bc, bf, ts = _batches_from(sampler, sym_tokens, args.batch, SEQ_LEN, rng)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = model.compute_loss(bc.to(dev), bf.to(dev), ts.to(dev))["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if not torch.isfinite(loss):
            raise RuntimeError(f"{spec.name}: stage-2 loss non-finite at step {step}")
        if step % args.eval_every == 0:
            vl = val_loss()
            entry = {"step": step, "val_nll": round(vl, 4)}
            if probe_cell:
                entry.update(
                    {
                        k: round(v, 4)
                        for k, v in _probes(model, tok, d0, b_c0, b_f0, device=dev).items()
                    }
                )
            trajectory.append(entry)
            print(f"    [{spec.name}] {entry}")
            if best - vl > args.epsilon:
                best, stall = vl, 0
            else:
                stall += 1
                if stall >= args.k_plateau:
                    break
    converged = stall >= args.k_plateau
    save_checkpoint(run_dir / "predictor.pt", model, model.get_config())
    print(
        f"    [{spec.name}] stopped at step {step} "
        f"({'PLATEAU (converged)' if converged else 'HARD CAP — not converged'}), "
        f"best val {best:.4f}, wall {time.time() - t0:.0f}s"
    )
    return {
        "cell": spec.name,
        "steps": step,
        "converged_by_plateau": converged,
        "best_val_nll": best,
        "trajectory": trajectory,
        "stopping_rule": {
            "epsilon_nats": args.epsilon,
            "k_plateau": args.k_plateau,
            "eval_every": args.eval_every,
            "max_steps": args.max_steps,
            "toy_lr": TOY_LR,
        },
    }


def _micro_recon_contribution(out_dir: Path, raw: dict, *, device: str) -> dict:
    """Per-dim recon MAE of micro dims 7-12 through Cell 4's trained tokenizer vs the
    predict-the-mean baseline (signal dim + block mean; iid fillers are incompressible and
    the hierarchy rationally triages bits away from them)."""
    ck = load_checkpoint(out_dir / "cell4_fsq_micro_seed0" / "tokenizer.pt", TokenizerAE)
    tok = ck.model.to(device).eval()
    d = raw[SYMBOLS[0]]
    n_win = min(128, d["x"].shape[0] // SEQ_LEN)
    n = n_win * SEQ_LEN
    x = torch.from_numpy(d["x"][:n].reshape(n_win, SEQ_LEN, -1)).to(device)
    m = torch.from_numpy(d["mask"][:n].astype(np.float32).reshape(n_win, SEQ_LEN, -1)).to(device)
    with torch.no_grad():
        cidx, fidx = tok.encode_tokens(x, m)
        x_hat = tok.decode_tokens(cidx, fidx)
    per_dim = {}
    for dim in range(7, 13):
        truth = d["x"][:n, dim].astype(np.float64)
        rec = x_hat[:, :, dim].reshape(-1).float().cpu().numpy().astype(np.float64)
        mae = float(np.abs(rec - truth).mean())
        baseline = float(np.abs(truth - truth.mean()).mean())
        per_dim[str(dim)] = {
            "recon_mae": mae,
            "mean_baseline_mae": baseline,
            "non_degenerate": bool(mae < 0.9 * baseline),
        }
    return per_dim


def run_arm(arm_name: str, *, planted: bool, args, boundary: int) -> dict:
    """Train the 5 cells to convergence on one arm; score through the REAL decision path."""
    t0 = time.time()
    raw = {
        s: synth_symbol(np.random.default_rng((zlib.crc32(arm_name.encode()), i)), planted=planted)
        for i, s in enumerate(SYMBOLS)
    }
    hold_ms = T0_MS + TRAIN_HOLD_BAR * 60_000  # the sampler boundary — reserves the VAL slice
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
                boundary_ms=hold_ms,
                seed=args.seed,
            )
            for s, d in raw.items()
        ]
        for arm in ARMS
    }
    out_dir = Path(args.out) / arm_name
    training = {}
    for spec in CELLS:
        print(f"  [{arm_name}] training {spec.name} to convergence…")
        training[spec.cell_id] = train_cell_convergent(spec, per_symbol_by_arm, raw, args, out_dir)

    xcfg = XSectionConfig(
        window_start=WINDOW[0],
        window_end=WINDOW[1],
        train_frac=TRAIN_FRAC,
        h=15,
        seq_len=SEQ_LEN,
        cap_per_symbol=None,
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

    pb45 = paired_delta_ir_bootstrap(scores[4].headline_series, scores[5].headline_series, h=xcfg.h)
    pb52 = paired_delta_ir_bootstrap(scores[5].headline_series, scores[2].headline_series, h=xcfg.h)
    fired = bool(pb45.ci_lower > 0.0)
    neutral = bool(
        (pb52.ci_lower <= 0.0 <= pb52.ci_upper) or (abs(pb52.delta_ir) <= 0.5 * abs(pb45.delta_ir))
    )
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
        "training": training,
        "all_cells_converged": all(t["converged_by_plateau"] for t in training.values()),
        "cell_ir": {str(c): float(s.ir_headline) for c, s in scores.items()},
        "delta_info": float(pb45.delta_ir),
        "pb_4_minus_5": {"ci_lower": pb45.ci_lower, "ci_upper": pb45.ci_upper, "fired": fired},
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--toy-shells",
        action="store_true",
        help="cpu validation of the MACHINERY (tiny dims, short caps) — detection not expected",
    )
    ap.add_argument("--stage1-steps", type=int, default=3000)
    ap.add_argument("--max-steps", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--epsilon", type=float, default=0.005, help="plateau epsilon (nats)")
    ap.add_argument("--k-plateau", type=int, default=5, help="consecutive non-improving evals")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/m6_canary")
    args = ap.parse_args()
    args.canonical = not args.toy_shells
    if args.toy_shells:
        args.stage1_steps = min(args.stage1_steps, 300)
        args.max_steps = min(args.max_steps, 600)
        args.eval_every = min(args.eval_every, 100)
    boundary = calendar_boundary_ms(*WINDOW, TRAIN_FRAC)
    dims = "CANONICAL d512" if args.canonical else "toy shells (machinery validation only)"
    print(f"=== M6 CANARY — convergence-based, {dims}, device={args.device} ===")
    print(
        f"stopping: val-loss improvement < {args.epsilon} nats over {args.k_plateau} "
        f"consecutive evals (every {args.eval_every} steps), cap {args.max_steps}; "
        f"toy lr {TOY_LR} (TOY-ONLY)"
    )
    if args.canonical:
        parity = assert_cells_parity()
        print(f"[parity] canonical FSQ/BSQ arm parity asserted: {parity}")

    planted = run_arm("planted", planted=True, args=args, boundary=boundary)
    noise = run_arm("noise", planted=False, args=args, boundary=boundary)

    verdicts = {
        "planted_fires": planted["pb_4_minus_5"]["fired"],
        "planted_placebo_neutral": planted["pb_5_minus_2"]["placebo_neutral"],
        "planted_all_cells_converged": planted["all_cells_converged"],
        "noise_does_not_fire": not noise["pb_4_minus_5"]["fired"],
        "noise_placebo_neutral": noise["pb_5_minus_2"]["placebo_neutral"],
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
        "canary": "m6_preflight_item2_convergence",
        "arms": {"planted": planted, "noise": noise},
        "verdicts": verdicts,
        "PASS": ok,
        "recipe": {
            "canonical_dims": args.canonical,
            "signal_lag": SIGNAL_LAG,
            "c_signal": C_SIGNAL,
            "sigma": SIGMA,
            "n_bars": N_BARS,
            "train_hold_bar": TRAIN_HOLD_BAR,
            "symbols": list(SYMBOLS),
            "seq_len": SEQ_LEN,
            "seed": args.seed,
            "device": args.device,
        },
    }
    out = Path(args.out) / "canary_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[canary] verdicts: {verdicts}")
    print(f"[canary] {'PASS' if ok else 'FAIL'} — manifest → {out}")
    if not ok and args.canonical and planted["all_cells_converged"]:
        print(
            "[canary] FULL STOP (supervisor ruling): detection failed with CONVERGED curves "
            "at real dims — pre-spend architectural finding; nothing further runs."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
