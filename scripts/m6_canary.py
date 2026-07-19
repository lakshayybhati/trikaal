"""M6 pre-flight Item 2 CANARY v5 — FIXED budget, REAL schedule, LEVEL-based convergence.

    PYTHONPATH=src python3 scripts/m6_canary.py --cell4-only --device cuda   # Stage-1 (~1.9h)
    PYTHONPATH=src python3 scripts/m6_canary.py --device cuda                # Stage-2 full 10-cell
    PYTHONPATH=src python3 scripts/m6_canary.py --toy-shells                 # cpu machinery check

Supervisor ruling (2026-07-19, from the committed lr{1e-3,3e-4} trajectories): the v3/v4
plateau rule MISFIRED — every "plateau" fired 1,000 steps after a best-val set in the first
200-600 steps, lr3e4 vals were RISING at kill, and each model got ~1% of a real run's
optimization. The plateau rule is therefore DELETED. v5:

- FIXED budget: --steps per cell (default 20,000), no early stop. One pre-authorized
  extension if clearly still descending at the cap = a fresh run at --steps 40000.
- REAL schedule: the orchestrator's own cosine_warmup_lr with its defaults — peak
  3e-4 (stage-2) / 1e-3 (stage-1), warmup_frac 0.1, floor 0.1*peak. Toy-lr is retired.
- Every --eval-every steps (default 500): train-loss mean, val NLL, and the committed
  probes (teacher-forced corr(x_hat0_next, state) + h=2 rollout corr). Full trajectories
  land in the manifest — the trajectory, not an endpoint, is the evidence.
- Convergence is judged by LEVEL, not slope: converged <=> final val NLL <= the d48
  reference D48_REFERENCE_NATS (12.6 — the d48 toy shells reached best val 12.524
  [planted cell4] / 12.496 [cell5] / 12.598 [noise cell4] on the IDENTICAL 200k-bar
  stream within 600 steps; receipt: runs_manifest/m6_canary_d48_reference.json).
  A canonical d512 sitting ABOVE that level is INCONCLUSIVE — never a finding.
- Instability: a val spike > --spike-nats (default 1.0) above best-so-far => halve the
  peak lr ONCE, restart the cell's AR stage from scratch, log it. A second spike is
  recorded and training continues to budget.

Staged execution (supervisor): Stage-1 = planted-arm Cell 4 ONLY (--cell4-only) ->
decision node on the tf_corr trajectory + the val level; only (a) material corr rise
authorizes the full 10-cell Stage-2. (b) val <= 12.6 AND corr flat over the final 5k
steps = GENUINE architectural FULL STOP with converged evidence.

Arms and pass conditions (Stage-2, unchanged):
(a) planted — micro dim 9 carries an iid state reaching returns at lag 2 (inside y[t];
    structurally invisible to OHLCV/placebo filtrations): must DETECT (probe corr rises
    materially toward the 0.95 ceiling; paired CI of dIR(4-5) clears 0) with placebo
    neutrality (IR(5) ~ IR(2));
(b) noise — must NOT fire, placebo neutral;
(c) micro-dim recon through Cell 4's tokenizer non-degenerate and codebook diagnostics
    non-collapsed for both quantizers.

Verdicts come from the SAME instruments as the real decision (paired_delta_ir_bootstrap
on score_cell's pooled headline series). Writes ``<out>/canary_manifest.json``.
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
from trikaal.train.gates import cosine_warmup_lr
from trikaal.train.token_stream import tokenize_features
from trikaal.utils.seeding import set_determinism

SYMBOLS = ("SYNAUSDT", "SYNBUSDT", "SYNCUSDT")
# >=200k bars (supervisor spec): kills the joint-id one-shot sparsity measured at 26k
# (24,695 distinct pairs / 26k tokens, median occurrence 1) and gives the eval grid T~4,000
N_BARS = 200_000
WINDOW = ("2024-01-01", "2024-05-19")
TRAIN_FRAC = 0.7
SEQ_LEN = 32
SIGMA = 0.01  # constant per-bar vol -> x0 = r/sigma (standardized ret_close)
SIGNAL_LAG = 2  # state_t first touches r at t+2 — inside y[t], outside the OHLCV filtration
C_SIGNAL = 3.0  # r_{t+2} = C*sigma*state_t + sigma*eps — per-bar corr(x0_{t+2}, state_t) ~ 0.95
T0_MS = 1_704_067_200_000  # 2024-01-01 UTC
TRAIN_HOLD_BAR = 130_000  # train draw < here; [here, true boundary) = the held-out VAL slice

# The orchestrator-REAL schedule (trikaal.train.orchestrator defaults, v5 ruling: no toy lr)
STAGE1_PEAK_LR = 1e-3
STAGE2_PEAK_LR = 3e-4
WARMUP_FRAC = 0.1  # cosine_warmup_lr floor_frac stays its own default (0.1*peak)

# LEVEL reference (v5): d48 toy shells on the IDENTICAL 200k-bar stream reached best val
# 12.523924827575684 (planted cell4) / 12.495724042256674 (planted cell5) /
# 12.598373095194498 (noise cell4) within 600 steps on cpu.
# Receipt: runs_manifest/m6_canary_d48_reference.json. Threshold set at the supervisor's 12.6.
D48_REFERENCE_NATS = 12.6

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
    (iid states => past returns reveal only states <= t-SIGNAL_LAG, orthogonal to y[t])."""
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


def train_cell_budget(spec, per_symbol_by_arm, raw, args, out_dir: Path) -> dict:
    """Stage-1 tokenizer (fixed steps) -> stage-2 AR on a FIXED --steps budget (v5).

    Both stages run the orchestrator-REAL schedule (cosine_warmup_lr, orchestrator default
    peaks, warmup_frac 0.1). No early stop. Convergence is judged afterward by LEVEL
    (final val NLL <= D48_REFERENCE_NATS); a val spike > --spike-nats above best-so-far
    halves the peak lr ONCE and restarts the AR stage from scratch (logged)."""
    set_determinism(args.seed, deterministic_algorithms=False)
    dev = args.device
    run_dir = out_dir / f"{spec.name}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tok_kw = {} if args.canonical else dict(TOY_TOK_KW)
    tok = build_cell_tokenizer(spec, **tok_kw).to(dev)

    per_symbol = per_symbol_by_arm[spec.arm]
    sampler = MultiSymbolWindowSampler(per_symbol, alpha=0.5, seed=args.seed)

    # ---- stage 1: tokenizer, fixed steps, real schedule (recon convergence is fast) --------
    opt = torch.optim.AdamW(tok.parameters(), lr=STAGE1_PEAK_LR)
    warmup1 = int(WARMUP_FRAC * args.stage1_steps)
    for step in range(1, args.stage1_steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_warmup_lr(step, STAGE1_PEAK_LR, warmup1, args.stage1_steps)
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

    uniform_nats = float(np.log(tok.v_c) + np.log(tok.v_f))
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
    probe_cell = spec.arm in ("micro", "micro_shuffled")
    bb_kw = {} if args.canonical else dict(TOY_BB_KW)

    def _ar_attempt(peak_lr: float, *, abort_on_spike: bool) -> dict:
        """One from-scratch AR run over the full budget; returns the attempt record.

        abort_on_spike=True: the first val spike > --spike-nats above best aborts the
        attempt (the caller restarts once at peak/2). False: spikes are recorded and
        training continues to budget."""
        set_determinism(args.seed, deterministic_algorithms=False)
        model = build_cell_backbone(
            tok.v_c,
            tok.v_f,
            expect_base_params=21_301_248 if args.canonical else None,
            max_len=SEQ_LEN + 64,
            **bb_kw,
        ).to(dev)
        opt = torch.optim.AdamW(
            model.parameters(), lr=peak_lr, weight_decay=0.01, betas=(0.9, 0.95)
        )
        warmup2 = int(WARMUP_FRAC * args.steps)

        def val_loss() -> float:
            model.eval()
            tot = 0.0
            with torch.no_grad():
                for bc, bf, ts in val:
                    tot += float(model.compute_loss(bc.to(dev), bf.to(dev), ts.to(dev))["nll_main"])
            return tot / len(val)

        best = float("inf")
        trajectory, spikes = [], []
        train_acc, train_n = 0.0, 0
        t0 = time.time()
        for step in range(1, args.steps + 1):
            lr_now = cosine_warmup_lr(step, peak_lr, warmup2, args.steps)
            for g in opt.param_groups:
                g["lr"] = lr_now
            bc, bf, ts = _batches_from(sampler, sym_tokens, args.batch, SEQ_LEN, rng)
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = model.compute_loss(bc.to(dev), bf.to(dev), ts.to(dev))["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{spec.name}: stage-2 loss non-finite at step {step}")
            train_acc += float(loss.detach())
            train_n += 1
            if step % args.eval_every == 0:
                vl = val_loss()
                entry = {
                    "step": step,
                    "lr": round(lr_now, 8),
                    "train_nll": round(train_acc / max(1, train_n), 4),
                    "val_nll": round(vl, 4),
                }
                train_acc, train_n = 0.0, 0
                if probe_cell:
                    entry.update(
                        {
                            k: round(v, 4)
                            for k, v in _probes(model, tok, d0, b_c0, b_f0, device=dev).items()
                        }
                    )
                trajectory.append(entry)
                print(f"    [{spec.name}] {entry}", flush=True)
                if vl > best + args.spike_nats:
                    spikes.append({"step": step, "val_nll": round(vl, 4), "best": round(best, 4)})
                    print(
                        f"    [{spec.name}] VAL SPIKE at step {step}: {vl:.4f} > "
                        f"best {best:.4f} + {args.spike_nats}",
                        flush=True,
                    )
                    if abort_on_spike:
                        return {
                            "peak_lr": peak_lr,
                            "aborted_by_spike": True,
                            "spikes": spikes,
                            "trajectory": trajectory,
                            "wall_s": round(time.time() - t0, 1),
                        }
                best = min(best, vl)
        save_checkpoint(run_dir / "predictor.pt", model, model.get_config())
        final_val = trajectory[-1]["val_nll"] if trajectory else float("nan")
        return {
            "peak_lr": peak_lr,
            "aborted_by_spike": False,
            "spikes": spikes,
            "steps": args.steps,
            "final_val_nll": final_val,
            "best_val_nll": round(best, 4),
            "trajectory": trajectory,
            "wall_s": round(time.time() - t0, 1),
        }

    attempts = [_ar_attempt(STAGE2_PEAK_LR, abort_on_spike=True)]
    if attempts[0]["aborted_by_spike"]:
        print(
            f"    [{spec.name}] RESTART (v5 instability rule): halving peak lr "
            f"{STAGE2_PEAK_LR} -> {STAGE2_PEAK_LR / 2} and re-running the cell",
            flush=True,
        )
        attempts.append(_ar_attempt(STAGE2_PEAK_LR / 2, abort_on_spike=False))
    fin = attempts[-1]
    converged = bool(fin["final_val_nll"] <= D48_REFERENCE_NATS)
    print(
        f"    [{spec.name}] budget done: final val {fin['final_val_nll']:.4f} "
        f"(best {fin['best_val_nll']:.4f}, uniform {uniform_nats:.3f}) -> "
        f"{'CONVERGED by level' if converged else 'INCONCLUSIVE (above d48 reference)'} "
        f"[threshold {D48_REFERENCE_NATS}], wall {fin['wall_s']:.0f}s",
        flush=True,
    )
    return {
        "cell": spec.name,
        "steps": fin.get("steps", args.steps),
        "final_val_nll": fin["final_val_nll"],
        "best_val_nll": fin["best_val_nll"],
        "uniform_nll_nats": round(uniform_nats, 4),
        "converged_by_level": converged,
        "lr_halved": len(attempts) > 1,
        "attempts": attempts,
        "trajectory": fin["trajectory"],
        "rule": {
            "budget_steps": args.steps,
            "eval_every": args.eval_every,
            "schedule": "cosine_warmup_lr (orchestrator default)",
            "peak_lr_stage2": STAGE2_PEAK_LR,
            "peak_lr_stage1": STAGE1_PEAK_LR,
            "warmup_frac": WARMUP_FRAC,
            "spike_nats": args.spike_nats,
            "level_threshold_nats": D48_REFERENCE_NATS,
            "d48_receipts": {
                "planted_cell4_best_val": 12.523924827575684,
                "planted_cell5_best_val": 12.495724042256674,
                "noise_cell4_best_val": 12.598373095194498,
                "source": "runs_manifest/m6_canary_d48_reference.json "
                "(d48 toy shells, identical 200k-bar stream, 600-step cap, cpu)",
            },
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


def _build_arm_inputs(arm_name: str, *, planted: bool, args) -> tuple[dict, dict]:
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
    return raw, per_symbol_by_arm


def run_arm(arm_name: str, *, planted: bool, args) -> dict:
    """Train the 5 cells on the fixed budget on one arm; score through the REAL decision path."""
    t0 = time.time()
    raw, per_symbol_by_arm = _build_arm_inputs(arm_name, planted=planted, args=args)
    out_dir = Path(args.out) / arm_name
    training = {}
    for spec in CELLS:
        print(f"  [{arm_name}] training {spec.name} on the fixed budget…", flush=True)
        training[spec.cell_id] = train_cell_budget(spec, per_symbol_by_arm, raw, args, out_dir)

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
            f"κ*={scores[spec.cell_id].kappa_chosen} act={scores[spec.cell_id].activity:.2f}",
            flush=True,
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
        "all_cells_converged": all(t["converged_by_level"] for t in training.values()),
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


def run_stage1_cell4(args) -> dict:
    """Stage-1 (v5 staged execution): planted-arm Cell 4 ONLY — the decision-node run."""
    raw, per_symbol_by_arm = _build_arm_inputs("planted", planted=True, args=args)
    out_dir = Path(args.out) / "planted"
    spec4 = next(s for s in CELLS if s.cell_id == 4)
    print("  [stage1] training cell4_fsq_micro (planted arm) on the fixed budget…", flush=True)
    training = train_cell_budget(spec4, per_symbol_by_arm, raw, args, out_dir)
    recon = _micro_recon_contribution(out_dir, raw, device=args.device)

    traj = training["trajectory"]
    tf = [e["teacher_forced_corr"] for e in traj if "teacher_forced_corr" in e]
    last_5k_evals = max(1, 5000 // args.eval_every)
    tail = tf[-last_5k_evals:] if tf else []
    decision_inputs = {
        "final_val_nll": training["final_val_nll"],
        "best_val_nll": training["best_val_nll"],
        "converged_by_level": training["converged_by_level"],
        "level_threshold_nats": D48_REFERENCE_NATS,
        "tf_corr_first": tf[0] if tf else None,
        "tf_corr_max": max(tf) if tf else None,
        "tf_corr_final": tf[-1] if tf else None,
        "tf_corr_last5k_min": min(tail) if tail else None,
        "tf_corr_last5k_max": max(tail) if tail else None,
        "micro_recon_dim9_non_degenerate": recon["9"]["non_degenerate"],
    }
    print(f"[stage1] decision inputs: {decision_inputs}", flush=True)
    return {
        "canary": "m6_preflight_item2_v5",
        "stage": "stage1_cell4_only",
        "training": training,
        "micro_recon": recon,
        "decision_inputs": decision_inputs,
        "decision_node": (
            "(a) tf_corr rises materially -> full 10-cell Stage-2; "
            "(b) val <= d48 reference AND corr flat over final 5k steps -> "
            "GENUINE architectural FULL STOP with converged evidence; "
            "(c) still descending at cap, corr ~0 -> ONE pre-authorized extension "
            "(fresh run, --steps 40000) then (a)/(b); "
            "(d) spike-restart already applied automatically if it occurred"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--toy-shells",
        action="store_true",
        help="cpu validation of the MACHINERY (tiny dims, short caps) — detection not expected",
    )
    ap.add_argument(
        "--cell4-only",
        action="store_true",
        help="Stage-1 (v5): planted-arm Cell 4 only — the staged decision-node run",
    )
    ap.add_argument("--stage1-steps", type=int, default=3000)
    ap.add_argument(
        "--steps",
        type=int,
        default=20_000,
        help="FIXED stage-2 budget per cell (v5; extension = a fresh run at 40000)",
    )
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument(
        "--spike-nats",
        type=float,
        default=1.0,
        help="val spike above best-so-far that triggers the one lr-halving restart",
    )
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/m6_canary")
    args = ap.parse_args()
    args.canonical = not args.toy_shells
    if args.toy_shells:
        args.stage1_steps = min(args.stage1_steps, 300)
        args.steps = min(args.steps, 600)
        args.eval_every = min(args.eval_every, 100)
    dims = "CANONICAL d512" if args.canonical else "toy shells (machinery validation only)"
    print(f"=== M6 CANARY v5 — fixed budget, real schedule, {dims}, device={args.device} ===")
    print(
        f"budget {args.steps} steps/cell (NO early stop; plateau rule deleted per supervisor "
        f"adjudication of the lr1e3/lr3e4 trajectories); schedule cosine_warmup_lr peak "
        f"{STAGE2_PEAK_LR} warmup_frac {WARMUP_FRAC}; evals every {args.eval_every}; "
        f"converged <=> final val <= {D48_REFERENCE_NATS} nats (d48 reference); "
        f"spike rule: > {args.spike_nats} nats above best -> halve lr once, restart, log"
    )
    if args.canonical:
        parity = assert_cells_parity()
        print(f"[parity] canonical FSQ/BSQ arm parity asserted: {parity}")

    if args.cell4_only:
        manifest = run_stage1_cell4(args)
        ok = True  # stage-1 has no PASS/FAIL — it feeds the decision node
    else:
        planted = run_arm("planted", planted=True, args=args)
        noise = run_arm("noise", planted=False, args=args)
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
            "canary": "m6_preflight_item2_v5",
            "stage": "full_10_cell",
            "arms": {"planted": planted, "noise": noise},
            "verdicts": verdicts,
            "PASS": ok,
        }
        print(f"[canary] verdicts: {verdicts}")
        if not ok and args.canonical and planted["all_cells_converged"]:
            print(
                "[canary] FULL STOP (supervisor ruling): detection failed with curves "
                "CONVERGED BY LEVEL at real dims — pre-spend architectural finding; "
                "nothing further runs."
            )
        elif not ok and args.canonical:
            print(
                "[canary] INCONCLUSIVE — detection failed but val never reached the d48 "
                "reference level; NOT a finding (v5 anti-masquerade rule). Escalate."
            )

    manifest["recipe"] = {
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
        "budget_steps": args.steps,
        "eval_every": args.eval_every,
        "schedule": {
            "fn": "cosine_warmup_lr",
            "peak_lr_stage1": STAGE1_PEAK_LR,
            "peak_lr_stage2": STAGE2_PEAK_LR,
            "warmup_frac": WARMUP_FRAC,
            "floor_frac": 0.1,
        },
        "level_threshold_nats": D48_REFERENCE_NATS,
        "spike_nats": args.spike_nats,
    }
    out = Path(args.out) / "canary_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[canary] manifest → {out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
