"""M6 pre-flight Item 2 CANARY v6 — SUB-EPOCH stream, NOISE-ARM yardstick, exhaustive branches.

    PYTHONPATH=src python3 scripts/m6_canary.py --stage1 --device cuda      # Stage-1 (two runs)
    PYTHONPATH=src python3 scripts/m6_canary.py --device cuda               # Stage-2 full 10-cell
    PYTHONPATH=src python3 scripts/m6_canary.py --toy-shells --stage1       # cpu machinery check
    # pre-authorized (c) extension = fresh runs, BOTH arms, sub-epoch preserved:
    PYTHONPATH=src python3 scripts/m6_canary.py --stage1 --steps 40000 --bars-total 84000000

Supervisor ruling (v6, after the v5 INCONCLUSIVE): v5's regime was mismatched — best-val at
the FIRST evals proves memorization dominated from step 1 on the 200k-bar stream, while the
REAL M6 run is sub-epoch on 285M bars where memorization is impossible. v6 makes the canary
faithful. Kept from v5: canonical dims, orchestrator-real schedule (cosine_warmup_lr, peak
3e-4 / warmup_frac 0.1 / floor 0.1*peak), probes+val+train every 500 steps, spike > 1 nat ->
halve-lr-once restart, fixed budget with NO early stopping. New in v6:

1. SUB-EPOCH STREAM: total bars sized so the TRAIN region (HOLD_FRAC of the stream) holds
   >= 1.3x the budget's total bar-visits (steps x batch x seq_len). Exact arithmetic is
   computed, asserted (canonical mode), and recorded in the manifest. Default 42M bars
   (3 symbols x 14M): train region 27.3M >= 1.3 x 20,480,000 = 26,624,000. Same generator,
   same 3-sigma lag-2 planted rule as v5 — only the size changes.
2. THE YARDSTICK IS THE NOISE ARM (the d48 reference is RETIRED): Stage-1 = TWO runs at
   identical budget/schedule/stream-size — planted-arm Cell 4 and noise-arm Cell 4. The
   noise run is the measured no-rule floor. Both full trajectories side by side.
3. BRANCHES (exhaustive):
   (a) DETECT: planted tf_corr >= 0.3 on >= 2 consecutive evals (crossing step recorded)
       AND final val(planted) < val(noise) - 0.1 nats -> Stage-2 = full 10-cell canary at
       ~1.5x the crossing step (driver states cost vs balance first).
   (b) CLEAN ARCHITECTURAL FULL STOP: schedule complete, planted tf_corr < 0.05 at EVERY
       eval, |val(planted) - val(noise)| < 0.1, AND the sub-epoch check holds (final
       val - train nll_main gap < 0.5 nats on BOTH arms — proves no memorization regime)
       -> stop everything, ship both trajectories; the pre-named next step is the
       token-space-planted control, NOT more canary tuning.
   (c) still clearly descending at the cap (val improved > 0.05 nats over the last 5k
       steps, either arm) -> ONE 20k extension, BOTH arms (fresh runs at --steps 40000
       --bars-total 84000000 so sub-epoch still holds), then re-apply (a)/(b).
   (d) spike -> the halve-once restart (automatic, logged, as in v5).
   (e) ANY other signature -> INCONCLUSIVE: report with the trajectories, spend nothing
       further, await ruling.

Stage-2 pass conditions (full 10-cell, unchanged from v5): planted arm fires with placebo
neutrality, noise arm quiet, recon non-degenerate, codebooks non-collapsed — through the
SAME instruments as the real decision (paired_delta_ir_bootstrap on score_cell's pooled
headline series). Writes ``<out>/canary_manifest.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import zlib
from datetime import UTC, datetime, timedelta
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
from trikaal.train.gates import cosine_warmup_lr, digit_onehots, id_legibility_sign_acc
from trikaal.train.token_stream import tokenize_features
from trikaal.utils.seeding import set_determinism

SYMBOLS = ("SYNAUSDT", "SYNBUSDT", "SYNCUSDT")
BATCH = 32
SEQ_LEN = 32
SIGMA = 0.01  # constant per-bar vol -> x0 = r/sigma (standardized ret_close)
SIGNAL_LAG = 2  # state_t first touches r at t+2 — inside y[t], outside the OHLCV filtration
C_SIGNAL = 3.0  # r_{t+2} = C*sigma*state_t + sigma*eps — per-bar corr(x0_{t+2}, state_t) ~ 0.95
T0_MS = 1_704_067_200_000  # 2024-01-01 UTC
HOLD_FRAC = 0.65  # train draws < hold bar; [hold, 0.70) = held-out VAL/probe slice (v5 ratio)
TRAIN_FRAC = 0.7  # Stage-2 scoring boundary (eval region [0.70, 1.0) — after the train region)
SUB_EPOCH_FACTOR = 1.3  # v6 rule: train-region bars >= factor x budget bar-visits

# v6 default: 42M total (3 x 14M). Train region 0.65 x 42M = 27.3M >= 1.3 x 20k x 32 x 32.
BARS_TOTAL_DEFAULT = 42_000_000

# The orchestrator-REAL schedule (trikaal.train.orchestrator defaults; toy-lr retired in v5)
STAGE1_PEAK_LR = 1e-3
STAGE2_PEAK_LR = 3e-4
WARMUP_FRAC = 0.1  # cosine_warmup_lr floor_frac stays its own default (0.1*peak)

# GATE-2 CALIBRATION RIDER (supervisor ruling, 2026-07-20): the fixture's 11 non-state,
# non-return filler dims are CALIBRATED to the real lake's measured non-micro feature
# entropy — a measurement-matched fixture, not a difficulty knob. Measured on 30 symbols /
# 3,895,808 unmasked bars (sample seed 20260720): Gaussian-copula corr logdet -6.4819 over
# the 7 non-micro dims => per-dim entropy deficit 0.4631 nats. The fixture matches it with
# an AR(1)-ACROSS-DIMS chain over the 11 fillers: deficit = (10/22)*ln(1/(1-rho^2)) =>
# rho = 0.7993 (exact by construction; plug-in receipt recorded per run, tolerance 0.05
# nats/dim). Marginals stay N(0,1); bars stay temporally iid; state dim, plant, and return
# rule UNTOUCHED.
FILLER_DIMS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12)
FILLER_RHO = 0.7993
FILLER_TARGET_DEFICIT = 0.4631  # nats/dim, from the lake measurement above
FILLER_DEFICIT_TOL = 0.05

# §7 v1.4.1 fixture legibility gate, RESTATED AS RULED (2026-07-21): over the three
# calibration seeds, mean >= 0.9 AND every seed >= 0.85 (seeded receipts at lambda*=3:
# mean 0.9067, min 0.9000). The
# in-run hard stop below enforces the PER-SEED floor before any AR spend; the mean clause
# is enforced by the formal 3-seed cell-path re-run receipt
# (runs_manifest/m6_lambda_search_receipt.json).
LEGIBILITY_MIN = 0.85

# v6 branch thresholds (supervisor spec, verbatim)
DETECT_TF = 0.3  # (a): >= on 2 consecutive evals
DETECT_VAL_MARGIN = 0.1  # (a): final val(planted) < val(noise) - this
FLAT_TF = 0.05  # (b): planted tf_corr below this at EVERY eval
ARM_VAL_TOL = 0.1  # (b): |val(planted) - val(noise)| below this
MEMO_GAP = 0.5  # (b): final val - train nll_main below this on both arms
DESC_NATS = 0.05  # (c): val improved by more than this over the final 5k steps

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


def filler_calibration_receipt(n: int = 500_000) -> dict:
    """Plug-in Gaussian-copula deficit of a freshly generated filler sample vs the target."""
    rng = np.random.default_rng(20260720)
    g = rng.standard_normal((n, len(FILLER_DIMS)))
    for j in range(1, len(FILLER_DIMS)):
        g[:, j] = FILLER_RHO * g[:, j - 1] + np.sqrt(1.0 - FILLER_RHO**2) * g[:, j]
    z = np.empty_like(g)
    for j in range(g.shape[1]):
        rk = g[:, j].argsort().argsort().astype(np.float64)
        u = (rk + 0.5) / n
        z[:, j] = (torch.erfinv(torch.tensor(2 * u - 1)) * np.sqrt(2)).numpy()
    _sign, logdet = np.linalg.slogdet(np.corrcoef(z, rowvar=False))
    achieved = float(-0.5 * logdet / len(FILLER_DIMS))
    return {
        "lake_measurement": {
            "symbols": 30,
            "unmasked_bars": 3_895_808,
            "dims": "x_0..x_6 (non-micro)",
            "copula_corr_logdet": -6.4819,
            "per_dim_deficit_nats": 0.4631,
            "sample_seed": 20260720,
        },
        "filler_dims": list(FILLER_DIMS),
        "filler_rho": FILLER_RHO,
        "target_per_dim_deficit_nats": FILLER_TARGET_DEFICIT,
        "achieved_per_dim_deficit_nats": round(achieved, 4),
        "tolerance_nats": FILLER_DEFICIT_TOL,
        "within_tolerance": bool(abs(achieved - FILLER_TARGET_DEFICIT) <= FILLER_DEFICIT_TOL),
        "note": "measurement-matched fixture (gate-2 rider): state dim, plant, and return "
        "rule untouched; fillers N(0,1) marginals, cross-dim AR(1), temporally iid",
    }


def sub_epoch_arithmetic(steps: int, bars_total: int) -> dict:
    """The v6 sizing rule, computed once and recorded verbatim in the manifest."""
    n_per = bars_total // len(SYMBOLS)
    total = n_per * len(SYMBOLS)
    visits = steps * BATCH * SEQ_LEN
    required_train_bars = int(np.ceil(SUB_EPOCH_FACTOR * visits))
    train_bars = int(HOLD_FRAC * n_per) * len(SYMBOLS)
    return {
        "rule": "train-region bars >= 1.3 x (steps x batch x seq_len) bar-visits",
        "steps": steps,
        "batch": BATCH,
        "seq_len": SEQ_LEN,
        "bar_visits": visits,
        "required_train_bars": required_train_bars,
        "bars_total": total,
        "n_per_symbol": n_per,
        "hold_frac": HOLD_FRAC,
        "train_bars": train_bars,
        "epochs_equivalent_train_region": round(visits / train_bars, 4),
        "holds": bool(train_bars >= required_train_bars),
    }


def synth_symbol(rng: np.random.Generator, *, planted: bool, n_bars: int) -> dict:
    """One synthetic symbol: micro dim 9 = an iid state that (if planted) drives r two bars on.

    Causal AND asymmetric by construction: state_t is known at the close of bar t (dim 9) and
    first moves the price at bar t+SIGNAL_LAG — inside the entry-next-bar label
    y[t] = log(C_{t+h}/C_{t+1}) but invisible to any function of returns available at t
    (iid states => past returns reveal only states <= t-SIGNAL_LAG, orthogonal to y[t]).
    Identical generator to v5 — v6 changes ONLY the stream size."""
    state = rng.standard_normal(n_bars)
    eps = rng.standard_normal(n_bars)
    r = SIGMA * eps
    if planted:
        r[SIGNAL_LAG:] = r[SIGNAL_LAG:] + C_SIGNAL * SIGMA * state[:-SIGNAL_LAG]
    x = rng.standard_normal((n_bars, N_FEATURES)).astype(np.float32)
    # gate-2 calibration rider: fillers are CROSS-DIM correlated (AR(1) chain, rho matched
    # to the lake's measured non-micro copula entropy), N(0,1) marginals, iid across bars
    g = x[:, list(FILLER_DIMS)].astype(np.float64)
    for j in range(1, len(FILLER_DIMS)):
        g[:, j] = FILLER_RHO * g[:, j - 1] + np.sqrt(1.0 - FILLER_RHO**2) * g[:, j]
    x[:, list(FILLER_DIMS)] = g.astype(np.float32)
    x[:, 0] = (r / SIGMA).astype(np.float32)  # standardized ret_close (return rule UNTOUCHED)
    x[:, 9] = (state if planted else rng.standard_normal(n_bars)).astype(np.float32)
    m = np.zeros((n_bars, N_FEATURES), dtype=np.uint8)
    m[:, 13:16] = 1  # perp dims masked everywhere (the lake convention; placebo tripwire)
    return {
        "x": x,
        "mask": m,
        "segment_id": np.zeros(n_bars, dtype=np.int64),
        "ts": (T0_MS + np.arange(n_bars) * 60_000).astype(np.int64),
        "sigma": np.full(n_bars, SIGMA),
        "raw_ret_close": r.astype(np.float64),
    }


# The §7 v1.4 legibility instruments live in trikaal.train.gates (ONE implementation —
# the orchestrator's standing M6-time gate and this canary share it).
_digit_onehots = digit_onehots


def legibility_probe(tok, state: np.ndarray, b_c: np.ndarray, b_f: np.ndarray) -> float:
    """§7 v1.4 standing gate: logistic sign-accuracy of the state from bar t's OWN id."""
    return id_legibility_sign_acc(tok, state, b_c, b_f)


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


def _probes(model, tok, d: dict, b_c, b_f, hold_bar: int, *, device: str) -> dict:
    """The committed probes on the held-out VAL slice: teacher-forced corr + h=2 rollout corr."""
    lo = hold_bar + SEQ_LEN
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


def train_cell_budget(
    spec, per_symbol_by_arm, raw, args, out_dir: Path, *, legibility_gate: bool = False
) -> dict:
    """Stage-1 tokenizer (fixed steps) -> stage-2 AR on the FIXED --steps budget.

    Both stages run the orchestrator-REAL schedule. No early stop. Every eval logs
    train loss (total AND nll_main — the memorization gap check needs the comparable
    number), val nll_main, and the probes. A val spike > --spike-nats above best-so-far
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
        xb, mb, _ = sampler.sample_batch(BATCH, SEQ_LEN)
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
    t_tok = time.time()
    sym_tokens = {}
    for sw in per_symbol:
        b_c, b_f = tokenize_features(
            tok, sw.x, sw.mask, sw.segment_id, window=SEQ_LEN, batch_windows=512, device=dev
        )
        sym_tokens[sw.symbol] = (b_c, b_f, sw.ts, sw.starts)
    tokenize_wall_s = round(time.time() - t_tok, 1)

    n_per = raw[SYMBOLS[0]]["x"].shape[0]
    hold_bar = int(HOLD_FRAC * n_per)
    uniform_nats = float(np.log(tok.v_c) + np.log(tok.v_f))
    rng = sampler._rng
    d0 = raw[SYMBOLS[0]]
    b_c0, b_f0 = sym_tokens[SYMBOLS[0]][0], sym_tokens[SYMBOLS[0]][1]

    # ---- §7 v1.4 standing PER-BAR LEGIBILITY gate (before ANY AR spend) ---------------------
    legibility = None
    if spec.arm in ("micro", "micro_shuffled"):
        legibility = legibility_probe(tok, d0["x"][:, 9], b_c0, b_f0)
        print(
            f"    [{spec.name}] per-bar id legibility (logistic sign-acc): {legibility:.4f} "
            f"(gate {LEGIBILITY_MIN} {'ENFORCED' if legibility_gate else 'recorded'})",
            flush=True,
        )
        if legibility_gate and args.canonical and legibility < LEGIBILITY_MIN:
            raise SystemExit(
                f"§7 v1.4 LEGIBILITY GATE FAILED before AR spend: {legibility:.4f} < "
                f"{LEGIBILITY_MIN} — the fine subtoken does not carry bar t's state"
            )
    # val batches: windows from the HELD-OUT slice [hold_bar, boundary) — never drawn in train
    val_starts = np.arange(hold_bar, hold_bar + 96 * SEQ_LEN, SEQ_LEN)
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
        """One from-scratch AR run over the full budget; returns the attempt record."""
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
        acc_tot, acc_main, train_n = 0.0, 0.0, 0
        t0 = time.time()
        for step in range(1, args.steps + 1):
            lr_now = cosine_warmup_lr(step, peak_lr, warmup2, args.steps)
            for g in opt.param_groups:
                g["lr"] = lr_now
            bc, bf, ts = _batches_from(sampler, sym_tokens, BATCH, SEQ_LEN, rng)
            model.train()
            opt.zero_grad(set_to_none=True)
            out = model.compute_loss(bc.to(dev), bf.to(dev), ts.to(dev))
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{spec.name}: stage-2 loss non-finite at step {step}")
            acc_tot += float(loss.detach())
            acc_main += float(out["nll_main"].detach())
            train_n += 1
            if step % args.eval_every == 0:
                vl = val_loss()
                entry = {
                    "step": step,
                    "lr": round(lr_now, 8),
                    "train_nll": round(acc_tot / max(1, train_n), 4),
                    "train_nll_main": round(acc_main / max(1, train_n), 4),
                    "val_nll": round(vl, 4),
                }
                acc_tot, acc_main, train_n = 0.0, 0.0, 0
                if probe_cell:
                    entry.update(
                        {
                            k: round(v, 4)
                            for k, v in _probes(
                                model, tok, d0, b_c0, b_f0, hold_bar, device=dev
                            ).items()
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
            f"    [{spec.name}] RESTART (branch d, halve-once rule): peak lr "
            f"{STAGE2_PEAK_LR} -> {STAGE2_PEAK_LR / 2}, re-running the cell from scratch",
            flush=True,
        )
        attempts.append(_ar_attempt(STAGE2_PEAK_LR / 2, abort_on_spike=False))
    fin = attempts[-1]
    tr = fin["trajectory"]
    gap = round(tr[-1]["val_nll"] - tr[-1]["train_nll_main"], 4) if tr else float("nan")
    print(
        f"    [{spec.name}] budget done: final val {fin['final_val_nll']:.4f} "
        f"(best {fin['best_val_nll']:.4f}, uniform {uniform_nats:.3f}, "
        f"final val-train_main gap {gap:+.4f}), wall {fin['wall_s']:.0f}s",
        flush=True,
    )
    return {
        "cell": spec.name,
        "arm_stream": None,  # filled by the caller (planted / noise)
        "steps": fin.get("steps", args.steps),
        "final_val_nll": fin["final_val_nll"],
        "best_val_nll": fin["best_val_nll"],
        "uniform_nll_nats": round(uniform_nats, 4),
        "final_val_train_main_gap": gap,
        "lr_halved": len(attempts) > 1,
        "attempts": attempts,
        "trajectory": tr,
        "tokenize_wall_s": tokenize_wall_s,
        "per_bar_id_legibility": legibility,
        "rule": {
            "budget_steps": args.steps,
            "eval_every": args.eval_every,
            "schedule": "cosine_warmup_lr (orchestrator default)",
            "peak_lr_stage2": STAGE2_PEAK_LR,
            "peak_lr_stage1": STAGE1_PEAK_LR,
            "warmup_frac": WARMUP_FRAC,
            "spike_nats": args.spike_nats,
            "yardstick": "noise-arm cell4 at identical budget/schedule/stream (v6; "
            "d48 level reference RETIRED)",
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


def _build_arm_inputs(arm_name: str, *, planted: bool, args, arms: tuple[str, ...]) -> tuple:
    """Synthesize one arm's stream and build windows for ONLY the requested training arms
    (select_arm copies columns — at 42M bars every unneeded arm costs ~2GB RAM)."""
    n_per = args.bars_total // len(SYMBOLS)
    raw = {
        s: synth_symbol(
            np.random.default_rng((zlib.crc32(arm_name.encode()), i)),
            planted=planted,
            n_bars=n_per,
        )
        for i, s in enumerate(SYMBOLS)
    }
    hold_ms = T0_MS + int(HOLD_FRAC * n_per) * 60_000  # sampler boundary — reserves VAL slice
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
        for arm in arms
    }
    return raw, per_symbol_by_arm


def run_arm(arm_name: str, *, planted: bool, args) -> dict:
    """Stage-2: train the 5 cells on the fixed budget on one arm-stream; score through the
    REAL decision path."""
    t0 = time.time()
    raw, per_symbol_by_arm = _build_arm_inputs(arm_name, planted=planted, args=args, arms=ARMS)
    out_dir = Path(args.out) / arm_name
    training = {}
    for spec in CELLS:
        print(f"  [{arm_name}] training {spec.name} on the fixed budget…", flush=True)
        rec = train_cell_budget(
            spec,
            per_symbol_by_arm,
            raw,
            args,
            out_dir,
            legibility_gate=(arm_name == "planted" and spec.cell_id == 4),
        )
        rec["arm_stream"] = arm_name
        training[spec.cell_id] = rec

    n_per = args.bars_total // len(SYMBOLS)
    end_dt = datetime.fromtimestamp((T0_MS + n_per * 60_000) / 1000, tz=UTC) + timedelta(days=1)
    xcfg = XSectionConfig(
        window_start="2024-01-01",
        window_end=end_dt.strftime("%Y-%m-%d"),
        train_frac=TRAIN_FRAC,
        h=15,
        seq_len=SEQ_LEN,
        cap_per_symbol=args.eval_cap,
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


def _descending_at_cap(trajectory: list[dict], eval_every: int) -> bool:
    """(c): val still clearly descending at the cap — improved > DESC_NATS over the last
    5,000 steps."""
    back = max(1, 5000 // eval_every)
    if len(trajectory) <= back:
        return False
    return trajectory[-1]["val_nll"] < trajectory[-1 - back]["val_nll"] - DESC_NATS


def stage1_decision(planted: dict, noise: dict, sub_epoch: dict, args) -> dict:
    """The v6 exhaustive branch rule, computed mechanically from the two trajectories."""
    tr_p, tr_n = planted["trajectory"], noise["trajectory"]
    tf = [e["teacher_forced_corr"] for e in tr_p if "teacher_forced_corr" in e]
    cross_step = None
    for i in range(len(tf) - 1):
        if tf[i] >= DETECT_TF and tf[i + 1] >= DETECT_TF:
            cross_step = tr_p[i]["step"]
            break
    fvp, fvn = planted["final_val_nll"], noise["final_val_nll"]
    schedule_complete = not (
        planted["attempts"][-1]["aborted_by_spike"] or noise["attempts"][-1]["aborted_by_spike"]
    )
    tf_flat_every_eval = bool(tf) and all(t < FLAT_TF for t in tf)
    arms_val_close = bool(abs(fvp - fvn) < ARM_VAL_TOL)
    no_memorization = bool(
        sub_epoch["holds"]
        and planted["final_val_train_main_gap"] < MEMO_GAP
        and noise["final_val_train_main_gap"] < MEMO_GAP
    )
    a = bool(cross_step is not None and fvp < fvn - DETECT_VAL_MARGIN)
    b = bool(schedule_complete and tf_flat_every_eval and arms_val_close and no_memorization)
    c = bool(
        (_descending_at_cap(tr_p, args.eval_every) or _descending_at_cap(tr_n, args.eval_every))
        and not a
        and not b
    )
    branch = (
        "a_detect"
        if a
        else "b_architectural_full_stop"
        if b
        else "c_extension"
        if c
        else "e_inconclusive"
    )
    return {
        "branch": branch,
        "detect_cross_step": cross_step,
        "tf_corr_max": max(tf) if tf else None,
        "tf_corr_final": tf[-1] if tf else None,
        "tf_flat_every_eval": tf_flat_every_eval,
        "final_val_planted": fvp,
        "final_val_noise": fvn,
        "val_planted_minus_noise": round(fvp - fvn, 4),
        "arms_val_close": arms_val_close,
        "schedule_complete": schedule_complete,
        "gap_planted": planted["final_val_train_main_gap"],
        "gap_noise": noise["final_val_train_main_gap"],
        "no_memorization_regime": no_memorization,
        "descending_at_cap_planted": _descending_at_cap(tr_p, args.eval_every),
        "descending_at_cap_noise": _descending_at_cap(tr_n, args.eval_every),
        "lr_halved": {"planted": planted["lr_halved"], "noise": noise["lr_halved"]},
        "thresholds": {
            "detect_tf": DETECT_TF,
            "detect_val_margin": DETECT_VAL_MARGIN,
            "flat_tf": FLAT_TF,
            "arm_val_tol": ARM_VAL_TOL,
            "memo_gap": MEMO_GAP,
            "desc_nats": DESC_NATS,
        },
    }


def run_stage1(args) -> dict:
    """v6 Stage-1: planted-arm Cell 4 AND noise-arm Cell 4 at identical budget/schedule/
    stream size. The noise run is the yardstick."""
    spec4 = next(s for s in CELLS if s.cell_id == 4)
    sub_epoch = sub_epoch_arithmetic(args.steps, args.bars_total)
    print(f"[stage1] sub-epoch arithmetic: {sub_epoch}", flush=True)
    if args.canonical and not sub_epoch["holds"]:
        raise SystemExit(
            "v6 sub-epoch rule violated: train region "
            f"{sub_epoch['train_bars']} < required {sub_epoch['required_train_bars']} — "
            "raise --bars-total"
        )

    runs, recon = {}, None
    for arm_name, planted in (("planted", True), ("noise", False)):
        print(f"  [stage1] {arm_name}-arm cell4 run…", flush=True)
        raw, per_symbol_by_arm = _build_arm_inputs(
            arm_name, planted=planted, args=args, arms=(spec4.arm,)
        )
        out_dir = Path(args.out) / arm_name
        rec = train_cell_budget(
            spec4, per_symbol_by_arm, raw, args, out_dir, legibility_gate=(arm_name == "planted")
        )
        rec["arm_stream"] = arm_name
        runs[arm_name] = rec
        if arm_name == "planted":
            recon = _micro_recon_contribution(out_dir, raw, device=args.device)
        del raw, per_symbol_by_arm
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    decision = stage1_decision(runs["planted"], runs["noise"], sub_epoch, args)
    decision["micro_recon_dim9_non_degenerate"] = recon["9"]["non_degenerate"]
    print(f"[stage1] decision: {json.dumps(decision)}", flush=True)
    return {
        "canary": "m6_preflight_item2_v6",
        "stage": "stage1_two_arm_cell4",
        "sub_epoch": sub_epoch,
        "runs": runs,
        "micro_recon": recon,
        "decision_inputs": decision,
        "branch_taken": decision["branch"],
        "branch_semantics": {
            "a_detect": "Stage-2 full 10-cell at ~1.5x the crossing step (cost vs balance "
            "stated by the driver first)",
            "b_architectural_full_stop": "stop everything, ship both trajectories; pre-named "
            "next step = token-space-planted control, NOT more canary tuning",
            "c_extension": "ONE 20k extension BOTH arms (fresh --steps 40000 "
            "--bars-total 84000000), then re-apply (a)/(b)",
            "d_spike": "halve-lr-once restart (automatic, recorded per run as lr_halved)",
            "e_inconclusive": "report with trajectories, spend nothing further, await ruling",
        },
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
        "--stage1",
        action="store_true",
        help="v6 Stage-1: planted-arm cell4 + noise-arm cell4 (the yardstick pair)",
    )
    ap.add_argument("--stage1-steps", type=int, default=3000)
    ap.add_argument(
        "--steps",
        type=int,
        default=20_000,
        help="FIXED stage-2 budget per cell (extension = fresh runs at 40000 + 84M bars)",
    )
    ap.add_argument(
        "--bars-total",
        type=int,
        default=BARS_TOTAL_DEFAULT,
        help="total synthetic bars across the 3 symbols (v6 sub-epoch rule asserts the size)",
    )
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument(
        "--spike-nats",
        type=float,
        default=1.0,
        help="val spike above best-so-far that triggers the one lr-halving restart",
    )
    ap.add_argument(
        "--eval-cap",
        type=int,
        default=4000,
        help="Stage-2 scoring: decision points per symbol (full streams are 14M bars)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/m6_canary")
    args = ap.parse_args()
    args.canonical = not args.toy_shells
    if args.toy_shells:
        args.stage1_steps = min(args.stage1_steps, 300)
        args.steps = min(args.steps, 600)
        args.eval_every = min(args.eval_every, 100)
        args.bars_total = min(args.bars_total, 600_000)
    dims = "CANONICAL d512" if args.canonical else "toy shells (machinery validation only)"
    print(f"=== M6 CANARY v6 — sub-epoch stream, noise-arm yardstick, {dims} ===")
    print(
        f"budget {args.steps} steps/cell (NO early stop); schedule cosine_warmup_lr peak "
        f"{STAGE2_PEAK_LR} warmup_frac {WARMUP_FRAC}; evals every {args.eval_every}; "
        f"bars_total {args.bars_total}; spike rule > {args.spike_nats} nats -> halve once"
    )
    if args.canonical:
        parity = assert_cells_parity()
        print(f"[parity] canonical FSQ/BSQ arm parity asserted: {parity}")

    if args.stage1:
        manifest = run_stage1(args)
        ok = True  # stage-1 has no PASS/FAIL — the branch feeds the driver
    else:
        planted = run_arm("planted", planted=True, args=args)
        noise = run_arm("noise", planted=False, args=args)
        verdicts = {
            "planted_fires": planted["pb_4_minus_5"]["fired"],
            "planted_placebo_neutral": planted["pb_5_minus_2"]["placebo_neutral"],
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
            "canary": "m6_preflight_item2_v6",
            "stage": "full_10_cell",
            "arms": {"planted": planted, "noise": noise},
            "verdicts": verdicts,
            "PASS": ok,
        }
        print(f"[canary] verdicts: {verdicts}")

    manifest["recipe"] = {
        "canonical_dims": args.canonical,
        "signal_lag": SIGNAL_LAG,
        "c_signal": C_SIGNAL,
        "sigma": SIGMA,
        "bars_total": args.bars_total,
        "n_per_symbol": args.bars_total // len(SYMBOLS),
        "hold_frac": HOLD_FRAC,
        "sub_epoch": sub_epoch_arithmetic(args.steps, args.bars_total),
        "generator": "np.random.default_rng((zlib.crc32(arm_name), symbol_index)); "
        "arm_name in {'planted','noise'} — identical to v5, only the size changed",
        "symbols": list(SYMBOLS),
        "seq_len": SEQ_LEN,
        "batch": BATCH,
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
        "spike_nats": args.spike_nats,
        "filler_calibration": filler_calibration_receipt(),
        "legibility_gate_min": LEGIBILITY_MIN,
    }
    out = Path(args.out) / "canary_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[canary] manifest → {out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
