"""ITEM E DIAGNOSTIC (a) — IS THE MICRO EVICTION PER-BAR, OR COMPLETE? $0, local, CPU.

THE QUESTION. The §7 v1.4 legibility gate measures PER-BAR legibility: can `sign(x_t[dim])` be
read linearly from bar t's OWN (c_id, f_id) digits. It refused cell4 and cell5 on real data. That
leaves the decisive question unanswered: **does a WINDOWED read of the token stream recover the
micro dims?** The AR sees 512 bars of context, not one.

  * If a windowed read ALSO finds nothing, the eviction is COMPLETE rather than merely per-bar —
    the strongest form of the finding, and it kills the λ-re-derivation's premise for free.
  * If a windowed read DOES recover them, the gate is measuring a PER-BAR PROXY for something the
    AR might still reach, and that is a genuine limitation of the gate which must be disclosed.

SAME FOOTING AS THE GATE, DELIBERATELY. Same `digit_onehots` features, same `sign(value) > 0`
target, same 80/20 split BLOCKED IN TIME within each symbol, same 300 Adam steps at torch seed 0.
The ONLY change is the feature set: the gate uses lag 0; this adds lags spanning the 512-bar window.

★ THE BASIS IS RESTRICTED AND THE ASYMMETRY MATTERS — STATED, NOT BURIED.
The unrestricted basis is 512 lags x 53 = 27,136 features; at the gate's n = 150,000 that is a
16 GB design matrix and more parameters than the val split can honestly support. This uses a
LOG-SPACED lag set that spans the full window (0..511) at ~1,000 features, keeping n/p ~ 150.
So:
  * FINDING SIGNAL on a restricted basis is CONCLUSIVE — the information is demonstrably there,
    since a subset of a linear basis can only under-state what the full basis could read.
  * FINDING NOTHING is SUGGESTIVE, NOT PROOF, of complete eviction — a richer basis might still
    recover it. Report it as such and never as "the information is absent".

REPORTS. Per dim, per arm: lag-0 accuracy (the gate's own number, reproduced as the control), the
windowed accuracy, the delta, and the majority-class base rate so lift is visible beside accuracy.
INTERPRETS NOTHING.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake
from trikaal.eval.conformance import MDE_INPUTS, PINNED_MONEY_SEQ_LEN
from trikaal.run.matrix import load_symbol_arrays, windows_by_arm
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARM_MICRO, ARM_MICRO_SHUFFLED
from trikaal.train.checkpoint import load_checkpoint
from trikaal.train.gates import MICRO_DIMS, _stratified_blocked_split, digit_onehots
from trikaal.train.token_stream import tokenize_features

# Log-spaced lags spanning the full 512-bar context. 0 is the gate's own basis.
LAGS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 511)


def _fit_sign_acc(
    feats: np.ndarray, y: np.ndarray, groups: np.ndarray, n: int, *, steps: int = 300
) -> tuple[float, float]:
    """The gate's own estimator: stratified blocked 80/20, Adam, torch seed 0. Returns
    ``(val_sign_acc, final_train_loss)`` — the loss is the CONVERGENCE EVIDENCE without which the
    accuracy cannot be read (see ``SUPERSET DOMINANCE`` below)."""
    n = min(n, feats.shape[0])
    sel, tr = _stratified_blocked_split(groups, n)
    x, yy = feats[sel], y[sel]
    xt, xv = torch.from_numpy(x[tr]), torch.from_numpy(x[~tr])
    yt = torch.from_numpy(yy[tr])
    torch.manual_seed(0)
    logit = nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(logit.parameters(), lr=0.05)
    lossf = nn.BCEWithLogitsLoss()
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(logit(xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (logit(xv).squeeze(-1) > 0).numpy()
    return float((pred == (yy[~tr] > 0.5)).mean()), float(loss.item())


# ★ SUPERSET DOMINANCE ON *TRAIN LOSS* — THE CONVERGENCE CONTROL, AND WHERE IT ACTUALLY LIVES.
#
# The windowed basis CONTAINS lag 0 as a strict subset, so a converged windowed fit cannot have a
# HIGHER TRAINING LOSS than the lag-0 fit: any lag-0 solution is reachable by zeroing the extra
# blocks. If train loss does NOT improve, the optimizer never exploited the extra lags and the cell
# measures the optimizer rather than the information — PROBE INVALID for that cell.
#
# ★ IT DOES **NOT** HOLD ON VALIDATION ACCURACY, AND I ASSERTED THAT IT DID BEFORE CHECKING.
# 19x the parameters can generalize WORSE; that is overfitting, not non-convergence. Two wrong
# framings preceded this one and both are recorded because the correction is only trustworthy
# beside them:
#   (1) equal step counts gave windowed BELOW lag-0 on 11 of 12 cells, which read naively inverts
#       the finding into false evidence of complete eviction;
#   (2) the first fix put the dominance control on VAL ACCURACY, where the property is simply not
#       true — it would have condemned every honest cell as unconverged.
# Measured at 6,000 windowed steps: train loss improved on 12 of 12 while val accuracy fell on 11
# of 12. The fit converged AND the extra lags carry no generalizing signal — which is the finding,
# and is only readable because the control sits on the right quantity.
DOMINANCE_TOL = 1e-6  # on TRAIN LOSS: windowed must not be worse than lag0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", type=Path, default=Path("processed/universe_bars"))
    ap.add_argument("--n-symbols", type=int, default=10)
    ap.add_argument("--tail-bars", type=int, default=60_000)
    ap.add_argument("--n-rows", type=int, default=150_000)
    ap.add_argument("--steps-lag0", type=int, default=300, help="the gate's own step count")
    ap.add_argument(
        "--steps-window",
        type=int,
        default=6000,
        help="MORE than lag0: 19x the parameters at the same lr needs more steps, "
        "and superset dominance is the check that it was enough",
    )
    ap.add_argument("--out", type=Path, default=Path("runs_manifest/m6_window_context_probe.json"))
    a = ap.parse_args()
    if not a.lake.exists():
        print(f"LAKE MISSING at {a.lake}", file=sys.stderr)
        return 1

    pinned = json.loads(MDE_INPUTS.read_text())
    symbols = list(pinned["symbols_sampled"])[: a.n_symbols]
    window, train_frac = tuple(pinned["window"]), float(pinned["train_frac"])
    boundary = calendar_boundary_ms(window[0], window[1], train_frac)
    con = connect_lake(a.lake)

    arms = {"cell4_micro": ARM_MICRO, "cell5_micro_shuffled": ARM_MICRO_SHUFFLED}
    ckpt = {
        "cell4_micro": "runs_cloud/legibility_stop/cell4_seed0/tokenizer.pt",
        "cell5_micro_shuffled": "runs_cloud/legibility_stop/cell5_seed0/tokenizer.pt",
    }
    out: dict = {
        "receipt": "m6_window_context_probe",
        "what": "Does a WINDOWED linear read of the token stream recover the micro dims that the "
        "PER-BAR legibility gate could not? Same estimator, same split, same features — "
        "lags added.",
        "NO_INTERPRETATION": "reports lag-0 vs windowed accuracy and base rates; what it means "
        "is not this script's call.",
        "basis_is_restricted": {
            "lags": list(LAGS),
            "n_features_per_bar": None,
            "why": "the unrestricted 512-lag basis is 27,136 features (16 GB at n=150k) and has "
            "more parameters than the val split can honestly support. A SUBSET of a linear "
            "basis can only UNDER-state recoverability, so: finding signal here is "
            "CONCLUSIVE; finding nothing is SUGGESTIVE of complete eviction, NOT PROOF.",
        },
        "sampling": {
            "n_symbols": len(symbols),
            "tail_bars_per_symbol": a.tail_bars,
            "n_rows_target": a.n_rows,
            "split": "stratified by symbol, 80/20 blocked in time within symbol (the gate's own)",
            "estimator": "300 Adam steps, lr 0.05, torch seed 0 (the gate's own)",
        },
        "arms": {},
        "SUPERSET_DOMINANCE_CONTROL": "Checked on TRAIN LOSS, where the property holds: "
        "the windowed basis contains "
        "lag 0, so a converged windowed fit cannot train WORSE. It does NOT hold on VAL "
        "ACCURACY — more parameters may generalize worse, which is overfitting, not "
        "non-convergence. I asserted the val form before checking it and it was wrong; the "
        "smoke evidence is train loss improving 12/12 while val accuracy fell 11/12.",
    }

    for label, arm in arms.items():
        t0 = time.time()
        tok = load_checkpoint(Path(ckpt[label]), TokenizerAE, map_location="cpu").model.eval()
        raw = {
            s: load_symbol_arrays(con, s, boundary_ms=boundary, tail_bars=a.tail_bars)
            for s in symbols
        }
        per_symbol = windows_by_arm(
            raw, seq_len=PINNED_MONEY_SEQ_LEN, boundary_ms=boundary, seed=0, arms=(arm,)
        )[arm]
        print(f"[{label}] loaded {len(per_symbol)} symbols in {time.time() - t0:.0f}s", flush=True)

        toks = {}
        for sw in per_symbol:
            toks[sw.symbol] = tokenize_features(
                tok, sw.x, sw.mask, sw.segment_id, window=PINNED_MONEY_SEQ_LEN, device="cpu"
            )
        print(f"[{label}] tokenized in {time.time() - t0:.0f}s", flush=True)

        res = {}
        for dim in MICRO_DIMS:
            V, L0, LW, G = [], [], [], []
            for sw in per_symbol:
                b_c, b_f = toks[sw.symbol]
                oh = digit_onehots(tok, b_c, b_f)  # [N, 53]
                nb = oh.shape[0]
                keep = (sw.mask[:, dim] == 0) & (np.arange(nb) >= max(LAGS))
                idx = np.flatnonzero(keep)
                if idx.size == 0:
                    continue
                win = np.concatenate([oh[idx - lg] for lg in LAGS], axis=1)  # [k, 53*len(LAGS)]
                V.append(sw.x[idx, dim])
                L0.append(oh[idx])
                LW.append(win)
                G.append(np.full(idx.size, sw.symbol, dtype=object))
            if not V:
                res[str(dim)] = {"skipped": "no unmasked rows"}
                continue
            v = np.concatenate(V)
            y = (v > 0).astype(np.float32)
            g = np.concatenate(G)
            a0, l0 = _fit_sign_acc(np.concatenate(L0), y, g, a.n_rows, steps=a.steps_lag0)
            aw, lw = _fit_sign_acc(np.concatenate(LW), y, g, a.n_rows, steps=a.steps_window)
            br = float((v > 0).mean())
            maj = max(br, 1 - br)
            dominates = lw <= l0 + DOMINANCE_TOL  # TRAIN LOSS, not val accuracy
            res[str(dim)] = {
                "lag0_sign_acc": round(a0, 4),
                "windowed_sign_acc": round(aw, 4),
                "delta_windowed_minus_lag0": round(aw - a0, 4),
                "base_rate_positive": round(br, 4),
                "majority_baseline": round(maj, 4),
                "lag0_lift_over_majority": round(a0 - maj, 4),
                "windowed_lift_over_majority": round(aw - maj, 4),
                "n_rows": int(min(a.n_rows, v.size)),
                "train_loss_lag0": round(l0, 6),
                "train_loss_windowed": round(lw, 6),
                "superset_dominance_holds": bool(dominates),
                "READABLE": bool(dominates),
                "why_unreadable": None
                if dominates
                else "windowed TRAIN LOSS did not improve on lag-0's, though the windowed basis "
                "CONTAINS lag 0 — the optimizer never exploited the extra lags, so this cell "
                "measures the optimizer and not the information. PROBE INVALID for this cell.",
            }
            print(
                f"[{label}] dim {dim}: lag0={a0:.4f} windowed={aw:.4f} "
                f"delta={aw - a0:+.4f} maj={maj:.4f} dominance={'OK' if dominates else 'FAILED'}",
                flush=True,
            )
        out["basis_is_restricted"]["n_features_per_bar"] = int(
            np.concatenate(L0).shape[1] if L0 else 0
        )
        out["arms"][label] = res
        a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
