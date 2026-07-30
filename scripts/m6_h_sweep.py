"""§7 v1.4.5 CANARY-ONLY horizon sweep + degeneracy-band grid-stability check ($0, CPU).

    PYTHONPATH=src .venv/bin/python scripts/m6_h_sweep.py --sweep
    PYTHONPATH=src .venv/bin/python scripts/m6_h_sweep.py --guard-stability

SCOPE GUARDRAIL (supervisor ruling, §7 v1.4.5): this is a FIXTURE diagnostic. The real run's
h=15 and the entire pinned §3a money surface are NOT in scope and are not touched by anything
here. Nothing in this file may be quoted as a reason to move them; if a result argues for it,
that is a PROPOSAL for the supervisor, never an executed change.

MECHANISM UNDER TEST (h-sweep). At h=15 the KV-cache rollout accumulates ~15 decode steps of
drift into a large constant offset in μ̂, while the fixture's planted information lives entirely
at lag SIGNAL_LAG=2. If drift accumulation is what saturates the sign, lowering h should shrink
the offset roughly in proportion to h while preserving the state-driven component — raising
dispersion-over-mean and letting the θ = κ·c execution filter bind.

WHY A MIRROR OF ``score_cell`` AND NOT ``score_cell`` ITSELF. ``score_cell`` tokenizes the full
42M-bar stream on every call; the sweep is 5 checkpoints x 5 horizons and tokenization is
h-INDEPENDENT, so calling it 25 times would re-pay ~7.8 CPU-minutes of tokenization 20 times
over, and holding three 14M-bar 16-dim feature matrices at once does not fit this box's RAM.
``score_pretokenized`` below is a LINE-FOR-LINE mirror with the token stream hoisted out of the
h-loop; it imports the pooling, cost, position, netting and IR arithmetic from the instrument
modules rather than reimplementing any of it. Its equivalence to ``score_cell`` is a KAT
(tests/eval/test_h_sweep_mirror.py) — the mirror is never trusted on assertion alone.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from trikaal.eval.costs import SPREAD_DECILE_FRAC, CostModel
from trikaal.eval.harness import _per_bar_cost, forward_log_returns
from trikaal.eval.metrics import information_ratio
from trikaal.eval.strategy import net_trade_returns, positions
from trikaal.eval.verdict import FRAC_NEG_BAND
from trikaal.eval.xsection import (
    SymbolEval,
    XSectionConfig,
    _pool_periods,
    global_decision_grid_ms,
    symbol_decisions,
)
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARM_MICRO_SHUFFLED, select_arm
from trikaal.train.checkpoint import load_checkpoint

HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 15)
OUT_SWEEP = Path("runs_manifest/m6_h_sweep.json")
OUT_GUARD = Path("runs_manifest/m6_guard_band_stability.json")

# The L3 reference the band-stability check compares against (n=4000, h=15, capped).
L3_REFERENCE_FRAC_NEG = 0.5585
BAND_DRIFT_LIMIT = 0.15  # pre-declared: >this absolute move => band NOT grid-stable => STOP


@dataclass(frozen=True)
class Ckpt:
    """One trained (run, arm-stream, cell) whose weights survived teardown."""

    key: str
    path: str
    arm_stream: str  # the synth stream: "planted" | "noise"
    planted: bool
    arm: str  # the cell's feature arm: "ohlcv" | "micro"
    cell_id: int


CKPTS: tuple[Ckpt, ...] = (
    Ckpt(
        "acceptance_planted_cell4",
        "runs_cloud/acceptance/m6_acceptance/planted/cell4_fsq_micro_seed0",
        "planted",
        True,
        "micro",
        4,
    ),
    Ckpt(
        "acceptance_noise_cell4",
        "runs_cloud/acceptance/m6_acceptance/noise/cell4_fsq_micro_seed0",
        "noise",
        False,
        "micro",
        4,
    ),
    Ckpt(
        "moneyleg_noise_cell1",
        "runs_cloud/moneyleg/m6_moneyleg/noise/cell1_bsq_ohlcv_seed0",
        "noise",
        False,
        "ohlcv",
        1,
    ),
    Ckpt(
        "moneyleg_noise_cell2",
        "runs_cloud/moneyleg/m6_moneyleg/noise/cell2_fsq_ohlcv_seed0",
        "noise",
        False,
        "ohlcv",
        2,
    ),
    Ckpt(
        "moneyleg_noise_cell3",
        "runs_cloud/moneyleg/m6_moneyleg/noise/cell3_bsq_micro_seed0",
        "noise",
        False,
        "micro",
        3,
    ),
)


def _env(device: str) -> str:
    """Interpreter provenance, stamped into EVERY receipt this script writes.

    The v1.4.4 environment slip (a whole session run on the system interpreter instead of the
    uv-locked `.venv`) was caught only by chance, from a two-line diff in an unrelated manifest.
    Recording numpy/torch on the artifact itself makes the same mistake self-announcing.
    """
    import numpy
    import torch

    return f"numpy {numpy.__version__} / torch {torch.__version__}; device {device}"


def _canary():
    """The fixture module is a script, not a package — import it by path at call time."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import m6_canary as mc

    return mc


def canary_cfg(h: int, *, cap: int | None, n_per: int, seed: int, device: str) -> XSectionConfig:
    """The EXACT XSectionConfig ``m6_canary.run_arm`` builds, with h and cap parameterized."""
    mc = _canary()
    end_dt = datetime.fromtimestamp((mc.T0_MS + n_per * 60_000) / 1000, tz=UTC) + timedelta(days=1)
    return XSectionConfig(
        window_start="2024-01-01",
        window_end=end_dt.strftime("%Y-%m-%d"),
        train_frac=mc.TRAIN_FRAC,
        h=h,
        seq_len=mc.SEQ_LEN,
        cap_per_symbol=cap,
        device=device,
        seed=seed,
    )


# --------------------------------------------------------------- the mirror (KAT'd vs score_cell)
def score_pretokenized(
    model,
    tok,
    prepared_tokens: dict[str, dict],
    cfg: XSectionConfig,
) -> dict:
    """``xsection.score_cell`` with the token stream hoisted out.

    ``prepared_tokens[symbol]`` = {"b_c", "b_f", "se"} where ``se`` is a SymbolEval carrying at
    least ts / sigma / raw_ret_close / segment_id. Returns the sweep's per-(cell, h) record.

    Mirrors score_cell exactly: ONE rng seeded at cfg.seed shared across the VAL and headline
    grid passes (so the cap subsample draw ORDER matches), cap-then-finite-filter ordering, κ*
    argmax over the pooled VAL per-κ IR, headline netted at the flat cfg.headline_cost.
    """
    from trikaal.eval.predict import predict_mu

    cost = CostModel()
    rng = np.random.default_rng(cfg.seed)

    def spread_for(sym: str) -> float:
        if cfg.spread_frac_by_symbol is None:
            return SPREAD_DECILE_FRAC["major"]
        return float(cfg.spread_frac_by_symbol[sym])

    prepared = {
        s: {
            "b_c": d["b_c"],
            "b_f": d["b_f"],
            "y": forward_log_returns(d["se"].raw_ret_close, d["se"].segment_id, cfg.h),
            "se": d["se"],
        }
        for s, d in prepared_tokens.items()
    }

    def grid_series(grid: np.ndarray):
        n_periods = grid.shape[0]
        nets_flat: dict[float, dict] = {k: {} for k in cfg.kappas}
        n_dec: dict[str, int] = {}
        mu_pool: list[np.ndarray] = []
        traded_by_k: dict[float, int] = dict.fromkeys(cfg.kappas, 0)
        n_scored = 0
        for sym, d in prepared.items():
            se = d["se"]
            dec = symbol_decisions(se, grid, cfg)
            if cfg.cap_per_symbol is not None and dec.size > cfg.cap_per_symbol:
                dec = np.sort(rng.choice(dec, size=cfg.cap_per_symbol, replace=False))
            finite = np.isfinite(d["y"][dec])
            dec = dec[finite]
            if dec.size == 0:
                continue
            n_dec[sym] = int(dec.size)
            n_scored += int(dec.size)
            mu = predict_mu(
                model,
                tok,
                d["b_c"],
                d["b_f"],
                se.ts,
                se.sigma,
                dec,
                h=cfg.h,
                seq_len=cfg.seq_len,
                device=cfg.device,
            )
            mu_pool.append(mu)
            y_d = d["y"][dec]
            c_mod = _per_bar_cost(cost, se.sigma, dec, spread_for(sym))
            pidx = np.searchsorted(grid, se.ts[dec])
            for k in cfg.kappas:
                s = positions(mu, k * c_mod)
                act = s != 0
                traded_by_k[k] += int(act.sum())
                nets_flat[k][sym] = (
                    pidx[act],
                    net_trade_returns(s[act], y_d[act], np.full(int(act.sum()), cfg.headline_cost)),
                )
        series = {k: _pool_periods(nets_flat[k], n_periods) for k in cfg.kappas}
        counts = {
            "traded_by_kappa": {f"{k:g}": traded_by_k[k] for k in cfg.kappas},
            "scored": n_scored,
        }
        mu_all = np.concatenate(mu_pool) if mu_pool else np.array([])
        mu_diag = (
            {
                "mean": float(mu_all.mean()),
                "std": float(mu_all.std()),
                "frac_negative": float((mu_all < 0).mean()),
                "n": int(mu_all.size),
                "estimator": "expectation",
                "activity_decisions_by_kappa": {
                    f"{k:g}": (traded_by_k[k] / n_scored) for k in cfg.kappas
                },
            }
            if mu_all.size
            else {}
        )
        return series, n_dec, mu_diag, counts

    val_series, _, _, _ = grid_series(global_decision_grid_ms(cfg, cfg.val_block))
    val_ir = {k: information_ratio(val_series[k], cfg.h) for k in cfg.kappas}
    finite_ir = {k: v for k, v in val_ir.items() if np.isfinite(v)}
    kappa_star = max(finite_ir, key=finite_ir.get) if finite_ir else cfg.kappas[0]

    hd_series, n_dec, mu_diag, counts = grid_series(
        global_decision_grid_ms(cfg, cfg.headline_block)
    )
    if mu_diag:
        mu_diag["activity_decisions"] = float(
            mu_diag["activity_decisions_by_kappa"][f"{kappa_star:g}"]
        )
    head = hd_series[kappa_star]
    return {
        "h": int(cfg.h),
        "kappa_star": float(kappa_star),
        "ir_at_kstar": float(information_ratio(head, cfg.h)),
        "ir_by_kappa": {
            f"{k:g}": float(information_ratio(hd_series[k], cfg.h)) for k in cfg.kappas
        },
        "val_ir_by_kappa": {f"{k:g}": float(v) for k, v in val_ir.items()},
        "n_decisions": int(sum(n_dec.values())),
        # L2's raw counts: decision-activity is traded/scored, and "traded == scored at every
        # kappa" is the direct statement that theta = kappa*c never binds.
        "traded_by_kappa": counts["traded_by_kappa"],
        "n_scored": counts["scored"],
        "mu_diag": mu_diag,
    }


# ------------------------------------------------------------------------------- data + tokenizing
@torch.no_grad()
def tokenize_needed(
    tok,
    x_np: np.ndarray,
    m_np: np.ndarray,
    segment_id: np.ndarray,
    needed: np.ndarray,
    *,
    window: int,
    batch_windows: int = 128,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """``tokenize_features`` restricted to the aligned chunks that contain a needed bar.

    WHY THIS IS EXACT, NOT AN APPROXIMATION. ``tokenize_features`` encodes the stream in
    NON-OVERLAPPING chunks of ``window`` bars that never cross a segment, so bar t's token is a
    function of exactly one chunk — the aligned block containing t — and of nothing else. Skipping
    a chunk therefore cannot perturb any bar inside a chunk we do keep. This routine reuses the
    instrument's own ``_chunk_ranges`` so the chunk boundaries are identical by construction, and
    keeps the same batch size and traversal order.

    Decisions live only in eval blocks 0 (VAL) and 3 (headline) — about 10% of the 14M-bar
    fixture stream — so this removes ~90% of the tokenizer forward passes AND the full-length
    float32 torch copy that ``tokenize_features`` makes (windows are gathered per batch instead),
    which is what took the full-stream version into swap on a 16 GB box.

    Unwritten positions are left at -1 so any accidental read of a bar we never tokenized is a
    loud out-of-range index rather than a silently plausible token id.
    """
    from trikaal.train.token_stream import _chunk_ranges

    tok.eval()
    dev = torch.device(device)
    tok.to(dev)
    n = x_np.shape[0]
    b_c = np.full(n, -1, dtype=np.int32)
    b_f = np.full(n, -1, dtype=np.int32)
    ranges = [r for r in _chunk_ranges(segment_id, window) if needed[r[0] : r[1]].any()]
    full = [(a, b) for a, b in ranges if b - a == window]
    tail = [(a, b) for a, b in ranges if b - a < window]

    for i in range(0, len(full), batch_windows):
        chunk = full[i : i + batch_windows]
        starts = np.array([a for a, _ in chunk], dtype=np.int64)
        idx = starts[:, None] + np.arange(window)[None, :]
        xb = torch.from_numpy(np.ascontiguousarray(x_np[idx], dtype=np.float32)).to(dev)
        mb = torch.from_numpy(np.ascontiguousarray(m_np[idx], dtype=np.float32)).to(dev)
        cidx, fidx = tok.encode_tokens(xb, mb)
        cidx, fidx = cidx.cpu().numpy(), fidx.cpu().numpy()
        for j, (a, b) in enumerate(chunk):
            b_c[a:b], b_f[a:b] = cidx[j], fidx[j]
    for a, b in tail:
        xb = torch.from_numpy(np.ascontiguousarray(x_np[a:b], dtype=np.float32))[None].to(dev)
        mb = torch.from_numpy(np.ascontiguousarray(m_np[a:b], dtype=np.float32))[None].to(dev)
        cidx, fidx = tok.encode_tokens(xb, mb)
        b_c[a:b], b_f[a:b] = cidx[0].cpu().numpy(), fidx[0].cpu().numpy()
    return b_c, b_f


def needed_bar_mask(cfg_probe: XSectionConfig, ts: np.ndarray, blocks=(0, 3)) -> np.ndarray:
    """Bars whose tokens any (block, horizon) in scope could read.

    predict_mu reads the ``seq_len`` bars ending at each decision. The densest decision grid is
    h=1 (stride one bar), which selects every valid bar in the block, so taking h=1 UNCAPPED and
    extending each block back by seq_len gives a superset for every horizon and every cap.
    """
    se = SymbolEval(
        symbol="probe",
        x=None,
        mask=None,
        segment_id=None,
        ts=ts,
        sigma=None,
        raw_ret_close=None,
    )
    needed = np.zeros(ts.shape[0], dtype=bool)
    for blk in blocks:
        dec = symbol_decisions(se, global_decision_grid_ms(cfg_probe, blk), cfg_probe)
        if dec.size == 0:
            continue
        lo = max(0, int(dec.min()) - cfg_probe.seq_len + 1)
        needed[lo : int(dec.max()) + 1] = True
    return needed


def build_tokens(ck: Ckpt, *, n_per: int, device: str, symbols=None) -> tuple[dict, object, object]:
    """Generate the arm stream symbol-by-symbol and tokenize with the cell's frozen tokenizer.

    Memory discipline (this box has 16 GB and a nearly full swap file): each symbol's 16-dim
    feature matrix is ~0.9 GB plus a ~1.2 GB float64 scratch inside the fixture generator, so
    symbols are generated ONE AT A TIME and their features dropped the moment the token stream
    exists. Only the needed chunks are tokenized (see ``tokenize_needed``), token ids are kept as
    int32, and ``ts`` / ``segment_id`` / ``sigma`` — which are byte-identical across symbols by
    construction (same T0, same length, constant SIGMA) — are allocated once and shared.
    """
    mc = _canary()
    syms = list(symbols if symbols is not None else mc.SYMBOLS)
    d = Path(ck.path)
    tok = load_checkpoint(d / "tokenizer.pt", TokenizerAE, map_location=device).model
    model = load_checkpoint(d / "predictor.pt", TrikaalAR, map_location=device).model
    tok.to(device).eval()
    model.to(device).eval()
    if ck.arm == ARM_MICRO_SHUFFLED:  # score_cell re-applies the training shuffle first
        raise NotImplementedError("cell5 (shuffled placebo) has no surviving checkpoint")

    ts_shared = (mc.T0_MS + np.arange(n_per) * 60_000).astype(np.int64)
    seg_shared = np.zeros(n_per, dtype=np.int64)
    sigma_shared = np.full(n_per, mc.SIGMA)
    probe = canary_cfg(1, cap=None, n_per=n_per, seed=0, device=device)
    needed = needed_bar_mask(probe, ts_shared)
    print(
        f"    needed bars: {int(needed.sum()):,} / {n_per:,} "
        f"({100 * needed.mean():.1f}% — eval blocks 0 and 3 only)",
        flush=True,
    )

    prepared: dict[str, dict] = {}
    for i, s in enumerate(syms):
        t0 = time.time()
        raw = mc.synth_symbol(
            np.random.default_rng((zlib.crc32(ck.arm_stream.encode()), i)),
            planted=ck.planted,
            n_bars=n_per,
        )
        assert np.array_equal(raw["ts"], ts_shared) and np.array_equal(
            raw["segment_id"], seg_shared
        ), "shared ts/segment_id must match the fixture generator exactly"
        assert np.array_equal(raw["sigma"], sigma_shared), "shared sigma must match the fixture"
        x_arm, m_arm = select_arm(np.asarray(raw["x"], np.float32), raw["mask"], ck.arm)
        del raw["x"], raw["mask"]
        b_c, b_f = tokenize_needed(
            tok, x_arm, m_arm, seg_shared, needed, window=mc.SEQ_LEN, device=device
        )
        del x_arm, m_arm
        prepared[s] = {
            "b_c": b_c,
            "b_f": b_f,
            "se": SymbolEval(
                symbol=s,
                x=None,
                mask=None,
                segment_id=seg_shared,
                ts=ts_shared,
                sigma=sigma_shared,
                raw_ret_close=raw["raw_ret_close"],
            ),
        }
        print(f"    tokenized {s} ({n_per:,} bars) in {time.time() - t0:.1f}s", flush=True)
    return prepared, model, tok


# ------------------------------------------------------------------------------------ the h-sweep
def run_sweep(args) -> int:
    mc = _canary()
    n_per = args.bars_total // len(mc.SYMBOLS)
    cells: dict[str, dict] = {}
    for ck in CKPTS:
        if not (Path(ck.path) / "predictor.pt").exists():
            print(f"[skip] {ck.key}: no checkpoint on disk ({ck.path})")
            continue
        print(f"[sweep] {ck.key} — tokenizing once, then h in {HORIZONS}", flush=True)
        prepared, model, tok = build_tokens(ck, n_per=n_per, device=args.device)
        by_h = {}
        for h in HORIZONS:
            t0 = time.time()
            cfg = canary_cfg(h, cap=args.eval_cap, n_per=n_per, seed=args.seed, device=args.device)
            rec = score_pretokenized(model, tok, prepared, cfg)
            md = rec["mu_diag"]
            mean, std = md.get("mean", float("nan")), md.get("std", float("nan"))
            rec["dispersion_over_mean"] = float(std / abs(mean)) if mean else float("inf")
            rec["wall_s"] = round(time.time() - t0, 1)
            by_h[str(h)] = rec
            print(
                f"  h={h:>2}: IR@k*={rec['ir_at_kstar']:+.4f} k*={rec['kappa_star']} "
                f"mu(mean={mean:+.5f} std={std:.5f} disp/|mean|={rec['dispersion_over_mean']:.3f} "
                f"neg={md.get('frac_negative', float('nan')):.4f}) "
                f"act@k*={md.get('activity_decisions', float('nan')):.4f} "
                f"n={rec['n_decisions']} [{rec['wall_s']}s]",
                flush=True,
            )
        cells[ck.key] = {"checkpoint": ck.path, "arm": ck.arm, "cell_id": ck.cell_id, "by_h": by_h}
        del prepared, model, tok

    lo, hi = FRAC_NEG_BAND
    upgrade = _upgrade_test(cells.get("acceptance_planted_cell4"), lo, hi)
    out = {
        "receipt": "m6_h_sweep",
        "scope": (
            "CANARY-ONLY fixture diagnostic. The real run's h=15 and the pinned 3a money "
            "surface are NOT in scope and are unchanged by this receipt."
        ),
        "mechanism_under_test": (
            "h=15 rollout drift accumulation saturating the sign of mu-hat; lowering h should "
            "shrink the constant offset ~proportionally while preserving the state component."
        ),
        "horizons": list(HORIZONS),
        "bars_total": int(args.bars_total),
        "n_per_symbol": int(args.bars_total // len(mc.SYMBOLS)),
        "h1_structural_note": (
            "h=1 is STRUCTURALLY VOID for this estimator, not a measurement: mu-hat covers "
            "[t+1, t+h] = log(C_{t+h}/C_{t+1}), so predict.py excludes rollout step k=1 "
            "(entry-next-bar, spec 8.B.3). At h=1 the accumulator is never touched and mu-hat "
            "== 0 identically for every model. Its all-zero row is the estimator's definition, "
            "NOT evidence about drift, sign-lock, or filter binding. The fixture's plant sits at "
            "SIGNAL_LAG=2, so h=2 is both the plant's own horizon and the smallest valid one."
        ),
        "eval_cap_per_symbol": args.eval_cap,
        "device": args.device,
        "environment": _env(args.device),
        "estimator": "expectation",
        "frac_neg_band": [lo, hi],
        "pre_declared_upgrade_test": (
            "Leg (c) upgrades to demonstrated-at-the-plant-horizon ONLY if, on the acceptance "
            "PLANTED cell4, at some h: decision-activity at kappa* strictly inside (0,1) AND "
            "frac_negative inside [0.05,0.95] AND IR at kappa* > 0. All three, or no upgrade."
        ),
        "upgrade_test": upgrade,
        "cells": cells,
    }
    OUT_SWEEP.parent.mkdir(parents=True, exist_ok=True)
    OUT_SWEEP.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\n[upgrade-test] {upgrade['verdict']}")
    print(f"[receipt] -> {OUT_SWEEP}")
    return 0


def _upgrade_test(cell: dict | None, lo: float, hi: float) -> dict:
    """The three-part pre-declared test on the acceptance PLANTED cell4 — all three or nothing."""
    if not cell:
        return {"verdict": "NOT EVALUABLE — acceptance planted cell4 absent", "per_h": {}}
    per_h = {}
    for h, rec in cell["by_h"].items():
        md = rec["mu_diag"]
        act = float(md.get("activity_decisions", float("nan")))
        fn = float(md.get("frac_negative", float("nan")))
        ir = float(rec["ir_at_kstar"])
        legs = {
            "activity_binds": bool(act == act and 0.0 < act < 1.0),
            "sign_balanced": bool(fn == fn and lo <= fn <= hi),
            "ir_positive": bool(ir == ir and ir > 0.0),
        }
        per_h[h] = {
            "activity_decisions": act,
            "frac_negative": fn,
            "ir_at_kstar": ir,
            **legs,
            "all_three": bool(all(legs.values())),
            # h=1 CANNOT be a measurement: mu-hat covers [t+1, t+h] = log(C_{t+h}/C_{t+1}), so
            # predict.py excludes rollout step k=1 (entry-next-bar, 8.B.3). At h=1 nothing is
            # ever accumulated and mu-hat == 0 IDENTICALLY. That is the estimator's definition,
            # not a property of any model, and it must never be read as "the filter did not bind".
            "structurally_void": h == "1",
        }
    hits = [h for h, v in per_h.items() if v["all_three"] and not v["structurally_void"]]
    binds = [h for h, v in per_h.items() if v["activity_binds"] and not v["structurally_void"]]
    if hits:
        verdict = f"UPGRADE — all three legs hold at h={sorted(hits, key=int)}"
    elif binds:
        verdict = (
            f"NO UPGRADE — activity binds at h={sorted(binds, key=int)} but not all three legs; "
            "leg (c) stays a NAMED RESIDUAL, recorded as a horizon diagnostic with the "
            "mechanism CONFIRMED."
        )
    else:
        verdict = (
            "NO UPGRADE — decision-activity never binds at any h: the drift-accumulation "
            "hypothesis is REFUTED. Leg (c) stays a named residual; close conditional."
        )
    return {"verdict": verdict, "per_h": per_h, "horizons_all_three": sorted(hits, key=int)}


# ------------------------------------------------------------ BLOCKING band grid-stability check
def run_guard_stability(args) -> int:
    """Re-decode the acceptance PLANTED cell4 at h=15 across decision-set sizes.

    Separates GRID/CAP dependence of frac_negative from MODEL divergence between the acceptance
    and money-leg runs (whose recipes are byte-identical). Pre-declared: a move > 0.15 absolute
    means the [0.05,0.95] band is not grid-stable -> report and STOP, do NOT adjust the band.
    """
    mc = _canary()
    ck = CKPTS[0]
    assert ck.key == "acceptance_planted_cell4"
    n_per = args.bars_total // len(mc.SYMBOLS)
    variants = [
        ("cap4000_1symbol", 4000, mc.SYMBOLS[:1]),
        ("cap4000_3symbols", 4000, mc.SYMBOLS),
        ("uncapped_3symbols", None, mc.SYMBOLS),
    ]
    results = {}
    for name, cap, syms in variants:
        if name == "uncapped_3symbols" and args.skip_uncapped:
            print(f"[skip] {name} (--skip-uncapped)")
            continue
        print(f"[guard] {name}: cap={cap} symbols={list(syms)}", flush=True)
        prepared, model, tok = build_tokens(ck, n_per=n_per, device=args.device, symbols=syms)
        cfg = canary_cfg(15, cap=cap, n_per=n_per, seed=args.seed, device=args.device)
        t0 = time.time()
        rec = score_pretokenized(model, tok, prepared, cfg)
        rec["wall_s"] = round(time.time() - t0, 1)
        results[name] = rec
        md = rec["mu_diag"]
        print(
            f"  n={rec['n_decisions']} frac_neg={md['frac_negative']:.4f} "
            f"act@k*={md['activity_decisions']:.4f} IR@k*={rec['ir_at_kstar']:+.4f} "
            f"[{rec['wall_s']}s]",
            flush=True,
        )
        del prepared, model, tok

    fns = {k: v["mu_diag"]["frac_negative"] for k, v in results.items()}
    drifts = {k: abs(v - L3_REFERENCE_FRAC_NEG) for k, v in fns.items()}
    worst = max(drifts, key=drifts.get) if drifts else None
    stable = bool(worst is not None and drifts[worst] <= BAND_DRIFT_LIMIT)
    out = {
        "receipt": "m6_guard_band_stability",
        "scope": "CANARY-ONLY. Blocking check on the v1.4.4 FRAC_NEG_BAND's grid stability.",
        "checkpoint": ck.path,
        "h": 15,
        "environment": _env(args.device),
        "bars_total": int(args.bars_total),
        "n_per_symbol": int(n_per),
        "l3_reference_frac_negative": L3_REFERENCE_FRAC_NEG,
        "band_drift_limit_absolute": BAND_DRIFT_LIMIT,
        "frac_negative_by_variant": fns,
        "absolute_drift_vs_l3": drifts,
        "worst_variant": worst,
        "band_grid_stable": stable,
        "what_the_uncapped_variant_settles": (
            "The ruling described the money-leg's n=12000 decision set as UNCAPPED. Measured "
            "here: 12,000 is 3 symbols x the 4,000 per-symbol eval cap; the genuinely uncapped "
            "block-3 grid is 46,671 decisions per symbol = 140,013 total. Both are scored so cap "
            "dependence is settled by measurement rather than by the label."
        ),
        "variant_subsampling_note": (
            "The 1-symbol and 3-symbol capped variants do NOT share a decision subset. score_cell "
            "draws the cap subsample from ONE rng in symbol order across the VAL pass then the "
            "headline pass, so SYNAUSDT's headline draw is the 2nd draw with one symbol and the "
            "4th with three. The variants therefore probe subsample-to-subsample variation as "
            "well as cap size — a stricter stability test, not a weaker one."
        ),
        "pre_declared_action": (
            "PASS -> the band stands. FAIL (>0.15) -> report and STOP for a band re-derivation "
            "or a replacement statistic; the builder does NOT adjust the band."
        ),
        "variants": results,
    }
    OUT_GUARD.parent.mkdir(parents=True, exist_ok=True)
    OUT_GUARD.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\n[band] grid_stable={stable} worst={worst} drift={drifts.get(worst)}")
    print(f"[receipt] -> {OUT_GUARD}")
    return 0


# ------------------------------------------- L1/L2/L4 regeneration on the pinned .venv (C1 fix)
# The cloud money-leg noise IRs at kappa*, at full float64 precision, from
# runs_manifest/m6_moneyleg_rerun_manifest.json (a441cfa).
CLOUD_NOISE_IR = {
    1: -7.240328510778865,
    2: -7.563377128272953,
    3: -7.563377128272953,
}
L1_CRITERION_ABS = 1e-3  # the declared reproduction criterion (NOT a bit-identity claim)


def run_rescore(args) -> int:
    """L1 (fidelity) + L2 (per-kappa binding) + L4 (SIGMA counterfactual), FULL PRECISION.

    C1: the v1.4.4 receipt stored the local IR ROUNDED TO 6 DECIMALS in its display field while
    reporting L1_abs_diff = 0.0, which made the zero look like it could be a rounding artifact.
    Here the local IR is retained at FULL float64 so |Delta| is measured rather than inferred.

    L4 exploits training-invariance: the feature matrix is SIGMA-independent (x[:,0] = r/sigma
    cancels), so the token stream is built ONCE and both sigmas are scored by scaling
    raw_ret_close and sigma exactly — no retokenization, no retraining.
    """
    mc = _canary()
    n_per = args.bars_total // len(mc.SYMBOLS)
    sigmas = (0.01, 0.05)
    cells = {}
    for ck in CKPTS:
        if not ck.key.startswith("moneyleg_noise"):
            continue
        if not (Path(ck.path) / "predictor.pt").exists():
            print(f"[skip] {ck.key}: no checkpoint on disk")
            continue
        print(f"[rescore] {ck.key}", flush=True)
        prepared, model, tok = build_tokens(ck, n_per=n_per, device=args.device)
        base_sigma = float(mc.SIGMA)
        per_sigma = {}
        for sg in sigmas:
            scale = sg / base_sigma
            scaled = {
                s: {
                    "b_c": d["b_c"],
                    "b_f": d["b_f"],
                    "se": SymbolEval(
                        symbol=s,
                        x=None,
                        mask=None,
                        segment_id=d["se"].segment_id,
                        ts=d["se"].ts,
                        sigma=d["se"].sigma * scale,
                        raw_ret_close=d["se"].raw_ret_close * scale,
                    ),
                }
                for s, d in prepared.items()
            }
            cfg = canary_cfg(15, cap=args.eval_cap, n_per=n_per, seed=args.seed, device=args.device)
            rec = score_pretokenized(model, tok, scaled, cfg)
            binds = {
                k: (rec["traded_by_kappa"][k] != rec["n_scored"]) for k in rec["traded_by_kappa"]
            }
            rec["filter_binds_by_kappa"] = binds
            rec["filter_ever_binds"] = bool(any(binds.values()))
            per_sigma[f"{sg:g}"] = rec
            md = rec["mu_diag"]
            print(
                f"  sigma={sg:g}: IR@k*={rec['ir_at_kstar']!r} k*={rec['kappa_star']} "
                f"traded/scored={rec['traded_by_kappa']}/{rec['n_scored']} "
                f"frac_neg={md['frac_negative']} binds={rec['filter_ever_binds']}",
                flush=True,
            )
        local = per_sigma[f"{base_sigma:g}"]["ir_at_kstar"]
        cloud = CLOUD_NOISE_IR[ck.cell_id]
        delta = abs(local - cloud)
        cells[str(ck.cell_id)] = {
            "checkpoint": ck.path,
            "L1_cloud_ir_full_precision": cloud,
            "L1_local_ir_full_precision": local,
            "L1_abs_delta": delta,
            "L1_abs_delta_sci": f"{delta:.3e}",
            "L1_criterion_abs": L1_CRITERION_ABS,
            "L1_reproduced": bool(delta <= L1_CRITERION_ABS),
            "L1_orders_of_margin": (
                float(np.log10(L1_CRITERION_ABS / delta)) if delta > 0 else float("inf")
            ),
            "per_sigma": per_sigma,
        }
        print(f"  L1: |delta| = {delta:.3e} vs criterion {L1_CRITERION_ABS:g}", flush=True)
        del prepared, model, tok

    deltas = [c["L1_abs_delta"] for c in cells.values()]
    out = {
        "receipt": "m6_moneyleg_local_rescore_L1L2L4",
        "supersedes": "the v1.4.4 receipt of the same name (display field rounded to 6dp)",
        "environment": _env(args.device),
        "C1_correction": (
            "MEASURED, NOT INFERRED. The v1.4.4 receipt stored the local IR rounded to 6 decimals "
            "in its DISPLAY field (-7.240329) next to L1_abs_diff = 0.0, which made the zero look "
            "like it could be a rounding artifact; the v1.4.5 ruling inferred a true |Delta| of "
            "about 4.9e-7 on that basis. Regenerated here at full float64 on the pinned .venv, "
            "the local IR equals the cloud IR EXACTLY for all three cells (|Delta| = 0.0, not "
            "4.9e-7), so the original L1_abs_diff was CORRECT and only the display precision was "
            "misleading. The inferred 4.9e-7 discrepancy does not exist."
        ),
        "C1_builder_disclosure": (
            "The builder wrote the C1 correction into docs/m6_prereg.md BEFORE regenerating this "
            "receipt, acting on the ruling's inference rather than on a measurement. That was the "
            "error the project's own rule forbids (claims must match receipts). The prereg text "
            "is withdrawn and replaced with this measured result."
        ),
        "C2_epistemic_status": (
            "The exact agreement is a SYMPTOM of the sign-only degeneracy, not evidence of "
            "instrument health — and the C1 measurement STRENGTHENS this reading rather than "
            "weakening it. Because traded == scored at every kappa, the book is constant, so "
            "mu-hat's MAGNITUDES drop out of the IR entirely: the computation reduces to the "
            "same return series under the same fixed positions. That is why |Delta| is EXACTLY "
            "0.0 across CUDA/numpy 2.4.6, CPU/numpy 2.3.3 and CPU/numpy 2.4.6 rather than merely "
            "small — the arithmetic never touches the numbers that differ between environments. "
            "On a NON-degenerate cell those same ULP-scale differences would flip marginal bars "
            "and reproduction would be looser. L1 therefore validates the SIGN PATTERN and the "
            "scoring arithmetic — which is all L2 and L4 needed, so their conclusions stand — "
            "and nothing more."
        ),
        "L2_L4_finding": (
            "Decision-activity is 1.0 at every kappa in {1, 1.5, 2, 3} at BOTH sigmas: traded == "
            "scored everywhere, so the execution filter never binds and raising SIGMA only "
            "rescales |IR|. The v1.4.3 economic recalibration is a measured NON-FIX; SIGMA is "
            "reverted to 0.01 and the negative result is retained rather than deleted."
        ),
        "worst_abs_delta": max(deltas) if deltas else None,
        "worst_abs_delta_sci": f"{max(deltas):.3e}" if deltas else None,
        "L1_criterion_abs": L1_CRITERION_ABS,
        "L1_all_reproduced": bool(all(c["L1_reproduced"] for c in cells.values())),
        "sigmas": list(sigmas),
        "eval_cap_per_symbol": args.eval_cap,
        "n_per_symbol": n_per,
        "cells": cells,
    }
    out_path = Path("runs_manifest/m6_moneyleg_local_rescore.json")
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(
        f"\n[L1] worst |delta| = {out['worst_abs_delta_sci']} vs {L1_CRITERION_ABS:g} — "
        f"all_reproduced={out['L1_all_reproduced']}"
    )
    print(f"[receipt] -> {out_path}")
    return 0


# ------------------------------------------------------- item-1 estimator bias bound (n >= 256)
BIAS_CELLS = ("acceptance_planted_cell4", "moneyleg_noise_cell2")


def run_bias_bound(args) -> int:
    """§7 v1.4.3 item 1: expectation vs the unbiased mc_mean reference, paired on one decision set.

    WHAT CHANGED vs v1.4.3. The order read "expectation vs mc_mean at n >= 256". The v1.4.3
    receipt used mc_samples = 256 with n_dec = 24, so it met that bar on the MC-sample reading;
    the DECISION count was the weak leg, and it is the one the supervisor challenged. With
    tokenization now restricted to the needed chunks, n_dec = 256 is affordable, so BOTH readings
    are satisfied here. This is a strengthening of the weak leg, not the closing of a missed
    requirement -- the distinction matters because the v1.4.3 receipt was not out of compliance.

    mc_mean accumulates in float64 and is therefore CPU/CUDA only (not MPS); production is CUDA.
    """
    mc = _canary()
    from trikaal.eval.predict import MC_DEFAULT_SEED, predict_mu

    n_per = args.bars_total // len(mc.SYMBOLS)
    per_cell = []
    for key in BIAS_CELLS:
        ck = next(c for c in CKPTS if c.key == key)
        if not (Path(ck.path) / "predictor.pt").exists():
            print(f"[skip] {key}: no checkpoint on disk")
            continue
        print(f"[bias] {key}", flush=True)
        prepared, model, tok = build_tokens(ck, n_per=n_per, device=args.device)
        cfg = canary_cfg(15, cap=args.eval_cap, n_per=n_per, seed=args.seed, device=args.device)
        grid = global_decision_grid_ms(cfg, cfg.headline_block)
        rng = np.random.default_rng(cfg.seed)

        mu_exp_full, dec_by_sym = [], {}
        for s, d in prepared.items():
            se = d["se"]
            dec = symbol_decisions(se, grid, cfg)
            if cfg.cap_per_symbol is not None and dec.size > cfg.cap_per_symbol:
                dec = np.sort(rng.choice(dec, size=cfg.cap_per_symbol, replace=False))
            dec_by_sym[s] = dec
            mu_exp_full.append(
                predict_mu(
                    model,
                    tok,
                    d["b_c"],
                    d["b_f"],
                    se.ts,
                    se.sigma,
                    dec,
                    h=cfg.h,
                    seq_len=cfg.seq_len,
                    device=cfg.device,
                    estimator="expectation",
                )
            )
        mu_exp_full = np.concatenate(mu_exp_full)

        # the paired MC subsample: the FIRST n_dec decisions of the first symbol (deterministic)
        s0 = next(iter(prepared))
        sub = dec_by_sym[s0][: args.bias_n_dec]
        d0 = prepared[s0]
        se0 = d0["se"]
        kw = dict(h=cfg.h, seq_len=cfg.seq_len, device=cfg.device)
        mu_e = predict_mu(
            model,
            tok,
            d0["b_c"],
            d0["b_f"],
            se0.ts,
            se0.sigma,
            sub,
            estimator="expectation",
            **kw,
        )
        t0 = time.time()
        mu_m = predict_mu(
            model,
            tok,
            d0["b_c"],
            d0["b_f"],
            se0.ts,
            se0.sigma,
            sub,
            estimator="mc_mean",
            mc_samples=args.bias_mc_samples,
            **kw,
        )
        gap = mu_e - mu_m

        def _st(a):
            return {
                "mean": float(a.mean()),
                "std": float(a.std()),
                "frac_neg": float((a < 0).mean()),
            }

        rec = {
            "cell": key,
            "checkpoint": ck.path,
            "n_dec_subsample": int(sub.size),
            "n_dec_full": int(mu_exp_full.size),
            "expectation_full": _st(mu_exp_full),
            "expectation_on_mc_subsample": _st(mu_e),
            "mc_mean_subsample": _st(mu_m),
            "gap_expectation_minus_mcmean": {
                **_st(gap),
                "abs_mean": float(np.abs(gap).mean()),
            },
            "sign_agreement_frac": float(((mu_e < 0) == (mu_m < 0)).mean()),
            "mc_wall_s": round(time.time() - t0, 1),
        }
        per_cell.append(rec)
        print(
            f"  n={sub.size} exp(mean={rec['expectation_on_mc_subsample']['mean']:+.5f} "
            f"neg={rec['expectation_on_mc_subsample']['frac_neg']:.4f}) "
            f"mc(mean={rec['mc_mean_subsample']['mean']:+.5f} "
            f"neg={rec['mc_mean_subsample']['frac_neg']:.4f}) "
            f"gap_mean={rec['gap_expectation_minus_mcmean']['mean']:+.5f} "
            f"sign_agree={rec['sign_agreement_frac']:.4f} [{rec['mc_wall_s']}s]",
            flush=True,
        )
        del prepared, model, tok

    gaps = [c["gap_expectation_minus_mcmean"]["mean"] for c in per_cell]
    spread = float(max(gaps) - min(gaps)) if len(gaps) > 1 else 0.0
    out = {
        "receipt": "m6_estimator_bias_bound",
        "amendment": "prereg s7 v1.4.3 item 1, REGENERATED under v1.4.5 on the pinned .venv",
        "environment": _env(args.device),
        "estimator_stated": (
            "per-step mean-field decode_latent(E[z]) along a GREEDY argmax token path; exact only "
            "under a linear decoder in z + a deterministic path; mc_mean = unbiased-in-expectation "
            "MC reference over sampled chains"
        ),
        "what_changed_vs_v1_4_3": (
            f"The order read 'expectation vs mc_mean at n >= 256'. v1.4.3 used mc_samples = 256 "
            f"with n_dec = 24, meeting that bar on the MC-sample reading; the DECISION count was "
            f"the weak leg and the one challenged. Here n_dec = {args.bias_n_dec} and mc_samples "
            f"= {args.bias_mc_samples}, so both readings are satisfied. This STRENGTHENS a weak "
            f"leg; it does not close a compliance gap."
        ),
        "h": 15,
        "mc_samples": args.bias_mc_samples,
        "mc_seed": MC_DEFAULT_SEED,
        "gap_spread_across_cells": spread,
        "per_cell": per_cell,
        "n24_claim_status": (
            "The v1.4.3 'balanced frac_neg=0.5' diagnostic was n_dec=24 on a FIXTURE-PLANTED cell. "
            "It supports ONLY that expectation and mc_mean agree on that cell, never that the AR "
            "avoids sign-lock on real data. That withdrawal STANDS regardless of this regeneration."
        ),
    }
    p = Path("runs_manifest/m6_estimator_bias_bound.json")
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[receipt] -> {p}")
    return 0


# -------------------------------- §7 v1.4.6 item 4: sign agreement ON THE TRADED SET at kappa*
TRADED_AGREEMENT_THRESHOLD = (
    0.97  # pre-declared: >= this -> expectation stands; < -> mc_mean headline
)


def run_traded_sign_agreement(args) -> int:
    """Sign agreement between expectation and mc_mean RESTRICTED TO THE TRADED SET at kappa*.

    WHY RESTRICTED. The v1.4.5 figure (0.8398) is over ALL decisions, but only bars that actually
    TRADE turn a sign into a position. The disagreements must concentrate where |mu-hat| is small
    (gap std 0.0114 against mu-hat std 0.0292), and the theta = kappa*c filter already excludes
    ~11% of bars at h=15 — precisely the small-|mu-hat| ones. So the all-decisions figure is a
    LOWER BOUND on the agreement that matters, and this measures the population whose signs
    become positions.

    PRE-DECLARED (supervisor, §7 v1.4.6): traded-set agreement >= 0.97 -> expectation STANDS with
    the bias disclosed; < 0.97 -> the HEADLINE block at kappa* must be scored with mc_mean, with
    expectation retained for VAL and kappa*-selection (a fully pre-specified deterministic
    two-stage procedure bounding MC cost to one block at one kappa).

    kappa* is taken from the FAITHFUL VAL pass under the pre-registered estimator (expectation) —
    the selection stage is unchanged by this measurement.
    """
    mc = _canary()
    from trikaal.eval.predict import MC_DEFAULT_SAMPLES, MC_DEFAULT_SEED, predict_mu

    n_per = args.bars_total // len(mc.SYMBOLS)
    ck = next(c for c in CKPTS if c.key == "acceptance_planted_cell4")
    print(f"[traded] {ck.key} @ h=15", flush=True)
    prepared, model, tok = build_tokens(ck, n_per=n_per, device=args.device)
    cfg = canary_cfg(15, cap=args.eval_cap, n_per=n_per, seed=args.seed, device=args.device)

    # (1) kappa* from the faithful VAL pass under expectation (the pre-registered selector)
    kappa_star = score_pretokenized(model, tok, prepared, cfg)["kappa_star"]
    print(f"  kappa* (VAL, expectation) = {kappa_star}", flush=True)

    # (2) headline grid: expectation mu-hat, then the traded set at kappa*
    cost = CostModel()
    rng = np.random.default_rng(cfg.seed)
    grid = global_decision_grid_ms(cfg, cfg.headline_block)
    kw = dict(h=cfg.h, seq_len=cfg.seq_len, device=cfg.device)
    s0 = next(iter(prepared))
    d0 = prepared[s0]
    se0 = d0["se"]

    traded_all, scored_all = 0, 0
    sub = None
    for sym, d in prepared.items():  # replicate score_cell's rng draw ORDER exactly
        se = d["se"]
        dec = symbol_decisions(se, grid, cfg)
        if cfg.cap_per_symbol is not None and dec.size > cfg.cap_per_symbol:
            dec = np.sort(rng.choice(dec, size=cfg.cap_per_symbol, replace=False))
        y = forward_log_returns(se.raw_ret_close, se.segment_id, cfg.h)
        dec = dec[np.isfinite(y[dec])]
        mu = predict_mu(model, tok, d["b_c"], d["b_f"], se.ts, se.sigma, dec, **kw)
        c_mod = _per_bar_cost(cost, se.sigma, dec, SPREAD_DECILE_FRAC["major"])
        s = positions(mu, kappa_star * c_mod)
        scored_all += int(dec.size)
        traded_all += int((s != 0).sum())
        if sym == s0:
            sub = dec[s != 0][: args.bias_n_dec]  # the TRADED subsample, deterministic
            mu_exp_traded = mu[s != 0][: args.bias_n_dec]
    activity = traded_all / scored_all
    print(
        f"  traded/scored = {traded_all}/{scored_all} (activity {activity:.4f}); "
        f"MC subsample n = {sub.size}",
        flush=True,
    )

    # (3) mc_mean on the SAME traded decisions
    t0 = time.time()
    mu_mc_traded = predict_mu(
        model,
        tok,
        d0["b_c"],
        d0["b_f"],
        se0.ts,
        se0.sigma,
        sub,
        estimator="mc_mean",
        mc_samples=args.bias_mc_samples,
        **kw,
    )
    mc_wall = time.time() - t0

    # (3b) THE PINNED mc_mean is n=32, not the 256 used above as a precise reference. If the <0.97
    # branch fires, production would run n=32 — so its MC noise is a PRECONDITION of that branch
    # being sound, not a detail. Measured here rather than assumed.
    t1 = time.time()
    mu_mc32 = predict_mu(
        model,
        tok,
        d0["b_c"],
        d0["b_f"],
        se0.ts,
        se0.sigma,
        sub,
        estimator="mc_mean",
        mc_samples=MC_DEFAULT_SAMPLES,
        **kw,
    )
    mc32_wall = time.time() - t1

    def _agree(a, b):
        return float(((a < 0) == (b < 0)).mean())

    agree = _agree(mu_exp_traded, mu_mc_traded)
    # would any of these bars ALSO be excluded under mc_mean? (a position can vanish, not just flip)
    c_sub = _per_bar_cost(cost, se0.sigma, sub, SPREAD_DECILE_FRAC["major"])
    s_exp = positions(mu_exp_traded, kappa_star * c_sub)
    s_mc = positions(mu_mc_traded, kappa_star * c_sub)
    s_mc32 = positions(mu_mc32, kappa_star * c_sub)
    same_position = float((s_exp == s_mc).mean())
    agree_pinned = _agree(mu_exp_traded, mu_mc32)
    agree_mc_noise = _agree(mu_mc32, mu_mc_traded)
    print(
        f"  pinned mc_mean n={MC_DEFAULT_SAMPLES}: agree(exp,mc32)={agree_pinned:.4f} "
        f"agree(mc32,mc256)={agree_mc_noise:.4f} [{mc32_wall:.0f}s]",
        flush=True,
    )

    passes = bool(agree >= TRADED_AGREEMENT_THRESHOLD)
    # eval-leg cost of the fallback: mc_mean on the headline block at ONE kappa, whole grid
    per_dec = mc_wall / max(1, sub.size)
    out = {
        "receipt": "m6_traded_sign_agreement",
        "amendment": "prereg s7 v1.4.6 item 4 (supervisor-ordered)",
        "environment": _env(args.device),
        "checkpoint": ck.path,
        "h": 15,
        "kappa_star_from_val_expectation": kappa_star,
        "bars_total": int(args.bars_total),
        "n_per_symbol": int(n_per),
        "eval_cap_per_symbol": args.eval_cap,
        "traded": traded_all,
        "scored": scored_all,
        "activity_decisions_at_kstar": activity,
        "mc_samples": args.bias_mc_samples,
        "mc_seed": MC_DEFAULT_SEED,
        "n_traded_subsample": int(sub.size),
        "sign_agreement_traded_set": agree,
        "identical_position_traded_set": same_position,
        "pinned_mc_samples": MC_DEFAULT_SAMPLES,
        "sign_agreement_vs_PINNED_mc32": agree_pinned,
        "identical_position_vs_PINNED_mc32": float((s_exp == s_mc32).mean()),
        "mc_noise_floor_agreement_mc32_vs_mc256": agree_mc_noise,
        "why_mc32_matters": (
            "The PRE-REGISTERED mc_mean is n=32 (predict.MC_DEFAULT_SAMPLES); the n=256 run above "
            "is a precise reference, not the production setting. If the <0.97 branch fires, the "
            "headline would be scored at n=32, so agreement between mc32 and mc256 is the MC noise "
            "floor -- the amount of position churn the fallback introduces on its own. A fallback "
            "whose own noise floor is comparable to the disagreement it is meant to remove would "
            "trade estimator bias for Monte-Carlo variance rather than fix anything."
        ),
        "threshold": TRADED_AGREEMENT_THRESHOLD,
        "expectation_stands": passes,
        "all_decisions_reference_v1_4_5": 0.8398,
        "pre_declared_rule": (
            "traded-set agreement >= 0.97 -> expectation STANDS (bias disclosed, this measurement "
            "cited); < 0.97 -> the HEADLINE block at kappa* is scored with mc_mean while "
            "expectation is retained for VAL and kappa* selection."
        ),
        "fallback_eval_cost_estimate": {
            "mc_wall_s_for_subsample_n256": round(mc_wall, 1),
            "mc_wall_s_for_subsample_n32_PINNED": round(mc32_wall, 1),
            "measured_s_per_decision_n256": round(per_dec, 4),
            "measured_s_per_decision_n32_PINNED": round(mc32_wall / max(1, sub.size), 4),
            "expectation_multiplier": MC_DEFAULT_SAMPLES,
            "note": (
                "Cost of the <0.97 branch = mc_mean on the headline block at ONE kappa only. "
                "Scale measured_s_per_decision by the headline decision count; the money surface "
                "is UNCAPPED (46,671/symbol) so this is the number that matters, and it is per "
                "GPU not per CPU — this figure is CPU-measured and is an upper bound."
            ),
        },
    }
    p = Path("runs_manifest/m6_traded_sign_agreement.json")
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(
        f"  sign agreement ON TRADED SET = {agree:.4f} (identical position {same_position:.4f}) "
        f"vs threshold {TRADED_AGREEMENT_THRESHOLD} -> expectation_stands={passes} "
        f"[mc {mc_wall:.0f}s]",
        flush=True,
    )
    print(f"[receipt] -> {p}")
    return 0


# --------------- §7 v1.4.7 item 4 forensics: M4a split-half floor + M4b benign perturbation
def run_estimator_forensics(args) -> int:
    """M4a (split-half MC noise floor) + M4b (is the perturbation benign?) in ONE rollout budget.

    M4a AS SPECIFIED IS IMPOSSIBLE: ``_rollout_mc_mean`` accumulates ``acc += cum`` and returns
    ``acc / mc_samples`` — the 256 per-decision samples were NEVER materialized, so they cannot be
    partitioned after the fact. Retaining them would mean editing predict.py, which is the Gate-A
    anchored instrument. Instead the SAME 256 sample-rollouts are spent as TWO INDEPENDENT n=128
    halves at different mc_seeds, which yields the split-half floor DIRECTLY (measured, not
    1/sqrt(n)-extrapolated) and their average is an n=256 estimate. So M4a costs nothing beyond the
    M4b run that was ordered anyway — no extra rollouts are spent to recover discarded samples.

    M4b tests the argument that makes a biased estimator acceptable in a pre-registered test: that
    the disagreements sit where |mu-hat| is small, so the induced position noise pulls measured IR
    TOWARD ZERO (biasing to NULL, the conservative direction) and cannot manufacture a false
    SURVIVES. On the disagreement set it reports the realized forward-return distribution and
    whether flip direction correlates with realized return. The raw per-decision vectors are
    PERSISTED this time so no future question about them needs another rollout.
    """
    mc = _canary()
    from trikaal.eval.predict import MC_DEFAULT_SAMPLES, MC_DEFAULT_SEED, predict_mu

    n_per = args.bars_total // len(mc.SYMBOLS)
    ck = next(c for c in CKPTS if c.key == "acceptance_planted_cell4")
    print(f"[forensics] {ck.key} @ h=15", flush=True)
    prepared, model, tok = build_tokens(ck, n_per=n_per, device=args.device)
    cfg = canary_cfg(15, cap=args.eval_cap, n_per=n_per, seed=args.seed, device=args.device)
    kappa_star = score_pretokenized(model, tok, prepared, cfg)["kappa_star"]
    print(f"  kappa* = {kappa_star}", flush=True)

    cost = CostModel()
    rng = np.random.default_rng(cfg.seed)
    grid = global_decision_grid_ms(cfg, cfg.headline_block)
    kw = dict(h=cfg.h, seq_len=cfg.seq_len, device=cfg.device)
    s0 = next(iter(prepared))
    d0, se0 = prepared[s0], prepared[s0]["se"]
    y_full = forward_log_returns(se0.raw_ret_close, se0.segment_id, cfg.h)

    sub = mu_exp = None
    for sym, d in prepared.items():  # replicate score_cell's rng draw ORDER
        se = d["se"]
        dec = symbol_decisions(se, grid, cfg)
        if cfg.cap_per_symbol is not None and dec.size > cfg.cap_per_symbol:
            dec = np.sort(rng.choice(dec, size=cfg.cap_per_symbol, replace=False))
        y = forward_log_returns(se.raw_ret_close, se.segment_id, cfg.h)
        dec = dec[np.isfinite(y[dec])]
        if sym != s0:
            continue
        mu = predict_mu(model, tok, d["b_c"], d["b_f"], se.ts, se.sigma, dec, **kw)
        c_mod = _per_bar_cost(cost, se.sigma, dec, SPREAD_DECILE_FRAC["major"])
        traded = positions(mu, kappa_star * c_mod) != 0
        sub, mu_exp = dec[traded][: args.bias_n_dec], mu[traded][: args.bias_n_dec]

    half = args.bias_mc_samples // 2
    halves = []
    for i, seed_off in enumerate((0, 7919)):  # two INDEPENDENT mc streams, same total budget
        t0 = time.time()
        halves.append(
            predict_mu(
                model,
                tok,
                d0["b_c"],
                d0["b_f"],
                se0.ts,
                se0.sigma,
                sub,
                estimator="mc_mean",
                mc_samples=half,
                mc_seed=MC_DEFAULT_SEED + seed_off,
                **kw,
            )
        )
        print(f"  mc half {i + 1} (n={half}) in {time.time() - t0:.0f}s", flush=True)
    a, b = halves
    mc_combined = 0.5 * (a + b)
    y_sub = y_full[sub]

    def _agree(u, v):
        return float(((u < 0) == (v < 0)).mean())

    # ---- M4a: the split-half floor, MEASURED
    m4a = {
        "n_per_half": int(half),
        "agree_half_a_vs_half_b": _agree(a, b),
        "agree_expectation_vs_combined_n256": _agree(mu_exp, mc_combined),
        "pinned_n32_floor_v1_4_6": 0.8164,
        "expectation_vs_mc256_v1_4_6": 0.9102,
    }
    # disagreement WITH the reference decomposed: how much is reference variance?
    m4a["expectation_disagreement"] = 1.0 - m4a["agree_expectation_vs_combined_n256"]
    m4a["reference_self_disagreement_at_half"] = 1.0 - m4a["agree_half_a_vs_half_b"]

    # §7 v1.4.7 CORRECTION (supervisor-issued). The two agreements have DIFFERENT error budgets, so
    # their equality is NOT the null expectation and "indistinguishable" is wrong. Under one
    # error-scale model in units of the per-sample MC sd (sigma):
    #   D(half_a, half_b)  ~ sqrt(2)*sigma/sqrt(n_half)          (two independent halves)
    #   D(exp, combined)   ~ sqrt(s_exp^2 + (sigma/sqrt(2*n_half))^2)
    # Equal disagreement therefore IMPLIES s_exp, which is DERIVED here rather than asserted.
    s_half = float(np.sqrt(2.0) / np.sqrt(half))  # scale of the split-half comparison
    s_ref = float(1.0 / np.sqrt(2 * half))  # the reference's own noise at n = 2*half
    s_exp_sq = s_half**2 - s_ref**2
    s_exp = float(np.sqrt(s_exp_sq)) if s_exp_sq > 0 else float("nan")
    s_mc32 = float(1.0 / np.sqrt(MC_DEFAULT_SAMPLES))
    m4a["CORRECTION_v1_4_7"] = (
        "BOTH earlier readings are WITHDRAWN. agree(half_a,half_b) compares two INDEPENDENT "
        "n=half estimates while agree(expectation,combined) compares expectation to ONE n=2*half "
        "estimate, so the error budgets differ and equal disagreement is NOT the null expectation. "
        "It IMPLIES s_exp = sqrt(2/n_half - 1/(2*n_half)) sigma, larger than the reference's own "
        "noise -- expectation's deviation is REAL, not indistinguishable from reference variance. "
        "WITHDRAWN: the builder's 'statistically indistinguishable'. EQUALLY WITHDRAWN: the "
        "ruling's 'most of the gap is reference variance'. The exact 228/256 tie should not be "
        "expected even in principle under this model."
    )
    m4a["DECISION_RELEVANT_v1_4_7"] = (
        "The decision rests on this, not on 'indistinguishable': mc@32's noise scale is "
        "sigma/sqrt(32), so EXPECTATION'S ERROR IS ~0.61x THE PINNED ESTIMATOR'S AT 1/32 THE COST."
    )
    m4a["error_scale_model"] = {
        "units": "per-sample MC standard deviation (sigma)",
        "split_half_comparison_scale": round(s_half, 4),
        "reference_noise_at_n256": round(s_ref, 4),
        "implied_s_expectation": round(s_exp, 4),
        "ratio_expectation_over_reference_noise": round(s_exp / s_ref, 2),
        "share_of_squared_budget_that_is_reference_variance": round(s_ref**2 / s_half**2, 4),
        "mc32_noise_scale": round(s_mc32, 4),
        "expectation_error_over_mc32": round(s_exp / s_mc32, 2),
        "self_consistency": {
            "D(exp,mc256)": {
                "observed": round(1.0 - m4a["agree_expectation_vs_combined_n256"], 4),
                "predicted_scale": round(float(np.hypot(s_exp, s_ref)), 4),
            },
            "D(half_a,half_b)": {
                "observed": round(1.0 - m4a["agree_half_a_vs_half_b"], 4),
                "predicted_scale": round(s_half, 4),
            },
            "D(mc32,mc256)": {
                "observed": round(1.0 - m4a["pinned_n32_floor_v1_4_6"], 4),
                "predicted_scale": round(float(np.hypot(s_mc32, s_ref)), 4),
            },
            "D(exp,mc32)": {
                "observed": round(1.0 - 0.8125, 4),
                "predicted_scale": round(float(np.hypot(s_exp, s_mc32)), 4),
            },
        },
        "note": (
            "Four measurements, ONE error-scale model, no free parameters once s_exp is fixed "
            "by the first pair. The D(*,mc32) values come from the v1.4.6 traded-set receipt."
        ),
    }

    # ---- M4b: is the perturbation benign?
    dis = (mu_exp < 0) != (mc_combined < 0)
    n_dis = int(dis.sum())
    out_m4b: dict = {"n_disagreements": n_dis, "n_traded_subsample": int(sub.size)}
    if n_dis >= 2:
        yd = y_sub[dis]
        flip_to_long = np.where(mc_combined[dis] > 0, 1.0, -1.0)
        out_m4b.update(
            {
                "abs_mu_on_disagreements": float(np.abs(mu_exp[dis]).mean()),
                "abs_mu_on_agreements": float(np.abs(mu_exp[~dis]).mean()),
                "realized_return_on_disagreements": {
                    "mean": float(yd.mean()),
                    "std": float(yd.std()),
                    "median": float(np.median(yd)),
                    "frac_positive": float((yd > 0).mean()),
                },
                "corr_flip_direction_vs_realized_return": float(
                    np.corrcoef(flip_to_long, yd)[0, 1]
                ),
                # the DIRECT test: on the disagreement bars, does either estimator's position earn?
                "mean_signed_return_expectation_positions": float(
                    (np.sign(mu_exp[dis]) * yd).mean()
                ),
                "mean_signed_return_mc_positions": float((np.sign(mc_combined[dis]) * yd).mean()),
                "mean_signed_return_agreement_set": float(
                    (np.sign(mu_exp[~dis]) * y_sub[~dis]).mean()
                ),
            }
        )
    out = {
        "receipt": "m6_estimator_forensics",
        "amendment": "prereg s7 v1.4.7 item 4 M4a/M4b (supervisor-ordered)",
        "environment": _env(args.device),
        "checkpoint": ck.path,
        "h": 15,
        "kappa_star": kappa_star,
        "mc_total_sample_budget": int(args.bias_mc_samples),
        "pinned_mc_samples": MC_DEFAULT_SAMPLES,
        "M4a_split_half": m4a,
        "M4b_benign_perturbation": out_m4b,
        "M4a_method_note": (
            "The 256 per-decision MC samples of the v1.4.6 run were NOT retained (_rollout_mc_mean "
            "accumulates and returns only the mean), so partitioning them after the fact is "
            "impossible without editing the anchored predict.py. The same 256-sample budget is "
            "instead spent as two independent n=128 halves, giving the split-half floor DIRECTLY "
            "rather than by 1/sqrt(n) extrapolation, at no extra rollout cost over the ordered M4b."
        ),
        "caveats": (
            "n=256 traded decisions on ONE fixture-planted cell; point estimates, NOT significance "
            "claims; no paired CI computed on any difference."
        ),
    }
    p = Path("runs_manifest/m6_estimator_forensics.json")
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    np.savez_compressed(
        "runs_manifest/m6_estimator_forensics_vectors.npz",
        decisions=sub,
        mu_expectation=mu_exp,
        mc_half_a=a,
        mc_half_b=b,
        realized_return=y_sub,
    )
    print(
        f"  M4a split-half agree(a,b) = {m4a['agree_half_a_vs_half_b']:.4f} "
        f"| agree(exp, combined) = {m4a['agree_expectation_vs_combined_n256']:.4f}"
    )
    if n_dis >= 2:
        print(
            f"  M4b n_dis={n_dis} corr(flip, y) = "
            f"{out_m4b['corr_flip_direction_vs_realized_return']:+.4f} "
            f"| signed-return exp {out_m4b['mean_signed_return_expectation_positions']:+.6f} "
            f"mc {out_m4b['mean_signed_return_mc_positions']:+.6f} "
            f"agreeset {out_m4b['mean_signed_return_agreement_set']:+.6f}"
        )
        print(
            f"  |mu| on dis {out_m4b['abs_mu_on_disagreements']:.5f} vs "
            f"agree {out_m4b['abs_mu_on_agreements']:.5f}"
        )
    print(f"[receipt] -> {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true", help="h-sweep over every checkpoint on disk")
    ap.add_argument("--guard-stability", action="store_true", help="BLOCKING band-stability check")
    ap.add_argument("--rescore", action="store_true", help="regenerate L1/L2/L4 at full precision")
    ap.add_argument("--bias-bound", action="store_true", help="item-1 expectation vs mc_mean")
    ap.add_argument("--bias-n-dec", type=int, default=256, help="paired MC subsample size (>=256)")
    ap.add_argument("--bias-mc-samples", type=int, default=256)
    ap.add_argument(
        "--traded-sign-agreement",
        action="store_true",
        help="item 4: sign agreement restricted to the traded set at kappa*",
    )
    ap.add_argument(
        "--estimator-forensics",
        action="store_true",
        help="item 4 M4a split-half MC floor + M4b benign-perturbation test",
    )
    ap.add_argument("--bars-total", type=int, default=42_000_000)
    ap.add_argument("--eval-cap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-uncapped", action="store_true")
    args = ap.parse_args()
    if args.sweep:
        return run_sweep(args)
    if args.guard_stability:
        return run_guard_stability(args)
    if args.rescore:
        return run_rescore(args)
    if args.bias_bound:
        return run_bias_bound(args)
    if args.traded_sign_agreement:
        return run_traded_sign_agreement(args)
    if args.estimator_forensics:
        return run_estimator_forensics(args)
    ap.error(
        "pass --sweep, --guard-stability, --rescore, --bias-bound, --traded-sign-agreement "
        "or --estimator-forensics"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
