"""EXACT A_p SKILL-VS-BIAS — seed 4 only. Re-scores per-symbol mu-hat over the money grid.

WHAT THIS FIXES. ``scripts/m6_skill_vs_bias.py`` had to benchmark over ``V_p`` (every symbol with
a VALID decision) because ``A_p`` — the symbols the model actually TRADED — is not persisted. This
script recomputes A_p directly, so the benchmark becomes the one that was specified:

    b_p = sigma * (1/|A_p|) * sum_{n in A_p} r_{n,p}       sigma = the seed's MODAL sign

SEED 4 ONLY, BY RULING. Seeds 0 and 2 are not authorized and buying them would be waste: their
V_p benchmark is -0.7607 and -1.2640 against models at +1.2665 and +2.1715, so an A_p re-basing
would have to swing the benchmark by more than two full IR units to touch the conclusion. Their
inversion is structural. Seed 4 is the only seed whose answer is unknown.

★ THE PRE-DECLARED READING (supervisor, BEFORE this was run — recorded so no outcome can be
re-read afterwards as a surprise). All three are publishable; none is a failure:
  * delta EXCLUDES ZERO POSITIVE -> seed 4 also demonstrates timing skill; the claim broadens to
    three of three and is DEFENDED rather than asserted.
  * delta ~ ZERO or STRADDLES     -> unchanged from V_p; publish "cannot exclude", now on the
    EXACT benchmark. The V_p-vs-A_p caveat DROPS, which is itself worth the money.
  * delta NEGATIVE                -> seed 4 UNDERPERFORMS holding long on the names it traded.
    That goes in the abstract's own limitation sentence, not a footnote.

★ THE BUILT-IN CONTROL ARM. This script recomputes the model's OWN gross period series from
scratch. That series is already banked (``headline_series`` + the flat cost), so the recomputation
is checked against it before any benchmark number is read. If the reconstruction does not match,
the pipeline is wrong and NO VERDICT IS EMITTED — a probe must refuse when its own control fails.
This is not decoration: it validates the tokenization, the decision grid, the arm transform, the
cost model, the position filter and the pooling, all at once, against a number produced weeks ago
on different hardware.

★ THE DATA SOURCE IS PINNED EXPLICITLY, per the shared-input rule this project just landed:
``--lake processed/universe_bars`` is a REQUIRED default and never inherited from
``trikaal.data.lake.connect()``, whose default is the one-symbol M2 lake. That exact default
silently served a 4x-shorter normalization history to a gate that passed 9/9.

Resumable: each symbol's per-decision arrays are written under ``--work`` as they complete, so an
interrupted run resumes instead of re-paying. Exit 0 PASS · 2 control-arm failure or refusal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake  # noqa: E402
from trikaal.demo.inference import PINNED_TRAIN_FRAC, PINNED_WINDOW, load_unit  # noqa: E402
from trikaal.eval.costs import CostModel, load_spread_deciles  # noqa: E402
from trikaal.eval.diagnostics import gross_series  # noqa: E402
from trikaal.eval.harness import (  # noqa: E402
    HEADLINE_COST,
    _per_bar_cost,
    forward_log_returns,
)
from trikaal.eval.metrics import information_ratio  # noqa: E402
from trikaal.eval.paired_bootstrap import (  # noqa: E402
    BOOT_B,
    BOOT_SEED,
    block_length,
    paired_delta_ir_bootstrap,
)
from trikaal.eval.predict import predict_mu  # noqa: E402
from trikaal.eval.strategy import positions  # noqa: E402
from trikaal.eval.xsection import (  # noqa: E402
    SymbolEval,
    XSectionConfig,
    primary_region_grid_ms,
    symbol_decisions,
)
from trikaal.model.attention_mode import MODE_SDPA, set_attention_backend  # noqa: E402
from trikaal.run.matrix import load_symbol_arrays  # noqa: E402
from trikaal.train.arms import select_arm  # noqa: E402
from trikaal.train.token_stream import tokenize_features  # noqa: E402
from trikaal.utils.seeding import set_determinism  # noqa: E402

SEED = 4
ARTIFACT = REPO / "runs_cloud/results/r2/cell1_seed4_eval.json"
ARM = "ohlcv"
H = 15
SEQ_LEN = 512
# The money-run pin. Read from the artifact and asserted, never assumed.
KAPPA_STAR = 3.0


def build_cfg(device: str) -> XSectionConfig:
    return XSectionConfig(
        window_start=PINNED_WINDOW[0],
        window_end=PINNED_WINDOW[1],
        train_frac=PINNED_TRAIN_FRAC,
        h=H,
        seq_len=SEQ_LEN,
        cap_per_symbol=None,
        device=device,
        money=False,
    )


def score_symbol(sym, unit, con, cfg, grid, spread, boundary, device, limit_dec=0):
    """Per-decision (period, position, forward return) for ONE symbol — the production path."""
    raw = load_symbol_arrays(con, sym, boundary_ms=boundary, tail_bars=None)
    if raw["x"].shape[0] == 0:
        return None
    se = SymbolEval(
        symbol=sym,
        x=raw["x"],
        mask=raw["mask"],
        segment_id=raw["segment_id"],
        ts=raw["ts"],
        sigma=raw["sigma"],
        raw_ret_close=raw["raw_ret_close"],
    )
    y = forward_log_returns(se.raw_ret_close, se.segment_id, cfg.h)
    dec = symbol_decisions(se, grid, cfg)
    finite = np.isfinite(y[dec])
    dec = dec[finite]
    # ★ THE DRY-RUN CAP MUST BOUND THE WORK, NOT THE RESULT. The first version truncated the
    # returned arrays AFTER predict_mu had already scored all 35,063 decisions, so the "2 symbols
    # x 8 decisions" rehearsal was a full-scale run wearing a cap, and it timed out at ten
    # minutes. A rehearsal that costs what the real thing costs cannot rehearse anything.
    if limit_dec:
        dec = dec[:limit_dec]
    if dec.size == 0:
        return None
    x_arm, m_arm = select_arm(np.asarray(se.x, np.float32), se.mask, ARM)
    b_c, b_f = tokenize_features(
        unit.tok, x_arm, m_arm, se.segment_id, window=cfg.seq_len, device=device
    )
    mu = predict_mu(
        unit.model,
        unit.tok,
        b_c,
        b_f,
        se.ts,
        se.sigma,
        dec,
        h=cfg.h,
        seq_len=cfg.seq_len,
        device=device,
        estimator=cfg.mu_estimator,
    )
    c_mod = _per_bar_cost(CostModel(), se.sigma, dec, spread)
    s = positions(mu, KAPPA_STAR * c_mod)
    pidx = np.searchsorted(grid, se.ts[dec])
    return {
        "pidx": pidx.astype(np.int64),
        "s": s.astype(np.int64),
        "y": y[dec].astype(np.float64),
        "mu": mu.astype(np.float64),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", type=Path, default=Path("processed/universe_bars"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--work", type=Path, default=REPO / "runs_cloud/skill_exact")
    ap.add_argument("--out", type=Path, default=REPO / "runs_manifest/m6_skill_vs_bias_exact.json")
    ap.add_argument("--limit-symbols", type=int, default=0, help="DRY RUN ONLY: cap symbols")
    ap.add_argument("--limit-decisions", type=int, default=0, help="DRY RUN ONLY: cap decisions")
    args = ap.parse_args()
    dry = bool(args.limit_symbols or args.limit_decisions)

    doc = json.loads(ARTIFACT.read_text())
    assert int(doc["seed"]) == SEED and int(doc["cell_id"]) == 1
    ks = float(doc["kappa_star_by_h"][str(H)])
    if ks != KAPPA_STAR:
        print(f"kappa* is {ks}, expected {KAPPA_STAR} — REFUSING", file=sys.stderr)
        return 2
    symbols = list(doc["meta"]["symbols"])
    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]

    frac_neg = float(doc["mu_diag"]["frac_negative"])
    sigma_modal = -1.0 if frac_neg > 0.5 else 1.0
    print(
        f"[cfg]  seed {SEED}  kappa* {ks}  modal {'SHORT' if sigma_modal < 0 else 'LONG'} "
        f"(frac_neg {frac_neg:.4f})  device {args.device}  lake {args.lake}"
    )

    # ★ THE MONEY RUN'S DETERMINISM POSTURE, REPRODUCED — the control arm compares against a
    # series produced weeks ago on a 4090 under sdpa_deterministic + deterministic_algorithms +
    # CUBLAS_WORKSPACE_CONFIG=:4096:8. Invariant 7 is honest that deterministic attention is
    # NECESSARY, NOT SUFFICIENT, so this does not guarantee bit-exactness across boxes; it removes
    # the sources we CAN remove, and the control arm is what actually decides whether the
    # reproduction is close enough to read a benchmark off.
    set_determinism(SEED, deterministic_algorithms=True)
    unit = load_unit(SEED, device=args.device)
    n_attn = set_attention_backend(unit.model, MODE_SDPA)
    print(f"[det]  attention={MODE_SDPA} on {n_attn} modules; deterministic_algorithms=True")
    print(
        f"[unit] identity verified; {unit.n_params_realized:,} params ({unit.n_params_mtp:,} MTP)"
    )

    cfg = build_cfg(args.device)
    con = connect_lake(args.lake)
    n_sym_lake = con.execute("SELECT count(DISTINCT symbol) FROM bars").fetchone()[0]
    print(f"[lake] {args.lake} -> {n_sym_lake} distinct symbols (PINNED, not inherited)")
    boundary = calendar_boundary_ms(cfg.window_start, cfg.window_end, cfg.train_frac)
    grid = primary_region_grid_ms(cfg)
    t = int(grid.size)
    spreads = load_spread_deciles(REPO / "runs_manifest/m6_spread_deciles.json")
    args.work.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    parts = {}
    for k, sym in enumerate(symbols, 1):
        cache = args.work / f"{sym}.npz"
        if cache.exists() and not dry:
            z = np.load(cache)
            parts[sym] = {kk: z[kk] for kk in z.files}
            print(f"[{k}/{len(symbols)}] {sym} resumed", flush=True)
            continue
        r = score_symbol(
            sym, unit, con, cfg, grid, spreads[sym], boundary, args.device, args.limit_decisions
        )
        if r is None:
            print(f"[{k}/{len(symbols)}] {sym} SKIP (no valid decisions)", flush=True)
            continue
        parts[sym] = r
        if not dry:
            np.savez(cache, **r)
        el = time.perf_counter() - t0
        print(
            f"[{k}/{len(symbols)}] {sym} n={r['s'].size:,} traded={(r['s'] != 0).sum():,} "
            f"elapsed {el / 60:.1f}m eta {el / k * (len(symbols) - k) / 60:.1f}m",
            flush=True,
        )

    # ── pool into the two period series ───────────────────────────────────────────────────────
    g_sum = np.zeros(t)
    g_cnt = np.zeros(t, np.int64)
    b_sum = np.zeros(t)
    for r in parts.values():
        act = r["s"] != 0
        p, s, y = r["pidx"][act], r["s"][act].astype(np.float64), r["y"][act]
        np.add.at(g_sum, p, s * y)  # the model: signed
        np.add.at(b_sum, p, sigma_modal * y)  # constant direction on the SAME names
        np.add.at(g_cnt, p, 1)
    nz = g_cnt > 0
    g = np.zeros(t)
    b = np.zeros(t)
    g[nz] = g_sum[nz] / g_cnt[nz]
    b[nz] = b_sum[nz] / g_cnt[nz]

    # ── CONTROL ARM: the recomputed model series must reproduce the banked one ────────────────
    banked = gross_series(
        np.array(doc["headline_series"], dtype=np.float64), headline_cost=HEADLINE_COST
    )
    banked_active = np.array(doc["headline_series"], dtype=np.float64) != 0.0
    ctrl = {
        "n_active_recomputed": int(nz.sum()),
        "n_active_banked": int(banked_active.sum()),
        "active_set_identical": bool(np.array_equal(nz, banked_active)),
        "max_abs_diff_gross": float(np.max(np.abs(g - banked))) if not dry else None,
        "corr_gross": float(np.corrcoef(g[nz], banked[nz])[0, 1]) if nz.sum() > 2 else None,
        "ir_recomputed": information_ratio(g, H),
        "ir_banked": information_ratio(banked, H),
    }
    print(
        f"[ctrl] active recomputed {ctrl['n_active_recomputed']:,} vs banked "
        f"{ctrl['n_active_banked']:,}; identical={ctrl['active_set_identical']}; "
        f"IR recomputed {ctrl['ir_recomputed']:+.4f} vs banked {ctrl['ir_banked']:+.4f}"
    )

    if dry:
        print("[dry]  DRY RUN — no verdict emitted, control arm not applicable at reduced scale")
        return 0

    # A probe must REFUSE when its own control fails.
    if not ctrl["active_set_identical"] or abs(ctrl["ir_recomputed"] - ctrl["ir_banked"]) > 0.01:
        print(
            "REFUSING TO EMIT: the recomputed model series does not reproduce the banked one. "
            "The pipeline is not the scored pipeline, so no benchmark number is readable.",
            file=sys.stderr,
        )
        json.dump(
            {"verdict": "CONTROL ARM FAILED", "control": ctrl},
            args.out.open("w"),
            indent=2,
            default=str,
        )
        return 2

    ir_a = information_ratio(g, H)
    ir_b = information_ratio(b, H)
    boot = paired_delta_ir_bootstrap(g, b, h=H)
    out = {
        "script": "scripts/m6_skill_vs_bias_exact.py",
        "schema": "m6_skill_vs_bias_exact_v1",
        "seed": SEED,
        "benchmark": (
            "EXACT A_p — constant direction at the modal sign on the names THE MODEL TRADED"
        ),
        "modal_sign": "SHORT" if sigma_modal < 0 else "LONG",
        "frac_negative_full_grid": frac_neg,
        "a_ir_gross_model": ir_a,
        "b_ir_gross_constant_direction_Ap": ir_b,
        "delta_a_minus_b": boot.delta_ir,
        "delta_ci95_lower": boot.ci_lower,
        "se_boot": boot.se_boot,
        "bias_only_share_of_model_ir": (ir_b / ir_a) if ir_a else None,
        "control_arm": ctrl,
        "pins": {
            "h": H,
            "kappa_star": ks,
            "headline_cost": HEADLINE_COST,
            "bootstrap_B": BOOT_B,
            "bootstrap_seed": BOOT_SEED,
            "block_length": block_length(t),
            "lake_root": str(args.lake),
            "n_symbols": len(symbols),
            "n_periods": t,
        },
        "PRE_DECLARED_READING": {
            "declared_by": "supervisor, BEFORE the run",
            "excludes_zero_positive": (
                "seed 4 also demonstrates timing skill; the claim broadens to 3 of 3 and is "
                "DEFENDED rather than asserted"
            ),
            "straddles_zero": (
                "unchanged from V_p; publish 'cannot exclude' on the EXACT benchmark, and the "
                "V_p-vs-A_p caveat DROPS"
            ),
            "negative": (
                "seed 4 UNDERPERFORMS holding long on the names it traded; goes in the abstract's "
                "own limitation sentence, not a footnote"
            ),
        },
        "INTERPRETATION": "NONE — the builder reports; the ruling is the supervisor's.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(
        f"\n(a) model {ir_a:+.4f}   (b) constant-direction on A_p {ir_b:+.4f}   "
        f"delta {boot.delta_ir:+.4f}   CI95_lower {boot.ci_lower:+.4f}"
    )
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
