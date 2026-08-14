"""IS THE GROSS IR SKILL, OR DIRECTIONAL BIAS TIMES SAMPLE DRIFT? $0, banked artifacts + lake.

THE CLAIM UNDER TEST. The paper states the models have "real predictive skill", resting on
positive GROSS information ratios (+1.27 / +2.17 / +1.14 for seeds 0 / 2 / 4). But the authoritative
full-grid mu-diag says seed 4 is LONG on 92.66% of its 1,402,520 decisions while seeds 0 and 2 are
SHORT on 67.73% and 65.52%. A model that holds one direction almost always, in a sample that drifts
that way, earns a positive gross IR with NO TIMING SKILL AT ALL. Bias x drift is indistinguishable
from skill in the number currently quoted, so this measures the bias-only alternative directly.

    (a) IR_gross of the model AS SCORED — from the banked headline series.
    (b) IR_gross of a CONSTANT-DIRECTION strategy at that seed's MODAL SIGN, on the same active
        periods, same flat netting, same full-calendar-grid construction.

(b) keeps the direction and throws the timing away. If (b) reproduces (a), the skill claim
collapses into a beta claim. If (a) clearly exceeds (b), the claim survives and is DEFENDED.

--------------------------------------------------------------------------------------------
★ WHAT IS $0-EXACT AND WHAT IS NOT — READ BEFORE QUOTING ANY NUMBER
--------------------------------------------------------------------------------------------
The benchmark needs, per period, the mean forward return of the symbols THE MODEL TRADED:

    b_p = sigma * (1/|A_p|) * sum_{n in A_p} r_{n,p}

``A_p`` is model-dependent, and ``write_cell_eval_artifact`` persists only the POOLED net series —
no per-symbol positions, no per-symbol forward returns. So A_p is NOT recoverable from banked
artifacts. (Same class as ``break_even`` being computed on the rented box and discarded.)

It would be recoverable if the execution filter never bound, because then traded == scored and
A_p would equal the model-independent valid-decision set. IT DOES BIND HERE: decision-activity at
kappa* is 0.0926 / 0.0193 / 0.4702, and it moves with kappa (0.339 -> 0.0926 for seed 0). The
v1.4.4 "theta never binds" finding was about the FIXTURE and does not transfer to the money run.

SO THIS SCRIPT IMPLEMENTS THE BENCHMARK OVER ``V_p`` — every symbol with a VALID DECISION at
period p — RATHER THAN ``A_p``. That substitution is the one deviation from the specified test and
it is stated in the receipt, not buried:

  * V_p is model-INDEPENDENT and lake-only, hence exact and free.
  * V_p is the full evaluated cross-section; A_p is the model's selected subset of it. So (b) here
    measures "constant direction on the cross-section the model was scored over", not "constant
    direction on the names this model happened to pick".
  * DIRECTION OF THE DIFFERENCE IS NOT KNOWN A PRIORI. Selection could help or hurt the benchmark,
    so this is NOT a conservative bound in a provable sense, and it is not claimed as one.

The exactly-specified version needs a re-score (per-symbol mu-hat over the money grid) — local CPU,
still $0 in money, but hours in wall-clock. Costed in the receipt; not run without a ruling.

--------------------------------------------------------------------------------------------
Statistics: the PINNED recipe, unmodified — paired moving-block bootstrap, B=10,000, seed
20260704, block length ceil(sqrt(T)) = 188. Paired is correct here and not a convenience: both
series live on the same grid and share the market component, which cancels inside every replicate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake  # noqa: E402
from trikaal.demo.inference import (  # noqa: E402
    DEMO_H,
    LAKE_ROOT,
    PINNED_TRAIN_FRAC,
    PINNED_WINDOW,
)
from trikaal.eval.diagnostics import gross_series  # noqa: E402
from trikaal.eval.harness import HEADLINE_COST, forward_log_returns  # noqa: E402
from trikaal.eval.metrics import information_ratio  # noqa: E402
from trikaal.eval.paired_bootstrap import (  # noqa: E402
    BOOT_B,
    BOOT_SEED,
    block_length,
    paired_delta_ir_bootstrap,
)
from trikaal.eval.xsection import (  # noqa: E402
    SymbolEval,
    XSectionConfig,
    primary_region_grid_ms,
    symbol_decisions,
)
from trikaal.run.matrix import load_symbol_arrays  # noqa: E402

ARTIFACTS = {
    0: REPO / "runs_cloud/results/r0/cell1_seed0_eval.json",
    2: REPO / "runs_cloud/results/r1/cell1_seed2_eval.json",
    4: REPO / "runs_cloud/results/r2/cell1_seed4_eval.json",
}


def market_mean_forward_return(symbols: list[str], *, device: str = "cpu") -> tuple:
    """Per-period cross-sectional mean forward log-return over every VALID decision (V_p).

    Lake-only: no model is loaded and no mu-hat is formed, so this is exact and free. The
    per-period breadth is returned beside it because a mean over one symbol is not a market.
    """
    cfg = XSectionConfig(
        window_start=PINNED_WINDOW[0],
        window_end=PINNED_WINDOW[1],
        train_frac=PINNED_TRAIN_FRAC,
        h=DEMO_H,
        seq_len=512,
        cap_per_symbol=None,
        device=device,
        money=False,
    )
    con = connect_lake(LAKE_ROOT)
    boundary = calendar_boundary_ms(cfg.window_start, cfg.window_end, cfg.train_frac)
    grid = primary_region_grid_ms(cfg)
    t = int(grid.size)
    tot = np.zeros(t, dtype=np.float64)
    cnt = np.zeros(t, dtype=np.int64)

    for k, sym in enumerate(symbols, 1):
        raw = load_symbol_arrays(con, sym, boundary_ms=boundary, tail_bars=None)
        if raw["x"].shape[0] == 0:
            print(f"  [skip] {sym}: no bars", flush=True)
            continue
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
        if dec.size == 0:
            print(f"  [skip] {sym}: no valid decisions on the money grid", flush=True)
            continue
        pidx = np.searchsorted(grid, se.ts[dec])
        yv = y[dec]
        ok = np.isfinite(yv)
        np.add.at(tot, pidx[ok], yv[ok])
        np.add.at(cnt, pidx[ok], 1)
        if k % 10 == 0:
            print(f"  [{k}/{len(symbols)}] {sym}", flush=True)

    market = np.zeros(t, dtype=np.float64)
    nz = cnt > 0
    market[nz] = tot[nz] / cnt[nz]
    return market, cnt, grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "runs_manifest/m6_skill_vs_bias.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    docs = {s: json.loads(p.read_text()) for s, p in ARTIFACTS.items()}
    symbols = list(docs[0]["meta"]["symbols"])
    print(f"[market] cross-sectional mean forward return over {len(symbols)} evaluated symbols")
    market, breadth, _grid = market_mean_forward_return(symbols, device=args.device)
    t = int(market.size)
    print(f"[market] T={t:,} periods; breadth mean {breadth.mean():.2f}, max {breadth.max()}")

    rows = []
    for seed, doc in sorted(docs.items()):
        net = np.array(doc["headline_series"], dtype=np.float64)
        if net.size != t:
            raise ValueError(f"seed {seed}: series {net.size} != market grid {t}")
        g = gross_series(net, headline_cost=HEADLINE_COST)
        active = net != 0.0

        frac_neg = float(doc["mu_diag"]["frac_negative"])
        sigma = -1.0 if frac_neg > 0.5 else 1.0

        # (b) constant direction at the modal sign, SAME active periods, same construction
        bench = np.where(active, sigma * market, 0.0)

        ir_a = information_ratio(g, DEMO_H)
        ir_b = information_ratio(bench, DEMO_H)
        boot = paired_delta_ir_bootstrap(g, bench, h=DEMO_H)

        # how much of the model's gross IR the bias-only strategy reproduces
        share = (ir_b / ir_a) if ir_a != 0 else float("nan")
        rows.append(
            {
                "seed": seed,
                "modal_sign": "SHORT" if sigma < 0 else "LONG",
                "frac_negative_full_grid": frac_neg,
                "modal_share": max(frac_neg, 1.0 - frac_neg),
                "decision_activity_at_kappa_star": float(doc["mu_diag"]["activity_decisions"]),
                "n_active_periods": int(active.sum()),
                "a_ir_gross_model": ir_a,
                "b_ir_gross_constant_direction": ir_b,
                "delta_a_minus_b": boot.delta_ir,
                "delta_ci95_lower": boot.ci_lower,
                "se_boot": boot.se_boot,
                "bias_only_share_of_model_ir": share,
            }
        )
        print(
            f"  seed {seed} ({'SHORT' if sigma < 0 else 'LONG'}): "
            f"(a) {ir_a:+.4f}  (b) {ir_b:+.4f}  delta {boot.delta_ir:+.4f}  "
            f"CI95_lower {boot.ci_lower:+.4f}"
        )

    out = {
        "script": "scripts/m6_skill_vs_bias.py",
        "schema": "m6_skill_vs_bias_v1",
        "question": (
            "Is the positive GROSS IR evidence of predictive skill, or is it directional bias "
            "times sample drift? (a) the model as scored vs (b) constant direction at the modal "
            "sign on the same active periods."
        ),
        "★_BENCHMARK_SUBSTITUTION_READ_FIRST": (
            "The specified benchmark averages over A_p, the symbols THE MODEL TRADED. A_p is NOT "
            "recoverable: write_cell_eval_artifact persists only the pooled net series, and the "
            "execution filter DOES bind here (decision-activity 0.0926/0.0193/0.4702 at kappa*, "
            "moving with kappa), so traded != scored and A_p is model-dependent. This script "
            "therefore averages over V_p — every symbol with a VALID DECISION at that period — "
            "which is model-independent and lake-only. (b) is thus 'constant direction on the "
            "evaluated cross-section', NOT 'constant direction on the names this model picked'. "
            "The direction of that difference is NOT known a priori and no bound is claimed."
        ),
        "market_drift": {
            "what": "cross-sectional mean forward log-return per stride-15 period over V_p",
            "mean_per_period": float(market.mean()),
            "mean_per_period_active_only": None,
            "std_per_period": float(market.std(ddof=0)),
            "annualized_ir_of_always_long_market": information_ratio(market, DEMO_H),
            "mean_breadth": float(breadth.mean()),
            "n_periods": t,
        },
        "pins": {
            "h": DEMO_H,
            "headline_cost": HEADLINE_COST,
            "bootstrap_B": BOOT_B,
            "bootstrap_seed": BOOT_SEED,
            "block_length": block_length(t),
            "paired": True,
        },
        "n_symbols": len(symbols),
        "rows": rows,
        "COST_OF_THE_EXACT_VERSION": (
            "Computing (b) over A_p requires re-scoring per-symbol mu-hat across the money grid "
            "(1,402,520 decisions per unit). Local CPU, so $0 in money, but tokenization alone is "
            "~29 s per (symbol, seed) x 40 symbols x 3 seeds ~= 1 hour, plus the rollouts. Not run "
            "without a ruling."
        ),
        "INTERPRETATION": "NONE — the builder reports; the ruling is the supervisor's.",
    }
    act_mask = np.zeros(t, dtype=bool)
    for _s, doc in docs.items():
        act_mask |= np.array(doc["headline_series"], dtype=np.float64) != 0.0
    out["market_drift"]["mean_per_period_active_only"] = float(market[act_mask].mean())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"WROTE {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
