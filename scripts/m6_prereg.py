"""Compute MDE_prereg from the realized lake covariance structure → docs/m6_prereg.md.

    PYTHONPATH=src python3 scripts/m6_prereg.py [--n-symbols 40] [--horizons 5,15,60]

Pre-registration mechanics (m6_design §2 — computed from the DATA's covariance structure and the
design, NEVER from any treatment comparison; no M6 model exists when this runs):

* The primary test statistic is ΔIR_info = IR(Cell 4) − IR(Cell 5) on the POOLED cross-sectional
  portfolio period series over the forward (eval) region.
* SE mechanics: an IR estimated on T_eff independent periods has SE ≈ √(1/T_eff) in per-period
  Sharpe units; two cells scored on the same grid, conservatively treated as independent
  (ρ₄₅ = 0 → the true SE is smaller), give SE(Δ) = √(2/T_eff).
* MDE (one-sided α = 0.05, power = 0.80): (z₀.₉₅ + z₀.₈₀)·√(2/T_eff)·√(periods_per_year(h)).
* T_eff deflations, BOTH measured from the realized data:
  - temporal: T_eff = T·(1−ρ₁)/(1+ρ₁) with ρ₁ = lag-1 autocorrelation of the pooled
    equal-weight period-return series (the moving-block-bootstrap-consistent shrinkage);
  - cross-symbol (context for breadth, reported): N_breadth_eff = N/(1+(N−1)·ρ̄) with ρ̄ = the
    mean pairwise correlation of per-symbol period returns AFTER removing the market factor
    (the equal-weight cross-sectional mean) — crypto's common factor makes this mandatory.
* Per-regime: under the anchored 0.7 train split, the OOS forward region covers late-2023 + 2024
  ONLY; 2021/2022 lie inside the train region, so their per-regime reads are in-sample (Q1/Q2)
  secondary diagnostics — recorded as such, not as OOS tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake
from trikaal.eval.dsr import norm_ppf
from trikaal.eval.metrics import periods_per_year

LAKE = Path("processed/universe_bars")
WINDOW = ("2021-01-01", "2025-01-01")
TRAIN_FRAC = 0.7
BAR_MS = 60_000


def load_period_returns(con, symbol: str, grid: np.ndarray, h: int) -> np.ndarray:
    """One symbol's h-bar forward log return at each global grid instant (NaN where absent)."""
    t = con.execute(
        "SELECT bar_open_ms, raw_ret_close, segment_id FROM bars WHERE symbol = ? "
        "ORDER BY bar_open_ms",
        [symbol],
    ).to_arrow_table()
    ts = t.column("bar_open_ms").to_numpy().astype(np.int64)
    rrc = t.column("raw_ret_close").to_numpy().astype(np.float64)
    seg = t.column("segment_id").to_numpy().astype(np.int64)
    clp = np.cumsum(rrc)
    out = np.full(grid.shape[0], np.nan)
    pos = np.searchsorted(ts, grid)
    ok = (pos + h < ts.shape[0]) & (pos > 0)
    hit = np.zeros_like(ok)
    hit[ok] = ts[pos[ok]] == grid[ok]
    idx = np.nonzero(ok & hit)[0]
    p = pos[idx]
    same_seg = seg[p] == seg[np.minimum(p + h, seg.shape[0] - 1)]
    idx, p = idx[same_seg], p[same_seg]
    out[idx] = clp[np.minimum(p + h, clp.shape[0] - 1)] - clp[p]
    return out


def mde_from_t_eff(t_eff: float, h: int, *, alpha: float = 0.05, power: float = 0.80) -> float:
    z = norm_ppf(1.0 - alpha) + norm_ppf(power)
    return z * np.sqrt(2.0 / max(t_eff, 1.0)) * np.sqrt(periods_per_year(h))


def analyze(returns: np.ndarray, h: int) -> dict:
    """[T, N] per-symbol period returns → ρ̄, ρ₁, T_eff, breadth, MDE."""
    _t_grid, n_sym = returns.shape
    market = np.nanmean(returns, axis=1)
    valid_p = np.isfinite(market)
    resid = returns - market[:, None]
    # mean pairwise residual correlation over symbol pairs with enough overlap
    cors = []
    for i in range(n_sym):
        for j in range(i + 1, n_sym):
            ok = np.isfinite(resid[:, i]) & np.isfinite(resid[:, j])
            if ok.sum() >= 500:
                c = np.corrcoef(resid[ok, i], resid[ok, j])[0, 1]
                if np.isfinite(c):
                    cors.append(c)
    rho_bar_resid = float(np.mean(cors)) if cors else float("nan")
    # raw (pre-residualization) pairwise correlation — the common-factor strength context
    cors_raw = []
    for i in range(n_sym):
        for j in range(i + 1, n_sym):
            ok = np.isfinite(returns[:, i]) & np.isfinite(returns[:, j])
            if ok.sum() >= 500:
                c = np.corrcoef(returns[ok, i], returns[ok, j])[0, 1]
                if np.isfinite(c):
                    cors_raw.append(c)
    rho_bar_raw = float(np.mean(cors_raw)) if cors_raw else float("nan")
    # temporal deflation from the pooled series' lag-1 autocorrelation
    m = market[valid_p]
    ac1 = float(np.corrcoef(m[:-1], m[1:])[0, 1]) if m.size > 100 else 0.0
    ac1 = max(ac1, 0.0)  # negative ac1 would inflate T_eff; stay conservative
    t = int(valid_p.sum())
    t_eff = t * (1.0 - ac1) / (1.0 + ac1)
    avg_breadth = float(np.isfinite(returns).sum(axis=1)[valid_p].mean())
    n_breadth_eff = avg_breadth / (1.0 + (avg_breadth - 1.0) * max(rho_bar_raw, 0.0))
    return {
        "T": t,
        "T_eff": round(t_eff, 1),
        "ac1_pooled": round(ac1, 4),
        "rho_bar_residual": round(rho_bar_resid, 4),
        "rho_bar_raw": round(rho_bar_raw, 4),
        "avg_breadth": round(avg_breadth, 1),
        "N_breadth_eff": round(n_breadth_eff, 1),
        "MDE_annualized_IR": round(mde_from_t_eff(t_eff, h), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-symbols", type=int, default=40)
    ap.add_argument("--horizons", default="5,15,60")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]

    con = connect_lake(LAKE)
    syms = [
        r[0]
        for r in con.execute(
            "SELECT symbol FROM bars GROUP BY symbol ORDER BY count(*) DESC, symbol LIMIT ?",
            [args.n_symbols],
        ).fetchall()
    ]
    boundary = calendar_boundary_ms(*WINDOW, TRAIN_FRAC)
    end_ms = calendar_boundary_ms(*WINDOW, 1.0)
    print(f"[prereg] {len(syms)} symbols; eval region {boundary} → {end_ms}")

    results: dict[str, dict] = {}
    for h in horizons:
        grid = np.arange(boundary, end_ms, h * BAR_MS, dtype=np.int64)
        mat = np.column_stack([load_period_returns(con, s, grid, h) for s in syms])
        results[f"h{h}_pooled"] = analyze(mat, h)
        # per-regime: the OOS forward region only contains late-2023 + 2024
        for year, (lo_s, hi_s) in {
            "2023_oos_tail": ("2023-01-01", "2024-01-01"),
            "2024": ("2024-01-01", "2025-01-01"),
        }.items():
            lo = calendar_boundary_ms(lo_s, hi_s, 0.0)
            hi = calendar_boundary_ms(lo_s, hi_s, 1.0)
            sel = (grid >= lo) & (grid < hi)
            if sel.sum() > 500:
                results[f"h{h}_{year}"] = analyze(mat[sel], h)
        print(f"[prereg] h={h}: {results[f'h{h}_pooled']}")

    out = {"symbols_sampled": syms, "window": WINDOW, "train_frac": TRAIN_FRAC, **results}
    Path("runs_manifest").mkdir(exist_ok=True)
    p = Path("runs_manifest/m6_mde_inputs.json")
    import json

    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[prereg] inputs+results → {p} (fold into docs/m6_prereg.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
