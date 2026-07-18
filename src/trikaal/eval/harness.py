"""End-to-end eval harness — wires every KAT'd primitive into the §8 pipeline (the machine).

``run_harness`` executes the whole apparatus on the M3 checkpoint + BTCUSDT lake: purged
walk-forward folds → KV-cache rollout predictions → execution-filter strategy → cost-aware net-IR
at the 0.30% headline + cost-stress curve + break-even → DSR/PBO over the κ-search configs →
the Cell-5 placebo comparison (real vs shuffled-micro tokens) → secondary diagnostics. It also
exercises the held-out-symbol Q3/Q4 code path on a synthetic 2-symbol fixture.

**The numbers are meaningless** (dev-grade model, one symbol, synthetic Q3/Q4) — M5 proves the
machine runs and is leak-free, not any result. Crash-safe and content-hashed by the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trikaal.data.segments import segment_bounds
from trikaal.eval import dsr as dsr_mod
from trikaal.eval.costs import SPREAD_DECILE_FRAC, CostModel
from trikaal.eval.diagnostics import picp_mpiw, vol_mae_r2
from trikaal.eval.folds import FoldPlan, make_fold_plan, quadrant
from trikaal.eval.ic_screen import pearson, spearman_rankic
from trikaal.eval.metrics import break_even_cost, information_ratio, periods_per_year
from trikaal.eval.placebo import assert_perp_dims_masked, block_time_permute, placebo_verdict
from trikaal.eval.predict import predict_mu, sigma_hat
from trikaal.eval.strategy import net_trade_returns, portfolio_period_returns, positions

KAPPAS = (1.0, 1.5, 2.0, 3.0)
FLAT_COSTS = (0.0010, 0.0020, 0.0030)  # cost-stress curve 0.10/0.20/0.30%
HEADLINE_COST = 0.0030  # §8.B.2 conservative 0.30% upper bound


def forward_log_returns(raw_ret_close: np.ndarray, segment_id: np.ndarray, h: int) -> np.ndarray:
    """y_{t,h} = log(C_{t+h}/C_{t+1}) per bar (entry next-bar, §8.B.3), within-segment; NaN OOB."""
    rrc = np.asarray(raw_ret_close, dtype=np.float64)
    y = np.full(rrc.shape[0], np.nan)
    for s0, s1 in segment_bounds(segment_id):
        clp = np.cumsum(rrc[s0:s1])  # log-price relative to segment start
        last = (s1 - s0) - h
        if last > 1:
            p = np.arange(0, last)
            y[s0 + p] = clp[p + h] - clp[p + 1]  # log(C_{t+h}/C_{t+1})
    return y


def decision_grid(
    fp: FoldPlan, lo: int, hi: int, h: int, *, seq_len: int, cap: int, rng: np.random.Generator
) -> np.ndarray:
    """Stride-h decision bars in [lo,hi) with ≥seq_len history and t+h valid; capped at ``cap``."""
    grid = np.arange(max(lo, seq_len), min(hi, fp.n - h), h, dtype=np.int64)
    if grid.size > cap:
        grid = np.sort(rng.choice(grid, size=cap, replace=False))
    return grid


def _per_bar_cost(cost: CostModel, sigma: np.ndarray, decisions: np.ndarray, spread_frac: float):
    """Vol-scaled round-trip cost per decision (q/ADV at a small constant — ADV is an M4 input).

    σ̄ is a **causal expanding mean** of σ (bars ≤ t) — never the full-series mean, which would peek
    into the future of a past decision. M4 supplies the causal per-symbol σ̄/ADV at scale.
    """
    sig = np.asarray(sigma, dtype=np.float64)
    cummean = np.cumsum(sig) / np.arange(1, sig.shape[0] + 1)  # σ̄_t over bars ≤ t (causal)
    return np.array(
        [
            cost.c_total(
                spread_frac=spread_frac, sigma_t=sig[t], sigma_bar=cummean[t], q_over_adv=1e-3
            )
            for t in decisions
        ]
    )


@dataclass
class BacktestOut:
    ir: float
    n_trades: int
    series: np.ndarray  # FULL stride-h calendar grid, flat periods 0.0 (m6_design §6 item 10b)
    breadth: np.ndarray
    activity: float = float("nan")  # traded fraction of the calendar grid (reported separately)
    period_returns_by_kappa: dict[float, np.ndarray] = field(default_factory=dict)


def single_symbol_backtest(
    mu: np.ndarray,
    y: np.ndarray,
    c_total: np.ndarray,
    h: int,
    kappa: float,
    *,
    c_net: np.ndarray | float | None = None,
) -> BacktestOut:
    """Single-symbol stride-h backtest at filter multiple ``kappa`` → portfolio period-IR.

    Positions come from the execution filter θ = κ·``c_total`` (the modeled cost); the *netting*
    cost is ``c_net`` (default: the same ``c_total``) — so the pre-registered headline can net at
    the flat 0.30 % round-trip while the filter still uses the modeled cost (§6 item 10a).

    The returned ``series`` covers the **full stride-h calendar grid with flat periods as 0.0**
    (§6 item 10b): annualizing an active-only series at full calendar frequency inflates the IR by
    ~1/√(activity) and biases cross-cell ΔIR whenever trade rates differ. Activity is reported
    separately, never folded into the annualization.
    """
    theta = kappa * c_total
    s = positions(mu, theta)
    active = s != 0
    cn = c_total if c_net is None else np.broadcast_to(np.asarray(c_net, np.float64), mu.shape)
    net = net_trade_returns(s[active], y[active], cn[active])
    period_idx = np.arange(mu.shape[0])[active]  # one symbol → each decision is its own period
    ps = portfolio_period_returns(period_idx, net, n_periods=mu.shape[0])
    calendar = np.zeros(mu.shape[0], dtype=np.float64)  # flat periods = 0.0 (held cash)
    if ps.returns.size:
        calendar[np.unique(period_idx)] = ps.returns
    return BacktestOut(
        ir=information_ratio(calendar, h),
        n_trades=int(active.sum()),
        series=calendar,
        breadth=ps.breadth,
        activity=float(active.mean()) if mu.shape[0] else float("nan"),
    )


def cost_stress_curve(
    mu: np.ndarray, y: np.ndarray, c_theta: np.ndarray, h: int, kappa: float
) -> dict[float, float]:
    """IR of the SAME strategy (positions from θ = κ·``c_theta``, the modeled cost) re-netted at
    flat c ∈ {0.10,0.20,0.30 %} (§8.E.2 cost-stress). Netting-only stress keeps the position set
    fixed, so IR(c) is monotone in c and the 0.30 % point IS the headline (§6 item 10a)."""
    out: dict[float, float] = {}
    for c in FLAT_COSTS:
        out[c] = single_symbol_backtest(mu, y, c_theta, h, kappa, c_net=c).ir
    return out


@dataclass
class HarnessResult:
    horizon: int
    kappa_chosen: float
    ir_headline: float  # net-IR NETTED at the flat HEADLINE_COST (0.30 %/round-trip, §6 item 10a)
    ir_modeled_cost: float  # secondary: netted at the modeled per-decision cost (the old headline)
    cost_stress: dict[float, float]
    break_even: float
    dsr: float
    pbo: float
    placebo: dict[str, object]
    diagnostics: dict[str, float]
    quadrants_run: list[str]
    n_decisions: int
    notes: list[str]
    val_ir_by_kappa: dict[float, float] = field(default_factory=dict)  # §6 item 10d: κ* audit trail
    activity: float = float("nan")  # traded fraction of the headline calendar grid


def run_harness(
    lake: dict,
    model,
    tok,
    *,
    h: int = 15,
    seq_len: int = 128,
    cap: int = 400,
    device: str = "cpu",
    seed: int = 0,
) -> HarnessResult:
    """Run the full machine on the BTCUSDT lake; return a structured (meaningless-number) result."""
    rng = np.random.default_rng(seed)
    notes: list[str] = []
    b_c, b_f = lake["b_c"], lake["b_f"]
    ts, seg, sigma = lake["ts"], lake["segment_id"], lake["sigma"]
    rrc = lake["raw_ret_close"]
    n = b_c.shape[0]
    fp = make_fold_plan(n, train_frac=0.7, k=6, seq_len=seq_len)
    y = forward_log_returns(rrc, seg, h)
    cost = CostModel()
    spread = SPREAD_DECILE_FRAC["major"]  # BTC liquidity decile

    # VAL = forward block 0 (κ tuning); HEADLINE = forward block 3 (a later block; Q4-real needs M4)
    val_lo, val_hi = fp.blocks[0]
    hd_lo, hd_hi = fp.blocks[3]
    val_dec = decision_grid(fp, val_lo, val_hi, h, seq_len=seq_len, cap=cap, rng=rng)
    hd_dec = decision_grid(fp, hd_lo, hd_hi, h, seq_len=seq_len, cap=cap, rng=rng)

    def mu_y_c(dec):
        mu = predict_mu(model, tok, b_c, b_f, ts, sigma, dec, h=h, seq_len=seq_len, device=device)
        return mu, y[dec], _per_bar_cost(cost, sigma, dec, spread)

    mu_val, y_val, c_val = mu_y_c(val_dec)
    mu_hd, y_hd, c_hd = mu_y_c(hd_dec)

    # κ search on VAL (never the headline block); collect per-κ period series for PBO
    per_kappa_val: dict[float, np.ndarray] = {}
    ir_val: dict[float, float] = {}
    for k in KAPPAS:
        bt = single_symbol_backtest(mu_val, y_val, c_val, h, k)
        per_kappa_val[k] = bt.series
        ir_val[k] = bt.ir
    kappa_star = max(KAPPAS, key=lambda k: ir_val[k] if np.isfinite(ir_val[k]) else -1e9)
    notes.append(f"κ chosen on VAL block (never headline): κ*={kappa_star}")

    # HEADLINE backtest: positions from θ = κ*·c_modeled, NETTED at the flat pre-registered
    # HEADLINE_COST (§6 item 10a). The modeled-cost IR is kept as a named secondary.
    head = single_symbol_backtest(mu_hd, y_hd, c_hd, h, kappa_star, c_net=HEADLINE_COST)
    head_modeled = single_symbol_backtest(mu_hd, y_hd, c_hd, h, kappa_star)
    stress = cost_stress_curve(mu_hd, y_hd, c_hd, h, kappa_star)
    c_break = break_even_cost(np.array(FLAT_COSTS), np.array([stress[c] for c in FLAT_COSTS]))

    # DSR over the multiple-testing trial budget; PBO via CSCV on the κ-config matrix (VAL).
    # n_trials=240 here is the HISTORICAL M5 budget, kept numerically because run_harness is the
    # M5 machine-validation instrument frozen under the Gate-A anchor (5eead7b6…) — it is NOT the
    # M6 decision path and its DSR is never quoted as the M6 outcome. The governing recipe is
    # prereg §3 clause 5: N=180 ENUMERATED trials (5 cells × 3 seeds × horizons {5,15,60} × 4 κ;
    # h=1 is not evaluated in M6), computed ONLY by scripts/m6_verdict.py under the
    # conformance.PINNED_DSR pin.
    n_trials = 5 * 3 * 4 * len(KAPPAS)
    # cross-trial variance of the per-period (de-annualized) Sharpe across the κ configs
    ppy = periods_per_year(h)
    var_sr = float(np.var([ir_val[k] / np.sqrt(ppy) for k in KAPPAS])) + 1e-12
    dsr = (
        dsr_mod.deflated_sharpe_ratio(head.series, n_trials=n_trials, var_sr=var_sr)
        if head.series.size >= 2
        else float("nan")
    )
    # PBO input is the TIME-ALIGNED [T_grid, |κ|] matrix (§6 item 10c): every κ's series covers the
    # same full VAL calendar grid with per-κ flat periods as 0 — never active-only series truncated
    # to a common min-length (rows would compare different time periods; statistically invalid).
    pbo = float("nan")
    t_grid = mu_val.shape[0]
    if t_grid >= 16:
        m = dsr_mod.time_aligned_pbo_matrix(per_kappa_val)
        pbo = dsr_mod.pbo_cscv(m, n_splits=8)
    else:
        notes.append("PBO skipped: VAL calendar grid too short (dev-grade subsample)")

    # Placebo: re-tokenize the shuffled-micro surrogate and re-backtest (same model, off-dist)
    placebo_info = _placebo_codepath(
        lake,
        model,
        tok,
        h=h,
        seq_len=seq_len,
        kappa=kappa_star,
        dec=hd_dec,
        y_hd=y_hd,
        c_hd=c_hd,
        ir_real=head.ir,
        device=device,
    )

    # Secondary diagnostics on the dense headline decisions
    diags = _diagnostics(mu_hd, y_hd, sigma[hd_dec], h)

    # Q1 (in-sample sanity) for the quadrant table; Q3/Q4 code path on the synthetic fixture
    q_run = ["Q1", "Q2"]
    _synthetic_q3q4_codepath(notes)
    q_run += ["Q3", "Q4"]

    return HarnessResult(
        horizon=h,
        kappa_chosen=kappa_star,
        ir_headline=head.ir,
        ir_modeled_cost=head_modeled.ir,
        cost_stress=stress,
        break_even=c_break,
        dsr=dsr,
        pbo=pbo,
        placebo=placebo_info,
        diagnostics=diags,
        quadrants_run=q_run,
        n_decisions=int(hd_dec.size + val_dec.size),
        notes=notes,
        val_ir_by_kappa={k: float(ir_val[k]) for k in KAPPAS},
        activity=head.activity,
    )


def _diagnostics(mu: np.ndarray, y: np.ndarray, sig_t: np.ndarray, h: int) -> dict[str, float]:
    ok = np.isfinite(mu) & np.isfinite(y)
    ric = spearman_rankic(mu[ok], y[ok]) if ok.sum() > 2 else float("nan")
    ic = pearson(mu[ok], y[ok]) if ok.sum() > 2 else float("nan")
    sh = sigma_hat(sig_t, h)
    realized_vol = np.abs(
        y
    )  # crude realized |move| proxy over the window (M6 uses the §7 estimator)
    vol = vol_mae_r2(sh[ok], realized_vol[ok]) if ok.sum() > 2 else {"mae": float("nan")}
    low, high = mu - 1.2816 * sh, mu + 1.2816 * sh  # 80% Gaussian interval
    picp, mpiw = (
        picp_mpiw(y[ok], low[ok], high[ok]) if ok.sum() > 2 else (float("nan"), float("nan"))
    )
    return {"rank_ic": ric, "ic": ic, "vol_mae": vol["mae"], "picp80": picp, "mpiw80": mpiw}


def _placebo_codepath(
    lake, model, tok, *, h, seq_len, kappa, dec, y_hd, c_hd, ir_real, device
) -> dict[str, object]:
    """Exercise §8.C.4: build the shuffled-micro surrogate, re-tokenize, re-backtest, compare."""
    from trikaal.train.token_stream import tokenize_features

    # Tripwire (§6 item 10f): MICRO_DIMS (7-12) deliberately excludes funding/OI (13-15) because
    # this build masks them everywhere; if they ever ACTIVATE, the shuffle would silently
    # under-shuffle (real funding/OI info left aligned in a "shuffled-micro" cell) — fail loudly.
    assert_perp_dims_masked(lake["mask"])
    x_surr, m_surr = block_time_permute(lake["x"], lake["mask"], lake["segment_id"], seed=0)
    sc, sf = tokenize_features(
        tok, x_surr, m_surr, lake["segment_id"], window=seq_len, device=device
    )
    mu_pl = predict_mu(
        model, tok, sc, sf, lake["ts"], lake["sigma"], dec, h=h, seq_len=seq_len, device=device
    )
    ir_placebo = single_symbol_backtest(mu_pl, y_hd, c_hd, h, kappa).ir
    # proxy cell IRs: Cell-2 (OHLCV-only) is a no-micro baseline, stood at 0 for the code-path
    # exercise. Real Cell-2/4/5 IRs come from M6's trained models; here we only prove the machinery.
    v = placebo_verdict(ir_cell2=0.0, ir_cell4=ir_real, ir_cell5=ir_placebo)
    v["ir_real_tokens"] = ir_real
    v["ir_placebo_tokens"] = ir_placebo
    return v


def _synthetic_q3q4_codepath(notes: list[str]) -> None:
    """Exercise the held-out-symbol Q3/Q4 + cross-symbol aggregation on a 2-symbol synth fixture."""
    rng = np.random.default_rng(0)
    # two synthetic symbols, period-aligned; trades on both → breadth-2 portfolio periods
    n_periods = 50
    sym_a = rng.standard_normal(n_periods) * 0.01
    sym_b = rng.standard_normal(n_periods) * 0.01
    period_idx = np.concatenate([np.arange(n_periods), np.arange(n_periods)])
    net = np.concatenate([sym_a, sym_b])
    ps = portfolio_period_returns(period_idx, net, n_periods=n_periods)
    assert ps.breadth.max() == 2, "Q3/Q4 cross-symbol aggregation should reach breadth 2"
    assert quadrant(False, True) == "Q3" and quadrant(False, False) == "Q4"
    notes.append(
        "Q3/Q4 + cross-symbol breadth exercised on a synthetic 2-symbol fixture "
        "(real Q3/Q4 numbers require the M4 universe)"
    )


__all__ = [
    "FLAT_COSTS",
    "HEADLINE_COST",
    "KAPPAS",
    "BacktestOut",
    "HarnessResult",
    "cost_stress_curve",
    "decision_grid",
    "forward_log_returns",
    "run_harness",
    "single_symbol_backtest",
]
