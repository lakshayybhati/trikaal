"""Cross-sectional eval driver — the 5-model × N-symbol ablation scorer (m6_design §6 item 6).

Extends the CORRECTED single-symbol conventions (§6 item 10) to the cross-section:

* **global stride-h rebalance periods**: one calendar grid over the forward region, shared by
  every symbol — period ``p`` is the same instant everywhere, so pooling is time-aligned by
  construction;
* **equal-weight pooling across active symbols** per period (``portfolio_period_returns``), then
  the FULL calendar grid with flat periods as 0.0 before annualization;
* **headline netted at the flat 0.30 %** round-trip (positions from θ = κ*·c_modeled);
* **κ\\* chosen on the VAL block** (forward block 0) over the pooled cross-section, per-κ curve
  persisted; **time-aligned PBO** over the κ configs;
* **Cell-5 at eval uses the SAME ``arms.shuffle_micro``** the training loader used (per-(seed,
  symbol) derivation) — train/eval identity is structural;
* the **5-model + placebo verdict**: ΔIR_info = IR(Cell 4) − IR(Cell 5) plus the 2×2 marginals.

G-causal (m6_design §3): eval decisions live STRICTLY inside the forward blocks — a decision
before the train/eval boundary is structurally impossible (KAT'd), mirroring the loader's
train-side guarantee from the other direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trikaal.data.universe_loader import calendar_boundary_ms, eval_block_bounds_ms
from trikaal.eval.costs import SPREAD_DECILE_FRAC, CostModel
from trikaal.eval.dsr import pbo_cscv, time_aligned_pbo_matrix
from trikaal.eval.harness import HEADLINE_COST, KAPPAS, _per_bar_cost, forward_log_returns
from trikaal.eval.metrics import break_even_cost, information_ratio
from trikaal.eval.placebo import placebo_verdict
from trikaal.eval.predict import predict_mu
from trikaal.eval.strategy import net_trade_returns, portfolio_period_returns, positions
from trikaal.train.arms import ARM_MICRO_SHUFFLED, select_arm, shuffle_micro
from trikaal.train.token_stream import tokenize_features

BAR_MS = 60_000


@dataclass
class XSectionConfig:
    window_start: str = "2021-01-01"
    window_end: str = "2025-01-01"
    train_frac: float = 0.7
    h: int = 15
    seq_len: int = 128
    # decisions per (symbol, grid) — a DEV bound (like M5's cap). None = no cap; money mode
    # REQUIRES None (§3a pins the full primary region, uncapped — conformance-gated).
    cap_per_symbol: int | None = 200
    kappas: tuple[float, ...] = KAPPAS
    headline_cost: float = HEADLINE_COST
    val_block: int = 0
    headline_block: int = 3
    n_blocks: int = 6
    device: str = "cpu"
    seed: int = 0
    # per-symbol spread deciles from TRAIN-REGION liquidity (audit item 4). None = the flat
    # "major" dev fallback — the money run REQUIRES the committed decile map (conformance-gated).
    spread_frac_by_symbol: dict[str, float] | None = None
    # MONEY MODE (§3a): the headline is ONE continuous stride grid over forward blocks 1..5
    # (VAL block 0 excluded) — the exact MDE-recompute basis — with no dev cap and per-symbol
    # deciles mandatory. Dev/smoke mode (money=False) keeps the single-block headline.
    money: bool = False

    def __post_init__(self) -> None:
        if self.money:
            if self.cap_per_symbol is not None:
                raise ValueError("money mode forbids cap_per_symbol (§3a: uncapped primary)")
            if self.spread_frac_by_symbol is None:
                raise ValueError(
                    "money mode requires per-symbol spread deciles "
                    "(runs_manifest/m6_spread_deciles.json via costs.load_spread_deciles)"
                )


@dataclass
class SymbolEval:
    """One symbol's raw eval inputs (FULL 16-dim layout — arms are applied per cell)."""

    symbol: str
    x: np.ndarray  # [T,16]
    mask: np.ndarray  # [T,16]
    segment_id: np.ndarray
    ts: np.ndarray
    sigma: np.ndarray
    raw_ret_close: np.ndarray


@dataclass
class CellScore:
    run: str
    kappa_chosen: float
    ir_headline: float  # pooled cross-section, netted at the flat 0.30 %
    ir_modeled_cost: float
    val_ir_by_kappa: dict[float, float]
    cost_stress: dict[float, float]
    break_even: float
    pbo: float
    activity: float
    n_decisions: int
    per_symbol_decisions: dict[str, int] = field(default_factory=dict)


def global_decision_grid_ms(cfg: XSectionConfig, block: int) -> np.ndarray:
    """The GLOBAL stride-h rebalance instants (calendar ms) inside forward block ``block``."""
    lo, hi = eval_block_bounds_ms(
        cfg.window_start, cfg.window_end, train_frac=cfg.train_frac, k=cfg.n_blocks
    )[block]
    return np.arange(lo, hi, cfg.h * BAR_MS, dtype=np.int64)


def primary_region_grid_ms(cfg: XSectionConfig) -> np.ndarray:
    """The §3a PRIMARY-REGION grid: ONE continuous stride-h grid over forward blocks 1..k−1
    (VAL block 0 excluded) — the exact basis `scripts/m6_prereg.py` computed the MDE on.
    A per-block concatenation would re-anchor the stride phase at every block edge (the edges
    are not stride multiples); the money headline uses THIS grid, never a concatenation."""
    bounds = eval_block_bounds_ms(
        cfg.window_start, cfg.window_end, train_frac=cfg.train_frac, k=cfg.n_blocks
    )
    lo, hi = bounds[1][0], bounds[cfg.n_blocks - 1][1]
    return np.arange(lo, hi, cfg.h * BAR_MS, dtype=np.int64)


def symbol_decisions(se: SymbolEval, grid_ms: np.ndarray, cfg: XSectionConfig) -> np.ndarray:
    """Map the global grid onto one symbol's bar indices; keep only VALID decisions.

    Valid: the bar exists at that instant, has ≥seq_len history, an in-range label bar, and its
    calendar instant is ≥ the train/eval boundary (G-causal: eval never reaches into train —
    structurally true since the grid starts at the boundary, asserted anyway)."""
    boundary = calendar_boundary_ms(cfg.window_start, cfg.window_end, cfg.train_frac)
    assert grid_ms.size == 0 or grid_ms.min() >= boundary  # eval lives in the forward region
    pos = np.searchsorted(se.ts, grid_ms)
    ok = (pos < se.ts.shape[0]) & (pos >= cfg.seq_len)
    ok &= pos + cfg.h < se.ts.shape[0]
    hit = np.zeros_like(ok)
    hit[ok] = se.ts[pos[ok]] == grid_ms[ok]  # the bar genuinely exists at the instant
    dec = pos[ok & hit]
    return dec.astype(np.int64)


def _pool_periods(
    per_symbol_nets: dict[str, tuple[np.ndarray, np.ndarray]], n_periods: int
) -> np.ndarray:
    """Equal-weight pool per global period → FULL calendar series with flat periods as 0.0."""
    all_pidx = (
        np.concatenate([p for p, _ in per_symbol_nets.values()])
        if per_symbol_nets
        else np.array([], dtype=np.int64)
    )
    all_nets = (
        np.concatenate([r for _, r in per_symbol_nets.values()])
        if per_symbol_nets
        else np.array([])
    )
    ps = portfolio_period_returns(all_pidx, all_nets, n_periods=n_periods)
    calendar = np.zeros(n_periods, dtype=np.float64)
    if ps.returns.size:
        calendar[np.unique(all_pidx)] = ps.returns
    return calendar


def score_cell(
    run: str,
    model,
    tok,
    arm: str,
    symbols: list[SymbolEval],
    cfg: XSectionConfig,
) -> CellScore:
    """Score ONE trained (cell, seed) model cross-sectionally: VAL κ-search → pooled headline."""
    cost = CostModel()
    rng = np.random.default_rng(cfg.seed)

    def spread_for(sym: str) -> float:
        """The symbol's train-region liquidity decile (audit item 4); flat dev fallback if unset."""
        if cfg.spread_frac_by_symbol is None:
            return SPREAD_DECILE_FRAC["major"]
        if sym not in cfg.spread_frac_by_symbol:
            raise KeyError(
                f"no spread decile for {sym} — regenerate runs_manifest/m6_spread_deciles.json "
                "(scripts/m6_spread_deciles.py) to cover every evaluated symbol"
            )
        return float(cfg.spread_frac_by_symbol[sym])

    # per-symbol arm transform (Cell-5: the SAME shuffle as training) + tokenize + labels
    prepared: dict[str, dict] = {}
    for se in symbols:
        x, mask = se.x, se.mask
        if arm == ARM_MICRO_SHUFFLED:
            x, mask = shuffle_micro(x, se.mask, se.segment_id, symbol=se.symbol, seed=cfg.seed)
        x_arm, m_arm = select_arm(np.asarray(x, np.float32), mask, arm)
        b_c, b_f = tokenize_features(
            tok, x_arm, m_arm, se.segment_id, window=cfg.seq_len, device=cfg.device
        )
        y = forward_log_returns(se.raw_ret_close, se.segment_id, cfg.h)
        prepared[se.symbol] = {"b_c": b_c, "b_f": b_f, "y": y, "se": se}

    def grid_series(
        grid: np.ndarray,
    ) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[str, int], int]:
        """Per-κ pooled FULL-grid calendar series for one decision grid, netted BOTH ways:
        at the flat pre-registered headline cost AND at the modeled per-decision cost (the
        named secondary, §6 item 10a). Positions come from θ = κ·c_modeled in both."""
        n_periods = grid.shape[0]
        nets_flat: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            k: {} for k in cfg.kappas
        }
        nets_mod: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            k: {} for k in cfg.kappas
        }
        n_dec: dict[str, int] = {}
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
            y_d = d["y"][dec]
            c_mod = _per_bar_cost(cost, se.sigma, dec, spread_for(sym))
            pidx = np.searchsorted(grid, se.ts[dec])  # the decision's global period id
            for k in cfg.kappas:
                s = positions(mu, k * c_mod)
                act = s != 0
                nets_flat[k][sym] = (
                    pidx[act],
                    net_trade_returns(s[act], y_d[act], np.full(int(act.sum()), cfg.headline_cost)),
                )
                nets_mod[k][sym] = (pidx[act], net_trade_returns(s[act], y_d[act], c_mod[act]))
        series = {k: _pool_periods(nets_flat[k], n_periods) for k in cfg.kappas}
        series_mod = {k: _pool_periods(nets_mod[k], n_periods) for k in cfg.kappas}
        return series, series_mod, n_dec, n_periods

    # κ* on the pooled VAL block; per-κ curve persisted; time-aligned PBO over the κ configs
    val_series, _val_mod, _, _ = grid_series(global_decision_grid_ms(cfg, cfg.val_block))
    val_ir = {k: information_ratio(val_series[k], cfg.h) for k in cfg.kappas}
    finite_ir = {k: v for k, v in val_ir.items() if np.isfinite(v)}
    kappa_star = max(finite_ir, key=finite_ir.get) if finite_ir else cfg.kappas[0]
    pbo = float("nan")
    if val_series[kappa_star].shape[0] >= 16:
        pbo = pbo_cscv(time_aligned_pbo_matrix(val_series), n_splits=8)

    # HEADLINE at κ*: pooled, netted at the flat 0.30 %; modeled-cost secondary + stress.
    # Money mode (§3a): ONE continuous grid over blocks 1..5 — the MDE-recompute basis;
    # dev/smoke mode: the single headline block (the historical M5-style read).
    hd_grid = (
        primary_region_grid_ms(cfg)
        if cfg.money
        else global_decision_grid_ms(cfg, cfg.headline_block)
    )
    hd_series, hd_mod, n_dec, _ = grid_series(hd_grid)
    head = hd_series[kappa_star]
    ir_headline = information_ratio(head, cfg.h)
    ir_modeled = information_ratio(hd_mod[kappa_star], cfg.h)
    activity = float((head != 0.0).mean()) if head.size else float("nan")

    def _renet(flat_c: float) -> float:
        # same positions (θ from modeled cost inside block_series), re-netted at flat_c: shift
        # every traded period's pooled net by the cost delta (uniform unit netting per trade)
        delta = flat_c - cfg.headline_cost
        adj = np.where(head != 0.0, head - delta, 0.0)
        return information_ratio(adj, cfg.h)

    stress = {c: _renet(c) for c in (0.0010, 0.0020, 0.0030)}
    c_break = break_even_cost(
        np.array(sorted(stress)), np.array([stress[c] for c in sorted(stress)])
    )
    return CellScore(
        run=run,
        kappa_chosen=float(kappa_star),
        ir_headline=float(ir_headline),
        ir_modeled_cost=float(ir_modeled),
        val_ir_by_kappa={float(k): float(v) for k, v in val_ir.items()},
        cost_stress=stress,
        break_even=float(c_break),
        pbo=float(pbo),
        activity=activity,
        n_decisions=int(sum(n_dec.values())),
        per_symbol_decisions=n_dec,
    )


def ablation_verdict(cell_ir: dict[int, float]) -> dict:
    """The pre-registered comparisons from the 5 pooled cell IRs (means across seeds).

    Primary: ΔIR_info = IR(4) − IR(5). Secondary (reported, non-gating): the 2×2 marginals.
    The CI/MDE gating happens in the M6 analysis (m6_prereg.md) — this emits the point verdict."""
    v = placebo_verdict(
        ir_cell2=cell_ir.get(2, float("nan")),
        ir_cell4=cell_ir.get(4, float("nan")),
        ir_cell5=cell_ir.get(5, float("nan")),
    )
    v["delta_info"] = cell_ir.get(4, float("nan")) - cell_ir.get(5, float("nan"))
    v["fsq_effect_ohlcv"] = cell_ir.get(2, float("nan")) - cell_ir.get(1, float("nan"))
    v["fsq_effect_micro"] = cell_ir.get(4, float("nan")) - cell_ir.get(3, float("nan"))
    v["micro_marginal_fsq"] = cell_ir.get(4, float("nan")) - cell_ir.get(2, float("nan"))
    v["micro_marginal_bsq"] = cell_ir.get(3, float("nan")) - cell_ir.get(1, float("nan"))
    v["cell_ir"] = dict(cell_ir)
    return v


__all__ = [
    "CellScore",
    "SymbolEval",
    "XSectionConfig",
    "ablation_verdict",
    "global_decision_grid_ms",
    "primary_region_grid_ms",
    "score_cell",
    "symbol_decisions",
]
