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

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from trikaal.data.universe_loader import calendar_boundary_ms, eval_block_bounds_ms
from trikaal.eval.costs import SPREAD_DECILE_FRAC, CostModel
from trikaal.eval.diagnostics import (
    cell_codebook_diagnostic,
    ohlcv_recon_diagnostic,
    single_bar_decode_diagnostic,
)
from trikaal.eval.dsr import pbo_cscv, time_aligned_pbo_matrix
from trikaal.eval.harness import HEADLINE_COST, KAPPAS, _per_bar_cost, forward_log_returns
from trikaal.eval.metrics import break_even_cost, information_ratio
from trikaal.eval.placebo import placebo_verdict
from trikaal.eval.predict import predict_mu
from trikaal.eval.strategy import net_trade_returns, portfolio_period_returns, positions
from trikaal.train.arms import ARM_MICRO_SHUFFLED, select_arm, shuffle_micro
from trikaal.train.token_stream import tokenize_features

BAR_MS = 60_000


# §7 v1.4.2 / v1.6.15 C-18: the pinned decode. conformance.PINNED_MU_ESTIMATOR must agree.
MU_ESTIMATOR = "expectation"


@dataclass
class XSectionConfig:
    window_start: str = "2021-01-01"
    window_end: str = "2025-01-01"
    train_frac: float = 0.7
    h: int = 15
    # ★ 512, NOT 128. The old default was the Stage-1 TOKENIZER training length on a
    # Stage-2/eval config: every real call site passes 512, the shipped tokenizer's own
    # `max_len` IS 512, and `conformance.PINNED_MONEY_SEQ_LEN` is 512 — so a caller who
    # accepted the default silently evaluated at a QUARTER of the shipped context with the
    # signature and the docstring agreeing with each other and disagreeing with the artifact.
    # The literal is spelled out rather than imported because `conformance` imports THIS
    # module; `test_seq_len_defaults.py` binds the two so they cannot drift apart.
    seq_len: int = 512
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
    # §7 v1.6.15 C-18: the mu estimator was the v1.4.2 PIN existing only as a predict_mu function
    # default, so nothing on the money path asserted it. It is now a config field the conformance
    # gate reads, cross-checked against conformance.PINNED_MU_ESTIMATOR (two copies, one truth —
    # the shape used for DSR_VAR_SR_BASIS_CELL). Declared here rather than imported to keep the
    # dependency running one way: conformance imports xsection, never the reverse.
    mu_estimator: str = MU_ESTIMATOR

    def __post_init__(self) -> None:
        if self.money:
            if self.mu_estimator != MU_ESTIMATOR:
                raise ValueError(
                    f"money mode pins mu_estimator={MU_ESTIMATOR!r} (§7 v1.4.2 "
                    f"expectation-decode); got {self.mu_estimator!r} — 'argmax' is the biased "
                    "mode-decode v1.4.2 removed"
                )
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
    # non-gating referee-preempt diagnostic (CO2 item 5): per-cell codebook utilization +
    # empirical code entropy (effective bits) for coarse and fine, over the evaluated streams
    codebook: dict = field(default_factory=dict)
    # §7 v1.6 C-12 M1 (supervisor-adopted): OHLCV-restricted reconstruction, the capacity
    # disclosure that bounds the placebo's handicap. NON-GATING — reported beside the headline,
    # never a clause.
    ohlcv_recon: dict = field(default_factory=dict)
    # §7 v1.6.11 C-1 (supervisor-adopted): single-bar vs in-window decode agreement, the SECOND
    # handicap channel. NON-GATING; both channels inflate dIR(4-5) in the same direction if real.
    decode_agreement: dict = field(default_factory=dict)
    # the pooled FULL-calendar headline series itself (flat periods = 0.0) — what the verdict
    # artifacts persist (verdict.write_cell_eval_artifact) and the paired bootstrap resamples.
    # None under val_only (no headline grid was scored).
    headline_series: np.ndarray | None = None
    # §7 v1.4.2 standing μ̂ receipt (acceptance adjudication): the decision-signal's mean/std
    # and negative-fraction pooled across symbols. A constant-sign μ̂ (frac_negative ≈ 0 or 1)
    # is the mode-estimator sign-saturation pathology the acceptance run surfaced — with the
    # mean estimator this should be balanced; recorded per cell so any residual drift is visible
    # in the artifact rather than silently collapsing the κ-threshold decisions.
    mu_diag: dict = field(default_factory=dict)


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
    *,
    val_only: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> CellScore:
    """Score ONE trained (cell, seed) model cross-sectionally: VAL κ-search → pooled headline.

    ``val_only=True`` stops after the VAL block (the per-κ curve + κ*): the verdict artifacts
    need the h∈{5,60} secondary horizons ONLY as VAL trial entries (prereg §3 clause 5), and
    the headline grid at those horizons would be wasted rollout compute. Headline fields come
    back NaN/None and must never be read from a val_only score.

    ``progress`` (§7 finding 12) is an OPTIONAL OBSERVER called once per scored symbol with a
    plain dict. It is ``None`` by default, computes nothing this function returns, and CANNOT
    RAISE — see the call site. The measured cost of not having it: a 14.65-hour eval leg that
    emitted NOTHING, so a healthy run and a hung one were indistinguishable for a whole day. The
    driver supplies the callback, which keeps ``trikaal.eval`` free of W&B exactly as it is
    today."""
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
    # §7 v1.6.25 (RE-AUDIT R12e): the FIRST symbol's ARM-TRANSFORMED tensors, captured here rather
    # than rebuilt from `se.x` below. See the diagnostics block for why that distinction decided
    # two required disclosures.
    first_arm: tuple[np.ndarray, np.ndarray] | None = None
    for se in symbols:
        x, mask = se.x, se.mask
        if arm == ARM_MICRO_SHUFFLED:
            x, mask = shuffle_micro(x, se.mask, se.segment_id, symbol=se.symbol, seed=cfg.seed)
        x_arm, m_arm = select_arm(np.asarray(x, np.float32), mask, arm)
        if first_arm is None:
            first_arm = (x_arm, m_arm)
        b_c, b_f = tokenize_features(
            tok, x_arm, m_arm, se.segment_id, window=cfg.seq_len, device=cfg.device
        )
        y = forward_log_returns(se.raw_ret_close, se.segment_id, cfg.h)
        prepared[se.symbol] = {"b_c": b_c, "b_f": b_f, "y": y, "se": se}

    # non-gating codebook diagnostic over the cell's FULL evaluated token streams (all symbols):
    # utilization + empirical code entropy in bits, coarse and fine (CO2 item 5, referee-preempt)
    codebook = cell_codebook_diagnostic(
        np.concatenate([d["b_c"] for d in prepared.values()]),
        np.concatenate([d["b_f"] for d in prepared.values()]),
        v_c=tok.v_c,
        v_f=tok.v_f,
    )

    # §7 v1.6 C-12 M1 + v1.6.11 C-1: computed on the FIRST symbol's arm-selected window — the same
    # tensors the tokenizer was fed.
    #
    # §7 v1.6.25 (RE-AUDIT R12e) — IT WAS NOT THE SAME TENSORS, AND ONLY FOR CELL 5. The previous
    # form re-derived the window with `select_arm(_first.se.x, ...)`, i.e. from the RAW feature
    # matrix, while the Cell-5 shuffle above wrote to LOCAL `x, mask` that were never stored back.
    # So on the placebo arm — and ONLY there — both required disclosures described a tensor the
    # cell was never fed: the tokenizer was trained on shuffled micro and then measured on
    # UNSHUFFLED micro, which is out-of-distribution input, not "the capacity handicap". The C-12
    # disclosure's entire argument is about what Cell 5 spends bits on, so measuring it on Cell 4's
    # input made the number uninterpretable in exactly the arm it exists to characterise.
    # Every other arm was unaffected (`select_arm` is a pure column selection there), which is why
    # nothing looked wrong. Now the arm-transformed tensors travel from the loop that built them.
    if first_arm is None:  # an empty symbol set cannot be scored (and would break codebook above)
        raise ValueError("score_cell received no symbols — there is nothing to score")
    _x0, _m0 = first_arm
    ohlcv_recon = ohlcv_recon_diagnostic(
        tok, _x0[: cfg.seq_len], _m0[: cfg.seq_len], device=cfg.device
    )
    decode_agreement = single_bar_decode_diagnostic(
        tok, _x0[: cfg.seq_len], _m0[: cfg.seq_len], device=cfg.device
    )

    def grid_series(
        grid: np.ndarray,
        *,
        phase: str,
    ) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[str, int], int, dict]:
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
        mu_pool: list[np.ndarray] = []
        # §7 v1.4.4 DEGENERACY-GUARD input: per-κ TRADED / SCORED decisions (the FILTER-BINDING
        # ratio). Decision-activity ∈ {0, 1} means θ = κ·c never binds — the IR is a pure function
        # of sign(μ̂), a constant-direction book (not a strategy). Distinct from `activity` below,
        # which is traded/full-calendar-grid (diluted by the eval cap; never reaches 1.0).
        traded_by_k: dict[float, int] = {k: 0 for k in cfg.kappas}
        n_scored = 0
        _t_pass, _n_syms = time.perf_counter(), len(prepared)
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
                estimator=cfg.mu_estimator,
            )
            mu_pool.append(mu)
            if progress is not None:
                # ★ AN OBSERVER MUST NEVER BE ABLE TO KILL THE MEASUREMENT IT WATCHES.
                # SWALLOWING IS CORRECT HERE AND WRONG IN A GATE, and the distinction is the whole
                # point. A GATE that hides its own failure is this project's oldest defect ("a
                # check that cannot fail is not a check" — the pipefail rider, the control-arm
                # rule, the fixture-discrimination rule). An OBSERVER that throws destroys a
                # 14.65-hour eval at hour 12 to protect a progress line. DO NOT "fix" this into a
                # bare call: that converts a readability feature into a matrix-killer. Nothing
                # inside this block feeds any value this function returns.
                try:
                    _el = time.perf_counter() - _t_pass
                    _done = len(n_dec)
                    _nan = float("nan")
                    progress(
                        {
                            "run": run,
                            # ★ THE LABEL MUST DESCRIBE THE GRID BEING SCORED, NOT THE
                            # ENCLOSING CALL. It read `val_only` — a parameter of score_cell —
                            # so on the HEADLINE pass (val_only=False) the VAL kappa-search
                            # emitted "grid" while actually scoring the val block, and the
                            # money driver printed that straight to the operator console.
                            "phase": phase,
                            "h": cfg.h,
                            "symbol": sym,
                            "symbols_done": _done,
                            "symbols_total": _n_syms,
                            "decisions_done": n_scored,
                            "elapsed_s": _el,
                            "ms_per_decision": (_el / n_scored * 1e3) if n_scored else _nan,
                            # ETA FOR THIS PASS ONLY. A unit is four passes of UNEQUAL size (the
                            # headline grid spans forward blocks 1..k-1; VAL is block 0, measured
                            # at exactly 0.2000x), so this must never be read as a unit ETA.
                            "eta_pass_s": (_el / _done * (_n_syms - _done)) if _done else _nan,
                        }
                    )
                except Exception:
                    pass
            y_d = d["y"][dec]
            c_mod = _per_bar_cost(cost, se.sigma, dec, spread_for(sym))
            pidx = np.searchsorted(grid, se.ts[dec])  # the decision's global period id
            for k in cfg.kappas:
                s = positions(mu, k * c_mod)
                act = s != 0
                traded_by_k[k] += int(act.sum())
                nets_flat[k][sym] = (
                    pidx[act],
                    net_trade_returns(s[act], y_d[act], np.full(int(act.sum()), cfg.headline_cost)),
                )
                nets_mod[k][sym] = (pidx[act], net_trade_returns(s[act], y_d[act], c_mod[act]))
        series = {k: _pool_periods(nets_flat[k], n_periods) for k in cfg.kappas}
        series_mod = {k: _pool_periods(nets_mod[k], n_periods) for k in cfg.kappas}
        mu_all = np.concatenate(mu_pool) if mu_pool else np.array([])
        mu_diag = (
            {
                "mean": float(mu_all.mean()),
                "std": float(mu_all.std()),
                "frac_negative": float((mu_all < 0).mean()),
                "n": int(mu_all.size),
                "estimator": "expectation",  # §7 v1.4.2 default (conditional mean)
                # §7 v1.4.4: decision-activity (traded/scored) per κ — the guard reads κ*'s value
                "activity_decisions_by_kappa": {
                    f"{k:g}": (traded_by_k[k] / n_scored) for k in cfg.kappas
                },
            }
            if mu_all.size
            else {}
        )
        return series, series_mod, n_dec, n_periods, mu_diag

    # κ* on the pooled VAL block; per-κ curve persisted; time-aligned PBO over the κ configs
    val_series, _val_mod, _, _, _ = grid_series(
        global_decision_grid_ms(cfg, cfg.val_block), phase="val"
    )
    val_ir = {k: information_ratio(val_series[k], cfg.h) for k in cfg.kappas}
    finite_ir = {k: v for k, v in val_ir.items() if np.isfinite(v)}
    kappa_star = max(finite_ir, key=finite_ir.get) if finite_ir else cfg.kappas[0]
    pbo = float("nan")
    if val_series[kappa_star].shape[0] >= 16:
        pbo = pbo_cscv(time_aligned_pbo_matrix(val_series), n_splits=8)

    if val_only:
        return CellScore(
            run=run,
            kappa_chosen=float(kappa_star),
            ir_headline=float("nan"),
            ir_modeled_cost=float("nan"),
            val_ir_by_kappa={float(k): float(v) for k, v in val_ir.items()},
            cost_stress={},
            break_even=float("nan"),
            pbo=float(pbo),
            activity=float("nan"),
            n_decisions=0,
            per_symbol_decisions={},
            codebook=codebook,
            ohlcv_recon=ohlcv_recon,
            decode_agreement=decode_agreement,
            headline_series=None,
        )

    # HEADLINE at κ*: pooled, netted at the flat 0.30 %; modeled-cost secondary + stress.
    # Money mode (§3a): ONE continuous grid over blocks 1..5 — the MDE-recompute basis;
    # dev/smoke mode: the single headline block (the historical M5-style read).
    hd_grid = (
        primary_region_grid_ms(cfg)
        if cfg.money
        else global_decision_grid_ms(cfg, cfg.headline_block)
    )
    # only reached when val_only is False (the val_only branch returns above), so this grid is
    # always the headline one.
    hd_series, hd_mod, n_dec, _, mu_diag = grid_series(hd_grid, phase="grid")
    if mu_diag:  # §7 v1.4.4: pin the decision-activity AT κ* (the guard's binding-filter input)
        mu_diag["activity_decisions"] = float(
            mu_diag["activity_decisions_by_kappa"][f"{kappa_star:g}"]
        )
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
        codebook=codebook,
        # §7 v1.6.25 (RE-AUDIT R5, FOUND BY THE P1 DRY RUN) — THESE TWO LINES DID NOT EXIST, AND
        # THIS IS THE RETURN THE VERDICT ARTIFACT IS BUILT FROM. The `val_only` return above
        # carried both diagnostics; the FULL return did not, so they defaulted to `{}` and
        # `write_cell_eval_artifact` REFUSED every artifact. Two REQUIRED disclosures (C-12 M1 and
        # C-1) were computed thirty lines earlier and dropped on the exact path that feeds the
        # decision — the R4 shape again, in a different file. Consequence: THE MONEY DRIVER COULD
        # NEVER HAVE WRITTEN A SINGLE EVAL ARTIFACT, and it would have failed AFTER paying for a
        # shard's training. Invisible because the driver had never been run to completion at this
        # configuration, which is precisely what R5 said.
        ohlcv_recon=ohlcv_recon,
        decode_agreement=decode_agreement,
        headline_series=head,
        mu_diag=mu_diag,
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
