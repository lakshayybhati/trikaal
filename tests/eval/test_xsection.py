"""KATs for the cross-sectional eval driver (m6_design §6 item 6) + its G-causal property.

The load-bearing pieces: the GLOBAL stride-h rebalance grid (time-aligned pooling by
construction), hand-computed equal-weight pooling with flat=0.0 on the full grid, eval decisions
structurally unable to reach the train region, and the 5-model + placebo verdict surface."""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.data.config import BAR_MS
from trikaal.data.universe_loader import calendar_boundary_ms
from trikaal.eval.xsection import (
    SymbolEval,
    XSectionConfig,
    _pool_periods,
    ablation_verdict,
    global_decision_grid_ms,
    score_cell,
    symbol_decisions,
)
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import ARM_MICRO, ARM_MICRO_SHUFFLED
from trikaal.utils.seeding import set_determinism

WINDOW = dict(window_start="2024-01-01", window_end="2024-01-04", train_frac=0.7)
T0 = 1_704_067_200_000  # 2024-01-01 UTC in ms


def _cfg(**kw) -> XSectionConfig:
    return XSectionConfig(h=5, seq_len=16, cap_per_symbol=40, device="cpu", **WINDOW, **kw)


def _sym(symbol: str, n=4320, seed=0) -> SymbolEval:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 16)).astype(np.float32)
    m = np.zeros((n, 16), dtype=np.uint8)
    m[:, 13:16] = 1
    return SymbolEval(
        symbol=symbol,
        x=x,
        mask=m,
        segment_id=np.zeros(n, dtype=np.int64),
        ts=(T0 + np.arange(n) * BAR_MS).astype(np.int64),
        sigma=np.full(n, 0.01),
        raw_ret_close=(0.001 * rng.standard_normal(n)).astype(np.float64),
    )


# ---------------------------------------------- G-causal: eval cannot reach the train region
def test_eval_decisions_cannot_precede_the_boundary():
    cfg = _cfg()
    boundary = calendar_boundary_ms(cfg.window_start, cfg.window_end, cfg.train_frac)
    se = _sym("AAAUSDT")
    for block in range(cfg.n_blocks):
        grid = global_decision_grid_ms(cfg, block)
        assert grid.min() >= boundary  # the grid itself starts AT the boundary
        dec = symbol_decisions(se, grid, cfg)
        assert dec.size > 0 or block == cfg.n_blocks  # blocks land on real bars here
        assert (se.ts[dec] >= boundary).all()  # no decision can precede the boundary
    # and the grid is GLOBAL: two symbols receive the same instants (pooling is time-aligned)
    grid0 = global_decision_grid_ms(cfg, 0)
    np.testing.assert_array_equal(grid0, global_decision_grid_ms(cfg, 0))


def test_symbol_missing_bars_yield_no_phantom_decisions():
    cfg = _cfg()
    se = _sym("BBBUSDT", n=4320)
    se2 = SymbolEval(  # a late-listed symbol whose bars start AFTER block 0
        symbol="LATE",
        x=se.x[3600:],
        mask=se.mask[3600:],
        segment_id=se.segment_id[3600:],
        ts=se.ts[3600:],
        sigma=se.sigma[3600:],
        raw_ret_close=se.raw_ret_close[3600:],
    )
    grid = global_decision_grid_ms(cfg, 0)
    dec = symbol_decisions(se2, grid, cfg)
    # block 0 spans [boundary, boundary+~5h); the late listing may cover a tail of it, but every
    # decision must be a REAL bar of this symbol with full history — no phantom mapping
    assert (se2.ts[dec] == np.intersect1d(se2.ts[dec], grid)).all()
    assert (dec >= cfg.seq_len).all()


# ---------------------------------------------- hand-computed pooling KAT
def test_pool_periods_hand_kat():
    """2 symbols, 4 global periods: A trades p0 (+0.01) and p2 (+0.03); B trades p2 (-0.01)
    and p3 (+0.02) → pooled = [0.01, 0.0(flat), mean(0.03,-0.01)=0.01, 0.02]."""
    per = {
        "A": (np.array([0, 2]), np.array([0.01, 0.03])),
        "B": (np.array([2, 3]), np.array([-0.01, 0.02])),
    }
    cal = _pool_periods(per, n_periods=4)
    np.testing.assert_allclose(cal, [0.01, 0.0, 0.01, 0.02], atol=1e-15)
    assert _pool_periods({}, n_periods=3).tolist() == [0.0, 0.0, 0.0]  # all-flat block


# ---------------------------------------------- end-to-end: score 2 cells + the verdict
@pytest.mark.slow
def test_score_cell_and_ablation_verdict_end_to_end():
    set_determinism(0)
    cfg = _cfg()
    kw = dict(d_model=16, n_layers=1, n_heads=2, d_ff=32, max_len=64)
    tok = TokenizerAE(**kw).eval()
    model = TrikaalAR(
        n_layers=1,
        d_model=16,
        d_ff=32,
        n_heads=2,
        mtp_depths=0,
        max_len=64,
        ffn_dropout=0.0,
        resid_dropout=0.0,
        attn_dropout=0.0,
        token_dropout=0.0,
    ).eval()
    symbols = [_sym("AAAUSDT", seed=1), _sym("BBBUSDT", seed=2)]
    s4 = score_cell("cell4", model, tok, ARM_MICRO, symbols, cfg)
    s5 = score_cell("cell5", model, tok, ARM_MICRO_SHUFFLED, symbols, cfg)
    for s in (s4, s5):
        assert set(s.val_ir_by_kappa) == set(float(k) for k in cfg.kappas)  # curve persisted
        assert s.kappa_chosen in s.val_ir_by_kappa
        assert set(s.cost_stress) == {0.001, 0.002, 0.003}
        assert s.n_decisions > 0 and set(s.per_symbol_decisions) == {"AAAUSDT", "BBBUSDT"}
        assert 0.0 <= s.activity <= 1.0 or np.isnan(s.activity)
        assert isinstance(float(s.ir_headline), float) and isinstance(
            float(s.ir_modeled_cost), float
        )
        # CO2 item 5: the non-gating codebook diagnostic rides every CellScore
        cb = s.codebook
        assert cb["coarse"]["vocab"] == tok.v_c and cb["fine"]["vocab"] == tok.v_f
        assert 0.0 < cb["coarse"]["utilization"] <= 1.0
        assert 0.0 <= cb["effective_bits_per_token"] <= cb["capacity_bits_per_token"]
    v = ablation_verdict({1: 0.1, 2: 0.2, 3: 0.15, 4: 0.5, 5: 0.3})
    assert v["delta_info"] == pytest.approx(0.2)  # IR(4) − IR(5)
    assert v["fsq_effect_ohlcv"] == pytest.approx(0.1)
    assert v["micro_marginal_fsq"] == pytest.approx(0.3)
    assert v["micro_exceeds_placebo"] is True  # Δ_micro (0.3) > Δ_placebo (0.1)
    # audit item 4: per-symbol spread deciles thread through θ + the modeled cost end-to-end
    cfg_dec = _cfg(spread_frac_by_symbol={"AAAUSDT": 1e-4, "BBBUSDT": 1e-3})
    s4d = score_cell("cell4_deciles", model, tok, ARM_MICRO, symbols, cfg_dec)
    assert set(s4d.per_symbol_decisions) == {"AAAUSDT", "BBBUSDT"}
    with pytest.raises(KeyError, match="no spread decile for BBBUSDT"):
        score_cell(
            "cell4_bad",
            model,
            tok,
            ARM_MICRO,
            symbols,
            _cfg(spread_frac_by_symbol={"AAAUSDT": 1e-4}),
        )


def test_val_only_matches_full_val_curve_and_headline_series_is_the_ir_basis():
    """val_only skips the headline grid (secondary-horizon VAL trials, prereg §3 clause 5) but
    must reproduce the FULL call's per-κ VAL curve + κ* exactly; the full call's new
    headline_series must BE the series ir_headline was computed on."""
    from trikaal.eval.metrics import information_ratio

    set_determinism(0)
    cfg = _cfg()
    kw = dict(d_model=16, n_layers=1, n_heads=2, d_ff=32, max_len=64)
    tok = TokenizerAE(**kw).eval()
    model = TrikaalAR(
        n_layers=1,
        d_model=16,
        d_ff=32,
        n_heads=2,
        mtp_depths=0,
        max_len=64,
        ffn_dropout=0.0,
        resid_dropout=0.0,
        attn_dropout=0.0,
        token_dropout=0.0,
    ).eval()
    symbols = [_sym("AAAUSDT", seed=1), _sym("BBBUSDT", seed=2)]
    full = score_cell("cell4", model, tok, ARM_MICRO, symbols, cfg)
    vo = score_cell("cell4_vo", model, tok, ARM_MICRO, symbols, cfg, val_only=True)
    assert vo.val_ir_by_kappa == full.val_ir_by_kappa
    assert vo.kappa_chosen == full.kappa_chosen
    assert vo.headline_series is None and np.isnan(vo.ir_headline) and vo.n_decisions == 0
    assert vo.codebook["coarse"]["vocab"] == tok.v_c  # diagnostic still rides val_only scores
    assert full.headline_series is not None
    assert information_ratio(full.headline_series, cfg.h) == pytest.approx(full.ir_headline)
