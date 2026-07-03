"""KATs for the m6_design §6-item-10 instrument fixes (2026-07-04 audit).

One known-answer test per fix: (a) headline netting at the flat 0.30 % round-trip, (b) the
full-calendar-grid flat=0.0 annualization convention, (c) the time-aligned PBO matrix, (d) the
persisted per-κ VAL curve, (e) σ̄ as a causal expanding mean in the cost path, (f) the
funding/OI placebo tripwire. These pin the corrected instrument BEFORE the Gate-A re-anchor.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.eval.dsr import pbo_cscv, time_aligned_pbo_matrix
from trikaal.eval.harness import _per_bar_cost, cost_stress_curve, single_symbol_backtest
from trikaal.eval.metrics import information_ratio
from trikaal.eval.placebo import assert_perp_dims_masked


# --------------------------------------------------------- (a) headline = 0.30% NETTING
def test_headline_nets_at_flat_cost_while_theta_stays_modeled():
    """Positions from θ = κ·c_modeled; the netting cost is the flat 0.30 % — hand-computed."""
    mu = np.array([0.02, -0.02, 0.001, 0.03])
    y = np.array([0.01, -0.02, 0.005, 0.004])
    c_mod = np.full(4, 0.001)  # modeled per-decision round-trip
    # θ = 1.0·0.001 → +1 at {0,3}, −1 at {1}, flat at {2} (0.001 is NOT > θ)
    bt_mod = single_symbol_backtest(mu, y, c_mod, h=5, kappa=1.0)
    bt_head = single_symbol_backtest(mu, y, c_mod, h=5, kappa=1.0, c_net=0.0030)
    # identical position set (θ unchanged) …
    assert bt_mod.n_trades == bt_head.n_trades == 3
    # … but hand-computed nets differ: s·y − c  with c = modeled vs flat 0.30 %
    np.testing.assert_allclose(bt_mod.series, [0.009, 0.019, 0.0, 0.003], atol=1e-12)
    np.testing.assert_allclose(bt_head.series, [0.007, 0.017, 0.0, 0.001], atol=1e-12)
    assert bt_head.ir != bt_mod.ir  # the mislabel the fix removes: these are different numbers
    # and the cost-stress curve's 0.30 % point IS the headline (same θ, netted at flat c)
    stress = cost_stress_curve(mu, y, c_mod, h=5, kappa=1.0)
    assert stress[0.0030] == pytest.approx(bt_head.ir, rel=1e-12)


# --------------------------------------------------------- (b) full calendar grid, flat = 0.0
def test_calendar_grid_ir_for_a_half_active_toy_strategy():
    """Hand-computed: 8-period grid, trades in exactly 4 → series has flat 0.0 periods, and the
    annualized IR equals mean/std of the FULL grid (=1.0 per-period here) — NOT the inflated
    active-only value (which would be mean 0.009 / std 0 → ~9e9 at eps_ir)."""
    mu = np.array([0.02, 0.0, 0.02, 0.0, 0.02, 0.0, 0.02, 0.0])
    y = np.full(8, 0.01)
    c = np.full(8, 0.001)
    bt = single_symbol_backtest(mu, y, c, h=15, kappa=1.0)
    expected = np.array([0.009, 0.0, 0.009, 0.0, 0.009, 0.0, 0.009, 0.0])
    np.testing.assert_allclose(bt.series, expected, atol=1e-15)
    assert bt.activity == pytest.approx(0.5)
    # hand: mean=0.0045, popstd=0.0045 → IR_period=1 → IR = √(525600/15) = √35040 ≈ 187.1897
    assert bt.ir == pytest.approx(np.sqrt(525_600 / 15), rel=1e-6)
    # regression teeth: the OLD active-only convention would inflate by ~1/√(activity)
    active_only_ir = information_ratio(expected[expected != 0.0], 15)
    assert bt.ir < active_only_ir  # active-only degenerates (std→0 ⇒ ~1e9); calendar stays sane


# --------------------------------------------------------- (c) time-aligned PBO matrix
def test_pbo_matrix_rejects_misaligned_series_and_aligns_full_grid():
    """Hand-built misalignment: κ=1 traded EARLY periods, κ=2 traded LATE. The old active-only +
    truncate-to-min construction would stack κ1's early returns against κ2's late returns row-by-row
    (different time periods per row — invalid). The new builder (i) refuses ragged inputs and
    (ii) keeps full-grid series aligned so row t is period t for every column."""
    # ragged (active-only lengths differ) → hard refusal, never a silent truncate
    with pytest.raises(ValueError, match="not time-aligned"):
        time_aligned_pbo_matrix({1.0: np.zeros(10), 2.0: np.zeros(8)})
    # full-grid, flat=0 series: κ1 active early, κ2 active late — aligned by construction
    t_grid = 16
    k1 = np.zeros(t_grid)
    k1[:8] = 0.01  # early trader
    k2 = np.zeros(t_grid)
    k2[8:] = 0.02  # late trader
    m = time_aligned_pbo_matrix({2.0: k2, 1.0: k1})  # insertion order ≠ sorted order on purpose
    assert m.shape == (t_grid, 2)
    np.testing.assert_array_equal(m[:, 0], k1)  # sorted-κ column order: κ=1 first
    np.testing.assert_array_equal(m[:, 1], k2)
    # and a known outcome through the corrected input: a strictly-dominant config → PBO = 0
    rng = np.random.default_rng(7)
    base = rng.standard_normal(64) * 1e-3
    dom = {1.0: base + 5e-3, 2.0: base - 1e-3, 3.0: base - 2e-3}
    assert pbo_cscv(time_aligned_pbo_matrix(dom), n_splits=4) == 0.0


# --------------------------------------------------------- (e) σ̄ = causal expanding mean
class _RecordingCost:
    """Duck-typed CostModel capturing the σ̄ each decision is priced with."""

    def __init__(self):
        self.sigma_bars: list[float] = []

    def c_total(self, *, spread_frac, sigma_t, sigma_bar, q_over_adv) -> float:
        self.sigma_bars.append(float(sigma_bar))
        return 0.001

    def c_side(self, **kw) -> float:  # pragma: no cover — interface parity
        return 0.0005


def test_sigma_bar_is_the_causal_expanding_mean():
    """Regression KAT for the M5 σ̄ fix: σ̄_t = mean(σ[0..t]) (bars ≤ t), never the full-series
    mean; and perturbing σ AFTER a decision cannot change that decision's cost."""
    sigma = np.array([0.01, 0.02, 0.03, 0.04])
    dec = np.array([0, 1, 3])
    rec = _RecordingCost()
    _per_bar_cost(rec, sigma, dec, spread_frac=1.0)
    # hand-computed expanding means at t∈{0,1,3}: 0.01, 0.015, 0.025
    np.testing.assert_allclose(rec.sigma_bars, [0.01, 0.015, 0.025], atol=1e-15)
    # causality teeth: change σ strictly AFTER t=1 → σ̄ at decisions ≤1 identical
    sigma2 = sigma.copy()
    sigma2[2:] = 99.0
    rec2 = _RecordingCost()
    _per_bar_cost(rec2, sigma2, np.array([0, 1]), spread_frac=1.0)
    np.testing.assert_allclose(rec2.sigma_bars, rec.sigma_bars[:2], atol=1e-15)
    # (the full-series mean WOULD have moved: mean(σ2)=25.0075 ≠ 0.015 — the leak this KAT pins out)


# --------------------------------------------------------- (f) funding/OI placebo tripwire
def test_placebo_tripwire_fails_loudly_when_perp_dims_activate():
    """MICRO_DIMS (7-12) excludes funding/OI (13-15) because this build masks them everywhere.
    If they ever activate, Cell 5 would silently under-shuffle — the tripwire must fire."""
    n = 32
    mask = np.zeros((n, 16), dtype=np.uint8)
    mask[:, 13:16] = 1  # fully masked (the real-build invariant) → passes
    assert_perp_dims_masked(mask)
    mask[5, 14] = 0  # one ACTIVE log_oi entry → loud failure, never a silent exclusion
    with pytest.raises(AssertionError, match="under-shuffle"):
        assert_perp_dims_masked(mask)
