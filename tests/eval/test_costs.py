"""Known-answer tests for the transaction-cost model (§8.B.2) — every component hand-computed."""

from __future__ import annotations

import numpy as np

from trikaal.eval.costs import SPREAD_DECILE_FRAC, CostModel, funding_cost


def test_taker_only_round_trip():
    c = CostModel()
    side = c.c_side(spread_frac=0.0, sigma_t=1.0, sigma_bar=1.0, q_over_adv=0.0)
    assert side == 4e-4  # pure taker fee
    assert c.c_total(spread_frac=0.0, sigma_t=1.0, sigma_bar=1.0, q_over_adv=0.0) == 8e-4


def test_half_spread_vol_scaling():
    c = CostModel()
    s = SPREAD_DECILE_FRAC["major"]  # 1e-4
    # σ_t == σ̄ → multiplier max(1,1)=1 → half-spread = 0.5·1e-4 = 5e-5
    assert c.half_spread(s, 1.0, 1.0) == 5e-5
    # σ_t = 2·σ̄ → multiplier 2 → 1e-4
    assert np.isclose(c.half_spread(s, 2.0, 1.0), 1e-4)
    # σ_t = 0.5·σ̄ → floored at 1 (never below the decile spread) → still 5e-5
    assert c.half_spread(s, 0.5, 1.0) == 5e-5


def test_linear_impact():
    c = CostModel()
    assert np.isclose(c.impact(0.01), 1e-3)  # k_impact 0.1 · 0.01
    assert c.impact(-5.0) == 0.0  # negative participation clamped


def test_full_round_trip_hand_computed():
    c = CostModel()
    # f_taker 4e-4 + half-spread (top20 3e-4, no vol scale → 1.5e-4) + impact (0.1·0.005=5e-4)
    side = c.c_side(
        spread_frac=SPREAD_DECILE_FRAC["top20"], sigma_t=1.0, sigma_bar=1.0, q_over_adv=0.005
    )
    assert np.isclose(side, 4e-4 + 1.5e-4 + 5e-4)  # 1.05e-3
    assert np.isclose(
        c.c_total(spread_frac=3e-4, sigma_t=1.0, sigma_bar=1.0, q_over_adv=0.005), 2.1e-3
    )  # 0.21% round trip


def test_funding_sign_and_empty():
    assert np.isclose(funding_cost(+1, [1e-4, 1e-4]), 2e-4)  # long pays positive funding
    assert np.isclose(funding_cost(-1, [1e-4, 1e-4]), -2e-4)  # short receives it
    assert funding_cost(+1, []) == 0.0  # no settlement crossed
    assert np.isclose(funding_cost(+1, [-3e-4]), -3e-4)  # negative funding → long receives
