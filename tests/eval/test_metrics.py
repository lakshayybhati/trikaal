"""Known-answer tests for the net-IR metrics (§8.B.3) — hand-computed."""

from __future__ import annotations

import numpy as np

from trikaal.eval.metrics import (
    break_even_cost,
    incremental_ir,
    information_ratio,
    periods_per_year,
)


def test_periods_per_year():
    assert periods_per_year(1) == 525_600
    assert periods_per_year(5) == 105_120
    assert periods_per_year(15) == 35_040
    assert periods_per_year(60) == 8_760


def test_information_ratio_known_answer():
    # r_p = [2,0,2,0]: mean 1, population std 1 → IR_period ≈ 1 → IR_net = sqrt(periods_per_year)
    r = np.array([2.0, 0.0, 2.0, 0.0])
    assert np.isclose(information_ratio(r, 15), np.sqrt(35_040), rtol=1e-9)
    assert np.isclose(information_ratio(r, 60), np.sqrt(8_760), rtol=1e-9)
    # zero-mean series → IR 0
    assert np.isclose(information_ratio(np.array([1.0, -1.0, 1.0, -1.0]), 15), 0.0)


def test_break_even_cost_interpolates_zero_crossing():
    cost = np.array([0.001, 0.002, 0.003, 0.004])
    ir = np.array([3.0, 1.0, -1.0, -3.0])  # crosses 0 halfway between 0.002 and 0.003
    assert np.isclose(break_even_cost(cost, ir), 0.0025)
    assert break_even_cost(cost, np.array([3.0, 2.0, 1.0, 0.5])) == float("inf")  # robust
    assert break_even_cost(cost, np.array([-1.0, -2.0, -3.0, -4.0])) == float("-inf")  # fragile


def test_incremental_ir():
    strat = np.array([2.0, 0.0, 2.0, 0.0])  # IR_net = sqrt(35040)
    bench = np.array([1.0, -1.0, 1.0, -1.0])  # IR_net = 0
    assert np.isclose(incremental_ir(strat, bench, 15), np.sqrt(35_040))
