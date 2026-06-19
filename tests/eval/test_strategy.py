"""Known-answer tests for signal→position→net-P&L→portfolio aggregation (§8.B.1/§8.B.3)."""

from __future__ import annotations

import numpy as np

from trikaal.eval.strategy import (
    net_trade_returns,
    portfolio_period_returns,
    positions,
)


def test_execution_filter_positions():
    mu = np.array([0.01, -0.01, 0.0005, -0.0005, 0.002])
    s = positions(mu, theta=0.001)  # trade only when |μ̂| > θ
    assert list(s) == [1, -1, 0, 0, 1]


def test_net_trade_returns_hand_computed():
    side = np.array([1, -1])
    ret = np.array([0.02, 0.02])  # log(C_{t+h}/C_{t+1})
    c = np.array([0.001, 0.001])
    funding = np.array([0.0002, -0.0002])  # long pays, short receives
    r = net_trade_returns(side, ret, c, funding)
    # long: +0.02 - 0.001 - 0.0002 = 0.0188 ; short: -0.02 - 0.001 + 0.0002 = -0.0208
    assert np.allclose(r, [0.0188, -0.0208])


def test_portfolio_period_aggregation_equal_weight():
    period = np.array([0, 0, 1])
    net = np.array([0.01, 0.03, -0.02])
    ps = portfolio_period_returns(period, net, n_periods=4)
    assert np.allclose(ps.returns, [0.02, -0.02])  # period 0 = mean(0.01,0.03)
    assert list(ps.breadth) == [2, 1]
    assert ps.n_flat_periods == 2  # 4 grid periods, 2 had no position


def test_empty_portfolio_is_safe():
    ps = portfolio_period_returns(np.array([], dtype=np.int64), np.array([]), n_periods=10)
    assert ps.returns.size == 0 and ps.n_flat_periods == 10
