"""Headline metric — cost-aware net Information Ratio on a portfolio period-return series (§8.B.3).

The IR is computed on the **portfolio period-return series** ``{r_p}`` (one return per stride-h
rebalance period, equal-weighted across concurrently-active symbols — built in ``strategy.py``),
NOT on pooled per-trade returns. Annualization multiplies the period-series std by
``sqrt(periods_per_year)`` so horizons are comparable:

    IR_period = mean_p / (std_p + eps_ir)            # population std over portfolio periods
    IR_net    = IR_period · sqrt(525600 / h)

``c_break`` (the round-trip cost at which IR_net crosses 0) is a first-class robustness number.
Known-answer-tested in ``tests/eval/test_metrics.py``.
"""

from __future__ import annotations

import numpy as np

MINUTES_PER_YEAR = 525_600
EPS_IR = 1e-12


def periods_per_year(h: int) -> float:
    """Stride-h rebalance periods in a year: 525600 / h (h=1→525600, 15→35040, 60→8760)."""
    return MINUTES_PER_YEAR / h


def information_ratio(portfolio_returns: np.ndarray, h: int, *, eps_ir: float = EPS_IR) -> float:
    """Annualized net-IR of a portfolio period-return series at horizon ``h`` (population std)."""
    r = np.asarray(portfolio_returns, dtype=np.float64)
    if r.size < 2:
        return float("nan")
    ir_period = float(r.mean()) / (float(r.std(ddof=0)) + eps_ir)
    return ir_period * float(np.sqrt(periods_per_year(h)))


def net_sharpe(portfolio_returns: np.ndarray, h: int) -> float:
    """Net Sharpe on the same series — identical to IR_net here (risk-free 0, no benchmark)."""
    return information_ratio(portfolio_returns, h)


def incremental_ir(strategy_returns: np.ndarray, benchmark_returns: np.ndarray, h: int) -> float:
    """Headline is INCREMENTAL net-IR over the best naive benchmark (§8.B.4)."""
    ir = information_ratio
    return ir(strategy_returns, h) - ir(benchmark_returns, h)


def break_even_cost(cost_grid: np.ndarray, ir_at_cost: np.ndarray) -> float:
    """``c_break`` — the round-trip cost where IR_net(c) crosses 0, by linear interpolation.

    ``cost_grid`` ascending, ``ir_at_cost`` the IR_net at each. Returns the first zero-crossing; if
    IR > 0 across the whole grid returns ``+inf`` (robust beyond the swept band), if IR ≤ 0 across
    it returns ``-inf`` (cost-fragile below the band).
    """
    c = np.asarray(cost_grid, dtype=np.float64)
    ir = np.asarray(ir_at_cost, dtype=np.float64)
    order = np.argsort(c)
    c, ir = c[order], ir[order]
    if np.all(ir > 0.0):
        return float("inf")
    if np.all(ir <= 0.0):
        return float("-inf")
    for i in range(len(c) - 1):
        if ir[i] == 0.0:
            return float(c[i])
        if ir[i] > 0.0 >= ir[i + 1] or ir[i] < 0.0 <= ir[i + 1]:
            # linear interpolation for IR(c)=0 between grid points i, i+1
            t = ir[i] / (ir[i] - ir[i + 1])
            return float(c[i] + t * (c[i + 1] - c[i]))
    return float(c[-1])


__all__ = [
    "EPS_IR",
    "MINUTES_PER_YEAR",
    "break_even_cost",
    "incremental_ir",
    "information_ratio",
    "net_sharpe",
    "periods_per_year",
]
