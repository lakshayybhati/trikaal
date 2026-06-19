"""Signal → position → per-trade net P&L → portfolio period-return series (§8.B.1 / §8.B.3).

The execution filter trades only when the predicted edge clears cost with a tuned margin:
``θ = κ·c_total``, ``s_t = +1 if μ̂ > θ, −1 if μ̂ < −θ, 0 otherwise`` (uniform unit notional, v1).
Per executed position the **net** realized log-return is

    r_net = s·log(C_{t+h}/C_{t+1}) − c_total − funding_cost

and the portfolio period-return at each stride-h rebalance period is the equal-weighted mean of
the net returns of all symbols active that period: ``r_p = (1/|A_p|) Σ_{n∈A_p} r_{n,p}^net``.
Periods with no active position are flat and excluded from the series (breadth reported separately).

Known-answer-tested in ``tests/eval/test_strategy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def positions(mu_hat: np.ndarray, theta: np.ndarray | float) -> np.ndarray:
    """Filtered ±1/0 position: +1 if μ̂ > θ, −1 if μ̂ < −θ, else 0 (θ = κ·c_total ≥ 0)."""
    mu = np.asarray(mu_hat, dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    s = np.zeros_like(mu, dtype=np.int64)
    s[mu > th] = 1
    s[mu < -th] = -1
    return s


def net_trade_returns(
    side: np.ndarray,
    fwd_log_return: np.ndarray,
    c_total: np.ndarray,
    funding: np.ndarray | None = None,
) -> np.ndarray:
    """Per-position net log-return ``s·r − c_total − funding`` (round-trip cost subtracted once).

    ``funding`` is the already-signed ``funding_cost`` (long pays positive funding); None → 0.
    """
    s = np.asarray(side, dtype=np.float64)
    r = np.asarray(fwd_log_return, dtype=np.float64)
    c = np.asarray(c_total, dtype=np.float64)
    f = np.zeros_like(s) if funding is None else np.asarray(funding, dtype=np.float64)
    return s * r - c - f


@dataclass(frozen=True)
class PortfolioSeries:
    returns: np.ndarray  # r_p, one per active period (in period order)
    breadth: np.ndarray  # |A_p| concurrent positions backing each r_p
    n_flat_periods: int  # periods on the grid with no active position (held cash)


def portfolio_period_returns(
    period_index: np.ndarray, net_returns: np.ndarray, *, n_periods: int | None = None
) -> PortfolioSeries:
    """Aggregate per-trade net returns into the equal-weighted portfolio period-return series.

    ``period_index[i]`` is the stride-h rebalance period of trade ``i`` (only executed trades,
    side≠0, are passed). ``n_periods`` (if given, the total grid size) lets us count flat periods.
    """
    p = np.asarray(period_index, dtype=np.int64)
    r = np.asarray(net_returns, dtype=np.float64)
    if p.size == 0:
        return PortfolioSeries(np.array([]), np.array([], dtype=np.int64), n_periods or 0)
    uniq = np.unique(p)
    sums = np.zeros(uniq.shape[0], dtype=np.float64)
    counts = np.zeros(uniq.shape[0], dtype=np.int64)
    pos = {int(u): j for j, u in enumerate(uniq)}
    for pi, ri in zip(p, r, strict=True):
        j = pos[int(pi)]
        sums[j] += ri
        counts[j] += 1
    r_p = sums / counts  # equal-weight across active symbols in the period
    total = n_periods if n_periods is not None else int(uniq.shape[0])
    n_flat = max(0, total - int(uniq.shape[0]))
    return PortfolioSeries(returns=r_p, breadth=counts, n_flat_periods=n_flat)


__all__ = [
    "PortfolioSeries",
    "net_trade_returns",
    "portfolio_period_returns",
    "positions",
]
