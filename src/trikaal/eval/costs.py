"""Transaction-cost model — taker fee + vol-scaled half-spread + impact + perp funding (§8.B.2).

Every quantity is a **fraction of notional** (not bps): ``f_taker = 4e-4`` is 0.04%. The round-trip
execution cost is ``c_total = 2·c_side`` with

    c_side = f_taker + 0.5·spread_frac·max(1, σ_t/σ̄) + k_impact·(Q/ADV)

The half-spread is **volatility-scaled** because the execution filter fires disproportionately on
high-vol bars where the effective spread is wider (a flat spread would understate cost exactly on
the bars we trade); the multiplier is causal (`σ_t` reads ≤ t). Perp **funding** is a separate
position-dependent carry: a long pays positive funding, a short receives it.

Written from scratch; this is the cost half of the headline net-IR, so every component is
known-answer-tested in ``tests/eval/test_costs.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Liquidity-decile default half-spread *fractions* (§8.B.2): BTC/ETH 1bp, top-20 3bp, long tail
# 10bp. Used only when a per-bar effective spread is not reconstructible from data.
SPREAD_DECILE_FRAC: dict[str, float] = {
    "major": 1e-4,  # BTC/ETH — 1 bp
    "top20": 3e-4,  # 3 bp
    "tail": 1e-3,  # long tail — 10 bp
}


@dataclass(frozen=True)
class CostModel:
    """Per-(symbol, bar) round-trip cost in fraction-of-notional units (spec §8.B.2 defaults)."""

    f_taker: float = 4e-4  # 0.04% taker fee
    k_impact: float = 0.1  # linear temporary-impact coefficient (per unit Q/ADV)

    def half_spread(self, spread_frac: float, sigma_t: float, sigma_bar: float) -> float:
        """Vol-scaled half-spread: ``0.5·spread_frac·max(1, σ_t/σ̄)`` (causal; σ_t reads ≤ t)."""
        ratio = sigma_t / sigma_bar if sigma_bar > 0.0 else 1.0
        return 0.5 * spread_frac * max(1.0, ratio)

    def impact(self, q_over_adv: float) -> float:
        """Linear temporary impact ``k_impact·(Q/ADV)`` (Q = notional, ADV = causal 24h $vol)."""
        return self.k_impact * max(0.0, q_over_adv)

    def c_side(
        self, *, spread_frac: float, sigma_t: float, sigma_bar: float, q_over_adv: float
    ) -> float:
        """Per-side cost fraction = taker + vol-scaled half-spread + linear impact."""
        return (
            self.f_taker
            + self.half_spread(spread_frac, sigma_t, sigma_bar)
            + self.impact(q_over_adv)
        )

    def c_total(
        self, *, spread_frac: float, sigma_t: float, sigma_bar: float, q_over_adv: float
    ) -> float:
        """Round-trip (enter + exit) execution cost fraction = ``2·c_side``."""
        return 2.0 * self.c_side(
            spread_frac=spread_frac, sigma_t=sigma_t, sigma_bar=sigma_bar, q_over_adv=q_over_adv
        )


def funding_cost(side: int, funding_rates_in_window: np.ndarray | list[float]) -> float:
    """Perp funding carry over a holding span (§8.B.2): ``s_t · Σ funding_rate_g`` for settlements
    ``g ∈ (t+1, t+h]``. A long (s=+1) pays positive funding (a cost); a short receives it. Spot or a
    span crossing no 8h settlement → 0. The caller supplies only the rates that settle in-window."""
    r = np.asarray(funding_rates_in_window, dtype=np.float64)
    return float(side) * float(r.sum()) if r.size else 0.0


__all__ = ["SPREAD_DECILE_FRAC", "CostModel", "funding_cost"]
