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

# §8.B.2 defines the "major" tier BY NAME (BTC/ETH 1bp) — a pre-data spec constant, not a
# data-derived rank. The remaining tiers are ranked from TRAIN-REGION liquidity (2026-07-06
# audit item 4: the flat "major" for every symbol under-costed the tail by ~9bp of half-spread).
MAJOR_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
TOP_TIER_N = 20  # majors + next (20 − len(majors)) by rank form the 3bp tier


def assign_spread_deciles(train_stats: dict[str, tuple[float, int]]) -> dict[str, str]:
    """``{symbol: (train_liquidity_proxy, n_bars_train)}`` → ``{symbol: decile label}``.

    Majors are the §8.B.2 spec constants; the rest rank by the CAUSAL train-region proxy
    (higher = more liquid), ties broken by train bar count then symbol name — fully
    deterministic. Ranks 1..(TOP_TIER_N − |majors|) → "top20"; the rest → "tail"."""
    majors = [s for s in MAJOR_SYMBOLS if s in train_stats]
    rest = sorted(
        (s for s in train_stats if s not in MAJOR_SYMBOLS),
        key=lambda s: (-train_stats[s][0], -train_stats[s][1], s),
    )
    n_top = max(0, TOP_TIER_N - len(majors))
    out = {s: "major" for s in majors}
    out.update({s: "top20" for s in rest[:n_top]})
    out.update({s: "tail" for s in rest[n_top:]})
    return out


def load_spread_deciles(path) -> dict[str, float]:
    """Read the committed decile artifact → ``{symbol: spread_frac}`` (the eval driver's input)."""
    import json
    from pathlib import Path

    doc = json.loads(Path(path).read_text())
    return {s: float(rec["spread_frac"]) for s, rec in doc["deciles"].items()}


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


__all__ = [
    "MAJOR_SYMBOLS",
    "SPREAD_DECILE_FRAC",
    "TOP_TIER_N",
    "CostModel",
    "assign_spread_deciles",
    "funding_cost",
    "load_spread_deciles",
]
