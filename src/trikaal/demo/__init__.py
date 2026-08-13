"""Trikaal v1 live demo — FORECAST ONLY.

★★ THIS PACKAGE COMPUTES NO PROFIT AND LOSS, AND THAT IS ARCHITECTURAL, NOT EDITORIAL.

There is no equity curve, no cumulative return, no position, no Sharpe, no trade count and no
"strategy" anywhere in this package. Those quantities are never CALCULATED, so they cannot be
rendered, exported, screenshotted or accidentally surfaced by a later contributor's small
addition without a visible code change. ``tests/demo/test_no_pnl.py`` asserts the absence and is
designed to FAIL if any of it is reintroduced.

The reason is our own published measurement, not squeamishness. On the three banked cell-1 money
units the net information ratio is -28.47 / -67.02 / -146.49 across the entire pre-registered
0.10-0.30 % cost band, and the break-even round-trip cost is 0.0023-0.0208 % against a realistic
~0.10 % taker round trip (``runs_manifest/m6_horizon_break_even.json``). A demo that implied
tradeability would be this project's own artifact contradicting this project's own headline
result, and under the context-stripping rule a caption cannot be trusted to prevent that — only
the absence of the number can.

WHAT IT DOES SHOW: mu-hat, the model's expected cumulative log-return over the next h minutes,
for THREE SEEDS AT ONCE, never averaged.
"""

from trikaal.demo.inference import (
    DEMO_CELL,
    DEMO_H,
    ForecastBundle,
    SeedForecast,
    available_seeds,
    forecast_at,
    load_unit,
)

__all__ = [
    "DEMO_CELL",
    "DEMO_H",
    "ForecastBundle",
    "SeedForecast",
    "available_seeds",
    "forecast_at",
    "load_unit",
]
