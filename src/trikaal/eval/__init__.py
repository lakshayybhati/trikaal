"""Evaluation metrics — written from scratch (no imported implementations).

v1 scope note: this package currently holds ONLY the Milestone-2 *diagnostic* IC screen
(``ic_screen``) — the spec §8.E.8 "first cut of MDE / effective-N on the single coin-year,
alongside the per-feature causal IC screen." It is **not** the full eval harness (purged
walk-forward + embargo, the 2×2 ablation runner, the cost-aware net-IR backtest, the Kronos
external-validation target) — that subsystem is a later milestone and is deliberately not
started here.
"""
