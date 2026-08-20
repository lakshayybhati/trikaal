"""Evaluation — written from scratch (no imported implementations).

The purged walk-forward harness with embargo (``folds``, ``harness``), the cross-sectional
decision path (``xsection``), the cost model and net-IR backtest (``costs``, ``strategy``,
``metrics``), the placebo and DSR machinery (``placebo``, ``dsr``, ``paired_bootstrap``,
``tdist``), the verdict assembly and its conformance pins (``verdict``, ``conformance``) and the
M2 diagnostic IC screen (``ic_screen``) all live here.

★ THIS DOCSTRING PREVIOUSLY SAID this package held "ONLY" the IC screen and that the purged
walk-forward harness was "deliberately not started here" — beside sixteen other modules that
implement exactly that. An agent asked to "add a purged walk-forward backtest" would read this,
believe it absent, and write a second one. The external-validation target it also named is
``external_validation``, whose gate was DROPPED as binding and never executed
(``GATE_IS_BINDING = False``); pulling Kronos weights is forbidden by invariant 8.
"""
