"""§7 v1.6.15 — the FAIL-OPEN class, KAT'd. Every guard that was fixed, plus the pins C-18 added.

Each test here follows the ninth standing norm: it asserts the gate REJECTS the predecessor
behaviour, and (where the assertion could pass vacuously) it first proves the fixture can
discriminate the two cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.eval import conformance as C
from trikaal.eval import verdict as V
from trikaal.eval.xsection import MU_ESTIMATOR, XSectionConfig
from trikaal.run.matrix import shard_partition_failures
from trikaal.train.gates import MICRO_DIMS, _stratified_blocked_split, micro_legibility_gate

# ------------------------------------------------------------------ S-2: power_guard fails closed
_CLAIM = {"d45": {"delta_ir": 1.1, "cells": (4, 5)}}


def _evals(series_by_cell) -> dict:
    return {
        (c, s): {"headline_series": list(series_by_cell(c, s))}
        for c in (1, 2, 3, 4, 5)
        for s in V.DSR_SEEDS
    }


def test_power_guard_is_armed_only_when_it_read_its_inputs():
    healthy = _evals(lambda c, s: np.linspace(-1, 1, 64) + 0.01 * (c + s))
    ok = V.power_guard(healthy, _CLAIM)
    # FIXTURE DISCRIMINATION: the healthy arm must be armed, or "not armed" proves nothing.
    assert ok["armed"] is True and ok["unreadable_inputs"] == []

    blind = _evals(lambda c, s: [float("nan")] * 64)
    got = V.power_guard(blind, _CLAIM)
    assert got["armed"] is False, "a hardcoded armed=True is the C-2 defect (S-2)"
    assert got["halted"] is True, "a guard that read nothing must HALT, not pass quietly"
    assert got["unreadable_inputs"], "the unreadable units must be NAMED, not merely counted"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_power_guard_halts_on_an_unevaluable_claim(bad):
    healthy = _evals(lambda c, s: np.linspace(-1, 1, 64) + 0.01 * (c + s))
    got = V.power_guard(healthy, {"d45": {"delta_ir": bad, "cells": (4, 5)}})
    assert got["halted"] is True and got["armed"] is False


def test_power_guard_stays_halt_only():
    """It may never flip SURVIVES<->NULL — it only refuses to report."""
    healthy = _evals(lambda c, s: np.linspace(-1, 1, 64) + 0.01 * (c + s))
    assert set(V.power_guard(healthy, _CLAIM)) >= {"armed", "halted", "underpowered_claims"}
    assert "primary" not in V.power_guard(healthy, _CLAIM)


# ------------------------------------------- the decode disclosure refuses to report a null result
def _decode_evals(present: bool) -> dict:
    da = {"sign_agreement_dim0": 0.9, "single_bar_degenerate": False}
    return {
        (c, s): {"decode_agreement": (dict(da) if present else None)}
        for c in (1, 2, 3, 4, 5)
        for s in V.DSR_SEEDS
    }


def test_decode_disclosure_reports_unmeasurable_instead_of_a_quiet_null():
    ok = V.decode_agreement_disclosure(_decode_evals(True))
    assert ok["unmeasurable"] is False, "the healthy arm must be measurable (discrimination)"

    blind = V.decode_agreement_disclosure(_decode_evals(False))
    assert blind["unmeasurable"] is True
    assert blind["unreadable_inputs"]
    assert "UNMEASURABLE" in blind["primary_pair_test_cell4_vs_cell5"]["reading"]


# ------------------------------------------------------- pinned_threshold_failures: a deleted pin
def test_deleting_a_pin_is_a_failure_not_an_absence(monkeypatch):
    assert C.pinned_threshold_failures() == [], "baseline must be clean (discrimination)"
    key = next(iter(C.PINNED_THRESHOLDS))
    monkeypatch.setattr(
        C, "PINNED_THRESHOLDS", {k: v for k, v in C.PINNED_THRESHOLDS.items() if k != key}
    )
    fails = C.pinned_threshold_failures()
    assert any("NO entry in PINNED_THRESHOLDS" in f for f in fails), fails


def test_an_extra_pin_with_no_live_constant_is_a_failure(monkeypatch):
    monkeypatch.setattr(C, "PINNED_THRESHOLDS", {**C.PINNED_THRESHOLDS, "GHOST": 1.0})
    assert any("not a live verdict constant" in f for f in C.pinned_threshold_failures())


# ------------------------------------------------------------------- an empty matrix is not a cover
def test_empty_unit_matrix_is_rejected():
    units = [(type("S", (), {"name": "cell1"})(), 0), (type("S", (), {"name": "cell2"})(), 0)]
    assert shard_partition_failures(units, 1) == [], "a real matrix must pass (discrimination)"
    assert shard_partition_failures([], 1), "a partition of zero units is not a covered matrix"


# ------------------------------------------------------------------------------ C-10, both legs
class _W:
    def __init__(self, symbol, x, mask):
        self.symbol, self.x, self.mask = symbol, x, mask


def _gate(monkeypatch, n_thin: int, acc: float):
    import trikaal.train.gates as G

    monkeypatch.setattr(G, "id_legibility_sign_acc", lambda *a, **k: acc)
    rng = np.random.default_rng(0)
    t, f = 12_000, 16
    x = rng.standard_normal((t, f))
    mask = np.zeros((t, f), dtype=np.uint8)
    for d in list(MICRO_DIMS)[:n_thin]:
        mask[:-500, d] = 1
    toks = {"S": (rng.integers(0, 8, t), rng.integers(0, 8, t))}
    return micro_legibility_gate(None, [_W("S", x, mask)], toks)


def test_a_healthy_gate_still_passes(monkeypatch):
    """DISCRIMINATION FIRST: if this failed, every rejection below would be vacuous."""
    assert _gate(monkeypatch, 0, 0.95)["pass"] is True


@pytest.mark.parametrize("n_thin", [1, 5, 6])
def test_a_skipped_dim_can_never_contribute_to_a_pass(monkeypatch, n_thin):
    """C-10 leg 1. The predecessor `continue`d BEFORE the `ok` update, so five skipped dims plus
    one dense passing dim returned pass=True on a gate whose contract is all six."""
    with pytest.raises(RuntimeError, match="could not be measured"):
        _gate(monkeypatch, n_thin, 0.95)


def test_stratified_split_covers_every_symbol_and_blocks_in_time():
    """C-10 leg 2. Measured on the real lake, the old head-150k window spanned 1 of 200 symbols."""
    groups = np.repeat([f"S{i}" for i in range(200)], 1000).astype(object)
    sel, tr = _stratified_blocked_split(groups, 150_000)
    assert len({groups[i] for i in sel}) == 200, "every stratum must contribute"
    assert 0.75 < tr.mean() < 0.85
    # blocked in time WITHIN a stratum: no train row may follow a val row of the same stratum
    for g in ("S0", "S199"):
        idx = [k for k, i in enumerate(sel) if groups[i] == g]
        flags = tr[idx]
        assert not (flags[:-1] < flags[1:]).any(), "train must precede val inside a stratum"


def test_stratified_split_fails_closed_on_an_empty_grouping():
    with pytest.raises(ValueError, match="selected no rows"):
        _stratified_blocked_split(np.array([], dtype=object), 100)


# ---------------------------------------------------------------------------------- C-18 the pins
def test_mu_estimator_pin_is_enforced_on_the_money_path():
    assert MU_ESTIMATOR == C.PINNED_MU_ESTIMATOR == "expectation"
    with pytest.raises(ValueError, match="mu_estimator"):
        XSectionConfig(
            money=True,
            cap_per_symbol=None,
            spread_frac_by_symbol={"A": 1e-4},
            mu_estimator="argmax",
        )
    with pytest.raises(ValueError, match="mu_estimator"):
        XSectionConfig(
            money=True,
            cap_per_symbol=None,
            spread_frac_by_symbol={"A": 1e-4},
            mu_estimator="mc_mean",
        )
    # discrimination: the pinned value must CONSTRUCT, or the rejections prove nothing
    XSectionConfig(money=True, cap_per_symbol=None, spread_frac_by_symbol={"A": 1e-4})


def test_a_drifted_xsection_constant_is_caught_by_the_gate(monkeypatch):
    import trikaal.eval.xsection as X

    assert C.pinned_threshold_failures() == []
    monkeypatch.setattr(X, "MU_ESTIMATOR", "argmax")
    assert any("MU_ESTIMATOR" in f for f in C.pinned_threshold_failures())


@pytest.mark.parametrize("field", ["steps_stage1", "steps_stage2"])
def test_step_budget_is_pinned_on_a_money_run(field):
    from trikaal.eval.conformance import ConformanceError
    from trikaal.train.orchestrator import OrchestratorConfig

    OrchestratorConfig(money_run=True)  # discrimination: the pinned config must construct
    with pytest.raises(ConformanceError, match=field):
        OrchestratorConfig(money_run=True, **{field: 1234})


def test_step_pins_are_referenced_not_restated():
    from trikaal.train.orchestrator import OrchestratorConfig

    cfg = OrchestratorConfig(money_run=False)
    assert cfg.steps_stage1 == C.PINNED_STEPS_STAGE1
    assert cfg.steps_stage2 == C.PINNED_STEPS_STAGE2
