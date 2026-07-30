"""KATs for the §7 v1.5 pin changes — including MUTATION tests that the conformance gate REJECTS
the pre-v1.5 values.

A pin that is merely changed is not pinned. Each amendment below is asserted twice: the new value
is in force, AND restoring the old value is caught by the gate before any DSR is computed.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.eval.conformance import PINNED_DSR, PINNED_SEEDS, verdict_dsr_failures
from trikaal.eval.tdist import student_t_ppf, welch_satterthwaite_df
from trikaal.eval.verdict import (
    DSR_HORIZONS,
    DSR_KAPPAS,
    DSR_N_TRIALS,
    DSR_SEEDS,
    DSR_VAR_SR_BASIS_CELL,
    PLACEBO_DISPERSION_TRIPWIRE,
)


def _trials() -> dict:
    """A full enumerated trial set with distinct per-cell dispersions."""
    rng = np.random.default_rng(7)
    return {
        (c, s, h, k): float(rng.normal(0.0, 0.001 * c))
        for c in (1, 2, 3, 4, 5)
        for s in DSR_SEEDS
        for h in DSR_HORIZONS
        for k in DSR_KAPPAS
    }


def _var_placebo(trials: dict) -> float:
    keys = sorted(k for k in trials if k[0] == DSR_VAR_SR_BASIS_CELL)
    return float(np.var(np.array([trials[k] for k in keys], dtype=np.float64)))


# ------------------------------------------------------------------ the pins themselves
def test_v1_5_pins_are_in_force():
    assert PINNED_SEEDS == (0, 1, 2, 3, 4)  # item D / C.1 Option 2: S=5 UP FRONT
    assert DSR_SEEDS == PINNED_SEEDS
    assert PINNED_DSR["n_trials"] == DSR_N_TRIALS == 60  # A.5: seeds EXCLUDED
    assert DSR_N_TRIALS == 5 * len(DSR_HORIZONS) * len(DSR_KAPPAS)
    assert PINNED_DSR["var_sr_basis_cell"] == DSR_VAR_SR_BASIS_CELL == 5  # A.4: the PLACEBO arm
    assert PINNED_DSR["placebo_dispersion_tripwire"] == PLACEBO_DISPERSION_TRIPWIRE == 1.5


def test_n_trials_is_independent_of_the_seed_count():
    """THE POINT of A.5: replicates must not move the multiplicity count. If N ever becomes a
    function of len(seeds) again, the misspecification is back."""
    assert DSR_N_TRIALS == 5 * len(DSR_HORIZONS) * len(DSR_KAPPAS)
    assert DSR_N_TRIALS != 5 * len(DSR_SEEDS) * len(DSR_HORIZONS) * len(DSR_KAPPAS)


# ------------------------------------------------------------------ MUTATION tests (the old values)
def test_gate_rejects_the_pre_v1_5_n_trials():
    """Restoring N=180 (or the seeds-as-trials 300) must FAIL the gate."""
    tr = _trials()
    for stale in (180, len(DSR_SEEDS) * 5 * len(DSR_HORIZONS) * len(DSR_KAPPAS)):
        fails = verdict_dsr_failures(
            n_trials=stale, threshold=0.95, trials=tr, var_sr=_var_placebo(tr)
        )
        assert any("n_trials" in f for f in fails), f"stale n_trials {stale} was NOT rejected"


def test_gate_rejects_the_pre_v1_5_all_arms_var_sr_basis():
    """The old all-arms var_sr is the basis that made clause 5 anti-correlated with its own
    hypothesis. Supplying it must FAIL, naming the placebo basis."""
    tr = _trials()
    stale_var = float(np.var(np.array([tr[k] for k in sorted(tr)], dtype=np.float64)))
    assert stale_var != _var_placebo(tr)  # the fixture really does distinguish them
    fails = verdict_dsr_failures(n_trials=DSR_N_TRIALS, threshold=0.95, trials=tr, var_sr=stale_var)
    assert any("var_sr" in f and "placebo" in f for f in fails)


def test_gate_accepts_the_v1_5_recipe():
    tr = _trials()
    assert (
        verdict_dsr_failures(
            n_trials=DSR_N_TRIALS, threshold=0.95, trials=tr, var_sr=_var_placebo(tr)
        )
        == []
    )


def test_gate_still_rejects_a_missing_seed_in_the_enumeration():
    """The ENUMERATION stays the full cross-product even though seeds left the COUNT — the audit
    trail must not silently shrink."""
    tr = _trials()
    tr.pop((3, DSR_SEEDS[-1], DSR_HORIZONS[0], DSR_KAPPAS[0]))
    fails = verdict_dsr_failures(
        n_trials=DSR_N_TRIALS, threshold=0.95, trials=tr, var_sr=_var_placebo(tr)
    )
    assert any("cross product" in f for f in fails)


# ------------------------------------------------------------------ item B: the MDE multiplier
@pytest.mark.parametrize(("nu", "want"), [(2.0, 3.980646), (4.0, 3.072812)])
def test_t_multiplier_at_the_pinned_seed_counts(nu, want):
    """S=3 -> nu=2 -> 3.9806; S=5 -> nu=4 -> 3.0728. The z multiplier is 2.4865, so B is a
    1.24-1.60x TIGHTENING — recorded pre-data as the direction of the change."""
    got = student_t_ppf(0.95, nu) + student_t_ppf(0.80, nu)
    assert abs(got - want) < 1e-5


def test_welch_satterthwaite_limits():
    """Training-dominant -> nu = S-1; scoring-dominant -> nu = B-1."""
    assert abs(welch_satterthwaite_df(1e-12, 9999.0, 1.0, 4.0) - 4.0) < 1e-6
    assert abs(welch_satterthwaite_df(1.0, 9999.0, 1e-12, 4.0) - 9999.0) < 1e-3
