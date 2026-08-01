"""§7 v1.6 C-7 — the thresholds that DECIDE the verdict, and whether anything guards them.

THE DEFECT THESE PIN. ``ECON_FLOOR_IR`` (§3 clause 4's materiality floor), ``BOOT_POWER`` (the
power point inside MDE_paired) and ``PLACEBO_DISPERSION_TRIPWIRE`` (§7 v1.5 A.4.3) appeared
NOWHERE in ``conformance.py``. Each is a bare module constant in the file that consumes it, so an
edit to any one of them would have moved a §3 clause with nothing objecting — the C-2 defect one
level up: not a gate that examines nothing, but a pinned surface that pins nothing.

Four ``PINNED_DSR`` keys were equally decorative — ``n_trials_factorization``, the ``var_sr``
prose, ``statistic`` and ``placebo_dispersion_tripwire`` were carried in the dict and read by no
code path. The last of those was worse than unread: it was a SECOND independent copy of the 1.5
that ``verdict.PLACEBO_DISPERSION_TRIPWIRE`` also holds, i.e. the same two-files-own-the-truth
shape that produced the C-6 seed/seq_len split-brain.

EVERY TEST HERE IS A MUTATION TEST. Asserting that a pin currently equals its own value is
circular and would pass against a gate that does nothing. Each case instead PERTURBS the live
constant and requires the gate to name it — the fixture-discrimination norm applied to a pin
registry.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from trikaal.eval import conformance as C
from trikaal.eval import paired_bootstrap as PB
from trikaal.eval import verdict as V


def _trials() -> dict[tuple[int, int, int, float], float]:
    """The full pinned cross-product, values distinct so a wrong var_sr basis is detectable."""
    rng = np.random.default_rng(0)
    return {
        (c, s, h, k): float(rng.normal(0.4, 0.2))
        for c in C.PINNED_DSR["cells"]
        for s in C.PINNED_DSR["seeds"]
        for h in C.PINNED_DSR["horizons"]
        for k in C.PINNED_DSR["kappas"]
    }


def _var_sr(trials: dict) -> float:
    basis = int(C.PINNED_DSR["var_sr_basis_cell"])
    keys = sorted(k for k in trials if k[0] == basis)
    return float(np.var(np.array([trials[k] for k in keys], dtype=np.float64)))


def _assemble(tmp_path) -> dict:
    """A real verdict manifest over a benign 25-artifact fixture, via the shipped emission path."""
    out = tmp_path / "art"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    common = rng.standard_normal(600) * 1e-3
    entries: dict[str, str] = {}
    for c in C.PINNED_DSR["cells"]:
        for sd in C.PINNED_DSR["seeds"]:
            name, sha = V.write_cell_eval_artifact(
                out,
                cell_id=c,
                seed=sd,
                grid={"h": V.PRIMARY_H, "start_ms": 0, "n_periods": 600},
                headline_series=common + rng.standard_normal(600) * 1e-3,
                kappa_star_by_h={h: 1.5 for h in V.DSR_HORIZONS},
                val_ir_by_kappa_by_h={
                    h: {k: float(rng.normal(0.4, 0.2)) for k in V.DSR_KAPPAS}
                    for h in V.DSR_HORIZONS
                },
                mu_diag={"frac_negative": 0.5, "activity_decisions": 0.5},
                meta={},
            )
            entries[name] = sha
    V.write_eval_index(out, entries)
    evals, shas = V.load_cell_evals(out)
    return V.assemble_verdict(evals, shas, tabled_mde_h15=3.518)


def _dsr_fails(**over) -> list[str]:
    t = _trials()
    kw = {
        "n_trials": C.PINNED_DSR["n_trials"],
        "threshold": C.PINNED_DSR["threshold"],
        "trials": t,
        "var_sr": _var_sr(t),
    }
    kw.update(over)
    return C.verdict_dsr_failures(**kw)


# --------------------------------------------------- the fixture must be clean before it judges
def test_the_unmutated_surface_passes():
    """FIXTURE DISCRIMINATION: if this failed, every mutation below would pass vacuously."""
    assert C.pinned_threshold_failures() == []
    assert _dsr_fails() == []


# --------------------------------------------------- 1. the three constants, one mutation each
@pytest.mark.parametrize(
    ("module", "name", "bad"),
    [
        (V, "ECON_FLOOR_IR", 0.25),  # halving the materiality floor — clause 4 gets easier
        (PB, "BOOT_POWER", 0.5),  # a weaker power point shrinks MDE_paired — clause 2 easier
        (V, "PLACEBO_DISPERSION_TRIPWIRE", 3.0),  # a laxer tripwire hides placebo degradation
    ],
    ids=["econ_floor", "boot_power", "placebo_tripwire"],
)
def test_moving_a_verdict_threshold_is_caught(module, name, bad):
    with mock.patch.object(module, name, bad):
        fails = C.pinned_threshold_failures()
    assert any(name in f for f in fails), f"moving {name} to {bad} was NOT caught: {fails}"


def test_the_money_run_gate_also_enforces_the_thresholds():
    """A run must not start under an edited floor and discover it only at the verdict."""
    with mock.patch.object(V, "ECON_FLOOR_IR", 0.25):
        assert any("ECON_FLOOR_IR" in f for f in C.pinned_threshold_failures())
    # and the DSR gate carries them too, so neither entry point is the only guard
    with mock.patch.object(PB, "BOOT_POWER", 0.5):
        assert any("BOOT_POWER" in f for f in _dsr_fails())


# --------------------------------------------------- 2. the duplicate-source-of-truth leg
def test_the_two_independent_tripwire_copies_must_agree():
    """The C-6 shape: two files each holding their own copy of one pinned value."""
    with mock.patch.dict(C.PINNED_DSR, {"placebo_dispersion_tripwire": 2.0}):
        fails = C.pinned_threshold_failures()
    assert any("placebo_dispersion_tripwire" in f for f in fails)
    assert any("two copies of one truth" in f for f in fails)


def test_the_var_sr_basis_cell_copies_must_agree():
    with mock.patch.object(V, "DSR_VAR_SR_BASIS_CELL", 2):
        fails = C.pinned_threshold_failures()
    assert any("DSR_VAR_SR_BASIS_CELL" in f for f in fails)


# --------------------------------------------------- 3. the formerly-decorative DSR keys
def test_reintroducing_seeds_into_the_multiplicity_count_is_caught():
    """§7 v1.5 A.5: replicating a configuration must not make the adjustment stricter."""
    with mock.patch.dict(
        C.PINNED_DSR, {"n_trials_factorization": ("cells", "seeds", "horizons", "kappas")}
    ):
        fails = _dsr_fails()
    assert any("contains 'seeds'" in f for f in fails), fails
    assert any("!= cells x seeds x horizons x kappas" in f for f in fails), fails


def test_the_factorization_drives_the_count_rather_than_describing_it():
    """Dropping an axis must change n_trials and fail — proof the key is READ, not prose."""
    with mock.patch.dict(C.PINNED_DSR, {"n_trials_factorization": ("cells", "horizons")}):
        fails = _dsr_fails()
    assert any("n_trials 60 != cells x horizons = 15" in f for f in fails), fails


def test_a_sample_variance_ddof_is_caught():
    """var_sr_ddof is READ by the re-derivation AND asserted to be the population value."""
    with mock.patch.dict(C.PINNED_DSR, {"var_sr_ddof": 1}):
        fails = _dsr_fails()
    assert any("var_sr_ddof 1 != 0" in f for f in fails), fails
    # and it must not merely move the expectation to match a drifted computation
    assert any("POPULATION variance" in f for f in fails), fails


def test_the_shipped_rule_string_is_built_from_the_pins(tmp_path):
    """``statistic`` is wired — asserted on the EMITTED manifest, not on the source text.

    A source-substring check is the wrong level here: it cannot tell a live rule string from a
    comment describing the one that was removed (it flagged this file's own explanation on the
    first run). What matters is what the ARTIFACT carries, so the verdict is actually assembled
    and its clause-5 rule inspected. The predecessor said "var_sr over the 180 de-annualized VAL
    IRs" — wrong on BOTH counts after v1.5 (N is 60; the basis is the cell-5 placebo subset, not
    all arms) — and would have shipped inside the verdict manifest as a false account of how its
    own var_sr was computed."""
    rule = _assemble(tmp_path)["clauses"]["5_dsr"]["rule"]
    assert C.PINNED_DSR["statistic"] in rule, f"statistic not sourced from the pin: {rule}"
    assert "180" not in rule, f"superseded N is back in the shipped rule string: {rule}"
    assert f"CELL-{C.PINNED_DSR['var_sr_basis_cell']}" in rule, rule
    assert f"N={C.PINNED_DSR['n_trials']}" in rule, rule
    assert "seeds excluded as replicates" in rule, rule


def test_the_shipped_rule_string_follows_the_pin_when_it_moves(tmp_path):
    """MUTATION: change the pin, the manifest text must change with it — proof it is READ."""
    with mock.patch.dict(C.PINNED_DSR, {"statistic": "SENTINEL-STATISTIC"}):
        rule = _assemble(tmp_path)["clauses"]["5_dsr"]["rule"]
    assert "SENTINEL-STATISTIC" in rule, f"statistic is decorative again: {rule}"


def _unread_pinned_dsr_keys(keys) -> list[str]:
    from pathlib import Path

    sources = "".join(Path(m.__file__).read_text() for m in (C, V))
    return [
        k
        for k in keys
        if f'PINNED_DSR["{k}"]' not in sources and f"PINNED_DSR['{k}']" not in sources
    ]


def test_the_unread_key_detector_can_actually_fail():
    """FIXTURE DISCRIMINATION, before the check below is trusted."""
    assert _unread_pinned_dsr_keys(["a_key_no_code_reads"]) == ["a_key_no_code_reads"]
    assert _unread_pinned_dsr_keys(["n_trials"]) == []


def test_no_pinned_dsr_key_is_unread():
    """The registry must stay free of decoration — a pin never read gives false assurance."""
    unread = _unread_pinned_dsr_keys(C.PINNED_DSR)
    assert not unread, f"decorative PINNED_DSR keys (never read by any code path): {unread}"
