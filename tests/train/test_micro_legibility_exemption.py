"""§7 v1.6.27 — THE CONSTANT-MICRO ARM'S GATE EXEMPTION IS EXPLICIT, NAMED, AND CANNOT BE A SKIP.

The C-12 bracket needs a Cell 6 whose micro channels are CONSTANT. The micro-legibility gate
measures recovery of ``sign(x_t[dim])`` — above-vs-below the causal rolling mean — and **a constant
channel has no sign**, so on that arm the statistic is UNDEFINED, not merely low.

**THE DANGER IS NOT THE EXEMPTION, IT IS HOW THE EXEMPTION ARRIVES.** The gate fired on
``spec.arm in ("micro", "micro_shuffled")`` — an INCLUSION TEST. A newly added arm falls through it
and the gate silently does not run, with no field anywhere saying so. That is the C-10 shape at
cell scope rather than dim scope: *"skipped is a third state"*, and after C-10 a silent skip is the
one thing that cannot ship.

So: membership is a REGISTRY keyed on the ARM (never on "this dim looked degenerate"), an
unregistered arm is a hard error, and an exemption writes a NAMED reason into the run manifest.
"""

from __future__ import annotations

import pytest

from trikaal.train.arms import (
    ARM_MICRO,
    ARM_MICRO_CONSTANT,
    ARM_MICRO_SHUFFLED,
    ARM_OHLCV,
    ARMS,
    MICRO_LEGIBILITY_APPLIES,
    MICRO_LEGIBILITY_EXEMPTION_REASON,
    arm_n_features,
    micro_legibility_applies,
)


def test_the_gate_APPLIES_to_the_two_informative_micro_arms():
    """Discrimination first: if the gate applied to nothing, every exemption below is vacuous."""
    assert micro_legibility_applies(ARM_MICRO) is True
    assert micro_legibility_applies(ARM_MICRO_SHUFFLED) is True


def test_the_constant_arm_is_EXEMPT_and_says_why():
    assert micro_legibility_applies(ARM_MICRO_CONSTANT) is False
    why = MICRO_LEGIBILITY_EXEMPTION_REASON[ARM_MICRO_CONSTANT]
    assert "no sign" in why.lower() or "HAS NO SIGN" in why
    assert "ARM" in why, "the exemption must be keyed on the arm, not on a degenerate-looking dim"


def test_EVERY_arm_is_classified_and_an_unknown_arm_is_a_HARD_ERROR():
    """The whole point. An inclusion test lets a new arm fall through in silence; a registry
    lookup makes the author decide."""
    for arm in ARMS:
        assert arm in MICRO_LEGIBILITY_APPLIES, f"{arm} is unclassified"
    assert set(MICRO_LEGIBILITY_APPLIES) == set(ARMS)
    with pytest.raises(ValueError, match="not in MICRO_LEGIBILITY_APPLIES"):
        micro_legibility_applies("micro_something_new")


def test_every_exempt_arm_carries_a_reason():
    for arm, applies in MICRO_LEGIBILITY_APPLIES.items():
        if not applies:
            assert MICRO_LEGIBILITY_EXEMPTION_REASON.get(arm, "").strip(), (
                f"{arm} is exempt with no recorded reason — an absent reason is not a considered one"
            )


def test_the_constant_arm_is_SIXTEEN_dims_like_cells_4_and_5():
    """It must be present-but-uninformative, NOT removed. A 7-dim arm would be cell 2 again, and
    the bracket needs the same input width to isolate what the micro SLOTS carry."""
    assert arm_n_features(ARM_MICRO_CONSTANT) == arm_n_features(ARM_MICRO) == 16
    assert arm_n_features(ARM_OHLCV) == 7


def test_constant_micro_zeroes_only_the_micro_slots_and_leaves_OHLCV_byte_identical():
    """The bracket's like-for-like premise, same as the placebo's: both arms must reconstruct the
    SAME OHLCV targets, or the comparison measures the OHLCV channels too."""
    import numpy as np

    from trikaal.train.arms import constant_micro

    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 16)).astype(np.float32)
    m = np.zeros((64, 16), dtype=np.uint8)
    m[:, 13:16] = 1  # funding/OI masked everywhere — the tripwire's precondition
    out, out_m = constant_micro(x, m)

    np.testing.assert_array_equal(out[:, :7], x[:, :7])  # OHLCV byte-identical
    assert np.all(out[:, 7:13] == 0.0)  # micro slots constant
    np.testing.assert_array_equal(out[:, 13:16], x[:, 13:16])  # perp dims untouched
    np.testing.assert_array_equal(out_m, m)
    assert not np.shares_memory(out, x), "must not mutate the caller's array"


def test_the_perp_tripwire_guards_DISJOINT_dims_from_the_ones_constant_micro_writes():
    """Checked, not assumed. `assert_perp_dims_masked` guards 13-15; the transform writes 7-12.
    If they overlapped, the transform could mask the very dims the tripwire asserts about."""
    from trikaal.eval.placebo import PERP_DIMS
    from trikaal.train.gates import MICRO_DIMS

    assert PERP_DIMS == (13, 14, 15)
    assert MICRO_DIMS == (7, 8, 9, 10, 11, 12)
    assert not set(PERP_DIMS) & set(MICRO_DIMS)


def test_the_tripwire_still_FIRES_on_an_activated_perp_dim_through_this_transform():
    """Discrimination on the tripwire itself: it must be reachable from the new transform, or
    'the tripwire covers this' is an assertion about code that never runs here."""
    import numpy as np

    from trikaal.train.arms import constant_micro

    m = np.zeros((8, 16), dtype=np.uint8)
    m[:, 13:16] = 1
    m[3, 14] = 0  # funding/OI ACTIVE on one bar
    with pytest.raises(AssertionError, match="funding/OI"):
        constant_micro(np.zeros((8, 16), dtype=np.float32), m)


# ------------------------------------------------------------------ cell 6 stays OUT of the verdict
def test_cell_6_is_NOT_in_the_verdict_path_registry():
    """§7 v1.6.27 — a CONTROL is not a configuration under test. `verdict.n_trials_v12` is computed
    from `len(_CELL_BY_ID)`, so adding cell 6 to `CELLS` would move the v1.2 dual-specification
    leg's multiplicity from 300 to 360 — a REPORTED statistic changing because a control was
    added. The conformance gate asserts this; here it is asserted directly."""
    from trikaal.eval.conformance import PINNED_DSR
    from trikaal.eval.verdict import _CELL_BY_ID

    assert tuple(PINNED_DSR["cells"]) == (1, 2, 3, 4, 5)
    assert tuple(sorted(_CELL_BY_ID)) == (1, 2, 3, 4, 5)


def test_the_conformance_gate_CATCHES_a_sixth_cell_in_the_live_registry(monkeypatch):
    """The mutation, in-process: put a 6th cell in the registry the verdict path reads and the
    gate must refuse. Without this the assertion above is a statement about today, not a gate."""
    import trikaal.eval.verdict as V
    from trikaal.eval.conformance import pinned_threshold_failures

    assert pinned_threshold_failures() == []  # DISCRIMINATION FIRST
    monkeypatch.setitem(V._CELL_BY_ID, 6, object())
    fails = pinned_threshold_failures()
    assert fails and any("n_trials_v12" in f or "_CELL_BY_ID" in f for f in fails)
