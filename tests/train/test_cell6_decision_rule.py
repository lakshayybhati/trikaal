"""§7 v1.6.27 — THE CELL-6 DECISION RULE IS PRE-COMMITTED, IN CODE, AND CAN SAY NO.

A ~$0.50 Stage-1 measurement decides a $16–22 purchase. The rule is written **before the numbers
exist** so it is not a judgement made on a rented box with a meter running — and it is tested at
its boundaries so it cannot be a rule that only ever says yes.

Both gates are Stage-1 outputs: codebook utilization (≥ 0.95, or every cell-6 artifact is refused
by `_validate_artifact`) and realized bits-per-token (inside FSQ_BPT_BAND, or the bracket's lower
end sits on a different-capacity instrument).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from trikaal.constants import FSQ_BPT_BAND
from trikaal.eval.conformance import PINNED_CODEBOOK_MIN_UTILIZATION

REPO = Path(__file__).resolve().parents[2]


def _decide():
    spec = importlib.util.spec_from_file_location(
        "cell6_probe", REPO / "scripts/m6_cell6_stage1_probe.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_BOTH_gates_passing_wires_cell_6():
    """Discrimination first: a rule that never says yes proves nothing about the noes."""
    m = _decide()
    d = m.decide(0.99, 0.98, 20.0578)  # the realized FSQ bpt
    assert d["decision"] == m.WIRE
    assert d["reasons"] == []


@pytest.mark.parametrize(
    "uc,uf,bpt,expect",
    [
        (0.94, 0.99, 20.06, "utilization"),  # coarse collapses
        (0.99, 0.80, 20.06, "utilization"),  # fine collapses
        (0.99, 0.98, 19.49, "bits-per-token"),  # below the band
        (0.99, 0.98, 20.51, "bits-per-token"),  # above the band
        (0.10, 0.10, 12.00, "utilization"),  # both fail
    ],
)
def test_EITHER_gate_failing_DECLINES_the_purchase(uc, uf, bpt, expect):
    m = _decide()
    d = m.decide(uc, uf, bpt)
    assert d["decision"] == m.DECLINE
    assert any(expect in r for r in d["reasons"])


def test_the_boundaries_are_INCLUSIVE_and_read_the_live_pins():
    """The rule must move with the pins, not restate them — one copy of one truth."""
    m = _decide()
    lo, hi = FSQ_BPT_BAND
    u = PINNED_CODEBOOK_MIN_UTILIZATION
    assert m.decide(u, u, lo)["decision"] == m.WIRE  # exactly on both floors
    assert m.decide(u, u, hi)["decision"] == m.WIRE  # exactly on the ceiling
    assert m.decide(u - 1e-9, u, lo)["decision"] == m.DECLINE


def test_declining_is_a_PUBLISHABLE_outcome_not_a_failure():
    """The rule must carry what happens on a NO, or a NO becomes an invitation to re-litigate."""
    m = _decide()
    d = m.decide(0.5, 0.5, 12.0)
    assert "UNQUANTIFIED IN IR UNITS" in d["if_declined"]
    assert "costs nothing" in d["if_declined"]


def test_wiring_carries_the_INVERSION_pre_commitment():
    """ΔIR(4−6) ≤ ΔIR(4−5) is an empirical claim about two trained models, NOT an identity. If it
    inverts the interval is empty AND the larger number flatters us — so the commitment to report
    both ends whichever way they fall travels with the YES."""
    m = _decide()
    d = m.decide(0.99, 0.98, 20.0)
    assert "whichever way they fall" in d["if_wired"]
    assert "not an identity" in d["if_wired"]
    assert "OPPOSITE directions" in d["if_wired"]


# ---------------------------------------------------------------- stage2_never_entered
# THE FIELD COULD NEVER RETURN True. It was `final_loss["stage2"] is None`, but the key always
# exists and carries nan when steps_stage2 == 0, so the receipt asserted "Stage 2 WAS entered" on
# a run where it provably was not. These assert the predicate DISCRIMINATES — without the finite
# case, a predicate hardcoded to True would pass just as well.
def test_stage2_never_entered_discriminates():
    f = _decide().stage2_never_entered
    assert f({"final_loss": {"stage1": 0.38, "stage2": float("nan")}}) is True, "nan => never ran"
    assert f({"final_loss": {"stage1": 0.38, "stage2": None}}) is True, "None => never ran"
    assert f({"final_loss": {"stage1": 0.38}}) is True, "absent => never ran"
    assert f({}) is True, "no final_loss at all => never ran"
    # the case that makes the others meaningful: a REAL Stage-2 loss must report False
    assert f({"final_loss": {"stage1": 0.38, "stage2": 4.21}}) is False, "finite loss => it RAN"
    assert f({"final_loss": {"stage1": 0.38, "stage2": 0.0}}) is False, "0.0 is finite => it RAN"


def test_stage2_predicate_rejects_the_pre_fix_form():
    """The pre-fix predicate on the ACTUAL manifest shape, so the regression is pinned to reality
    rather than to a hand-made dict: nan is not None, so `is None` returned False."""
    real_shape = {"final_loss": {"stage1": 0.3853527307510376, "stage2": float("nan")}}
    assert (real_shape["final_loss"]["stage2"] is None) is False  # what the old code computed
    assert _decide().stage2_never_entered(real_shape) is True  # what is true
