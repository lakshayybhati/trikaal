"""G-§8.C.3 — the gate must HALT while unresolved, and must be capable of passing.

Audit C-4: this gate was named binding in three documents and had no implementation. The tests
below are written so that the CURRENT state (no published number obtained) is a HALT, and so that
a future "we filled the number in" cannot silently become a pass without the number being real.
"""

from __future__ import annotations

import pytest

from trikaal.eval.external_validation import (
    BLOCKED,
    FAIL,
    GATE_RATIO,
    PASS,
    ExternalReference,
    ExternalValidationBlocked,
    assert_clear_to_compute_deltas,
    evaluate,
    gate_threshold,
)


def test_the_gate_is_BLOCKED_today_and_that_is_the_honest_state():
    """No published Kronos-small RankIC has been obtained. The gate must say so and HALT."""
    v = evaluate(cell1_rankic=0.05, ref=ExternalReference())
    assert v["status"] == BLOCKED
    assert gate_threshold(ExternalReference()) is None
    with pytest.raises(ExternalValidationBlocked, match="BLOCKED"):
        assert_clear_to_compute_deltas(v)


def test_an_unmeasured_cell1_is_BLOCKED_not_passed():
    ref = ExternalReference(published_rankic=0.04, source="paper table N", slice_id="btc-2024")
    v = evaluate(cell1_rankic=None, ref=ref)
    assert v["status"] == BLOCKED
    with pytest.raises(ExternalValidationBlocked):
        assert_clear_to_compute_deltas(v)


def test_a_strong_cell1_PASSES_and_clears_the_delta_path():
    """DISCRIMINATION: if the gate could never pass, every HALT above would prove nothing."""
    ref = ExternalReference(published_rankic=0.04, source="paper table N", slice_id="btc-2024")
    v = evaluate(cell1_rankic=0.04, ref=ref)
    assert v["status"] == PASS and v["protocol"] is None
    assert_clear_to_compute_deltas(v)  # must NOT raise


@pytest.mark.parametrize("frac,expect", [(0.849, FAIL), (0.851, PASS), (1.0, PASS), (0.5, FAIL)])
def test_the_0p85_boundary_is_exactly_where_the_prereg_puts_it(frac, expect):
    ref = ExternalReference(published_rankic=0.04, source="s", slice_id="x")
    assert evaluate(0.04 * frac, ref)["status"] == expect
    assert gate_threshold(ref) == pytest.approx(GATE_RATIO * 0.04)


def test_a_failing_gate_names_the_precommitted_protocol_rather_than_describing_it():
    ref = ExternalReference(published_rankic=0.04, source="s", slice_id="x")
    v = evaluate(0.01, ref)
    assert v["status"] == FAIL
    assert "HALT_BEFORE_ANY_DELTA" in v["protocol"]
    assert "CELL1_ONLY_FIX" in v["protocol"]
    assert "FULL_5_CELL_SAME_SEED_RETRAIN" in v["protocol"]
    with pytest.raises(ExternalValidationBlocked, match="No between-cell"):
        assert_clear_to_compute_deltas(v)


def test_blocked_names_the_invariant_8_question_rather_than_hiding_it():
    """The gate cannot be run without Kronos model code; invariant 8 forbids Kronos code. The
    gate must surface that as the blocking question, not report a vague 'not configured'."""
    v = evaluate(0.05, ExternalReference())
    assert "invariant 8" in v["blocking_question"]
    assert "Lakshay" in v["blocking_question"]


# ---------------------------------------------------- the WIRING, not just the gate function
def _fixture_evals(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.test_verdict import MU_PLANTED, _build_fixture
    from trikaal.eval.verdict import load_cell_evals

    return load_cell_evals(_build_fixture(tmp_path, MU_PLANTED, fixture_seed=101))


def test_a_MONEY_verdict_HALTS_while_the_gate_is_blocked(tmp_path):
    """C-4's whole point: the gate must fire BEFORE any Δ exists. `money_verdict` defaults TRUE,
    so a caller that says nothing gets the safe behaviour (the C-6 rule)."""
    from trikaal.eval.verdict import assemble_verdict

    evals, shas = _fixture_evals(tmp_path)
    with pytest.raises(ExternalValidationBlocked, match=r"G-§8\.C\.3"):
        assemble_verdict(evals, shas, tabled_mde_h15=3.518)


def test_a_declared_fixture_still_assembles(tmp_path):
    """DISCRIMINATION: if nothing could assemble, the HALT above would prove nothing."""
    from trikaal.eval.verdict import assemble_verdict

    evals, shas = _fixture_evals(tmp_path)
    m = assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)
    assert m["verdict"]["primary"] in {"SURVIVES", "NULL", "INCONCLUSIVE", "HALT_ADJUDICATE"}


def test_a_PASSED_gate_clears_a_money_verdict(tmp_path):
    from trikaal.eval.verdict import assemble_verdict

    evals, shas = _fixture_evals(tmp_path)
    ok = evaluate(0.04, ExternalReference(published_rankic=0.04, source="s", slice_id="x"))
    assert ok["status"] == PASS
    m = assemble_verdict(evals, shas, tabled_mde_h15=3.518, external_validation=ok)
    assert m["verdict"]["primary"] in {"SURVIVES", "NULL", "INCONCLUSIVE", "HALT_ADJUDICATE"}
