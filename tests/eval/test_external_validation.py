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


def test_the_gate_is_DROPPED_and_carries_its_disclosures():
    """§7 v1.6.22 — Lakshay DROPPED it as binding. It must still be visible, must carry the
    reasons and the required disclosure, and must no longer halt."""
    from trikaal.eval.external_validation import BSQ_DISCLOSURE, DROPPED, GATE_IS_BINDING

    assert GATE_IS_BINDING is False
    v = evaluate(cell1_rankic=None, ref=ExternalReference())
    assert v["status"] == DROPPED
    assert len(v["why"]) == 3 and any("0.0254" in w for w in v["why"])
    assert v["required_disclosure"] == BSQ_DISCLOSURE
    assert "IR(4)-IR(2) is clean" in v["also_withdrawn"]
    assert_clear_to_compute_deltas(v)  # must NOT raise — the gate is dropped


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


def test_the_binding_path_still_exists_and_still_halts(monkeypatch):
    """DISCRIMINATION: if the gate were re-armed it must HALT again. A dropped gate whose
    binding path had rotted would be undetectable."""
    import trikaal.eval.external_validation as X

    monkeypatch.setattr(X, "GATE_IS_BINDING", True)
    v = X.evaluate(0.05, ExternalReference())
    assert v["status"] == BLOCKED
    assert "invariant 8" in v["blocking_question"]
    with pytest.raises(ExternalValidationBlocked):
        X.assert_clear_to_compute_deltas(v)


# ---------------------------------------------------- the WIRING, not just the gate function
def _fixture_evals(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.test_verdict import MU_PLANTED, _build_fixture
    from trikaal.eval.verdict import load_cell_evals

    return load_cell_evals(_build_fixture(tmp_path, MU_PLANTED, fixture_seed=101))


def test_a_MONEY_verdict_now_CLEARS_because_the_gate_was_dropped(tmp_path):
    """§7 v1.6.22: with the gate dropped a money verdict assembles. The wiring stays in place so
    re-arming it is a one-constant change, and the test above proves the halt path still works."""
    from trikaal.eval.verdict import assemble_verdict

    evals, shas = _fixture_evals(tmp_path)
    m = assemble_verdict(evals, shas, tabled_mde_h15=3.518)
    assert m["verdict"]["primary"] in {"SURVIVES", "NULL", "INCONCLUSIVE", "HALT_ADJUDICATE"}


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
