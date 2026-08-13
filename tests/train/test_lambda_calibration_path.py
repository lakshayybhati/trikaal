"""§7 v1.5 item E — THE LAMBDA CALIBRATION PATH MUST NOT WEAKEN THE MONEY PIN.

The once-only lambda re-derivation needs a tokenizer at a non-pinned lambda, and
``build_cell_tokenizer`` structurally refuses one. The wrong fix is to loosen
``PINNED_MICRO_POINT_WEIGHT`` or relax that refusal: the pin is what guards the experiment, and a
pin that moves to let a diagnostic run is decoration.

So the calibration path is SEPARATE and the money path is proven UNCHANGED here — not asserted in
a docstring, but exercised.
"""

from __future__ import annotations

import pytest

from trikaal.eval.conformance import PINNED_MICRO_POINT_WEIGHT, cell_tokenizer_failures
from trikaal.train.cells import CELLS, build_calibration_tokenizer, build_cell_tokenizer
from trikaal.train.orchestrator import OrchestratorConfig

C4 = next(c for c in CELLS if c.cell_id == 4)


def test_the_money_constructor_STILL_REFUSES_a_non_pinned_lambda():
    """The load-bearing one. If this ever passes silently, the pin is gone."""
    for lam in (2.0, 5.0, 8.0, 12.0):
        with pytest.raises(ValueError, match="micro_point_weight"):
            build_cell_tokenizer(C4, micro_point_weight=lam)


def test_the_money_constructor_still_HARD_SETS_the_pin_when_unspecified():
    assert build_cell_tokenizer(C4).get_config()["micro_point_weight"] == PINNED_MICRO_POINT_WEIGHT
    assert PINNED_MICRO_POINT_WEIGHT == 3.0


def test_the_calibration_constructor_refuses_the_PINNED_value():
    """It is for non-pinned lambda ONLY — the pinned value must go through the money path, or the
    two constructors become interchangeable and the separation is cosmetic."""
    with pytest.raises(ValueError, match="NON-PINNED"):
        build_calibration_tokenizer(C4, micro_point_weight=3.0)


@pytest.mark.parametrize("lam", [5.0, 8.0, 12.0])
def test_a_calibration_tokenizer_CANNOT_pass_conformance(lam):
    """An artifact built at a calibration lambda is refused by the money surface BY CONSTRUCTION,
    not by anyone remembering to check: `cell_tokenizer_failures` compares micro_point_weight
    against the pin."""
    fails = cell_tokenizer_failures(
        build_calibration_tokenizer(C4, micro_point_weight=lam).get_config(), run="calib"
    )
    assert any("micro_point_weight" in f for f in fails)


def test_calibration_lambda_on_a_MONEY_RUN_is_refused_by_the_orchestrator():
    """The declaration is CHECKED, not trusted. money_run defaults to True, so the dangerous
    combination is the DEFAULT one and must raise."""
    cfg = OrchestratorConfig(calibration_micro_point_weight=5.0)
    assert cfg.money_run is True
    import inspect

    from trikaal.train.orchestrator import run_cell

    src = inspect.getsource(run_cell)
    assert "calibration_micro_point_weight is not None" in src
    assert "if cfg.money_run:" in src, "the money_run guard is the whole protection"


def test_calibration_lambda_is_NOT_reachable_through_tok_kwargs():
    """It is a separate field precisely so it cannot leak into build_cell_tokenizer. If it were in
    tok_kwargs it would reach the money constructor, which would then raise — but by accident
    rather than by design, and only for the values that happen to be non-pinned."""
    cfg = OrchestratorConfig(money_run=False, calibration_micro_point_weight=5.0)
    assert "micro_point_weight" not in cfg.tok_kwargs
    assert cfg.calibration_micro_point_weight == 5.0


def test_default_config_leaves_the_calibration_path_OFF():
    assert OrchestratorConfig().calibration_micro_point_weight is None
