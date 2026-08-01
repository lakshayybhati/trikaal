"""§7 v1.6 C-6 — the training side and the eval side each held their own copy of the truth.

THE DEFECT. ``OrchestratorConfig`` carried ``seeds = (0, 1, 2)`` and ``seq_len = 128`` against a
pinned ``PINNED_SEEDS = (0, 1, 2, 3, 4)`` and a money eval surface of 512. Two consequences, and
the second is larger than the finding as originally stated:

  1. the S=5 the operator approved and funded on 2026-07-31 never reached the training side;
  2. a TRAIN/EVAL REGIME MISMATCH — a model optimised over 128 bars of context, then scored on
     512. That is not a stale default, it is a different experiment.

It stayed invisible because every existing driver overrides both fields; the money driver (C-5)
does not exist yet, and would have inherited 128 and three seeds straight from the dataclass.

WHY AN ASSERTION AND NOT A CORRECTED DEFAULT. Right-looking literals fix today and guarantee
nothing about tomorrow — the defect existed precisely because two files each restated the same
truth. So the fields now REFERENCE the pins (no second copy remains to drift) AND
``__post_init__`` hard-fails on divergence, which also catches a caller passing bad values by
argument, which a default cannot.

WHY ``money_run`` IS AN EXPLICIT DECLARATION AND NOT AN INFERENCE. Deciding toy-ness from the
values would re-create the hole exactly: a 128-context money run would be classified as a toy and
skip the check. The default is TRUE so the DANGEROUS direction is the one that must be declared —
a toy shell announcing itself is safe; a money run silently inheriting toy values is the defect.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from trikaal.eval.conformance import (
    PINNED_BACKBONE_PARAMS,
    PINNED_MICRO_LEGIBILITY_MIN,
    PINNED_MONEY_SEQ_LEN,
    PINNED_SEEDS,
    ConformanceError,
    money_config,
)
from trikaal.train.orchestrator import OrchestratorConfig

REPO = Path(__file__).resolve().parents[2]
TOY = dict(
    money_run=False,
    seeds=(0,),
    seq_len=16,
    micro_legibility_min=None,
    expect_backbone_params=None,
    enforce_parity=False,
)


def test_the_bare_default_is_the_money_surface():
    """FIXTURE DISCRIMINATION: if the clean config failed, every mutation below would pass."""
    cfg = OrchestratorConfig()
    assert tuple(cfg.seeds) == tuple(PINNED_SEEDS)
    assert cfg.seq_len == PINNED_MONEY_SEQ_LEN
    assert cfg.micro_legibility_min == PINNED_MICRO_LEGIBILITY_MIN
    assert cfg.expect_backbone_params == PINNED_BACKBONE_PARAMS


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"seeds": (0, 1, 2)}, "seeds (0, 1, 2) != pinned"),  # the exact historical value
        ({"seq_len": 128}, "train/eval regime"),  # the exact historical value
        ({"seeds": (0, 1, 2, 3)}, "!= pinned"),  # near-miss, not just the old literal
        ({"micro_legibility_min": None}, "standing gate"),
        ({"micro_legibility_min": 0.5}, "!= pinned 0.9"),
        ({"expect_backbone_params": None}, "realized count"),
        ({"enforce_parity": False}, "G-parity"),
    ],
    ids=[
        "old_seeds",
        "old_seq_len",
        "near_miss_seeds",
        "gate_off",
        "gate_lowered",
        "params_off",
        "parity_off",
    ],
)
def test_a_money_run_hard_fails_on_divergence(kw, needle):
    with pytest.raises(ConformanceError, match="pinned money surface"):
        OrchestratorConfig(**kw)
    with pytest.raises(ConformanceError) as e:
        OrchestratorConfig(**kw)
    assert needle in str(e.value), f"{kw} failed for the wrong reason: {e.value}"


def test_a_declared_toy_shell_is_exempt():
    """Fail-closed must not become fail-on-everything, or the check above passes vacuously."""
    cfg = OrchestratorConfig(**TOY)
    assert cfg.money_run is False and cfg.seq_len == 16


def test_toyness_is_declared_not_inferred():
    """The hole this would re-create: a 128-context MONEY run classified as a toy and skipped."""
    with pytest.raises(ConformanceError):
        OrchestratorConfig(seq_len=128, seeds=(0, 1, 2))  # toy-LOOKING values, money_run default
    assert OrchestratorConfig(**{**TOY, "seq_len": 128}).seq_len == 128  # only the flag exempts


def test_the_two_sides_read_one_constant():
    """C-6's root cause: eval seq_len and training seq_len must not be independent copies."""
    assert money_config().seq_len == PINNED_MONEY_SEQ_LEN == OrchestratorConfig().seq_len


def test_no_orchestrator_default_restates_a_pinned_value():
    """A pinned value must be REFERENCED in the dataclass, never re-typed as a literal."""
    src = (REPO / "src/trikaal/train/orchestrator.py").read_text()
    tree = ast.parse(src)
    cls = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "OrchestratorConfig"
    )
    literals = {
        n.target.id: ast.unparse(n.value)
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and n.value is not None
    }
    for fieldname, pin in (
        ("seeds", "PINNED_SEEDS"),
        ("seq_len", "PINNED_MONEY_SEQ_LEN"),
        ("micro_legibility_min", "PINNED_MICRO_LEGIBILITY_MIN"),
        ("expect_backbone_params", "PINNED_BACKBONE_PARAMS"),
    ):
        assert literals.get(fieldname) == pin, (
            f"{fieldname} default is {literals.get(fieldname)!r}, not the pin {pin} — a second "
            "copy of a pinned value is exactly what produced C-6"
        )


def test_every_orchestrator_caller_declares_its_intent():
    """AST sweep: a driver must not construct the config without saying which surface it is on.

    Bare ``OrchestratorConfig()`` is permitted — it IS the money surface and is asserted as such.
    What must not exist is a construction that overrides pinned fields without declaring
    ``money_run``, because that is the shape that would silently train a toy and score it money.
    """
    offenders = []
    for path in sorted((REPO / "scripts").glob("*.py")) + sorted(
        (REPO / "tests").rglob("test_*.py")
    ):
        if path.name == Path(__file__).name:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name != "OrchestratorConfig" or not node.keywords:
                continue
            kws = {k.arg for k in node.keywords}
            if "money_run" not in kws and kws & {
                "seeds",
                "seq_len",
                "micro_legibility_min",
                "expect_backbone_params",
                "enforce_parity",
            }:
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, (
        f"OrchestratorConfig overrides a pinned field without declaring money_run at: {offenders}"
    )


def test_the_caller_sweep_can_actually_fail():
    """FIXTURE DISCRIMINATION for the sweep above — a checker that never fires is not a check."""

    def offends(src: str) -> bool:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "OrchestratorConfig"
            ):
                kws = {k.arg for k in node.keywords}
                if "money_run" not in kws and kws & {"seeds", "seq_len"}:
                    return True
        return False

    assert offends("OrchestratorConfig(seeds=(0,), seq_len=16)"), "sweep missed a real offender"
    assert not offends("OrchestratorConfig(money_run=False, seeds=(0,), seq_len=16)")
    assert not offends("OrchestratorConfig()"), "the money default must not be flagged"
