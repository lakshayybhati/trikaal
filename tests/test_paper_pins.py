"""Appendix D says its pins are imported live. They are typed by hand. This makes the claim true.

THE FINDING. ``paper/sections/appendix_repro.tex`` opens with a comment stating that every value in
the pinned-decision-surface table is "imported live from src/trikaal/eval/conformance.py and
src/trikaal/eval/verdict.py -- none is transcribed", and the section text repeats it: "The values
are reproduced from the constants themselves, not from prose." No generator exists anywhere in the
repository. They are transcribed, and they happen to be correct.

WHY A TEST AND NOT A GENERATOR. A generator would make the claim literally true for one build and
would still let the table drift the moment someone edits the LaTeX by hand — and `paper/` belongs
to the writer, so emitting into it is not mine to do. What actually matters to a referee is not
how the number arrived on the page but whether it MATCHES, and that is a property this file can
assert from outside the boundary: it reads `paper/`, writes nothing there, and fails when a pin and
its printed value disagree. The claim becomes operationally true — a drifted pin fails CI — and the
wording of the header is the writer's to soften or keep.

FIXTURE DISCRIMINATION. Parsing LaTeX is exactly the kind of check that quietly matches nothing and
passes: `test_the_table_was_actually_parsed` requires every expected pin to have been FOUND, and
`test_a_drifted_value_is_caught` runs the comparator against a deliberately corrupted table and
requires it to fail. Without those two, a regex that stopped matching would read as a clean bill.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from trikaal.eval import conformance as C

REPO = Path(__file__).resolve().parents[1]
APPENDIX = REPO / "paper" / "sections" / "appendix_repro.tex"

# constant name -> how its live value is rendered in the table
EXPECTED: dict[str, str] = {
    "PINNED_SEEDS": "tuple",
    "PINNED_KAPPAS": "tuple",
    "PINNED_HEADLINE_COST": "float",
    "PINNED_MONEY_SEQ_LEN": "int",
    "PINNED_BACKBONE_PARAMS": "int",
    "PINNED_STEPS_STAGE1": "int",
    "PINNED_STEPS_STAGE2": "int",
    "PINNED_MU_ESTIMATOR": "str",
    "PINNED_ENCODER_CAUSAL": "bool",
    "PINNED_FINE_POINTWISE": "bool",
    "PINNED_MICRO_POINT_WEIGHT": "float",
    "PINNED_MICRO_LEGIBILITY_MIN": "float",
    "PINNED_CODEBOOK_MIN_UTILIZATION": "float",
}

_ROW = re.compile(r"\\texttt\{(PINNED(?:\\_[A-Z0-9]+)+)\}\s*&\s*(.+?)\s*&", re.S)


def _tex_scalar(cell: str) -> str:
    """Strip the LaTeX a value is dressed in, leaving the value."""
    s = cell.strip()
    s = re.sub(r"\\texttt\{(.*?)\}", r"\1", s)
    s = s.replace("$", "").replace("{,}", "").replace("\\,", "").replace("\\ldots", "…")
    s = s.replace("\\emph{", "").replace("}", "").replace("{", "")
    return s.strip()


def _parsed_rows() -> dict[str, str]:
    text = APPENDIX.read_text()
    out: dict[str, str] = {}
    for name, cell in _ROW.findall(text):
        out.setdefault(name.replace("\\_", "_"), _tex_scalar(cell))
    return out


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _matches(printed: str, live) -> bool:
    """Compare by VALUE, not by rendering.

    The first version compared strings and failed on ``PINNED_KAPPAS``: the table prints
    ``(1.0, 1.5, 2.0, 3.0)`` and ``f"{1.0:g}"`` is ``1``. The pin was right and the comparator was
    wrong — which is the way round that matters, because a comparator that fails on correct input
    gets loosened until it stops failing on incorrect input too.
    """
    p = printed.strip()
    if isinstance(live, bool):
        return p.lower() == str(live).lower()
    if isinstance(live, tuple):
        got = [float(x) for x in _NUM.findall(p)]
        return len(got) == len(live) and all(a == float(b) for a, b in zip(got, live, strict=True))
    if isinstance(live, (int, float)):
        got = [float(x) for x in _NUM.findall(p)]
        return len(got) == 1 and got[0] == float(live)
    return p.replace(" ", "") == str(live).replace(" ", "")


def test_the_table_was_actually_parsed() -> None:
    """A regex that matches nothing passes every comparison below. It must match everything."""
    rows = _parsed_rows()
    missing = sorted(set(EXPECTED) - set(rows))
    assert not missing, f"Appendix D rows not found by the parser: {missing}"
    assert len(rows) >= len(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_printed_pin_matches_the_live_constant(name: str) -> None:
    printed = _parsed_rows()[name]
    live = getattr(C, name)
    assert _matches(printed, live), (
        f"Appendix D prints {name} = {printed!r}; the live constant is {live!r}. The appendix "
        f"claims these values are imported live and none is transcribed — they ARE transcribed, "
        f"so this is the only thing keeping the claim true."
    )


def test_the_symbols_hash_prefix_matches() -> None:
    """Printed as an elided prefix, so it is compared as one rather than skipped."""
    printed = _parsed_rows()["PINNED_SYMBOLS_SHA256"]
    head = printed.split("…")[0].strip()
    assert len(head) >= 8, f"prefix too short to be evidence: {printed!r}"
    assert C.PINNED_SYMBOLS_SHA256.startswith(head), (
        f"Appendix D prints {printed!r}; live hash starts {C.PINNED_SYMBOLS_SHA256[:16]}"
    )


def test_a_drifted_value_is_caught(tmp_path) -> None:
    """The comparator must fail on a corrupted table, or the passes above mean nothing."""
    text = APPENDIX.read_text().replace(f"${C.PINNED_MICRO_LEGIBILITY_MIN:.2f}$", "$0.42$", 1)
    assert "$0.42$" in text, "the mutation did not land — the fixture cannot discriminate"
    doctored = tmp_path / "appendix_repro.tex"
    doctored.write_text(text)
    rows = {}
    for name, cell in _ROW.findall(doctored.read_text()):
        rows.setdefault(name.replace("\\_", "_"), _tex_scalar(cell))
    assert not _matches(rows["PINNED_MICRO_LEGIBILITY_MIN"], C.PINNED_MICRO_LEGIBILITY_MIN), (
        "a drifted pin was NOT caught — this whole file is decorative"
    )
