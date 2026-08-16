"""``display_path`` + the regression it exists to prevent: a success message that crashes.

The failure mode is specific and worth naming in a test file, because the shape recurs: THE WORK
SUCCEEDS AND THE REPORTING OF IT FAILS. A file is written, then the run dies formatting the line
that announces it, and the operator reads a traceback as "the tool is broken".
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from trikaal.utils.paths import display_path

REPO = Path(__file__).resolve().parents[2]


def test_inside_root_is_relative():
    assert display_path(REPO / "scripts" / "x.py", REPO) == "scripts/x.py"


def test_outside_root_returns_absolute_and_DOES_NOT_RAISE(tmp_path):
    outside = tmp_path / "somewhere" / "out.html"
    got = display_path(outside, REPO)
    assert got == str(outside)
    assert Path(got).is_absolute()


def test_bare_relative_to_would_have_raised_on_that_input(tmp_path):
    """FIXTURE DISCRIMINATION: prove the input actually exercises the failure being guarded.

    Without this, the test above would pass just as happily against a fixture that never posed
    the problem — an assertion that cannot distinguish a fixed helper from an unnecessary one.
    """
    outside = tmp_path / "somewhere" / "out.html"
    with pytest.raises(ValueError):
        outside.relative_to(REPO)


def test_scripts_do_not_print_through_bare_relative_to():
    """THE CLASS, SWEPT — no script may format user-facing output with a bare ``relative_to``.

    Read as AST, not text, so this file and the docstrings above (which necessarily discuss
    ``relative_to``) cannot trip it. Two scripts had this line simultaneously; a grep-once fix
    would have left the second standing and closed the finding.
    """
    offenders = []
    for path in sorted((REPO / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "relative_to":
                continue
            # only flag calls whose result flows into a printed f-string / print() argument
            parent_is_print = False
            for outer in ast.walk(tree):
                if isinstance(outer, ast.Call) and getattr(outer.func, "id", None) == "print":
                    for sub in ast.walk(outer):
                        if sub is node:
                            parent_is_print = True
            if parent_is_print:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "these print a path through a bare relative_to, which RAISES when the path is outside "
        f"the repo — use trikaal.utils.paths.display_path: {offenders}"
    )


@pytest.mark.slow
def test_demo_build_exits_zero_with_out_outside_the_repo(tmp_path):
    """THE REPORTED BUG, END TO END. --out outside the repo must exit 0, not traceback.

    Skipped when the acceptance receipt is absent or the lake is not present locally, because the
    builder legitimately REFUSES in those cases and this test is about the reporting path only.
    """
    acc = REPO / "runs_manifest" / "m6_demo_acceptance.json"
    if not acc.exists() or not (REPO / "processed" / "universe_bars").exists():
        pytest.skip("needs a PASSING acceptance receipt and the local universe lake")
    out = tmp_path / "nested" / "demo.html"
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "m6_demo_build.py"),
            "--symbols",
            "BTCUSDT",
            "--per-symbol",
            "1",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=3600,
    )
    assert r.returncode == 0, f"exit {r.returncode}; stderr tail:\n{r.stderr[-800:]}"
    assert out.exists() and out.stat().st_size > 0
    assert str(out) in r.stdout, "the success line must name the absolute path it actually wrote"
