"""THE PAPER IS OUTSIDE EVERY GUARD THIS PROJECT BUILT, AND A PLAIN GREP CANNOT SEE IT EITHER.

Two independent failures compound here.

STRUCTURAL: every guard walks ``git ls-files``, and ``paper/`` is ignored at ``.gitignore:52``. The
claim-drift sweep, ``claim_site``, the model-card guard and the receipt sweeps are therefore
incapable of reading the project's most important document. Tiers 1, 1b, 2, 3 and 4 all corrected
``docs/MODEL_CARD.md`` and ``README.md``; none of it could reach the paper.

LEXICAL, AND WORSE BECAUSE IT IS SILENT: the manuscript writes ``304{,}625{,}181`` and ``$97.3\\%$``.
MEASURED on the restored paper — a raw search for ``97.3% of the shortfall`` finds **0**, and the
same search after ``latex_plain`` finds **3**. A raw search for ``304,625,181`` finds 2 (both in
comments); normalized it finds 8. A guard pointed at the paper WITHOUT normalization returns a
confident clean result on a document that states exactly those things. Same shape as the
``git grep``/``\\b`` defect: a pattern that cannot match looks like a document that does not say it.

WHAT THIS FILE IS, AND WHAT IT IS NOT. It proves the TOOL works — that ``latex_plain`` reveals what
a raw read hides and that ``scripts/m6_paper_claim_audit.py`` fires on each planted defect. It does
NOT assert the paper is clean, because the paper is NOT clean and fixing it is the writer's job.
The audit is a script that exits non-zero; the suite stays green so CI keeps meaning something.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from tests.claim_site import latex_plain, paper_dir, paper_docs

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "scripts/m6_paper_claim_audit.py"


# ── the normalizer, which is the load-bearing part ────────────────────────────────────────────
@pytest.mark.parametrize(
    ("src", "must_contain"),
    [
        (r"The realized ingest is \textbf{304{,}625{,}181} bars", "304,625,181"),
        (r"the headline metric is scored on $40$;", "on 40;"),
        (r"$97.3\%$ of the shortfall", "97.3% of the shortfall"),
        (r"9 & \texttt{trade\_count} & 0.8975", "0.8975"),
        (r"$7{,}024$ Parquet files", "7,024"),
        (r"\emph{selectively}: $97.3\%$", "selectively"),
    ],
)
def test_latex_plain_reveals_what_a_raw_read_hides(src, must_contain) -> None:
    assert must_contain in latex_plain(src), latex_plain(src)


def test_NEGATIVE_CONTROL_a_raw_read_really_does_miss_them() -> None:
    """Fixture discrimination: if the raw form already matched, the normalizer would be pointless
    and these tests would prove nothing."""
    src = r"is \textbf{304{,}625{,}181} bars and $97.3\%$ of the shortfall"
    assert "304,625,181" not in src and "97.3% of the shortfall" not in src
    plain = latex_plain(src)
    assert "304,625,181" in plain and "97.3% of the shortfall" in plain


def test_comments_are_removed_before_anything_else() -> None:
    """A ``%`` line is not a claim a reader reads, and once whitespace collapses it becomes
    indistinguishable from the prose around it."""
    assert latex_plain("% artifact: 304{,}625{,}181 bars\n").strip() == ""
    assert "100" in latex_plain(r"we report 100\% coverage")  # \% is not a comment


# ── the auditor has teeth on planted defects ──────────────────────────────────────────────────
def _run_audit(tmp: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(AUDIT), "--paper-dir", str(tmp)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def _clean_paper(tmp: Path) -> Path:
    """A minimal manuscript that satisfies every rule — the fixture must pass before it can
    discriminate."""
    d = tmp / "paper"
    (d / "sections").mkdir(parents=True)
    (d / "main.tex").write_text(
        "The gate fired. $97.3\\%$ of cell 4's shortfall falls on the two signed channels. "
        "The plant injects $\\tfrac12\\ln(1+c^2) = 1.151$ nats at $c=3$. "
        "The draw is $40$ symbols and the split is at 2023-10-20. "
        "Stage 2 was never entered for cells 2--5.\n"
    )
    (d / "sections" / "s.tex").write_text("Nothing to see here.\n")
    return d


def test_a_clean_manuscript_passes(tmp_path) -> None:
    r = _run_audit(_clean_paper(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize(
    ("rule", "inject"),
    [
        ("97.3-unscoped", "Separately, $97.3\\%$ of the shortfall sits on the signed channels."),
        (
            "trained-on-the-lake",
            "The other 160 instruments are lake breadth and training data, not evaluation breadth.",
        ),
        (
            "magnitude-range-0.8975",
            "The magnitude dims span 0.8975--0.9320 across the four channels.",
        ),
        ("canary-1.151-underived", "Left: 1.151 nats planted in feature space, never recovered."),
        ("cells-never-trained", "Cells 2--5 were never trained at any stage."),
        (
            "replicate-spread-unqualified",
            "The 18 stored replicates span 77 to 93 percent of the shortfall.",
        ),
    ],
)
def test_each_planted_defect_is_caught(tmp_path, rule, inject) -> None:
    d = _clean_paper(tmp_path)
    (d / "sections" / "s.tex").write_text(inject + "\n")
    r = _run_audit(d)
    assert r.returncode == 1, f"{rule} not caught\n{r.stdout}{r.stderr}"
    assert rule in r.stderr, f"caught something, but not {rule}:\n{r.stderr}"


@pytest.mark.parametrize("missing", ["40", "2023-10-20"])
def test_a_figure_the_reader_needs_but_the_paper_omits_is_reported(tmp_path, missing) -> None:
    """Absence is a defect too: the paper states neither the 40-symbol draw nor 84,153,600 bars."""
    d = _clean_paper(tmp_path)
    (d / "main.tex").write_text((d / "main.tex").read_text().replace(missing, "XX"))
    r = _run_audit(d)
    assert r.returncode == 1 and "ABSENT" in r.stdout, r.stdout + r.stderr


def test_the_audit_REFUSES_when_there_is_no_manuscript(tmp_path) -> None:
    """A sweep with no subject has proven nothing — it must not exit 0."""
    r = _run_audit(tmp_path / "nothing-here")
    assert r.returncode == 2 and "REFUSAL" in r.stderr, r.stderr


# ── the document set is a parameter, and the default is unchanged ─────────────────────────────
def test_the_paper_location_is_a_parameter_not_a_path(monkeypatch, tmp_path) -> None:
    d = _clean_paper(tmp_path)
    monkeypatch.setenv("TRIKAAL_PAPER_DIR", str(d))
    assert paper_dir() == d
    assert len(paper_docs()) == 2


def test_no_manuscript_is_not_an_error_for_the_library(monkeypatch, tmp_path) -> None:
    """A public clone has no paper. ``paper_dir`` returns None rather than raising, so guards that
    consult it degrade to today's behaviour instead of breaking."""
    monkeypatch.setenv("TRIKAAL_PAPER_DIR", str(tmp_path / "absent"))
    assert paper_dir() is None
    assert paper_docs() == []


# ── what the audit found on the real manuscript, when it is present ───────────────────────────
def test_the_real_manuscript_if_present_is_reported_not_asserted_clean() -> None:
    """★ DELIBERATELY NOT AN ASSERTION THAT THE PAPER IS CLEAN. It is not clean; the corrections
    are the writer's to make. What is asserted is that IF the manuscript is here, the audit
    reaches it and returns a decision rather than silence."""
    if paper_dir() is None:
        pytest.skip("no manuscript present (a public clone) — nothing to reach")
    r = subprocess.run([sys.executable, str(AUDIT)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr
    assert re.search(r"manuscript: .*\(\d+ \.tex files\)", r.stdout), r.stdout
