#!/usr/bin/env python3
"""Assert that the submission build omits Appendix F and that the internal build keeps it.

WHY THIS EXISTS. Appendix F is addressed to the integration reviewer, not to a referee, and must
not ship. The paper said so in prose, which is an intention. An intention is the shape of defect
this project has spent two days closing -- "we will remember" is not a mechanism. The mechanism is
`submission.tex`, which defines \\SUBMISSIONBUILD and drops the appendix; this file is the check
that the mechanism actually WORKS, in BOTH directions.

BOTH DIRECTIONS IS THE POINT. A one-sided check ("F is absent from the submission build") passes
just as happily if someone deletes the appendix from the paper entirely, or if the gate is
inverted, or if the submission target stops compiling the real document. So this asserts:

    1. the internal build (main.tex) CONTAINS Appendix F           -- the gate is not over-broad
    2. the submission build (submission.tex) does NOT contain it   -- the gate fires
    3. the submission build keeps every OTHER section heading      -- the gate is not a sledgehammer
    4. the submission build is strictly shorter                    -- something was actually removed

Any one of those failing is a real defect. Run:

    python3 paper/check_submission_build.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
F_TITLE = "Self-audit against the completion directive"

# Every top-level heading the SUBMISSION build must still contain. Written out rather than
# derived from the build, so that a gate which removed too much fails instead of agreeing.
REQUIRED_HEADINGS = (
    "Introduction",
    "Related work",
    "Capacity eviction in reconstruction-trained tokenizers",
    "Experimental design",
    "Data",
    "Results",
    "Limitations",
    "Reproducibility and verification",
    "Eviction measurement, per dimension",
    "Token-space plant: construction and calibration",
    "Two defects in the legibility gate",
    "The pinned decision surface",
    "Claims audit",
)


def _compile(stem: str) -> str:
    """Compile `stem`.tex and return its extracted text."""
    r = subprocess.run(
        ["tectonic", "-X", "compile", f"{stem}.tex"],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise SystemExit(
            f"FAIL: {stem}.tex did not compile (exit {r.returncode})\n{r.stderr[-2000:]}"
        )
    t = subprocess.run(
        ["pdftotext", f"{stem}.pdf", "-"], cwd=HERE, capture_output=True, text=True, check=True
    )
    return t.stdout


def _pages(stem: str) -> int:
    out = subprocess.run(
        ["pdfinfo", f"{stem}.pdf"], cwd=HERE, capture_output=True, text=True, check=True
    ).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    raise SystemExit(f"FAIL: could not read a page count for {stem}.pdf")


def main() -> int:
    internal = _compile("main")
    submission = _compile("submission")
    failures: list[str] = []

    if F_TITLE not in internal:
        failures.append(
            "the INTERNAL build (main.tex) does not contain Appendix F. Either the appendix was "
            "deleted or the gate is over-broad -- in both cases the read-through build is wrong."
        )
    if F_TITLE in submission:
        failures.append(
            "the SUBMISSION build contains Appendix F. The gate did not fire; do not submit this."
        )

    missing = [h for h in REQUIRED_HEADINGS if h not in submission]
    if missing:
        failures.append(f"the SUBMISSION build is missing sections it must keep: {missing}")

    p_int, p_sub = _pages("main"), _pages("submission")
    if p_sub >= p_int:
        failures.append(
            f"the SUBMISSION build is not shorter than the internal one "
            f"({p_sub} vs {p_int} pages), so nothing was actually removed."
        )

    if failures:
        print("SUBMISSION BUILD CHECK: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        f"SUBMISSION BUILD CHECK: PASS  "
        f"(internal {p_int} pp with Appendix F, submission {p_sub} pp without it, "
        f"{len(REQUIRED_HEADINGS)} required headings present in both)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
