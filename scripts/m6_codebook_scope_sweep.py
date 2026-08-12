"""SWEEP: every statement about codebook utilization / occupancy / dead-code / collapse,
with its QUANTIZER SCOPE. Supervisor ruling item 2. REPORTS ONLY — proposes nothing.

THE QUESTION IT ANSWERS. `PINNED_CODEBOOK_MIN_UTILIZATION = 0.95` is enforced against EVERY cell,
citing design :1859 — a sentence about FSQ. Establishing that the sentence is about FSQ is only
HALF the argument; the other half is that NO OTHER authority extends a codebook criterion to BSQ.
Nobody has checked. This checks.

★ THE TOOL IS SUSPECT UNTIL SHOWN CAPABLE OF FAILING (class rule). `--self-test` plants sentences
whose scope is known and asserts the classifier returns it — including a BSQ-scoped one, which is
the case whose ABSENCE from the real corpus would be the finding. A sweep that cannot detect a BSQ
statement cannot report that there are none.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

FILES = [
    Path("docs/superpowers/specs/2026-06-18-trikaal-v1-design.md"),
    Path("docs/m6_prereg.md"),
    Path("docs/m6_design.md"),
]

# Terms that would carry a codebook-health criterion. Deliberately WIDER than "utilization".
TERMS = re.compile(
    r"codebook\s+(usage|utili[sz]ation|occupancy)|utili[sz]ation|occupancy|dead[-\s]?code|"
    r"collapse[sd]?|codes?\s+used|perplexit",
    re.I,
)


def scope_of(text: str) -> str:
    """Quantizer scope of one statement: FSQ / BSQ / BOTH / UNSCOPED."""
    t = text.lower()
    fsq = bool(re.search(r"\bfsq\b|891|1225|11,\s*9,\s*9|20\.06", t))
    bsq = bool(re.search(r"\bbsq\b|binary spherical|2\*\*20|2\^20", t))
    if fsq and bsq:
        return "BOTH"
    if fsq:
        return "FSQ"
    if bsq:
        return "BSQ"
    return "UNSCOPED"


def self_test() -> int:
    """The classifier must DISCRIMINATE, or 'no BSQ statement exists' is unfalsifiable."""
    cases = [
        ("fraction of FSQ codes used over the eval set (target >= 95%)", "FSQ"),
        ("the BSQ codebook must not collapse; utilization >= 0.9", "BSQ"),
        ("both FSQ and BSQ codebooks are checked for dead-code collapse", "BOTH"),
        ("codebook utilization is reported per cell", "UNSCOPED"),
        ("coarse group {11, 9, 9} -> vocab 891; utilization tracked", "FSQ"),
        ("binary spherical quantization occupancy", "BSQ"),
    ]
    bad = [(t, want, scope_of(t)) for t, want in cases if scope_of(t) != want]
    for t, want, got in bad:
        print(f"  SELF-TEST FAIL want={want} got={got}: {t[:70]}", file=sys.stderr)
    if bad:
        print(f"SELF_TEST_EXIT=1 ({len(bad)}/{len(cases)} failed)", file=sys.stderr)
        return 1
    # NEGATIVE CONTROL: a term-free sentence must not match TERMS at all.
    if TERMS.search("the autoregressive backbone has eight layers"):
        print("SELF-TEST FAIL: TERMS matched a sentence with no codebook term", file=sys.stderr)
        return 1
    print(f"SELF_TEST_EXIT=0 ({len(cases)}/{len(cases)} scope cases + negative control)")
    return 0


def _read(p: Path, rev: str | None) -> str:
    """File contents at ``rev`` (``git show``), or the working tree when ``rev`` is None."""
    if rev is None:
        return p.read_text()
    out = subprocess.run(["git", "show", f"{rev}:{p}"], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"git show {rev}:{p} failed: {out.stderr.strip()}")
    return out.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--rev",
        default=None,
        help="git revision to read the corpus AT. ★ THE ANSWER DEPENDS ON THIS. Run against the "
        "working tree AFTER §7 v1.6.30 landed and the sweep reads the AMENDMENT ITSELF quoting "
        "the evidence for the amendment — 50 hits, 2 BSQ-scoped, both inside the new prereg "
        "entry. The finding that justified the amendment was measured at 055d14f (37 hits, ZERO "
        "BSQ-scoped) and must be re-checked there, never here. Baseline rule: check a claim "
        "against the commit the claim was made about.",
    )
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    if self_test() != 0:
        print(
            "REFUSING TO SWEEP: the classifier failed its own discrimination test", file=sys.stderr
        )
        return 2

    total = 0
    by_scope: dict[str, int] = {}
    print(f"CORPUS READ AT: {a.rev or 'WORKING TREE (uncommitted)'}")
    for p in FILES:
        if a.rev is None and not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            return 2
        lines = _read(p, a.rev).splitlines()
        hits = [(i + 1, ln) for i, ln in enumerate(lines) if TERMS.search(ln)]
        print(f"\n{'=' * 100}\n{p}  —  {len(hits)} hit(s)\n{'=' * 100}")
        for n, ln in hits:
            sc = scope_of(ln)
            by_scope[sc] = by_scope.get(sc, 0) + 1
            total += 1
            s = ln.strip()
            print(f"  [{sc:8}] :{n}  {s[:240]}")
    print(f"\n{'=' * 100}\nTOTAL HITS {total}   by scope: {dict(sorted(by_scope.items()))}")
    print(
        "BSQ-scoped statements:", by_scope.get("BSQ", 0), " BOTH-scoped:", by_scope.get("BOTH", 0)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
