"""Audit a manuscript against the corrections that landed in the repo AFTER it left the repo.

★ WHY THIS EXISTS. Every guard in this project walks ``git ls-files``, and ``paper/`` is ignored at
``.gitignore:52``. The claim-drift sweep, ``tests/claim_site.py``, the model-card guard and the
receipt sweeps are therefore STRUCTURALLY INCAPABLE of seeing the project's most important
document. Tiers 1, 1b, 2, 3 and 4 all corrected ``docs/MODEL_CARD.md`` and ``README.md``; none of
those corrections could reach the paper, because the paper was already gone before those tiers ran.

★ AND A PLAIN GREP IS BLIND TO IT. The manuscript writes ``304{,}625{,}181`` and ``$40$``. Searching
for ``304,625,181`` or ``40 symbols`` returns NOTHING — a confident clean result on a document that
states exactly those figures. That is the same shape as the ``git grep``/``\\b`` defect: a pattern
that cannot match is indistinguishable from a document that does not say it. Everything here runs
through ``claim_site.latex_plain`` first.

THE PAPER LOCATION IS A PARAMETER. It may live beside the repo, in a private sibling repository, or
nowhere. ``--paper-dir``, else ``$TRIKAAL_PAPER_DIR``, else ``<repo>/paper``.

    .venv/bin/python scripts/m6_paper_claim_audit.py            # exit 1 if any claim is stale
    .venv/bin/python scripts/m6_paper_claim_audit.py --json OUT
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

from claim_site import latex_plain, paper_dir, paper_docs  # noqa: E402
from trikaal.utils.paths import display_path  # noqa: E402

# Sentence end, not every period: "0.90" and "\cref{sec:x}." are not boundaries.
SENTENCE_END = re.compile(r"(?<=[a-z0-9\)\}])\.(?=\s+(?:[A-Z\\]|$))")


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", latex_plain(text))
    out, last = [], 0
    for m in SENTENCE_END.finditer(flat):
        out.append(flat[last : m.end()].strip())
        last = m.end()
    if flat[last:].strip():
        out.append(flat[last:].strip())
    return out


#: (id, claim pattern, forbidden/required, mode, why). mode: "forbid" fires when BOTH match;
#: "require" fires when the claim matches and the second pattern does NOT.
RULES = (
    (
        "97.3-unscoped",
        r"97\.3\s*%",
        r"cell\s*4",
        "require",
        "97.3% is CELL 4's shortfall — one cell, one seed, one run — and cell 4's weights are not "
        "published and never will be. Unqualified it reads as a property of the released model. "
        "Corrected in docs/MODEL_CARD.md and README.md in tier 1.",
    ),
    (
        "trained-on-the-lake",
        r"\b(other\s+)?160\b|\b200\b[^.]{0,60}(instrument|symbol)",
        r"(?:training data|trained on)(?![^.]*?(?:same \$?40|identical to the scored|"
        r"never trained|40 symbols they are scored))",
        "forbid",
        "The units were trained on the SAME 40 symbols they were scored on — "
        "draw.drawn_by_symbol_stage1 is 40 in all three units and equals m6_mde_inputs "
        "symbols_sampled exactly. The other 160 instruments were never trained on. Saying they "
        "are 'training data' overstates what the model saw by 5x. "
        "\u2605 THE NEGATIVE LOOKAHEAD IS LOAD-BEARING: a CORRECT statement of this fact must "
        "name BOTH halves — '160' and 'trained on' — so a rule matching the two together fires on "
        "the FIX. It did, on the writer's correction at the paper repo's 7d2d6c1, which is the "
        "eighth instance of the class in tests/claim_site.py. A sentence that asserts the "
        "40-symbol identity is the fix, not the defect.",
    ),
    (
        "magnitude-range-0.8975",
        r"0\.8975\s*(?:-|--|—|to)\s*0\.9320",
        r".",
        "forbid",
        "0.8975 is dim 9's value; the magnitude MINIMUM is 0.8962 (dim 11). As a range low end it "
        "makes the signed-vs-magnitude contrast look cleaner than it measured. Tier 1.4.",
    ),
    (
        "replicate-spread-unqualified",
        r"77\s*(?:-|--|—|to|and)\s*93|77\s*%[^.]{0,20}93",
        r"NOT_COMPARABLE|not comparable|spread rather than|dispersion rather than|"
        r"not a comparison|rather than as a comparison",
        "require",
        "m6_lambda_sweep.json declares itself NOT_COMPARABLE_TO_THE_GATE_FIRING_RUN. Quoting the "
        "spread beside 97.3% invites a delta that file forbids. Tier 1b.2. The prose form of the "
        "prohibition counts — a rule that demanded the field NAME would cry wolf on a sentence "
        "that carries the meaning, and a guard that fires on correct text gets ignored.",
    ),
    (
        "canary-1.151-underived",
        r"1\.151",
        r"\\?tfrac|\\?ln|0\.5\s*\*?\s*ln|closed[- ]form|derived|c\s*=\s*3|C_SIGNAL",
        "require",
        "1.151 is EXACT — 1/2 ln(1+c^2) at C_SIGNAL = 3.0 (scripts/m6_canary.py:100) = 1/2 ln 10 = "
        "1.1512925. It is a derived closed form, not a measurement, so it must appear with its "
        "derivation or a pointer to it, or a reader cannot tell manufactured precision from real.",
    ),
    (
        "cells-never-trained",
        r"cells?\s*2\s*(?:-|--|—|through|to)\s*5|cells?\s*3/4/5",
        r"never trained",
        "forbid",
        "Stage 2 was never ENTERED for cells 2-5, but cells 4 and 5 ran through Stage 1 and that "
        "run produced the legibility receipt the whole finding rests on. Tier 1.5.",
    ),
)

#: Things a reader needs that the paper may simply not say. Absence is the defect.
REQUIRED_PRESENT = (
    ("draw-40", r"\b40\b", "the 40-symbol training draw"),
    ("holdout-boundary", r"2023-10-20", "the train/holdout boundary date"),
)


def audit(docs: list[Path]) -> tuple[list[dict], list[dict]]:
    findings: list[dict] = []
    for path in docs:
        text = path.read_text(errors="ignore")
        for sent in sentences(text):
            for rid, claim, other, mode, why in RULES:
                if not re.search(claim, sent, re.I):
                    continue
                hit = re.search(other, sent, re.I)
                if (mode == "forbid" and hit) or (mode == "require" and not hit):
                    findings.append(
                        {
                            "rule": rid,
                            "file": display_path(path, REPO),
                            "why": why,
                            "sentence": sent[:240],
                        }
                    )
    joined = " ".join(latex_plain(p.read_text(errors="ignore")) for p in docs)
    absent = [
        {"rule": rid, "missing": what}
        for rid, pat, what in REQUIRED_PRESENT
        if not re.search(pat, joined)
    ]
    return findings, absent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=None, help="manuscript root; else $TRIKAAL_PAPER_DIR")
    ap.add_argument("--json", type=Path, default=None, help="write findings here")
    args = ap.parse_args()

    d = paper_dir(args.paper_dir)
    if d is None:
        print(
            "NO MANUSCRIPT FOUND. Pass --paper-dir or set TRIKAAL_PAPER_DIR. This is a REFUSAL, "
            "not a pass: a sweep with no subject has proven nothing.",
            file=sys.stderr,
        )
        return 2

    docs = paper_docs(args.paper_dir)
    findings, absent = audit(docs)
    print(f"manuscript: {d}  ({len(docs)} .tex files)")

    by_rule: dict[str, int] = {}
    for f in findings:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
    for rid, *_ in RULES:
        print(f"  {rid:32s} {by_rule.get(rid, 0)} finding(s)")
    for a in absent:
        print(f"  {a['rule']:32s} ABSENT — {a['missing']}")

    if findings or absent:
        print(
            f"\nCLAIM AUDIT: {len(findings)} stale claim(s), {len(absent)} absent", file=sys.stderr
        )
        for f in findings:
            print(
                f"\n  [{f['rule']}] {f['file']}\n      …{f['sentence']}\n      why: {f['why']}",
                file=sys.stderr,
            )
        for a in absent:
            print(f"\n  [{a['rule']}] the manuscript never states {a['missing']}", file=sys.stderr)

    if args.json:
        args.json.write_text(
            json.dumps({"findings": findings, "absent": absent}, indent=2, sort_keys=True) + "\n"
        )
    return 1 if (findings or absent) else 0


if __name__ == "__main__":
    sys.exit(main())
