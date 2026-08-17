#!/usr/bin/env python3
"""Sweep the paper for RETIRED CLAIMS, by their distinctive terms rather than their phrasing.

WHY THIS EXISTS, AND WHY IT IS A SCRIPT RATHER THAN A GREP.

The same defect has now appeared five times: a correction propagates to what you edit, not to what
derives from or restates it.

    prose  -> generators    fig1 kept "21.3M" through the parameter sweep
    title  -> figures       fig12 kept "duplicates price" through the title fix
    source -> exports       main_full.txt kept the superseded title for four days
    fix    -> the sites its own grep missed   (twice, in one hour)

Three separate hand-greps missed the abstract. Each was NARROWER THAN THE CLAIM:

    * a sweep over sections/ that did not include main.tex, where the abstract lives
    * a match on the exact phrase "duplicates price", where the abstract said
      "duplicates what price already carries"
    * a match on "vs" where the text said "against"

THE LESSON, AND THE DESIGN OF THIS FILE: grepping the superseded STRING catches verbatim
survivals. It does not catch THE CLAIM RESTATED IN DIFFERENT WORDS. So each rule below pairs a
CLAIM pattern with a FORBIDDEN pattern, and fires on any sentence matching both -- which is a
statement about meaning rather than about wording. Sentences are reconstructed from whole text,
not read line by line, because in a tree that wraps at 100 characters most claims span lines.

    python3 paper/check_claim_drift.py        # exit 1 if any retired claim survives
"""

from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent

# Each rule: (name, claim-pattern, forbidden-pattern, why, allowed exceptions by substring).
# An exception must carry the reason it is allowed, so a silence cannot outlive its justification.
RULES: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    (
        "eviction-claim names PRICE where the referent is the OHLCV block",
        r"(duplicat|co-?vari|covarian|spends its bits|evict|keeps the micro|preserves the micro"
        r"|already carries|already reconstruct|survivors)",
        r"\bprice\b",
        "The surviving microstructure channels co-vary with VOLUME, not with price. This is the "
        "reviewer's first quibble and the reason the title changed.",
        (
            # the fixture's own geometry, where the return dimension IS the price analogue and the
            # sentence is about what we do NOT know rather than about which channels survived
            "we do not know the variance and covariance geometry of real trade-flow imbalance",
            # the SHUFFLE genuinely does make the block independent of its own bar's price -- that
            # is what block_time_permute does, and it is a statement about the placebo's
            # construction rather than about which channels survived
            "independent of its own bar's price",
        ),
    ),
    (
        "a measurement over the OHLCV block described as covering only PRICE",
        r"(takes nothing from|costs .{0,20}nothing|reconstruction|recon MAE|buy capacity from"
        r"|does not degrade|spends its bits)",
        r"\bprice channels?\b",
        "OHLCV reconstruction is measured over SEVEN dimensions, two of which are volume "
        "(log_volume, log_amount). Naming the price channels alone understates the measurement "
        "AND drops the volume channels the surviving microstructure co-varies with.",
        (),
    ),
    (
        "the cumulative ladder claimed NECESSITY of each intervention",
        r"(each was necessary|necessary and not sufficient)",
        r".",
        "A cumulative ladder establishes insufficiency of the PREFIXES and sufficiency of the "
        "TRIPLE. Necessity needs each part removed from the final triple, which was never done.",
        (),
    ),
    (
        "the magnitude channels described as passing when two of four fail",
        r"essentially pass",
        r"magnitude",
        "Two of the four magnitude dimensions miss the gate by 0.0025 and 0.0038. §6.1 says so; "
        "summaries must not round that away.",
        (),
    ),
    (
        "the gate's firing attributed to the full lake rather than its probe",
        r"gate fired",
        r"200 instruments",
        "The lake is the TRAINING draw. The gate's probe measured 150,000 rows across 40 symbols.",
        (),
    ),
    (
        "a placebo comparison described as capacity-neutral",
        r"placebo",
        r"capacity (is )?(held )?(constant|fixed|neutral|equal)",
        "C-12: cell 5 reconstructs byte-identical OHLCV targets 1.51x worse than cell 4. The "
        "argument must rest on the WITHIN-arm signed-vs-magnitude split.",
        (
            # the paper states the false framing in order to reject it, which is the point
            "is therefore false",
        ),
    ),
)

# A sentence ends at a period followed by whitespace and a capital or a TeX control sequence --
# NOT at every period. "0.90" and "\cref{sec:x}." are not sentence boundaries, and an earlier
# version of this splitter broke on both, producing fragments that spanned unrelated claims and
# fired two false positives on a clean tree. A gate with false positives becomes decoration.
SENTENCE_END = re.compile(r"(?<=[a-z0-9\)\}])\.(?=\s+(?:[A-Z\\]|$))")


def _sentences(text: str) -> list[str]:
    """Whole-text sentences with LaTeX comments stripped first.

    Comments must go before flattening: a `%` line in the middle of a paragraph is not a claim a
    reader sees, but once newlines collapse it merges into the surrounding prose and is
    indistinguishable from it. That merge was the other false positive.
    """
    lines = []
    for raw in text.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("%"):
            continue
        lines.append(re.sub(r"(?<!\\)%.*$", "", raw))  # trailing comments too, but not \%
    flat = re.sub(r"\s+", " ", " ".join(lines))
    out, last = [], 0
    for m in SENTENCE_END.finditer(flat):
        out.append(flat[last : m.end()])
        last = m.end()
    if flat[last:].strip():
        out.append(flat[last:])
    return out


def main() -> int:
    targets = [*sorted(HERE.glob("sections/*.tex")), HERE / "main.tex"]
    missing = [t for t in targets if not t.exists()]
    if missing or len(targets) < 5:
        raise SystemExit(f"FAIL: target set looks wrong ({len(targets)} files, missing {missing})")

    failures: list[str] = []
    for path in targets:
        text = path.read_text()
        for sentence in _sentences(text):
            stripped = sentence.strip()
            if stripped.startswith("%"):
                continue  # a source comment, not a claim the reader sees
            for name, claim, forbidden, why, exceptions in RULES:
                if not re.search(claim, sentence, re.I):
                    continue
                if not re.search(forbidden, sentence, re.I):
                    continue
                if any(exc in sentence for exc in exceptions):
                    continue
                failures.append(f"{path.name}: [{name}]\n      …{stripped[:160]}\n      why: {why}")

    if failures:
        print(f"CLAIM DRIFT: {len(failures)} retired claim(s) survive", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        f"CLAIM DRIFT: CLEAN  ({len(RULES)} rules over {len(targets)} files, "
        "sentences reconstructed from whole text)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
