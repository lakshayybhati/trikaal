"""The paper's amendment-log counts must equal what the log actually contains.

WHY THIS EXISTS. The paper asserted **65 revision tags / 62 top-level entries**. The log measured
**64 / 61**. Both numbers were wrong by exactly one, in four `.tex` files across five sites, and
**the entire test suite was green while it was true** — 968 passing, ruff clean, claim-drift CLEAN,
the Gate-A anchor reproducing. Nothing in `scripts/` or `tests/` counted either quantity, so every
instrument the project owns was silent on a factual error in a number the paper presents as
*measured*. A claim labelled "measured" with nothing measuring it is the defect this file closes.

★ THE TAG PATTERN IS THE SUBTLE PART, AND A THREE-PART PATTERN IS WRONG. Tags come in two shapes:
`v1.6.49` and `v1.5`. A pattern of the form `v\\d+\\.\\d+\\.\\d+` sees only the first shape and
silently drops **v1.0, v1.1, v1.2, v1.3, v1.4, v1.5, v1.6** — seven tags, an undercount that looks
like a plausible answer. The paper's own prose names `v1.0` as a tag, so any pattern that misses it
is refuted by the document it is measuring. `test_the_tag_pattern_sees_two_part_tags` pins that
directly rather than trusting the total.

WHAT IS READ AND WHAT IS NEVER WRITTEN. This reads `docs/m6_prereg.md` and `paper/**/*.tex`, both
outside the builder's domain, and writes neither. The counting functions take text so the mutation
tests can run them against temporary copies — a fixture that had to edit the real files to prove
the check works would be a fixture nobody could run twice.

WHOLE-TEXT, NOT LINE-BY-LINE. Two of the five assertion sites wrap across a newline
(`...carries $65$ distinct revision tags across\\n$62$ top-level entries`). A line-based extractor
finds three of five and reports agreement, which is the failure mode this project has hit before in
a different medium.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "m6_prereg.md"
PAPER = REPO / "paper"

# A top-level amendment entry: a list item whose first bold token is the revision tag.
# ★ ANCHORED ON `v<digit>`, NOT ON `v1.`. My first version hardcoded the major version, and the
# mutation fixture below — which plants a synthetic entry to prove the count moves — then failed
# because the planted tag was not a v1. The tempting fix was to plant a v1 entry instead; that
# would have been tailoring the fixture to the pattern, and the pattern is the thing under test.
ENTRY = re.compile(r"^- \*\*v\d+\.\d", re.M)
# ★ TWO-PART-AWARE. The optional third group is GREEDY, so `v1.6.49` matches whole rather than
# yielding a spurious `v1.6`, while a bare `v1.5` still matches.
TAG = re.compile(r"\bv1\.\d+(?:\.\d+)?\b")

# `$N$ ... revision tags` / `$N$ ... top-level entries`, where the gap may not cross another `$N$`.
CLAIM_TAGS = re.compile(r"\$(\d+)\$[^$]{0,60}?revision tags")
CLAIM_ENTRIES = re.compile(r"\$(\d+)\$[^$]{0,60}?top-level entries")


def count_entries(log_text: str) -> int:
    return len(ENTRY.findall(log_text))


def count_tags(log_text: str) -> int:
    return len(set(TAG.findall(log_text)))


def _flat(text: str) -> str:
    return " ".join(text.split())


def asserted(tex_text: str) -> tuple[list[int], list[int]]:
    """(tag counts asserted, entry counts asserted) in one .tex file."""
    flat = _flat(tex_text)
    return (
        [int(n) for n in CLAIM_TAGS.findall(flat)],
        [int(n) for n in CLAIM_ENTRIES.findall(flat)],
    )


def _sites() -> dict[str, tuple[list[int], list[int]]]:
    out = {}
    for tex in sorted(PAPER.rglob("*.tex")):
        tags, entries = asserted(tex.read_text(errors="replace"))
        if tags or entries:
            out[str(tex.relative_to(REPO))] = (tags, entries)
    return out


def test_the_sources_exist() -> None:
    assert PREREG.exists(), "the amendment log is the subject; without it this proves nothing"
    assert PAPER.is_dir()


def test_the_tag_pattern_sees_two_part_tags() -> None:
    """★ Pinned directly: a three-part pattern undercounts by seven and looks plausible."""
    for t in ("v1.0", "v1.1", "v1.5", "v1.6"):
        assert TAG.fullmatch(t), f"{t} must match — the paper's own prose names v1.0 as a tag"
    for t in ("v1.6.49", "v1.2.1"):
        assert TAG.fullmatch(t), f"{t} must match whole, not as its two-part prefix"
    # greedy third group: no spurious two-part match inside a three-part tag
    assert TAG.findall("see v1.6.49 and v1.5 here") == ["v1.6.49", "v1.5"]
    live = {t for t in TAG.findall(PREREG.read_text()) if t.count(".") == 1}
    assert len(live) >= 7, (
        f"only {len(live)} two-part tags found; a real undercount looks like this"
    )


def test_the_extractor_actually_finds_the_assertion_sites() -> None:
    """An extractor that matches nothing agrees with everything."""
    sites = _sites()
    assert len(sites) >= 4, f"expected the counts asserted in several .tex files, found {sites}"
    n_tag = sum(len(t) for t, _ in sites.values())
    assert n_tag >= 4, f"expected several tag-count assertions, found {n_tag}"


def test_the_asserted_counts_equal_the_measured_counts() -> None:
    log = PREREG.read_text()
    tags, entries = count_tags(log), count_entries(log)
    bad = []
    for path, (t_claims, e_claims) in sorted(_sites().items()):
        for c in t_claims:
            if c != tags:
                bad.append(f"{path}: asserts {c} revision tags; the log has {tags}")
        for c in e_claims:
            if c != entries:
                bad.append(f"{path}: asserts {c} top-level entries; the log has {entries}")
    assert not bad, (
        "the paper's amendment-log counts disagree with the log:\n  "
        + "\n  ".join(bad)
        + f"\n\n(measured: {tags} distinct revision tags across {entries} top-level entries)"
    )


def test_a_mutated_tex_count_is_caught(tmp_path) -> None:
    """MUTATION 1 — corrupt the number in a .tex copy; the comparison must fail."""
    log = PREREG.read_text()
    tags = count_tags(log)
    good = f"The log carries ${tags}$ distinct revision tags."
    assert asserted(good)[0] == [tags], "control: the honest sentence must parse and agree"
    doctored = tmp_path / "fake.tex"
    doctored.write_text(f"The log carries ${tags + 1}$ distinct revision tags.\n")
    claimed = asserted(doctored.read_text())[0]
    assert claimed == [tags + 1]
    assert claimed[0] != tags, "a wrong .tex count was NOT caught — this file is decorative"


def test_a_mutated_log_is_caught(tmp_path) -> None:
    """MUTATION 2 — change the LOG instead; the same comparison must fail from the other side."""
    log = PREREG.read_text()
    base_entries, base_tags = count_entries(log), count_tags(log)
    grown = log + "\n- **v9.9.9 (2099-01-01, A PLANTED ENTRY):** not a real amendment.\n"
    assert count_entries(grown) == base_entries + 1, "planting an entry must move the entry count"
    grown_tag = log + "\n- **v1.7 (2099-01-01, A PLANTED TAG):** not a real amendment.\n"
    assert count_tags(grown_tag) == base_tags + 1, "planting a tag must move the tag count"
    # and the paper's (unchanged) assertion must now disagree with the mutated log
    for tex, (t_claims, _) in _sites().items():
        if t_claims:
            assert t_claims[0] != count_tags(grown_tag), (
                f"{tex} still agrees with a log that gained a tag — the check is inert"
            )
            break


@pytest.mark.parametrize("wrapped", [True, False])
def test_claims_that_wrap_across_lines_are_still_found(wrapped: bool) -> None:
    """Two of the five real sites wrap; a line-based extractor finds three and reports agreement."""
    sep = "\n" if wrapped else " "
    text = (
        f"the amendment log carries $65$ distinct revision tags across{sep}$62$ top-level entries"
    )
    t, e = asserted(text)
    assert t == [65] and e == [62], f"wrapped={wrapped} lost a claim: {t} {e}"
