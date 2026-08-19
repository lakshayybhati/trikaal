"""The amendment log must account for every revision tag it contains.

THE IDENTITY: ``headers + in-body-only == total distinct tags``.

WHY THIS SHAPE. Its predecessor compared the log against numbers the manuscript asserted. That
caught a real one-off error, and it had a structural weakness the writer named: **equality between
two documents can hold while both are wrong**, and it dies the moment either side goes away — which
is exactly what happened when the manuscript left the repository on 2026-08-19. This identity is
INTERNAL. It needs no second document, so it cannot be defeated by one going missing, and it
catches the failure that started the whole episode: **a tag silently gaining or losing a header.**

WHAT EACH TERM MEANS. A *header* is a top-level list item whose first bold token is the tag — an
entry with its own dated body. An *in-body-only* tag is one referenced inside another entry's prose
without an entry of its own; there are legitimately three (``v1.0``, ``v1.6.24``, ``v1.6.27``). Any
tag must be exactly one of the two. If a header is deleted while its tag survives in prose, or a
new tag appears with no entry, the sum stops matching the total and this fails.

★ IT REFUSES RATHER THAN PASSING VACUOUSLY. Its predecessor's most useful moment was asserting
``0 >= 4`` and FAILING when its input vanished, instead of reporting "no disagreements found" over
an empty scan. That property is preserved deliberately: ``test_the_log_is_actually_populated``
requires a plausible number of headers and tags before any identity is believed, so an emptied,
truncated or renamed log fails loudly instead of going quiet.

★ THE TAG PATTERN IS TWO-PART AWARE, and a three-part pattern is wrong. Tags come as ``v1.6.49``
and as ``v1.5``. A ``v\\d+\\.\\d+\\.\\d+`` pattern silently drops ``v1.0 v1.1 v1.2 v1.3 v1.4 v1.5
v1.6`` — seven tags, an undercount that looks like a plausible answer.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "m6_prereg.md"

# Greedy optional third group: `v1.6.49` matches whole; a bare `v1.5` still matches.
TAG = re.compile(r"\bv1\.\d+(?:\.\d+)?\b")
# A top-level entry: list item whose first bold token is the tag. Not anchored to a major version.
HEADER = re.compile(r"^- \*\*(v\d+\.\d+(?:\.\d+)?)", re.M)


def headers(text: str) -> set[str]:
    return set(HEADER.findall(text))


def tags(text: str) -> set[str]:
    return set(TAG.findall(text))


def in_body_only(text: str) -> set[str]:
    return tags(text) - headers(text)


def _log() -> str:
    return PREREG.read_text()


def test_the_log_exists() -> None:
    assert PREREG.exists(), "the amendment log is the subject; without it this proves nothing"


def test_the_log_is_actually_populated() -> None:
    """★ REFUSE RATHER THAN PASS VACUOUSLY. An empty log satisfies every identity below."""
    t = _log()
    assert len(t) > 10_000, f"log is {len(t)} bytes — implausibly small, refusing to check it"
    assert len(headers(t)) >= 40, f"only {len(headers(t))} entry headers found; the scan is broken"
    assert len(tags(t)) >= 40, f"only {len(tags(t))} distinct tags found; the scan is broken"


def test_the_tag_pattern_sees_two_part_tags() -> None:
    """A three-part pattern undercounts by seven and looks plausible while doing it."""
    for t in ("v1.0", "v1.1", "v1.5", "v1.6"):
        assert TAG.fullmatch(t), f"{t} must match — two-part tags are real tags"
    for t in ("v1.6.49", "v1.2.1"):
        assert TAG.fullmatch(t), f"{t} must match whole, not as its two-part prefix"
    assert TAG.findall("see v1.6.49 and v1.5 here") == ["v1.6.49", "v1.5"]
    two = {t for t in tags(_log()) if t.count(".") == 1}
    assert len(two) >= 7, (
        f"only {len(two)} two-part tags; a real undercount looks exactly like this"
    )


def test_every_tag_is_either_a_header_or_an_in_body_reference() -> None:
    """THE IDENTITY. headers + in-body-only == total distinct tags."""
    t = _log()
    h, ib, all_t = headers(t), in_body_only(t), tags(t)
    assert len(h) + len(ib) == len(all_t), (
        f"{len(h)} headers + {len(ib)} in-body-only != {len(all_t)} distinct tags"
    )
    assert h.isdisjoint(ib), f"a tag counted twice: {sorted(h & ib)}"
    assert h | ib == all_t


def test_the_in_body_only_set_is_small_and_named() -> None:
    """In-body-only tags are legitimate but should stay few; a jump means a header was lost."""
    ib = in_body_only(_log())
    assert len(ib) <= 6, (
        f"{len(ib)} tags have no entry of their own ({sorted(ib)}) — a header was probably deleted "
        f"while its tag survived in prose, which is the failure this identity exists to catch"
    )


def test_a_tag_without_a_header_is_caught() -> None:
    """MUTATION 1 — plant a tag in prose with no entry; in-body-only must grow."""
    t = _log()
    before = len(in_body_only(t))
    planted = t + "\n  Some prose mentioning v1.9.9 with no entry of its own.\n"
    after = len(in_body_only(planted))
    assert after == before + 1, f"a headerless tag was NOT caught: {before} -> {after}"
    assert "v1.9.9" in in_body_only(planted)


def test_a_header_without_its_tag_in_the_body_is_still_counted(tmp_path) -> None:
    """MUTATION 2 — plant an entry header; headers must grow and the identity must hold."""
    t = _log()
    h0, all0 = len(headers(t)), len(tags(t))
    planted = t + "\n- **v1.9.8 (2099-01-01, A PLANTED ENTRY):** not a real amendment.\n"
    assert len(headers(planted)) == h0 + 1, "planting an entry must move the header count"
    assert len(tags(planted)) == all0 + 1, "its tag must also join the total"
    assert len(headers(planted)) + len(in_body_only(planted)) == len(tags(planted))


def test_deleting_a_header_while_its_tag_survives_breaks_the_identity() -> None:
    """★ MUTATION 3 — THE ACTUAL FAILURE THIS EXISTS FOR, and the identity must move.

    Strip one entry's header line while its tag still appears in another entry's prose. The tag
    total is unchanged; the header count drops; the tag migrates into in-body-only. The identity
    still SUMS — that is what makes it an identity — so what must be caught is the MIGRATION, which
    is why ``test_the_in_body_only_set_is_small_and_named`` bounds that set rather than trusting
    the sum alone. Both halves are needed; neither alone would see this.
    """
    t = _log()
    victim = sorted(headers(t))[0]
    h0, ib0 = len(headers(t)), len(in_body_only(t))
    mutated = re.sub(rf"^- \*\*{re.escape(victim)}", f"  (was {victim})", t, count=1, flags=re.M)
    assert len(headers(mutated)) == h0 - 1, f"header removal not detected for {victim}"
    assert len(in_body_only(mutated)) == ib0 + 1, "the orphaned tag did not migrate as expected"
    assert victim in in_body_only(mutated)
