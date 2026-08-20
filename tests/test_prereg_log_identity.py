"""The amendment log must account for every revision tag it contains — checked against a COUNT.

★ WHAT THIS FILE USED TO DO, AND WHY IT WAS WORTHLESS. It asserted an "identity":
``len(headers) + len(in_body_only) == len(tags)``, plus ``headers.isdisjoint(in_body_only)`` and
``headers | in_body_only == tags``. But ``in_body_only`` is DEFINED as ``tags - headers``, so all
three reduce to ``headers ⊆ tags`` and are otherwise true of every possible input. Measured: the
identity holds on the empty string, on 200 random strings, and on the real log **with every header
line stripped out**. Its only non-tautological behaviour was a LATENT FALSE POSITIVE — ``TAG`` was
anchored to ``v1.`` while ``HEADER`` accepted any major version, so the first legitimate ``v2.0``
entry would have failed it. I specified that identity and called it the durable win of the day.

WHAT SURVIVES. The file was not worthless: its ``>10k`` / ``>=40`` / ``>=40`` bounds do reject an
emptied log, a truncated log and a header-stripped log, which is exactly the vacuity the identity
waves through. Those are kept.

WHAT REPLACES THE IDENTITY. Two checks that can actually fail:

1. **A CENSUS**, pinned as numbers. 62 entry headers, 65 distinct tags, and three named
   in-body-only tags. Any drift fails and must be updated deliberately, which is the point — a
   count that moves silently is how an entry gets deleted without anyone noticing.
2. **SEQUENCE CONTINUITY**, derived from the tags themselves and independent of the header regex.
   Amendments are numbered, so a deleted entry leaves a HOLE: ``v1.6`` runs 1–51 with exactly one
   justified gap at 21, ``v1.4`` runs 1–7 with none. An unexplained gap is a missing amendment,
   and no arrangement of headers can hide it.

``TAG`` is also no longer anchored to major version 1, so it and ``HEADER`` see the same universe
and the false positive is gone.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "m6_prereg.md"

# Greedy optional third group: `v1.6.49` matches whole; a bare `v1.5` still matches. NOT anchored
# to major version 1 — that mismatch against HEADER was the old file's one live defect.
TAG = re.compile(r"\bv\d+\.\d+(?:\.\d+)?\b")
HEADER = re.compile(r"^- \*\*(v\d+\.\d+(?:\.\d+)?)", re.M)

# ── THE CENSUS. Measured 2026-08-20; update deliberately, never to make a failure go away. ────
EXPECTED_HEADERS = 62
EXPECTED_TAGS = 65
EXPECTED_IN_BODY_ONLY = {"v1.0", "v1.6.24", "v1.6.27"}
#: Numbering holes that are real and accounted for. A gap NOT listed here is a deleted amendment.
JUSTIFIED_GAPS = {"v1.6": {21}}


def headers(text: str) -> set[str]:
    return set(HEADER.findall(text))


def tags(text: str) -> set[str]:
    return set(TAG.findall(text))


def in_body_only(text: str) -> set[str]:
    return tags(text) - headers(text)


def families(text: str) -> dict[str, set[int]]:
    """``{"v1.6": {1, 2, ...}}`` — patch numbers per major.minor family."""
    fam: dict[str, set[int]] = defaultdict(set)
    for tag in tags(text):
        parts = tag[1:].split(".")
        if len(parts) == 3:
            fam[f"v{parts[0]}.{parts[1]}"].add(int(parts[2]))
    return dict(fam)


def _log() -> str:
    return PREREG.read_text()


# ── the subject exists and is real ────────────────────────────────────────────────────────────
def test_the_log_exists() -> None:
    assert PREREG.exists(), "the amendment log is the subject; without it this proves nothing"


def test_the_log_is_actually_populated() -> None:
    """★ REFUSE RATHER THAN PASS VACUOUSLY — and these bounds are the part of the old file that
    genuinely worked, so they are kept verbatim in spirit."""
    t = _log()
    assert len(t) > 10_000, f"log is {len(t)} bytes — implausibly small, refusing to check it"
    assert len(headers(t)) >= 40, f"only {len(headers(t))} entry headers found; the scan is broken"
    assert len(tags(t)) >= 40, f"only {len(tags(t))} distinct tags found; the scan is broken"


# ── the census ────────────────────────────────────────────────────────────────────────────────
def test_the_entry_census_matches() -> None:
    t = _log()
    h, a, ib = len(headers(t)), len(tags(t)), in_body_only(t)
    assert (h, a) == (EXPECTED_HEADERS, EXPECTED_TAGS), (
        f"census moved: {h} headers / {a} tags, expected "
        f"{EXPECTED_HEADERS} / {EXPECTED_TAGS}. If an amendment was genuinely added or removed, "
        "update the constants in this file in the same commit and say so."
    )
    assert ib == EXPECTED_IN_BODY_ONLY, (
        f"the in-body-only set changed: {sorted(ib)} vs {sorted(EXPECTED_IN_BODY_ONLY)} — a header "
        "was probably deleted while its tag survived in another entry's prose"
    )


# ── sequence continuity: independent of the header regex ─────────────────────────────────────
def test_no_amendment_number_is_missing() -> None:
    """A DELETED ENTRY LEAVES A HOLE, and no arrangement of headers can conceal it."""
    holes = {}
    for fam, patches in sorted(families(_log()).items()):
        gaps = {i for i in range(1, max(patches) + 1) if i not in patches}
        unexplained = gaps - JUSTIFIED_GAPS.get(fam, set())
        if unexplained:
            holes[fam] = sorted(unexplained)
    assert not holes, f"amendment numbers missing from the log: {holes}"


def test_the_justified_gaps_are_still_gaps() -> None:
    """An allow-list must rot loudly: if v1.6.21 ever appears, this exemption is stale."""
    fam = families(_log())
    for name, gaps in JUSTIFIED_GAPS.items():
        assert name in fam, f"family {name} vanished from the log"
        for g in gaps:
            assert g not in fam[name], (
                f"{name}.{g} now EXISTS — remove it from JUSTIFIED_GAPS rather than leaving a "
                "dead exemption behind"
            )


def test_the_tag_pattern_sees_two_part_tags() -> None:
    """A three-part pattern undercounts by seven and looks plausible while doing it."""
    for t in ("v1.0", "v1.1", "v1.5", "v1.6", "v2.0"):
        assert TAG.fullmatch(t), f"{t} must match — two-part tags are real tags"
    for t in ("v1.6.49", "v1.2.1"):
        assert TAG.fullmatch(t), f"{t} must match whole, not as its two-part prefix"
    assert TAG.findall("see v1.6.49 and v1.5 here") == ["v1.6.49", "v1.5"]
    two = {t for t in tags(_log()) if t.count(".") == 1}
    assert len(two) >= 7, (
        f"only {len(two)} two-part tags; a real undercount looks exactly like this"
    )


def test_TAG_and_HEADER_see_the_same_universe() -> None:
    """THE OLD FILE'S ONE LIVE DEFECT: TAG was anchored to v1 and HEADER was not, so the first v2
    entry would have failed a check that was otherwise true of every input."""
    v2 = "- **v2.0 (2099-01-01):** the first v2 entry.\n"
    assert headers(v2) == {"v2.0"} and tags(v2) == {"v2.0"}
    assert not (headers(_log() + v2) - tags(_log() + v2)), (
        "a v2 header is not seen as a tag — the anchoring mismatch is back"
    )


# ── a deliberately corrupted log must FAIL ────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("name", "corrupt"),
    [
        ("emptied", lambda t: ""),
        ("truncated", lambda t: t[:5000]),
        (
            "headers stripped",
            lambda t: "\n".join(ln for ln in t.split("\n") if not ln.startswith("- **v")),
        ),
        ("one entry deleted", lambda t: re.sub(r"^- \*\*v1\.6\.30.*$", "", t, count=1, flags=re.M)),
    ],
)
def test_a_corrupted_log_fails(name, corrupt) -> None:
    """★ THE ACCEPT. The old identity survived every one of these."""
    t = corrupt(_log())
    populated = len(t) > 10_000 and len(headers(t)) >= 40 and len(tags(t)) >= 40
    census = (len(headers(t)), len(tags(t))) == (EXPECTED_HEADERS, EXPECTED_TAGS)
    assert not (populated and census), f"a {name} log passed both checks"


def test_the_old_identity_would_have_passed_all_of_those() -> None:
    """NEGATIVE CONTROL, and the reason this file was rewritten. Kept so nobody reintroduces it."""

    def old_identity(t: str) -> bool:
        h, ib, a = headers(t), in_body_only(t), tags(t)
        return len(h) + len(ib) == len(a) and h.isdisjoint(ib) and (h | ib) == a

    stripped = "\n".join(ln for ln in _log().split("\n") if not ln.startswith("- **v"))
    assert old_identity("") and old_identity("nonsense") and old_identity(stripped), (
        "the old identity no longer passes on empty input — re-read this file's premise"
    )
