"""The audit findings file asserts its OWN completeness and its OWN integrity.

Two properties, both machine-enforced, because the file has exactly two failure modes and a
human reading it would catch neither reliably:

1. **COMPLETENESS** — every one of C-1..C-20 and S-1..S-4 has a real body inside the verbatim
   region. The register existed for two tranches with 20 of 24 bodies missing; nobody noticed
   because a partial file reads exactly like a complete one.
2. **INTEGRITY** — the verbatim region is byte-pinned by sha256. The instruction was *do not
   paraphrase, do not tidy, do not correct the drifted line numbers*. Prose does not enforce
   that; a hash does. Any edit inside the markers — including a well-meant typo fix — fails here
   and must be made deliberately by re-pinning, which leaves a diff a reviewer can see.

The boundary this protects is load-bearing: PART 1 is the external auditor's text and PART 2 is
our disposition, and a re-auditor must never read the second as the first.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "m6_readiness_audit_findings.md"
BEGIN = "<!-- ===== BEGIN VERBATIM AUDIT — DO NOT EDIT BELOW THIS LINE ===== -->"
END = "<!-- ===== END VERBATIM AUDIT ===== -->"

# Re-pin ONLY with a deliberate, reviewable diff. This is what stops the auditor's prose from
# being silently "improved".
VERBATIM_SHA256 = "c440b4a8755bbb7cb5863470d971b2d7d13b3c36f5a38915730decb0fb184dd7"

REQUIRED_IDS = tuple(f"C-{i}" for i in range(1, 21)) + tuple(f"S-{i}" for i in range(1, 5))
MIN_BODY_CHARS = 200  # a heading with a stub under it is not a body


def _doc() -> str:
    return DOC.read_text()


def verbatim_region(text: str) -> str:
    assert BEGIN in text, "the BEGIN marker is missing — the boundary is gone"
    assert END in text, "the END marker is missing — the boundary is gone"
    return text.split(BEGIN, 1)[1].split(END, 1)[0]


def _bodies(region: str) -> dict[str, str]:
    """{id: body}. C-N are '## C-N · …' headings; S-N are bold list entries in the SUSPECTED
    section. Both shapes are the AUDITOR'S, not ours, so the parser adapts to the file rather
    than the file being reshaped to suit the parser."""
    out: dict[str, str] = {}
    heads = list(re.finditer(r"^## (C-\d+)\s*·", region, flags=re.M))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(region)
        out[m.group(1)] = region[m.end() : end]
    for m in re.finditer(r"^- \*\*(S-\d+)\s*—(.*?)(?=^- \*\*S-\d+|\Z)", region, flags=re.M | re.S):
        out[m.group(1)] = m.group(2)
    return out


def test_the_boundary_markers_exist_and_disposition_is_outside_them():
    text = _doc()
    region = verbatim_region(text)
    assert "PART 2 — OUR DISPOSITION" in text
    assert "PART 2 — OUR DISPOSITION" not in region, (
        "our disposition leaked INSIDE the verbatim region — a re-auditor would read our "
        "verdicts as the auditor's findings"
    )
    assert "Disposition" not in region


@pytest.mark.parametrize("fid", REQUIRED_IDS)
def test_every_finding_has_a_real_body(fid):
    bodies = _bodies(verbatim_region(_doc()))
    assert fid in bodies, f"{fid} has NO body in the verbatim region"
    assert len(bodies[fid].strip()) >= MIN_BODY_CHARS, (
        f"{fid} has a heading but only {len(bodies[fid].strip())} chars under it — a stub, "
        "not the auditor's finding"
    )


def test_all_twenty_four_are_present_and_none_extra_was_invented():
    bodies = _bodies(verbatim_region(_doc()))
    assert set(bodies) == set(REQUIRED_IDS), (
        f"missing {sorted(set(REQUIRED_IDS) - set(bodies))}, "
        f"unexpected {sorted(set(bodies) - set(REQUIRED_IDS))}"
    )


def test_the_verbatim_region_is_byte_identical_to_what_was_recovered():
    got = hashlib.sha256(verbatim_region(_doc()).encode()).hexdigest()
    assert got == VERBATIM_SHA256, (
        "the verbatim audit text CHANGED. If this was a deliberate re-recovery, re-pin "
        f"VERBATIM_SHA256 to {got} in a reviewable commit. If it was a tidy-up, a typo fix or a "
        "line-number correction, revert it — the instruction is that the auditor's prose is not "
        "ours to edit, and drifted references are annotated BESIDE the text, never inside it."
    )


def test_the_disposition_table_covers_every_finding():
    """PART 2 must dispose of all 24 — a body with no verdict is a finding we stopped tracking."""
    table = _doc().split("PART 2 — OUR DISPOSITION", 1)[1]
    missing = [f for f in REQUIRED_IDS if not re.search(rf"\|\s*{re.escape(f)}\s*\|", table)]
    assert not missing, f"no disposition row for {missing}"
