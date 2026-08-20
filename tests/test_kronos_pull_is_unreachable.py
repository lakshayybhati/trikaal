"""AN AGENT ASKED TO "FINISH THE EXTERNAL-VALIDATION GATE" MUST ARRIVE AT "NO".

THE HAZARD, WHICH WAS A CHAIN AND NOT A TYPO. ``docs/ENGINEERING.md`` used to end its sources-of-
truth list with *"if anything here conflicts with the spec, the spec wins"*. The spec is a FROZEN
pre-work design record, and at §8.C.3 it lays out a step-by-step protocol for **downloading
published Kronos weights into the eval harness** — the exact thing invariant 8 calls
non-negotiable and forbids. So an agent that did what the project told it to do would follow the
stated precedence to the spec, pull third-party weights, and break the invariant while believing
it was obeying the project. Four further documents ordered or detailed that pull with no banner,
including one that says the weights "load anyway" in M6, in the present tense.

The precedence is now inverted and every surviving description of the pull carries a superseded
banner. This file holds that shut.

★ CHECKED AT THE CLAIM SITE. A mention of the pull is not an instruction to perform it — the
correction notices, the prohibition itself, and this docstring all contain the words. So a mention
is judged by its OWN paragraph (does it negate?) or by whether a banner precedes it in the file,
never by the document merely containing a disclaimer somewhere. That distinction is the reason the
first three guards written in this repo passed when they should have failed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENGINEERING = REPO / "docs/ENGINEERING.md"
BANNER = "★ SUPERSEDED"

#: A description of obtaining Kronos's published weights, in either word order.
PULL = re.compile(
    r"(download|pull|load)[^.\n]{0,60}\bKronos[^.\n]{0,40}\bweights"
    r"|\bKronos[^.\n]{0,40}\bweights[^.\n]{0,60}(download|pull|load)",
    re.I,
)
#: Words that make a paragraph a REFUSAL or a RECORD rather than an instruction.
NEGATED = re.compile(
    r"never|no Kronos|forbid|superseded|dropped|not in force|must not|corrected|false|"
    r"unexecutable|hazard|was found",
    re.I,
)


def _docs() -> list[Path]:
    return [*sorted(REPO.joinpath("docs").rglob("*.md")), REPO / "README.md"]


def _bare_mentions(text: str) -> list[str]:
    """Pull descriptions that neither negate themselves nor sit under a banner."""
    out = []
    for m in PULL.finditer(text):
        lo = text.rfind("\n\n", 0, m.start()) + 2
        hi = text.find("\n\n", m.end())
        para = text[lo : hi if hi != -1 else len(text)]
        if NEGATED.search(para) or BANNER in text[: m.start()]:
            continue
        out.append(para.strip()[:200])
    return out


# ── the detector has teeth ────────────────────────────────────────────────────────────────────
def test_NEGATIVE_CONTROL_a_bare_instruction_is_detected() -> None:
    """Fixture discrimination: without this, a broken regex would make every test below pass."""
    bad = "# doc\n\nStep 1: download the published Kronos-small weights into the eval harness.\n"
    assert _bare_mentions(bad), "the detector cannot see a bare pull instruction"


def test_NEGATIVE_CONTROL_a_negated_mention_is_not_flagged() -> None:
    """And it must not fire on the prohibition, or it becomes noise and gets ignored."""
    ok = "# doc\n\nNo Kronos weights are ever pulled; downloading Kronos weights is forbidden.\n"
    assert not _bare_mentions(ok)


def test_the_corpus_actually_contains_pull_mentions() -> None:
    """Anti-vacuity: if the phrase vanished entirely, the sweep would pass by absence."""
    total = sum(len(PULL.findall(f.read_text())) for f in _docs())
    assert total >= 5, (
        f"only {total} pull mentions found — is this sweep still looking at anything?"
    )


# ── the sweep ─────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("doc", _docs(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_document_carries_a_live_kronos_pull_instruction(doc: Path) -> None:
    bare = _bare_mentions(doc.read_text())
    assert not bare, f"{doc.relative_to(REPO)} describes pulling Kronos weights unqualified: {bare}"


# ── the precedence chain itself ───────────────────────────────────────────────────────────────
def test_the_spec_no_longer_outranks_the_invariants() -> None:
    """★ AND THIS TEST ITSELF FELL INTO THE TRAP IT GUARDS. The first version asserted
    ``"the spec wins" not in text`` and failed — because the corrected document QUOTES the retired
    sentence in order to explain why it was dangerous. A retired directive and a description of a
    retired directive are not the same string's two meanings; they are two different claims that
    happen to share a string. Judged by paragraph, as everywhere else in this file."""
    t = ENGINEERING.read_text()
    live = []
    for m in re.finditer(r"the spec wins", t, re.I):
        lo = t.rfind("\n\n", 0, m.start()) + 2
        hi = t.find("\n\n", m.end())
        para = t[lo : hi if hi != -1 else len(t)]
        if not re.search(r"previously read|used to read|superseded|hazard|opposite", para, re.I):
            live.append(para.strip()[:200])
    assert not live, f"'the spec wins' survives as a live directive: {live}"

    inv = t.lower().find("non-negotiable invariants")
    spec = t.find("Blueprint spec (design)")
    assert -1 < inv < spec, (
        f"the invariants must be ranked above the blueprint spec (inv@{inv} spec@{spec})"
    )


def test_the_answer_is_written_down_rather_than_left_to_be_derived() -> None:
    t = ENGINEERING.read_text()
    assert re.search(r"may I download Kronos weights\?\*{0,2}\s*\*\*No\.\*\*", t), (
        "ENGINEERING.md no longer answers the question outright"
    )


def test_the_gate_is_not_binding_in_code() -> None:
    """The prose must agree with the executable state, not merely assert it."""
    from trikaal.eval.external_validation import GATE_IS_BINDING

    assert GATE_IS_BINDING is False


def test_the_document_whose_subject_IS_the_pull_says_so_at_the_top() -> None:
    """``m6_c4_kronos_gate_requirements.md`` is 36 mentions of Kronos and a HuggingFace pull
    recipe. Its section-level banner already precedes the recipe, so the sweep above stays green
    without this — that redundancy was measured by mutation, not assumed. It is guarded anyway
    because an agent skimming a document titled "what implementing G-§8.C.3 actually requires"
    decides whether to keep reading from the first screen, not from §3.1."""
    doc = REPO / "docs/m6_c4_kronos_gate_requirements.md"
    head = doc.read_text()[:1500]
    assert BANNER in head, "the requirements doc no longer refuses at the top"
    assert "DO NOT EXECUTE" in head.upper(), "the top banner no longer says not to act on it"
