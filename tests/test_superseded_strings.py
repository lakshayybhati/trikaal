"""A tracked list of superseded claims, and a test that fails if any of them comes back.

WHY THIS IS AN INSTRUMENT AND NOT A HABIT. The same defect has now taken four forms in one thread,
and each time the fix reached the site an audit pointed at and stopped:

  prose → generators      a correction landed in the text and not in the code that emits the text
  title → figures         fig12 kept "Duplicates Price" through the title fix
  sources → exports       the sweep covered .tex and stopped at main_full.txt and the review PDFs
  paper → shipping        README, the model card, the release script and its receipt all still
                          said "duplicates price" after the paper was corrected

The operational rule the fourth one produced is the one this file mechanises:
**GREP THE SUPERSEDED STRING, NOT THE CORRECTED FILE.** Re-reading what you changed can only
confirm the change. Only the OLD string finds where it survives — and all four instances above were
found that way, none by re-reading a fix.

THE FAILURE MODE OF A TOOL LIKE THIS IS ITS ALLOW-LIST, so the allow-list here is built to rot
loudly. Entries are keyed on (pattern, path), never on path alone — allow-listing a FILE would have
silenced the paper's abstract along with the prereg's record of the correction. Every entry carries
a reason, and every entry is asserted to STILL MATCH: when a listed hit is fixed, this file FAILS
and tells you to delete the line. A list that can only grow is a list that becomes a permanent
silence.

★ AND THE FIRST DRAFT OF THIS FILE WAS THE BOY WHO CRIED WOLF, which is the tooling rule arriving
on schedule. Broad patterns fired on eleven CORRECT usages: ``21,301,248-param gate`` and
``the 21,301,248-param pin`` name a real pinned quantity, and ``is NOT externally validated`` is
the correction itself. An instrument whose output must be manually filtered is one nobody reads.
Two fixes, both borrowed from defects found earlier in this same thread: a POLARITY GUARD, so a
negated statement never counts as an assertion of the thing it negates (the fact-verifier's
negation vector, generalized), and patterns narrowed to the phrasing that is actually wrong —
``21.3M-parameter MODEL``, not every mention of the backbone count. The instrument's own
false-positive rate is a property to measure, not to tolerate.

WHAT IS BLOCKING AND WHAT IS REPORTED. ``paper/`` belongs to the writer, so its surviving hits are
recorded in ``EXPECTED_HITS`` with a routing note rather than treated as build failures the builder
cannot fix. Everything in the builder's domain is blocking. The distinction is about who can act,
not about which is more important — the highest-value hit this instrument has found to date is in
``paper/main.tex``, and it is in the abstract.

★ THE KNOWN BLIND SPOT, AND WHERE IT IS COVERED. This file matches SUPERSEDED STRINGS. It therefore
catches verbatim survivals and near-misses of the wording it lists, and it does NOT catch a retired
claim RESTATED IN DIFFERENT WORDS. That limit nearly bit on its first outing: the abstract said
"duplicates what price already carries", and the only reason ``duplicates_price`` matched is that
its pattern happens to carry an optional ``what`` — design and luck in roughly equal measure.
``paper/check_claim_drift.py`` (the writer's) is the stronger construction for that failure mode:
each of its rules pairs a CLAIM pattern with a FORBIDDEN pattern and fires on any sentence matching
both, which is a statement about meaning rather than about wording. The two are complementary and
neither subsumes the other — a string sweep sees restatements the claim/forbidden pairing has no
rule for, and the pairing sees paraphrases a string sweep cannot. Read both before concluding a
retired claim is gone.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------------
# THE SUPERSEDED CLAIMS. One entry per retired statement, with what replaced it and when.
# --------------------------------------------------------------------------------------------
SUPERSEDED: dict[str, dict[str, str]] = {
    "duplicates_price": {
        # prereg §7 v1.6.43 — the title correction. The surviving channels co-vary with VOLUME.
        "pattern": r"duplicat\w*(\s+what)?\s+(the\s+)?price",
        "correct": "keeps the microstructure that DUPLICATES OHLCV and drops the SIGNED CHANNELS",
        "why": (
            "the four magnitude dims (trade_count, mean_trade_size, trade_size_dispersion, "
            "large_trade_share) co-vary with log_volume/log_amount, not with price; the two "
            "SIGNED dims carry 97.3% of the shortfall"
        ),
    },
    "price_correlated_channel": {
        # PARTICIPLE AND VERB FORMS ONLY. "micro--price correlation = 0.2037" is a MEASURED
        # quantity, not the superseded framing, and an earlier pattern flagged it — the difference
        # between describing a channel as price-correlated and reporting a correlation with price.
        "pattern": r"price[- ]correlated|weakly[- ]price"
        r"|(weakly[ -])?co-?vari(es|ant|ance)\s+with\s+(the\s+)?price",
        "correct": "co-vary with VOLUME (or, of the fixture: the correlated FILLER dims)",
        "why": (
            "same correction, in the adjectival form. In the SYNTHETIC FIXTURE the fillers really "
            "are correlated with the return channel, so this phrasing was accurate there and "
            "still seeded the wrong generalization to real data — the receipts call them fillers"
        ),
    },
    "backbone_is_the_model_size": {
        # §7 v1.6.17/v1.6.34 — 21,301,248 is backbone-EXCLUDING-MTP, a NAMED SECONDARY. Naming
        # that quantity is correct ("the 21,301,248-param pin", "params EXCLUDING the MTP heads");
        # calling it the MODEL is not. The pattern targets only the second, because a pattern that
        # fires on both makes the instrument unreadable.
        # One optional intervening word: "21,301,248-parameter MEASUREMENT instrument" is the exact
        # phrasing the paper retracts, and the tighter form missed it. "backbone" is deliberately
        # NOT in the alternation -- calling 21,301,248 the backbone is correct.
        "pattern": r"(21[,.]?301[,.]?248|21\.3\s?M)[- ]param\w*\s+(\w+\s+)?(model|instrument|decoder)",
        "correct": "31,795,200 realized total (21,301,248 is the backbone excluding MTP heads)",
        "why": "quoting the backbone as the model size understates the shipped checkpoint by 33%",
    },
    "externally_validated": {
        # Invariant 4 / prereg §7 v1.6.22 — the gate was dropped as binding, never executed.
        "pattern": r"externally\s+validated",
        "correct": "our own BSQ baseline at matched bits-per-token; the §8.C.3 gate was dropped "
        "as binding on 2026-08-03 and never executed",
        "why": "no Kronos weights were ever pulled; invariant 8 forbids it",
    },
    "mde_z_formula": {
        # §7 v1.5 item B — t-quantiles at Welch-Satterthwaite nu over SE_total.
        "pattern": r"\(z0?\.95\s*\+\s*z0?\.80\)\s*[·*x]\s*SE_boot",
        "correct": "(t_0.95,nu + t_0.80,nu) x SE_total, built by paired_bootstrap.mde_terms",
        "why": "the pre-v1.5 MDE shipped inside clause 2's rule string for a month",
    },
}

# --------------------------------------------------------------------------------------------
# WHERE A SUPERSEDED STRING IS LEGITIMATE. Keyed on (claim, path prefix) — NEVER on path alone.
# --------------------------------------------------------------------------------------------
#
# ★ THE RESIDUAL RISK, STATED RATHER THAN HIDDEN. Each entry silences one claim in one file, and
# nothing here can distinguish a NEW wrong usage added to such a file from the old one it was
# opened for. That is a real hole and it is the price of keeping historical records readable. It is
# narrowed two ways: entries are (claim, path) rather than path, and each carries a `requires`
# marker that must be present in the file — so an annotation cannot be deleted while the silence
# survives it.
RECORD_FILES: dict[tuple[str, str], tuple[str, str]] = {}


def _record(claim: str, path: str, reason: str, requires: str) -> None:
    RECORD_FILES[(claim, path)] = (reason, requires)


for _c in SUPERSEDED:
    _record(_c, "docs/m6_prereg.md", "the amendment log IS the record of each correction", "§7")
    _record(_c, "docs/BUILD_RECORD.md", "quotes what was wrong in order to date it", "#")
    _record(
        _c, "tests/test_superseded_strings.py", "this file names its own patterns", "SUPERSEDED"
    )

for _c in ("price_correlated_channel", "backbone_is_the_model_size"):
    _record(
        _c,
        "docs/paper_skeleton.md",
        "a PRE-RUN document under a provisional-framing banner, annotated with what the gate "
        "later measured rather than edited into agreement with its own result",
        "NOTE (added 2026-08-17)",
    )
for _p in ("docs/m6_decisions_pending.md", "docs/m6_training_budget_decision.md"):
    _record(
        "backbone_is_the_model_size",
        _p,
        "records the candidate PAPER SENTENCES offered at a decision point; editing a recorded "
        "option into correctness would destroy the record of what was actually on the table",
        "PARAMETER-COUNT NOTE",
    )
_record(
    "backbone_is_the_model_size",
    "docs/superpowers/specs/2026-06-18-trikaal-v1-design.md",
    "the design record, using the design-time target; annotated with the realized totals",
    "PARAMETER-COUNT NOTE",
)
for _c in SUPERSEDED:
    _record(
        _c,
        "paper/check_claim_drift.py",
        "the writer's own drift checker, which names the retired claims it searches for — the "
        "same exemption this file holds for itself, and read the note in the module docstring "
        "about which of the two instruments is the stronger design",
        "RETIRED CLAIMS",
    )

# --------------------------------------------------------------------------------------------
# SURVIVING HITS OUTSIDE THE BUILDER'S DOMAIN. Exact — fixing one must FAIL this file.
# --------------------------------------------------------------------------------------------
EXPECTED_HITS: dict[tuple[str, str], str] = {}
# ★ EMPTY, AND IT WAS EMPTIED BY THE ROT-CHECK RATHER THAN BY ANYONE REMEMBERING. On 2026-08-17
# this list held five routed hits in paper/: the ABSTRACT (`main.tex` said "duplicates what price
# already carries" while §1 and §6 had been corrected to OHLCV), §6's allocation claim, §7.2's
# placebo sentence, and two lower-confidence "weakly covariant with price" sites I flagged as the
# writer's judgement rather than as defects. The writer fixed ALL FIVE — including both
# lower-confidence ones, and generalized rather than substituting: "weakly covariant with the
# high-variance channels already being coded", "with the rest", "with OHLCV shape".
#
# The mechanism is the point. Nobody re-read this file; the rot-check went red on the next run and
# named each entry to delete. A list that can only grow is a permanent silence, and this is what
# the alternative looks like in practice.
#
# paper/main_full.txt held a sixth entry for about an hour before the writer deleted the export.
# NOTE ON WHAT IS NOT LISTED: paper/main_full.txt carried a fourth hit when this file was first
# written and the writer deleted the export before it was committed. The rot-check below is what
# surfaced that within one run, which is the property the list is built for.

_SKIP_DIRS = (".git/", ".venv/", "node_modules/", "__pycache__/")
_EXTS = (".md", ".py", ".json", ".tex", ".txt", ".yaml", ".yml", ".toml")

# ★ THE POLARITY GUARD. "is NOT externally validated" contains "externally validated" and is the
# CORRECTION, not the defect — the identical containment-inverts-meaning vector that made the fact
# verifier confirm a claim against its own negation. A line whose match is immediately preceded by
# a negation, or which is visibly narrating the correction, is not an assertion of the superseded
# claim. Kept deliberately tight: it looks only just before the match, so it cannot excuse a
# sentence that merely mentions "not" somewhere else.
_NEGATION_BEFORE = re.compile(
    r"(?:\bnot\b|\bnever\b|\bno longer\b|\bwas\b|\bused to\b|\bformerly\b|\bsuperseded\b"
    r"|\bwithdrawn\b|\bdropped\b|\bfalse\b|\bincorrect\b|\bwrong\b|\bstale\b)[\s\W]*$",
    re.I,
)
# Prose that is explicitly ABOUT the correction. Narrower than "contains a negation anywhere".
_NARRATES_CORRECTION = re.compile(
    r"used to (read|say)|superseded|corrected (to|at|in)|the (old|previous|former) wording"
    r"|this (line|table|string|docstring) said|NOT externally valid|pre-v1\.5|the defect"
    r"|specified as|GATE DROPPED|\bCLAIMED:|dropped as binding|never executed",
    re.I,
)


def _asserts_it(line: str, m: re.Match) -> bool:
    """Is this line ASSERTING the superseded claim, or narrating that it was wrong?"""
    if _NARRATES_CORRECTION.search(line):
        return False
    return not _NEGATION_BEFORE.search(line[: m.start()])


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return [
        p for p in out if p and p.endswith(_EXTS) and not any(p.startswith(d) for d in _SKIP_DIRS)
    ]


def scan(paths: list[str] | None = None) -> dict[tuple[str, str], list[str]]:
    """{(claim, path): [matching excerpts]} over every tracked text file.

    ★ SCANNED OVER THE WHOLE TEXT, NOT LINE BY LINE, AND THAT IS NOT A DETAIL. The first version
    matched per line and silently missed every claim that WRAPS — which in a repository whose
    markdown wraps at 100 characters is most of them. ``the compute-optimal point for a
    21.3M-parameter\\n  model`` is a live superseded claim that a line-based scanner cannot see,
    because ``\\s+`` in the pattern has to cross the newline. A sweep that cannot see wrapped prose
    is a sweep that reports clean on the majority of a documentation tree.
    """
    compiled = {k: re.compile(v["pattern"], re.I) for k, v in SUPERSEDED.items()}
    found: dict[tuple[str, str], list[str]] = {}
    for rel in paths if paths is not None else _tracked_files():
        try:
            text = (REPO / rel).read_text(errors="replace")
        except OSError:
            continue
        for claim, rx in compiled.items():
            if (claim, rel) in RECORD_FILES:
                continue
            hits = []
            for m in rx.finditer(text):
                # the surrounding sentence, flattened, is what polarity is judged on
                lo = max(0, text.rfind("\n", 0, max(0, m.start() - 200)) + 1)
                hi = text.find("\n", m.end() + 120)
                ctx = " ".join(text[lo : hi if hi != -1 else len(text)].split())
                cm = rx.search(ctx)
                if cm is not None and _asserts_it(ctx, cm):
                    hits.append(ctx[:170])
            if hits:
                found[(claim, rel)] = hits
    return found


def test_the_scanner_actually_matches_something() -> None:
    """FIXTURE DISCRIMINATION — a regex that matches nothing passes every assertion below."""
    compiled = {k: re.compile(v["pattern"], re.I) for k, v in SUPERSEDED.items()}
    probes = {
        "duplicates_price": "the tokenizer keeps microstructure that duplicates price",
        "price_correlated_channel": "a low-variance, weakly-price-correlated channel",
        "backbone_is_the_model_size": "the 21,301,248-parameter measurement instrument",
        "externally_validated": "our BSQ baseline, externally validated against Kronos-small",
        "mde_z_formula": "MDE_paired = (z0.95+z0.80)·SE_boot",
    }
    assert set(probes) == set(SUPERSEDED), "every superseded claim needs a discriminating probe"
    for claim, sentence in probes.items():
        assert compiled[claim].search(sentence), f"{claim!r} no longer matches its own defect"
    # and each must NOT match its own replacement, or the check would fire on the fix
    for claim, spec in SUPERSEDED.items():
        assert not compiled[claim].search(spec["correct"]), (
            f"{claim!r} matches its own CORRECTED wording — it would fail on the fix"
        )


def test_the_patterns_do_not_fire_on_correct_usages() -> None:
    """★ THE FALSE-POSITIVE FLOOR, MEASURED. An instrument nobody trusts is worse than none.

    Every string below is a CORRECT sentence that the first draft of this file flagged. If any of
    them fires again, the patterns have broadened back and the output will need manual filtering —
    which is how a gate becomes decoration.
    """
    compiled = {k: re.compile(v["pattern"], re.I) for k, v in SUPERSEDED.items()}
    correct = {
        "backbone_is_the_model_size": [
            "21,301,248-param gate enforced, canonical geometry (batch 32 x seq 512)",
            "V_C, V_F = 891, 1225  # canonical FSQ vocab (the 21,301,248-param pin)",
            "Kronos_small backbone (8L / d512 / ff1024 / 8h; 21,301,248 params EXCLUDING the MTP)",
        ],
        "price_correlated_channel": [
            r"placebo: micro--price correlation & $0.2037 \to 0.0005$ & measured",
        ],
    }
    for claim, sentences in correct.items():
        for s in sentences:
            m = compiled[claim].search(s)
            assert m is None or not _asserts_it(s, m), (
                f"{claim!r} fires on a CORRECT usage: {s!r}. Narrow the pattern rather than "
                f"widening the allow-list."
            )
    # the polarity guard specifically: the correction must never read as the defect
    for s in (
        "The BSQ baseline is our own and is not externally validated.",
        "| 1 | BSQ | ohlcv | our BSQ baseline - NOT externally valid. |",
    ):
        m = compiled["externally_validated"].search(s)
        assert m is None or not _asserts_it(s, m), f"polarity guard failed on {s!r}"


def test_the_scan_reaches_the_tree() -> None:
    files = _tracked_files()
    assert len(files) > 100, f"scanner saw only {len(files)} tracked text files"
    assert any(f == "README.md" for f in files)
    assert any(f.startswith("paper/") for f in files), "paper/ must be in scope, read-only"


def test_no_superseded_claim_survives_in_the_builders_domain() -> None:
    """BLOCKING. Anything outside paper/ is mine and must carry the corrected wording."""
    live = {k: v for k, v in scan().items() if not k[1].startswith("paper/")}
    unexpected = {k: v for k, v in live.items() if k not in EXPECTED_HITS}
    if unexpected:
        lines = []
        for (claim, path), hits in sorted(unexpected.items()):
            spec = SUPERSEDED[claim]
            lines.append(f"  {path}  [{claim}]\n    correct wording: {spec['correct']}")
            lines += [f"      > {h}" for h in hits[:3]]
        raise AssertionError(
            "SUPERSEDED CLAIMS ARE LIVE in files the builder owns:\n" + "\n".join(lines)
        )


@pytest.mark.parametrize("key", sorted(EXPECTED_HITS))
def test_each_routed_hit_is_still_there_or_the_list_is_stale(key: tuple[str, str]) -> None:
    """The allow-list must ROT LOUDLY. A fixed hit fails here so the entry gets deleted.

    Without this, ``EXPECTED_HITS`` becomes a permanent silence — which is precisely how a
    superseded string survives four rounds of correction.
    """
    claim, path = key
    if not (REPO / path).exists():
        pytest.fail(f"{path} no longer exists — delete its EXPECTED_HITS entries")
    rx = re.compile(SUPERSEDED[claim]["pattern"], re.I)
    text = (REPO / path).read_text(errors="replace")
    assert rx.search(text), (
        f"GOOD NEWS, AND AN ACTION: {path} no longer contains the {claim!r} wording. "
        f"Delete its entry from EXPECTED_HITS in {Path(__file__).name} — the list is only "
        f"honest while it is exact."
    )


@pytest.mark.parametrize("key", sorted(RECORD_FILES))
def test_each_record_exemption_is_still_earned(key: tuple[str, str]) -> None:
    """A record exemption must still describe a real file that still carries its marker.

    Two ways this rots and both are covered: the file stops containing the superseded string (the
    exemption is dead weight and hides the next one), or its ANNOTATION is deleted while the
    exemption survives it (the silence outlives the thing that justified it).
    """
    claim, path = key
    reason, requires = RECORD_FILES[key]
    p = REPO / path
    if not p.exists():
        pytest.fail(f"{path} no longer exists — delete its RECORD_FILES entry ({reason})")
    text = p.read_text(errors="replace")
    if not re.search(SUPERSEDED[claim]["pattern"], text, re.I):
        pytest.skip(f"{path} no longer contains the {claim!r} wording at all")
    assert requires in text, (
        f"{path} still carries the {claim!r} wording, but its justifying marker {requires!r} is "
        f"GONE. The exemption was granted because: {reason}. Restore the annotation or remove the "
        f"exemption — a silence must not outlive its reason."
    )
