"""Judge a claim at its own site, never by whether a page mentions a string.

★ THIS MODULE EXISTS BECAUSE ONE MISTAKE WAS MADE FIVE TIMES IN A ROW, BY ME, IN GUARDS WHOSE
WHOLE PURPOSE WAS TO CATCH THAT KIND OF MISTAKE.

Every time a defect gets corrected, the correction NAMES the thing it retired — that is what makes
a correction legible. So the corrected document contains the retired string, and a guard written
as ``assert BAD not in text`` fires on the fix, while ``assert GOOD in text`` passes on a page that
merely discusses the topic. ★ FIVE INSTANCES IN **TWO OPPOSITE MODES**, and that distinction was itself a correction to my
own account of it. I first described these as one failure — checks firing on the fix — and that
covered only three of them. Repairing only the false positives would have left two checks that
CANNOT FAIL, which is the worse half: a noisy guard gets investigated, a silent one never does.

MODE A — passes vacuously (the check cannot fail):

1. a tilde required "somewhere in the file", satisfied by the sentence EXPLAINING the tilde;
2. ``cell 4`` required within 200 characters, satisfied by an unrelated neighbouring sentence.

MODE B — fires on the correction (the check cannot pass):

3. ``weights_only=False`` banned as a string, firing on the sentence WARNING against it;
4. ``the spec wins`` banned as a string, firing on the line quoting it as superseded;
5. ``deliberately not started here`` banned as a string, firing on the docstring that quotes it in
   order to say it was false.

6. ``tests/test_no_routable_addresses.py`` flagged ITSELF: its docstring quotes CUDA versions that
   parse as addresses, and its negative control must plant a REAL routable address (RFC-5737
   documentation ranges report ``is_global=False``, which would make the control vacuous). Mode B
   again, in the file whose whole subject is this distinction — and it could not see itself until
   it was committed, because the sweep walks ``git ls-files``.

Both modes have one cause — the document was the unit of judgement instead of the claim — so both
are fixed by the same move, and a repair that addresses only Mode B leaves Mode A intact.

★ AND A GUARD THAT EXEMPTS ITSELF MUST CHECK THE EXEMPTION. Instance 6 is exempt from its own
sweep, so it asserts that every routable value it carries sits in a docstring or a test body and
never in a data field, and that removing the exemption WOULD have caught it. An unchecked
self-exemption is just Mode A with extra steps.

The discipline is the same in all five: a mention is judged by the PARAGRAPH it sits in, or by the
code block it sits in, and never by the document as a whole. A mention whose own paragraph retires
it is a record, not a claim.
"""

from __future__ import annotations

import re

#: Words that turn a mention into a record, a refusal, or a correction.
RETIRED = re.compile(
    r"previously|used to|no longer|never existed|never happened|was cited|old block|superseded|"
    r"forbid|must not|never|dropped|not in force|corrected|false|hazard|unexecutable|instead of|"
    r"this line read|read 128|opposite",
    re.I,
)


def paragraph_at(text: str, pos: int) -> str:
    """The blank-line-delimited paragraph containing ``pos``."""
    lo = text.rfind("\n\n", 0, pos) + 2
    hi = text.find("\n\n", pos)
    return text[lo : hi if hi != -1 else len(text)]


def live_mentions(
    text: str, pattern: str | re.Pattern[str], *, retired: re.Pattern[str] = RETIRED
) -> list[str]:
    """Occurrences of ``pattern`` whose own paragraph does NOT retire them.

    Returns the offending paragraphs (truncated), so a failure message shows the claim rather than
    just its offset.
    """
    pat = re.compile(pattern, re.I) if isinstance(pattern, str) else pattern
    out: list[str] = []
    for m in pat.finditer(text):
        para = paragraph_at(text, m.start())
        if not retired.search(para):
            out.append(para.strip()[:220])
    return out


def code_blocks(md: str, lang: str = "python") -> list[str]:
    """The fenced ``lang`` blocks only — a claim about a recipe is a claim about its code."""
    return re.findall(rf"```{lang}\n(.*?)```", md, re.S)
