"""Judge a claim at its own site, never by whether a page mentions a string.

★ THIS MODULE EXISTS BECAUSE ONE MISTAKE WAS MADE FIVE TIMES IN A ROW, BY ME, IN GUARDS WHOSE
WHOLE PURPOSE WAS TO CATCH THAT KIND OF MISTAKE.

Every time a defect gets corrected, the correction NAMES the thing it retired — that is what makes
a correction legible. So the corrected document contains the retired string, and a guard written
as ``assert BAD not in text`` fires on the fix, while ``assert GOOD in text`` passes on a page that
merely discusses the topic. The five instances, all in this repository, all mine:

1. a tilde required "somewhere in the file", satisfied by the sentence EXPLAINING the tilde;
2. ``cell 4`` required within 200 characters, satisfied by an unrelated neighbouring sentence;
3. ``weights_only=False`` banned as a string, firing on the sentence WARNING against it;
4. ``the spec wins`` banned as a string, firing on the line quoting it as superseded;
5. ``deliberately not started here`` banned as a string, firing on the docstring that quotes it in
   order to say it was false.

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
