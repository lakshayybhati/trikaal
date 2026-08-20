"""THE THREE FILES AN AGENT READS FIRST DESCRIBED A PROJECT THAT ENDED THREE MILESTONES AGO.

``README.md`` listed the Binance lake, the eval harness, the generation path and full-corpus
training under "Deferred to later milestones ... explicitly out of this slice" — sixty-seven lines
below a paragraph stating the 200-symbol lake was complete. ``src/trikaal/__init__.py`` said the
current milestone was "the synthetic vertical slice ... before any real Binance ingest".
``src/trikaal/eval/__init__.py`` said the package held "ONLY" the M2 IC screen and that the purged
walk-forward harness was "deliberately not started here" — sitting beside sixteen other modules,
including ``harness.py`` and ``folds.py``, which are that harness.

THE FAILURE IS NOT UNTIDINESS. An agent asked to "add a purged walk-forward backtest" reads the
package's own docstring, believes the subsystem absent, and writes a second one. Present tense is
what makes it dangerous: nothing in those sentences signalled that they were a snapshot.

★ CITATIONS ARE JUDGED AT THE CLAIM SITE. ``BackboneOutput.kv_cache`` was cited in the README and
exists nowhere in ``src/`` — but the replacement text SAYS SO, in a sentence that names it in
order to retire it. A sweep that flagged every occurrence of the string would fire on the
correction. So each citation is judged by whether its own sentence negates it, exactly as the
Kronos-pull sweep does.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
from tests.claim_site import live_mentions

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"

#: Modules a README-cited attribute could plausibly live on.
LOOKUP = (
    "trikaal.model.predictor",
    "trikaal.model.attention",
    "trikaal.tokenizer.model",
    "trikaal.eval.harness",
    "trikaal.demo.inference",
)
#: Dotted names that are data paths (JSON keys, manifest fields), not Python attributes.
NOT_PYTHON = {"draw.drawn_by_symbol_stage1"}
NEGATED = re.compile(r"never existed|used to|previously|no longer|old block|was cited", re.I)


def _dotted_names(text: str) -> list[re.Match[str]]:
    out = []
    for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`", text):
        n = m.group(1)
        if "/" in n or n.endswith((".py", ".md", ".json", ".lock", ".toml")) or n in NOT_PYTHON:
            continue
        out.append(m)
    return out


def _resolves(dotted: str) -> bool:
    try:
        importlib.import_module(dotted)
        return True
    except ImportError:
        pass
    head, _, attr = dotted.rpartition(".")
    for mod in LOOKUP:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        obj = getattr(m, head.split(".")[-1], None)
        if obj is not None and hasattr(obj, attr):
            return True
    return False


# ── the detector works ────────────────────────────────────────────────────────────────────────
def test_the_readme_still_cites_apis() -> None:
    """Anti-vacuity: an empty citation list would make the sweep pass by absence."""
    assert len(_dotted_names(README.read_text())) >= 4


def test_NEGATIVE_CONTROL_a_bogus_live_citation_would_be_caught() -> None:
    assert not _resolves("BackboneOutput.kv_cache")
    assert _resolves("TrikaalAR.step"), "the resolver cannot find an API that does exist"


# ── the sweep ─────────────────────────────────────────────────────────────────────────────────
def test_every_live_api_citation_in_the_readme_resolves() -> None:
    text = README.read_text()
    dead = []
    for m in _dotted_names(text):
        lo = text.rfind("\n\n", 0, m.start()) + 2
        hi = text.find("\n\n", m.end())
        para = text[lo : hi if hi != -1 else len(text)]
        if NEGATED.search(para):
            continue  # a citation named in order to retire it
        if not _resolves(m.group(1)):
            dead.append(m.group(1))
    assert not dead, f"README cites APIs that do not exist: {dead}"


# ── the package docstrings ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        ("trikaal", r"current milestone is the \*\*synthetic vertical slice\*\*"),
        ("trikaal", r"before any real Binance ingest"),
        ("trikaal", r"which is the single\s+source of truth"),
        ("trikaal.eval", r"currently holds ONLY"),
        ("trikaal.eval", r"deliberately not\s+started here"),
    ],
)
def test_package_docstrings_do_not_deny_what_is_built(module, forbidden) -> None:
    """Judged per paragraph: both docstrings QUOTE their retired claims in order to explain why
    those claims were dangerous, and a bare `not in` check fires on the fix. See
    ``tests/claim_site.py`` — this is the fifth instance of that mistake and the last one."""
    doc = importlib.import_module(module).__doc__ or ""
    live = live_mentions(doc, forbidden)
    assert not live, f"{module} still claims it, unqualified: {live}"


def test_the_eval_package_names_the_harness_it_used_to_deny() -> None:
    doc = importlib.import_module("trikaal.eval").__doc__ or ""
    for name in ("harness", "folds", "xsection", "verdict"):
        assert name in doc, f"trikaal.eval docstring does not mention {name}"
    n = len(list((REPO / "src/trikaal/eval").glob("*.py")))
    assert n >= 15, f"only {n} eval modules — this docstring's premise has changed"


def test_the_readme_does_not_call_shipped_subsystems_deferred() -> None:
    text = README.read_text()
    m = re.search(r"## Deferred to later milestones", text)
    assert m is None, "the deferred-milestones block is back; four of its entries have shipped"
