"""The two gitignored evidence trees must each be represented by a TRACKED mirror.

WHAT THIS WAS, AND WHY ONLY PART OF IT SURVIVES. This file began as
``test_paper_artifact_citations.py``: it swept every ``% artifact:`` citation in the manuscript and
required each path to EXIST and be TRACKED. It found the defect it was written for — the sole
artifact behind the λ subsection, present on one disk and committed nowhere — and it caught two
planted failures before anyone believed it.

The manuscript left the repository on 2026-08-19 (60 files, operator decision). With nothing to
scan, the citation sweep had **zero inputs**, and its central assertion — "no cited artifact is
missing" — became TRUE BY VACUITY: nothing cited, therefore nothing missing, therefore green. Its
own fixture-discrimination guard is what refused to let that stand: ``assert len(cited) >= 40``
failed rather than passing against an empty scan, which is the only reason the vacuity was visible
at all. **A check that goes quiet when its input disappears is worse than no check**, so the
vacuous half was deleted rather than kept for its green tick.

WHAT SURVIVES IS INDEPENDENT OF THE MANUSCRIPT. ``runs_cloud/`` (~1.7 GB of checkpoints) and
``processed/`` (the 15 GB lake) are gitignored BY DESIGN — evidence, not source. Anything that
points at them is pointing outside what a clone can verify, so each is allow-listed here and each
must name a tracked MIRROR that itself passes the tracked test. That constraint has nothing to do
with the manuscript: it is what stops the allow-list becoming a hole, and it holds as long as the
two trees are ignored.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Gitignored evidence trees. Each MUST name a tracked mirror, asserted below.
UNTRACKABLE_PREFIXES: dict[str, str] = {
    # ~1.7 GB of checkpoints and per-cell eval JSON pulled off the rented boxes.
    "runs_cloud/": "runs_manifest/m6_cell1_eval_mirror.json",
    # The 15 GB Parquet lake. Its identity is the Merkle root in the tracked manifest.
    "processed/": "runs_manifest/m6_lake_subset_manifest.json",
}


def _tracked() -> set[str]:
    proc = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return set(proc.stdout.split("\n")) - {""}


def test_git_is_the_oracle_here() -> None:
    """Fail loudly rather than skip: without git this file proves nothing and must say so."""
    assert (REPO / ".git").exists(), "no .git — this check cannot run and must not be skipped"
    assert len(_tracked()) > 100, "git ls-files returned an implausibly small set"


def test_the_prefixes_are_still_gitignored() -> None:
    """★ THE PRECONDITION, ASSERTED RATHER THAN ASSUMED — and it is what stops this file going
    vacuous the way its predecessor did. The mirrors exist BECAUSE these trees cannot be tracked.
    If one ever became trackable the exemption would be pointless and this file should be revisited
    rather than quietly continuing to pass."""
    for prefix in UNTRACKABLE_PREFIXES:
        probe = f"{prefix}__probe__.bin"
        r = subprocess.run(["git", "check-ignore", "-q", probe], cwd=REPO)
        assert r.returncode == 0, (
            f"{prefix} is no longer gitignored — the mirror exemption below assumes it is, "
            f"so this file's premise has changed and needs re-deciding, not re-running"
        )


@pytest.mark.parametrize("prefix,mirror", sorted(UNTRACKABLE_PREFIXES.items()))
def test_every_untrackable_prefix_names_a_tracked_mirror(prefix: str, mirror: str) -> None:
    """An exception is only an exception if a reader can still check something."""
    assert mirror in _tracked(), (
        f"{prefix} is allow-listed as untrackable, but its stated mirror {mirror} is not "
        f"tracked — the allow-list would then be a hole rather than a disclosure"
    )


def test_the_mirrors_are_not_empty_placeholders() -> None:
    """A tracked mirror that carries nothing satisfies the letter of the rule and none of it."""
    for prefix, mirror in sorted(UNTRACKABLE_PREFIXES.items()):
        p = REPO / mirror
        assert p.exists(), f"{mirror} is tracked but absent from the worktree"
        assert p.stat().st_size > 1024, (
            f"{mirror} is {p.stat().st_size} bytes — too small to mirror {prefix} meaningfully"
        )
