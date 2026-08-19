"""Every artifact the paper cites must EXIST and be TRACKED — or be an explicit, mirrored exception.

WHY THIS TEST EXISTS. ``runs_manifest/m6_lambda_sweep.json`` is the sole artifact behind §6.6 (the
once-only λ re-derivation, 0 of 18 clearing). It sat UNTRACKED for the whole life of that
subsection: not gitignored, simply never committed. A fresh clone could not verify the claim, and
nothing in the repository could tell. An independent auditor found it by sweeping the citations in
one command — which is exactly the shape of check that belongs in the suite rather than in an
audit, so this is the class fix and the commit was only the instance.

WHAT IS CHECKED, and why "tracked" is the operative word rather than "exists". A path that exists
on the builder's disk and nowhere else is the defect: it makes every local verification pass while
the published artifact is unverifiable. So the assertion is membership in ``git ls-files``, not
``Path.exists()`` — the two disagreed on precisely one file, and that file was the finding.

THE TWO EVIDENCE TREES THAT CANNOT BE TRACKED. ``runs_cloud/`` (~1.7 GB of checkpoints) and
``processed/`` (the 15 GB lake) are gitignored BY DESIGN — they are evidence, not source. Citing
them is legitimate, but it breaks the citation convention exactly where a reader most wants to
check, so each is an explicit entry in ``UNTRACKABLE_PREFIXES`` and each must name a tracked
MIRROR that itself passes the tracked test. The allow-list therefore cannot quietly grow into a
hole: adding a prefix without a mirror fails this file.

NOTE FOR THE WRITER. This test reads ``paper/`` and never writes it. A new ``% artifact:`` citation
naming a receipt that has not been committed yet will fail here — that is the intended behaviour,
and the fix is to commit the receipt, not to weaken the test.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"

# Repo-relative prefixes that a citation can name. Anything not under one of these (prose, "same
# receipt", inline formulae) is not a path claim and is not checked.
_DIR_PREFIXES = (
    "runs_manifest",
    "docs",
    "src",
    "scripts",
    "tests",
    "paper",
    "config",
    "runs_cloud",
    "processed",
)
# Repo-root files that are cited without a directory component.
_ROOT_FILES = ("docs/ENGINEERING.md", "README.md", "LICENSE", "pyproject.toml", "uv.lock")

# Gitignored evidence trees. Each MUST name a tracked mirror, asserted below.
UNTRACKABLE_PREFIXES: dict[str, str] = {
    # ~1.7 GB of checkpoints and per-cell eval JSON pulled off the rented boxes.
    "runs_cloud/": "runs_manifest/m6_cell1_eval_mirror.json",
    # The 15 GB Parquet lake. Its identity is the Merkle root in the tracked manifest.
    "processed/": "runs_manifest/m6_lake_subset_manifest.json",
}

_PATH = re.compile(
    r"(?:" + "|".join(_DIR_PREFIXES) + r")/[A-Za-z0-9_./-]+|(?:" + "|".join(_ROOT_FILES) + r")"
)


def _cited_paths(root: Path | None = None) -> dict[str, list[str]]:
    """{repo-relative path: [source locations]} over every ``% artifact:`` line under paper/."""
    out: dict[str, list[str]] = {}
    for tex in sorted((root or PAPER).rglob("*.tex")):
        for n, line in enumerate(tex.read_text().splitlines(), start=1):
            if not line.lstrip().startswith("% artifact:"):
                continue
            body = line.split("artifact:", 1)[1]
            for raw in _PATH.findall(body):
                # `verdict.py:519-521` / `features.py:180` — the line suffix is not part of the path
                path = raw.split(":", 1)[0].rstrip(".,;)")
                if path.endswith("/"):
                    continue
                out.setdefault(path, []).append(f"{tex.name}:{n}")
    return out


def _audit(tracked: set[str], root: Path | None = None) -> tuple[list[str], list[str]]:
    """(paths that do not exist, paths that exist but are not committed)."""
    missing: list[str] = []
    untracked: list[str] = []
    for path, where in sorted(_cited_paths(root).items()):
        if any(path.startswith(p) for p in UNTRACKABLE_PREFIXES):
            continue
        if not (REPO / path).exists():
            missing.append(f"{path}  (cited at {', '.join(where)})")
        elif path not in tracked:
            untracked.append(f"{path}  (cited at {', '.join(where)})")
    return missing, untracked


def _tracked() -> set[str]:
    proc = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return set(proc.stdout.split("\n")) - {""}


def test_git_is_the_oracle_here() -> None:
    """Fail loudly rather than skip: without git this file proves nothing and must say so."""
    assert (REPO / ".git").exists(), "no .git — this check cannot run and must not be skipped"
    assert len(_tracked()) > 100, "git ls-files returned an implausibly small set"


def test_the_citation_scan_finds_the_citations() -> None:
    """Fixture discrimination: an extractor that finds nothing would pass every assertion below."""
    cited = _cited_paths()
    assert len(cited) >= 40, f"expected the paper to cite many artifacts, found {len(cited)}"
    # the file that motivated this test must be among them, or the scan has stopped seeing §6.6
    assert "runs_manifest/m6_lambda_sweep.json" in cited


@pytest.mark.parametrize("prefix,mirror", sorted(UNTRACKABLE_PREFIXES.items()))
def test_every_untrackable_prefix_names_a_tracked_mirror(prefix: str, mirror: str) -> None:
    """An exception is only an exception if a reader can still check something."""
    assert mirror in _tracked(), (
        f"{prefix} is allow-listed as untrackable, but its stated mirror {mirror} is not "
        f"tracked — the allow-list would then be a hole rather than a disclosure"
    )


def test_the_check_catches_both_failure_modes(tmp_path: Path) -> None:
    """FIXTURE DISCRIMINATION — the check is run against citations that MUST fail it.

    ``test_every_cited_artifact_is_tracked`` passes when the paper is clean, which is exactly when
    a check that does nothing also passes. So the same ``_audit`` is pointed at a synthetic paper
    citing (a) a receipt that does not exist and (b) one that exists and is not committed. If
    either comes back clean the assertion below is worthless and says so.
    """
    ghost = "runs_manifest/m6_this_receipt_was_never_written.json"
    local_only = "runs_manifest/m6_exists_but_uncommitted.json"
    (REPO / local_only).write_text("{}\n")  # real file, deliberately never added to the index
    try:
        (tmp_path / "fake.tex").write_text(
            f"Some prose.\n% artifact: {ghost} :: a.b\n% artifact: {local_only} :: c.d\n"
        )
        missing, untracked = _audit(_tracked(), root=tmp_path)
        assert any(ghost in m for m in missing), f"a nonexistent citation was not caught: {missing}"
        assert any(local_only in u for u in untracked), (
            f"an uncommitted citation was not caught: {untracked}"
        )
    finally:
        (REPO / local_only).unlink()


def test_every_cited_artifact_is_tracked() -> None:
    missing, untracked = _audit(_tracked())
    assert not missing, "paper cites artifacts that do not exist:\n  " + "\n  ".join(missing)
    assert not untracked, (
        "paper cites artifacts that exist locally but are NOT COMMITTED — a fresh clone "
        "cannot verify them:\n  " + "\n  ".join(untracked)
    )
