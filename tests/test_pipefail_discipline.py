"""PIPEFAIL RIDER (gate-2 calibration adjudication, 2026-07-21) — exit-code masking KAT.

The incident: commit e8d2a06's "anchor re-proven" claim entered the record unverified because
a ``python | grep | tail`` pipeline returned the LAST command's exit code, masking the
harness's exit-1 (the M5 anchor had actually FAILED to load the old-schema checkpoint).
Standing rule: any gate whose exit code feeds a claim must run under pipefail (or capture the
exit code directly, never through a filter pipeline). This KAT polices the committed shell
surface; interactive discipline is process (documented in prereg §7 v1.4.1)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_every_committed_shell_script_sets_pipefail():
    """Every .sh under scripts/ must set pipefail before any pipeline can run."""
    scripts = sorted((REPO / "scripts").glob("*.sh"))
    assert scripts, "expected at least one shell script (setup_cloud.sh)"
    for sh in scripts:
        text = sh.read_text()
        assert re.search(r"set\s+-[A-Za-z]*o\s+pipefail|set\s+-o\s+pipefail", text), (
            f"{sh.name}: no pipefail — a masked pipeline exit can turn a failed gate into "
            "an unverified claim (the e8d2a06 incident class)"
        )


def test_no_shell_true_pipelines_in_gate_scripts():
    """Gate-running python drivers must not shell out through masking pipelines."""
    for py in sorted((REPO / "scripts").glob("*.py")):
        text = py.read_text()
        for bad in ("os.system(", "shell=True"):
            assert bad not in text, (
                f"{py.name}: {bad} found — gate drivers must not route exit codes through "
                "a shell where a pipeline can mask them"
            )
