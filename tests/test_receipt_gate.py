"""SIX SCRIPTS WROTE "VERIFIED" FROM ZERO MEASUREMENTS. One gate now stands in front of all of them.

Individually each of the six looked like a rough edge. Together they are the mechanism by which a
repository starts asserting things nobody measured:

  * ``m6_h_sweep.py`` skipped all three cells and wrote ``L1_all_reproduced: true`` from an EMPTY
    dict — ``all(...)`` over nothing is True — exiting 0 while deleting 317 lines of a tracked
    receipt, and printing ``worst |delta| = None`` beside ``all_reproduced=True``;
  * ``m6_pull_verify.py`` printed ``PASS ... verified 0/0`` and exited 0, rewriting 9/9 to 0/0,
    beneath its own docstring's ``Exit 0 ONLY if every file verifies``;
  * ``m6_determinism_probe.py`` printed ``PROBE INVALID — this measured nothing`` and overwrote a
    VALID receipt;
  * ``m6_c3_dsr_units.py`` printed ``PROBE INVALID`` and wrote anyway, deleting 103 lines of
    CONFIRMED audit findings;
  * ``m6_bit_exact_correction.py`` silently rewrote a disclosure receipt from 46 records to 19;
  * ``m4b_universe_ingest.py`` made an unprompted LIVE NETWORK CALL on a no-argument run and
    overwrote the committed ``config/universe_full.yaml`` that pins the 200-symbol universe.

The norm — *a probe must REFUSE to emit a verdict when its own control arm fails* — was already
written down, and had been implemented for exactly one receipt. It is now a write-path gate.

Each script's OWN failing condition is reproduced below, so these are not tests of an abstraction.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from trikaal.utils.receipts import ReceiptRefused, is_tracked, write_receipt

REPO = Path(__file__).resolve().parents[1]
WIRED = {
    "m6_h_sweep.py": "the three-skip empty-cells write",
    "m6_pull_verify.py": "PASS ... verified 0/0",
    "m6_determinism_probe.py": "PROBE INVALID, wrote anyway",
    "m6_c3_dsr_units.py": "PROBE INVALID, deleted 103 lines",
    "m6_bit_exact_correction.py": "46 records rewritten to 19",
    "m4b_universe_ingest.py": "unprompted network call + tracked config overwrite",
}


# ── the three refusals ────────────────────────────────────────────────────────────────────────
def test_an_empty_measured_set_is_refused(tmp_path) -> None:
    """m6_h_sweep's exact shape: `all()` over an empty dict is True, so the verdict was `true`."""
    cells: dict[str, dict] = {}
    assert all(c["ok"] for c in cells.values()) is True, "the defect's premise no longer holds"
    with pytest.raises(ReceiptRefused, match="measured NOTHING"):
        write_receipt(tmp_path / "r.json", {"all_reproduced": True}, measured=cells)


def test_zero_of_zero_verified_is_refused(tmp_path) -> None:
    """m6_pull_verify's exact shape: verified 0 of 0 files and called it PASS."""
    with pytest.raises(ReceiptRefused, match="measured NOTHING"):
        write_receipt(tmp_path / "r.json", {"all_verified": True}, measured=[])


def test_a_self_reported_invalid_run_is_refused(tmp_path) -> None:
    """m6_determinism_probe and m6_c3_dsr_units both printed PROBE INVALID and wrote."""
    with pytest.raises(ReceiptRefused, match="reports itself INVALID"):
        write_receipt(
            tmp_path / "r.json",
            {"verdict": "PROBE INVALID"},
            measured=1,
            valid=False,
            invalid_reason="control arm failed",
        )


def test_a_tracked_target_is_refused_without_force() -> None:
    """The rule that protects EXISTING evidence — every one of the six destroyed committed data
    as a side effect of merely being run."""
    tracked = REPO / "pyproject.toml"
    assert is_tracked(tracked), "pyproject.toml is not tracked — this check has lost its subject"
    with pytest.raises(ReceiptRefused, match="TRACKED"):
        write_receipt(tracked, {"x": 1}, measured=1)
    assert "[tool.pytest.ini_options]" in tracked.read_text(), "the refusal did not prevent a write"


# ── and it still writes when it should ────────────────────────────────────────────────────────
def test_a_real_measurement_to_an_untracked_path_writes(tmp_path) -> None:
    """FIXTURE DISCRIMINATION: a gate that refused everything would pass every test above."""
    out = write_receipt(tmp_path / "r.json", {"ok": True}, measured=[1, 2, 3])
    assert json.loads(out.read_text()) == {"ok": True}


def test_force_permits_a_tracked_overwrite(tmp_path) -> None:
    """--force must actually work, or the gate is a wall rather than a gate."""
    p = tmp_path / "r.json"
    p.write_text("{}")
    assert write_receipt(p, {"ok": 1}, measured=1, force=True).exists()


@pytest.mark.parametrize("n", [0, [], {}, None, False])
def test_every_empty_shape_counts_as_nothing_measured(tmp_path, n) -> None:
    with pytest.raises(ReceiptRefused, match="measured NOTHING"):
        write_receipt(tmp_path / "r.json", {}, measured=n)


# ── the six are actually wired ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("script", sorted(WIRED))
def test_each_named_writer_routes_through_the_gate(script) -> None:
    src = (REPO / "scripts" / script).read_text()
    if script == "m4b_universe_ingest.py":
        assert "REFUSING" in src and "args.force" in src, (
            "the unprompted network call / tracked-config overwrite is ungated again"
        )
        return
    assert "write_receipt(" in src, f"{script} no longer calls the gate ({WIRED[script]})"
    assert "ReceiptRefused" in src, f"{script} calls the gate but does not handle its refusal"


@pytest.mark.parametrize("script", sorted(WIRED))
def test_no_named_writer_still_writes_a_receipt_directly(script) -> None:
    """A bare `write_text(json.dumps(...))` is the shape the gate exists to replace."""
    src = (REPO / "scripts" / script).read_text()
    bare = re.findall(r"^\s*\S*\.write_text\(json\.dumps\(", src, re.M)
    assert not bare, f"{script} still writes a receipt without the gate: {bare}"


def test_each_named_writer_exits_non_zero_and_writes_nothing_without_force() -> None:
    """THE ACCEPT, end to end: run each one, and require a non-zero exit AND a clean tree."""
    before = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    failures = []
    for script in sorted(WIRED):
        r = subprocess.run(
            ["python", str(REPO / "scripts" / script)],
            cwd=REPO,
            capture_output=True,
            text=True,
            env={"PATH": str(REPO / ".venv/bin"), "HOME": str(Path.home())},
        )
        after = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
        if r.returncode == 0:
            failures.append(f"{script} exited 0 without --force")
        if after != before:
            failures.append(
                f"{script} modified the tree: {set(after.split(chr(10))) ^ set(before.split(chr(10)))}"
            )
    assert not failures, failures
