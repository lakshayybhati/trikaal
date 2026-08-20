"""NO ROUTABLE ADDRESS IN THE TRACKED TREE — and the sweep that said so was broken.

THE SWEEP THAT COULD NOT FAIL. This item was reported closed on the strength of
``git grep -E '\\b(...)\\b'`` returning nothing. ``git grep``'s ERE has no ``\\b``: the pattern
silently never matches, so the command returns "clean" for every possible repository. Plain
``grep`` supports it, which is why it looked right. Four public IPv4 addresses of rented GPU boxes
were sitting in two tracked receipts the whole time, under the field name ``public_ipaddr``.

This file therefore does its own matching in Python, where ``\\b`` means what it looks like, and
classifies with ``ipaddress`` rather than with a regex.

REDACTED, NOT DELETED. The receipts exist to prove the M6 shards ran on THREE DISTINCT MACHINES —
``host_id`` cannot establish that, because one host_id is a provider ACCOUNT operating many
machines. Three distinct digests carry that claim exactly as well as three addresses, so both
halves are asserted here: no address survives, AND the claim still verifies.

A DOTTED QUAD IS NOT AN ADDRESS. ``uv.lock`` carries CUDA wheel versions like ``12.0.0.61`` and
``9.20.0.48`` that parse as routable IPv4 and are nothing of the kind. They are exempted by
CONTEXT, not by allow-listing the values, and the exemption is checked rather than trusted — the
same reference-vs-subject discipline as ``tests/claim_site.py``.
"""

from __future__ import annotations

import ipaddress
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
QUAD = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
POOL = REPO / "runs_manifest/m6_pool_identity.json"
RUNBOX = REPO / "runs_manifest/m6_run_box_identity.json"
TEXT_SUFFIXES = {".md", ".py", ".txt", ".json", ".jsonl", ".yml", ".yaml", ".sh", ".toml", ".cfg"}
#: A quad on one of these lines is a package version, not a host.
VERSION_CONTEXT = re.compile(r"version\s*=|\.whl|\.tar\.gz|files\.pythonhosted\.org|upload-time")


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return [REPO / p for p in out if p and Path(p).suffix in TEXT_SUFFIXES]


def _routable_hits(path: Path) -> list[tuple[int, str]]:
    hits = []
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return hits
    for i, line in enumerate(lines, 1):
        if VERSION_CONTEXT.search(line):
            continue  # a package version that happens to parse as an address
        for m in QUAD.finditer(line):
            try:
                addr = ipaddress.ip_address(m.group(0))
            except ValueError:
                continue
            if addr.is_global:
                hits.append((i, str(addr)))
    return hits


# ── the sweep has teeth ───────────────────────────────────────────────────────────────────────
def test_NEGATIVE_CONTROL_the_matcher_finds_a_planted_address(tmp_path) -> None:
    f = tmp_path / "x.json"
    f.write_text('{"public_ipaddr": "76.121.3.151"}\n')
    assert _routable_hits(f) == [(1, "76.121.3.151")]


def test_NEGATIVE_CONTROL_private_and_loopback_are_not_flagged(tmp_path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("127.0.0.1 10.0.0.5 192.168.1.1 0.0.0.0\n")
    assert _routable_hits(f) == []


def test_the_version_exemption_is_justified_not_assumed() -> None:
    """The exempted quads must ALL be package versions. If uv.lock ever grew a real host, this
    fails rather than quietly widening the hole."""
    lock = REPO / "uv.lock"
    if not lock.is_file():
        pytest.skip("uv.lock absent")
    exempted = []
    for line in lock.read_text().splitlines():
        if not VERSION_CONTEXT.search(line):
            continue
        for m in QUAD.finditer(line):
            try:
                a = ipaddress.ip_address(m.group(0))
            except ValueError:
                continue
            if a.is_global:
                exempted.append(line.strip())
    assert exempted, "no exempted quads found — this exemption may no longer be needed"
    for line in exempted:
        # matched against the FULL line; truncating first hid the evidence past column 120
        assert re.search(r'version = "|nvidia_\w+-|\.whl|pythonhosted\.org', line), (
            f"exempted a line that is not a package version: {line[:140]}"
        )


def test_git_grep_cannot_be_used_for_this() -> None:
    """THE LESSON, PINNED. git grep's ERE has no \\b, so the pattern that closed this item could
    not match anything. Asserted rather than remembered."""
    r = subprocess.run(
        ["git", "grep", "-cE", r"\bnumpy\b", "--", "pyproject.toml"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and not r.stdout.strip(), (
        "git grep now supports \\b — this repo's sweeps may be re-checked, but do not assume it"
    )
    assert re.search(r"\bnumpy\b", (REPO / "pyproject.toml").read_text()), (
        "python re DOES support \\b — that asymmetry is the whole trap"
    )


def test_no_test_or_script_passes_a_word_boundary_to_git_grep() -> None:
    """Nothing in the repo may inherit the defect. Measured: no git grep exists here at all."""
    offenders = []
    for f in _tracked_text_files():
        if f.suffix not in {".py", ".sh"}:
            continue
        text = f.read_text(errors="ignore")
        for m in re.finditer(r"git[\"'\s,\]]+grep", text):
            window = text[m.start() : m.start() + 300]
            if r"\b" in window:
                offenders.append(f"{f.relative_to(REPO)}: {window[:90]}")
    assert not offenders, offenders


# ── the sweep itself ──────────────────────────────────────────────────────────────────────────
def test_no_tracked_file_carries_a_routable_address() -> None:
    bad = {}
    for f in _tracked_text_files():
        hits = _routable_hits(f)
        if hits:
            bad[str(f.relative_to(REPO))] = hits
    assert not bad, f"routable addresses in tracked files: {bad}"


# ── and the claim the addresses were carrying still verifies ─────────────────────────────────
def test_three_distinct_machines_still_verifies_from_the_receipt() -> None:
    d = json.loads(POOL.read_text())
    boxes = d["boxes"]
    assert len(boxes) == 3, boxes
    digests = {b["public_ipaddr"] for b in boxes.values()}
    machines = {b["machine_id"] for b in boxes.values()}
    assert len(digests) == 3, f"the distinctness claim did not survive redaction: {digests}"
    assert len(machines) == 3, machines
    assert all(v.startswith("sha256:") for v in digests), digests


def test_the_substitution_is_recorded_in_the_file_itself() -> None:
    """A redaction nobody can see is indistinguishable from a value that was never there."""
    for path in (POOL, RUNBOX):
        d = json.loads(path.read_text())
        note = d.get("ADDRESS_REDACTION", "")
        assert "REDACTED, NOT DELETED" in note, path
        assert "NOT A CRYPTOGRAPHIC GUARANTEE" in note, (
            f"{path} no longer states the limit of what the digest buys"
        )


def test_the_ssh_hostport_is_gone() -> None:
    text = RUNBOX.read_text()
    assert "HostPort REDACTED" in text
    assert not re.search(r"HostPort\s+\d+", text), "an SSH HostPort survives"
