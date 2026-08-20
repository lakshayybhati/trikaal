"""NO ROUTABLE ADDRESS IN THE TRACKED TREE — and the sweep that said so was not portable.

THE SWEEP THAT GAVE A DIFFERENT ANSWER ON A DIFFERENT MACHINE. This item was reported closed
on the strength of ``git grep -E '\\b(...)\\b'`` returning nothing. ``git grep``'s ERE is the
PLATFORM's regex engine: glibc (Linux, and CI) implements ``\\b`` as a GNU extension and
MATCHES, while BSD (macOS) does not and returns a silent, confident nothing. The same command
therefore reads "clean" on one machine and finds the problem on another — and a pattern that
cannot match is indistinguishable from a clean repository. Four public IPv4 addresses of rented
GPU boxes were sitting in two tracked receipts the whole time, under ``public_ipaddr``.

(The platform split was found the hard way: the first version of the test below asserted that
``git grep`` NEVER supports ``\\b``, passed on macOS, and FAILED IN CI on Linux.)

This file therefore does its own matching in Python, where ``\\b`` means what it looks like on
every platform, and classifies with ``ipaddress`` rather than with a regex.

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
import platform
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
#: An ACTUAL git-grep invocation: the list form with flags, or the shell form with a flag. Prose
#: that merely names ``git grep`` — which every explanation of the defect must — is not a usage.
INVOCATION = re.compile(r'"git"\s*,\s*"grep"|\bgit\s+grep\s+-')


#: THIS FILE IS EXEMPT FROM ITS OWN SWEEP, and that is the sixth instance of the class in
#: ``tests/claim_site.py`` — Mode B, a check firing on the document that describes it. It carries
#: a routable address ON PURPOSE (the negative control must use one; RFC-5737 documentation ranges
#: report ``is_global=False`` and would make the control vacuous) and quotes CUDA versions in its
#: docstring. It could not see itself until it was committed, because the sweep walks
#: ``git ls-files``. The exemption is CHECKED below rather than trusted.
SELF = Path(__file__).resolve()


def _tracked_text_files(*, include_self: bool = False) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    files = [REPO / p for p in out if p and Path(p).suffix in TEXT_SUFFIXES]
    return files if include_self else [f for f in files if f.resolve() != SELF]


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


def test_a_word_boundary_in_git_grep_is_NOT_PORTABLE() -> None:
    """THE LESSON, PINNED — AND IT IS WORSE THAN "git grep has no \\b".

    The first version of this test asserted that ``git grep -E '\\bnumpy\\b'`` never matches. It
    passed on macOS and FAILED IN CI, which is how the real rule was found: ``git grep``'s ERE is
    the PLATFORM's regex engine. glibc (Linux, and CI) implements ``\\b`` as a GNU extension and
    matches; BSD (macOS) does not, and returns a silent, confident nothing.

    So the same sweep gives DIFFERENT ANSWERS ON DIFFERENT MACHINES — which is exactly how this
    item came to be reported closed with four addresses sitting in the tree. A pattern that cannot
    match on the machine you ran it on looks identical to a clean repository.

    This test therefore asserts the PORTABLE rule and refuses to encode either platform's answer:
    the explicit character-class form must work everywhere, Python ``re`` supports ``\\b``
    everywhere, and ``\\b`` in ``git grep`` is recorded as unreliable rather than as absent.
    """
    portable = subprocess.run(
        ["git", "grep", "-cE", r"(^|[^A-Za-z])numpy([^A-Za-z]|$)", "--", "pyproject.toml"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert portable.returncode == 0 and portable.stdout.strip(), (
        "the explicit character-class form must match on every platform; if this fails the "
        "subject file changed, not the rule"
    )
    assert re.search(r"\bnumpy\b", (REPO / "pyproject.toml").read_text()), (
        "python re supports \\b on every platform — that asymmetry with git grep is the trap"
    )
    boundary = subprocess.run(
        ["git", "grep", "-cE", r"\bnumpy\b", "--", "pyproject.toml"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    # Deliberately NOT asserted either way. Recorded so a future reader sees which engine ran.
    print(
        f"[platform note] git grep -E '\\b' on {platform.system()}: "
        f"{'MATCHED' if boundary.returncode == 0 else 'silently matched NOTHING'}"
    )


def test_no_test_or_script_passes_a_word_boundary_to_git_grep() -> None:
    """Nothing in the repo may inherit the defect. Measured: no git grep exists here at all.

    ★ JUDGED PER PARAGRAPH, BECAUSE THIS FIRED ON A DESCRIPTION OF THE DEFECT. Explaining the
    trap requires naming it, so every document that teaches it contains both ``git grep`` and
    ``\\b`` — and the first version of this check flagged ``tests/claim_site.py`` for saying so.
    That is Mode B in ``tests/claim_site.py``'s own taxonomy, on the file that defines the
    taxonomy. A mention whose paragraph retires or describes it is a record, not a usage.
    """
    offenders = []
    for f in _tracked_text_files():
        if f.suffix not in {".py", ".sh"}:
            continue
        text = f.read_text(errors="ignore")
        for m in re.finditer(INVOCATION, text):
            window = text[m.start() : m.start() + 300]
            if r"\b" in window:
                offenders.append(f"{f.relative_to(REPO)}: {window[:90]}")
    assert not offenders, offenders


@pytest.mark.parametrize(
    ("src", "is_invocation"),
    [
        ('run(["git", "grep", "-E", r"\\bnumpy\\b"])', True),
        ("git grep -nE '\\bnumpy\\b' -- x", True),
        ("the ``git grep``/``\\b`` defect: a pattern that cannot match", False),
        ("git grep's ERE has no \\b, so the pattern never matches", False),
    ],
)
def test_the_invocation_pattern_tells_a_command_from_prose(src, is_invocation) -> None:
    """FIXTURE DISCRIMINATION, and the reason the check is shaped this way. Teaching this trap
    requires NAMING it, so every document that explains it contains both ``git grep`` and ``\\b``.
    A proximity check flagged ``tests/claim_site.py`` for explaining the defect it defines. What
    separates a command from prose is not distance — it is the flags and the list form."""
    assert bool(re.search(INVOCATION, src)) is is_invocation, src


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


# ── the self-exemption, checked rather than trusted ───────────────────────────────────────────
def test_this_files_own_addresses_are_fixtures_and_nothing_else() -> None:
    """The exemption above must rot loudly. Every routable value in this file has to sit in a
    docstring or a test body — never in a data field like ``public_ipaddr`` — or the file has
    become the very thing it exists to prevent."""
    import ast

    src = SELF.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)

    # docstring line spans, resolved by AST rather than guessed from how a line starts — a
    # continuation line of a docstring begins with a word, not a quote, which is what a
    # line-shape heuristic gets wrong.
    doc_spans = []
    for node in [tree, *ast.walk(tree)]:
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = node.body[0] if node.body else None
            if isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant):
                if isinstance(d.value.value, str):
                    doc_spans.append((d.lineno, d.end_lineno))
    fn_spans = [
        (n.lineno, n.end_lineno)
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    def inside(lineno, spans):
        return any(lo <= lineno <= hi for lo, hi in spans)

    for lineno, addr in _routable_hits(SELF):
        line = lines[lineno - 1]
        if inside(lineno, fn_spans):
            continue  # inside a test body it IS a fixture — that is what a control is
        assert inside(lineno, doc_spans), (
            f"line {lineno} carries {addr} at module scope, outside any docstring or test body — "
            f"which is the shape of real data, not of an example: {line.strip()}"
        )
        assert not re.search(r'"(public_ipaddr|endpoint|host|ip)"\s*:\s*"', line), (
            f"line {lineno} carries {addr} in a DATA field: {line.strip()}"
        )


def test_the_sweep_would_see_this_file_if_it_were_not_exempt() -> None:
    """ANTI-VACUITY FOR THE EXEMPTION ITSELF. If the exemption were silently doing nothing, or if
    the file stopped carrying a control address, this would pass for the wrong reason."""
    with_self = {f.resolve() for f in _tracked_text_files(include_self=True)}
    without = {f.resolve() for f in _tracked_text_files()}
    assert with_self - without == {SELF}, "the exemption is not excluding exactly this file"
    assert _routable_hits(SELF), (
        "this file no longer carries a routable control address — the negative control above is "
        "vacuous, and the exemption is no longer needed"
    )
