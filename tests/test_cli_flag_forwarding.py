"""A CLI flag that is ACCEPTED but never FORWARDED is a defect — at two instances, systematic.

THE PATTERN, both caught only by a $0 gate:

* ``m6_cuda_probe.py`` accepted ``--warmup`` and never passed it to the bench subprocess, so the
  bench used its own default (10) against 4 steps → ``timed_steps = -6`` → NaN in both arms. It
  would have "worked" at the real settings only because the two defaults coincide.
* ``m6_toy_rehearsal.py`` accepted ``--steps`` and ``--local-smoke`` overwrote it
  unconditionally, so ``--local-smoke --steps 200`` silently ran 30 steps — not enough for the
  loss to drop, so the tripwire aborted on whichever seed drew unluckily. That masqueraded as a
  training problem for three gate attempts.

Two scripts, same shape: argparse declares it, the code drops it, and the symptom appears
somewhere else entirely. These are cheap structural checks, not ceremony — each asserts that a
declared flag actually reaches the place it is supposed to affect.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _declared_flags(path: Path) -> set[str]:
    """Every ``--flag`` string passed to ``ap.add_argument`` in the file."""
    tree = ast.parse(path.read_text())
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


def _dest(flag: str) -> str:
    return flag.removeprefix("--").replace("-", "_")


def test_probe_forwards_warmup_to_the_bench_subprocess():
    """THE ORIGINAL: --warmup was declared and dropped, producing NaN rates in both arms."""
    src = (REPO / "scripts" / "m6_cuda_probe.py").read_text()
    assert "--warmup" in _declared_flags(REPO / "scripts" / "m6_cuda_probe.py")
    # NOT `'"--warmup," in src` — that matches the argparse DECLARATION, so it passes whether or
    # not the flag is forwarded. This test's first version did exactly that and survived a
    # mutation which deleted the flag from the argv: an assertion that cannot fail is not a
    # check (prereg §7 v1.5 fixture-discrimination rule), caught here by mutating it.
    assert src.count('"--warmup"') >= 2, (
        "--warmup appears only once: it is declared in the parser but never placed in the "
        "bench argv"
    )
    assert "str(warmup)" in src, "--warmup's value is never serialised into the subprocess argv"
    assert "warmup=args.warmup" in src, "--warmup never reaches bench_arm()"


def test_validation_forwards_its_flags_to_the_rehearsal():
    """The validation driver builds the rehearsal argv; every knob it exposes must be in it."""
    src = (REPO / "scripts" / "m6_cuda_validation.py").read_text()
    for flag in ("--steps", "--batch", "--seq-len", "--symbols", "--seeds", "--eval-window"):
        assert f'"{flag}",' in src, f"{flag} never reaches the rehearsal argv"
    # --local-smoke is conditional, so it appears in a guarded append rather than the base list
    assert "--local-smoke" in src
    assert "if args.local_smoke else []" in src, "--local-smoke passthrough lost (it has been"
    " lost to a formatter run before — re-verify after formatting)"
    # the timeouts must actually be handed to run(), not merely declared
    assert "timeout_s=args.train_timeout_s" in src, "train timeout declared but never applied"
    assert "timeout_s=args.verdict_timeout_s" in src, "verdict timeout declared but never applied"


def test_local_smoke_does_not_clobber_an_explicit_steps():
    """The exact defect: --local-smoke used to overwrite args.steps unconditionally."""
    path = REPO / "scripts" / "m6_toy_rehearsal.py"
    src = path.read_text()
    assert "--steps" in _declared_flags(path)
    # the tell-tale of the bug was a tuple assignment that included args.steps
    assert 'args.device, args.steps, args.batch = "cpu", 30, 16' not in src, (
        "--local-smoke is clobbering an explicitly-passed --steps again"
    )
    assert "if args.steps is None:" in src, "--steps must be resolved via a None sentinel"


def test_every_declared_flag_is_referenced_by_its_dest():
    """Structural: a flag nobody reads is either dead or dropped. Both are defects."""
    for name in ("m6_cuda_probe.py", "m6_cuda_validation.py", "m6_toy_rehearsal.py"):
        path = REPO / "scripts" / name
        src = path.read_text()
        for flag in _declared_flags(path):
            dest = _dest(flag)
            assert f"args.{dest}" in src, (
                f"{name}: {flag} is declared but args.{dest} is never read"
            )
