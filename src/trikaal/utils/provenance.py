"""§7 v1.6 C-5 A7 / C-18 — the per-unit environment stamp.

Lives in ``utils`` rather than beside either consumer because BOTH the artifact writer
(``eval.verdict``) and the matrix runner (``run.matrix``) need it, and ``run.matrix`` imports
``eval.verdict``. A copy in each would be the C-6 shape again.

WHY THE INTERPRETER IS CALLED OUT. ``uv.lock`` pins PACKAGES, not the interpreter:
``requires-python ">=3.11"`` admits 3.14, and a rented box silently ran 3.14.6 while the run was
being described as "the exact pinned environment". That is C-18, and the only reliable fix is to
record ``platform.python_version()`` per unit and compare it across shards.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

# The fields that must AGREE across shards for a verdict to assemble. The 25 units are ONE paired
# comparison; a cell computed on a different instrument is a different experiment. Deliberately
# excludes seed and unit label, which are meant to vary.
# §7 v1.6.23 (FAN-OUT): `image` and `lockfile_sha256` were MISSING from the identity surface.
# Two boxes can carry identical torch/numpy/driver strings and still differ in container image
# (different CUDA userspace, different BLAS) or in the resolved dependency set. Across 5 shards
# that is 5 chances for an unrecorded difference to sit inside one paired comparison. They are
# identity keys now, so a mismatch REFUSES rather than being invisible.
# §7 v1.6.25 (RE-AUDIT R6, BLOCKING): the surface recorded WHICH MACHINE ran a unit and said
# nothing about WHICH CODE. Two shards could carry byte-identical hardware, driver, image,
# interpreter and lockfile stamps while running different revisions of Trikaal — and the auditor
# named the concrete path: a shard built from a stale payload trains at the OLD 2,000-step budget,
# assembles beside four 26,003-step shards without a murmur, and its cell enters ΔIR(4−5) as
# "trained less" rather than "different arm". That is the cheapest remaining route to a
# silent WRONG number, so `git_commit` and the two step budgets are identity keys.
PROVENANCE_IDENTITY_KEYS = (
    "image",
    "git_commit",
    "steps_stage1",
    "steps_stage2",
    "lockfile_sha256",
    "gpu_name",
    "cuda_build",
    "driver_version",
    "torch",
    "numpy",
    "python",
    "platform_abi",
    "attention_mode",
    "deterministic_algorithms",
    "cudnn_deterministic",
    "cudnn_benchmark",
)

UNAVAILABLE = "unavailable"


def _git_commit() -> str:
    """The revision this unit ran, or ``"unavailable"``.

    TWO SOURCES, IN ORDER, because the money run has NO ``.git``: the runbook scp's a payload
    tarball to each box precisely so no credential ever reaches it, which also means no repository.
    So the launcher exports ``TRIKAAL_GIT_COMMIT`` from the machine that BUILT the tarball — the
    same mechanism as ``TRIKAAL_IMAGE`` — and the local `git rev-parse` is the fallback for runs
    that do have a checkout (CI, laptop, the dry runs).

    ★ THE PREMISE ABOVE NARROWED ON 2026-08-18. The tarball was scp'd "precisely so no credential
    ever reaches it" because the repository was PRIVATE and cloning needed a token. It is now
    PUBLIC, so a box can `git clone` with no credential — which means a fan-out MAY have a .git,
    and for those runs the fallback below is the primary source. Nothing in this function changes:
    the env var still wins when set, and `git rev-parse` still answers when there is a checkout.
    Recorded because the ORDER is deliberate and a reader would otherwise wonder why the weaker
    source is first.

    Being an IDENTITY key, a fan-out where some shards stamp it and others do not is a REFUSAL.
    """
    env = os.environ.get("TRIKAAL_GIT_COMMIT")
    if env:
        return env.strip()
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, ValueError, subprocess.SubprocessError):  # pragma: no cover — no git present
        pass
    return UNAVAILABLE


def _driver_version() -> str:
    """The NVIDIA driver version, or ``"unavailable"``.

    **§7 v1.6.26 (P2 finding F3) — THIS READ `unavailable` ON A REAL 4090.** The previous form was
    ``str(torch._C._cuda_getDriverVersion())`` inside ``except (AttributeError, RuntimeError)``.
    At torch 2.12.1 that private symbol no longer behaves as assumed, so on a box whose
    ``nvidia-smi`` reported **580.159.03** the identity key silently fell back to the placeholder.
    **That is the R6 class recurring inside the fix for R6:** a key carrying the same placeholder
    on every shard sees no disagreement across the fan-out and protects nothing while appearing to.

    ``nvidia-smi`` is authoritative here — it is the driver's own reporter, not a binding to it —
    so it is tried FIRST and the torch private symbol is the fallback, not the reverse.
    """
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except (OSError, ValueError, subprocess.SubprocessError):  # pragma: no cover — no nvidia-smi
        pass
    try:  # pragma: no cover — CUDA box only
        return str(torch._C._cuda_getDriverVersion())
    except (AttributeError, RuntimeError):
        return UNAVAILABLE


def platform_abi(full: str, system: str, release: str) -> str:
    """The ABI-bearing part of ``platform.platform()`` — architecture + libc, kernel REMOVED.

    **WE CHANGED WHAT WE REFUSE ON, NEVER WHAT WE RECORD.** ``platform`` is still stamped in full
    on every artifact, kernel included, so a reviewer can audit exactly which kernels ran. What
    moved is the COMPARISON: ``platform_abi`` is the identity key, ``platform`` is a recorded
    field.

    THE ARGUMENT. ``platform.platform()`` on Linux is ``Linux-<kernel>-<arch>-with-<libc>``.
    ``arch`` obviously bears on numerics; ``libc`` can, through libm. **A kernel patch release
    does not participate in GPU arithmetic** — 5.15.0-179 vs 5.15.0-186 changes no float
    operation, no CUDA kernel and no torch code path — and it was the one component of the one
    key that vast exposes no way to filter on, so it priced a multi-box pool out of the market
    while defending nothing.

    **FAILS CLOSED.** The kernel is identified by stripping the exact ``"{system}-{release}-"``
    prefix. When that prefix does not match — macOS reports ``platform()`` as
    ``macOS-26.5.1-arm64-arm-64bit`` while ``system()``/``release()`` say ``Darwin``/``25.5.0`` —
    the FULL string is returned and the key goes on comparing everything it compared before. A
    relaxation whose parser silently mangles an unrecognised format would be the fail-open class,
    and it would look exactly like success.

    Pure in its inputs so the LINUX behaviour is testable off Linux, on the literal strings the
    rented boxes actually produced.
    """
    prefix = f"{system}-{release}-"
    return full[len(prefix) :] if full.startswith(prefix) and len(full) > len(prefix) else full


def identity_placeholder_failures(prov: dict) -> list[str]:
    """Identity keys still carrying a PLACEHOLDER on a device that should have resolved them.

    **§7 v1.6.26 (P2 finding F3) — the assertion that would have caught it.** A placeholder is not
    a refusal *across* shards: every shard writes the same ``"unavailable"``, the comparison sees
    no disagreement, and the key protects nothing while appearing to. On a CUDA device every
    identity key must carry a real value, so this is checked BEFORE spend (the driver's early gate)
    and again at artifact-write time (the durable one).

    On CPU the CUDA-only keys are legitimately unresolvable and are exempt — recording
    ``"unavailable"`` there is the honest value the C-18 rule asks for, not a defect.
    """
    if not str(prov.get("device", "")).startswith("cuda"):
        return []
    bad = [
        k
        for k in PROVENANCE_IDENTITY_KEYS
        if str(prov.get(k)) in (UNAVAILABLE, "unknown", "None", "")
    ]
    return [
        f"identity key {k!r} = {prov.get(k)!r} on a CUDA device — a placeholder is IDENTICAL on "
        "every shard, so it can never produce a refusal and protects nothing (§7 v1.6.26 F3)"
        for k in bad
    ]


def identity_reference_failures(prov: dict, reference: dict) -> list[str]:
    """This box's identity vs the PINNED REFERENCE — refuse at second zero, not at assembly.

    THE DEFECT THIS CLOSES. Until now the only pre-spend identity check was
    :func:`identity_placeholder_failures`, which rejects PLACEHOLDER VALUES and nothing else — its
    own docstring says "a placeholder is not a refusal *across* shards". A box carrying driver
    ``580.x``, a different glibc, or a different 4090 SKU has all-REAL values, passes the
    pre-flight gate, computes its unit, and is caught only at assembly by
    ``verdict.provenance_failures`` — after every unit has been paid for. Survivable at three
    boxes rented in one driver-filtered batch and hand-verified; NOT survivable at 25 units across
    many boxes over days, where each new box is a silent chance to poison the whole matrix and the
    failure surfaces at the end.

    COMPARES EXACTLY :data:`PROVENANCE_IDENTITY_KEYS` — no more, no fewer. If this gate and
    ``provenance_failures`` compared different sets they could disagree, and a run could pass the
    cheap gate while being doomed at the expensive one; ``test_identity_gate_key_set_matches``
    asserts the equality directly rather than trusting that both call sites were kept in step.

    ``platform`` is RECORDED, NEVER COMPARED (§7 v1.6.29). Its kernel component differs
    legitimately across hosts — the live pool ran 5.15.0-52 beside 6.8.0-101 — and comparing it
    would have refused a healthy pool. ``platform_abi`` is the identity key.

    The reference is a COMMITTED FILE written at unit 1, never "the first box I happen to talk
    to": a reference taken from whichever machine answered first makes the instrument an accident
    of scheduling.
    """
    return [
        f"identity key {k!r} = {prov.get(k)!r} but the pinned reference is {reference.get(k)!r} "
        "— this box is a DIFFERENT INSTRUMENT and its unit could never join the matrix "
        "(refused before compute rather than at assembly)"
        for k in PROVENANCE_IDENTITY_KEYS
        if prov.get(k) != reference.get(k)
    ]


def run_provenance(device: str = "cpu", *, attention_mode: str = "unknown") -> dict:
    """Stamp the live environment. Recorded, never chosen.

    An unobtainable value is the literal ``"unavailable"`` rather than a plausible default, so a
    missing datum can never be read back as a measured one — the same rule the throughput record
    follows with ``measured=False`` + NaN."""
    gpu_name = cuda_build = driver = UNAVAILABLE
    if torch.cuda.is_available():  # pragma: no cover — no CUDA in CI
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except (RuntimeError, AssertionError):
            pass
        cuda_build = getattr(torch.version, "cuda", None) or UNAVAILABLE
        driver = _driver_version()
    # The container image is not introspectable from inside the process; the launcher stamps it
    # via TRIKAAL_IMAGE. "unavailable" is the honest value when it was not set, and because it is
    # an IDENTITY key, a run where SOME shards set it and others did not is a REFUSAL, not a
    # silent pass — which is the case that would otherwise slip through.
    image = os.environ.get("TRIKAAL_IMAGE", UNAVAILABLE)
    lock = Path("uv.lock")
    try:
        lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else UNAVAILABLE
    except OSError:
        lock_sha = UNAVAILABLE
    # §7 v1.6.25 R6: the step budgets come from the LIVE conformance pins, so a shard built from a
    # revision whose constant still read 2,000 stamps 2,000 and refuses against its four siblings.
    # Imported inside the call to keep `utils` free of any `eval` import edge.
    from trikaal.eval.conformance import PINNED_STEPS_STAGE1, PINNED_STEPS_STAGE2

    return {
        "image": image,
        "git_commit": _git_commit(),
        "steps_stage1": int(PINNED_STEPS_STAGE1),
        "steps_stage2": int(PINNED_STEPS_STAGE2),
        "lockfile_sha256": lock_sha,
        "gpu_name": gpu_name,
        "cuda_build": cuda_build,
        "driver_version": driver,
        "device": device,
        # ★ str(), AND IT IS NOT COSMETIC. `torch.__version__` is a `TorchVersion` object, not a
        # str. Pickled into a checkpoint it becomes a GLOBAL that `torch.load(weights_only=True)`
        # refuses by default — so every published checkpoint could only be opened with
        # `weights_only=False`, i.e. by executing arbitrary pickle from a 127 MB download. One
        # attribute made the safe path unavailable to every reader of these weights. Checkpoints
        # already published still carry the object; the model card shows the `add_safe_globals`
        # one-liner that opens them safely without it.
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "python": platform.python_version(),  # C-18 — the pin uv.lock does NOT carry
        "python_executable": sys.executable,
        # RECORDED IN FULL, KERNEL INCLUDED — never compared. `platform_abi` is the identity key.
        "platform": platform.platform(),
        "platform_abi": platform_abi(platform.platform(), platform.system(), platform.release()),
        "attention_mode": attention_mode,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
