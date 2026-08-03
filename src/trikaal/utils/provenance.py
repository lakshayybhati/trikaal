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
    "platform",
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
        try:
            # The lookup is deferred INSIDE the call on purpose: evaluating
            # ``torch._C._cuda_getDriverVersion`` at tuple-build time put the attribute access
            # outside its own try/except and crashed the CUDA probe at cell 1, seed 0.
            driver = str(torch._C._cuda_getDriverVersion())
        except (AttributeError, RuntimeError):
            pass
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
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),  # C-18 — the pin uv.lock does NOT carry
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "attention_mode": attention_mode,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
