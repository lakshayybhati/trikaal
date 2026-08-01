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

import platform
import sys

import numpy as np
import torch

# The fields that must AGREE across shards for a verdict to assemble. The 25 units are ONE paired
# comparison; a cell computed on a different instrument is a different experiment. Deliberately
# excludes seed and unit label, which are meant to vary.
PROVENANCE_IDENTITY_KEYS = (
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
    return {
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
