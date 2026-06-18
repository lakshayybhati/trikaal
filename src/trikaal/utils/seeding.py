"""Global seed pinning — the engineering embodiment of the determinism deliverable (§7.7).

One call fixes python / numpy / torch RNG and forces deterministic algorithms so two
back-to-back runs with identical seed+config produce bit-identical outputs (gate G2).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_determinism(seed: int, *, deterministic_algorithms: bool = True) -> None:
    """Pin every RNG and force deterministic kernels.

    Args:
        seed: the single pinned seed (config field ``run.seed``).
        deterministic_algorithms: when True, force ``torch.use_deterministic_algorithms``
            and set ``CUBLAS_WORKSPACE_CONFIG`` so CUDA matmuls are reproducible. Always
            safe on CPU/MPS; required for the G2 determinism gate and any published run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — intentional global legacy-RNG seed for full determinism
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        # CUBLAS workspace must be set before the first CUDA matmul for determinism.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seeded_generator(seed: int) -> torch.Generator:
    """A CPU ``torch.Generator`` seeded deterministically (for dataloader/sampler RNG)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
