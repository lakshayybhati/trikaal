"""Attention-backend selection + the per-run determinism record (m6_design §6 item 7, inv. 7).

Invariant 7 (scoped honestly): GPU training is bit-exact ONLY under the deterministic-attention
fallback — FlashAttention-2 is faster but non-deterministic — and **every run records its mode**.
This module is that hook: :func:`resolve_attention_mode` picks the backend,
:func:`set_attention_backend` applies it to every attention module, and
:func:`determinism_record` produces the provenance dict that goes into BOTH the per-run manifest
and the checkpoint metadata.

Modes:
* ``sdpa_deterministic`` — torch SDPA math path (CPU/MPS default; the CUDA bit-exact fallback).
* ``flash2`` — FlashAttention-2 kernels (CUDA only, requires ``flash_attn`` importable); NOT
  bit-reproducible run-to-run — a run recorded under this mode never claims bit-exactness.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch import nn

MODE_SDPA = "sdpa_deterministic"
MODE_FLASH2 = "flash2"


def flash2_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:  # pragma: no cover — CUDA box only
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_attention_mode(device: str, *, prefer_flash: bool = False) -> str:
    """Pick the backend for this run. Deterministic SDPA unless flash2 is explicitly preferred
    AND actually available (CUDA + importable flash_attn)."""
    if prefer_flash and device.split(":")[0] == "cuda" and flash2_available():
        return MODE_FLASH2  # pragma: no cover — CUDA box only
    return MODE_SDPA


def set_attention_backend(model: nn.Module, mode: str) -> int:
    """Stamp ``mode`` onto every attention module in the tree; returns how many were set.

    ``flash2`` demands availability up front — failing at step 1 of a 15-day run is the cheap
    failure; silently falling back (and mis-recording the mode) is the expensive one."""
    if mode not in (MODE_SDPA, MODE_FLASH2):
        raise ValueError(f"unknown attention mode {mode!r}")
    if mode == MODE_FLASH2 and not flash2_available():
        raise RuntimeError(
            "flash2 requested but FlashAttention-2 is unavailable (needs CUDA + flash-attn); "
            "use sdpa_deterministic or install the wheel (cloud_runbook §4)"
        )
    n = 0
    for m in model.modules():
        if hasattr(m, "attn_dropout") and hasattr(m, "qkv"):  # MultiHeadSelfAttention duck-type
            m.attention_backend = mode
            n += 1
    return n


def _cuda_provenance() -> dict:
    """GPU model / driver / CUDA / cuDNN — the fields a cross-hardware divergence needs.

    §7 v1.4.6 item 3: the acceptance and money-leg Stage-2 manifests recorded ``device: "cuda"``
    and nothing else, so when their cell4 models diverged (frac_negative 0.5662 vs 0.99883 under
    byte-identical recipes) the divergence was UNATTRIBUTABLE from the receipts. Every field here
    is one that was needed and absent. Each probe is guarded: provenance must never be the thing
    that crashes a 15-day run.
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    out: dict = {"cuda_available": True, "cuda_build": torch.version.cuda}
    for key, probe in (
        ("gpu_name", lambda: torch.cuda.get_device_name(0)),
        ("gpu_capability", lambda: ".".join(map(str, torch.cuda.get_device_capability(0)))),
        ("gpu_count", torch.cuda.device_count),
        ("gpu_total_memory_bytes", lambda: torch.cuda.get_device_properties(0).total_memory),
        ("cudnn_version", torch.backends.cudnn.version),
        # THE LOOKUP MUST BE DEFERRED, NOT JUST THE CALL. This entry previously passed the bound
        # function directly — `torch._C._cuda_getDriverVersion` — which resolves the attribute
        # while the tuple is being BUILT, i.e. before the try/except below. torch 2.12.1 (the
        # pinned version) does not define it, so the AttributeError escaped the guard entirely and
        # determinism_record() raised. Since orchestrator.run_cell calls determinism_record BEFORE
        # training, the real M6 run would have died on CUDA at the first cell. It was invisible
        # locally because every CPU/MPS path returns at the cuda_available check above and never
        # reaches this tuple. The comment said "guarded" and the code was not: a guard that cannot
        # fire is not a guard.
        ("driver_version", lambda: torch._C._cuda_getDriverVersion()),
    ):
        try:
            out[key] = probe()
        except Exception as e:  # pragma: no cover — probe availability is platform-dependent
            out[key] = f"unavailable: {type(e).__name__}"
    return out


def determinism_record(
    *, seed: int, device: str, attention_mode: str, extra_seeds: dict | None = None
) -> dict:
    """The per-run determinism provenance (manifest + checkpoint metadata; never stdout-only).

    ``bit_exact_claim`` is retained with its historical meaning (attention mode == SDPA) so
    existing manifests stay comparable, but §7 v1.4.6 adds ``bit_exact_preconditions_met``, which
    is the claim that can actually be relied on: SDPA attention AND forced deterministic
    algorithms AND cuDNN determinism AND cuDNN autotune off. These two DISAGREE today — every M6
    path calls ``set_determinism(..., deterministic_algorithms=False)``, so a CUDA+SDPA run stamps
    ``bit_exact_claim: true`` beside ``deterministic_algorithms: false``. The disagreement is
    surfaced in ``bit_exact_caveat`` rather than silently resolved: redefining a recorded claim is
    a supervisor decision, but a reader must not be able to miss the contradiction.
    """
    det_algos = bool(torch.are_deterministic_algorithms_enabled())
    cudnn_det = bool(torch.backends.cudnn.deterministic)
    cudnn_bench = bool(torch.backends.cudnn.benchmark)
    claim = attention_mode == MODE_SDPA
    preconditions = bool(claim and det_algos and cudnn_det and not cudnn_bench)
    rec = {
        "attention_mode": attention_mode,
        "bit_exact_claim": claim,
        "bit_exact_preconditions_met": preconditions,
        "seed": int(seed),
        "seeds": {"run": int(seed), **{k: int(v) for k, v in (extra_seeds or {}).items()}},
        "device": device,
        "deterministic_algorithms": det_algos,
        "determinism_flags": {
            "torch_use_deterministic_algorithms": det_algos,
            "cudnn_deterministic": cudnn_det,
            "cudnn_benchmark": cudnn_bench,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "hardware": _cuda_provenance(),
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    if claim and not preconditions:
        rec["bit_exact_caveat"] = (
            "bit_exact_claim is TRUE on attention mode alone, but the preconditions are NOT met "
            "(see determinism_flags). Run-to-run bit-exactness is NOT established for this run: "
            "CUDA reduction order is unconstrained. Do not quote this run as bit-exact."
        )
    return rec


__all__ = [
    "MODE_FLASH2",
    "MODE_SDPA",
    "determinism_record",
    "flash2_available",
    "resolve_attention_mode",
    "set_attention_backend",
]
