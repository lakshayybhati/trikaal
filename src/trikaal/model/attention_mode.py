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


def determinism_record(*, seed: int, device: str, attention_mode: str) -> dict:
    """The per-run determinism provenance (manifest + checkpoint metadata; never stdout-only)."""
    return {
        "attention_mode": attention_mode,
        "bit_exact_claim": attention_mode == MODE_SDPA,
        "seed": int(seed),
        "device": device,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


__all__ = [
    "MODE_FLASH2",
    "MODE_SDPA",
    "determinism_record",
    "flash2_available",
    "resolve_attention_mode",
    "set_attention_backend",
]
