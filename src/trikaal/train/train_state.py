"""Resumable + ATOMIC full-train-state checkpointing (m6_design §6 item 9).

The weights+config checkpoint (``checkpoint.py``) gives artifact IDENTITY; it cannot RESUME a
training run — that needs the full :func:`save_train_state` payload: model + optimizer state +
step counter (the LR schedule is a pure function of step) + every RNG stream (torch CPU/CUDA,
the trainer's numpy generator, python ``random``) + the data-sampler cursor. A resumed run that
restores all of these continues the EXACT trajectory (any miss diverges — that is what the
kill-resume test asserts).

Writes are atomic: serialize to ``<path>.tmp`` then ``os.replace`` — a crash mid-write can never
corrupt the last good state (``torch.save`` straight onto the live file could). This is the local
twin of pre-flight Item 3.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

TRAIN_STATE_FORMAT = "trikaal-train-state-1"


def capture_rng_states(np_rng: np.random.Generator | None = None) -> dict[str, Any]:
    """Snapshot every RNG stream a training loop consumes."""
    states: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if np_rng is not None:
        states["numpy_generator"] = np_rng.bit_generator.state
    if torch.cuda.is_available():  # pragma: no cover — CUDA box only
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        try:
            states["torch_mps"] = torch.mps.get_rng_state()
        except (AttributeError, RuntimeError):  # older torch — MPS state not exposed
            pass
    return states


def restore_rng_states(states: dict[str, Any], np_rng: np.random.Generator | None = None) -> None:
    torch.set_rng_state(states["torch_cpu"])
    random.setstate(states["python"])
    if np_rng is not None and "numpy_generator" in states:
        np_rng.bit_generator.state = states["numpy_generator"]
    if "torch_cuda" in states and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    if "torch_mps" in states and torch.backends.mps.is_available():
        try:
            torch.mps.set_rng_state(states["torch_mps"])
        except (AttributeError, RuntimeError):
            pass


def save_train_state(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    np_rng: np.random.Generator | None = None,
    sampler_state: dict | None = None,
    extra: dict | None = None,
) -> None:
    """Atomically persist the FULL resumable state (tmp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": TRAIN_STATE_FORMAT,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_states(np_rng),
        "sampler_state": sampler_state,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX — the live file is never half-written


def load_train_state(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    np_rng: np.random.Generator | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Restore model/optimizer/RNG in place; return ``{step, sampler_state, extra}``."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format") != TRAIN_STATE_FORMAT:
        raise ValueError(f"unrecognized train-state format: {payload.get('format')!r}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_states(payload["rng"], np_rng)
    return {
        "step": int(payload["step"]),
        "sampler_state": payload.get("sampler_state"),
        "extra": payload.get("extra", {}),
    }


def autocast_ctx(device: str, *, bf16: bool):
    """The bf16-autocast knob: enabled → ``torch.autocast(bfloat16)`` on the device type (the
    cloud CUDA target precision); disabled (MPS/CPU default) → a no-op context."""
    dev_type = device.split(":")[0]
    return torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=bf16)


__all__ = [
    "TRAIN_STATE_FORMAT",
    "autocast_ctx",
    "capture_rng_states",
    "load_train_state",
    "restore_rng_states",
    "save_train_state",
]
