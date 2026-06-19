"""Content-hashed checkpoint save/load for the tokenizer and AR predictor (§7.7).

Determinism is a deliverable: every checkpoint carries a content hash over its (sorted) weights +
construction config, so a past run is reconstructable and a downstream stage can *assert* it is
consuming the exact frozen artifact it expects (e.g. Stage-2 asserting the tokenizer hash before
materializing tokens). The hash is the artifact's identity — not a substitute for the
GPU-training-determinism caveat (FlashAttention-2 is non-deterministic; see §7.G2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from trikaal.utils.hashing import content_hash

CKPT_FORMAT = "trikaal-ckpt-1"


def state_dict_hash(sd: dict[str, torch.Tensor]) -> str:
    """Deterministic content hash over a state_dict (sorted keys + dtype-normalized weight values).

    Low-precision dtypes are upcast to fp32 before hashing: ``numpy()`` cannot view ``bfloat16``
    (the documented A100/H100 training precision), so a raw-bytes hash would crash the determinism
    deliverable the instant a bf16 checkpoint is saved. The hash is therefore a *value* address
    (fp32-normalized), not a raw-bit address — uniform across the bf16/fp16/fp32 a run may produce.
    """
    parts: list[Any] = []
    for k in sorted(sd):
        parts.append(k)
        t = sd[k].detach().cpu().contiguous()
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.to(torch.float32)
        parts.append(t.numpy())
    return content_hash(*parts)


def artifact_hash(sd: dict[str, torch.Tensor], config: dict[str, Any]) -> str:
    """Content address binding weights AND the reconstruction config.

    Folding ``config`` into the hash means a *behavior-only* config field (e.g. the tokenizer's
    ``encoder_causal``, which changes the forward pass but no tensor) cannot silently differ between
    save and load — the guard catches it. So ``config`` must be the COMPLETE constructor kwargs
    (use ``model.get_config()``), not a hand-authored subset."""
    return content_hash(state_dict_hash(sd), config)


def save_checkpoint(path: str | Path, model: nn.Module, config: dict[str, Any]) -> str:
    """Save ``{state_dict, config, hash}`` to ``path``; return the content hash.

    ``config`` must be the exact keyword arguments that reconstruct ``model`` via its class
    constructor — pass ``model.get_config()`` so every architecture/behavior knob is captured."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    h = artifact_hash(sd, config)
    torch.save({"format": CKPT_FORMAT, "state_dict": sd, "config": config, "hash": h}, path)
    return h


@dataclass
class LoadedCheckpoint:
    model: nn.Module
    config: dict[str, Any]
    hash: str


def load_checkpoint(
    path: str | Path,
    model_factory: Callable[..., nn.Module],
    *,
    map_location: str = "cpu",
    expect_hash: str | None = None,
) -> LoadedCheckpoint:
    """Rebuild a module from a saved checkpoint and verify its content hash.

    ``model_factory(**config)`` reconstructs the module. The stored hash is re-derived from the
    loaded weights and checked against the recorded one (corruption guard); if ``expect_hash`` is
    given it must also match (the consume-the-right-artifact assert)."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if ckpt.get("format") != CKPT_FORMAT:
        raise ValueError(f"unrecognized checkpoint format: {ckpt.get('format')!r}")
    model = model_factory(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    rehash = artifact_hash(model.state_dict(), ckpt["config"])
    if rehash != ckpt["hash"]:
        raise ValueError(f"checkpoint hash mismatch on load: {rehash} != {ckpt['hash']}")
    if expect_hash is not None and rehash != expect_hash:
        raise ValueError(f"checkpoint is not the expected artifact: {rehash} != {expect_hash}")
    return LoadedCheckpoint(model=model, config=ckpt["config"], hash=rehash)


__all__ = [
    "CKPT_FORMAT",
    "LoadedCheckpoint",
    "artifact_hash",
    "load_checkpoint",
    "save_checkpoint",
    "state_dict_hash",
]
