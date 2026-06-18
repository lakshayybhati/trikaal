"""Content-hashing — every dataset/config/checkpoint is content-addressed (§6, §7.7).

Determinism is a deliverable: a run is reproducible from one config + a pinned seed, and
every produced dataset is content-hashed so any past prediction can be reconstructed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding (sorted keys, no whitespace drift) for hashing configs."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(*parts: Any) -> str:
    """sha256 over a sequence of parts.

    Each part is normalised: numpy arrays by their raw bytes + dtype + shape; bytes as-is;
    everything else via canonical JSON. Returns ``sha256:<hex>``.
    """
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            arr = np.ascontiguousarray(part)
            h.update(b"\x00ndarray")
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        elif isinstance(part, (bytes, bytearray)):
            h.update(b"\x00bytes")
            h.update(part)
        else:
            h.update(b"\x00json")
            h.update(canonical_json(part).encode())
    return f"sha256:{h.hexdigest()}"
