"""Materialize the frozen-tokenizer token stream for Stage-2 (§7.2 — the freeze boundary).

Stage 2 trains on the *discrete* ``(coarse, fine)`` id stream, never the encoder weights. We
tokenize the lake once with the frozen tokenizer in **non-overlapping windows of the tokenizer's
training length** (``L_tok = 512`` for the shipped tokenizer; this line read 128 through the
M3 era and that value is no longer anything's training length), in eval mode (FSQ
rounding is deterministic),
and content-hash the result keyed to ``(tokenizer_hash, dataset_hash, window)`` so a re-run
reloads the identical stream and any downstream number is reconstructable.

Causal note (§Tokenizer-a): the encoder is bidirectional within a window, so ``b_t`` is a
*contextual* code. This is by design — ``b_t`` is the AR *label*, predicted from ``b_{<t}``; the
defensible-artifact ``encoder_causal=True`` variant is an M6 ablation knob, not an M3 concern.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from trikaal.data.segments import segment_bounds
from trikaal.tokenizer.model import TokenizerAE
from trikaal.utils.hashing import content_hash


def _chunk_ranges(segment_id: np.ndarray, window: int) -> list[tuple[int, int]]:
    """Non-overlapping ``[start, end)`` chunks of length ``<= window``, never crossing a segment."""
    ranges: list[tuple[int, int]] = []
    for s0, s1 in segment_bounds(segment_id):
        for start in range(s0, s1, window):
            ranges.append((start, min(start + window, s1)))
    return ranges


@torch.no_grad()
def tokenize_features(
    tok: TokenizerAE,
    x_np: np.ndarray,
    m_np: np.ndarray,
    segment_id: np.ndarray,
    *,
    window: int,
    batch_windows: int = 128,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize ``[N,16]`` features → ``(b_c[N], b_f[N])`` int64 with the frozen tokenizer.

    ``window`` IS REQUIRED, and used to default to 128 — the Stage-1 tokenizer length from
    the M3 era. The shipped tokenizer's ``max_len`` is **512** and every call site in this
    repository passes 512, so the default could only ever be taken by a caller who did not
    know it existed, and taking it silently tokenized at a quarter of the shipped context.
    Nothing errors when that happens: the ids are valid, the stream hashes cleanly, and the
    numbers are simply built on less context than the model was trained for. There is no
    safe default here, so there is no default.
    """
    tok.eval()
    dev = torch.device(device)
    tok.to(dev)
    n = x_np.shape[0]
    b_c = np.empty(n, dtype=np.int64)
    b_f = np.empty(n, dtype=np.int64)
    x = torch.from_numpy(x_np.astype(np.float32))
    m = torch.from_numpy(m_np.astype(np.float32))
    ranges = _chunk_ranges(segment_id, window)
    full = [(a, b) for a, b in ranges if b - a == window]
    tail = [(a, b) for a, b in ranges if b - a < window]

    for i in range(0, len(full), batch_windows):
        chunk = full[i : i + batch_windows]
        starts = np.array([a for a, _ in chunk], dtype=np.int64)
        idx = starts[:, None] + np.arange(window)[None, :]
        ti = torch.from_numpy(idx)
        cidx, fidx = tok.encode_tokens(x[ti].to(dev), m[ti].to(dev))
        cidx, fidx = cidx.cpu().numpy(), fidx.cpu().numpy()
        for j, (a, b) in enumerate(chunk):
            b_c[a:b], b_f[a:b] = cidx[j], fidx[j]
    for a, b in tail:  # ragged segment-tail windows, one at a time
        cidx, fidx = tok.encode_tokens(x[a:b][None].to(dev), m[a:b][None].to(dev))
        b_c[a:b], b_f[a:b] = cidx[0].cpu().numpy(), fidx[0].cpu().numpy()
    return b_c, b_f


def token_stream_hash(
    b_c: np.ndarray, b_f: np.ndarray, *, tokenizer_hash: str, dataset_hash: str, window: int
) -> str:
    """Content hash of the materialized stream, bound to its provenance."""
    return content_hash(
        b_c, b_f, {"tokenizer": tokenizer_hash, "dataset": dataset_hash, "window": window}
    )


def cache_token_stream(
    path: str | Path,
    b_c: np.ndarray,
    b_f: np.ndarray,
    ts: np.ndarray,
    segment_id: np.ndarray,
    *,
    tokenizer_hash: str,
    dataset_hash: str,
    window: int,
) -> str:
    """Save the stream + provenance to ``path`` (.npz); return its content hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = token_stream_hash(
        b_c, b_f, tokenizer_hash=tokenizer_hash, dataset_hash=dataset_hash, window=window
    )
    np.savez(
        path,
        b_c=b_c,
        b_f=b_f,
        ts=ts,
        segment_id=segment_id,
        tokenizer_hash=tokenizer_hash,
        dataset_hash=dataset_hash,
        window=np.int64(window),
        token_stream_hash=h,
    )
    return h


def load_token_stream(
    path: str | Path, *, tokenizer_hash: str | None = None, dataset_hash: str | None = None
) -> dict:
    """Load a cached stream and verify its content hash (and optionally its provenance)."""
    d = np.load(Path(path), allow_pickle=False)
    b_c, b_f = d["b_c"], d["b_f"]
    window = int(d["window"])
    tok_h, data_h = str(d["tokenizer_hash"]), str(d["dataset_hash"])
    rehash = token_stream_hash(b_c, b_f, tokenizer_hash=tok_h, dataset_hash=data_h, window=window)
    if rehash != str(d["token_stream_hash"]):
        raise ValueError("token-stream cache corrupted (hash mismatch)")
    if tokenizer_hash is not None and tok_h != tokenizer_hash:
        raise ValueError(f"token stream was built with a different tokenizer: {tok_h}")
    if dataset_hash is not None and data_h != dataset_hash:
        raise ValueError(f"token stream was built from a different dataset: {data_h}")
    return {
        "b_c": b_c,
        "b_f": b_f,
        "ts": d["ts"],
        "segment_id": d["segment_id"],
        "tokenizer_hash": tok_h,
        "dataset_hash": data_h,
        "window": window,
        "token_stream_hash": rehash,
    }


__all__ = [
    "cache_token_stream",
    "load_token_stream",
    "token_stream_hash",
    "tokenize_features",
]
