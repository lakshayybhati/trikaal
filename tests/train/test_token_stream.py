"""Token-stream materialization + content-hashed cache (§7.2 freeze boundary)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.token_stream import (
    cache_token_stream,
    load_token_stream,
    tokenize_features,
)
from trikaal.utils.seeding import set_determinism


def _toy():
    set_determinism(0)
    tok = TokenizerAE(d_model=32, n_layers=1, n_heads=2, d_ff=64).eval()
    n = 400
    x = np.random.default_rng(0).standard_normal((n, 16)).astype(np.float32)
    m = np.zeros((n, 16), dtype=np.uint8)
    seg = np.zeros(n, dtype=np.int64)
    seg[200:] = 1  # two segments → exercises per-segment chunking + a ragged tail
    ts = (1_600_000_000_000 + np.arange(n) * 60_000).astype(np.int64)
    return tok, x, m, seg, ts


def test_tokenize_features_shapes_and_determinism():
    tok, x, m, seg, _ = _toy()
    with torch.no_grad():
        b_c1, b_f1 = tokenize_features(tok, x, m, seg, window=128)
        b_c2, b_f2 = tokenize_features(tok, x, m, seg, window=128)
    assert b_c1.shape == (400,) and b_f1.shape == (400,)
    assert b_c1.dtype == np.int64 and b_f1.dtype == np.int64
    assert b_c1.max() < tok.v_c and b_f1.max() < tok.v_f and b_c1.min() >= 0
    assert np.array_equal(b_c1, b_c2) and np.array_equal(b_f1, b_f2)  # eval FSQ is deterministic


def test_token_stream_cache_roundtrip_and_provenance(tmp_path):
    tok, x, m, seg, ts = _toy()
    b_c, b_f = tokenize_features(tok, x, m, seg, window=128)
    h = cache_token_stream(
        tmp_path / "tok.npz",
        b_c,
        b_f,
        ts,
        seg,
        tokenizer_hash="sha256:aa",
        dataset_hash="sha256:bb",
        window=128,
    )
    loaded = load_token_stream(
        tmp_path / "tok.npz", tokenizer_hash="sha256:aa", dataset_hash="sha256:bb"
    )
    assert loaded["token_stream_hash"] == h
    assert np.array_equal(loaded["b_c"], b_c) and np.array_equal(loaded["b_f"], b_f)
    with pytest.raises(ValueError, match="different tokenizer"):
        load_token_stream(tmp_path / "tok.npz", tokenizer_hash="sha256:zz")
