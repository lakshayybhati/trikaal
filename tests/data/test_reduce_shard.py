"""Resumable, content-hashed shard reduction (M4a) — lossless persistence + cache logic.

Real-data shard==direct equivalence (on BTCUSDT) runs in the M4a dry-run script; here we pin the
persistence + resumability machinery on a synthetic stream (no aggTrades zip needed)."""

from __future__ import annotations

import json

import numpy as np

from trikaal.data.reduce_shard import (
    _shard_paths,
    load_shard_npz,
    save_shard_npz,
    shard_cache_meta,
    shard_content_hash,
    shard_plan,
)
from trikaal.data.synthetic import make_synthetic_stream


def test_npz_round_trip_is_byte_exact(tmp_path):
    stream, _ = make_synthetic_stream(seed=5, n_bars=500)
    p = tmp_path / "AAA" / "AAA-2023-01.npz"
    save_shard_npz(stream, p)
    back = load_shard_npz(p)
    for attr in (
        "bar_open_ms",
        "open",
        "high",
        "low",
        "close",
        "v_kline",
        "quote_volume",
        "v_buy",
        "v_sell",
        "n_buy",
        "n_sell",
    ):
        assert np.array_equal(getattr(back, attr), getattr(stream, attr)), attr
    assert len(back.sizes) == len(stream.sizes)
    for a, b in zip(back.sizes, stream.sizes, strict=True):
        assert np.array_equal(a, b)
    # the content hash is invariant under save/load (resumable integrity check)
    assert shard_content_hash(back) == shard_content_hash(stream)


def test_shard_hash_is_sensitive_to_size_pools():
    stream, _ = make_synthetic_stream(seed=6, n_bars=400)
    h0 = shard_content_hash(stream)
    # perturb a single trade in one bar's size pool → hash must move
    for i in range(len(stream.sizes)):
        if stream.sizes[i].size:
            stream.sizes[i][0] += 1.0
            break
    assert shard_content_hash(stream) != h0


def test_shard_plan_is_sorted_flat_worklist():
    plan = shard_plan({"BBB": ["2023-02", "2023-01"], "AAA": ["2023-01"]})
    assert plan == [("AAA", "2023-01"), ("BBB", "2023-01"), ("BBB", "2023-02")]


def _write_shard(tmp_path, symbol="AAA", period="2023-01", seed=7, n_bars=300):
    stream, _ = make_synthetic_stream(seed=seed, n_bars=n_bars)
    npz_path, side_path = _shard_paths(tmp_path, symbol, period)
    save_shard_npz(stream, npz_path)
    side_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "period": period,
                "n_bars": len(stream),
                "content_hash": shard_content_hash(stream),
            },
            sort_keys=True,
        )
    )
    return npz_path, side_path


def test_cache_meta_accepts_a_consistent_shard(tmp_path):
    npz_path, side_path = _write_shard(tmp_path)
    meta = shard_cache_meta(npz_path, side_path, "AAA", "2023-01")
    assert meta is not None and meta["n_bars"] == 300


def test_cache_meta_rejects_a_stale_hash(tmp_path):
    """A sidecar whose content_hash does not match the npz is treated as a miss (self-heal)."""
    npz_path, side_path = _write_shard(tmp_path)
    meta = json.loads(side_path.read_text())
    meta["content_hash"] = "sha256:deadbeef"  # raw re-issued / tampered → hash drift
    side_path.write_text(json.dumps(meta, sort_keys=True))
    assert shard_cache_meta(npz_path, side_path, "AAA", "2023-01") is None


def test_cache_meta_rejects_a_torn_npz(tmp_path):
    """A corrupt/half-written npz (with a valid sidecar) is rejected, not trusted."""
    npz_path, side_path = _write_shard(tmp_path)
    npz_path.write_bytes(b"not a real npz")  # torn write
    assert shard_cache_meta(npz_path, side_path, "AAA", "2023-01") is None


def test_cache_meta_misses_when_sidecar_absent(tmp_path):
    npz_path, side_path = _write_shard(tmp_path)
    side_path.unlink()  # npz present but no commit marker → miss
    assert shard_cache_meta(npz_path, side_path, "AAA", "2023-01") is None
