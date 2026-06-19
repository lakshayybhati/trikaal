"""Resumable, content-hashed per-(symbol, month) aggTrades reduction (data-pipeline §2.2).

The aggTrades stream is the heavy one (≈ 1–3 GB/month/symbol compressed). At universe scale the
reduction is embarrassingly parallel over **shards** = one (symbol, month) each. This module wraps
the proven M2 reducer (``reduce_month`` + ``build_raw_stream``) with crash-safety:

  * each shard reduces to a per-month :class:`RawStream` persisted **losslessly** to ``.npz``
    (per-bar arrays stored as raw f64/i64; the ragged ``sizes`` pools flattened to a values+offsets
    pair — no pickle), plus a ``.shardhash.json`` sidecar recording the content hash + bar count;
  * a shard whose sidecar is present and consistent is **skipped** (status ``"cached"``), so a run
    that dies mid-universe resumes without recomputing finished shards (the cloud-money rule);
  * ``build_symbol_features_sharded`` concatenates the shards in time order and runs the *unchanged*
    ``compute_features`` — byte-identical to the in-memory ``build_symbol_features`` path (proven in
    the M4a dry-run on real BTCUSDT), so the shard store is a pure acceleration, never a divergence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trikaal.data.build import _concat_streams, _raw_paths
from trikaal.data.config import FeatureConfig
from trikaal.data.features import PerBarOutput, compute_features
from trikaal.data.raw_store import RAW_ROOT_DEFAULT
from trikaal.data.reduce import build_raw_stream, load_klines, reduce_aggtrades
from trikaal.data.synthetic import RawStream
from trikaal.utils.hashing import content_hash

SHARD_ROOT_DEFAULT = Path("processed/shards")


def shard_content_hash(stream: RawStream) -> str:
    """Content hash of one shard's RawStream (per-bar arrays + flattened size pools)."""
    sizes_flat = np.concatenate(stream.sizes) if stream.sizes else np.zeros(0, dtype=np.float64)
    offsets = np.cumsum([0, *[s.size for s in stream.sizes]]).astype(np.int64)
    return content_hash(
        stream.bar_open_ms,
        stream.open,
        stream.high,
        stream.low,
        stream.close,
        stream.v_kline,
        stream.quote_volume,
        stream.v_buy,
        stream.v_sell,
        stream.n_buy,
        stream.n_sell,
        sizes_flat,
        offsets,
    )


def save_shard_npz(stream: RawStream, path: Path) -> None:
    """Lossless single-month RawStream → ``.npz`` (no pickle; ragged sizes as values+offsets)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes_flat = np.concatenate(stream.sizes) if stream.sizes else np.zeros(0, dtype=np.float64)
    offsets = np.cumsum([0, *[s.size for s in stream.sizes]]).astype(np.int64)
    np.savez(
        path,
        bar_open_ms=stream.bar_open_ms,
        open=stream.open,
        high=stream.high,
        low=stream.low,
        close=stream.close,
        v_kline=stream.v_kline,
        quote_volume=stream.quote_volume,
        v_buy=stream.v_buy,
        v_sell=stream.v_sell,
        n_buy=stream.n_buy,
        n_sell=stream.n_sell,
        sizes_flat=sizes_flat,
        sizes_offsets=offsets,
        is_perp=np.array([stream.is_perp]),
    )


def load_shard_npz(path: Path) -> RawStream:
    """Inverse of :func:`save_shard_npz`. Funding/OI are empty (per-month shards carry none)."""
    z = np.load(path)
    offsets = z["sizes_offsets"]
    flat = z["sizes_flat"]
    sizes = [flat[offsets[i] : offsets[i + 1]].copy() for i in range(offsets.size - 1)]
    empty = np.zeros(0, dtype=np.int64)
    return RawStream(
        bar_open_ms=z["bar_open_ms"],
        open=z["open"],
        high=z["high"],
        low=z["low"],
        close=z["close"],
        v_kline=z["v_kline"],
        quote_volume=z["quote_volume"],
        v_buy=z["v_buy"],
        v_sell=z["v_sell"],
        n_buy=z["n_buy"],
        n_sell=z["n_sell"],
        sizes=sizes,
        is_perp=bool(z["is_perp"][0]),
        funding_ts=empty.copy(),
        funding_val=np.zeros(0, dtype=np.float64),
        oi_ts=empty.copy(),
        oi_val=np.zeros(0, dtype=np.float64),
    )


@dataclass(frozen=True)
class ShardResult:
    symbol: str
    period: str
    n_bars: int
    content_hash: str
    status: str  # "computed" | "cached"
    npz_path: Path


def _shard_paths(shard_root: Path, symbol: str, period: str) -> tuple[Path, Path]:
    base = shard_root / symbol
    return base / f"{symbol}-{period}.npz", base / f"{symbol}-{period}.shardhash.json"


def shard_cache_meta(npz_path: Path, side_path: Path, symbol: str, period: str) -> dict | None:
    """Return the sidecar meta iff a present shard is **content-verified**, else ``None``.

    A clean kill writes the npz first then the sidecar (the commit marker), so a half-written npz
    has no sidecar → miss. This additionally re-hashes the npz and checks it against the recorded
    ``content_hash`` (self-healing against a corrupt npz or a re-issued raw file) — a torn or
    tampered shard is treated as a miss and recomputed (the crash-safety rule).
    """
    if not (npz_path.exists() and side_path.exists()):
        return None
    try:
        meta = json.loads(side_path.read_text())
        if meta.get("symbol") != symbol or meta.get("period") != period:
            return None
        if shard_content_hash(load_shard_npz(npz_path)) != meta.get("content_hash"):
            return None  # torn / stale npz — recompute
    except (OSError, ValueError, KeyError):
        return None
    return meta


def reduce_shard(
    symbol: str,
    period: str,
    *,
    raw_root: Path = RAW_ROOT_DEFAULT,
    shard_root: Path = SHARD_ROOT_DEFAULT,
    is_perp: bool = True,
    force: bool = False,
) -> ShardResult:
    """Reduce one (symbol, month) to a content-hashed shard. Resumable: a present, **content-
    verified** sidecar is reused (status ``"cached"``) unless ``force``."""
    npz_path, side_path = _shard_paths(shard_root, symbol, period)
    if not force:
        meta = shard_cache_meta(npz_path, side_path, symbol, period)
        if meta is not None:
            return ShardResult(
                symbol=symbol,
                period=period,
                n_bars=int(meta["n_bars"]),
                content_hash=meta["content_hash"],
                status="cached",
                npz_path=npz_path,
            )
    kz, az = _raw_paths(raw_root, symbol, period)
    stream = build_raw_stream(load_klines(kz), reduce_aggtrades(az), is_perp=is_perp)
    h = shard_content_hash(stream)
    save_shard_npz(stream, npz_path)
    side_path.write_text(
        json.dumps(
            {"symbol": symbol, "period": period, "n_bars": len(stream), "content_hash": h},
            sort_keys=True,
        )
    )
    return ShardResult(symbol, period, len(stream), h, "computed", npz_path)


def build_symbol_features_sharded(
    symbol: str,
    months: list[str],
    *,
    cfg: FeatureConfig | None = None,
    raw_root: Path = RAW_ROOT_DEFAULT,
    shard_root: Path = SHARD_ROOT_DEFAULT,
    is_perp: bool = True,
    force: bool = False,
) -> tuple[PerBarOutput, list[ShardResult]]:
    """Shard-store path: reduce each month (resumable) → concat in time order → compute_features.

    Byte-identical to ``build_symbol_features`` — the concat + transform are the same code; only
    the per-month reduce is cached. Returns ``(features, shard_results)``.
    """
    cfg = cfg or FeatureConfig()
    results = [
        reduce_shard(
            symbol, p, raw_root=raw_root, shard_root=shard_root, is_perp=is_perp, force=force
        )
        for p in months
    ]
    parts = [load_shard_npz(_shard_paths(shard_root, symbol, p)[0]) for p in months]
    stream = _concat_streams(parts, is_perp)
    if not np.all(np.diff(stream.bar_open_ms) > 0):
        raise ValueError("stitched shards not strictly increasing — month overlap/disorder")
    out = compute_features(stream, cfg)
    return out, results


def shard_plan(symbol_to_months: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Flat list of (symbol, period) shards — the parallel work-list for M4b (one task each)."""
    return [(sym, p) for sym in sorted(symbol_to_months) for p in sorted(symbol_to_months[sym])]


__all__ = [
    "SHARD_ROOT_DEFAULT",
    "ShardResult",
    "build_symbol_features_sharded",
    "load_shard_npz",
    "reduce_shard",
    "save_shard_npz",
    "shard_cache_meta",
    "shard_content_hash",
    "shard_plan",
]
