"""N-symbol dataset build (data-pipeline §2/§3) — the generalization of ``build.py``.

This is a **pure fan-out** over the single-symbol path: each symbol's raw stream is truncated to
its delisting-aware active window and run through the *unchanged* ``compute_features`` transform,
then written to its own ``(symbol, frequency, date)`` Parquet partition. There is NO cross-symbol
state: no global normalization fit, no universe-wide statistic, no shared tau pool. The frozen
stats are per-(symbol, feature), computed inside ``compute_features`` on one stream.

Consequences that the tests pin:
  * **regression** — building BTCUSDT alone through here reproduces M2's ``dataset_hash``
    bit-exactly (the active window for a live symbol is a no-op, so the stream and transform are
    byte-identical to ``build_symbol_features``);
  * **per-symbol independence** — a symbol's ``dataset_hash`` is identical whether it is built
    alone or inside an N-symbol universe (no symbol can perturb another's statistics).

The universe hash is a Merkle-style combine over the symbol-sorted ``(symbol, dataset_hash)``
pairs, so the whole dataset is content-addressed and any past prediction is reconstructible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trikaal.data.build import build_symbol_stream
from trikaal.data.config import FeatureConfig
from trikaal.data.features import compute_features
from trikaal.data.lake import PROCESSED_ROOT_DEFAULT, write_lake
from trikaal.data.raw_store import RAW_ROOT_DEFAULT
from trikaal.data.synthetic import RawStream
from trikaal.data.universe import SymbolSpec, UniverseSpec
from trikaal.utils.hashing import content_hash


def slice_stream_by_ms(stream: RawStream, lo_ms: float, hi_ms: float) -> RawStream:
    """Keep bars with ``lo_ms <= bar_open_ms < hi_ms`` (delisting-truncation; §1.3).

    Per-bar arrays and the ragged ``sizes`` list are sliced together. Funding/OI events are kept
    on their own effective timestamps within ``[lo_ms, close_of_last_kept_bar]`` — strictly
    causal: an event after the last kept bar's close cannot inform any kept bar. Head-truncation
    (drop pre-listing bars) correctly restarts per-symbol warm-up at the listing edge; for a
    fully-live symbol (lo=-inf, hi=+inf) this is the identity, preserving the M2 byte-stream.
    """
    bom = stream.bar_open_ms.astype(np.int64)
    keep = (bom >= lo_ms) & (bom < hi_ms)
    if keep.all():
        return stream
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        raise ValueError("active window removes every bar — symbol has no live bars in window")
    last_close = int(bom[idx[-1]]) + 60_000  # close of last kept bar
    fmask = (stream.funding_ts >= lo_ms) & (stream.funding_ts <= last_close)
    omask = (stream.oi_ts >= lo_ms) & (stream.oi_ts <= last_close)
    return RawStream(
        bar_open_ms=stream.bar_open_ms[keep].copy(),
        open=stream.open[keep].copy(),
        high=stream.high[keep].copy(),
        low=stream.low[keep].copy(),
        close=stream.close[keep].copy(),
        v_kline=stream.v_kline[keep].copy(),
        quote_volume=stream.quote_volume[keep].copy(),
        v_buy=stream.v_buy[keep].copy(),
        v_sell=stream.v_sell[keep].copy(),
        n_buy=stream.n_buy[keep].copy(),
        n_sell=stream.n_sell[keep].copy(),
        sizes=[stream.sizes[i].copy() for i in idx],
        is_perp=stream.is_perp,
        funding_ts=stream.funding_ts[fmask].copy(),
        funding_val=stream.funding_val[fmask].copy(),
        oi_ts=stream.oi_ts[omask].copy(),
        oi_val=stream.oi_val[omask].copy(),
    )


@dataclass(frozen=True)
class SymbolBuild:
    """Per-symbol build provenance (one row of the universe manifest)."""

    symbol: str
    n_bars: int
    n_segments: int
    first_bar_ms: int
    last_bar_ms: int
    dataset_hash: str
    lo_ms: float
    hi_ms: float
    head_truncated: bool  # pre-listing bars dropped
    tail_truncated: bool  # post-delisting bars dropped


@dataclass(frozen=True)
class UniverseBuildResult:
    per_symbol: dict[str, SymbolBuild]
    universe_hash: str
    n_symbols: int
    total_bars: int

    def symbol_hashes(self) -> dict[str, str]:
        return {s: b.dataset_hash for s, b in self.per_symbol.items()}


def _merkle_universe_hash(symbol_hashes: dict[str, str]) -> str:
    """Order-independent combine over per-symbol dataset hashes (symbol-sorted JSON list)."""
    return content_hash([[s, symbol_hashes[s]] for s in sorted(symbol_hashes)])


def build_one_symbol(
    stream: RawStream,
    spec: SymbolSpec,
    *,
    window_start: str,
    window_end: str,
    cfg: FeatureConfig,
    root: Path | None,
    frequency: str = "1m",
) -> SymbolBuild:
    """Truncate one stream to its active window, run the per-symbol transform, optionally write.

    ``root=None`` skips the lake write (pure feature/hash check). The ``dataset_hash`` is exactly
    ``content_hash(x, m, ts)`` — the same triple M2 hashed — so a live symbol matches M2.
    """
    bom = stream.bar_open_ms.astype(np.int64)
    lo, hi = spec.active_window_ms(window_start, window_end)
    head = bool((bom < lo).any())
    tail = bool((bom >= hi).any())
    sliced = slice_stream_by_ms(stream, lo, hi)
    out = compute_features(sliced, cfg)
    if not np.all(np.isfinite(out.x_f64)):
        raise ValueError(f"non-finite features for {spec.symbol} — an eps/segment rule broke")
    if root is not None:
        write_lake(out, spec.symbol, root=root, frequency=frequency)
    return SymbolBuild(
        symbol=spec.symbol,
        n_bars=len(out),
        n_segments=int(out.segment_id.max()) + 1 if len(out) else 0,
        first_bar_ms=int(out.ts[0]),
        last_bar_ms=int(out.ts[-1]),
        dataset_hash=content_hash(out.x, out.m, out.ts),
        lo_ms=lo,
        hi_ms=hi,
        head_truncated=head,
        tail_truncated=tail,
    )


def build_universe_from_streams(
    streams: dict[str, RawStream],
    specs: dict[str, SymbolSpec],
    *,
    window_start: str,
    window_end: str,
    cfg: FeatureConfig | None = None,
    root: Path | None = PROCESSED_ROOT_DEFAULT,
    frequency: str = "1m",
) -> UniverseBuildResult:
    """Core N-symbol build: per-symbol independent transform + Merkle universe hash.

    Symbols are processed in sorted order. Each goes through ``build_one_symbol`` in complete
    isolation — the only shared object is ``cfg`` (read-only). This is the function the synthetic
    multi-symbol test drives; the real path (:func:`build_universe`) just supplies the streams.
    """
    cfg = cfg or FeatureConfig()
    builds: dict[str, SymbolBuild] = {}
    for sym in sorted(streams):
        builds[sym] = build_one_symbol(
            streams[sym],
            specs[sym],
            window_start=window_start,
            window_end=window_end,
            cfg=cfg,
            root=root,
            frequency=frequency,
        )
    symbol_hashes = {s: b.dataset_hash for s, b in builds.items()}
    return UniverseBuildResult(
        per_symbol=builds,
        universe_hash=_merkle_universe_hash(symbol_hashes),
        n_symbols=len(builds),
        total_bars=sum(b.n_bars for b in builds.values()),
    )


def build_universe(
    spec: UniverseSpec,
    *,
    cfg: FeatureConfig | None = None,
    raw_root: Path = RAW_ROOT_DEFAULT,
    root: Path | None = PROCESSED_ROOT_DEFAULT,
) -> UniverseBuildResult:
    """Real path: reduce each active symbol's months from the raw store, then fan-out the build.

    Each symbol's stream is built ONLY from its active months (delisting-aware) via the existing
    ``build_symbol_stream`` (unchanged M2 reducer), then truncated to ``[listing, delisting)`` and
    transformed. A fully-live symbol (e.g. BTCUSDT over 2023) loads exactly its window months and
    the truncation is a no-op ⇒ byte-identical to ``build_symbol_features`` ⇒ M2 hash preserved.
    """
    streams: dict[str, RawStream] = {}
    specs: dict[str, SymbolSpec] = {}
    for s in spec.active_symbols():
        months = s.active_months(spec.window_start, spec.window_end)
        if not months:
            continue
        streams[s.symbol] = build_symbol_stream(
            s.symbol, months, raw_root=raw_root, is_perp=s.perp_available
        )
        specs[s.symbol] = s
    return build_universe_from_streams(
        streams,
        specs,
        window_start=spec.window_start,
        window_end=spec.window_end,
        cfg=cfg,
        root=root,
        frequency=spec.frequency,
    )


__all__ = [
    "SymbolBuild",
    "UniverseBuildResult",
    "build_one_symbol",
    "build_universe",
    "build_universe_from_streams",
    "slice_stream_by_ms",
]
