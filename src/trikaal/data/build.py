"""End-to-end single-symbol dataset build: raw dumps → reduction → features → Parquet lake.

Ties §2.1 ingest, §2.2 reduction, the §1/§3/§5 causal feature transform, and the §3 Parquet lake
into one call for a single symbol over a set of months. Months are stitched in time order BEFORE
the feature transform so segment/normalization/tau state is continuous across month boundaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trikaal.data.config import FeatureConfig
from trikaal.data.features import PerBarOutput, compute_features
from trikaal.data.ingest import months_in_year
from trikaal.data.raw_store import RAW_ROOT_DEFAULT
from trikaal.data.reduce import build_raw_stream, load_klines, reduce_aggtrades
from trikaal.data.synthetic import RawStream


def _raw_paths(root: Path, symbol: str, period: str) -> tuple[Path, Path]:
    k = root / "um" / "klines" / symbol / f"{symbol}-1m-{period}.zip"
    a = root / "um" / "aggTrades" / symbol / f"{symbol}-aggTrades-{period}.zip"
    return k, a


def _concat_streams(parts: list[RawStream], is_perp: bool) -> RawStream:
    """Concatenate per-month RawStreams in time order (no overlap between monthly dumps)."""
    cat = np.concatenate
    sizes: list[np.ndarray] = []
    for p in parts:
        sizes.extend(p.sizes)
    empty = np.zeros(0, dtype=np.int64)
    return RawStream(
        bar_open_ms=cat([p.bar_open_ms for p in parts]),
        open=cat([p.open for p in parts]),
        high=cat([p.high for p in parts]),
        low=cat([p.low for p in parts]),
        close=cat([p.close for p in parts]),
        v_kline=cat([p.v_kline for p in parts]),
        quote_volume=cat([p.quote_volume for p in parts]),
        v_buy=cat([p.v_buy for p in parts]),
        v_sell=cat([p.v_sell for p in parts]),
        n_buy=cat([p.n_buy for p in parts]),
        n_sell=cat([p.n_sell for p in parts]),
        sizes=sizes,
        is_perp=is_perp,
        funding_ts=empty.copy(),
        funding_val=np.zeros(0, dtype=np.float64),
        oi_ts=empty.copy(),
        oi_val=np.zeros(0, dtype=np.float64),
    )


def build_symbol_stream(
    symbol: str,
    months: list[str],
    *,
    raw_root: Path = RAW_ROOT_DEFAULT,
    is_perp: bool = True,
) -> RawStream:
    """Reduce + stitch all months into one time-ordered RawStream (klines spine, agg attached).

    Built per-month then concatenated so the big Polars frames are freed promptly (memory-bounded).
    """
    parts: list[RawStream] = []
    for period in months:
        kz, az = _raw_paths(raw_root, symbol, period)
        kl = load_klines(kz)
        ag = reduce_aggtrades(az)
        parts.append(build_raw_stream(kl, ag, is_perp=is_perp))
        del kl, ag
    stream = _concat_streams(parts, is_perp)
    if not np.all(np.diff(stream.bar_open_ms) > 0):
        raise ValueError("stitched bar_open_ms not strictly increasing — month overlap/disorder")
    return stream


def build_symbol_features(
    symbol: str,
    months: list[str],
    *,
    cfg: FeatureConfig | None = None,
    raw_root: Path = RAW_ROOT_DEFAULT,
    is_perp: bool = True,
) -> tuple[PerBarOutput, RawStream]:
    """Full per-bar feature build for a symbol over ``months``. Returns (features, raw stream)."""
    cfg = cfg or FeatureConfig()
    stream = build_symbol_stream(symbol, months, raw_root=raw_root, is_perp=is_perp)
    out = compute_features(stream, cfg)
    if not np.all(np.isfinite(out.x_f64)):
        raise ValueError("non-finite features on real data — an eps/segment rule was violated")
    return out, stream


def year_months(year: int) -> list[str]:
    return months_in_year(year)
