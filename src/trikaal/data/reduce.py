"""aggTrades → 1m bar reduction (data-pipeline §2.2/§2.3) — single-symbol slice.

Reduces Binance futures aggTrades to per-bar microstructure with Polars (taker buy/sell split via
``is_buyer_maker``, exact integer-ms bucketing ``bar_id = transact_time // 60000``), left-joins it
onto the canonical 1m kline grid, and emits a :class:`~trikaal.data.synthetic.RawStream` — the
SAME structure the synthetic fixture produces — so the validated ``compute_features`` transform
runs unchanged on real bars. Funding/OI are left empty (deferred), so their mask bits are set.

Real futures schema (grounded by inspection, not the spec's spot schema):
  aggTrades: agg_trade_id, price, quantity, first/last_trade_id, transact_time, is_buyer_maker
  klines:    open_time, open, high, low, close, volume, close_time, quote_volume, count,
             taker_buy_volume, taker_buy_quote_volume, ignore
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from trikaal.data.config import BAR_MS
from trikaal.data.synthetic import RawStream

_AGG_SCHEMA = {
    "agg_trade_id": pl.Int64,
    "price": pl.Float64,
    "quantity": pl.Float64,
    "first_trade_id": pl.Int64,
    "last_trade_id": pl.Int64,
    "transact_time": pl.Int64,
    "is_buyer_maker": pl.Boolean,
}
_AGG_COLS = list(_AGG_SCHEMA)
_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def _extract_csv(zip_path: Path, first_col: str) -> tuple[Path, bool]:
    """Extract the single CSV member to a temp file. Returns (path, has_header)."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            has_header = f.read(len(first_col)).decode("ascii", "ignore") == first_col
        out_dir = Path(tempfile.mkdtemp(prefix="trikaal_csv_"))
        z.extract(name, out_dir)
    return out_dir / name, has_header


def reduce_aggtrades(zip_path: Path) -> pl.DataFrame:
    """Per-bar microstructure from one aggTrades archive. Columns: bar_id, V_buy, V_sell, N_buy,
    N_sell, V_agg, N_trades, sizes(list[f64]). Sorted by bar_id."""
    csv_path, has_header = _extract_csv(zip_path, "agg_trade_id")
    try:
        lf = pl.scan_csv(
            csv_path,
            has_header=has_header,
            new_columns=None if has_header else _AGG_COLS,
            schema_overrides=_AGG_SCHEMA,
        ).select(["quantity", "transact_time", "is_buyer_maker"])
        buy = pl.col("is_buyer_maker").not_()  # taker BUY (§2.1): is_buyer_maker == False
        agg = (
            lf.with_columns((pl.col("transact_time") // BAR_MS).alias("bar_id"))
            .group_by("bar_id")
            .agg(
                pl.col("quantity").filter(buy).sum().alias("V_buy"),
                pl.col("quantity").filter(buy.not_()).sum().alias("V_sell"),
                buy.sum().alias("N_buy"),
                buy.not_().sum().alias("N_sell"),
                pl.col("quantity").sum().alias("V_agg"),
                pl.len().alias("N_trades"),
                pl.col("quantity").alias("sizes"),
            )
            .sort("bar_id")
            .collect(engine="streaming")
        )
    finally:
        csv_path.unlink(missing_ok=True)
        csv_path.parent.rmdir()
    return agg.with_columns(
        pl.col("V_buy").fill_null(0.0),
        pl.col("V_sell").fill_null(0.0),
        pl.col("N_buy").fill_null(0),
        pl.col("N_sell").fill_null(0),
    )


def load_klines(zip_path: Path) -> pl.DataFrame:
    """Canonical 1m kline grid: bar_open_ms, open, high, low, close, v_kline, quote_volume."""
    csv_path, has_header = _extract_csv(zip_path, "open_time")
    try:
        df = pl.read_csv(
            csv_path,
            has_header=has_header,
            new_columns=None if has_header else _KLINE_COLS,
            schema_overrides=dict.fromkeys(
                ("open", "high", "low", "close", "volume", "quote_volume"), pl.Float64
            ),
        )
    finally:
        csv_path.unlink(missing_ok=True)
        csv_path.parent.rmdir()
    return df.select(
        pl.col("open_time").cast(pl.Int64).alias("bar_open_ms"),
        pl.col("open"),
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        pl.col("volume").alias("v_kline"),
        pl.col("quote_volume"),
    ).sort("bar_open_ms")


def build_raw_stream(klines: pl.DataFrame, agg: pl.DataFrame, *, is_perp: bool = True) -> RawStream:
    """Left-join per-bar microstructure onto the kline grid → a RawStream for compute_features.

    klines are the canonical spine; bars with no aggTrades get zero microstructure + empty sizes.
    Multiple monthly frames must be concatenated (sorted, deduped) BEFORE calling this so segment
    /normalization state is continuous across month boundaries.
    """
    k = klines.with_columns((pl.col("bar_open_ms") // BAR_MS).alias("bar_id"))
    joined = k.join(agg, on="bar_id", how="left").sort("bar_open_ms")

    v_buy = joined["V_buy"].fill_null(0.0).to_numpy().astype(np.float64)
    v_sell = joined["V_sell"].fill_null(0.0).to_numpy().astype(np.float64)
    n_buy = joined["N_buy"].fill_null(0).to_numpy().astype(np.int64)
    n_sell = joined["N_sell"].fill_null(0).to_numpy().astype(np.int64)
    sizes_col = joined["sizes"].to_list()
    sizes = [
        np.asarray(s, dtype=np.float64) if s is not None else np.zeros(0, dtype=np.float64)
        for s in sizes_col
    ]
    empty = np.zeros(0, dtype=np.int64)
    return RawStream(
        bar_open_ms=joined["bar_open_ms"].to_numpy().astype(np.int64),
        open=joined["open"].to_numpy().astype(np.float64),
        high=joined["high"].to_numpy().astype(np.float64),
        low=joined["low"].to_numpy().astype(np.float64),
        close=joined["close"].to_numpy().astype(np.float64),
        v_kline=joined["v_kline"].to_numpy().astype(np.float64),
        quote_volume=joined["quote_volume"].to_numpy().astype(np.float64),
        v_buy=v_buy,
        v_sell=v_sell,
        n_buy=n_buy,
        n_sell=n_sell,
        sizes=sizes,
        is_perp=is_perp,
        funding_ts=empty.copy(),
        funding_val=np.zeros(0, dtype=np.float64),
        oi_ts=empty.copy(),
        oi_val=np.zeros(0, dtype=np.float64),
    )


def reduce_month(klines_zip: Path, agg_zip: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reduce one month → (klines_df, agg_df) for concatenation across months."""
    return load_klines(klines_zip), reduce_aggtrades(agg_zip)
