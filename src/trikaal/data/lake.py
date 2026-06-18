"""Parquet lake (§3) + DuckDB query layer (§4) — single-symbol slice.

Per-bar features land in a Hive-partitioned Parquet lake keyed by ``(symbol, frequency, date)``
(zstd). DuckDB is the single read interface: a view over the lake plus a split-aware temporal-slice
assembly query that returns exactly the model-visible + hook columns in ``bar_open_ms`` order.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from trikaal.constants import N_FEATURES
from trikaal.data.features import PerBarOutput

PROCESSED_ROOT_DEFAULT = Path("processed/bars")


def _date_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).strftime("%Y-%m-%d")


def perbar_to_arrow(out: PerBarOutput, symbol: str, frequency: str = "1m") -> pa.Table:
    """Per-bar features → an Arrow table with the §3.2 column schema + partition columns."""
    ts = out.ts
    cols: dict[str, object] = {
        "bar_open_ms": ts,
        "symbol": pa.array([symbol] * len(out)).dictionary_encode(),
        "frequency": pa.array([frequency] * len(out)).dictionary_encode(),
        "date": [_date_utc(int(t)) for t in ts],
        "segment_id": out.segment_id,
    }
    for i in range(N_FEATURES):
        cols[f"x_{i}"] = out.x[:, i]
    for i in range(N_FEATURES):
        cols[f"m_{i}"] = out.m[:, i]
    cols["raw_ret_close"] = out.raw[:, 0].astype(np.float32)
    cols["raw_range"] = out.raw[:, 1].astype(np.float32)
    cols["sigma"] = out.sigma.astype(np.float32)  # divisor (hook), kept for target de-norm
    cols["is_stale"] = out.is_stale  # QA only
    cols["dq_flags"] = out.dq_flags  # QA only
    return pa.table(cols)


def write_lake(
    out: PerBarOutput, symbol: str, *, root: Path = PROCESSED_ROOT_DEFAULT, frequency: str = "1m"
) -> Path:
    """Write per-bar features to the Hive-partitioned, zstd Parquet lake."""
    table = perbar_to_arrow(out, symbol, frequency)
    root.mkdir(parents=True, exist_ok=True)
    pads.write_dataset(
        table,
        root,
        format="parquet",
        partitioning=["symbol", "frequency", "date"],
        partitioning_flavor="hive",
        existing_data_behavior="overwrite_or_ignore",
        file_options=pads.ParquetFileFormat().make_write_options(compression="zstd"),
    )
    return root


def connect(root: Path = PROCESSED_ROOT_DEFAULT) -> duckdb.DuckDBPyConnection:
    """A DuckDB connection with a ``bars`` view over the whole lake (§4.1)."""
    con = duckdb.connect()
    glob = f"{root}/**/*.parquet"
    con.execute(f"CREATE VIEW bars AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)")
    return con


def assemble_window(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    lo_ms: int,
    hi_ms: int,
    *,
    frequency: str = "1m",
) -> dict[str, np.ndarray]:
    """Split-aware temporal slice (§4.1a): model-visible + hook columns, ``bar_open_ms`` order.

    Returns ``{x:[N,16] f32, mask:[N,16] u8, ts:[N] i64, segment_id:[N] i64}``. QA-only columns
    (is_stale, dq_flags, vol_recon_err) are NEVER selected into the training path.
    """
    xcols = ", ".join(f"x_{i}" for i in range(N_FEATURES))
    mcols = ", ".join(f"m_{i}" for i in range(N_FEATURES))
    df = con.execute(
        f"SELECT bar_open_ms, segment_id, {xcols}, {mcols} "
        "FROM bars WHERE symbol = ? AND frequency = ? AND bar_open_ms >= ? AND bar_open_ms < ? "
        "ORDER BY bar_open_ms",
        [symbol, frequency, lo_ms, hi_ms],
    ).fetch_arrow_table()
    n = df.num_rows
    x = np.empty((n, N_FEATURES), dtype=np.float32)
    m = np.empty((n, N_FEATURES), dtype=np.uint8)
    for i in range(N_FEATURES):
        x[:, i] = df.column(f"x_{i}").to_numpy()
        m[:, i] = df.column(f"m_{i}").to_numpy()
    return {
        "x": x,
        "mask": m,
        "ts": df.column("bar_open_ms").to_numpy().astype(np.int64),
        "segment_id": df.column("segment_id").to_numpy().astype(np.int64),
    }
