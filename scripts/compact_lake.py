"""Compact the M4b universe lake — day-partitioned (211k tiny files) → month-partitioned — with a
HARD content-preservation gate. The lake cost ~52.8 h to build, so this NEVER touches the source: it
writes a NEW directory, then proves every symbol's dataset_hash and the universe Merkle are
byte-identical to the ledger anchor BEFORE any swap (the swap is a manual `mv`, done only on PASS).

Why this is safe as a pure file-layout change: dataset_hash = content_hash(x, m, ts) is over the
per-bar arrays in bar_open_ms order, regardless of file grouping. assemble_window re-sorts by
bar_open_ms, so reshuffling rows across month files cannot change any hash. The gate
re-derives all 200 hashes from the compacted lake and asserts the Merkle is unchanged; a single
altered/dropped/duplicated bar would flip a hash and FAIL.

Usage:
  PYTHONPATH=src python3 scripts/compact_lake.py \
      [--src processed/universe_bars] [--dst processed/universe_bars_compacted] \
      [--ledger processed/universe/ingest_ledger_2021-01-01_2025-01-01.jsonl] \
      [--expected-merkle sha256:5dfd667d…] [--skip-compact]
Exit 0 = PASS (anchors to the Merkle; safe to swap). Exit 1 = FAIL (keep the original).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import duckdb

from trikaal.data.lake import assemble_window
from trikaal.data.lake import connect as lake_connect
from trikaal.utils.hashing import content_hash

ANCHOR = "sha256:5dfd667d05b97bda5a35946433b96d40738df8a926d5fa8a8d9dfc924f5a04b7"


def ledger_hashes(path: Path) -> dict[str, str]:
    built: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("event") == "lake_built":
                built[rec["symbol"]] = rec["dataset_hash"]
    return built


def compact(src: Path, dst: Path) -> None:
    """ONE streaming pass over the source (globs the day-partition files once) → month-partitioned
    output. `date` stays a data column; (symbol, frequency, month) go to the partition path."""
    if dst.exists():
        shutil.rmtree(dst)  # dst is regenerable output; never the source
    con = duckdb.connect()
    # 7024 (symbol,month) partitions; keep them all open so each is ONE file (ulimit -n is 1M here).
    con.execute("PRAGMA partitioned_write_max_open_files=8192")
    con.execute(
        f"""
        COPY (
          SELECT *, substr(CAST(date AS VARCHAR), 1, 7) AS month
          FROM read_parquet('{src}/**/*.parquet', hive_partitioning=true)
        )
        TO '{dst}'
        (FORMAT PARQUET, PARTITION_BY (symbol, frequency, month), COMPRESSION zstd,
         OVERWRITE_OR_IGNORE)
        """
    )
    con.close()


def symbol_hash(con: duckdb.DuckDBPyConnection, sym: str) -> tuple[str, int]:
    """Re-derive a symbol's dataset_hash from the lake exactly as build_one_symbol did:
    content_hash(x, m, ts) over assemble_window's bar_open_ms-ordered arrays (proven equal by
    tests/data/test_universe.py::test_universe_lake_roundtrip_per_symbol)."""
    d = assemble_window(con, sym, 0, 10**15)
    return content_hash(d["x"], d["mask"], d["ts"]), len(d["ts"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="processed/universe_bars")
    ap.add_argument("--dst", default="processed/universe_bars_compacted")
    ap.add_argument(
        "--ledger", default="processed/universe/ingest_ledger_2021-01-01_2025-01-01.jsonl"
    )
    ap.add_argument("--expected-merkle", default=ANCHOR)
    ap.add_argument("--skip-compact", action="store_true", help="verify an already-written --dst")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    led = ledger_hashes(Path(args.ledger))
    n_src = sum(1 for _ in src.rglob("*.parquet"))
    print(f"ledger symbols: {len(led)} | src: {src} ({n_src:,} parquet files)")

    if not args.skip_compact:
        print(f"compacting → {dst} (one source pass, month-partitioned, zstd) …")
        compact(src, dst)
    n_dst = sum(1 for _ in dst.rglob("*.parquet"))
    print(f"dst: {dst} ({n_dst:,} parquet files) — {n_src / max(1, n_dst):.0f}x fewer")

    # HARD GATE: every symbol's dataset_hash from the COMPACTED lake must match the ledger
    con = lake_connect(dst)
    lake_h: dict[str, str] = {}
    mism: list[str] = []
    tot = 0
    for i, sym in enumerate(sorted(led), 1):
        h, n = symbol_hash(con, sym)
        lake_h[sym] = h
        tot += n
        if h != led[sym]:
            mism.append(sym)
        if i % 50 == 0:
            print(f"  verified {i}/{len(led)} symbols …")
    merkle = content_hash([[s, lake_h[s]] for s in sorted(lake_h)])

    print(f"\ntotal bars (compacted) : {tot:,}")
    print(f"per-symbol hash mismatches: {len(mism)}  {mism[:10]}")
    print(f"compacted universe Merkle : {merkle}")
    print(f"expected anchor           : {args.expected_merkle}")
    ok = (not mism) and merkle == args.expected_merkle
    if ok:
        print("\nPASS — compacted lake is byte-for-byte content-identical (Merkle preserved).")
        print(f"  Safe to swap:  mv {src} {src}_orig && mv {dst} {src}")
        return 0
    print("\nFAIL — content differs from the anchor. KEEP THE ORIGINAL; do NOT swap.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
