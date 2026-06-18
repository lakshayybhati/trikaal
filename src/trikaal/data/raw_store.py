"""Immutable raw store + ingest manifest (data-pipeline §1).

Raw bytes are written once under a content-addressed path and never mutated; all downstream
stages read raw and none write back. Every fetched file is recorded in an append-only JSONL
manifest with its SHA-256, byte count, source URL, and fetch time — the head of the
content-hash chain that makes any past prediction reconstructible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

RAW_ROOT_DEFAULT = Path("raw/binance")


@dataclass(frozen=True)
class RawFile:
    symbol: str
    market: str  # "um" (USDⓂ perp) | "spot"
    stream: str  # "klines" | "aggTrades"
    period: str  # "YYYY-MM"
    url: str
    sha256: str
    n_bytes: int
    local_path: str
    fetched_at_utc: str


def raw_zip_path(root: Path, symbol: str, market: str, stream: str, filename: str) -> Path:
    """``raw/binance/{market}/{stream}/{SYMBOL}/{filename}`` — the immutable destination."""
    return root / market / stream / symbol / filename


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def append_manifest(root: Path, run_id: str, record: RawFile) -> None:
    manifest_dir = root / "_ingest_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), sort_keys=True)
    with (manifest_dir / f"{run_id}.jsonl").open("a") as f:
        f.write(line + "\n")
