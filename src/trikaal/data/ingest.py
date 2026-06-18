"""Binance dumps ingest (data-pipeline §2.1) — single-symbol slice.

Downloads monthly klines + aggTrades archives for one symbol from ``data.binance.vision``,
verifies each against its published ``.CHECKSUM`` (SHA-256; a mismatch hard-fails, never a silent
skip), and writes them into the immutable raw store. Idempotent: a verified file already present
is not re-downloaded. The funding/OI futures-API path (§1.2) is deliberately deferred.
"""

from __future__ import annotations

import ssl
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import certifi

from trikaal.data.raw_store import (
    RAW_ROOT_DEFAULT,
    RawFile,
    append_manifest,
    raw_zip_path,
    sha256_file,
)

BASE = "https://data.binance.vision/data"

# macOS Python.org builds don't use the system trust store; pin certifi's CA bundle explicitly.
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


@dataclass(frozen=True)
class DumpSpec:
    """Locates one monthly archive on data.binance.vision."""

    symbol: str
    market: str  # "futures/um" path segment vs raw-store "um"
    stream: str  # "klines" | "aggTrades"
    period: str  # "YYYY-MM"

    @property
    def store_market(self) -> str:
        return "um" if "um" in self.market else "spot"

    @property
    def filename(self) -> str:
        if self.stream == "klines":
            return f"{self.symbol}-1m-{self.period}.zip"
        return f"{self.symbol}-{self.stream}-{self.period}.zip"

    @property
    def url(self) -> str:
        sub = (
            f"{self.stream}/{self.symbol}/1m"
            if self.stream == "klines"
            else f"{self.stream}/{self.symbol}"
        )
        return f"{BASE}/{self.market}/monthly/{sub}/{self.filename}"


def _download(url: str, dest: Path, *, retries: int = 4, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=timeout, context=_SSL_CTX) as r:
                with tmp.open("wb") as f:
                    while block := r.read(1 << 20):
                        f.write(block)
            tmp.replace(dest)
            return
        except Exception as e:  # retry on any transient network error
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {url}") from last


def _fetch_checksum(url: str, *, timeout: int = 60) -> str:
    with urllib.request.urlopen(url + ".CHECKSUM", timeout=timeout, context=_SSL_CTX) as r:
        # format: "<sha256-hex>  <filename>"
        return r.read().decode().split()[0].strip().lower()


def ingest_month(
    spec: DumpSpec, *, root: Path = RAW_ROOT_DEFAULT, run_id: str = "ingest"
) -> RawFile:
    """Download + checksum-verify one monthly archive into the immutable raw store."""
    dest = raw_zip_path(root, spec.symbol, spec.store_market, spec.stream, spec.filename)
    expected = _fetch_checksum(spec.url)

    if dest.exists() and sha256_file(dest) == expected:
        actual = expected  # already present + verified → idempotent skip
    else:
        _download(spec.url, dest)
        actual = sha256_file(dest)
        if actual != expected:
            dest.unlink(missing_ok=True)
            raise ValueError(
                f"CHECKSUM MISMATCH {spec.filename}: got {actual[:12]}…, want {expected[:12]}…"
            )

    record = RawFile(
        symbol=spec.symbol,
        market=spec.store_market,
        stream=spec.stream,
        period=spec.period,
        url=spec.url,
        sha256=actual,
        n_bytes=dest.stat().st_size,
        local_path=str(dest),
        fetched_at_utc=datetime.now(UTC).isoformat(),
    )
    append_manifest(root, run_id, record)
    return record


def months_in_year(year: int) -> list[str]:
    return [f"{year}-{m:02d}" for m in range(1, 13)]


def ingest_symbol_year(
    symbol: str,
    year: int,
    *,
    market: str = "futures/um",
    streams: tuple[str, ...] = ("klines", "aggTrades"),
    root: Path = RAW_ROOT_DEFAULT,
    run_id: str = "ingest",
    months: list[str] | None = None,
) -> list[RawFile]:
    """Ingest a symbol's klines + aggTrades for a year (or a subset of months)."""
    periods = months if months is not None else months_in_year(year)
    out: list[RawFile] = []
    for period in periods:
        for stream in streams:
            spec = DumpSpec(symbol=symbol, market=market, stream=stream, period=period)
            out.append(ingest_month(spec, root=root, run_id=run_id))
    return out
