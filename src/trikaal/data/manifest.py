"""Delisting-aware universe manifest (data-pipeline §1.3/§6) — generalized to N symbols.

The manifest is the content-addressed record of a universe build: per-symbol coverage
(listing/delisting, OI-coverage start), the live span actually ingested, per-symbol
``dataset_hash``, raw-file checksums, and the Merkle ``universe_hash``. It makes the whole
dataset reconstructible and lets the eval harness state survivorship honestly (delisted symbols
are present for exactly their live span).

``manifest_content_hash`` excludes wall-clock provenance (``built_at_utc``) so two builds of the
same data from the same config hash identically — determinism is a deliverable (§7.7).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trikaal.data.universe import UniverseSpec, ms_to_date
from trikaal.data.universe_build import UniverseBuildResult
from trikaal.utils.hashing import content_hash


@dataclass(frozen=True)
class SymbolCoverage:
    """One symbol's coverage row. Dates are UTC ``YYYY-MM-DD`` (or ``None``)."""

    symbol: str
    market: str
    listing_date: str | None
    delisting_date: str | None
    oi_coverage_start: str | None
    perp_available: bool
    # actually-ingested live span (post delisting-truncation)
    first_bar_date: str
    last_bar_date: str
    n_bars: int
    n_segments: int
    head_truncated: bool
    tail_truncated: bool
    dataset_hash: str
    # raw provenance (file checksums); populated at ingest, empty in a streams-only build
    raw_files: list[dict[str, Any]] = field(default_factory=list)

    def content_fields(self) -> dict[str, Any]:
        """The deterministic subset hashed into the universe manifest hash."""
        d = asdict(self)
        d.pop("raw_files", None)  # checksums recorded but kept out of the data-content hash
        return d


@dataclass
class UniverseManifest:
    """Full universe manifest: config + per-symbol coverage + Merkle hash + provenance."""

    name: str
    frequency: str
    market: str
    window_start: str
    window_end: str
    config_hash: str
    universe_hash: str
    n_symbols: int
    total_bars: int
    symbols: list[SymbolCoverage]
    built_at_utc: str | None = None  # provenance only — excluded from the content hash

    def manifest_content_hash(self) -> str:
        """Deterministic hash over config + symbol-sorted coverage (NO wall-clock field)."""
        payload = {
            "name": self.name,
            "frequency": self.frequency,
            "market": self.market,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "config_hash": self.config_hash,
            "universe_hash": self.universe_hash,
            "symbols": [s.content_fields() for s in sorted(self.symbols, key=lambda s: s.symbol)],
        }
        return content_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["manifest_content_hash"] = self.manifest_content_hash()
        return d

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path


def build_manifest(
    spec: UniverseSpec,
    result: UniverseBuildResult,
    *,
    raw_files: dict[str, list[dict[str, Any]]] | None = None,
    built_at_utc: str | None = None,
) -> UniverseManifest:
    """Assemble the manifest from a :class:`UniverseSpec` + :class:`UniverseBuildResult`.

    ``raw_files`` maps symbol → list of raw-file checksum records (from the ingest manifest);
    omit in a streams-only (synthetic / dry-run) build.
    """
    spec_by_sym = {s.symbol: s for s in spec.symbols}
    raw_files = raw_files or {}
    rows: list[SymbolCoverage] = []
    for sym in sorted(result.per_symbol):
        b = result.per_symbol[sym]
        s = spec_by_sym.get(sym)
        rows.append(
            SymbolCoverage(
                symbol=sym,
                market=s.market if s else spec.market,
                listing_date=s.listing_date if s else None,
                delisting_date=s.delisting_date if s else None,
                oi_coverage_start=s.oi_coverage_start if s else None,
                perp_available=s.perp_available if s else True,
                first_bar_date=ms_to_date(b.first_bar_ms),
                last_bar_date=ms_to_date(b.last_bar_ms),
                n_bars=b.n_bars,
                n_segments=b.n_segments,
                head_truncated=b.head_truncated,
                tail_truncated=b.tail_truncated,
                dataset_hash=b.dataset_hash,
                raw_files=raw_files.get(sym, []),
            )
        )
    return UniverseManifest(
        name=spec.name,
        frequency=spec.frequency,
        market=spec.market,
        window_start=spec.window_start,
        window_end=spec.window_end,
        config_hash=spec.config_hash(),
        universe_hash=result.universe_hash,
        n_symbols=result.n_symbols,
        total_bars=result.total_bars,
        symbols=rows,
        built_at_utc=built_at_utc,
    )


__all__ = ["SymbolCoverage", "UniverseManifest", "build_manifest"]
