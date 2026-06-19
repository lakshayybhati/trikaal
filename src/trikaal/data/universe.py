"""Universe definition + selection rule (data-pipeline §1.3) — the N-symbol generalization.

A :class:`UniverseSpec` is the immutable, content-hashable description of *which* symbols and
*what window* the ingest/build run over. It is the ONLY place the symbol set is coupled — and it
couples nothing statistical: every normalization / volatility / tau statistic stays strictly
per-symbol and causal (computed inside ``compute_features`` on one symbol's stream, never across
symbols). Going 1 symbol → N is therefore a pure fan-out over the *unchanged* single-symbol path.

Delisting is half-open: a symbol contributes bars ``t`` with
``listing_ms <= bar_open_ms(t) < delisting_ms`` (delisting day onward dropped). A still-live
symbol has ``delisting_ms = +inf``; a symbol with no listing date floor has ``listing_ms = -inf``.

The full 100–200-pair list is populated at M4b by :func:`bootstrap_universe` from the data
(ranked by cumulative quote volume, dates from a first/last-kline probe). This module ships the
schema, the loader, the active-window logic, and the selection rule; M4a never crosses to cloud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from trikaal.data.config import BAR_MS
from trikaal.utils.hashing import content_hash

UNIVERSE_CONFIG_DEFAULT = Path("config/universe.yaml")


def date_to_ms(date_str: str) -> int:
    """UTC midnight of ``YYYY-MM-DD`` → epoch milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).strftime("%Y-%m-%d")


def months_between(start_date: str, end_date: str) -> list[str]:
    """``YYYY-MM`` periods whose month overlaps the half-open window ``[start, end)``.

    Used to enumerate the monthly Binance dumps a symbol's window touches. The end date is
    exclusive; a window ending exactly on a month boundary does not pull that month.
    """
    s = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    e = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    if e <= s:
        return []
    out: list[str] = []
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        # include month (y,m) iff its first day is strictly before the exclusive end
        first = datetime(y, m, 1, tzinfo=UTC)
        if first < e:
            out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


@dataclass(frozen=True)
class SymbolSpec:
    """One symbol's coverage metadata (§1.3). Dates are UTC ``YYYY-MM-DD`` or ``None``."""

    symbol: str
    market: str = "um"
    listing_date: str | None = None
    delisting_date: str | None = None
    perp_available: bool = True
    oi_coverage_start: str | None = None

    @property
    def listing_ms(self) -> float:
        return float(date_to_ms(self.listing_date)) if self.listing_date else -math.inf

    @property
    def delisting_ms(self) -> float:
        return float(date_to_ms(self.delisting_date)) if self.delisting_date else math.inf

    @property
    def oi_coverage_start_ms(self) -> float:
        return float(date_to_ms(self.oi_coverage_start)) if self.oi_coverage_start else math.inf

    def active_window_ms(self, window_start: str, window_end: str) -> tuple[float, float]:
        """The symbol's live span ∩ observation window, in ms (half-open ``[lo, hi)``).

        ``lo = max(listing, window_start)``, ``hi = min(delisting, window_end)``. If
        ``hi <= lo`` the symbol has no live bars in the window.
        """
        lo = max(self.listing_ms, float(date_to_ms(window_start)))
        hi = min(self.delisting_ms, float(date_to_ms(window_end)))
        return lo, hi

    def is_live_in(self, window_start: str, window_end: str) -> bool:
        lo, hi = self.active_window_ms(window_start, window_end)
        return hi > lo

    def active_months(self, window_start: str, window_end: str) -> list[str]:
        """Monthly dump periods (``YYYY-MM``) the symbol's active window touches."""
        lo, hi = self.active_window_ms(window_start, window_end)
        if hi <= lo:
            return []
        return months_between(ms_to_date(int(lo)), ms_to_date(math.ceil(hi / BAR_MS) * BAR_MS))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "listing_date": self.listing_date,
            "delisting_date": self.delisting_date,
            "perp_available": self.perp_available,
            "oi_coverage_start": self.oi_coverage_start,
        }


@dataclass(frozen=True)
class SelectionRule:
    """§1.3 selection rule. Applied by :func:`bootstrap_universe` over live USDⓂ perps at M4b."""

    rule: str = "cumulative_quote_volume"
    top_n: int = 150
    min_live_days: int = 30
    min_live_bars: int = 43_200
    include_delisted: bool = True
    quote_asset: str = "USDT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "top_n": self.top_n,
            "min_live_days": self.min_live_days,
            "min_live_bars": self.min_live_bars,
            "include_delisted": self.include_delisted,
            "quote_asset": self.quote_asset,
        }


@dataclass(frozen=True)
class UniverseSpec:
    """Immutable, content-hashable universe definition (the symbol set + window + rule)."""

    name: str
    frequency: str
    market: str
    window_start: str
    window_end: str
    selection: SelectionRule
    symbols: tuple[SymbolSpec, ...]

    def active_symbols(self) -> tuple[SymbolSpec, ...]:
        """Symbols with ≥1 live bar in the window, in deterministic (symbol-sorted) order."""
        live = [s for s in self.symbols if s.is_live_in(self.window_start, self.window_end)]
        return tuple(sorted(live, key=lambda s: s.symbol))

    def window_months(self) -> list[str]:
        return months_between(self.window_start, self.window_end)

    def spec_dict(self) -> dict[str, Any]:
        """Deterministic dict for content-hashing (symbol-sorted; no wall-clock fields)."""
        return {
            "name": self.name,
            "frequency": self.frequency,
            "market": self.market,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "selection": self.selection.to_dict(),
            "symbols": [s.to_dict() for s in sorted(self.symbols, key=lambda s: s.symbol)],
        }

    def config_hash(self) -> str:
        return content_hash(self.spec_dict())

    def with_symbols(self, symbols: tuple[SymbolSpec, ...]) -> UniverseSpec:
        return replace(self, symbols=tuple(symbols))


def load_universe(path: Path = UNIVERSE_CONFIG_DEFAULT) -> UniverseSpec:
    """Parse ``config/universe.yaml`` into a :class:`UniverseSpec`."""
    raw = yaml.safe_load(Path(path).read_text())
    win = raw["window"]
    sel = raw.get("selection", {})
    symbols = tuple(
        SymbolSpec(
            symbol=s["symbol"],
            market=s.get("market", raw.get("market", "um")),
            listing_date=s.get("listing_date"),
            delisting_date=s.get("delisting_date"),
            perp_available=bool(s.get("perp_available", True)),
            oi_coverage_start=s.get("oi_coverage_start"),
        )
        for s in raw.get("symbols", [])
    )
    return UniverseSpec(
        name=raw["name"],
        frequency=raw.get("frequency", "1m"),
        market=raw.get("market", "um"),
        window_start=win["start"],
        window_end=win["end"],
        selection=SelectionRule(
            rule=sel.get("rule", "cumulative_quote_volume"),
            top_n=int(sel.get("top_n", 150)),
            min_live_days=int(sel.get("min_live_days", 30)),
            min_live_bars=int(sel.get("min_live_bars", 43_200)),
            include_delisted=bool(sel.get("include_delisted", True)),
            quote_asset=sel.get("quote_asset", "USDT"),
        ),
        symbols=symbols,
    )


def rank_by_cumulative_quote_volume(
    cumulative_quote_volume: dict[str, float], *, top_n: int
) -> list[str]:
    """The selection rule, as a pure function: top-``n`` symbols by Σ quote_volume.

    Ties are broken by symbol name (deterministic). This is *what* the bootstrap applies; the
    volumes themselves are measured from the data at M4b (this function takes them as input so it
    is unit-testable offline).
    """
    ordered = sorted(cumulative_quote_volume.items(), key=lambda kv: (-kv[1], kv[0]))
    return [sym for sym, _ in ordered[: max(0, top_n)]]


def bootstrap_universe(
    base: UniverseSpec,
    *,
    cumulative_quote_volume: dict[str, float],
    metadata: dict[str, SymbolSpec],
) -> UniverseSpec:
    """Populate the full `symbols:` list at M4b from measured data (NO network in this function).

    ``cumulative_quote_volume`` maps symbol → Σ quote_volume over its live window (measured from
    the ingested klines at M4b); ``metadata`` maps symbol → its resolved :class:`SymbolSpec`
    (listing/delisting from a first/last-kline probe). This applies the selection rule and the
    ``min_live_*`` / ``include_delisted`` filters and returns a fully-populated UniverseSpec. The
    *measurement* (klines probe, volume sum) is the cloud step; this is the pure rule applied to
    its outputs, so it is testable offline.
    """
    rule = base.selection
    eligible = {
        sym: v
        for sym, v in cumulative_quote_volume.items()
        if sym in metadata
        and (rule.include_delisted or metadata[sym].delisting_date is None)
        and metadata[sym].is_live_in(base.window_start, base.window_end)
    }
    ranked = rank_by_cumulative_quote_volume(eligible, top_n=rule.top_n)
    chosen = tuple(metadata[sym] for sym in sorted(ranked))
    return base.with_symbols(chosen)


__all__ = [
    "UNIVERSE_CONFIG_DEFAULT",
    "SelectionRule",
    "SymbolSpec",
    "UniverseSpec",
    "bootstrap_universe",
    "date_to_ms",
    "load_universe",
    "months_between",
    "ms_to_date",
    "rank_by_cumulative_quote_volume",
]
