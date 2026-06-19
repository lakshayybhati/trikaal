"""Bootstrap the real universe from the Binance dump index (data-pipeline §1.3) — M4b.

Survivorship is OWNED at the source: the S3 object index at ``data.binance.vision`` retains the
historical dumps of **delisted** symbols, so listing the index — not the live ``exchangeInfo`` —
gives the survivorship-correct symbol set. From one paginated scan of the ``aggTrades`` index we
read, per symbol, **for free (no bar download)**:

  * the set of available monthly dumps  → ``listing`` = first month, ``delisting`` = month after the
    last available month *iff* that last month trails the globally-latest available month (a symbol
    whose dumps stop early has delisted; one current to the latest month is still live);
  * the total ``aggTrades`` bytes        → a liquidity/volume **rank proxy** (trade-byte volume is
    monotone in trade count; used to pick the top-N — documented as a proxy for Σ quote_volume,
    which would require downloading every candidate's klines first).

Listing/delisting are therefore derived **FROM THE DATA**, never asserted from memory. Network is
read-only metadata (object keys + sizes); no bar data is downloaded here.
"""

from __future__ import annotations

import re
import ssl
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from xml.etree import ElementTree as ET

import certifi

from trikaal.data.universe import SelectionRule, SymbolSpec, UniverseSpec, months_between

S3_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def _list_page(
    prefix: str, marker: str | None, *, timeout: int = 60
) -> tuple[list[tuple[str, int]], str | None]:
    """One S3 ListBucket page → (list of (key, size), next_marker | None)."""
    url = f"{S3_BASE}?prefix={prefix}&max-keys=1000"
    if marker:
        url += f"&marker={urllib.parse.quote(marker)}"
    with urllib.request.urlopen(url, timeout=timeout, context=_SSL_CTX) as r:
        root = ET.fromstring(r.read())
    items: list[tuple[str, int]] = []
    last_key = None
    for c in root.findall(f"{_NS}Contents"):
        key = c.findtext(f"{_NS}Key") or ""
        size = int(c.findtext(f"{_NS}Size") or 0)
        items.append((key, size))
        last_key = key
    truncated = (root.findtext(f"{_NS}IsTruncated") or "false") == "true"
    next_marker = (root.findtext(f"{_NS}NextMarker") or last_key) if truncated else None
    return items, next_marker


def list_objects(prefix: str, *, max_pages: int = 1000) -> list[tuple[str, int]]:
    """All ``(key, size)`` under ``prefix`` (paginated). ``max_pages`` is a runaway backstop."""
    out: list[tuple[str, int]] = []
    marker: str | None = None
    for _ in range(max_pages):
        page, marker = _list_page(prefix, marker)
        out.extend(page)
        if not marker:
            break
    return out


@dataclass
class SymbolIndex:
    symbol: str
    months: list[str]  # sorted YYYY-MM with an aggTrades dump
    total_agg_bytes: int

    @property
    def first_month(self) -> str:
        return self.months[0]

    @property
    def last_month(self) -> str:
        return self.months[-1]


# Strict: monthly aggTrades zip only — {SYM}/{SYM}-aggTrades-YYYY-MM.zip (same symbol both sides,
# 4-digit year, 2-digit month). Rejects .CHECKSUM, daily dumps, and any malformed keys.
_AGG_KEY_RE = re.compile(r"(?P<sym>[A-Z0-9]+)/(?P=sym)-aggTrades-(?P<ym>\d{4}-\d{2})\.zip$")


def _parse_agg_key(key: str) -> tuple[str, str] | None:
    m = _AGG_KEY_RE.search(key)
    return (m.group("sym"), m.group("ym")) if m else None


def scan_aggtrades_index(market: str = "um") -> dict[str, SymbolIndex]:
    """Scan the monthly aggTrades index → per-symbol month coverage + bytes (no bar download)."""
    prefix = f"data/futures/{market}/monthly/aggTrades/"
    months: dict[str, set[str]] = defaultdict(set)
    nbytes: dict[str, int] = defaultdict(int)
    for key, size in list_objects(prefix):
        parsed = _parse_agg_key(key)
        if parsed is None:
            continue
        sym, ym = parsed
        months[sym].add(ym)
        nbytes[sym] += size
    return {
        s: SymbolIndex(symbol=s, months=sorted(months[s]), total_agg_bytes=nbytes[s])
        for s in months
    }


def _shift_month(period: str, k: int) -> str:
    """``YYYY-MM`` shifted by ``k`` months (month-index arithmetic; handles year wrap)."""
    y, m = (int(x) for x in period.split("-"))
    idx = y * 12 + (m - 1) + k
    return f"{idx // 12}-{idx % 12 + 1:02d}"


def _next_month(period: str) -> str:
    return _shift_month(period, 1)


def derive_symbol_spec(
    si: SymbolIndex, *, global_latest_month: str, market: str = "um", lag_months: int = 1
) -> SymbolSpec:
    """listing = first dump month (day 01); delisting = month-after-last iff the symbol trails the
    latest globally-available month by MORE than ``lag_months`` (else still live → None).

    The lag tolerance matters: Binance publishes monthly dumps with non-uniform per-symbol lag, so a
    still-live symbol legitimately trails the single freshest-published symbol by ~1 month in the
    publish window. A zero-tolerance comparison would fabricate a delisting for live symbols and
    drop their freshest (most valuable) month — a survivorship corruption. Dates from the data,
    never from memory. (``lag_months`` compared by month index, not string.)

    Interior gaps: a delist-then-relist (or a never-published interior month) leaves a hole in
    ``si.months``. We do NOT fabricate a delisting at the gap — listing/delisting bracket the full
    available span ``[first, last]`` and the ingest plan is built from the **actual dump months**
    (``index[sym].months`` ∩ window), so non-existent months are never requested (no 404), and the
    interior gap becomes a price-gap → a segment boundary in the build (causally safe). delisting
    therefore reflects only when the data series ENDS, which is the survivorship-relevant signal."""
    listing = f"{si.first_month}-01"
    live_threshold = _shift_month(global_latest_month, -max(0, lag_months))
    if si.last_month >= live_threshold:
        delisting = None  # within the publish-lag window of the latest dumps → still live
    else:
        delisting = f"{_next_month(si.last_month)}-01"  # delisted: first day after last live month
    return SymbolSpec(
        symbol=si.symbol,
        market=market,
        listing_date=listing,
        delisting_date=delisting,
        perp_available=True,
        oi_coverage_start=None,  # §1.2 OI retention trap → masked historically
    )


def bootstrap_from_index(
    index: dict[str, SymbolIndex],
    *,
    window_start: str,
    window_end: str,
    selection: SelectionRule,
    name: str = "trikaal-v1-universe",
    market: str = "um",
    lag_months: int = 1,
) -> tuple[UniverseSpec, dict[str, int]]:
    """Apply the selection rule to a scanned index → (UniverseSpec, {symbol: est_bytes_in_window}).

    Pure function over the scanned index (the network scan is separate, so this is unit-testable
    offline). Ranks by total aggTrades bytes (volume proxy), filters by quote asset, in-window live
    span, and the ``min_live_*`` floors. ``lag_months`` is the publish-lag tolerance for the
    still-live decision (see :func:`derive_symbol_spec`).
    """
    base = UniverseSpec(
        name=name,
        frequency="1m",
        market=market,
        window_start=window_start,
        window_end=window_end,
        selection=selection,
        symbols=(),
    )
    if not index:
        return base, {}  # empty / fully-filtered index → empty universe (not a crash)
    global_latest = max(si.last_month for si in index.values())
    win_months = set(months_between(window_start, window_end))
    candidates: list[tuple[str, int, SymbolSpec]] = []
    for sym, si in index.items():
        if not sym.endswith(selection.quote_asset):
            continue
        spec = derive_symbol_spec(
            si, global_latest_month=global_latest, market=market, lag_months=lag_months
        )
        if not spec.is_live_in(window_start, window_end):
            continue
        if not selection.include_delisted and spec.delisting_date is not None:
            continue
        in_window_months = [m for m in si.months if m in win_months]
        if len(in_window_months) < max(1, selection.min_live_days // 30):
            continue
        # est in-window bytes = total bytes prorated by the fraction of months inside the window
        est = int(si.total_agg_bytes * len(in_window_months) / max(1, len(si.months)))
        candidates.append((sym, est, spec))
    candidates.sort(key=lambda t: (-t[1], t[0]))
    chosen = candidates[: selection.top_n]
    specs = tuple(sorted((c[2] for c in chosen), key=lambda s: s.symbol))
    est_bytes = {c[0]: c[1] for c in chosen}
    return base.with_symbols(specs), est_bytes


def bootstrap_from_s3(
    *,
    window_start: str,
    window_end: str,
    selection: SelectionRule,
    market: str = "um",
    name: str = "trikaal-v1-universe",
    lag_months: int = 1,
) -> tuple[UniverseSpec, dict[str, int]]:
    """Network entry point: scan the live S3 index, then apply the rule. (M4b only.)"""
    index = scan_aggtrades_index(market)
    return bootstrap_from_index(
        index,
        window_start=window_start,
        window_end=window_end,
        selection=selection,
        name=name,
        market=market,
        lag_months=lag_months,
    )


def latest_available_month() -> str:
    """Current month minus one (the dumps lag ~1 month) — a quick sanity reference."""
    now = datetime.now(UTC)
    y, m = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)
    return f"{y}-{m:02d}"


__all__ = [
    "S3_BASE",
    "SymbolIndex",
    "bootstrap_from_index",
    "bootstrap_from_s3",
    "derive_symbol_spec",
    "latest_available_month",
    "list_objects",
    "scan_aggtrades_index",
]
