"""N-symbol ingest PLANNER + cost model (data-pipeline §2.1) — generalizes ``ingest.py``.

M4a is **plan-only**: :func:`plan_universe_ingest` enumerates every (symbol, month, stream)
archive URL the universe needs and estimates its bytes — it performs NO network I/O. The real
fetch (:func:`execute_plan`) reuses the proven single-symbol ``ingest_month`` (SHA-256-verified,
immutable raw, idempotent) and is GATED: it runs only at M4b once cloud is green-lit.

Funding / OI hooks (§1.2): funding-rate history is fully public (fapi ``/fundingRate``,
paginated, free) and is planned in full. Open-interest history hits the **retention trap** —
Binance ``/futures/data/openInterestHist`` retains only ~30 days — so historically OI is
**zero-filled and masked** (the OI mask bits are set in ``compute_features`` when the OI event
series is empty). The plan flags this so no one mistakes a zero for a real value.

Cost model (§ deliverable): byte estimates are anchored on the measured BTCUSDT-2023 dumps
(klines ≈ 1.8 MB/symbol-month; aggTrades ≈ 417 MB/symbol-month for the #1-volume symbol) and a
volume-tier decay, so the M4b $-range can be quoted at 100 and 200 pairs before any spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trikaal.data.ingest import DumpSpec
from trikaal.data.universe import UniverseSpec

# --- size anchors (bytes), measured from the real BTCUSDT-2023 raw store -----------------------
KLINE_MB_PER_MONTH: float = 1.8  # ~44.6k 1m rows; ~volume-independent
# aggTrades MB/month by volume tier. Anchored: BTCUSDT-2023 ≈ 417 MB/mo (5.0 GB / 12).
AGG_MB_PER_MONTH_BY_TIER: dict[str, float] = {
    "major": 400.0,  # BTC / ETH-class
    "top20": 70.0,  # high-volume alts
    "mid": 22.0,  # mid-cap
    "tail": 9.0,  # long tail (still > min_live_bars)
}
# reduction throughput anchor: measured 389 MB aggTrades zip → bar features in 23.2 s on one
# CPU core (≈ 16.8 MB compressed/s). Used for the CPU-hours estimate.
REDUCE_MB_PER_CORE_SEC: float = 16.8


def _tier_for(symbol: str) -> str:
    """Heuristic tier for the *sample* plan; at M4b the tier comes from the measured volume rank."""
    if symbol in ("BTCUSDT", "ETHUSDT"):
        return "major"
    if symbol in ("SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"):
        return "top20"
    return "mid"


@dataclass(frozen=True)
class DownloadItem:
    symbol: str
    stream: str  # "klines" | "aggTrades"
    period: str  # "YYYY-MM"
    url: str
    est_bytes: int


@dataclass
class IngestPlan:
    items: list[DownloadItem]
    n_symbols: int
    funding_symbols: list[str] = field(default_factory=list)  # full-history funding fetch
    oi_zero_filled_symbols: list[str] = field(default_factory=list)  # OI retention trap → masked

    @property
    def n_files(self) -> int:
        return len(self.items)

    @property
    def est_total_bytes(self) -> int:
        return sum(i.est_bytes for i in self.items)

    @property
    def est_total_gb(self) -> float:
        return self.est_total_bytes / 1e9

    def by_stream_gb(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for i in self.items:
            out[i.stream] = out.get(i.stream, 0.0) + i.est_bytes / 1e9
        return out


def _est_bytes(symbol: str, stream: str) -> int:
    if stream == "klines":
        return int(KLINE_MB_PER_MONTH * 1e6)
    return int(AGG_MB_PER_MONTH_BY_TIER[_tier_for(symbol)] * 1e6)


def plan_universe_ingest(
    spec: UniverseSpec,
    *,
    streams: tuple[str, ...] = ("klines", "aggTrades"),
    market: str = "futures/um",
) -> IngestPlan:
    """Enumerate every monthly archive the universe needs (NO network). Delisting-aware: a
    symbol contributes only the months its active window touches."""
    items: list[DownloadItem] = []
    funding: list[str] = []
    oi_masked: list[str] = []
    for s in spec.active_symbols():
        months = s.active_months(spec.window_start, spec.window_end)
        if not months:
            continue
        if s.perp_available:
            funding.append(s.symbol)
            # OI history only retained ~30d → historically zero-filled + masked (§1.2 trap)
            if s.oi_coverage_start is None:
                oi_masked.append(s.symbol)
        for period in months:
            for stream in streams:
                dump = DumpSpec(symbol=s.symbol, market=market, stream=stream, period=period)
                items.append(
                    DownloadItem(
                        symbol=s.symbol,
                        stream=stream,
                        period=period,
                        url=dump.url,
                        est_bytes=_est_bytes(s.symbol, stream),
                    )
                )
    return IngestPlan(
        items=items,
        n_symbols=len(spec.active_symbols()),
        funding_symbols=sorted(funding),
        oi_zero_filled_symbols=sorted(oi_masked),
    )


@dataclass(frozen=True)
class CostEstimate:
    n_symbols: int
    n_months_per_symbol: float
    download_gb: float
    raw_storage_gb: float
    processed_lake_gb: float
    reduce_core_hours: float
    total_bars: float
    notes: list[str]


def estimate_universe_cost(
    n_symbols: int,
    *,
    months_per_symbol: float,
    tier_mix: dict[str, float],
    bars_per_symbol_month: int = 44_640,
    processed_bytes_per_bar: float = 88.0,
) -> CostEstimate:
    """One-shot M4b cost estimate. ``tier_mix`` is the fraction of symbols in each volume tier
    (must sum to ~1). Numbers are deliberately rough (±50%); they exist to size the cloud box,
    not to bill to the cent.

    ``processed_bytes_per_bar`` ≈ 88 (measured: 45 MB lake / 525,600 BTCUSDT bars)."""
    agg_mb = sum(frac * AGG_MB_PER_MONTH_BY_TIER[t] for t, frac in tier_mix.items())
    agg_gb = n_symbols * months_per_symbol * agg_mb / 1e3
    kline_gb = n_symbols * months_per_symbol * KLINE_MB_PER_MONTH / 1e3
    download_gb = agg_gb + kline_gb
    total_bars = n_symbols * months_per_symbol * bars_per_symbol_month
    processed_gb = total_bars * processed_bytes_per_bar / 1e9
    # reduction is aggTrades-bound; klines are negligible to reduce
    reduce_core_hours = (agg_gb * 1e3) / REDUCE_MB_PER_CORE_SEC / 3600.0
    notes = []
    if total_bars > 1e9:
        notes.append(
            f"total_bars={total_bars / 1e9:.1f}B exceeds the ~1-2B saturation point - "
            "more data mostly slows iteration; consider trimming the window or top_n."
        )
    return CostEstimate(
        n_symbols=n_symbols,
        n_months_per_symbol=months_per_symbol,
        download_gb=download_gb,
        raw_storage_gb=download_gb,  # raw is stored as the downloaded zips (immutable)
        processed_lake_gb=processed_gb,
        reduce_core_hours=reduce_core_hours,
        total_bars=total_bars,
        notes=notes,
    )


def execute_plan(
    plan: IngestPlan, *, root=None, run_id: str = "universe", max_files: int | None = None
):
    """GATED real fetch (M4b only). Reuses the proven single-symbol ``ingest_month`` per file.

    NOT called in M4a. Kept here so the M4b runbook is one call; the caller must explicitly opt in.
    """
    from trikaal.data.ingest import ingest_month  # local import: M4a never reaches here
    from trikaal.data.raw_store import RAW_ROOT_DEFAULT

    root = root or RAW_ROOT_DEFAULT
    records = []
    market_map = {"klines": "futures/um", "aggTrades": "futures/um"}
    for item in plan.items[: (max_files or len(plan.items))]:
        spec = DumpSpec(
            symbol=item.symbol,
            market=market_map[item.stream],
            stream=item.stream,
            period=item.period,
        )
        records.append(ingest_month(spec, root=root, run_id=run_id))
    return records


__all__ = [
    "AGG_MB_PER_MONTH_BY_TIER",
    "KLINE_MB_PER_MONTH",
    "REDUCE_MB_PER_CORE_SEC",
    "CostEstimate",
    "DownloadItem",
    "IngestPlan",
    "estimate_universe_cost",
    "execute_plan",
    "plan_universe_ingest",
]
