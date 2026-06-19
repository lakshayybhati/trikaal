"""Universe bootstrap from the dump index (M4b) — survivorship correctness (offline, no network).

These pin the two HIGH/MED findings the adversarial review caught: dump-publish-lag must NOT
fabricate a delisting for a still-live symbol, and an interior gap must not silently drop a symbol.
"""

from __future__ import annotations

from trikaal.data.universe import SelectionRule
from trikaal.data.universe_bootstrap import (
    SymbolIndex,
    _parse_agg_key,
    _shift_month,
    bootstrap_from_index,
    derive_symbol_spec,
)


def _si(symbol, months, total_bytes=10_000):
    return SymbolIndex(symbol=symbol, months=sorted(months), total_agg_bytes=total_bytes)


def _months(start, n):
    return [_shift_month(start, k) for k in range(n)]


# ---------------------------------------------------------------- month arithmetic + key parsing
def test_shift_month_wraps_years():
    assert _shift_month("2024-11", 3) == "2025-02"
    assert _shift_month("2024-01", -1) == "2023-12"
    assert _shift_month("2024-06", 0) == "2024-06"


def test_agg_key_parser_is_strict():
    base = "data/futures/um/monthly/aggTrades"
    assert _parse_agg_key(f"{base}/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip") == ("BTCUSDT", "2024-01")
    # rejects checksum, daily, mismatched symbol, malformed
    assert _parse_agg_key(f"{base}/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip.CHECKSUM") is None
    assert _parse_agg_key(f"{base}/BTCUSDT/ETHUSDT-aggTrades-2024-01.zip") is None
    assert _parse_agg_key(f"{base}/BTCUSDT/BTCUSDT-aggTrades-2024.zip") is None


# ---------------------------------------------------------------- HIGH: dump-lag false delisting
def test_publish_lag_does_not_fabricate_delisting():
    # global latest = 2026-05; a live symbol whose dump trails by exactly lag_months stays live
    si_live = _si(
        "AAAUSDT", _months("2024-01", 17)
    )  # ...2025-05  (trails 2026-05 by 12 — delisted)
    si_edge = _si("BBBUSDT", _months("2021-01", 65))  # ...2026-05  (current → live)
    si_lag1 = _si(
        "CCCUSDT", _months("2021-01", 64)
    )  # ...2026-04  (trails by 1 → still live, lag=1)
    si_lag2 = _si("DDDUSDT", _months("2021-01", 63))  # ...2026-03  (trails by 2 → delisted)
    glob = "2026-05"
    assert derive_symbol_spec(si_edge, global_latest_month=glob).delisting_date is None
    assert derive_symbol_spec(si_lag1, global_latest_month=glob).delisting_date is None  # lag-tol
    assert derive_symbol_spec(si_lag2, global_latest_month=glob).delisting_date == "2026-04-01"
    assert derive_symbol_spec(si_live, global_latest_month=glob).delisting_date == "2025-06-01"


def test_lag_months_is_a_knob():
    si = _si("AAAUSDT", _months("2021-01", 62))  # ...2026-02 (trails 2026-05 by 3)
    glob = "2026-05"
    assert (
        derive_symbol_spec(si, global_latest_month=glob, lag_months=2).delisting_date
        == "2026-03-01"
    )
    assert derive_symbol_spec(si, global_latest_month=glob, lag_months=3).delisting_date is None


# ---------------------------------------------------------------- MED: interior gap not dropped
def test_interior_gap_brackets_full_span():
    # delist-then-relist: dumps [2023-01,2023-02, 2024-11,2024-12]; latest global 2026-05
    si = _si("GAPUSDT", ["2023-01", "2023-02", "2024-11", "2024-12"])
    spec = derive_symbol_spec(si, global_latest_month="2026-05")
    assert spec.listing_date == "2023-01-01"
    assert spec.delisting_date == "2025-01-01"  # ends after last dump (trails latest → delisted)
    # the [listing, delisting) hull spans the gap; the ingest plan (built elsewhere) filters to the
    # real dump months, so no non-existent month is requested.
    active = set(spec.active_months("2021-01-01", "2025-06-01"))
    real_in_window = [m for m in si.months if m in active]
    assert real_in_window == ["2023-01", "2023-02", "2024-11", "2024-12"]  # all 4, gap preserved


# ---------------------------------------------------------------- LOW: empty index
def test_empty_index_returns_empty_universe_not_crash():
    spec, est = bootstrap_from_index(
        {}, window_start="2021-01-01", window_end="2025-01-01", selection=SelectionRule()
    )
    assert spec.symbols == () and est == {}


# ---------------------------------------------------------------- ranking + filters
def test_bootstrap_ranks_by_bytes_and_filters_quote_and_min_live():
    idx = {
        "BIGUSDT": _si("BIGUSDT", _months("2022-01", 30), total_bytes=10_000_000),
        "SMALLUSDT": _si("SMALLUSDT", _months("2022-01", 30), total_bytes=1_000),
        "SHORTUSDT": _si("SHORTUSDT", ["2024-12"], total_bytes=5_000_000),  # < min_live → dropped
        "ETHBUSD": _si("ETHBUSD", _months("2022-01", 30), total_bytes=9_000_000),  # wrong quote
    }
    sel = SelectionRule(top_n=10, min_live_days=60, include_delisted=True, quote_asset="USDT")
    spec, est = bootstrap_from_index(
        idx, window_start="2021-01-01", window_end="2025-01-01", selection=sel
    )
    syms = [s.symbol for s in spec.active_symbols()]
    assert syms == ["BIGUSDT", "SMALLUSDT"]  # USDT-only, min-live met; ranked BIG > SMALL by bytes
    assert est["BIGUSDT"] > est["SMALLUSDT"]
