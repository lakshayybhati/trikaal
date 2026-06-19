"""N-symbol ingest planner + cost model (M4a) — plan-only, NO network."""

from __future__ import annotations

from trikaal.data.universe import SelectionRule, SymbolSpec, UniverseSpec
from trikaal.data.universe_ingest import (
    AGG_MB_PER_MONTH_BY_TIER,
    estimate_universe_cost,
    plan_universe_ingest,
)


def _universe(*specs):
    return UniverseSpec(
        name="u",
        frequency="1m",
        market="um",
        window_start="2023-01-01",
        window_end="2023-04-01",
        selection=SelectionRule(),
        symbols=tuple(specs),
    )


def test_plan_enumerates_delisting_aware_files_without_network():
    spec = _universe(
        SymbolSpec("BTCUSDT", listing_date="2020-01-01"),  # full 3 months
        SymbolSpec("DEADUSDT", listing_date="2020-01-01", delisting_date="2023-02-10"),  # 2 months
    )
    plan = plan_universe_ingest(spec)
    # BTCUSDT: 3 months × 2 streams = 6; DEADUSDT: 2 months × 2 = 4
    assert plan.n_files == 10
    assert plan.n_symbols == 2
    btc = [i for i in plan.items if i.symbol == "BTCUSDT"]
    dead = [i for i in plan.items if i.symbol == "DEADUSDT"]
    assert {i.period for i in btc} == {"2023-01", "2023-02", "2023-03"}
    assert {i.period for i in dead} == {"2023-01", "2023-02"}  # delisting truncates Feb-10 → no Mar
    # urls point at data.binance.vision monthly futures dumps
    assert all(
        i.url.startswith("https://data.binance.vision/data/futures/um/monthly/") for i in plan.items
    )
    assert "aggTrades" in {i.stream for i in plan.items}


def test_plan_flags_funding_and_oi_retention_trap():
    spec = _universe(
        SymbolSpec(
            "BTCUSDT", listing_date="2020-01-01", perp_available=True, oi_coverage_start=None
        ),
        SymbolSpec("SPOTONLY", listing_date="2020-01-01", perp_available=False),
    )
    plan = plan_universe_ingest(spec)
    assert plan.funding_symbols == ["BTCUSDT"]  # only perps get funding
    assert plan.oi_zero_filled_symbols == ["BTCUSDT"]  # OI hist retention trap → masked


def test_aggtrades_dominates_byte_estimate():
    spec = _universe(SymbolSpec("BTCUSDT", listing_date="2020-01-01"))
    plan = plan_universe_ingest(spec)
    by_stream = plan.by_stream_gb()
    assert by_stream["aggTrades"] > 50 * by_stream["klines"]  # aggTrades is the heavy stream


def test_cost_estimate_scales_and_flags_saturation():
    mix = {"major": 0.02, "top20": 0.13, "mid": 0.85}
    c100 = estimate_universe_cost(100, months_per_symbol=24, tier_mix=mix)
    c200 = estimate_universe_cost(200, months_per_symbol=24, tier_mix=mix)
    assert c200.download_gb > c100.download_gb
    assert c200.processed_lake_gb > 0 and c200.reduce_core_hours > 0
    # 200 pairs × 24 months ≈ 214M bars — comfortably under the ~1–2B saturation point, no flag
    assert c200.total_bars < 1e9 and not c200.notes


def test_saturation_note_threshold():
    mix = {"mid": 1.0}
    # 400 symbols × 60 months × 44640 ≈ 1.07B bars → note fires
    big = estimate_universe_cost(400, months_per_symbol=60, tier_mix=mix)
    assert big.total_bars > 1e9 and any("saturation" in n for n in big.notes)
    small = estimate_universe_cost(150, months_per_symbol=24, tier_mix=mix)
    assert small.total_bars < 1e9 and not small.notes


def test_tier_table_is_monotone():
    t = AGG_MB_PER_MONTH_BY_TIER
    assert t["major"] > t["top20"] > t["mid"] > t["tail"]
