"""N-symbol universe generalization (M4a) — the causal/independence gates.

The load-bearing claims:
  * **per-symbol independence** — a symbol's dataset_hash is identical whether built alone or
    inside an N-symbol universe (no cross-symbol statistic can leak in);
  * **regression** — the universe build of a single full-coverage symbol equals the direct
    single-symbol transform bit-for-bit;
  * **delisting truncation** — a symbol contributes only ``[listing, delisting)`` bars;
  * **leak-free** — the exhaustive causal sweep still passes on every (possibly truncated) stream.

All synthetic / no network. The real BTCUSDT-2023 ``dataset_hash`` regression runs in the M4a
dry-run script (scripts/m4a_universe_dryrun.py), not here (it needs the 5 GB raw store).
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.data.causal_check import exhaustive_truncation_sweep
from trikaal.data.config import synthetic_test_config
from trikaal.data.features import compute_features
from trikaal.data.synthetic import make_synthetic_stream
from trikaal.data.universe import (
    SelectionRule,
    SymbolSpec,
    UniverseSpec,
    bootstrap_universe,
    date_to_ms,
    load_universe,
    months_between,
    ms_to_date,
    rank_by_cumulative_quote_volume,
)
from trikaal.data.universe_build import (
    build_one_symbol,
    build_universe_from_streams,
    slice_stream_by_ms,
)
from trikaal.utils.hashing import content_hash


def _cfg():
    return synthetic_test_config(
        half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20, vol_window=24, n_warm_vol=12
    )


def _spec(symbol, listing=None, delisting=None, perp=True):
    return SymbolSpec(
        symbol=symbol, listing_date=listing, delisting_date=delisting, perp_available=perp
    )


# --------------------------------------------------------------------------- config + helpers
def test_sample_config_parses_with_majors_and_a_delisted_pair():
    spec = load_universe()
    syms = {s.symbol for s in spec.symbols}
    assert {"BTCUSDT", "ETHUSDT"} <= syms  # majors present
    delisted = [s for s in spec.symbols if s.delisting_date is not None]
    assert delisted, "sample must include >=1 delisted pair to exercise truncation"
    assert spec.frequency == "1m" and spec.market == "um"
    assert spec.selection.rule == "cumulative_quote_volume"
    # active_symbols is deterministically symbol-sorted
    act = [s.symbol for s in spec.active_symbols()]
    assert act == sorted(act)


def test_window_and_active_month_helpers():
    assert months_between("2023-01-01", "2023-04-01") == ["2023-01", "2023-02", "2023-03"]
    assert months_between("2023-01-15", "2023-03-10") == ["2023-01", "2023-02", "2023-03"]
    assert months_between("2023-01-01", "2023-01-01") == []  # empty half-open window
    # a fully-live symbol spans the whole window
    s = _spec("BTCUSDT", listing="2019-09-08", delisting=None)
    assert s.active_months("2023-01-01", "2023-04-01") == ["2023-01", "2023-02", "2023-03"]
    # delisting truncates the month list
    d = _spec("SRMUSDT", listing="2020-01-01", delisting="2023-02-10")
    assert d.active_months("2023-01-01", "2023-04-01") == ["2023-01", "2023-02"]
    # a symbol not live in the window has no months
    gone = _spec("OLD", listing="2019-01-01", delisting="2020-01-01")
    assert gone.active_months("2023-01-01", "2023-04-01") == []
    assert not gone.is_live_in("2023-01-01", "2023-04-01")


def test_date_ms_roundtrip():
    assert ms_to_date(date_to_ms("2023-06-19")) == "2023-06-19"


# --------------------------------------------------------------------------- selection rule
def test_selection_rule_ranks_by_volume_then_name():
    vols = {"AAA": 10.0, "BBB": 30.0, "CCC": 30.0, "DDD": 5.0}
    # BBB/CCC tie on volume → broken by name; top_n caps
    assert rank_by_cumulative_quote_volume(vols, top_n=3) == ["BBB", "CCC", "AAA"]


def test_bootstrap_applies_rule_and_filters():
    base = UniverseSpec(
        name="u",
        frequency="1m",
        market="um",
        window_start="2023-01-01",
        window_end="2024-01-01",
        selection=SelectionRule(top_n=2, include_delisted=True),
        symbols=(),
    )
    meta = {
        "BTCUSDT": _spec("BTCUSDT", listing="2020-01-01"),
        "ETHUSDT": _spec("ETHUSDT", listing="2020-01-01"),
        "DEADUSDT": _spec("DEADUSDT", listing="2020-01-01", delisting="2023-06-01"),
    }
    vols = {"BTCUSDT": 100.0, "ETHUSDT": 50.0, "DEADUSDT": 80.0}
    got = bootstrap_universe(base, cumulative_quote_volume=vols, metadata=meta)
    assert {s.symbol for s in got.symbols} == {"BTCUSDT", "DEADUSDT"}  # top-2 by volume
    # excluding delisted drops DEADUSDT even though its volume is high
    base_excl = base.with_symbols(())
    base_excl = UniverseSpec(
        **{**base.__dict__, "selection": SelectionRule(top_n=2, include_delisted=False)}
    )
    got2 = bootstrap_universe(base_excl, cumulative_quote_volume=vols, metadata=meta)
    assert {s.symbol for s in got2.symbols} == {"BTCUSDT", "ETHUSDT"}


# --------------------------------------------------------------------------- regression
def test_single_symbol_universe_equals_direct_transform():
    """A full-coverage symbol through the universe build == direct compute_features, bit-for-bit."""
    cfg = _cfg()
    stream, _ = make_synthetic_stream(seed=5, n_bars=900)
    direct = compute_features(stream, cfg)
    direct_hash = content_hash(direct.x, direct.m, direct.ts)

    res = build_universe_from_streams(
        {"AAAUSDT": stream},
        {"AAAUSDT": _spec("AAAUSDT")},
        window_start="2019-01-01",
        window_end="2030-01-01",
        cfg=cfg,
        root=None,
    )
    assert res.per_symbol["AAAUSDT"].dataset_hash == direct_hash


# --------------------------------------------------------------------------- per-symbol independence
def test_per_symbol_independence():
    """B's presence cannot change A's dataset_hash — there is no cross-symbol state."""
    cfg = _cfg()
    a, _ = make_synthetic_stream(seed=1, n_bars=900)
    b, _ = make_synthetic_stream(seed=2, n_bars=700)
    specs = {"AAA": _spec("AAA"), "BBB": _spec("BBB")}
    win = dict(window_start="2019-01-01", window_end="2030-01-01")

    alone = build_universe_from_streams(
        {"AAA": a}, {"AAA": specs["AAA"]}, **win, cfg=cfg, root=None
    )
    together = build_universe_from_streams({"AAA": a, "BBB": b}, specs, **win, cfg=cfg, root=None)
    assert alone.per_symbol["AAA"].dataset_hash == together.per_symbol["AAA"].dataset_hash
    # and the universe hash is order-independent / Merkle over both
    assert together.n_symbols == 2 and together.universe_hash.startswith("sha256:")


# --------------------------------------------------------------------------- delisting truncation
def test_delisting_truncation_keeps_only_live_bars():
    cfg = _cfg()
    stream, _ = make_synthetic_stream(seed=8, n_bars=2880)  # ~2 days
    bom = stream.bar_open_ms
    cut_date = ms_to_date(int(bom[len(bom) // 2]))  # a date that falls mid-stream
    hi_ms = date_to_ms(cut_date)
    expected_kept = int((bom < hi_ms).sum())
    assert 0 < expected_kept < len(bom), "fixture must straddle the delisting date"

    spec = _spec("ZZZUSDT", listing=None, delisting=cut_date)
    build = build_one_symbol(
        stream, spec, window_start="2019-01-01", window_end="2030-01-01", cfg=cfg, root=None
    )
    assert build.n_bars == expected_kept
    assert build.tail_truncated and not build.head_truncated
    assert build.last_bar_ms < hi_ms


def test_slice_is_identity_for_full_coverage():
    stream, _ = make_synthetic_stream(seed=9, n_bars=400)
    assert slice_stream_by_ms(stream, -np.inf, np.inf) is stream  # identity, no copy


# --------------------------------------------------------------------------- leak-free multi-symbol
@pytest.mark.gate
def test_multi_symbol_streams_stay_leak_free():
    """The exhaustive causal sweep passes on every symbol stream in the universe (incl. truncated)."""
    cfg = _cfg()
    a, _ = make_synthetic_stream(seed=3, n_bars=600)
    b, _ = make_synthetic_stream(seed=4, n_bars=600)
    # truncate B by ms to also exercise truncate ∘ causality
    hi = int(b.bar_open_ms[500])
    b_trunc = slice_stream_by_ms(b, -np.inf, float(hi))
    for name, s in (("A", a), ("B_trunc", b_trunc)):
        rep = exhaustive_truncation_sweep(s, cfg)
        assert rep.passed, f"{name}: " + rep.summary()
        assert rep.coverage == 1.0


# --------------------------------------------------------------------------- lake round-trip
@pytest.mark.slow
def test_universe_lake_roundtrip_per_symbol(tmp_path):
    from trikaal.data.lake import assemble_window, connect

    cfg = _cfg()
    a, _ = make_synthetic_stream(seed=6, n_bars=900)
    b, _ = make_synthetic_stream(seed=7, n_bars=800)
    specs = {"AAA": _spec("AAA"), "BBB": _spec("BBB")}
    res = build_universe_from_streams(
        {"AAA": a, "BBB": b},
        specs,
        window_start="2019-01-01",
        window_end="2030-01-01",
        cfg=cfg,
        root=tmp_path / "bars",
    )
    con = connect(tmp_path / "bars")
    for sym, stream in (("AAA", a), ("BBB", b)):
        out = compute_features(stream, cfg)
        d = assemble_window(con, sym, int(out.ts.min()), int(out.ts.max()) + 1)
        assert np.array_equal(d["x"], out.x)
        assert content_hash(d["x"], d["mask"], d["ts"]) == res.per_symbol[sym].dataset_hash
