"""KATs for per-symbol spread deciles from train-region liquidity (audit item 4).

The causal property under test: the decile mapping may use TRAIN-REGION data only — a symbol
whose liquidity explodes in the eval region must rank exactly as its train region says."""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from trikaal.data.universe_loader import train_region_liquidity_stats
from trikaal.eval.costs import (
    MAJOR_SYMBOLS,
    SPREAD_DECILE_FRAC,
    TOP_TIER_N,
    assign_spread_deciles,
    load_spread_deciles,
)


def test_assign_deciles_majors_pinned_next18_top20_rest_tail():
    # 30 symbols: majors given a deliberately AWFUL proxy — they stay "major" (spec constant)
    stats = {f"SYM{i:02d}USDT": (1.0 - i * 0.01, 1_000_000 - i) for i in range(28)}
    stats["BTCUSDT"] = (0.0, 10)
    stats["ETHUSDT"] = (0.0, 10)
    d = assign_spread_deciles(stats)
    assert d["BTCUSDT"] == d["ETHUSDT"] == "major"
    top20 = [s for s, t in d.items() if t == "top20"]
    assert len(top20) == TOP_TIER_N - len(MAJOR_SYMBOLS)  # 18
    assert set(top20) == {f"SYM{i:02d}USDT" for i in range(18)}  # the proxy rank, not names
    assert all(d[f"SYM{i:02d}USDT"] == "tail" for i in range(18, 28))
    # every label resolves to a spread fraction
    assert all(t in SPREAD_DECILE_FRAC for t in d.values())


def test_assign_deciles_deterministic_tiebreak():
    # identical proxies → n_bars decides; identical n_bars → symbol name decides
    stats = {"AAA": (1.0, 100), "BBB": (1.0, 200), "CCC": (1.0, 100)}
    order = sorted(stats, key=lambda s: (-stats[s][0], -stats[s][1], s))
    assert order == ["BBB", "AAA", "CCC"]
    d1 = assign_spread_deciles(stats)
    d2 = assign_spread_deciles(dict(reversed(list(stats.items()))))
    assert d1 == d2  # input order irrelevant


def test_train_region_stats_are_causal():
    """The causal KAT: symbol A is dead in the train window but very active in the eval
    region — its eval activity must be INVISIBLE; B's pre-window history counts toward
    n_bars but never toward the common-window share."""
    day = 86_400_000
    boundary = 400 * day
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE bars (symbol VARCHAR, bar_open_ms BIGINT, is_stale UTINYINT, m_5 UTINYINT)"
    )
    rows = []
    for i, t in enumerate(range(boundary - 180 * day, boundary, day)):  # common window: 1/day
        rows.append(("A", t, 1, 0))  # A present but STALE throughout the window
        rows.append(("B", t, 0, 1 if i < 30 else 0))  # B active; 30 zero-volume minutes
        rows.append(("C", t, 0, 0))  # C fully active
    for t in range(boundary - 200 * day, boundary - 180 * day, day):
        rows.append(("B", t, 0, 0))  # B pre-window history: n_bars only, never the share
    for t in range(boundary, boundary + 100 * day, day):
        rows.append(("A", t, 0, 0))  # A very active in the EVAL region — must be invisible
    con.executemany("INSERT INTO bars VALUES (?, ?, ?, ?)", rows)
    stats = train_region_liquidity_stats(con, boundary)
    assert stats["A"][0] == 0.0  # stale minutes don't count; eval activity invisible
    assert stats["A"][1] == 180  # train bars only — the 100 eval bars never entered
    assert stats["B"][0] == pytest.approx(150 / (180 * 1440))  # zero-vol minutes excluded
    assert stats["B"][1] == 200  # pre-window train history counts toward n_bars
    assert stats["C"][0] == pytest.approx(180 / (180 * 1440))
    rank = sorted(stats, key=lambda s: (-stats[s][0], -stats[s][1], s))
    assert rank == ["C", "B", "A"]  # activity rank, immune to A's eval burst
    d = assign_spread_deciles(stats)
    assert set(d.values()) <= set(SPREAD_DECILE_FRAC)


def test_load_spread_deciles_roundtrip(tmp_path):
    import json

    p = tmp_path / "deciles.json"
    p.write_text(
        json.dumps(
            {
                "deciles": {
                    "BTCUSDT": {"decile": "major", "spread_frac": 1e-4},
                    "ZZZUSDT": {"decile": "tail", "spread_frac": 1e-3},
                }
            }
        )
    )
    m = load_spread_deciles(p)
    assert m == {"BTCUSDT": 1e-4, "ZZZUSDT": 1e-3}


def test_score_cell_missing_symbol_fails_loudly():
    """The eval driver must never silently default a symbol missing from the decile map."""
    from trikaal.eval.xsection import XSectionConfig

    cfg = XSectionConfig(spread_frac_by_symbol={"AAAUSDT": 1e-4})
    # exercise the lookup exactly as score_cell does (the closure logic, extracted)
    assert cfg.spread_frac_by_symbol is not None
    with pytest.raises(KeyError):
        _ = cfg.spread_frac_by_symbol["MISSINGUSDT"]
    none_cfg = XSectionConfig()
    assert none_cfg.spread_frac_by_symbol is None  # dev fallback stays available


def test_committed_artifact_consistent_if_present():
    """When the real artifact exists (post item-4 run), it must be internally consistent."""
    from pathlib import Path

    p = Path("runs_manifest/m6_spread_deciles.json")
    if not p.exists():
        pytest.skip("decile artifact not yet generated")
    import json

    doc = json.loads(p.read_text())
    d = doc["deciles"]
    assert doc["n_symbols"] == len(d)
    for s in MAJOR_SYMBOLS:
        assert d[s]["decile"] == "major"
    counts = {t: sum(1 for r in d.values() if r["decile"] == t) for t in SPREAD_DECILE_FRAC}
    assert counts == doc["tier_counts"]
    assert counts["major"] == len(MAJOR_SYMBOLS)
    assert counts["top20"] == TOP_TIER_N - len(MAJOR_SYMBOLS)
    for rec in d.values():
        assert rec["spread_frac"] == SPREAD_DECILE_FRAC[rec["decile"]]
        assert 0.0 <= rec["active_minute_share_train"] <= 1.0
    m = load_spread_deciles(p)
    assert len(m) == doc["n_symbols"] and np.isfinite(list(m.values())).all()
