"""Delisting-aware universe manifest (M4a) — deterministic content hash."""

from __future__ import annotations

import json

from trikaal.data.config import synthetic_test_config
from trikaal.data.manifest import build_manifest
from trikaal.data.synthetic import make_synthetic_stream
from trikaal.data.universe import SelectionRule, SymbolSpec, UniverseSpec
from trikaal.data.universe_build import build_universe_from_streams


def _cfg():
    return synthetic_test_config(
        half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20, vol_window=24, n_warm_vol=12
    )


def _spec_pair():
    a, _ = make_synthetic_stream(seed=1, n_bars=800)
    b, _ = make_synthetic_stream(seed=2, n_bars=700)
    specs = {
        "AAA": SymbolSpec("AAA", listing_date="2019-01-01"),
        "BBB": SymbolSpec("BBB", listing_date="2019-01-01", delisting_date="2099-01-01"),
    }
    uspec = UniverseSpec(
        name="u",
        frequency="1m",
        market="um",
        window_start="2019-01-01",
        window_end="2100-01-01",
        selection=SelectionRule(),
        symbols=tuple(specs.values()),
    )
    return {"AAA": a, "BBB": b}, specs, uspec


def test_manifest_round_trips_and_hash_excludes_wall_clock(tmp_path):
    streams, specs, uspec = _spec_pair()
    res = build_universe_from_streams(
        streams,
        specs,
        window_start=uspec.window_start,
        window_end=uspec.window_end,
        cfg=_cfg(),
        root=None,
    )
    m1 = build_manifest(uspec, res, built_at_utc="2026-06-19T00:00:00Z")
    m2 = build_manifest(uspec, res, built_at_utc="2030-12-31T23:59:59Z")
    # built_at differs but the content hash is identical (determinism deliverable)
    assert m1.manifest_content_hash() == m2.manifest_content_hash()
    assert m1.universe_hash == res.universe_hash
    assert m1.n_symbols == 2

    p = m1.write(tmp_path / "manifest.json")
    doc = json.loads(p.read_text())
    assert doc["manifest_content_hash"] == m1.manifest_content_hash()
    rows = {s["symbol"]: s for s in doc["symbols"]}
    assert set(rows) == {"AAA", "BBB"}
    assert all("dataset_hash" in r and "first_bar_date" in r for r in rows.values())


def test_manifest_changes_when_data_changes(tmp_path):
    streams, specs, uspec = _spec_pair()
    res = build_universe_from_streams(
        streams,
        specs,
        window_start=uspec.window_start,
        window_end=uspec.window_end,
        cfg=_cfg(),
        root=None,
    )
    base = build_manifest(uspec, res).manifest_content_hash()
    # perturb one symbol's data → its dataset_hash changes → manifest hash changes
    streams["AAA"].close[100] *= 1.01
    res2 = build_universe_from_streams(
        streams,
        specs,
        window_start=uspec.window_start,
        window_end=uspec.window_end,
        cfg=_cfg(),
        root=None,
    )
    assert build_manifest(uspec, res2).manifest_content_hash() != base
