"""M4a universe-ingest dry-run — LOCAL, no cloud, no bulk downloads.

Proves the N-symbol generalization on free local data:
  1. REGRESSION — build the universe over BTCUSDT-2023 alone via the generalized path and assert
     it reproduces the M2 ``dataset_hash`` (sha256:1d3ec4f8…) bit-for-bit.
  2. SHARDED EQUIVALENCE + RESUMABILITY — the resumable per-(symbol,month) shard reducer yields a
     byte-identical build to the in-memory path, and a second pass is served from cache.
  3. SYNTHETIC 2-SYMBOL SMOKE — full multi-symbol path end-to-end (one symbol delisted mid-window),
     per-symbol independence, lake round-trip, exhaustive causal sweep leak-free on each, manifest.
  4. INGEST PLAN (dry-run, no network) + COST ESTIMATE at 100 & 200 pairs.

    python scripts/m4a_universe_dryrun.py [--skip-regression]

The regression rebuilds the 5 GB BTCUSDT raw store (~5 min); pass --skip-regression for a fast
pass over the synthetic + plan + cost sections only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from trikaal.data.causal_check import exhaustive_truncation_sweep
from trikaal.data.config import FeatureConfig, synthetic_test_config
from trikaal.data.manifest import build_manifest
from trikaal.data.reduce_shard import (
    build_symbol_features_sharded,
)
from trikaal.data.synthetic import make_synthetic_stream
from trikaal.data.universe import (
    SelectionRule,
    SymbolSpec,
    UniverseSpec,
    date_to_ms,
    load_universe,
    ms_to_date,
)
from trikaal.data.universe_build import (
    build_one_symbol,
    build_universe,
    build_universe_from_streams,
)
from trikaal.data.universe_ingest import estimate_universe_cost, plan_universe_ingest
from trikaal.utils.hashing import content_hash

M2_HASH = "sha256:1d3ec4f8db6686e20cd57749db4ca441c3e0d13c62a97c5f82eb0120bc9fa961"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def regression_btcusdt() -> bool:
    section("1. REGRESSION — generalized build of BTCUSDT-2023 == M2 dataset_hash")
    spec = UniverseSpec(
        name="btcusdt-regression",
        frequency="1m",
        market="um",
        window_start="2023-01-01",
        window_end="2024-01-01",
        selection=SelectionRule(),
        symbols=(SymbolSpec("BTCUSDT", listing_date="2019-09-08", perp_available=True),),
    )
    t0 = time.time()
    res = build_universe(spec, root=None)  # root=None: do not clobber the existing M2 lake
    h = res.per_symbol["BTCUSDT"].dataset_hash
    ok = h == M2_HASH
    print(f"  built {res.per_symbol['BTCUSDT'].n_bars:,} bars in {time.time() - t0:.0f}s")
    print(f"  generalized dataset_hash = {h}")
    print(f"  M2 recorded dataset_hash = {M2_HASH}")
    print(f"  BIT-EXACT MATCH: {ok}")
    print(f"  universe_hash (Merkle, 1 symbol) = {res.universe_hash}")
    return ok


def sharded_equivalence() -> bool:
    section("2. SHARDED EQUIVALENCE + RESUMABILITY (BTCUSDT 2023-01)")
    from trikaal.data.build import build_symbol_features

    cfg = FeatureConfig()
    direct, _ = build_symbol_features("BTCUSDT", ["2023-01"], is_perp=True)
    direct_h = content_hash(direct.x, direct.m, direct.ts)

    shard_root = Path("processed/shards")
    t0 = time.time()
    s1, r1 = build_symbol_features_sharded("BTCUSDT", ["2023-01"], cfg=cfg, shard_root=shard_root)
    sharded_h = content_hash(s1.x, s1.m, s1.ts)
    t_first = time.time() - t0

    t0 = time.time()
    _s2, r2 = build_symbol_features_sharded("BTCUSDT", ["2023-01"], cfg=cfg, shard_root=shard_root)
    t_cached = time.time() - t0

    ok = sharded_h == direct_h
    print(f"  direct   dataset_hash = {direct_h[:30]}…")
    print(f"  sharded  dataset_hash = {sharded_h[:30]}…  (match={ok})")
    print(f"  pass 1: status={r1[0].status} ({t_first:.0f}s)")
    print(f"  pass 2: status={r2[0].status} ({t_cached:.1f}s)")
    print(f"  resumable: second pass served from cache = {r2[0].status == 'cached'}")
    return ok and r2[0].status == "cached"


def synthetic_universe_smoke() -> bool:
    section("3. SYNTHETIC 2-SYMBOL SMOKE — multi-symbol path, delisting, causal sweep, manifest")
    cfg = synthetic_test_config(
        half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20, vol_window=24, n_warm_vol=12
    )
    a, _ = make_synthetic_stream(seed=21, n_bars=2880)  # full coverage
    b, _ = make_synthetic_stream(seed=22, n_bars=2880)  # delisted mid-window
    cut = ms_to_date(int(b.bar_open_ms[len(b.bar_open_ms) // 2]))
    specs = {
        "AAAUSDT": SymbolSpec("AAAUSDT", listing_date="2019-01-01"),
        "ZZZUSDT": SymbolSpec("ZZZUSDT", listing_date="2019-01-01", delisting_date=cut),
    }
    uspec = UniverseSpec(
        name="synthetic-smoke",
        frequency="1m",
        market="um",
        window_start="2019-01-01",
        window_end="2030-01-01",
        selection=SelectionRule(),
        symbols=tuple(specs.values()),
    )
    out_root = Path("processed/m4a_smoke")
    res = build_universe_from_streams(
        {"AAAUSDT": a, "ZZZUSDT": b},
        specs,
        window_start=uspec.window_start,
        window_end=uspec.window_end,
        cfg=cfg,
        root=out_root,
    )
    kept = int((b.bar_open_ms < date_to_ms(cut)).sum())
    zb = res.per_symbol["ZZZUSDT"]
    ab = res.per_symbol["AAAUSDT"]
    print(f"  delisting cut={cut}  AAAUSDT bars={ab.n_bars} (full)  ZZZUSDT bars={zb.n_bars}")
    print(f"  ZZZUSDT: kept={kept}, tail_truncated={zb.tail_truncated} (delisting drops the rest)")

    # per-symbol independence
    alone = build_one_symbol(
        a,
        specs["AAAUSDT"],
        window_start=uspec.window_start,
        window_end=uspec.window_end,
        cfg=cfg,
        root=None,
    )
    indep = alone.dataset_hash == res.per_symbol["AAAUSDT"].dataset_hash
    print(f"  per-symbol independence (A alone == A in universe): {indep}")

    man = build_manifest(uspec, res, built_at_utc="2026-06-19T00:00:00Z")
    mpath = man.write(out_root / "manifest.json")
    print(f"  manifest → {mpath}  content_hash={man.manifest_content_hash()[:26]}…")

    # causal sweep leak-free on each (small streams; the O(T^2) exhaustive sweep needs T modest).
    # The truncated-stream case is covered too: B_small is sliced by ms before the sweep.
    from trikaal.data.universe_build import slice_stream_by_ms

    a_s, _ = make_synthetic_stream(seed=23, n_bars=600)
    b_s, _ = make_synthetic_stream(seed=24, n_bars=600)
    b_s = slice_stream_by_ms(b_s, -1e18, float(b_s.bar_open_ms[500]))  # truncate ∘ causality
    sweeps = {}
    for sym, stream in (("AAAUSDT", a_s), ("ZZZUSDT_trunc", b_s)):
        rep = exhaustive_truncation_sweep(stream, cfg)
        sweeps[sym] = rep.passed and rep.coverage == 1.0
        print(f"  causal sweep {sym}: {rep.summary()}")

    return indep and kept == zb.n_bars and all(sweeps.values())


def ingest_plan_and_cost() -> bool:
    section("4. INGEST PLAN (dry-run, no network) + COST ESTIMATE")
    spec = load_universe()
    plan = plan_universe_ingest(spec)
    by_stream = {k: round(v, 2) for k, v in plan.by_stream_gb().items()}
    print(f"  sample universe: {plan.n_symbols} live symbols, {plan.n_files} archive files")
    print(f"  est download (sample) = {plan.est_total_gb:.1f} GB  by stream(GB): {by_stream}")
    print(f"  funding (full history): {plan.funding_symbols}")
    print(f"  OI zero-filled/masked (retention trap): {plan.oi_zero_filled_symbols}")
    assert all(i.url.startswith("https://data.binance.vision/") for i in plan.items)

    print("\n  --- M4b cost estimate (24 months/symbol avg; mix major2%/top13%/mid85%) ---")
    mix = {"major": 0.02, "top20": 0.13, "mid": 0.85}
    for n in (100, 200):
        c = estimate_universe_cost(n, months_per_symbol=24, tier_mix=mix)
        flag = f"  WARN {c.notes}" if c.notes else ""
        print(
            f"  {n:>3} pairs: download≈{c.download_gb:,.0f}GB  lake≈{c.processed_lake_gb:,.0f}GB  "
            f"reduce≈{c.reduce_core_hours:,.0f}core-h  bars≈{c.total_bars / 1e6:,.0f}M{flag}"
        )
    return True


def main(skip_regression: bool = False) -> int:
    results: dict[str, bool] = {}
    results["synthetic_smoke"] = synthetic_universe_smoke()
    results["ingest_plan_cost"] = ingest_plan_and_cost()
    if not skip_regression:
        results["regression_btcusdt"] = regression_btcusdt()
        results["sharded_equivalence"] = sharded_equivalence()
    else:
        print(
            "\n[skip] regression + sharded equivalence (real BTCUSDT) skipped via --skip-regression"
        )

    section("M4a DRY-RUN SUMMARY")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(results.values())
    print(f"\n  {'ALL GREEN' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(skip_regression="--skip-regression" in sys.argv))
