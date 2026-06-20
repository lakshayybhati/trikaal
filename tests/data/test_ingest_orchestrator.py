"""Governed ingest orchestrator (M4b) — disk governor + EVICT-AFTER-RECORD + resume.

All fakes, no network/zips. The load-bearing test is that raw is NEVER deleted before its symbol's
lake partition is recorded (the invariant the adversarial review targets)."""

from __future__ import annotations

import pytest

from trikaal.data.ingest_orchestrator import (
    GB,
    DiskGovernor,
    DiskGovernorAbort,
    DownloadResult,
    EvictBeforeRecordError,
    FatalIngestError,
    GovernorKnobs,
    IngestOrchestrator,
    Ledger,
    evict_raw,
)


# ----------------------------------------------------------------- evict-after-record chokepoint
def test_evict_raw_refuses_unrecorded_symbol(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    raw = tmp_path / "BTC.zip"
    raw.write_bytes(b"x" * 100)
    with pytest.raises(EvictBeforeRecordError):
        evict_raw("BTCUSDT", [raw], led)  # not lake-built → refuse
    assert raw.exists(), "raw must survive a refused eviction"


def test_evict_raw_deletes_after_record(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    raw = tmp_path / "BTC.zip"
    raw.write_bytes(b"x" * 100)
    led.record_lake_built("BTCUSDT", "sha256:abc", {"BTC.zip": "sha"}, 44640)
    freed = evict_raw("BTCUSDT", [raw], led)
    assert freed == 100 and not raw.exists()


# ----------------------------------------------------------------- governor decision logic
def test_governor_ok_wait_abort():
    free = [600 * GB]
    gov = DiskGovernor(GovernorKnobs(raw_cache_cap_gb=10, min_free_disk_gb=50), lambda: free[0])
    assert gov.plan_evictions(2 * GB, []) == ("ok", [])  # room
    gov.raw_cache_bytes = 9 * GB
    # need 2 GB but cap is 10 and cache 9 → must evict; evict the 5GB built symbol
    status, plan = gov.plan_evictions(2 * GB, [("AAA", 5 * GB), ("BBB", 1 * GB)])
    assert status == "ok" and plan == ["AAA"]  # largest-first
    # nothing evictable, but an empty cache could still fit → wait (in-flight may free room)
    assert gov.plan_evictions(2 * GB, [])[0] == "wait"
    # a single incoming bigger than the whole cap → abort
    with pytest.raises(DiskGovernorAbort):
        gov.plan_evictions(11 * GB, [])


def test_governor_floor_binds():
    free = [51 * GB]
    gov = DiskGovernor(GovernorKnobs(raw_cache_cap_gb=380, min_free_disk_gb=50), lambda: free[0])
    # only 1 GB above the floor → a 2 GB incoming violates the floor, nothing evictable → abort
    with pytest.raises(DiskGovernorAbort):
        gov.plan_evictions(2 * GB, [])


# ----------------------------------------------------------------- orchestrator with fakes
def _fakes(tmp_path, shard_gb=1.0):
    """download writes a real (small) stand-in zip; finalize asserts raw still present (proves the
    evict-after-record ordering) and returns a deterministic hash."""
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    order: list[str] = []

    def download(sym, per):
        p = rawdir / f"{sym}-{per}.zip"
        p.write_bytes(b"\0" * 1024)  # tiny on disk; we account size via est_bytes_fn
        order.append(f"dl:{sym}:{per}")
        return DownloadResult(sym, per, [p], int(shard_gb * GB), {p.name: "sha256:fake"})

    def finalize(sym, periods):
        for per in periods:  # raw of EVERY month must still exist at finalize time
            assert (rawdir / f"{sym}-{per}.zip").exists(), f"raw {sym}-{per} evicted before build!"
        order.append(f"build:{sym}")
        return (
            f"sha256:{sym}",
            44640 * len(periods),
            {f"{sym}-{p}.zip": "sha256:fake" for p in periods},
        )

    return download, finalize, order, rawdir


def test_orchestrator_builds_all_and_records(tmp_path):
    plan = {"AAAUSDT": ["2023-01", "2023-02"], "BBBUSDT": ["2023-01"]}
    download, finalize, _order, _ = _fakes(tmp_path)
    led = Ledger(tmp_path / "ledger.jsonl")
    orch = IngestOrchestrator(
        plan=plan,
        knobs=GovernorKnobs(raw_cache_cap_gb=100, min_free_disk_gb=1, max_concurrent_downloads=3),
        ledger=led,
        download_fn=download,
        finalize_symbol_fn=finalize,
        est_bytes_fn=lambda s, p: GB,
        disk_free_fn=lambda: 600 * GB,
    )
    rep = orch.run()
    assert rep.n_symbols_built == 2
    assert rep.total_bars == 44640 * 3
    assert led.is_lake_built("AAAUSDT") and led.is_lake_built("BBBUSDT")
    # ledger persisted → a fresh Ledger sees them (resume substrate)
    assert Ledger(tmp_path / "ledger.jsonl").built_symbols() == ["AAAUSDT", "BBBUSDT"]


def test_orchestrator_evicts_under_tight_cap_after_record(tmp_path):
    # 3 single-month symbols, cap = 2 GB → can't hold all 3 raw at once → must evict built ones
    plan = {"AAAUSDT": ["2023-01"], "BBBUSDT": ["2023-01"], "CCCUSDT": ["2023-01"]}
    download, finalize, order, rawdir = _fakes(tmp_path)
    led = Ledger(tmp_path / "ledger.jsonl")
    orch = IngestOrchestrator(
        plan=plan,
        knobs=GovernorKnobs(raw_cache_cap_gb=2, min_free_disk_gb=1, max_concurrent_downloads=3),
        ledger=led,
        download_fn=download,
        finalize_symbol_fn=finalize,
        est_bytes_fn=lambda s, p: GB,
        disk_free_fn=lambda: 600 * GB,
    )
    rep = orch.run()
    assert rep.n_symbols_built == 3 and not rep.aborted
    assert rep.n_symbols_evicted == 3  # every symbol's raw reclaimed after its lake build
    assert not list(rawdir.glob("*.zip")), "all raw evicted post-build"
    # finalize never fired before its raw was present (asserted inside the fake) — invariant held
    assert order.count("build:AAAUSDT") == 1


def test_orchestrator_aborts_when_symbol_exceeds_cap(tmp_path):
    # one symbol needs 3 months * 1 GB staged at once, cap only 2 GB → unbuildable → clean abort
    plan = {"AAAUSDT": ["2023-01", "2023-02", "2023-03"]}
    download, finalize, _order, _ = _fakes(tmp_path)
    led = Ledger(tmp_path / "ledger.jsonl")
    orch = IngestOrchestrator(
        plan=plan,
        knobs=GovernorKnobs(raw_cache_cap_gb=2, min_free_disk_gb=1, max_concurrent_downloads=3),
        ledger=led,
        download_fn=download,
        finalize_symbol_fn=finalize,
        est_bytes_fn=lambda s, p: GB,
        disk_free_fn=lambda: 600 * GB,
    )
    rep = orch.run()
    assert rep.aborted and rep.n_symbols_built == 0
    assert "exceeds raw_cache_cap_gb" in (rep.error or "")


def test_orchestrator_isolates_a_failing_symbol(tmp_path):
    """A download/build failure on one symbol must not sink the run — others still build."""
    plan = {"GOODUSDT": ["2023-01"], "BADUSDT": ["2023-01"], "ALSOUSDT": ["2023-01"]}
    download, finalize, _order, _ = _fakes(tmp_path)

    def flaky_download(sym, per):
        if sym == "BADUSDT":
            raise RuntimeError("simulated 404 / verify failure")
        return download(sym, per)

    led = Ledger(tmp_path / "ledger.jsonl")
    orch = IngestOrchestrator(
        plan=plan,
        knobs=GovernorKnobs(raw_cache_cap_gb=100, min_free_disk_gb=1, max_concurrent_downloads=3),
        ledger=led,
        download_fn=flaky_download,
        finalize_symbol_fn=finalize,
        est_bytes_fn=lambda s, p: GB,
        disk_free_fn=lambda: 600 * GB,
    )
    rep = orch.run()
    assert not rep.aborted
    assert rep.failed_symbols == ["BADUSDT"]
    assert rep.n_symbols_built == 2  # GOOD + ALSO built despite BAD failing
    assert led.is_lake_built("GOODUSDT") and not led.is_lake_built("BADUSDT")


def _orch(tmp_path, plan, download, finalize, **knob):
    led = Ledger(tmp_path / "ledger.jsonl")
    return IngestOrchestrator(
        plan=plan,
        knobs=GovernorKnobs(
            max_concurrent_downloads=3, **{"raw_cache_cap_gb": 100, "min_free_disk_gb": 1, **knob}
        ),
        ledger=led,
        download_fn=download,
        finalize_symbol_fn=finalize,
        est_bytes_fn=lambda s, p: GB,
        disk_free_fn=lambda: 600 * GB,
    )


def test_mid_symbol_download_failure_leaves_no_phantom_or_leak(tmp_path):
    """HIGH regression: month-2 fails AFTER month-1 staged → release the full footprint + delete the
    staged zip. No counter phantom (raw_cache_bytes==0), no leaked raw on disk."""
    plan = {"AAAUSDT": ["2023-01", "2023-02"]}
    download, finalize, _order, rawdir = _fakes(tmp_path)

    def flaky(sym, per):
        if per == "2023-02":
            raise RuntimeError("transient download failure mid-symbol")
        return download(sym, per)

    orch = _orch(tmp_path, plan, flaky, finalize)
    rep = orch.run()
    assert rep.failed_symbols == ["AAAUSDT"] and rep.n_symbols_built == 0
    assert orch._gov.raw_cache_bytes == 0, "counter phantom: earlier staged month not released"
    assert not list(rawdir.glob("*.zip")), "disk leak: month-1 raw not deleted on failure"


def test_finalize_failure_does_not_over_release(tmp_path):
    """HIGH regression: finalize raises after est→real conversion → release exactly once (no negative
    counter from subtracting est twice)."""
    plan = {"AAAUSDT": ["2023-01"]}
    download, _finalize, _order, rawdir = _fakes(tmp_path)

    def bad_finalize(sym, periods):
        raise RuntimeError("reduce/features blew up")

    orch = _orch(tmp_path, plan, download, bad_finalize)
    rep = orch.run()
    assert rep.failed_symbols == ["AAAUSDT"]
    assert orch._gov.raw_cache_bytes == 0, "over-release drove the counter negative"
    assert not list(rawdir.glob("*.zip"))


def test_failure_phantom_does_not_spuriously_abort_a_later_symbol(tmp_path):
    """A prior failure must not strand a perfectly-buildable later symbol via a phantom cache."""
    plan = {"AAAUSDT": ["2023-01", "2023-02"], "BBBUSDT": ["2023-01"]}
    download, finalize, _order, _ = _fakes(tmp_path)

    def flaky(sym, per):
        if sym == "AAAUSDT" and per == "2023-02":
            raise RuntimeError("fail A mid-symbol")
        return download(sym, per)

    # tight cap: a phantom from A's failure would block B; correct release lets B build
    orch = _orch(tmp_path, plan, flaky, finalize, raw_cache_cap_gb=2)
    rep = orch.run()
    assert not rep.aborted
    assert rep.failed_symbols == ["AAAUSDT"]
    assert orch.ledger.is_lake_built("BBBUSDT") and rep.n_symbols_built == 1


def test_fatal_finalize_error_aborts_whole_run(tmp_path):
    """A FatalIngestError from finalize (e.g. the in-line causal sweep going RED) must ABORT the
    whole run — NOT be isolated like a transient per-symbol fault (contrast
    test_orchestrator_isolates_a_failing_symbol, which uses a plain RuntimeError) — and the leaking
    symbol must never be recorded lake-built (finalize raises before record_lake_built)."""
    plan = {"AAAUSDT": ["2023-01"], "BBBUSDT": ["2023-01"]}
    download, _finalize, _order, _rawdir = _fakes(tmp_path)

    def fatal_finalize(sym, periods):
        raise FatalIngestError("real-universe causal sweep RED on AAAUSDT [live] — lookahead leak")

    led = Ledger(tmp_path / "ledger.jsonl")
    orch = IngestOrchestrator(
        plan=plan,
        knobs=GovernorKnobs(raw_cache_cap_gb=100, min_free_disk_gb=1, max_concurrent_downloads=3),
        ledger=led,
        download_fn=download,
        finalize_symbol_fn=fatal_finalize,
        est_bytes_fn=lambda s, p: GB,
        disk_free_fn=lambda: 600 * GB,
    )
    rep = orch.run()
    assert rep.aborted and rep.n_symbols_built == 0
    assert "causal sweep RED" in (rep.error or "")
    assert not led.built_symbols()  # leaking symbol (and everything) never recorded


def test_orchestrator_path_per_symbol_independence(tmp_path):
    """The 1→N leak guard THROUGH the real production path (IngestOrchestrator): each symbol's
    recorded dataset_hash must be bit-identical to building that symbol ALONE — even under
    concurrent downloads + a tight evicting cap that interleaves raw-eviction between builds.

    test_per_symbol_independence pins this on the in-memory fan-out (build_universe_from_streams);
    this pins it on the orchestrator, whose injected finalize runs the REAL compute_features per
    symbol while the governor evicts. Synthetic streams stand in for the (downloaded) raw so the
    test needs no network/zips; the placeholder zips exercise the disk governor + eviction."""
    from trikaal.data.config import synthetic_test_config
    from trikaal.data.synthetic import make_synthetic_stream
    from trikaal.data.universe import SymbolSpec
    from trikaal.data.universe_build import build_one_symbol

    cfg = synthetic_test_config(
        half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20, vol_window=24, n_warm_vol=12
    )
    win = dict(window_start="2019-01-01", window_end="2030-01-01")
    syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    streams = {
        s: make_synthetic_stream(seed=i + 1, n_bars=720 + 40 * i)[0] for i, s in enumerate(syms)
    }
    specs = {s: SymbolSpec(symbol=s) for s in syms}

    # reference: each symbol built ALONE (root=None → hash only, no lake write, no orchestration)
    alone = {
        s: build_one_symbol(streams[s], specs[s], **win, cfg=cfg, root=None).dataset_hash
        for s in syms
    }

    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    lake = tmp_path / "lake"

    def download(sym, per):
        p = rawdir / f"{sym}-{per}.zip"
        p.write_bytes(b"\0" * 1024)
        return DownloadResult(sym, per, [p], GB, {p.name: "sha256:fake"})

    def finalize(sym, periods):
        # REAL feature build through the (concurrent + evicting) orchestrator, from the in-memory
        # stand-in stream → writes the lake partition exactly as production does.
        b = build_one_symbol(streams[sym], specs[sym], **win, cfg=cfg, root=lake)
        return b.dataset_hash, b.n_bars, {f"{sym}-{p}.zip": "sha256:fake" for p in periods}

    led = Ledger(tmp_path / "ledger.jsonl")
    orch = IngestOrchestrator(
        plan={s: ["2023-01"] for s in syms},
        # tight cap (2 GB vs 3×1 GB) FORCES eviction interleaving between builds; 3 concurrent
        knobs=GovernorKnobs(raw_cache_cap_gb=2, min_free_disk_gb=1, max_concurrent_downloads=3),
        ledger=led,
        download_fn=download,
        finalize_symbol_fn=finalize,
        est_bytes_fn=lambda s, p: GB,
        disk_free_fn=lambda: 600 * GB,
    )
    rep = orch.run()
    assert not rep.aborted and rep.n_symbols_built == 3
    assert rep.n_symbols_evicted == 3  # eviction genuinely interleaved with builds
    for s in syms:
        assert led._built[s]["dataset_hash"] == alone[s], (
            f"{s}: dataset_hash changed through the orchestrator — cross-symbol/eviction leak"
        )


def test_orchestrator_resumes_from_ledger(tmp_path):
    plan = {"AAAUSDT": ["2023-01"], "BBBUSDT": ["2023-01"]}
    download, finalize, order, _ = _fakes(tmp_path)
    led = Ledger(tmp_path / "ledger.jsonl")
    led.record_lake_built("AAAUSDT", "sha256:AAAUSDT", {}, 44640)  # pretend AAA already done
    orch = IngestOrchestrator(
        plan=plan,
        knobs=GovernorKnobs(raw_cache_cap_gb=100, min_free_disk_gb=1, max_concurrent_downloads=2),
        ledger=led,
        download_fn=download,
        finalize_symbol_fn=finalize,
        est_bytes_fn=lambda s, p: GB,
        disk_free_fn=lambda: 600 * GB,
    )
    rep = orch.run()
    # AAA skipped (not re-downloaded), only BBB processed this run
    assert "dl:AAAUSDT:2023-01" not in order
    assert "build:BBBUSDT" in order
    assert rep.n_symbols_built == 2  # ledger total
