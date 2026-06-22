"""M4b — universe ingest at scale (parallel + raw-cache, LOCAL). Per docs/milestone4b_runbook.md.

Bootstraps the survivorship-correct top-N universe from the Binance dump index, writes
``config/universe_full.yaml``, then runs the governed producer–consumer orchestrator: pooled
SHA-256-verified downloads fill a bounded raw cache; each symbol is reduced in-memory → feature
lake → manifest-recorded → its raw EVICTED (evict-after-record). Fully resumable from the ledger.

Usage:
    python scripts/m4b_universe_ingest.py --top-n 200
        --window-start 2021-01-01 --window-end 2025-01-01
        [--max-symbols K] [--max-gb G] [--raw-cache-cap-gb C] [--min-free-disk-gb F]
        [--max-concurrent-downloads N] [--config-only]

``--config-only`` writes the full universe config + plan (no download). ``--max-symbols`` /
``--max-gb`` bound the actually-ingested subset (a strict, resumable prefix of the full plan) so a
real run completes inside the local bandwidth budget; the lake/manifest are the same artifacts the
full run extends. The link is the throughput wall (~8 MB/s measured), so concurrency only overlaps
download with reduce — it does not multiply bandwidth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

from trikaal.data.build import build_symbol_stream
from trikaal.data.causal_check import exhaustive_truncation_sweep
from trikaal.data.config import FeatureConfig
from trikaal.data.ingest import DumpSpec, ingest_month
from trikaal.data.ingest_orchestrator import (
    GB,
    DownloadResult,
    FatalIngestError,
    GovernorKnobs,
    IngestOrchestrator,
    Ledger,
    default_disk_free,
)
from trikaal.data.lake import connect as lake_connect
from trikaal.data.manifest import SymbolCoverage, UniverseManifest
from trikaal.data.raw_store import RAW_ROOT_DEFAULT
from trikaal.data.universe import SelectionRule, date_to_ms
from trikaal.data.universe_bootstrap import (
    SymbolIndex,
    bootstrap_from_index,
    scan_aggtrades_index,
)
from trikaal.data.universe_build import build_one_symbol, slice_stream_by_ms
from trikaal.utils.hashing import content_hash

LAKE_ROOT = Path("processed/universe_bars")
LEDGER_PATH = Path("processed/universe/ingest_ledger.jsonl")
INDEX_CACHE = Path("processed/universe/agg_index.json")

CORPUS_PURPOSE = {
    "purpose": "eval + bounded-training corpus, sized for REGIME x SYMBOL breadth, NOT loss "
    "reduction",
    "saturation_note": "a 27M backbone is ~compute-optimal at ~540M tokens (Chinchilla ~20 "
    "tok/param) and saturates ~1-2B bars; 1m crypto bars are redundant so effective saturation "
    "comes sooner. The M6/M7 training token budget is set SEPARATELY and bounded near "
    "compute-optimal (single-pass-ish over a drawn subset), NOT all-bars x N-epochs. "
    "In-distribution training loss is EXPECTED to plateau in this range; by design, not a bug.",
    "hashing_scope": "universe Merkle root is over the LAKE; raw checksums recorded for "
    "provenance only",
}


def load_index() -> dict[str, SymbolIndex]:
    if INDEX_CACHE.exists():
        raw = json.loads(INDEX_CACHE.read_text())
        return {
            s: SymbolIndex(symbol=s, months=d["months"], total_agg_bytes=d["total_agg_bytes"])
            for s, d in raw.items()
        }
    idx = scan_aggtrades_index("um")
    INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_CACHE.write_text(
        json.dumps(
            {
                s: {"months": si.months, "total_agg_bytes": si.total_agg_bytes}
                for s, si in idx.items()
            }
        )
    )
    return idx


def write_config(spec, est: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "name": spec.name,
        "frequency": spec.frequency,
        "market": spec.market,
        "window": {"start": spec.window_start, "end": spec.window_end},
        "selection": spec.selection.to_dict(),
        "corpus_purpose": CORPUS_PURPOSE,
        "config_hash": spec.config_hash(),
        "symbols": [
            {**s.to_dict(), "est_in_window_agg_bytes": est.get(s.symbol, 0)}
            for s in sorted(spec.symbols, key=lambda s: s.symbol)
        ],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))


def select_subset(spec, est, *, max_symbols, max_gb, window):
    """A real, resumable PREFIX of the full plan that fits the bandwidth budget. Smallest-footprint
    symbols first (fast), guaranteeing >= 1 delisted symbol so truncation is exercised on real."""
    live = list(spec.active_symbols())
    by_size = sorted(live, key=lambda s: est.get(s.symbol, 0))
    chosen: list = []
    budget = (max_gb or 1e9) * GB
    used = 0
    for s in by_size:
        if max_symbols and len(chosen) >= max_symbols:
            break
        sz = est.get(s.symbol, 0)
        if used + sz > budget and chosen:
            break
        chosen.append(s)
        used += sz
    if not any(s.delisting_date for s in chosen):
        delisted = next((s for s in by_size if s.delisting_date), None)
        if delisted:
            chosen[-1] = delisted
    return chosen


def rollup_manifest(spec, win, ledger, *, lake_root=LAKE_ROOT) -> UniverseManifest:
    """Roll the ledger + lake into a content-hashed manifest in ONE GROUP-BY pass over the lake.

    Coverage (first/last date, n_bars, n_segments) comes from a single `GROUP BY symbol` — a
    per-symbol query loop re-globs the whole partitioned lake once per symbol (~40 min at 200
    symbols / 200k+ files vs ~1-2 min for one pass). The universe Merkle is `content_hash` over the
    ledger's sorted `(symbol, dataset_hash)` pairs — the reproducibility anchor, independent of file
    layout (so it survives lake compaction). Shared by the normal run and `--manifest-only`.
    """
    spec_by_sym = {s.symbol: s for s in spec.symbols}
    built = ledger.built_symbols()
    cov: dict[str, tuple[str, str, int, int]] = {}
    if built:
        con = lake_connect(lake_root)
        for sym, fd, ld, nb, ns in con.execute(
            "SELECT symbol, CAST(min(date) AS VARCHAR), CAST(max(date) AS VARCHAR), "
            "count(*), count(DISTINCT segment_id) FROM bars GROUP BY symbol"
        ).fetchall():
            cov[sym] = (fd or "", ld or "", int(nb or 0), int(ns or 0))

    rows, lake_hashes = [], {}
    for sym in built:
        rec = ledger._built[sym]
        s = spec_by_sym.get(sym)
        first_d, last_d, n_bars, n_seg = cov.get(sym, ("", "", 0, 0))
        lake_hashes[sym] = rec["dataset_hash"]
        # truncated = the symbol's window-span was bounded by its OWN lifecycle, not the window edge
        head = bool(s and s.listing_date and date_to_ms(s.listing_date) > date_to_ms(win[0]))
        tail = bool(s and s.delisting_date and date_to_ms(s.delisting_date) < date_to_ms(win[1]))
        rows.append(
            SymbolCoverage(
                symbol=sym,
                market="um",
                listing_date=s.listing_date if s else None,
                delisting_date=s.delisting_date if s else None,
                oi_coverage_start=None,
                perp_available=True,
                first_bar_date=first_d,
                last_bar_date=last_d,
                n_bars=n_bars,
                n_segments=n_seg,
                head_truncated=head,
                tail_truncated=tail,
                dataset_hash=rec["dataset_hash"],
                raw_files=[{"file": k, "sha256": v} for k, v in rec.get("checksums", {}).items()],
            )
        )
    universe_hash = content_hash([[s, lake_hashes[s]] for s in sorted(lake_hashes)])
    return UniverseManifest(
        name=spec.name,
        frequency="1m",
        market="um",
        window_start=win[0],
        window_end=win[1],
        config_hash=spec.config_hash(),
        universe_hash=universe_hash,
        n_symbols=len(built),
        total_bars=sum(r.n_bars for r in rows),
        symbols=rows,
        built_at_utc=None,
        metadata=CORPUS_PURPOSE,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--window-start", default="2021-01-01")
    ap.add_argument("--window-end", default="2025-01-01")
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--max-gb", type=float, default=None)
    ap.add_argument("--raw-cache-cap-gb", type=float, default=380.0)
    ap.add_argument("--min-free-disk-gb", type=float, default=50.0)
    ap.add_argument("--max-concurrent-downloads", type=int, default=4)
    ap.add_argument("--sweep-sample-bars", type=int, default=800)
    ap.add_argument("--config-only", action="store_true")
    ap.add_argument(
        "--manifest-only",
        action="store_true",
        help="regenerate the manifest from the ledger + lake (no download/build/sweep)",
    )
    args = ap.parse_args()

    win = (args.window_start, args.window_end)
    sel = SelectionRule(
        top_n=args.top_n, min_live_days=30, include_delisted=True, quote_asset="USDT"
    )
    print(f"[bootstrap] scanning dump index … (window {win[0]}..{win[1]}, top-{args.top_n})")
    index = load_index()
    spec, est = bootstrap_from_index(index, window_start=win[0], window_end=win[1], selection=sel)
    live = spec.active_symbols()
    total_gb = sum(est.values()) / GB
    print(
        f"[bootstrap] {len(live)} symbols ({sum(1 for s in live if s.delisting_date)} delisted), "
        f"full-plan agg download ~{total_gb:.0f} GB"
    )
    write_config(spec, est, Path("config/universe_full.yaml"))
    print("[bootstrap] wrote config/universe_full.yaml + corpus-purpose metadata")
    if args.config_only:
        return 0

    if args.manifest_only:
        ledger = Ledger(LEDGER_PATH.with_name(f"ingest_ledger_{win[0]}_{win[1]}.jsonl"))
        manifest = rollup_manifest(spec, win, ledger)
        mpath = manifest.write(Path("processed/universe/universe_manifest.json"))
        print(f"[manifest-only] {manifest.n_symbols} symbols, {manifest.total_bars:,} bars")
        print(f"[manifest-only] written → {mpath}")
        print(f"[manifest-only] universe Merkle      : {manifest.universe_hash}")
        print(f"[manifest-only] manifest content_hash: {manifest.manifest_content_hash()}")
        return 0

    subset = select_subset(spec, est, max_symbols=args.max_symbols, max_gb=args.max_gb, window=win)
    spec_by_sym = {s.symbol: s for s in subset}
    # Plan from the ACTUAL dump months (∩ active window), not the [listing,delisting) hull — so a
    # symbol with an interior gap (delist-relist / never-published month) never requests a
    # non-existent month (which would 404 and drop the symbol); the gap becomes a segment boundary.
    plan = {
        s.symbol: [m for m in index[s.symbol].months if m in set(s.active_months(*win))]
        for s in subset
    }
    n_shards = sum(len(m) for m in plan.values())
    sub_gb = sum(est.get(s, 0) for s in plan) / GB
    n_del = sum(1 for s in subset if s.delisting_date)
    eta_min = sub_gb * 1e3 / 8.3 / 60
    print(
        f"[ingest] SUBSET: {len(subset)} symbols, {n_shards} shards, ~{sub_gb:.1f} GB "
        f"({n_del} delisted) — eta ~{eta_min:.0f} min"
    )

    checks_by_symbol: dict[str, dict] = defaultdict(dict)
    # one-time real-bar exhaustive causal sweep per coverage class — keyed "live"/"tail" so BOTH a
    # full-window symbol AND a delisting-tail-truncated symbol (the highest-risk causal seam, the
    # truncate∘causality path that ALL 41 delisted symbols share) are gated, not just the first.
    sweeps: dict[str, dict] = {}

    def download(sym: str, per: str) -> DownloadResult:
        paths, total, checks = [], 0, {}
        for stream in ("klines", "aggTrades"):
            rf = ingest_month(
                DumpSpec(sym, "futures/um", stream, per), root=RAW_ROOT_DEFAULT, run_id="m4b"
            )
            p = Path(rf.local_path)
            paths.append(p)
            total += rf.n_bytes
            checks[p.name] = rf.sha256
        checks_by_symbol[sym].update(checks)
        return DownloadResult(sym, per, paths, total, checks)

    def run_sweep(label: str, sym: str, stream, lo: float, hi: float) -> None:
        """One-time exhaustive truncation sweep on a slice of REAL universe bars (raw still in
        memory; O(T^2) so a head-slice keeps it fast). HARD GATE: a RED sweep — or a degenerate
        0-check / <100%-coverage sweep — raises FatalIngestError BEFORE the lake write below, so a
        symbol that fails causal safety is NEVER written, recorded, or evicted, and the whole run
        aborts (invariant #2 is not isolatable like a transient fault)."""
        sl = slice_stream_by_ms(stream, lo, hi)
        n = min(args.sweep_sample_bars, len(sl.bar_open_ms))
        sl = slice_stream_by_ms(sl, -1e18, float(sl.bar_open_ms[n - 1]) + 1)
        rep_sweep = exhaustive_truncation_sweep(sl, FeatureConfig())
        sweeps[label] = {
            "symbol": sym,
            "passed": rep_sweep.passed,
            "coverage": rep_sweep.coverage,
            "n_checks": rep_sweep.n_checks,
            "bars": len(sl.bar_open_ms),
        }
        if not (rep_sweep.passed and rep_sweep.coverage == 1.0 and rep_sweep.n_checks > 0):
            raise FatalIngestError(
                f"real-universe causal sweep RED on {sym} [{label}]: passed={rep_sweep.passed} "
                f"coverage={rep_sweep.coverage:.3f} checks={rep_sweep.n_checks} — lookahead leak"
            )

    def finalize(sym: str, months: list[str]) -> tuple[str, int, dict]:
        spec_s = spec_by_sym[sym]
        stream = build_symbol_stream(
            sym, months, raw_root=RAW_ROOT_DEFAULT, is_perp=spec_s.perp_available
        )
        # Coverage class by the SAME spec rule the manifest uses for tail_truncated (delisting
        # within the window) — NOT a bar test: bootstrapped delistings sit on the first-of-month
        # AFTER the last dump, so slice is identity at the tail and a bar test would label every
        # one of the 41 delisted symbols "live" and the tail sweep would never fire. This fires it
        # on the first real delisted symbol — a structurally distinct (shorter, earlier-ending) real
        # stream. Gate runs BEFORE the build/record/evict (raw still in memory here).
        lo, hi = spec_s.active_window_ms(win[0], win[1])
        is_tail = bool(
            spec_s.delisting_date and date_to_ms(spec_s.delisting_date) < date_to_ms(win[1])
        )
        label = "tail" if is_tail else "live"
        if label not in sweeps:
            run_sweep(label, sym, stream, lo, hi)
        build = build_one_symbol(
            stream,
            spec_s,
            window_start=win[0],
            window_end=win[1],
            cfg=FeatureConfig(),
            root=LAKE_ROOT,
        )
        return build.dataset_hash, build.n_bars, dict(checks_by_symbol[sym])

    def est_bytes(sym: str, per: str) -> int:
        si = index[sym]
        return int(si.total_agg_bytes / max(1, len(si.months))) + 1_800_000  # + klines

    # ledger is window-scoped so different windows never cross-skip a symbol on resume
    ledger = Ledger(LEDGER_PATH.with_name(f"ingest_ledger_{win[0]}_{win[1]}.jsonl"))
    knobs = GovernorKnobs(
        raw_cache_cap_gb=args.raw_cache_cap_gb,
        min_free_disk_gb=args.min_free_disk_gb,
        max_concurrent_downloads=args.max_concurrent_downloads,
    )
    orch = IngestOrchestrator(
        plan=plan,
        knobs=knobs,
        ledger=ledger,
        download_fn=download,
        finalize_symbol_fn=finalize,
        est_bytes_fn=est_bytes,
        disk_free_fn=default_disk_free(Path(".")),
    )
    t0 = time.time()
    rep = orch.run(progress=lambda m: print(f"  [{time.time() - t0:.0f}s] {m}", flush=True))
    dt = time.time() - t0

    # ---- roll up the ledger + lake into the content-hashed manifest (ONE GROUP-BY pass) ----
    built = ledger.built_symbols()
    manifest = rollup_manifest(spec, win, ledger)
    universe_hash = manifest.universe_hash
    mpath = manifest.write(Path("processed/universe/universe_manifest.json"))

    print("\n=== M4b ingest summary ===")
    print(
        f"  symbols built : {len(built)}/{len(subset)}   shards: {n_shards}   "
        f"(this run +{rep.n_symbols_built})"
    )
    print(f"  total bars    : {manifest.total_bars:,} (lake)   this run: +{rep.total_bars:,}")
    print(
        f"  peak raw cache: {rep.peak_raw_cache_bytes / GB:.1f} GB   "
        f"evicted: {rep.n_symbols_evicted} symbols / {rep.raw_bytes_evicted / GB:.1f} GB"
    )
    for label in ("live", "tail"):
        sw = sweeps.get(label)
        if sw:
            print(
                f"  causal sweep  : [{label}] {sw['symbol']} {sw['bars']} bars → "
                f"passed={sw['passed']} cov={sw['coverage'] * 100:.0f}% checks={sw['n_checks']}"
            )
    print(f"  universe hash : {universe_hash}")
    print(f"  manifest      : {mpath} (content_hash={manifest.manifest_content_hash()[:30]}…)")
    if rep.failed_symbols:
        print(f"  failed        : {len(rep.failed_symbols)} symbols → {rep.failed_symbols}")
    abort_msg = f"  {rep.error}" if rep.aborted else ""
    print(f"  wall          : {dt:.0f}s   aborted={rep.aborted}{abort_msg}")
    print(
        "  NOTE: universe Merkle root is over the lake; raw checksums recorded for provenance only."
    )

    # ---- HARD EXIT GATE (invariant #2 on the real-data path) ----
    # A sweep that ran but is not clean-GREEN → fail. A RED sweep already raised inside finalize and
    # aborted the run; this also defends against a degenerate 0-check / <100%-coverage record.
    def sweep_ok(d: dict | None) -> bool:
        return bool(d) and d["passed"] and d["coverage"] == 1.0 and d["n_checks"] > 0

    red = sorted(lab for lab, d in sweeps.items() if not sweep_ok(d))
    if red:
        print(f"FATAL: real-universe causal sweep RED {red} — lookahead leak")
        return 1
    if rep.aborted:
        print(f"FATAL: ingest aborted — {rep.error}")
        return 1
    # On the FULL run (no --max-* bounds) the exit gate REQUIRES the sweep to have fired GREEN on
    # BOTH a live and a delisting-tail symbol; a bounded smoke run is exempt (it may lack a class).
    full_run = args.max_symbols is None and args.max_gb is None
    missing = [lab for lab in ("live", "tail") if lab not in sweeps]
    if full_run and missing:
        print(f"FATAL: exit gate requires GREEN live+tail real-bar sweeps; missing {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
