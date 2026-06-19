# Milestone 4b — Universe ingest at scale (parallel + raw-cache, LOCAL)

**Status:** machinery built + adversarially reviewed; real ingest **executed on a bounded, resumable
subset** of the survivorship-correct top-200 universe. **No cloud spend** (run locally on the
operator's ~660 GB-free disk). The full 200-pair × multi-year lake is a one-command, bandwidth-bound
continuation of this exact run (see §6).

This is **data-only** — no GPU, no Stage-2 training, no M6. Per the standing rule, M6 does not start
until this report clears the M6 entry-gate check.

---

## 1. The decisive finding: the link is the wall

Before building, I measured the actual ingest economics (the operator's "parallel mode" assumes
concurrency buys throughput):

| Measurement | Value |
|---|---|
| Single-stream download | 7.8 MB/s (real ADAUSDT aggTrades, 61.5 MB) |
| **6-way parallel aggregate** | **8.3 MB/s** (730 MB across 6 concurrent) |
| Reduce + features throughput | 16.8 MB/s/core (compressed aggTrades) |
| Free disk | 660 GB |

**Concurrency does not speed the download — the local link saturates at ~8 MB/s regardless.** So the
producer–consumer design is built to *overlap* download with reduce and keep the link busy, not to
multiply bandwidth; and the full-universe download time is a hard floor set by the link, not the CPU
or the parallelism. This reframes M4b: the bottleneck is bandwidth + storage, never compute.

## 2. The real universe (bootstrapped FROM THE DATA, survivorship-owned)

The survivorship-correct symbol set comes from the **dump index**, not the live `exchangeInfo`
(which omits delisted symbols). One paginated scan of the `aggTrades` S3 index yields, per symbol and
**for free (no bar download)**: the available monthly dumps (→ listing = first month; delisting =
month-after-last *iff* it trails the latest global month) and the total aggTrades bytes (→ a
liquidity rank proxy). Result for window **2021-01-01 → 2025-01-01, top-200 USDT perps**:

- **200 symbols selected, 41 delisted** (e.g. AGIXUSDT 2023-02→2024-07, AUDIOUSDT 2021-08→2024-06) —
  each contributes only its `[listing, delisting)` bars. (The delisting decision carries a 1-month
  **publish-lag tolerance** — see §7 — so a still-live symbol that merely trails the freshest dump by
  a month is not mis-stamped delisted; this corrected 3 symbols vs a zero-tolerance comparison.)
- **Full-plan download ≈ 432 GB aggTrades + ~13 GB klines ≈ 445 GB** → at the measured 8.3 MB/s,
  **≈ 14.5 hours** of download (bandwidth-bound; unbeatable by parallelism).
- Est ≈ **315 M bars** — below the nominal 500 M–1 B band (flagged): 200 pairs over the fullest
  window, with many partial-coverage (delisted / late-listed) symbols, lands ~315 M. Per the
  corpus-purpose note (§5) this is *adequate* — a 27 M backbone saturates ~1–2 B bars, so the band is
  itself past-saturation; reaching it would need ~320 pairs or a wider net, at proportional bandwidth.

The full populated config (200 symbols, real listing/delisting + per-symbol volume proxy +
corpus-purpose metadata) is written to **`config/universe_full.yaml`** (`config_hash` recorded).

## 3. Execution mode — governed producer–consumer (the machinery)

`src/trikaal/data/ingest_orchestrator.py`: a thread-pool of SHA-256-verified downloaders feeding a
single-threaded consumer that reduces each symbol **in memory** → feature lake → manifest-records it
→ **evicts that symbol's raw**. Single-threaded reduce/evict ⇒ the disk governor and the
evict-after-record invariant have no races.

**Disk model (measured, and it changed the design).** The reduced-shard npz still keeps every trade's
size pool, so it is ~60 % of the raw zip (231 MB vs 389 MB for BTCUSDT 2023-01) — *not* a cheap
intermediate. The genuinely small kept artifact is the **feature lake** (~3.9 MB/month/symbol). So
M4b does **not** persist npz; it reduces in memory straight to the lake and evicts raw. The bounded
cache is the raw of the *in-flight* symbols only.

**Disk governor knobs:** `raw_cache_cap_gb` (pause downloads above this), `min_free_disk_gb` (hard
floor), `max_concurrent_downloads`. The governor evicts lake-built symbols' raw (largest-first) to
make room; if even an empty cache can't fit one symbol it aborts cleanly (cap < largest symbol's
raw); if room can't be made now but in-flight work will free it, it waits.

**The load-bearing invariant — evict-after-record.** A raw shard is unlinked **only after** its
symbol's lake partition is built **and** the ledger commit marker is flushed. `evict_raw` is the
single chokepoint and hard-asserts `is_lake_built(symbol)`; the evictable set is populated *only
after* `record_lake_built`. **Resumable** from the append-only JSONL ledger (window-scoped): built
symbols are skipped; a non-built symbol's raw is re-fetched if evicted (idempotent verified
download) or reused if cached. One symbol's download/build failure is **isolated** (logged, skipped),
never sinking the run.

## 4. Realized run (bounded, real, resumable)

Bounded to a bandwidth-feasible, resumable **prefix** of the top-200 (smallest-footprint first +
guaranteed delisted symbols), full **2021–2025** window, **tight 3 GB cap to force real eviction**.
Executed as two runs (a 12-symbol run stopped after 2 builds to demonstrate resume, then a 4-symbol
run that **resumed** — the ledger correctly skipped the 2 already-built symbols):

| Symbol | Bars | Coverage (real, from data) | Regimes spanned |
|---|---|---|---|
| **KLAYUSDT** | 1,606,830 | 2021-10-12 → 2024-10-31 | **all four** (2021 bull / 2022 bear+FTX / 2023 recovery / 2024 ETF) |
| KEYUSDT | 846,000 | 2023-05-24 → 2024-12-31 | 2023 recovery, 2024 ETF |
| SAGAUSDT | 383,850 | 2024-04-09 → 2024-12-31 | 2024 ETF |
| ZROUSDT | 279,930 | 2024-06-20 → 2024-12-31 | 2024 ETF |
| **Total** | **3,116,610 bars** (DuckDB count over the lake) | 73 month-shards, 4 symbols | multi-regime ✓ |

**Realized invariants on REAL data:**
- **Eviction held the cache bounded:** peak raw cache **0.8 GB**, evicted 2 symbols / **1.4 GB** in
  the final run; cache → 0.0 GB after the last build. Final **free disk 661 GB**; only the **281 MB
  lake** is retained (all raw reclaimed; the M2 BTCUSDT raw + lake untouched).
- **Cross-section causal exhaustive sweep GREEN** on real universe bars: ZROUSDT 800-bar slice →
  `passed=True, coverage=100%, 6398 checks`.
- **Resume proven:** the 4-symbol run skipped KEYUSDT + SAGAUSDT (already in the ledger).
- **Real delisting truncation exercised:** KLAYUSDT delisted **2024-11** (within the window —
  derived from dump coverage, not memory) → manifest `tail_truncated=True`, lake ends 2024-10-31.
- **Universe Merkle root (over the lake):** `sha256:f0ca2d37225cbacf…`;
  **manifest content_hash:** `sha256:a428c807fd0c7b99…`. Per-symbol coverage in the manifest is read
  back from the **lake itself** (DuckDB min/max date + segment count), so it is accurate for every
  built symbol including ones only resumed.

The lake lands in `processed/universe_bars/` (separate from the M2/M3/M5 `processed/bars/` — those
are untouched); the content-hashed manifest in `processed/universe/universe_manifest.json`; the
resumable ledger in `processed/universe/ingest_ledger_2021-01-01_2025-01-01.jsonl`.

**Reproducibility survives raw eviction.** The universe Merkle root + per-symbol content-hashes are
over the **reduced lake** (the kept artifact). The raw Binance SHA-256 of every shard is recorded in
the manifest for **provenance only** — re-pullable raw → deterministic reducer (proven byte-identical
in M4a) → reproduces the recorded lake hash. No raw retention required.

## 5. Corpus purpose (recorded in manifest metadata)

This lake is the **eval + bounded-training corpus, sized for REGIME × SYMBOL breadth, not
training-loss reduction**. A 27 M backbone is ~compute-optimal at ~540 M tokens (Chinchilla ~20
tok/param) and saturates ~1–2 B bars; 1-minute crypto bars are redundant so effective saturation
comes sooner. The M6/M7 **training** token budget is therefore set *separately* and bounded near
compute-optimal (single-pass-ish over a drawn subset), **not** all-bars × N-epochs. In-distribution
training loss is **expected** to plateau in this range — by design, not a bug. (Verbatim in
`config/universe_full.yaml` and the manifest `metadata`.)

## 6. Exit-gate status

| Gate criterion | Status |
|---|---|
| Full universe lake (200 pairs, multi-year), Parquet+DuckDB, content-hashed manifest w/ listing+delisting + Merkle root + corpus-purpose | ⚠ **machinery + config + manifest done; lake realized on a bounded resumable subset** — the full 200-pair lake is the 14.5 h bandwidth-bound continuation (`python scripts/m4b_universe_ingest.py` with no `--max-*`). NOT completable in-session at 8 MB/s. |
| Realized bar count in ~500M–1B band (flag if off) | **Flagged:** full plan ≈ 315 M (below band; adequate for the corpus purpose — see §2/§5). Realized subset bars in §4. |
| Causal exhaustive sweep GREEN on a real-universe cross-section | ✅ run in-line on a real ingested symbol (§4 / summary) |
| Peak disk, final lake size, raw retention outcome | ✅ reported (§4) |
| "universe Merkle root over the lake; raw checksums for provenance only" | ✅ printed + in manifest metadata |
| Tests green + ruff clean; evict-after-record adversarially reviewed | ✅ **154 tests green, ruff clean.** Evict-after-record verdict: **invariant holds, no violations** (single chokepoint, structural ordering, crash-safe across all kill-windows). A 14-agent verification workflow over the whole M4b surface **found + fixed 2 HIGH + 1 MED** (§7). |

**Honest bottom line.** The M4b *machine* is built, tested, reviewed, and proven on real
multi-symbol, multi-regime, survivorship-correct data with the governor + eviction + resume all
exercised. The full 200-pair × 4-year lake was **not** completed in-session because the download is a
hard ~14.5-hour, 8 MB/s-bandwidth-bound job that parallelism cannot shorten — it is one resumable
command away, and every criterion except "all 200 symbols downloaded" is met. **Decision for the
supervisor:** let the resumable job run to completion locally (~14.5 h wall, free), or move the
download to a higher-bandwidth box (the runbook's cloud path) before M6.

## 7. Adversarial verification — multi-agent, every finding verified

A 14-agent workflow reviewed the M4b surface across four dimensions (bootstrap survivorship,
orchestrator concurrency/disk-accounting, manifest+hashing-scope, runner wiring), each finding then
adversarially verified by an independent skeptic. It earned its keep — **2 HIGH + 1 MED, all real,
all fixed:**

- **HIGH — dump-publish-lag fabricated delistings (survivorship corruption).** `derive_symbol_spec`
  marked a symbol delisted if its last dump trailed the single freshest-published symbol by even one
  month — but Binance publishes with non-uniform per-symbol lag, so a *live* symbol legitimately
  trails by ~1 month, and the code dropped its freshest (most valuable) month. **Fixed:** a
  month-index `lag_months` tolerance (default 1) — a symbol within the lag window stays live. This
  flipped 3 false-delistings back to live (44 → 41 delisted) on the real universe. Pinned by a new
  `tests/data/test_universe_bootstrap.py` boundary test.
- **HIGH — orchestrator failure-leg disk leak + counter phantom.** My per-symbol failure isolation
  under-released a multi-month symbol's *earlier* staged bytes (a phantom in the cache counter) and
  leaked its raw zips; a phantom could later spuriously abort a perfectly-buildable symbol. **Fixed:**
  release the symbol's full real footprint + the live reservation exactly once, delete its staged raw
  directly (not via `evict_raw`, which asserts `is_lake_built`), and discard any *late-arriving*
  download of an already-failed symbol. Writing the fix surfaced a second ordering bug (month-2 fails
  while month-1 is still in-flight) which the new regression tests caught and which is now also fixed.
- **MED — interior gap silently dropped a symbol.** A delist-relist or never-published interior
  month made the orchestrator plan (built from the `[listing, delisting)` hull) request a
  non-existent month → 404 → the symbol was dropped wholesale. **Fixed:** the plan is built from the
  **actual dump months** (`index[sym].months ∩ active-window`), so non-existent months are never
  requested; the gap becomes a segment boundary (causally safe), preserving all real data.
- **LOW** — `bootstrap_from_index({})` crashed on `max()` of an empty sequence → now returns an empty
  universe. (Other LOWs — the byte-proxy proration approximation, monthly-granularity delist day —
  are documented limitations, no code change.)

The evict-after-record invariant itself was re-verified clean. All fixes are covered by the +10 new
tests (154 total green).

## 8. Reproduce

```bash
# write the full config + plan, no download:
python scripts/m4b_universe_ingest.py --config-only --top-n 200 \
  --window-start 2021-01-01 --window-end 2025-01-01

# a bounded real run (what was executed here): smallest-footprint prefix, tight cap forces eviction:
python scripts/m4b_universe_ingest.py --max-symbols 12 --max-gb 10 \
  --raw-cache-cap-gb 3 --min-free-disk-gb 30

# the full run (resumable; ~14.5 h bandwidth-bound at ~8 MB/s):
python scripts/m4b_universe_ingest.py --top-n 200 \
  --window-start 2021-01-01 --window-end 2025-01-01 \
  --raw-cache-cap-gb 380 --min-free-disk-gb 50

# orchestrator + bootstrap unit gates:
pytest tests/data/test_ingest_orchestrator.py tests/data/test_universe.py -q
```
