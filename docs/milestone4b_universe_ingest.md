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
- **Full-plan download ≈ 432 GB aggTrades + ~13 GB klines ≈ 445 GB** → at the *burst* 8.3 MB/s this
  projected to **≈ 15 hours**. **Realized: ~52.8 h at ~2.3 MB/s _sustained_** (§4) — the burst rate
  did not hold over a 2-day pull, so the projection was ~3.6× optimistic. Either way the conclusion
  stands: it is **bandwidth-bound and unbeatable by parallelism** (the peak raw cache never pressured
  the 380 GB cap — disk and CPU were idle waiting on the link).
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

## 4. Realized run — the full 200-symbol universe

The full run completed locally (governed, resumable, per-symbol failure-isolated): **200 / 200
symbols, 304,625,181 bars**, 7,024 month-shards, window **2021-01-01 → 2025-01-01**. One symbol
(**AXSUSDT**) failed mid-run on a transient `TimeoutError` (download of its 2023-12 shard) and was
failure-isolated — the first pass finished **199/200 with `aborted=False`**; a **resume** (the same
launch command — the window-scoped ledger skipped the 199 built and re-attempted only AXSUSDT)
rebuilt it cleanly on a fresh download → **200/200**.

| Metric | Realized |
|---|---|
| Symbols | **200 / 200** (199 first pass + AXSUSDT recovered on resume) |
| Bars (DuckDB count over the lake) | **304,625,181** |
| Lake on disk | **23 GB** Hive-partitioned zstd Parquet (raw fully evicted; free disk 628 GB) |
| Peak raw cache / evicted | **30.0 GB** peak (380 GB cap — never pressured) ; **195 symbols / 503 GB** cycled through eviction → raw → 0 |
| Wall clock | **~52.8 h** (190,111 s) first pass + ~43 min AXS resume |
| Effective throughput | **~2.3 MB/s sustained** (445 GB / 190,111 s) — **~3.6× below** the 8.3 MB/s burst (§1); the home link's *sustained* rate, not its burst, set the wall, so the earlier ~15 h projection was optimistic (the AXS timeout at 34.5 h is consistent with a flaky link) |
| Universe Merkle (over the lake) | **`sha256:5dfd667d05b97bda…`** (200-sym; the 199-sym first-pass intermediate was `9400a271…`) |

**Causal safety held at scale (the hard gate, invariant #2).** The in-line exhaustive truncation
sweep fired GREEN on BOTH coverage classes on REAL universe bars: **`[live] HOOKUSDT`** and
**`[tail] YFIIUSDT`** (each an 800-bar slice → `passed=True, coverage=100%, 6398 checks`); the AXS
resume fired a third, **`[live] AXSUSDT`**, also GREEN. A RED sweep raises `FatalIngestError` and
aborts before recording/evicting — none did, across 200 symbols.

**Eviction / resume / governor at scale.** Peak raw cache **30 GB** against the 380 GB cap (downloads
never pressured the floor — the link, not disk, was the bottleneck); **503 GB** of raw cycled through
`download → SHA-256 verify → reduce-in-memory → record → evict`, leaving only the **23 GB lake** (raw
→ 0). The one transient failure was recovered by **resume** with zero data loss; `aborted=False`.

**Merkle anchor vs the manifest.** The 200-symbol universe Merkle (`5dfd667d…`) is `content_hash` over
the ledger's sorted `(symbol, lake dataset_hash)` pairs — the reproducibility anchor, derivable
directly from the append-only ledger (so it survives raw eviction). The rolled-up
`universe_manifest.json` is a derived, **gitignored** convenience; its per-symbol coverage rollup is
pathologically slow on this lake (see §4.1), so it currently reflects the 199-symbol first-pass
intermediate — a fast follow, **not** the anchor.

The lake lands in `processed/universe_bars/` (separate from the M2/M3/M5 `processed/bars/`, untouched);
the resumable ledger in `processed/universe/ingest_ledger_2021-01-01_2025-01-01.jsonl`.

### 4.1 Data quality — usability per ablation arm + the micro-availability read (for the M6 draw)

`scripts/m4b_data_quality.py` (single-pass DuckDB aggregation over the lake) reports per-symbol
usability for **both 2×2 arms**, with the mask-bit groups stated so the definition is auditable
(`m_i = 1` = masked: warm-up incomplete / zero-vol / vol-recon-err):
- **OHLCV arm** usable ⇔ bits **0-6** unmasked (ret_close, range, body, upper/lower wick, log_volume, log_amount);
- **+micro arm** usable ⇔ bits **0-12** unmasked (OHLCV **+** aggTrades microstructure 7-12: TFI, signed_count_imbalance, trade_count, mean_trade_size, trade_size_dispersion, large_trade_share);
- perp bits **13-15** (funding, log_oi, d_oi) are EXCLUDED — masked historically (§1.2 OI retention trap).

| | |
|---|---|
| Total bars | 304,625,181 |
| **OHLCV-usable** (post warm-up) | **303,283,972 — 99.56%** |
| **+micro-usable** | **300,311,536 — 98.58%** |
| **Micro-availability ratio** (micro-usable / OHLCV-usable) | **99.0% universe-wide ; 0 symbols below 0.7** |
| OHLCV-usable / symbol | min 279,121 · median 1,757,144 · max 2,102,886 |
| Avg is_stale% / zero-vol% | **0.175% / 0.376%** |
| Thin-coin candidates (OHLCV-usable < 0.5×median **or** stale > 10%) | **43** |

**Read for M6.** (1) **Usability is high** — warm-up costs only ~0.3% of bars per symbol (most coins
are long single segments, so the ~800-bar z-score warm-up is a one-time tax). (2) **Microstructure is
nearly always present** — 99.0% micro-availability and **zero micro-starved coins**, so the +micro
cells get ~full coverage and the **shuffled-micro placebo comparison is not diluted** by missing
micro. (3) The **43 thin coins are thin by bar-count, not micro-quality** — almost all are recent 2024
listings (ZRO/TURBO/NOT/TAO/SAGA/ENA/… ~280k-500k bars, micro-ratio ~1.00); the lone genuinely
illiquid one is the delisted **FRONTUSDT** (stale 2.5%, zero-vol 2.6%). **Recommendation:** treat the
thin coins as a *sampling-weight* matter for the training draw (fewer bars ⇒ proportional weight), not
a drop-for-missing-micro matter; all 200 stay in the eval lake. Full per-symbol table:
`docs/_m4b_dq_table.md`.

**Spot-check (3 thinnest) — per-symbol masking fired:** ZROUSDT dropped 809/279,930 OHLCV bars to
warm-up/QA (0.3%) and masked 1,363 more micro bars (0.5%) to zero-vol/vre; TURBOUSDT 811 + 1,313;
NOTUSDT 809 + 1,479 — confirming warm-up and zero-vol/vre feature-masking apply **per symbol**.

**Lake fragmentation (flag for M6).** The lake is Hive-partitioned by `(symbol, frequency, date)` =
**per-day** → **211,595 parquet files** for 304 M bars. A single GROUP-BY pass is ~1-2 min, but the
manifest rollup's **200 sequential per-symbol queries** each re-glob all 211k files → ~40 min (why the
AXS-retry rollup was stopped and the Merkle taken from the ledger). M6 training data-loading pays the
same tax — **compacting to month-partitioned / coalesced Parquet is a recommended pre-M6 step.**

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
| Full universe lake (200 pairs, multi-year), Parquet+DuckDB, content-hashed manifest w/ listing+delisting + Merkle root + corpus-purpose | ✅ **MET** — 200/200 symbols, 304,625,181 bars, 23 GB Hive-Parquet/DuckDB lake; universe Merkle `5dfd667d…` (over the lake, from the ledger). Listing/delisting in `config/universe_full.yaml`; corpus-purpose in the manifest metadata. (Manifest coverage rollup at 199 — a fast follow; §4.1.) |
| Realized bar count in ~500M–1B band (flag if off) | **Flagged (realized 304.6 M):** below the 500 M–1 B band but **adequate by design** — past saturation for a 27 M model; this lake is regime×symbol breadth, not training fuel (§2/§5). |
| Causal exhaustive sweep GREEN on a real-universe cross-section | ✅ **GREEN at scale** — hard-gated in-line sweep fired on BOTH classes: `[live] HOOKUSDT`, `[tail] YFIIUSDT`, + `[live] AXSUSDT` on resume (each 100% cov, 6398 checks). A RED sweep raises `FatalIngestError` and aborts before recording (§4 / §7). |
| Peak disk, final lake size, raw retention outcome | ✅ peak raw cache **30 GB**, **503 GB** evicted → raw 0; final lake **23 GB**; free disk 628 GB (§4) |
| "universe Merkle root over the lake; raw checksums for provenance only" | ✅ printed + in manifest metadata |
| Tests green + ruff clean; evict-after-record adversarially reviewed | ✅ **159 tests green, ruff clean.** Evict-after-record verdict: **invariant holds, no violations** (single chokepoint, structural ordering, crash-safe across all kill-windows). A 14-agent verification workflow over the whole M4b surface **found + fixed 2 HIGH + 1 MED** (§7). |

**Honest bottom line.** The full 200-pair × 4-year universe lake is **complete**: 200/200 symbols,
304.6 M bars, both-class causal sweeps GREEN at scale, eviction/resume/governor all exercised on real
data, universe Merkle `5dfd667d…`. The run took **~52.8 h** at the home link's **~2.3 MB/s sustained**
(not the 8.3 MB/s burst), with one transient `TimeoutError` recovered by resume (`aborted=False`). Two
**fast-follow** loose ends, neither blocking the gate: regenerate the 200-symbol manifest (its rollup
is slow on the 211k-file day-partitioned lake — §4.1) and consider compacting the lake before M6
data-loading. **The M4 exit gate is MET; this close-out goes to the supervisor for the M6 entry-gate
check.**

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

The evict-after-record invariant itself was re-verified clean.

**Post-review hardening (supervisor gate-review → GO_WITH_FIXES).** Two correctness gaps on the
real-data path, both now closed:

- **Enforce + broaden the in-line causal sweep (invariant #2, BLOCKER).** The sweep previously only
  *printed* its result and ran once (on the first symbol, always a live one), so a RED sweep would
  not fail the run and the delisting-tail seam shared by all 41 delisted symbols was never swept.
  Now `run_sweep` is a **hard gate**: a RED — or degenerate `<100%`-coverage / `0`-check — sweep
  raises `FatalIngestError` *before* the lake write, so a leaking symbol is never written, recorded,
  or evicted; the orchestrator treats `FatalIngestError` as fatal (not an isolatable per-symbol
  fault) and aborts the whole run non-zero. It is **broadened** to fire once on a **live** symbol and
  once on a **delisting-tail** symbol (labelled by the same `delisting < window_end` rule the
  manifest uses), and the full run additionally requires both to have fired GREEN.
- **Orchestrator-path per-symbol independence test.** Independence was pinned only on the in-memory
  fan-out; a new test drives the **real** `IngestOrchestrator` (3 concurrent workers, tight evicting
  cap) with a finalize that runs the real `compute_features`, asserting each symbol's recorded
  `dataset_hash` is bit-identical to building it alone — the 1→N leak guard through the production
  concurrency + eviction interleaving.

All fixes are covered by new tests; **159 total green, ruff clean.**

## 8. Reproduce

```bash
# write the full config + plan, no download:
python scripts/m4b_universe_ingest.py --config-only --top-n 200 \
  --window-start 2021-01-01 --window-end 2025-01-01

# a bounded real run (what was executed here): smallest-footprint prefix, tight cap forces eviction:
python scripts/m4b_universe_ingest.py --max-symbols 12 --max-gb 10 \
  --raw-cache-cap-gb 3 --min-free-disk-gb 30

# the full run (resumable; ~15 h bandwidth-bound at ~8 MB/s):
python scripts/m4b_universe_ingest.py --top-n 200 \
  --window-start 2021-01-01 --window-end 2025-01-01 \
  --raw-cache-cap-gb 380 --min-free-disk-gb 50

# orchestrator + bootstrap unit gates:
pytest tests/data/test_ingest_orchestrator.py tests/data/test_universe.py -q
```
