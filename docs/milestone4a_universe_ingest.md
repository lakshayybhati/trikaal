# Milestone 4a — Universe-ingest scaffolding + local dry-run

**Status:** design + scaffold complete, validated on free local data. **No cloud, no paid compute,
no bulk downloads.** M4b (the at-scale cloud ingest) is a separate, gated step — see
[`milestone4b_runbook.md`](milestone4b_runbook.md).

**Goal.** Generalize the proven single-symbol M2 pipeline (ingest → reduce → causal feature
transform → Parquet/DuckDB lake) to the delisting-aware N-symbol universe, parametrized over one
config file, with **every causal and content-hash guarantee preserved**, so the cloud run is
turnkey and pre-costed.

---

## 1. Verdict

| Exit-gate criterion | Result |
|---|---|
| Generalized N-symbol pipeline parametrized over `config/universe.yaml` (schema+rule+sample) | ✅ `universe.py`, `universe_build.py`, `universe_ingest.py`, `manifest.py`, `reduce_shard.py` |
| Dry-run on BTCUSDT-2023 reproduces M2 `dataset_hash` **bit-exactly** | ✅ `sha256:1d3ec4f8…` (full hash in §4) |
| Multi-symbol path runs end-to-end + causal sweep leak-free | ✅ synthetic 2-symbol fixture (one delisted), exhaustive sweep 100% coverage |
| Per-symbol independence (the cross-symbol-leak risk) | ✅ a symbol's hash is identical alone vs in-universe |
| Parallel-reduction plan + M4b cost estimate (100 & 200 pairs) | ✅ §5, §6 |
| Adversarial review clean; tests green + ruff clean | ✅ §7; full suite + ruff in §8 |

**One honest caveat:** the *exact* listing/delisting dates and the *full* 100–200-pair list are
**populated at M4b from the data** (a first/last-kline probe + cumulative-volume ranking). M4a
ships the schema, the selection rule, a hand-verified sample, and the bootstrap that fills the
list — it never asserts delisting dates from memory (that would be a survivorship-leak risk).

---

## 2. The one thing that could have broken — and didn't

Generalizing 1 symbol → N has exactly one classic failure mode: **a cross-symbol or universe-wide
normalization statistic leaks** (e.g. fitting a global z-score, or pooling the large-trade `tau`
percentile across symbols). That silently destroys the causal guarantee the whole project rests on
(docs/ENGINEERING.md invariant 2).

The generalization is built so this is **structurally impossible, not just tested-against**:

- `compute_features(stream, cfg)` is called **once per symbol, on that symbol's own stream**, with
  zero shared mutable state. All normalization (causal EWMA/rolling z-score), volatility `σ_t`,
  and the `tau` percentile are computed *inside* that call, per segment, within the single stream.
- The universe layer (`build_universe_from_streams`) is a **pure fan-out**: it loops symbols in
  isolation and the only shared object is the read-only `FeatureConfig`. The frozen-stats table is
  per-`(symbol, feature)` by construction.
- This is pinned by two tests that would fail loudly on any cross-symbol coupling:
  - **per-symbol independence** — `dataset_hash(A)` is bit-identical whether `A` is built alone or
    inside `{A, B}` (`test_per_symbol_independence`);
  - **regression** — a single full-coverage symbol through the universe path equals the direct
    `compute_features` output bit-for-bit (`test_single_symbol_universe_equals_direct_transform`),
    and the real BTCUSDT-2023 build reproduces the M2 hash (§4).

Delisting is **half-open**: a symbol contributes bars `t` with
`listing_ms ≤ bar_open_ms(t) < delisting_ms`. Head-truncation (drop pre-listing bars) correctly
restarts that symbol's warm-up at its listing edge; a fully-live symbol (e.g. BTCUSDT/2023) is the
identity slice, which is *why* the M2 byte-stream — and hash — survive unchanged.

---

## 3. What was built (`src/trikaal/data/`)

| Module | Role |
|---|---|
| `universe.py` | `UniverseSpec`/`SymbolSpec`/`SelectionRule`, date↔ms, delisting-aware `active_window_ms`/`active_months`, the `cumulative_quote_volume` selection rule, and the M4b `bootstrap_universe` (pure function over measured volumes + probed dates — no network) |
| `universe_build.py` | the N-symbol build: `slice_stream_by_ms` (delisting-truncation), `build_one_symbol`, `build_universe_from_streams` (core fan-out + Merkle universe hash), `build_universe` (real path off the raw store) |
| `manifest.py` | delisting-aware `UniverseManifest`/`SymbolCoverage`: per-symbol coverage, OI-coverage start, dataset hash, raw-file checksums + a deterministic `manifest_content_hash` that **excludes wall-clock** provenance |
| `universe_ingest.py` | the **dry-run planner** `plan_universe_ingest` (enumerates every archive URL + byte estimate, **no network**), funding/OI hooks (§1.2 retention trap), `estimate_universe_cost`, and a **gated** `execute_plan` (M4b-only real fetch, reusing the proven `ingest_month`) |
| `reduce_shard.py` | resumable, content-hashed per-`(symbol,month)` shard reducer (`.npz` + `.shardhash.json` sidecar, cache-skip on resume) and `build_symbol_features_sharded` — byte-identical to the in-memory path, the parallel unit for M4b |

Plus `config/universe.yaml` (schema + rule + hand-verified sample), `scripts/m4a_universe_dryrun.py`
(the local validation runner), and the test suite in `tests/data/test_universe*.py`,
`test_manifest.py`, `test_reduce_shard.py`.

---

## 4. Regression — BTCUSDT-2023 reproduces M2 bit-exactly

The generalized build path (`build_universe` → `build_symbol_stream` → `compute_features`) run over
BTCUSDT-2023 alone:

```
generalized dataset_hash = sha256:1d3ec4f8db6686e20cd57749db4ca441c3e0d13c62a97c5f82eb0120bc9fa961
M2 recorded  dataset_hash = sha256:1d3ec4f8db6686e20cd57749db4ca441c3e0d13c62a97c5f82eb0120bc9fa961
BIT-EXACT MATCH: True   (525,600 bars, 1 segment)
```

This is the regression that proves the generalization did not perturb the single-symbol path. It is
a **one-time dry-run** (`scripts/m4a_universe_dryrun.py`, ~5 min — it rebuilds the 5 GB raw store
through the reducer); the fast CI suite proves the same property on the synthetic fixture in
milliseconds.

**Sharded equivalence + resumability** (BTCUSDT 2023-01): `build_symbol_features_sharded` reproduces
the in-memory `dataset_hash` bit-for-bit, and a second pass is served entirely from the content-hashed
shard cache (`status=cached`) — the crash-safe, resumable unit M4b parallelizes over.

---

## 5. Parallel aggTrades-reduction plan (M4b)

aggTrades is the heavy stream (≈ 1–3 GB/month/symbol compressed; klines are ~1.8 MB/month). The
reduction is **embarrassingly parallel over shards = one `(symbol, month)` each**:

- **Work-list:** `shard_plan(symbol → active_months)` → a flat, sorted `[(symbol, period)]` list,
  one task per shard (delisting-aware: a symbol only emits its live months).
- **Per shard:** `reduce_shard` runs the M2 Polars reducer (`reduce_aggtrades` + `build_raw_stream`),
  writes a lossless `.npz` (per-bar f64/i64 arrays; ragged trade-size pools as a values+offsets pair,
  **no pickle**) plus a `.shardhash.json` sidecar (content hash + bar count).
- **Crash-safety / resume:** a shard whose sidecar is present and consistent is **skipped**
  (`status=cached`). A run that dies mid-universe resumes without recomputing finished shards — the
  rule that stops a dead cloud job from burning money twice.
- **Storage layout:** `processed/shards/{SYMBOL}/{SYMBOL}-{YYYY-MM}.npz` (+ `.shardhash.json`).
  Features/lake build reads shards back, concatenates in time order, and runs the *unchanged*
  `compute_features` — byte-identical to the direct path (proven on real BTCUSDT in §4).
- **Throughput anchor (measured):** BTCUSDT 2023-01 = 389 MB aggTrades zip → bar features in
  **23.2 s on one core ≈ 16.8 MB compressed/s** (reduce + transform combined). Embarrassingly
  parallel ⇒ divide wall-clock by core count.

---

## 6. M4b cost estimate

Anchored on the measured BTCUSDT-2023 dumps (klines 1.8 MB/symbol-month; aggTrades 417 MB/symbol-month
for the #1-volume symbol) and a volume-tier decay (`major 400 / top20 70 / mid 22 / tail 9` MB/month).
Assumed window **24 months/symbol average** and a mix of **2 % major / 13 % top-20 / 85 % mid-cap**
(the long tail dominates a 100–200-pair USDT-perp universe). Numbers are deliberately rough (±50 %) —
they size the box, they do not bill to the cent.

| Universe | Download / raw-storage | Processed lake | Reduce CPU | Total bars |
|---|---|---|---|---|
| **100 pairs** | ≈ 90 GB | ≈ 9 GB | ≈ 1.4 core-hours | ≈ 107 M |
| **200 pairs** | ≈ 180 GB | ≈ 19 GB | ≈ 2.9 core-hours | ≈ 214 M |

**Rough $ range (one-time):** a cloud CPU box (8–16 vCPU, ~32–64 GB RAM) for ingest+reduce runs
hours, not days — **a few dollars of compute**. The real costs are **storage** (200–400 GB of raw
zips at ~$0.02/GB-month ⇒ single-digit $/month, droppable after the lake is built) and **egress** if
the lake leaves the cloud (the raw zips can stay; only the ~19 GB lake travels). Total one-time M4b
spend is comfortably **< $25** of cloud at 200 pairs, dominated by storage/egress, not CPU.

**Bottleneck:** download bandwidth + storage, **not** CPU — the reducer is fast relative to the
network. Provision for I/O, not cores.

**Saturation check (honest):** 200 pairs × 24 months ≈ **214 M bars** — an order of magnitude under
the ~1–2 B-bar saturation point, so more universe/window is still informative, not just slower. The
cost model raises a `WARN` only past ~1 B bars (e.g. 400 pairs × 60 months); the v1 universe does not
approach it. **Recommendation:** 150 pairs × the available window is well inside the useful regime.

---

## 7. Adversarial review — causal/normalization path

Two independent reviewers, same rigor that caught the M5 lookahead, with distinct lenses:

**Reviewer A (cross-symbol causal-leak hunt): no leak.** The N-symbol fan-out shares only a
read-only `FeatureConfig` (a frozen dataclass); there is no accumulator, cache, or statistic
computed across symbols. `compute_features` runs per-symbol on the symbol's own stream. Truncation
causality is correct: head-truncation renumbers segments so per-symbol warm-up restarts at the
listing edge (no pre-listing history imports into a symbol's stats); tail-truncation's funding/OI
event masking keeps only events `≤` the last kept bar's close. Pinned by `test_per_symbol_independence`.

**Reviewer B (regression byte-equivalence): PASS bit-exact, traced.** `months_between` half-open
end correctly yields exactly the 12 months of 2023 (no 2024-01); the active-window slice is a true
identity object for a fully-live symbol (`keep.all() → return stream`), so `compute_features` sees a
byte-identical input and `content_hash(x,m,ts)` matches M2. Flagged the single most dangerous future
edit: a refactor of `slice_stream_by_ms` that copies arrays even when `keep.all()` — already guarded
by `test_slice_is_identity_for_full_coverage` (asserts `result is stream`).

**One MED finding actioned — shard cache integrity.** Reviewer A noted the resume cache validated
only `(symbol, period)`, so a corrupt/half-written `.npz` carrying a valid sidecar could be trusted
(a crash-safety hole, not a causal leak). **Fixed:** `shard_cache_meta` now re-hashes the npz and
checks it against the sidecar's recorded `content_hash`, treating a torn/stale/tampered shard as a
cache miss → recompute (self-healing). Covered by four new tests (consistent shard accepted; stale
hash, torn npz, and absent sidecar all rejected). The sidecar remains the commit marker (npz written
first, sidecar second), so a killed mid-write leaves no false-positive cache entry.

**Net:** the causal/per-symbol-independence invariant the whole project rests on (and that the M5
placebo machinery relies on) is preserved structurally, not just by test. The real BTCUSDT-2023
`dataset_hash` regression (§4) is the empirical backstop.

---

## 8. Reproduce

```bash
# fast: synthetic multi-symbol smoke + ingest plan + cost estimate (no real data)
python scripts/m4a_universe_dryrun.py --skip-regression

# full: also rebuild BTCUSDT-2023 through the generalized path and assert the M2 hash (~5 min)
python scripts/m4a_universe_dryrun.py

# the fast CI gates
pytest tests/data/test_universe.py tests/data/test_universe_ingest.py \
       tests/data/test_manifest.py tests/data/test_reduce_shard.py -q
```

Everything is seed-pinned and content-hashed; the universe build is deterministic from
`config/universe.yaml` + the raw store. No numbers here are model-quality claims — M4a validates the
*machine*, not a result.
