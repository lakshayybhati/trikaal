# M4b runbook — full-universe cloud ingest (GATED)

**Do not run any of this until Lakshay has provisioned a cloud box and green-lit the spend.** M4b is
the one-time crossing into paid compute. M4a proved the whole machine on free local data; this
runbook makes M4b ≈ one command. Estimated one-time cost: **< $25** of cloud at 200 pairs, dominated
by storage/egress, not CPU (see `milestone4a_universe_ingest.md` §6).

## Preconditions (entry gate)

- [ ] Cloud CPU box provisioned (8–16 vCPU, 32–64 GB RAM, ≥ 400 GB disk for raw + lake).
- [ ] `pip install -e ".[data]"` on the box (polars, duckdb, pyarrow, certifi, pyyaml).
- [ ] `config/universe.yaml` window confirmed (default: 2023-01-01 → 2024-01-01; widen for M6 if
      the design calls for a multi-year window).
- [ ] Lakshay's explicit green-light to spend.

## Step 0 — bootstrap the universe list from the data (one-time)

The committed `config/universe.yaml` ships the schema, rule, and a *sample*. M4b populates the full
100–200-pair list from the data — ranked by cumulative quote volume, with listing/delisting dates
resolved from a first/last-kline probe. This is the only step that needs exchange metadata.

```bash
# (to be run on the box) probe exchangeInfo + monthly kline index → measured volumes + dates,
# then apply the committed selection rule:
python - <<'PY'
from trikaal.data.universe import load_universe, bootstrap_universe
# measured at M4b: {symbol: Σ quote_volume}, {symbol: SymbolSpec(listing/delisting from kline probe)}
volumes, metadata = ...   # filled by the kline/exchangeInfo probe (no model code, pure data)
base = load_universe()
full = bootstrap_universe(base, cumulative_quote_volume=volumes, metadata=metadata)
# persist `full` back to config/universe_full.yaml for the run
PY
```

`bootstrap_universe` applies `top_n`, `min_live_*`, `include_delisted`, and the
`cumulative_quote_volume` ranking — survivorship-owned (delisted symbols kept for their live span).

## Step 1 — plan (dry-run, no spend) — sanity-check before fetching

```bash
python - <<'PY'
from trikaal.data.universe import load_universe
from trikaal.data.universe_ingest import plan_universe_ingest
spec = load_universe("config/universe_full.yaml")
plan = plan_universe_ingest(spec)
print(plan.n_symbols, "symbols,", plan.n_files, "files, est", round(plan.est_total_gb,1), "GB")
print("funding:", len(plan.funding_symbols), " OI-masked:", len(plan.oi_zero_filled_symbols))
PY
```

## Step 2 — ingest (the gated fetch) — klines + aggTrades, SHA-256-verified, resumable

```bash
python - <<'PY'
from trikaal.data.universe import load_universe
from trikaal.data.universe_ingest import plan_universe_ingest, execute_plan
spec = load_universe("config/universe_full.yaml")
plan = plan_universe_ingest(spec)
execute_plan(plan, run_id="universe_m4b")   # reuses ingest_month: idempotent, checksum-verified
PY
```

`execute_plan` reuses the M2 single-symbol `ingest_month`: every file is SHA-256-verified against
its published `.CHECKSUM` (mismatch hard-fails, never a silent skip), written immutably, and a
verified file already present is skipped (resume-safe). **Binance data is not redistributed** — it
stays in the cloud raw store.

## Step 3 — parallel reduce → lake (resumable shards)

The reduction is parallel over `(symbol, month)` shards. On a multi-core box, fan `reduce_shard`
across the work-list (e.g. with `xargs -P`, GNU `parallel`, or a process pool):

```bash
python - <<'PY'
from trikaal.data.universe import load_universe
from trikaal.data.reduce_shard import shard_plan
spec = load_universe("config/universe_full.yaml")
work = shard_plan({s.symbol: s.active_months(spec.window_start, spec.window_end)
                   for s in spec.active_symbols()})
# write the work-list; each line is one resumable, content-hashed shard task
open("shards.todo","w").write("\n".join(f"{sym} {per}" for sym,per in work))
print(len(work), "shards")
PY

# fan out across cores; a killed run resumes (cached shards are skipped):
cat shards.todo | xargs -P "$(nproc)" -L1 \
  python -c 'import sys; from trikaal.data.reduce_shard import reduce_shard; \
             reduce_shard(sys.argv[1], sys.argv[2])'
```

Then build the per-symbol lake + manifest (per-symbol independent transform; writes one Parquet
partition per `(symbol, frequency, date)`):

```bash
python - <<'PY'
from trikaal.data.universe import load_universe
from trikaal.data.universe_build import build_universe
from trikaal.data.manifest import build_manifest
spec = load_universe("config/universe_full.yaml")
res = build_universe(spec)                      # reads raw store, writes processed/bars/...
man = build_manifest(spec, res, built_at_utc="<stamp>")
man.write("processed/universe_manifest.json")
print("universe_hash", res.universe_hash, "| total_bars", f"{res.total_bars:,}")
PY
```

## Step 4 — verify before declaring M4b done

- [ ] `manifest.manifest_content_hash()` recorded; `universe_hash` recorded.
- [ ] **Causal sweep on a universe cross-section:** run `exhaustive_truncation_sweep` on a sample of
      symbols (≥ majors + ≥ 1 delisted-truncated) → 100 % coverage, leak-free.
- [ ] Per-symbol bar counts match the manifest; delisted symbols truncate at their delisting date.
- [ ] BTCUSDT partition `dataset_hash` still `sha256:1d3ec4f8…` (the regression survives at scale).

## Funding / OI (§1.2)

- **Funding** — full public history via the futures API (`/fundingRate`, paginated, free); planned
  for every perp in `plan.funding_symbols`.
- **Open interest** — the **retention trap**: Binance `/futures/data/openInterestHist` retains only
  ~30 days, so historically OI is **zero-filled and masked** (the OI mask bits are set automatically
  in `compute_features` when the OI event series is empty). `plan.oi_zero_filled_symbols` lists every
  symbol this applies to. Do not mistake a historical OI zero for a real value.

## After M4b → M6

With the universe lake built and manifest content-hashed, the M5 eval harness runs the 5-cell
ablation (M6). M6's §8.C.3 setup gate still stands: a universe-trained **Cell 1 (our BSQ baseline)**
within ~10–15 % RankIC of published Kronos-small before the 2×2 + placebo is trusted. All intensive
training is cloud CUDA (A100/H100), bf16 — not MPS.
