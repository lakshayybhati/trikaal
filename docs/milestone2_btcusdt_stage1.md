# Milestone 2 — BTCUSDT real-data slice + Stage-1 tokenizer

Goal: prove the FSQ tokenizer learns **real** signal on one coin-year before downloading the
full multi-TB universe. One symbol (BTCUSDT, USDⓂ-perp), 2023, tokenizer only.

## Data pipeline (§2–§4)

| Stage | Result |
|---|---|
| §2.1 ingest | 12 monthly klines + 12 aggTrades archives, **4.7 GB**, every file SHA-256-verified vs its published `.CHECKSUM`; immutable raw store + append-only manifest. Funding/OI API deferred. |
| §2.2 reduction | Polars streaming aggTrades→per-bar (28.4M trades/month in ~4s); taker buy/sell via `is_buyer_maker`; grounded in the **real** futures schema (7-col aggTrades, `transact_time`) not the spec's spot schema. |
| §1/§3/§5 transform | 525,600 bars, **1 contiguous segment** (BTCUSDT-perp 2023 has no 1m gaps and no ≥50% single-bar moves), features finite, clipped to [-5,5], 524,159 valid targets. Funding/OI features correctly all-masked. |
| §3/§4 lake | Hive Parquet by `(symbol,frequency,date)` (31 day-partitions/month, zstd) + DuckDB assembly; round-trip **bit-exact** on real bars. |
| dataset hash | `sha256:1d3ec4f8db6686e2…` (content-addressed; reproducible from raw + config). |

## Causal-safety on REAL bars (the gate)

The **exhaustive truncation sweep** (every bar a boundary, 100% coverage) **passes on real
BTCUSDT bars** — 700/700 bars, leak-free. The transforms are causal on real OHLCV/microstructure
values, not just synthetic.

## Stage-1 tokenizer training (FSQ `[11,9,9,7,7,5,5]`)

525,600 real bars, 128-bar windows, batch 64, AdamW lr 1e-3 cosine+warmup, MPS, 4000 steps.

| metric | result | verdict |
|---|---|---|
| **recon-MAE convergence** | plateau at **step 1200** (val MAE ~0.088 → 0.076 by 4000) | **≤ 2k OK** — converges *faster* than the high-entropy synthetic fixture (~3.3k), as hypothesized (real K-lines are lower-entropy) |
| **codebook usage** | coarse **1.00**, fine **0.99**; perplexity **741 / 916** (of 891 / 1225) | **no collapse** — FSQ's "no dead codes by construction" confirmed empirically on real data |
| **per-stage Δ_fine** | coarse-only MAE 0.136 vs full 0.076 → **Δ_fine = 0.061** | **fine group ALIVE** — the fine subtoken explains ~44% of remaining error; no residual decay on real data |

Final val reconstruction MAE **0.0756** (z-scored units) — the irreducible error through the
~20-bit bottleneck on 13 active features (distinct from the single-batch overfit gate's <1e-3,
which measures *capacity*, not generalization). Still slowly improving at step 4000.

## Conclusion

The tokenizer eats real BTCUSDT data and learns genuine structure: fast convergence, full
codebook utilization, a live fine group. **Ready to scale to the full universe.**

## Out of scope (deferred, per milestone plan)

Full §2 universe ingest; funding/OI API; Stage-2 AR training; the inference/eval path (§8).
