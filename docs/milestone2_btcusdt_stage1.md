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

## Add-on — per-feature causal IC screen + first-cut statistical power

M2 above proved the tokenizer *reconstructs* real microstructure (compression). It did **not**
show microstructure *predicts* forward returns. This screen is that missing half — and the
cheapest possible early read on whether the +micro leg can survive the full eval (spec §8.E.8
mandates a first cut here). It is **read-only on the lake, CPU/local, reported not gated**, and
independent of any training run. Reproduce: `python3 scripts/m2_feature_ic.py 1000 0`.

**Method.** Forward-return label `y_{t,h} = log(C_{t+h}/C_{t+1})` (entry next-bar, matches eval
§8.B), reconstructed within-segment from the lake's stored causal `raw_ret_close` (the *only*
forward-reading object; the features are read straight from the lake and the reload is asserted
to reproduce the frozen M2 `dataset_hash`, so there is no recompute that could peek). RankIC =
Spearman (primary), Pearson IC secondary; 95% CI = moving-block bootstrap (block = max(h, 1440
bars = 1 day), 1000 resamples, seed 0). A feature **carries standalone signal** only if its
RankIC CI excludes 0. (*h=1 convention:* the §8.B entry-next-bar return is identically 0 at h=1,
so the 1-bar column reports the decision-relative `log(C_{t+1}/C_t)` — the model's 1-step target;
h∈{5,15,60} use the §8.B return.)

### RankIC ± 95% CI (★ = CI excludes 0)

| feature (dim) | h=1 | h=5 | h=15 | h=60 | verdict |
|---|---|---|---|---|---|
| `ret_close` (0) | −0.0163 [−0.0204,−0.0127]★ | −0.0335 [−0.0369,−0.0303]★ | −0.0220 [−0.0245,−0.0193]★ | −0.0154 [−0.0178,−0.0125]★ | carries-signal |
| `range` (1) | +0.0024 [−0.0002,+0.0052] | +0.0050 [+0.0013,+0.0095]★ | +0.0071 [+0.0012,+0.0133]★ | +0.0043 [−0.0074,+0.0161] | carries-signal |
| `body` (2) | −0.0053 [−0.0095,−0.0014]★ | −0.0258 [−0.0290,−0.0227]★ | −0.0155 [−0.0179,−0.0132]★ | −0.0115 [−0.0140,−0.0087]★ | carries-signal |
| `upper_wick_frac` (3) | −0.0076 [−0.0102,−0.0047]★ | −0.0057 [−0.0087,−0.0031]★ | −0.0041 [−0.0075,−0.0005]★ | −0.0019 [−0.0071,+0.0036] | carries-signal |
| `lower_wick_frac` (4) | +0.0030 [−0.0004,+0.0065] | +0.0102 [+0.0073,+0.0134]★ | +0.0072 [+0.0038,+0.0107]★ | +0.0073 [+0.0024,+0.0122]★ | carries-signal |
| `log_volume` (5) | +0.0024 [−0.0003,+0.0054] | +0.0047 [+0.0006,+0.0091]★ | +0.0076 [+0.0006,+0.0152]★ | +0.0073 [−0.0053,+0.0202] | carries-signal |
| `log_amount` (6) | +0.0026 [−0.0001,+0.0055] | +0.0050 [+0.0007,+0.0098]★ | +0.0078 [+0.0005,+0.0155]★ | +0.0075 [−0.0058,+0.0198] | carries-signal |
| **`TFI` (7)** ·micro | −0.0124 [−0.0168,−0.0082]★ | **−0.0270 [−0.0304,−0.0236]★** | −0.0197 [−0.0230,−0.0166]★ | −0.0137 [−0.0166,−0.0106]★ | **carries-signal** |
| **`signed_count_imbalance` (8)** ·micro | −0.0039 [−0.0079,−0.0004]★ | **−0.0197 [−0.0235,−0.0163]★** | −0.0088 [−0.0128,−0.0050]★ | −0.0035 [−0.0090,+0.0014] | **carries-signal** |
| `trade_count` (9) ·micro | +0.0008 [−0.0019,+0.0033] | +0.0042 [−0.0001,+0.0088] | +0.0059 [−0.0018,+0.0128] | +0.0019 [−0.0125,+0.0150] | live-no-signal |
| `mean_trade_size` (10) ·micro | +0.0027 [+0.0001,+0.0054]★ | +0.0004 [−0.0032,+0.0038] | +0.0027 [−0.0022,+0.0079] | −0.0003 [−0.0082,+0.0071] | carries-signal (h=1 only) |
| `trade_size_dispersion` (11) ·micro | −0.0003 [−0.0030,+0.0024] | +0.0008 [−0.0020,+0.0038] | +0.0007 [−0.0027,+0.0042] | −0.0012 [−0.0055,+0.0038] | live-no-signal |
| `large_trade_share` (12) ·micro | −0.0002 [−0.0029,+0.0022] | −0.0004 [−0.0056,+0.0038] | −0.0020 [−0.0097,+0.0058] | −0.0043 [−0.0183,+0.0088] | live-no-signal |

Pearson IC (secondary) tracks the same signs but is smaller in magnitude on the dominant
features (e.g. `ret_close` Pearson +0.001/−0.011/−0.003/−0.004 vs RankIC ~−0.02 to −0.03),
i.e. the predictive content is largely **monotone-but-nonlinear** — exactly the regime RankIC is
meant to catch and a point in favor of a quantizer-based model. Funding/OI (`funding_rate` 13,
`log_oi` 14, `d_oi` 15) are **N/A in this build** — all bars masked (mask rate 1.000); screen
them when the futures-API ingest lands.

### Which of the 6 aggTrades channels predict vs are inert

- **Predict (standalone):** `TFI` (peak |RankIC| 0.027 @ h=5) and `signed_count_imbalance`
  (0.020 @ h=5) — the two **signed order-flow** channels — clear the noise floor at every short
  horizon. This is the encouraging early read: the spec's foreground microstructure feature
  (TFI, the "method foundation") has real, sign-stable standalone predictive content.
- **Largely inert (standalone):** `trade_count`, `trade_size_dispersion`, `large_trade_share`
  are live (non-constant) but their CIs straddle 0 at every horizon; `mean_trade_size` is
  significant only marginally at h=1 (0.0027). The **magnitude/size** channels carry little
  univariate signal on this one coin-year — consistent with their value being conditional
  (regime/interaction) rather than standalone.
- **Direction:** the signed-flow and `ret_close`/`body` ICs are **negative** at 1–60 min →
  short-horizon mean-reversion, the textbook 1m-crypto pattern; the model has real structure to
  learn, not noise.

### First-cut MDE / effective-N (single-symbol, temporal only)

- Per-bar forward-return autocorrelation barely deflates N: **N = 525,599**, **N_eff ≈ 525,599**
  (ratio 1.000 — 1m returns are near-white; net autocorrelation over the 4 significant lags is
  ≤ 0, so N_eff is conservatively capped at N).
- **SE(RankIC) ≈ 1/√N_eff = 0.0014** → single-feature MDE@95% = **0.0027**; two-cell IC-gap MDE
  (e.g. Cell 4 − Cell 1) = **0.0038**. Peak live-micro |RankIC| **0.027** sits ~7× above the
  two-cell MDE → not underpowered on this temporal single-symbol cut (necessary, not sufficient).
- *Caveat — this MDE is optimistic.* It uses only **return** autocorrelation (per the §8.E.8
  formula); it ignores **feature** autocorrelation, which is large for 1m microstructure. The
  binding per-feature significance is therefore the **block-bootstrap CI**, several of which are
  much wider than ±0.0027 (e.g. `range`/`log_volume` at h=60 span ±0.012) — and those are what
  the verdicts use.
- *Scope.* First cut on one coin-year, TEMPORAL effective-N only. The full **portfolio** MDE
  (cross-sectional breadth across ~one-factor symbols and K=6 forward blocks, §8.E.8) is computed
  at eval time and will be larger/harder.

### What this does and does NOT establish

This is a **univariate liveness screen, not the headline test.** A standalone RankIC for a
microstructure channel can be confounded with the OHLCV signal (TFI co-moves with `ret_close`,
which itself mean-reverts), so "TFI carries signal" does **not** prove microstructure adds
*information beyond OHLCV*. That incremental claim is exactly what the 2×2 marginal (Cell 4 − Cell
2) and the **shuffled-microstructure placebo (Cell 5, §8.C.4)** are designed to test, under the
Deflated-Sharpe discount. Read this screen as: the +micro leg is **worth carrying into the full
ablation** (its strongest channels are alive and above the temporal noise floor), not as evidence
the leg has already won.

Provenance: `dataset_hash = sha256:1d3ec4f8db6686e20cd57749db4ca441c3e0d13c62a97c5f82eb0120bc9fa961`
(stored x/m/ts — matches M2 ✓); `ic_results_hash = sha256:2952bc4e07d9e0c494723b2…`
(seed 0, 1000 resamples; reproducible).

## Out of scope (deferred, per milestone plan)

Full §2 universe ingest; funding/OI API; Stage-2 AR training; the inference/eval path (§8).
