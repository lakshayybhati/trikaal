# Trikaal — v1 Research Blueprint
### A microstructure-aware FSQ tokenizer for crypto K-line foundation models

| field | value |
|---|---|
| Status | Design — spine approved; **pending final user review** |
| Date | 2026-06-18 |
| Scope | v1 — single paper, single model |
| Parent work | Kronos (Shi et al. 2025, arXiv:2508.02739) |
| Model class | decoder-only, Kronos_small class (**~21.30M** realized params) |
| Data | crypto-only, Binance spot + USDT-perps, 1-minute spine, ~500M–1B bars |

## Executive summary

Trikaal is a from-scratch ~27M-class decoder-only foundation model for the *language of crypto 1-minute K-lines*, released as code + weights (HuggingFace) + a paper + a live demo. It is a **controlled** evolution of Kronos with exactly **one** headline research contribution:

> **A microstructure-aware FSQ tokenizer for financial K-lines** — two mechanically-independent legs, one claim: (1) **Finite Scalar Quantization (FSQ)** replaces Kronos's Binary Spherical Quantization (no auxiliary commitment loss, no codebook-collapse failure mode, bounded quantization error); (2) the per-bar input vector carries **free microstructure** — trade-flow imbalance (TFI), funding, open interest — beyond OHLCV.

Everything else — DeepSeek-V3-style **MTP** heads for multi-horizon output, **volatility-scaled** causal targets, and a first-class **evaluation harness** — is production engineering or inherited design, explicitly **secondary** to the single claim.

### The proof: a 2×2 ablation at matched bits-per-token (~20.06 bits)

| | OHLCV-only (F = 7) | +Microstructure (F = 16) |
|---|---|---|
| **BSQ** | Cell 1 — Kronos_small reproduction (validated vs public weights) | Cell 3 — isolates the microstructure leg |
| **FSQ** | Cell 2 — isolates the FSQ leg | Cell 4 — **Trikaal** (full model) |

Both legs must visibly earn their place across tokenizer-level metrics (reconstruction MAE/MSE, codebook usage/perplexity, collapse rate) **and** every downstream task — or it is reported honestly.

### Headline metric

Cost-aware **net Information Ratio** computed on a **portfolio return series**, under a forecast-magnitude execution filter (θ = κ·c_total, κ > 1, tuned on validation folds), with a 0.1–0.3% transaction-cost model that **includes perpetual funding**, evaluated on **purged walk-forward + leading-embargo** splits across held-out symbols and a held-out future window. IC / RankIC / MAE / R² / discriminative-score / TSTR / pinball / coverage are **secondary diagnostics**.

### Canonical FSQ configuration

`fsq_levels = [11, 9, 9, 7, 7, 5, 5]` (D = 7 dims, all **odd**) → **20.06 bits/token**, matching BSQ k = 20 within 0.06 bit. Coarse group {11, 9, 9} → vocab **891**; fine group {7, 7, 5, 5} → vocab **1225**. Bits-per-token — not an integer vocab — is the controlled variable of the ablation.

### Non-negotiable invariants

- **TFI, never OFI** — imbalance is computed from free aggTrades (signed executed-volume); true order-book OFI is v2.
- **Strict causal-safety** — no transform (feature, normalization, vol-scale, data-quality gate, or target) may read data with effective timestamp > t+1; enforced as a transform-agnostic CI gate.
- **One headline claim** — MTP, vol-scaling, and the eval harness are secondary, not competing contributions.
- **Bits-per-token is the control** — never a parameter-padding lever; the realized ~21.30M is reported honestly.
- **Determinism is a deliverable** — one config + pinned seed reproduces any run; every dataset is content-hashed.

### Scope

**In (v1):** crypto-only Binance (spot + USDT-perps), 1m spine (5m/15m optional), ~500M–1B bars, the tokenizer + backbone + MTP + data pipeline + eval harness + release.
**Out (firewalled to v2/v3):** equities / cross-asset, Bybit/OKX, order-book depth (→ true OFI), base-class (~100M) scale, MLA attention, architecture-level latency micro-optimization, regime-conditioning, and the meta-labeling / bet-sizing layer. See §9 roadmap.

### Document provenance

Drafted and adversarially verified by a multi-agent workflow: 9 subsystem drafts → a 4-lens audit (causal-safety, architecture/math, financial-ML rigor, scope/consistency) → targeted revision against a locked reconciliation ledger → re-verification. The audit caught and corrected real defects (wrong FSQ bits-per-token tables, a bad-tick lookahead, a mis-scaled net-IR annualization, a missing funding-cost term, a break-even execution filter, an inert trailing embargo). Two re-verifier residuals (unifying the param base to 21.30M; a 0.07M typo) were patched during assembly.

## Contents

1. Per-Bar Feature Specification & Causal Normalization
2. Data Pipeline (ingest → Parquet → DuckDB → dataset)
3. Volatility-Scaled, Regime-Relative Targets
4. Microstructure-Aware FSQ Tokenizer
5. Autoregressive Backbone (decoder-only, Kronos_small class)
6. Multi-Token Prediction (MTP) Heads
7. Training Plan (two-stage)
8. Evaluation Harness (first-class subsystem)
9. Repository, Release & Future-Work Roadmap

---


---

## 1. Per-Bar Feature Specification and Causal Normalization

This section defines the canonical transform from raw exchange data to the model-ready 16-dimensional bar vector. It is the contract every downstream subsystem (tokenizer, AR backbone, eval harness) keys off. **All times are bar-*close* times.** A 1-minute bar indexed `t` covers the half-open interval `[t, t+1)` (open timestamp `t`, in milliseconds `[t_ms, t_ms + 60000)`); its features become known and usable only at the **close** of that interval, i.e. at wall-clock time `t+1`. The model that consumes the token for bar `t` therefore makes its first prediction for bar `t+1` onward — no feature of bar `t` may depend on any datum whose **effective timestamp is `> t+1`**.

### 1. The 16-feature table

Notation: `O,H,L,C` = open/high/low/close price; `V` = base-asset volume; `QuoteTurnover` = quote-asset volume (USDT); aggTrades-derived buy/sell splits defined in §2. `eps_*` constants in §5. `t-1` denotes the immediately preceding **contiguous** bar (see §4 segment rule: across a segment boundary, `t-1` does not exist and the bar is dropped from windowed/diff features per the warm-up rule).

| # | Feature | Exact formula | Source | Known-at-close-`t` lookahead rule |
|---|---------|---------------|--------|-----------------------------------|
| 1 | `ret_close` | `log(C_t / C_{t-1})` | klines | Uses only closes of `t` and prior bar `t-1`. `C_t` is final at close of `[t,t+1)`. **Raw feature; vol-scaling hook in §7 may transform the *target* derived from it, never the input here.** |
| 2 | `range` | `log(H_t / L_t)` | klines | `H_t,L_t` are extrema over `[t,t+1)`, final at close. No future data. |
| 3 | `body` | `(C_t - O_t) / (H_t - L_t + eps_px)` | klines | All four OHLC of bar `t` only. Bounded in `[-1,1]`. |
| 4 | `upper_wick_frac` | `(H_t - max(O_t,C_t)) / (H_t - L_t + eps_px)` | klines | Bar `t` OHLC only. Bounded `[0,1]`. |
| 5 | `lower_wick_frac` | `(min(O_t,C_t) - L_t) / (H_t - L_t + eps_px)` | klines | Bar `t` OHLC only. Bounded `[0,1]`. |
| 6 | `log_volume` | `log(V_t + 1)` | klines | `V_t` = total base-asset volume traded in `[t,t+1)`, final at close. |
| 7 | `log_amount` | `log(QuoteTurnover_t + 1)` | klines | `QuoteTurnover_t` over `[t,t+1)`, final at close. |
| 8 | `TFI` | `(V_buy - V_sell) / (V_buy + V_sell + eps_cnt)` | aggTrades | Taker buy/sell base-volume split over trades with `T in [t,t+1)` (§2). Bounded `[-1,1]`. **TFI = trade-flow imbalance, NOT orderbook OFI.** |
| 9 | `signed_count_imbalance` | `(N_buy - N_sell) / (N_buy + N_sell + eps_cnt)` | aggTrades | Taker buy/sell trade *counts* over `[t,t+1)`. Bounded `[-1,1]`. |
| 10 | `trade_count` | `log(N_trades + 1)` | aggTrades | Total aggTrade rows with `T in [t,t+1)`. |
| 11 | `mean_trade_size` | `log( V_agg_t / (N_trades + 1) + eps_sz )` | aggTrades | Within-bar mean; uses only bar-`t` trades. Uses aggTrades-summed base volume `V_agg_t` for source consistency (§2.2 reconciliation note). |
| 12 | `trade_size_dispersion` | `sigma(sizes_t) / (mean(sizes_t) + eps_sz)` (coeff. of variation over per-trade base sizes in `[t,t+1)`) | aggTrades | Computed only from bar-`t` trades. Use population std (ddof=0). Single-trade or zero-trade bar -> 0 (see §4). |
| 13 | `large_trade_share` | `(sum of base volume from trades with size >= tau_{t}) / (V_agg_t + eps_sz)` | aggTrades | **`tau_t` is the rolling-past percentile (§2.3), computed from trades in bars `[t-W_tau, t-1]` ONLY — never bar `t`'s own size distribution.** Bounded `[0,1]`. |
| 14 | `funding_rate` | last settled/predicted funding value with settle-or-known effective timestamp `<= t+1` (causal forward-fill) | futures API | Perp funding settles every 8h; carry the **last value whose effective timestamp `<= t+1`**. Spot -> 0 + missing-flag (§4). Already in natural small-magnitude units; no z-score (§3). |
| 15 | `log_oi` | `log(OI_t + 1)` | futures API | `OI_t` = open-interest snapshot whose API effective timestamp is the **latest `<= t+1`** (causal forward-fill; OI polled at coarser cadence). Spot -> 0 + flag. **The `+1` is the same stability guard used by `log_volume`/`log_amount` (see §5 note); it deviates from the canonical bare `log(OI_t)` deliberately and is the single source of truth for OI epsilon handling, kept consistent with `d_oi`'s `eps_oi=1.0` (§5).** |
| 16 | `d_oi` | `log( (OI_t + eps_oi) / (OI_{t-1} + eps_oi) )` | futures API | Log-change of the causally forward-filled OI series. Spot -> 0 + flag. Across segment boundary -> 0 (warm-up). |

**Calendar features are NOT here.** Minute/hour/day-of-week/day-of-month/month-of-year enter as learnable temporal *embeddings* inside the AR backbone (per Kronos), keyed off the bar-open timestamp `t`. They are deterministic functions of `t` and trivially causal; the feature pipeline emits the raw open-timestamp alongside the 16-vector for the backbone to embed.

**OHLCV-only ablation subset (Kronos-equivalent).** The OHLCV-only arm used by the Tokenizer and Eval ablations is the `F = 7` subset = the **5 price/shape features (`ret_close`, `range`, `body`, `upper_wick_frac`, `lower_wick_frac`) + 2 volume/liquidity features (`log_volume`, `log_amount`)**. The full vector is `F = 16`. This 5+2 = 7 split is the canonical Kronos-equivalent subset referenced identically by the Tokenizer and Eval sections.

**Output of this stage per bar:** a `float32` vector `x_t in R^16` (post-normalization, §3) plus a `uint8` flag vector `m_t in {0,1}^16` (1 = field was imputed/missing — see §4) plus the `int64` open-timestamp. Batched shape into the tokenizer: `x in R^{B x L x 16}`, `mask in {0,1}^{B x L x 16}`, `ts in Z^{B x L}` with `L <= 512`. **`m_t` is the ONLY model-visible missingness signal; the QA-only `is_stale` flag and the packed `dq_flags` bitfield (§4.5) are never part of this output and never fed to the model.**

---

### 2. aggTrades → bar reduction

Binance `aggTrades` (CSV dumps `data.binance.vision`, schema: `aggTradeId, price, quantity, firstTradeId, lastTradeId, timestamp, isBuyerMaker, isBestMatch`) are aggregated trades. We reduce them to per-bar microstructure.

#### 2.1 Buy/sell classification (taker-side, deterministic — no tick rule)

The `isBuyerMaker` flag is exact taker-side labeling and is preferred over any heuristic (Lee–Ready tick rule) because it is ground truth:

- `isBuyerMaker == False` → the buyer is the **taker** (aggressive buy, lifted the ask) → **BUY**.
- `isBuyerMaker == True` → the seller is the taker (aggressive sell, hit the bid) → **SELL**.

Per bar `t` (trades with `timestamp T in [t_ms, t_ms+60000)`; left-closed, right-open — a trade at exactly `t+1` belongs to bar `t+1`):

```
V_buy   = sum(quantity  where isBuyerMaker == False)
V_sell  = sum(quantity  where isBuyerMaker == True)
N_buy   = count(rows     where isBuyerMaker == False)
N_sell  = count(rows     where isBuyerMaker == True)
V_agg   = V_buy + V_sell                      # base volume reconstructed from trades
N_trades= N_buy + N_sell                      # aggregated-trade row count
sizes_t = [quantity_i for trades i in bar t]  # per-aggTrade base sizes
```

**Causal rule:** the bucket assignment depends only on each trade's own `timestamp`; no trade outside `[t,t+1)` contributes. Bucketing is exact integer-millisecond arithmetic (`floor((T - epoch) / 60000)`), never floating binning.

#### 2.2 Reconciliation with klines `V_t`

`V_agg` (summed from aggTrades) and `V_t` (kline base volume) can differ slightly (aggTrades excludes nothing material for spot/perp, but rounding and the rare missing-chunk happen). Rule: **features 8–13 and `mean_trade_size`/`large_trade_share` use `V_agg` internally** so numerator/denominator come from the same source and ratios stay in `[0,1]`/`[-1,1]`. Features 6 (`log_volume`) and 7 (`log_amount`) use the **kline** `V_t`/`QuoteTurnover_t` (the canonical liquidity figure). Emit a per-bar QA scalar `vol_recon_err = |V_agg - V_t| / (V_t + eps_sz)`; if `> 0.01` for a bar, set that bar's aggTrades-derived flags (`m_t` bits 8–13) and route to the data-quality report. The `vol_recon_err` scalar is a **QA-only** signal, **not** used as a model input. (The `vol_recon_err > 0.01` condition is a strictly-causal per-bar comparison of bar-`t` quantities only, so setting `m_t` bits 8–13 from it introduces no lookahead — see §4.5.)

#### 2.3 Causal `tau` percentile for `large_trade_share`

`tau_t` is the `q`-th percentile of **individual aggTrade base sizes** observed over a **rolling past window** of `W_tau` bars ending at `t-1`:

```
pool_t   = { quantity_i : trade i in bars [t - W_tau, t - 1] }   # STRICTLY past bars
tau_t    = percentile(pool_t, q)                                  # q-th pct of past trade sizes
large_trade_share_t = sum( quantity_i for i in bar t if quantity_i >= tau_t ) / (V_agg_t + eps_sz)
```

- **Lookahead rule (critical):** `pool_t` excludes bar `t` itself. `tau_t` is "what counts as a large trade, learned from recent history," then applied to the *current* bar's trades. Using bar `t`'s own distribution to set `tau` would leak the current bar's realized activity into its own feature → forbidden.
- **Defaults & tuning (feasibility-reconciled):** `W_tau = 1440` bars (24h of 1m) as the rolling window. **Default `q = 90`.** The earlier `q = 99` default is removed because it is **infeasible against the saturation criterion**: at the 99th percentile of past trade sizes, by construction almost every current bar contains *no* trade exceeding the threshold, so `large_trade_share = 0` for the large majority of bars — far more than 1% — which can never satisfy a target median in `[0.05, 0.40]` with `< 1%` zeros. Tune `q in {80, 90, 95}` (a non-trivial fraction of bars then contains an above-threshold trade) and `W_tau in {720, 1440, 2880}` (12h/24h/48h). **Feasible selection criterion (state one, verify on a data slice before locking `q`):** choose `(q, W_tau)` so that **5–40% of bars are exactly `0`, the median of the NON-ZERO bars lies in `[0.1, 0.5]`, and `< 1%` of bars saturate at exactly `1`**. This criterion is achievable given the rolling-past-percentile definition; the previous "median bar value in `[0.05,0.40]` AND `< 1%` zeros" criterion was incompatible with any high percentile and is discarded. For implementation efficiency, maintain a per-symbol rolling reservoir/quantile sketch (e.g. a t-digest) updated bar-by-bar; the sketch state at the close of `t-1` produces `tau_t`. Persist the sketch with the dataset hash for replay.
- **Warm-up:** until `W_tau` past bars are accumulated within the current contiguous segment, `large_trade_share` is undefined → emit `0` and set flag bit 13 (treated like missing; see §4 warm-up handling). Do **not** backfill across a segment boundary.

---

### 3. Causal normalization

Two classes of features get different treatment.

#### 3.1 Already-bounded / relative features — NO z-score
`body`, `upper_wick_frac`, `lower_wick_frac`, `TFI`, `signed_count_imbalance`, `funding_rate` are already on stable, bounded, cross-symbol-comparable scales (`[-1,1]`, `[0,1]`, or small natural funding units). They pass through **un-z-scored**, only clipped to `[-5,5]` defensively (funding spikes, degenerate `eps` bars). `d_oi` and `ret_close`/`range` are log-ratios — already mean-near-zero and roughly scale-free; we **do** z-score them (below) because their *dispersion* varies wildly by symbol/regime and the tokenizer benefits from a unit-variance input.

#### 3.2 Unbounded / scale-varying features — causal z-score
`ret_close`, `range`, `log_volume`, `log_amount`, `trade_count`, `mean_trade_size`, `trade_size_dispersion`, `log_oi`, `d_oi` are z-scored per-symbol using **only past bars**. Two interchangeable estimators (config-selected); EWMA is the default for inference cheapness and graceful regime adaptation:

**(a) EWMA (default).** Maintain causal running mean/variance with half-life `H` bars:
```
alpha   = 1 - 2^(-1/H)                          # EWMA decay from half-life H
# update AFTER emitting bar t's normalized value, using bar t's RAW value f_t,
# so the stats used to normalize f_t depend only on bars < t:
mu_t    = mu_{t-1}                              # stat as of close of t-1
var_t   = var_{t-1}
z_t     = clip( (f_t - mu_t) / sqrt(var_t + eps_var), -5, 5 )
# then advance state with f_t:
delta       = f_t - mu_{t-1}
mu_{t}_new  = mu_{t-1} + alpha * delta
var_{t}_new = (1 - alpha) * (var_{t-1} + alpha * delta^2)
```
This is the **West/EWMA incremental variance** form; it is strictly causal because `f_t` is normalized with `(mu_{t-1}, var_{t-1})` and only *then* folded into the running stats.

**(b) Rolling window (alternative / audit reference).** Simple `z_t = (f_t - mean(f_{t-W..t-1})) / (std(f_{t-W..t-1}) + eps_var)` over a trailing window `W` of **past** bars (excludes `t`). Used as the reproducible ground-truth reference in tests and for the published "frozen-stats" inference mode.

- **Defaults & tuning:** EWMA half-life `H = 1440` bars (24h) for `ret_close`, `range`, `trade_count`, `mean_trade_size`, `trade_size_dispersion`; `H = 5760` bars (4 days) for the slower-moving `log_volume`, `log_amount`, `log_oi`, `d_oi`. Tune `H in {720, 1440, 5760, 11520}` per feature group; selection criterion: post-normalization features should have rolling empirical std in `[0.8, 1.25]` and `< 0.5%` of values hitting the `±5` clip on a held-out month. Rolling-window alternative: `W in {1440, 5760}`.
- **Clip:** every normalized feature is clipped to `[-5, 5]` (Kronos-style outlier guard). Track clip-hit rate per feature; a feature clipping `> 1%` of the time signals a too-short half-life or a heavy tail needing a longer `H`.

#### 3.3 Warm-up handling
Per contiguous segment (§4), z-score stats start empty. Until `n_warm` bars of that segment are seen, the estimator is unreliable:
- Bars `0 .. n_warm-1` of a segment: emit `z = 0` for the z-scored features and set the corresponding `m_t` flag bit (treated as "stats-not-ready," distinct from "data-missing" only in the QA log; both are 1 in the mask). Default `n_warm = max(64, H/8)`. Warm-up is a **strictly-causal** per-bar condition (count of in-segment bars seen so far at `t`), so it is one of the few conditions permitted to set `m_t` (§4.5).
- Bounded features (§3.1) need no warm-up and are emitted normally from bar 0 of a segment.

#### 3.4 Storage & inference replay (auditability)
- **Training:** normalization is computed **online and causally** in the streaming pipeline; no global statistics are ever fit over the full dataset (that would leak the future). The EWMA state trajectory is deterministic given (raw data, `H`, `eps_var`, seed-free) so it is reproducible from the content-hashed raw Parquet.
- **Inference (two modes, both persisted):**
  1. **Streaming mode** (live): carry the live EWMA `(mu, var)` state per symbol per feature, updated bar-by-bar exactly as in training. The state vector at any wall-clock time is checkpointed (and content-hashed) so any historical prediction can be replayed bit-exactly.
  2. **Frozen-stats mode** (published artifact / leaderboard reproducibility): freeze `(mu, var)` per (symbol, feature) at the train/test boundary timestamp and apply them statically across the test window. This is the rolling-window estimator's natural companion and makes the published eval a pure function of (weights, frozen-stats table, raw test bars). The frozen-stats table ships with the model card and is content-hashed.
- Every run records: estimator type, `H`/`W` per feature group, `eps_*`, clip bounds, `n_warm`, and the dataset content-hash. Given these, the entire normalization is byte-reproducible.

---

### 4. Missing-data handling

Throughout this section the governing rule is: **every gate, fill, clip, split, and flag must be a pure function of raw data with effective timestamp `<= t+1`, and the batch (training) path must be byte-identical to the live (inference) path.** No verdict about bar `t` may be confirmed, reverted, or rewritten by any bar with effective timestamp `> t+1`. This is the same causal discipline the Data Pipeline section enforces on its data-quality gates, and §6 tests it pipeline-wide.

#### 4.1 Price gaps = HARD segment boundaries (no imputation)
Crypto 1m data has real gaps (listing day, halts, dump-coverage holes). A **price gap** = any missing 1m bar in the kline series (a timestamp expected by the 1m grid but absent) **or** an exchange-flagged halt.
- **Never impute price.** A gap **splits the symbol's stream into contiguous segments.** All windowed/diff/normalization state (EWMA stats, `tau` sketch, `t-1` references for `ret_close`/`d_oi`) **reset at each segment boundary.** The first bar of a new segment has no `t-1`, so `ret_close` and `d_oi` are emitted as `0` with their mask bit set (warm-up rule), and z-score warm-up restarts.
- A training **sample** (length-`L` window) is drawn entirely **within one segment** — windows never straddle a boundary. Minimum segment length to be usable = `L_min` (default `L_min = 128`; segments shorter than this are dropped from training, kept in the raw store).
- **Strict causality of the boundary.** A gap/halt is detectable from the absence of an expected grid timestamp `<= t+1` alone; it requires no forward confirmation. The **structural-break** segment split (a large single-bar jump, owned by the Data Pipeline section) is likewise triggered on the **bar-`t` jump magnitude alone** (`>= P%` single-bar move known at close `t`), **independent of any forward reversion**. Whether bar `t` becomes a new-segment start (which resets `ret_close`, `d_oi`, EWMA, and `tau` state for bar `t`) therefore depends only on data `<= t+1`. If a non-reversion confirmation is ever desired, it is realized as a **lagged** correction affecting only bars from the confirmation bar onward; it **never retro-edits bar `t`'s `segment_id` or features**. `segment_id` is consumed only insofar as it gates windowing/resets; it is derived purely causally and is never itself a model input.
- Rationale: imputing a flat/interpolated price across a gap fabricates microstructure the model would learn as real. Segment-splitting is the same discipline Kronos applies to discontinuities, adapted to crypto's harsher gap profile.

#### 4.2 Volume / amount / OI gaps = zero-fill (not boundaries)
Liquidity fields can legitimately be zero or briefly absent without a price discontinuity (a 1m bar with price carried forward but no trades — Binance still emits the bar with `V=0`). Per Kronos's missing-volume treatment:
- `V_t = 0` / `QuoteTurnover_t = 0` → `log_volume`, `log_amount` evaluate to `log(1) = 0` naturally; aggTrades-derived features (`TFI`, counts, sizes) → all `0`; set mask bits. The trigger condition is **bar-`t` volume `== 0`**, a strictly-causal per-bar test, so it is permitted to set `m_t` (§4.5). This is **not** a segment boundary (price is continuous).
- Missing OI snapshot (API gap) → causal forward-fill the last known `OI` with effective timestamp `<= t+1` (§1 rows 15–16); if no prior OI in the segment, `log_oi = 0`, `d_oi = 0`, flag set.

#### 4.3 Spot pairs (no perp data)
For spot symbols there is no funding/OI by construction: `funding_rate = 0`, `log_oi = 0`, `d_oi = 0`, and mask bits 14–16 set to 1 for **every** bar. This is a **structural, time-invariant** property of the symbol (a strictly-causal "channel structurally absent" condition), handled identically to Kronos's missing-channel convention — the flag tells the model "this channel is structurally absent," not "momentarily zero."

#### 4.4 Kronos volume/amount dropout (training robustness)
Replicate Kronos's regularizer: **during training only**, with probability `p_drop = 0.05` per (bar, field) independently, zero out `log_volume` and/or `log_amount` (post-normalization set to the feature's neutral value = its running mean → `z = 0`) and set the mask bit. This teaches robustness to liquidity-data outages.
- **Causality/leakage note:** dropout is applied **after** normalization stats are computed from the *true* values (the dropout does not corrupt the EWMA state), and is **never** applied at inference/eval. Seed-pinned and logged so any training batch is reproducible. Extends optionally to `log_oi` at the same `p_drop` (config flag, default off to keep parity with Kronos which dropped only volume/amount).

#### 4.5 Model-visible mask `m_t` vs QA-only quality columns (strict causal separation)
This is the single most leakage-prone surface, so the rule is explicit and loader-enforced.

- **`m_t` (model-visible)** may be set **ONLY** from strictly-causal, per-bar conditions whose truth value at bar `t` depends on no datum with effective timestamp `> t+1`. The complete, closed set of permitted `m_t` triggers is:
  1. **bar-`t` volume `== 0`** (zero-liquidity bar, §4.2) → bits 6–13;
  2. **structurally-absent spot funding/OI** (§4.3) → bits 14–16;
  3. **warm-up** — z-score stats-not-ready or `tau` window-not-filled or first-bar-of-segment (no `t-1`), all counted from in-segment bars `<= t` (§3.3, §2.3, §4.1) → the corresponding feature bits;
  4. **bar-`t` `vol_recon_err > 0.01`** (§2.2), a comparison of bar-`t` quantities only → bits 8–13.
- **`is_stale` and the packed `dq_flags` bitfield are QA-ONLY columns.** They are **never fed to the model** and are **never OR-ed into `m_t`.** In particular, the **stale / constant-price** verdict (owned by the Data Pipeline section) is derived for bar `t` from a **strictly trailing** window — "the last `>= N` bars including `t` were flat" — never from forward bars; even so, because run-membership is a *quality* signal rather than a *missingness* signal, it lives only in `dq_flags`/`is_stale` and does **not** propagate into `m_t`. Likewise no forward-confirmed structural-break or bad-tick verdict may ever reach `m_t`.
- **Bad-tick handling (no retro-edit).** Any bad-tick decision (e.g. a single-bar MAD / return-excursion test) is decided and applied using **only** data with effective timestamp `<= t+1`, and **no forward-revert confirmation may rewrite bar `t`'s emitted OHLC or any feature derived from it** (`range`, `body`, wicks, `ret_close`, and the EWMA/`tau`/vol state that ingest them). If forward confirmation is desired, it is realized **only** as a lagged flag on bar `t+R` read at/after `t+R`, never as an edit to bar `t`. The batch and live paths are identical here by construction.
- **Loader assertion (enforced at data-load time):** the loader asserts that `m_t` is reconstructible from the four permitted causal trigger sets above and that **no `dq_flags`/`is_stale`/structural-break/bad-tick bit is OR-ed into `m_t`**. Any future-dependent gate bit reaching `m_t` is a hard load-time failure. The QA columns travel beside the tensor purely for the data-quality report and are dropped before the batch reaches the model.

---

### 5. Numerical constants & stability

| Constant | Default | Used in | Rationale |
|----------|---------|---------|-----------|
| `eps_px` | `1e-8` | features 3–5 (price-range denominators) | Guards `H==L` flat bars (denominator `0`); `1e-8` is far below any real price tick so it never distorts a non-flat bar. On a true `H==L` bar, `body/wicks → 0`, which is the correct "no shape" value. |
| `eps_cnt` | `1e-9` | TFI, signed-count imbalance | Guards a zero-trade bar (`V_buy+V_sell=0`) → imbalance `0`. |
| `eps_sz` | `1e-9` | mean/dispersion/large-share denominators | Guards zero base-volume bars. |
| `eps_oi` | `1.0` | `d_oi` log-ratio | OI is a large integer count; additive `1.0` inside the ratio is negligible yet keeps `log(0)` finite if OI momentarily reads `0`. **Kept intentionally consistent with `log_oi`'s `+1` form (same `1.0` offset) so OI epsilon handling is uniform across rows 15–16.** |
| `eps_var` | `1e-8` | all z-scores | Guards zero-variance warm-up / constant-feature segments; with `n_warm` gating this rarely binds but must be present. |
| `clip` | `[-5, 5]` | all emitted features | Outlier guard (Kronos parity). Applied last, after normalization. |

**OI `+1` note.** `log_oi` (row 15) uses `log(OI_t + 1)` rather than the canonical bare `log(OI_t)` for the **same stability reason** as `log_volume`/`log_amount`: it keeps `log` finite when a polled OI snapshot momentarily reads `0` and avoids a discontinuity at the channel's lower edge. The additive `1.0` is negligible against realistic OI magnitudes. `d_oi` uses the matching `eps_oi = 1.0` inside its log-ratio, so the two OI features share one epsilon convention.

**Stability rules.** (1) All `log(...)` use the `+1` or `+eps` forms above; no bare `log` ever sees a non-positive argument. (2) All division denominators carry an `eps`. (3) Compute in `float64` through the reduction and normalization, cast to `float32` only at the final emit (avoids catastrophic cancellation in the EWMA variance recursion). (4) `var` is floored at `0` before `sqrt` (the West recursion is non-negative analytically but clamp against fp drift). (5) NaN/Inf at emit is a **hard pipeline error**, never silently zero-filled — it means an upstream `eps`/segment rule was violated; the bar is quarantined and the run fails the CI check in §6.

---

### 6. Lookahead-safety invariant (CI-enforced, transform-agnostic)

**Invariant (single, testable, transform-AGNOSTIC property).** *The ENTIRE per-bar pipeline output for bar `t` — the feature vector `x_t`, the model-visible mask `m_t`, the `segment_id`, and any training target derived for bar `t` — is a pure function of raw exchange data with effective timestamp `<= t+1` (the close of bar `t`). This holds for **every** transform without exception: every data-quality gate (bad-tick clip, structural-break split, stale/constant-price flag), every fill (zero-fill, OI forward-fill, funding forward-fill), every clip, every segment split, every normalization (EWMA / rolling z-score), the `tau` percentile, and any input-path vol-scaling (§7). Equivalently: truncating the raw input stream at any point `>= t+1`, or arbitrarily corrupting all raw data strictly after `t+1`, leaves `(x_t, m_t, segment_id_t, target_t)` byte-identical.*

This invariant is **not** an enumerated list of "covered transforms." Any transform that reads beyond `t+1` — including any newly added gate, fill, or feature — violates it and must fail CI. The previous closed enumeration (which named only the already-safe `tau` percentile, forward-fills, and EWMA recursion) is **discarded**, because enumerating only the safe transforms gives false assurance precisely where the future-dependent gates hide.

**The unit test the CI enforces (future-invariance / truncation + perturbation test):**

```
For a sampled set of (symbol, bar index t):
  1. Compute the full pipeline over the entire stream → record
     (x_t, m_t, segment_id_t, target_t).
  2. Truncate the raw input to ONLY bars/trades/funding/OI with effective
     timestamp <= t+1  (delete everything in the future of bar t's close).
  3. Re-run the WHOLE pipeline on the truncated stream → primed (').
  4. ASSERT  x_t == x_t'  (bit-exact, float64 pre-cast)  AND  m_t == m_t'
            AND segment_id_t == segment_id_t'  AND target_t == target_t'.

  PERTURBATION test (exercises future-dependent code paths directly):
  5. Arbitrarily corrupt/replace ALL raw data strictly after t+1 (random noise,
     deletion, sign flips, fabricated reverts/mean-reversions) → re-run the WHOLE
     pipeline → double-primed (").
  6. ASSERT  x_t == x_t''  AND  m_t == m_t''  AND  segment_id_t == segment_id_t''
            AND target_t == target_t''.
     ANY dependence of bar t on future data — a forward-revert bad-tick rewrite,
     a future-confirmed structural break, a forward stale-run membership flag,
     a vol-scaling estimator that peeks ahead — makes step 6 fail.
```

**Sampling discipline (critical — bars are NOT drawn at random).** Because the leakage-prone transforms only fire in specific regions, the sampler **must** draw test bars **specifically from the high-risk regions** in addition to a random baseline:
- **bad-tick-adjacent** bars (bar `t` and `t±R` around any flagged tick),
- **segment-boundary** bars (first/last bars of every contiguous segment, and bars immediately across a gap/halt),
- **structural-break** bars (bars at and around every large single-bar jump),
- **stale-run** bars (bars inside and at the edges of every detected constant-price run).
A random-only sample would systematically miss the exact bars where a future-dependent gate could leak; sampling the high-risk regions guarantees the perturbation test actually exercises those code paths.

Passing both the truncation and perturbation tests for this stratified sample, per symbol per regime, is a **merge gate** — the pipeline cannot land if **any** transform leaks future information. The test is seeded and runs on a fixed fixture so it is deterministic. A complementary fast assertion runs every training batch: it re-derives the `(mu_{t-1}, var_{t-1})` used to normalize bar `t` and asserts they contain no contribution from `f_{>=t}`, and the §4.5 loader assertion (no future-dependent quality bit OR-ed into `m_t`) runs on every load.

---

### 7. Vol-scaling hook (interface only; logic lives in the vol-scaling section)

The volatility-normalization of prediction *targets* (and optionally `ret_close`/`range` inputs) is defined elsewhere; this stage exposes the exact plug point so the two sections compose without ambiguity.

- **What this stage provides to the vol-scaling section:**
  1. The **raw** (pre-z-score) `ret_close` and `range` series per (symbol, segment), plus their float64 values, so the vol estimator can build a strictly-causal realized-vol series from them.
  2. A causal-vol callback contract:
     ```
     sigma_t = causal_vol(symbol, t)        # realized/rolling vol as of close of t-1..t,
                                            # MUST use only bars with effective timestamp <= t+1
     ```
     The interface guarantees `causal_vol` is called with, and may read, only in-segment past/current bars (same segment-boundary reset rules as §4 apply to the vol state).
- **Where it plugs in:**
  - **Input path (optional):** if `config.vol_scale_inputs == True`, features 1 (`ret_close`) and 2 (`range`) are divided by `sigma_t` *before* the §3.2 z-score (the z-score then standardizes the already-vol-normalized quantity). The hook is applied at the raw-feature stage, inside the same segment/causal envelope; mask and warm-up rules are inherited (a bar with undefined `sigma_t` during vol warm-up sets the same flag bits via the §4.5 warm-up trigger).
  - **Target path (always on, per project contract):** the *prediction target* derived from bar `t+1`'s `ret_close` is normalized by `sigma_{t}` (vol known at decision time `t`), so the model predicts a regime-relative move. This transform is owned by the vol-scaling section and the training-target builder, **not** applied to the input vector here.
- **Causal-safety of the hook:** `sigma_t` must be computable from data with effective timestamp `<= t+1`; the vol-scaling section is responsible for that proof, and — because the §6 invariant is transform-agnostic — **any input-path vol-scaling is automatically inside the merge-gate's scope** (it is one of the transforms the truncation/perturbation test exercises, with sampled bars including segment-boundary and warm-up regions where the vol estimate is most fragile). No lookahead may enter through this hook.

---

## 2. Data Pipeline (ingest → Parquet → DuckDB → dataset)

This subsystem turns raw Binance dumps into the content-hashed, lookahead-safe `R^16` bar vectors the tokenizer consumes. It is **immutable-raw, append-only, deterministic, and seed-pinned**: every byte the model ever trains on is reproducible from the raw store + a versioned config. The pipeline is the data-side proof of the project's auditability contract (§6 of the feature spec is enforced here as a merge gate).

The single non-negotiable invariant of this section is **strict causality with batch/live parity**: every emitted `(x_t, m_t, segment_id_t)` is a pure function of raw data with effective timestamp `≤ t+1`, and the code path that produces a training batch is byte-for-byte the same code path that serves a live bar. No gate, fill, clip, split, normalization, or vol-scaling step is permitted to read a single bar beyond `t+1`, and no gate may retro-edit an already-emitted bar from future data.

```
data.binance.vision dumps ─┐
                           ├─► [1] INGEST ─► immutable raw (zip + manifest)
Binance futures REST/ws  ──┘                       │
                                                   ▼
                                   [2] aggTrades → 1m bar reduction (Polars/DuckDB, parallel)
                                                   │
                                                   ▼
                              [5] DATA-QUALITY GATES (strictly-causal clip / gap / stale / structural filters)
                                                   │
                                                   ▼
                                   [3] PARQUET  (symbol, frequency, date) zstd
                                                   │
                                                   ▼
                                   [4] DuckDB query / training-set assembly  ◄── [6] manifest + content hash
                                                   │
                                                   ▼
                                   [7] IterableDataset → 512-bar windows → tokenizer
```

The **feature-vector computation itself** (the `R^16` transform, causal normalization, masks) is owned by the feature-spec section and is invoked *inside* stage [2]/[3]; this pipeline owns ingest, reduction plumbing, partitioning, quality gates, versioning, and the streaming dataset.

---

### 1. INGEST (immutable raw)

Two sources, two cadences, **one immutability rule**: raw bytes are written once under a content-addressed path and never mutated. All downstream stages read raw; none write back to it.

#### 1.1 Bulk dumps — `data.binance.vision`

| Stream | URL template | Granularity | Schema (CSV columns) |
|---|---|---|---|
| klines | `data/{spot\|futures/um}/{monthly\|daily}/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY-MM[-DD]}.zip` | monthly + daily tail | `open_time, open, high, low, close, volume, close_time, quote_volume, count, taker_buy_base, taker_buy_quote, ignore` |
| aggTrades | `data/{spot\|futures/um}/{monthly\|daily}/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM[-DD]}.zip` | monthly + daily tail | `aggTradeId, price, quantity, firstTradeId, lastTradeId, timestamp, isBuyerMaker, isBestMatch` |

- **Backfill policy:** pull **monthly** archives for the historical body (2019→last full month) and **daily** archives for the ragged current-month tail; switch the most-recent month from daily→monthly once Binance publishes it (idempotent: re-download monthly, verify it supersedes the dailies, then retire the dailies in the manifest — never delete raw).
- **Integrity:** each archive ships a `.CHECKSUM` (SHA-256). Verify on download; a mismatch hard-fails ingest (no silent skip). Record `(url, sha256, bytes, http_etag, fetched_at_utc)` per file.
- **Idempotency / resumability:** ingest is a pure function of the URL list. A download job is skipped iff a verified file with the recorded SHA-256 already exists. Parallelism = `min(n_cpu, 16)` concurrent HTTP streams with exponential backoff (Binance rate-limits the static host lightly; be polite).
- **Raw layout (immutable):**
  ```
  raw/binance/{market}/{stream}/{SYMBOL}/{SYMBOL}-{stream}-{period}.zip
  raw/binance/_ingest_manifest/{ingest_run_id}.jsonl     # one line per file: url, sha256, bytes, etag, fetched_at
  ```
  `market ∈ {spot, um}` (um = USDⓂ perp). Raw zips are **never decompressed in place**; stage [2] streams them.

> **aggTrades is the heavy stream.** A liquid perp (BTCUSDT) emits ~30–80M aggTrade rows/month (~1–3 GB zstd-equivalent CSV). klines are ~43.2k rows/month/symbol (trivial). Plan storage and parallelism around aggTrades.

#### 1.2 API streams — funding & open interest

`data.binance.vision` does **not** carry funding/OI history, so use the futures REST API (perps only):

| Field | Endpoint | Native cadence | Causal rule (feature-spec §1 rows 14–16) |
|---|---|---|---|
| `funding_rate` | `GET /fapi/v1/fundingRate` (paginated by `startTime/endTime`, ≤1000 rows) | settles every 8h | carry last value with `fundingTime ≤ t+1` (forward-fill) |
| `open_interest` | `GET /futures/data/openInterestHist` (`period=5m`, ≤500 rows, **~30-day retention**) | 5m snapshots | latest snapshot with `timestamp ≤ t+1` (forward-fill) |

- **OI retention trap (flag explicitly):** `openInterestHist` only serves ~30 days. To reach 2019, OI **must be polled forward continuously and accumulated into raw from day one** of the project; there is no historical bulk source. Until enough forward history exists, **OI features are zero-filled + flagged** exactly like spot pairs (feature-spec §4.2/§4.3) — this is honest missingness, not lookahead. Document the per-symbol OI-coverage start date in the manifest.
- **Funding** has full history via `fundingRate`, paginate to genesis.
- Raw API responses are written as immutable JSONL shards partitioned by `(symbol, month)`, hashed identically to dumps:
  ```
  raw/binance/um/funding/{SYMBOL}/{SYMBOL}-funding-{YYYY-MM}.jsonl
  raw/binance/um/oi/{SYMBOL}/{SYMBOL}-oi-{YYYY-MM}.jsonl
  ```

#### 1.3 Universe selection

Top ~100–200 USDT pairs by cumulative quote volume, frozen at **project start** into `config/universe.yaml` (symbol, market(s), listing date, perp-available flag, OI-coverage start). Survivorship bias is acknowledged and bounded: v1 is a *modeling* claim (tokenizer), not a tradable-universe claim; the eval section's backtest must use the same frozen universe and disclose this. Spot+perp pairs are ingested for both markets; the fused token uses perp microstructure where available, spot otherwise (mask bits 14–16 carry the distinction).

---

### 2. aggTrades → 1m bar reduction (the compute-heavy step)

This stage reduces ~10¹¹ aggTrade rows to per-bar microstructure (feature-spec §2), then assembles the joined per-bar table that the feature transform consumes. It is **embarrassingly parallel by `(symbol, month)`** and streams so memory stays `O(one month of one symbol)`.

#### 2.1 Engine choice & shape

**Polars (lazy + streaming) is the default executor**; DuckDB is the equivalent SQL fallback (both read the zip'd CSV directly via Arrow). Polars wins here because the buy/sell split is a `group_by_dynamic` with conditional aggregations expressed cleanly in the expression API, and its streaming engine spills the rare oversized month.

Per `(symbol, month)` worker:

```python
# Polars lazy plan — executed with .collect(engine="streaming")
agg = (
    pl.scan_csv(zip_member, schema=AGGTRADES_SCHEMA)          # zero-copy Arrow read from zip
      .with_columns([
          (pl.col("timestamp") // 60_000).alias("bar_id"),    # integer-ms floor → 1m bucket (§2.1)
          pl.col("isBuyerMaker"),
      ])
      .group_by("bar_id")
      .agg([
          # taker BUY = isBuyerMaker == False ; SELL = True   (feature-spec §2.1, deterministic ground truth)
          pl.col("quantity").filter(~pl.col("isBuyerMaker")).sum().alias("V_buy"),
          pl.col("quantity").filter( pl.col("isBuyerMaker")).sum().alias("V_sell"),
          (~pl.col("isBuyerMaker")).sum().alias("N_buy"),
          ( pl.col("isBuyerMaker")).sum().alias("N_sell"),
          pl.col("quantity").sum().alias("V_agg"),
          pl.len().alias("N_trades"),
          pl.col("quantity").mean().alias("mean_sz"),
          pl.col("quantity").std(ddof=0).alias("std_sz"),      # population std → trade_size_dispersion
          # rolling τ percentile (large_trade_share) is NOT computed here — see note below
      ])
)
```

- **Bucketing is exact integer arithmetic** `bar_id = floor(timestamp_ms / 60000)` — never float binning, never `group_by_dynamic` on a parsed datetime (avoids DST/leap/rounding drift). Left-closed/right-open is automatic: a trade at exactly `t+1` ms maps to `bar_id+1`.
- **dtypes at read:** `price, quantity → Float64`; `timestamp → Int64` (ms); `isBuyerMaker → Boolean`; ids → `Int64` (dropped after use). All reductions in **float64** (feature-spec §5 rule 3), cast to float32 only at final Parquet emit.

#### 2.2 The rolling-τ feature (`large_trade_share`) — a strictly-causal sequential pass

`large_trade_share` needs `tau_t` = the `q`-th percentile of **per-trade** sizes over the **past** `W_tau`-bar window (feature-spec §2.3), which a single `group_by` cannot express. It is computed in a **second, strictly-causal sequential pass** per `(symbol, segment)` using a **t-digest quantile sketch** updated bar-by-bar. The default `q` is set so a non-trivial fraction of bars contain an above-threshold trade: `q ∈ {80, 90, 95}` (default `q = 90`) — a feasible point under the feature-spec §2.3 criterion "5–40% of bars are exactly 0; median of the non-zero bars ∈ [0.1, 0.5]" (the original `q=99` is infeasible, since a 99th-past-percentile threshold leaves almost every current bar at exactly 0). Feasibility is verified on a data slice before `q` is locked.

```
state = TDigest()                  # per (symbol, segment), persisted with dataset hash
ring  = deque(maxlen=W_tau)        # bars' size-pools for window eviction
for bar_id in segment_order:
    tau_t = state.quantile(q) if warm else None          # τ from sketch as of close(t-1)
    large_share_t = sum(sz >= tau_t for sz in sizes[bar_id]) / (V_agg_t + eps_sz) if warm else 0.0
    # ONLY AFTER emitting bar t do we fold bar t's sizes into the sketch (causal):
    state.add_batch(sizes[bar_id]); ring.append(sizes[bar_id])
    if len(ring) == W_tau: state = rebuild_or_subtract(ring)   # window eviction
```

This pass is cheap (it consumes the already-reduced per-bar size pools, not raw trades) and **inherits the segment-reset rule** (§4): the sketch and ring reset at every segment boundary; warm-up emits `0` + flag bit 13. The sketch state at any bar is checkpointed and content-hashed for bit-exact replay. Because `tau_t` reads the sketch as of `close(t−1)` and folds bar `t` only *after* emitting bar `t`, this transform reads no bar beyond `t` and is exercised by the §6 purity test.

#### 2.3 Join to klines + feature transform

The per-bar microstructure table is **left-joined onto the 1m kline grid** (klines are the canonical bar spine; aggTrades may be sparse on illiquid bars):

```sql
-- DuckDB equivalent of the join; klines define the grid, microstructure is attached
SELECT k.bar_id, k.open, k.high, k.low, k.close, k.volume AS V_kline, k.quote_volume,
       a.V_buy, a.V_sell, a.N_buy, a.N_sell, a.V_agg, a.N_trades, a.mean_sz, a.std_sz, a.large_share,
       f.funding_rate, o.open_interest
FROM klines_grid k
LEFT JOIN agg_bars a USING (bar_id)
LEFT JOIN funding_ff f USING (bar_id)      -- forward-filled to t+1 (§1.2)
LEFT JOIN oi_ff      o USING (bar_id)
ORDER BY k.bar_id;
```

`vol_recon_err = |V_agg − V_kline| / (V_kline + eps_sz)` is computed here as the QA scalar (feature-spec §2.2): `>0.01` sets aggTrades mask bits 8–13 and routes the bar to the DQ report. The joined, ordered, segment-tagged table is then handed to the **feature-spec transform** (§3 normalization, §4 missing-data, §7 vol hook) which emits the final `(x∈R^16, m∈{0,1}^16, ts∈int64)` per bar. Funding/OI forward-fill is performed **within the join, capped at `≤ t+1`**, and is covered by the §6 truncation test.

#### 2.4 Parallelism & throughput

- **Unit of work = `(symbol, month)`** → ~`200 symbols × ~80 months × 2 markets` ≈ low tens of thousands of independent jobs. Schedule on a process pool (`n_workers ≈ n_physical_cores`); each worker is memory-bounded by one symbol-month.
- **Determinism:** sort key is `bar_id` (total order); no job depends on another within a symbol-month *except* the cross-month sequential passes (τ-sketch, EWMA stats), which are stitched **per-symbol in time order** in a light second stage that carries only the small sketch/stat state across month boundaries (not the data). State carry is serialized and hashed. Because the carried state is strictly trailing, stitching introduces no forward dependence.
- **Throughput target:** the full ~500M–1B-bar build is a **once-per-config batch job** (hours on a multi-core box), not a hot path; correctness/auditability dominate speed. Incremental daily appends process only the new daily dumps + the resumed state.

---

### 3. Parquet partitioning, schema & compression

Processed bars land in a Hive-partitioned Parquet lake. **Partition key = `(symbol, frequency, date)`** (date = UTC calendar day of `bar_open`), the natural granularity for time-ranged training queries and daily incremental appends.

#### 3.1 Partition path template

```
processed/bars/symbol={SYMBOL}/frequency={FREQ}/date={YYYY-MM-DD}/part-000.parquet
#   e.g. processed/bars/symbol=BTCUSDT/frequency=1m/date=2023-07-15/part-000.parquet
```

- `FREQ ∈ {1m, 5m, 15m}` — 1m is the spine; 5m/15m are downstream rollups (optional multi-scale, generated by causal resampling of 1m, same schema).
- One file per partition (a full 1m day = 1440 rows → tiny). DuckDB/PyArrow prune partitions by `symbol`, `frequency`, and `date` range with zero scan of irrelevant data.

#### 3.2 Column schema & dtypes

| Column | Arrow dtype | Notes | Model-visible? |
|---|---|---|---|
| `bar_open_ms` | `int64` | UTC ms, bar-open timestamp `t`; the causal anchor + temporal-embedding key | yes (→ `ts`) |
| `symbol` | `dictionary<string>` | also the partition key (stored for self-describing files) | no (provenance) |
| `frequency` | `dictionary<string>` | partition key | no |
| `segment_id` | `int32` | contiguous-segment index (resets at every gap and at every strictly-causal structural break, §4/§5.4); windows must not straddle | structural only (windowing) |
| `x_0 … x_15` | `float32` ×16 | the normalized feature vector (feature-spec §3 output); column names = feature names | **yes** |
| `m_0 … m_15` | `uint8` ×16 | model-visible missingness mask (1 = imputed/missing/stats-not-ready). Set **only** from strictly-causal per-bar conditions (bar-`t` volume==0, structurally-absent spot funding/OI, warm-up). **Never** OR-ed with any forward-confirmed gate verdict. | **yes** |
| `raw_ret_close` | `float32` | **raw** (pre-z-score) `ret_close` for the vol-scaling hook (feature-spec §7.1) | hook only |
| `raw_range` | `float32` | **raw** `range` for the same hook | hook only |
| `vol_recon_err` | `float32` | QA scalar (§2.2); **not a model input** | **NO — QA only** |
| `is_stale` | `uint8` | DQ gate flag (§5.3), derived from a strictly-trailing window; **QA only, never fed to the model, never OR-ed into `m_t`** | **NO — QA only** |
| `dq_flags` | `uint16` | packed bitfield of all §5 gate verdicts (clip-hit, gap-adjacent, stale, structural-break, illiquid), including lagged forward-confirmation bits; **QA only, never fed to the model, never OR-ed into `m_t`** | **NO — QA only** |

- **Model-visible vs QA-only is an enforced contract.** Exactly four column groups reach the model: `x_*` (input vector), `m_*` (missingness mask), `bar_open_ms` (temporal embedding), and `raw_*` (vol-scaling hook). Everything else — `vol_recon_err`, `is_stale`, `dq_flags`, `segment_id` (used only to bound windows, never read as a feature), `symbol` — is provenance/QA and is **never** placed in the model input tensor. The loader (§7) asserts this.
- **`m_t` causality guarantee.** `m_t` is set **only** from strictly-causal per-bar conditions: bar-`t` `volume==0`, structurally-absent spot funding/OI, and feature/normalization warm-up. No forward-confirmed run-membership (`is_stale`) or structural-break verdict is ever OR-ed into `m_t`. Any forward-confirmation a gate produces lives **exclusively** in the QA-only `dq_flags` bitfield.
- **Compression:** **zstd level 3** (default; level 9 for the cold archive). zstd on these float32 columns with Parquet's `BYTE_STREAM_SPLIT` encoding + dictionary on `symbol/frequency` gives ~3–5× over raw and decompresses fast enough that DuckDB scans are I/O-bound, not CPU-bound. Row-group size **128 MB** (≈ many symbol-days), `data_page_size` default; statistics enabled on `bar_open_ms` and `segment_id` for predicate pushdown.
- **No nulls in `x_*`:** missingness is encoded in `m_*` + neutral values (feature-spec §4), so the model input tensor is dense and null-free. `funding/oi` raw values are *not* stored separately in the processed lake — they're already folded into `x_14..x_15`; the raw values live in the immutable raw store for replay.
- **Volume estimate:** ~1B bars × (16 f32 + 16 u8 + ~6 aux cols) ≈ ~100–130 bytes/row pre-compression → **~50–80 GB zstd** for the full processed lake. Raw aggTrades dwarfs this (multi-TB); the processed lake is the working set.

---

### 4. DuckDB query layer (training-set assembly)

DuckDB is the **single read interface** over the Parquet lake — the trainer never globs files directly. A thin Python module exposes parameterized, content-hashable queries.

#### 4.1 View & key queries

```sql
-- one external view over the whole lake; partition columns become real columns
CREATE VIEW bars AS
SELECT * FROM read_parquet('processed/bars/**/*.parquet', hive_partitioning=true);

-- (a) a temporal slice for one symbol within a split window (training-set assembly)
SELECT bar_open_ms, segment_id, x_0..x_15, m_0..m_15, raw_ret_close, raw_range
FROM bars
WHERE symbol = ? AND frequency = '1m'
  AND bar_open_ms >= ?  AND bar_open_ms < ?        -- [split_start, split_end), set by §6 contract
ORDER BY bar_open_ms;

-- (b) per-symbol usable-segment catalog: maximal contiguous runs ≥ L_min, for window planning
SELECT symbol, segment_id, MIN(bar_open_ms) lo, MAX(bar_open_ms) hi, COUNT(*) n
FROM bars WHERE frequency='1m'
GROUP BY symbol, segment_id HAVING n >= 128;       -- L_min default (feature-spec §4.1)

-- (c) global per-symbol bar counts → IterableDataset sampling weights (per-symbol balancing)
SELECT symbol, COUNT(*) n_bars FROM bars
WHERE frequency='1m' AND bar_open_ms < ?           -- train split end only (no val/test leakage)
GROUP BY symbol;
```

- **Split-aware by construction:** every assembly query is parameterized by `[lo, hi)` boundaries supplied by the eval section's split policy (§6 contract below). The query layer **never** sees val/test bars when building the train loader — boundaries are passed in, not discovered, so a misconfiguration can't silently leak.
- **`x_*`/`m_*`/`raw_*` only:** assembly query (a) selects exactly the model-visible + hook columns plus `segment_id` (for windowing). It **never** selects `is_stale`, `dq_flags`, or `vol_recon_err` into the training path; those are read only by the offline DQ report.
- **Output to the dataset:** the segment catalog (query b) is the **window-planning index** the IterableDataset draws from; the temporal slice (query a) is the actual data fetch, streamed in `bar_open_ms` order per segment. DuckDB results are zero-copied to Arrow → Polars/NumPy → torch tensors without a pandas round-trip.

---

### 5. Data-quality gates (ingest **and** inference) — strictly causal, batch == live

Gates run **identically at ingest and at live inference** — same code, same thresholds, no batch-only branch — so the distribution the model sees online equals the distribution it trained on. The governing rule for **every** gate below is: **the verdict and its effect on bar `t`'s emitted OHLC, features, `m_t`, and `segment_id` are decided from data with effective timestamp `≤ t+1` only.** No gate performs a forward-revert / forward-confirmation that rewrites an already-emitted bar `t`. Where a forward signal is genuinely informative, it is realized **only as a lagged QA flag** on a *later* bar's `dq_flags`, never as a retro-edit of bar `t`. Gates **never silently delete or impute price** (that fabricates microstructure). Thresholds below are 1m-crypto-adapted from Kronos's structural-break/illiquid/stagnant filters; all are config-pinned and tunable, with the stated selection criterion.

#### 5.1 Flash-crash / bad-tick clipping (per-bar, strictly causal, on raw OHLC, before features)

A bad tick is a single-bar OHLC excursion that is implausible *on the evidence available at close `t`*. Detection and clipping use **only bar `t` (and the trailing `≤ t−1` reference)** — there is **no forward-revert confirmation**:

- **Per-bar excursion test (decided at close `t`):** compute `|log(C_t / C_{t-1})|` and the intrabar wick excursion `(H_t − max(O_t,C_t))` / `(min(O_t,C_t) − L_t)`. A bar is flagged a bad-tick candidate iff its single-bar return or wick exceeds a **trailing MAD threshold** `> k · MAD_trailing(ret, W=1440)` (the MAD window is strictly `[t−1440, t−1]`, known at close `t`), default `k = 12` (a tail far beyond any plausible 1m crypto move).
- **Clip action (decided at close `t`, applied to bar `t` immediately, no look-forward):** for a flagged bar, **clip the offending wick** `H_t`/`L_t` into a single-bar-consistent envelope around `[O_t, C_t]` and the trailing-MAD bound — i.e. the implausible wick is capped using only `O_t, C_t` and the trailing reference. The bar is **kept** (never dropped, never interpolated). `C_t` and `O_t` are **not** rewritten, so the realized `ret_close` reference for `t+1` is untouched. Set the clip bit in the QA-only `dq_flags`.
- **No forward-revert rewriting of bar `t`.** The previous design — "flag, wait `R=3` bars, clip only if it mean-reverts" — is **removed**: it made bar `t`'s emitted OHLC (and every feature derived from it: `range`, upper/lower-wick fractions, body, and the `ret_close`/EWMA/σ state that ingests them) depend on bars `t+1..t+3`, which is genuine lookahead in training and is impossible at live serving. The strictly-causal single-bar test above produces the **same** verdict in the batch build and in live serving.
- **Optional lagged forward-confirmation (QA only).** If a forward "did it revert?" signal is still wanted for analysis, it is realized **only** as a separate `dq_flags` bit written on bar `t+R` (read at/after `t+R`), describing the *t+R* observation. It **never** rewrites bar `t`'s OHLC, features, `m_t`, or `segment_id`, and it is never fed to the model.
- Defensive normalized clip to `[-5,5]` still applies downstream (feature-spec §3.2); this gate catches the rawer OHLC corruption *before* it poisons the EWMA variance — and now does so without reading the future.

#### 5.2 Exchange-outage GAP detection (segment boundaries)

- A **gap** = a 1m grid timestamp present in neither klines nor (for a halt) flagged as trading. Detection: `bar_open_ms[i] − bar_open_ms[i−1] > 60_000` within a symbol's sorted stream. This is decided from `≤ t` data (the absence of intervening bars at and before `t` is known at close `t`).
- **Action (feature-spec §4.1):** increment `segment_id`; reset all stateful estimators (EWMA, τ-sketch, `t−1` refs). **No price imputation across a gap, ever.** A gap-adjacent bar (first of a new segment) carries the gap flag; `ret_close`/`d_oi` → 0 + mask (these mask bits are strictly-causal per-bar warm-up conditions, so they may legitimately enter `m_t`).
- **Long-outage policy:** gaps `>` `G_drop = 6h` (360 bars) additionally tag the segment with an `outage` flag in the QA-only `dq_flags` for the DQ report (often a delisting/relisting or major incident); segments shorter than `L_min=128` after splitting are excluded from training (kept in raw).

#### 5.3 Stale-bar detection (zero-volume / constant-price runs) — strictly trailing

- **Zero-volume bar (strictly causal, may enter `m_t`):** `V_t = 0` is a per-bar condition known at close `t`. Features degrade naturally (feature-spec §4.2) and the corresponding `m_t` bit is set; this is a legitimate strictly-causal per-bar missingness signal. Not a boundary (price continuous).
- **Constant-price run membership (strictly trailing, QA only):** `is_stale` for bar `t` is derived **only from a trailing window**: "the last `≥ S = 30` bars *including and ending at* `t` were flat" — i.e. `H==L==O==C` and `V==0` across `[t−S+1, t]`. This reads **no** forward bar; the previous "flag every bar in the run (including bars whose run-membership is only knowable from bars after `t`)" formulation is **removed**. `is_stale` is written **only to the QA-only `dq_flags`/`is_stale` columns and is NEVER OR-ed into `m_t`** (the model-visible missingness comes solely from the strictly-causal `V_t==0` per-bar bit and warm-up, not from forward-confirmed run membership).
- **Dead-pair exclusion:** a trailing run of `≥ S_drop = 240` bars (4h dead, known at the bar that completes the trailing window) tags the segment low-priority in `dq_flags`; such stretches are **excluded from training windows** (they teach nothing and dilute per-symbol balance) but retained in the lake. This is Kronos's "stagnant K-line" filter, tightened to crypto's 24/7 always-on grid where true 4h flatlines signal a dead/illiquid pair, not a market close.

#### 5.4 Illiquidity / structural-break filter (Kronos-adapted, per symbol-segment)

- **Illiquid pair gate (offline, segment-level):** a symbol-segment whose **median 1m `quote_volume` `< Q_min = 1,000 USDT`** OR whose **fraction of zero-volume bars `> 30%`** is dropped from training (kept in raw). Selection criterion: tune `Q_min ∈ {500, 1000, 5000}` so the retained universe holds the frozen top-100–200 pairs but prunes their dead early-listing tails. This is a whole-segment retain/drop decision applied *after* the segment is fully observed; it never edits any bar's emitted `x_t`/`m_t`/`segment_id` and never enters the model — it only governs which segments feed the window sampler.
- **Structural break (strictly causal split on the single-bar jump):** the segment is split **at bar `t` on the bar-`t` jump magnitude alone** — a **`≥ P = 50%` single-bar price move known at close `t`** (e.g. a redenomination/contract change). The split decision uses **no forward reversion check**: the previous "`>50%` jump that does *not* revert" wording is **removed**, because "does not revert" is only establishable from bars after `t` and would make bar `t`'s `segment_id` (and the post-reset `ret_close`/`d_oi`/EWMA/τ values) depend on the future. The `50%`/1m threshold is deliberately extreme so only genuine non-market events trip it. If a non-reversion confirmation is desired, it is realized **only as a lagged QA `dq_flags` bit affecting bars from the confirmation bar onward** — it never retro-edits bar `t`'s `segment_id` or features.
- **Provenance:** all §5.4 retain/drop verdicts and the §5.1–5.3 flags (including any lagged forward-confirmation bits) are **persisted only in the QA-only `dq_flags`** so the eval section's backtest can optionally re-filter at query time without recomputing — and none of them is ever fed to the model or OR-ed into `m_t`.

> **Inference parity (mandatory, now unconditional):** the *exact* gate code runs in the live feature pipeline, with **no batch-only vs live-only branch**. Because every gate above is strictly causal (decided from `≤ t+1` data, no forward-revert rewriting of bar `t`), the verdict and emitted `(OHLC, x_t, m_t, segment_id)` for any bar are **identical** in the training build and in live serving. The only inference-time difference is that **training-only** regularizers (Kronos volume dropout, feature-spec §4.4) are **off** at inference — and these touch `x`-augmentation only, never the gates, `segment_id`, or `m_t`. Any lagged forward-confirmation (§5.1 / §5.4) is a QA-only `dq_flags` write on a later bar and is forbidden from rewriting any past input.

---

### 6. Data versioning + pipeline-wide lookahead-purity merge gate

Every produced dataset is **content-addressed** so any model prediction is replayable from `(weights, dataset hash, config hash, raw)`. The same machinery hosts the **lookahead-safety invariant**, which is enforced as a CI merge gate.

#### 6.1 What is hashed

```
dataset_hash = sha256(
    sorted( sha256(file) for file in produced_parquet_set )   # the file-set Merkle leaves
  ⧺ canonical_json(feature_config)                            # eps_*, clip, feature formulas version
  ⧺ canonical_json(norm_config)                               # estimator, H/W per group, n_warm
  ⧺ canonical_json(dq_config)                                 # all §5 thresholds (k,S,S_drop,Q_min,P,G_drop,W…)
  ⧺ canonical_json(reduction_config)                          # W_tau, q, sketch params
  ⧺ raw_ingest_manifest_hash                                  # SHA-256 set of every raw file consumed
)
```

The hash binds **data + the exact transform that produced it**. Changing one `eps`, one threshold, or one raw file → a new hash → a new immutable dataset version. The per-symbol **estimator state trajectories** (EWMA `(mu,var)`, τ-sketch) and the **frozen-stats table** (feature-spec §3.4) are serialized and hashed alongside, so both streaming and frozen-stats inference modes replay bit-exactly.

#### 6.2 Lookahead-purity invariant (transform-agnostic; the CI merge gate)

The invariant is **not** an enumerated list of "covered" transforms (an enumeration gives false assurance by testing only the already-safe steps). It is a single, **pipeline-wide property**:

> **The entire pipeline — every gate, fill, clip, split, normalization, vol-scaling hook, mask bit, `segment_id` assignment, and target — is a pure function of raw data with effective timestamp `≤ t+1`.** Truncating the raw stream at any point `≥ t+1` must leave `(x_t, m_t, segment_id_t, raw_t)` byte-identical, and perturbing any raw bar at `> t+1` must leave them byte-identical.

The CI gate enforces this with two tests run over the candidate dataset build:

1. **Truncation test:** for a sampled set of anchor bars `t`, rebuild `(x_t, m_t, segment_id_t, raw_t)` from the raw stream truncated at `t+1` and assert byte-equality with the full-history build.
2. **Perturbation test:** randomly corrupt raw bars strictly after `t+1` and assert `(x_t, m_t, segment_id_t, raw_t)` is unchanged.

**High-risk-region sampling (mandatory):** the sampled anchor bars are **not** drawn uniformly. The sampler specifically targets the bar regions where future-dependence is most likely to hide — **bad-tick-adjacent bars (§5.1), segment-boundary bars (§5.2), structural-break bars (§5.4), and stale-run bars (§5.3)** — in addition to a random background sample. The test **fails the merge** if *any* transform (the bad-tick clip, the structural-break split, the stale-run flag, the τ-percentile, the forward-fills, the EWMA recursion, the vol-scaling hook, or any future step) reads a single bar beyond `t+1`. Because the gates of §5 are now strictly causal, they pass this test; any future regression that reintroduces a forward-revert rewrite of bar `t` will be caught here rather than passing silently.

The test result is recorded in the manifest (§6.3); a dataset whose lookahead CI did not pass **cannot be published**.

#### 6.3 Manifest format (`manifest.json`, ships with the model card)

```json
{
  "dataset_hash": "sha256:…",
  "created_utc": "2026-06-18T00:00:00Z",
  "universe_config": "config/universe.yaml@<git_sha>",
  "raw_ingest_manifest_hash": "sha256:…",
  "configs": { "feature": "…@sha", "norm": "…@sha", "dq": "…@sha", "reduction": "…@sha" },
  "fsq": {                                                         // referenced from the Tokenizer section (single source of truth)
    "fsq_levels": [11, 9, 9, 7, 7, 5, 5],                          // D=7, all odd; bpt = Σ log2(L_i) = 20.06
    "bits_per_token": 20.06,
    "derived_split": { "coarse_dims": [11, 9, 9], "fine_dims": [7, 7, 5, 5],
                       "bpt_coarse": 9.80, "bpt_fine": 10.26, "V_c": 891, "V_f": 1225 }
  },
  "splits": { "train": ["…","…"], "val": ["…","…"], "test": ["…","…"] },   // [lo,hi) ms boundaries (leading-embargo-inset by eval)
  "n_bars": { "total": 982113402, "train": 0, "val": 0, "test": 0 },
  "n_symbols": 174, "n_segments": 0,
  "files": [ { "path": "processed/bars/symbol=BTCUSDT/frequency=1m/date=2023-07-15/part-000.parquet",
               "sha256": "…", "rows": 1440 } ],
  "state_artifacts": { "frozen_stats": "sha256:…", "tau_sketches": "sha256:…", "ewma_state": "sha256:…" },
  "code_version": { "pipeline_git_sha": "…", "feature_spec_git_sha": "…" },
  "lookahead_ci": { "truncation_test": "pass", "perturbation_test": "pass", "seed": 1337,
                    "n_sampled_bars": 50000,
                    "high_risk_strata": { "bad_tick_adjacent": 12000, "segment_boundary": 12000,
                                          "structural_break": 6000, "stale_run": 12000, "random_background": 8000 } }
}
```

- The `fsq` block is **stored derived, not authored here**: the canonical `fsq_levels = [11,9,9,7,7,5,5]` (all odd, `bpt = Σ log2(L_i) = 20.06`) and the **rule-derived** coarse/fine split (`coarse_dims={11,9,9}`, `fine_dims={7,7,5,5}`; `V_c=891`, `V_f=1225`) are owned by the Tokenizer section; the manifest records the derived split (it is **not** hardcoded 3/3 anywhere in the YAML — it is computed by the §d grouping rule and persisted).
- **Reproduce-any-prediction contract:** given the manifest + the raw store, the pipeline re-derives every `(x_t, m_t, segment_id_t)` byte-identically; given the weights, the AR model re-emits the prediction. The §6.2 truncation/perturbation result is **recorded in the manifest** — a dataset whose lookahead CI didn't pass cannot be published.
- Manifests are append-only; supersession (e.g., a backfilled month) writes a *new* manifest referencing the prior `dataset_hash` as `parent`, never edits one.

---

### 7. Training IterableDataset

The trainer consumes a **`torch.utils.data.IterableDataset`** that streams 512-bar windows from the DuckDB/Parquet layer, never holding the lake in memory.

#### 7.1 Window construction

- **Window = `L = 512` contiguous bars within a single `segment_id`** (hard rule; a window may never straddle a gap or a strictly-causal structural break — enforced by drawing only from the §4 segment catalog).
- **Stride / overlap:** training stride `s = 256` (50% overlap) by default — dense enough to use the data, sparse enough to decorrelate adjacent samples. Tune `s ∈ {128, 256, 512}`; selection criterion: validation IC stability vs. epoch wall-clock. Val/test use **non-overlapping** `s = L` to avoid double-counting in metrics.
- A segment of length `n` yields `floor((n − L)/s) + 1` windows; segments with `n < L` are unusable (dropped, but `L_min=128 < L=512` segments still feed the *segment catalog* for diagnostics).

#### 7.2 Tensor interface (per yielded sample) — model-visible columns only

```
x    : float32 [L, 16]      # normalized feature vector  (feature-spec §3 output)
mask : uint8   [L, 16]      # m_t: strictly-causal per-bar missingness flags ONLY
ts   : int64   [L]          # bar_open_ms → temporal embeddings in the AR backbone
raw  : float32 [L, 2]       # raw_ret_close, raw_range → vol-scaling hook (feature-spec §7)
seg  : int64   scalar       # segment_id (provenance/debug; bounds windows, NOT a model input)
sym  : int64   scalar       # symbol index (provenance/debug, NOT a model input)
# collated batch: x∈[B,L,16], mask∈[B,L,16], ts∈[B,L], raw∈[B,L,2]   (feature-spec batched contract)
```

- **Loader QA-only assertion (enforced on every fetch):** the dataset asserts that the only tensors handed to the model are `x`, `mask` (= `m_t`), `ts`, and `raw`, and that **`is_stale`, `dq_flags`, and `vol_recon_err` are never materialized into any model-visible tensor and are never OR-ed into `mask`**. Concretely:
  ```python
  # in the collate / yield path, with row blocks loaded from query (a)
  assert "dq_flags" not in model_batch and "is_stale" not in model_batch \
         and "vol_recon_err" not in model_batch, "QA-only columns must never reach the model"
  # m_t carries ONLY strictly-causal per-bar bits; no forward-confirmed gate bit may be present
  assert mask is feature_spec_mask, "mask must be feature-spec m_t, never the dq_flags bitfield"
  assert (mask & ~CAUSAL_MASK_BITS).sum() == 0, \
         "no forward-confirmed (is_stale/structural-break) bit may be OR'd into m_t"
  ```
  This makes the "QA-only never enters the model, and `m_t` is strictly causal" guarantee a runtime invariant, not just a documentation claim.

The **target builder** (vol-scaled, regime-relative; produced by autoregressive rollout, with MTP as speculative drafter) is owned by the vol-scaling/MTP sections and reads `raw` + the causal-vol callback; this dataset's job ends at delivering the causal inputs + the raw hook channel.

#### 7.3 Per-symbol balancing

Raw bar counts are wildly imbalanced (BTC/ETH ≫ long tail). Sample windows with **per-symbol weights** that flatten dominance without starving majors:

```
w_sym ∝ (n_windows_sym) ** alpha ,   alpha = 0.5   # square-root temperature (Kronos-style flattening)
```

`alpha=0.5` (tunable `∈ {0.0(uniform-per-window), 0.5, 1.0(natural)}`; criterion: held-out IC on *minor* symbols must not collapse while majors stay healthy). Sampling is **two-stage**: draw a symbol ∝ `w_sym`, then a uniform random valid window-start within that symbol's segment catalog. Seed-pinned per epoch (the seed + epoch index reproduce the exact window stream) and recorded for replay.

#### 7.4 Sharded streaming & determinism

- **Multi-worker / DDP:** the **segment catalog (DuckDB query b) is the shard unit** — segments are deterministically partitioned across `(rank, worker_id)` by a hash of `(symbol, segment_id, epoch_seed)`, so every (rank,worker) draws a disjoint, reproducible window stream with no coordination. DuckDB fetches each segment's bars in `bar_open_ms` order on demand; a small per-worker buffer (a few thousand windows) is shuffled to break temporal autocorrelation.
- **Determinism contract:** `(dataset_hash, epoch_seed, world_size, alpha, s)` fully determine the exact sequence of training batches → any batch is reproducible for debugging and the run is byte-auditable.

#### 7.5 Train/val/test split contract (embargo deferred to eval section)

This pipeline **does not choose** the split or embargo — it **consumes boundaries** the eval section provides, and guarantees they are honored:

- **Contract:** the eval section supplies, per fold, the **temporal `[lo, hi)` ms boundaries** for train/val/test of a **purged walk-forward + embargo** scheme. This dataset enforces: (1) windows are drawn **only** from within a split's `[lo, hi)`; (2) **no window crosses a split boundary** (a 512-bar window straddling train→val is discarded, which *is* the purge at the window level); (3) the embargo is applied by the eval section as a **boundary inset on the binding side** — this dataset simply receives the already-inset ranges and never reads outside them.
- **Leading embargo is the binding guard (default anchored mode).** In the default anchored/expanding walk-forward (train precedes test), the operative serial-correlation guard is a **leading embargo**: the eval section drops training samples whose anchor `t ∈ [T_k_start − E, T_k_start − 1]` (equivalently insets the train block's right edge by `E`), with `E = H_max + L_corr = 120` bars. This dataset receives the train range already inset to `[lo, T_k_start − E)` and reads nothing past it. A trailing embargo is **not** applied in anchored mode (there are no post-test training bars to embargo); it is in scope only for rolling/CPCV modes, and the eval section scopes it there explicitly. `H_max = 60` is the autoregressive-rollout label look-forward that drives the purge.
- The **purged-walk-forward + embargo policy, the embargo width `E`, the leading/trailing scoping, and the headline cost-aware Net-IR metric live in the eval section** (per project contract). The pipeline's sole obligation is the *mechanical guarantee* that no window mixes splits and no query touches bars outside the supplied (already-embargo-inset) range — making the temporal firewall a property of the data layer, not just the metric layer.

---

## 3. Volatility-Scaled, Regime-Relative Targets

This section owns the volatility transform that the feature spec (§7) exposes as a hook. It defines the causal current-regime volatility estimator `sigma_t`, the exact set of quantities it scales, the precise order of operations relative to the §3.2 z-score (so nothing is double-normalized), and the decode-time de-normalization that recovers absolute moves, prices, and quantiles. **Core premise:** a `+0.4%` 1-minute move is a non-event when BTC's realized vol is `1.2%`/min during a liquidation cascade, but a `4-sigma` tail when vol is `0.1%`/min in an overnight drift. Standardizing every move by its current-regime vol lets a 27M-param model spend its capacity learning *shape and direction relative to regime* instead of re-learning the unconditional vol distribution it will see at decode time anyway.

---

### 1. Current-regime volatility `sigma_t` (strictly causal)

`sigma_t` is the model's estimate, **made at decision time `t` (close of bar `t`)**, of the per-bar return volatility that governs the *next* move. It is built from the **raw, pre-z-score** `ret_close` series the feature spec hands over via the §7 callback contract (`causal_vol(symbol, t)`), and it uses **only bars with effective timestamp `<= t+1`** — equivalently only `ret_close` values up to and including bar `t`. No datum at or after `t+1` may enter.

Let `r_s = ret_close_s = log(C_s / C_{s-1})` be the **raw** (un-z-scored, un-vol-scaled) close-to-close log return of bar `s`. We track a causal, zero-mean (returns are ~mean-zero at 1m; we do **not** subtract a running mean — see rationale below) variance.

**(a) EWMA realized variance (default).** RiskMetrics-style exponentially weighted variance with decay `lambda`:

```
sigma_sq_t = lambda * sigma_sq_{t-1} + (1 - lambda) * r_t^2     # uses r_t (known at close t)
sigma_t    = sqrt(sigma_sq_t + eps_vol)
```

Crucially, `sigma_t` here **includes** `r_t^2` because `r_t` (the return *into* the close of bar `t`) is known at decision time `t` and legitimately informs the next-bar vol. The lookahead boundary is `t+1`, not `t`: `sigma_t` may read bar `t`, it may **never** read bar `t+1`. (Contrast with the §3.2 z-score state `(mu_{t-1}, var_{t-1})`, which normalizes the *input* `f_t` and therefore must exclude `f_t` itself. The vol estimator normalizes a *target* about bar `t+1`, so conditioning on `r_t` is correct and not leakage.)

**(b) Rolling realized vol (audit/reference + frozen mode).** Simple trailing window of `W` past-inclusive bars:

```
sigma_t = sqrt( (1/W) * sum_{s = t-W+1}^{t} r_s^2  + eps_vol )
```

This is the reproducible ground-truth reference used in the §6-extended lookahead test and in **frozen-stats inference mode**.

- **Zero-mean choice:** we use `E[r^2]`, not the de-meaned variance. At 1-minute horizon the drift term is negligible (`mu ~ 1e-5`, `mu^2` swamped by `E[r^2] ~ 1e-6..1e-4`), and subtracting a running mean injects a second causal state with no measurable benefit and a small leakage surface. `sqrt(E[r^2])` is the standard RiskMetrics convention.

**Defaults & tuning.**
- EWMA: `lambda = 0.97` (default). This is the RiskMetrics daily value, deliberately re-purposed here at 1m cadence; `lambda = 0.97` gives a center-of-mass `1/(1-lambda) ≈ 33` bars and a half-life `ln(2)/ln(1/lambda) ≈ 22.8` bars (~23 min) — fast enough to track intraday regime flips (funding flushes, news candles) yet smooth enough not to chase single-bar noise. Tune `lambda in {0.94, 0.97, 0.99}` (half-lives ≈ 11 / 23 / 69 bars).
- Rolling: `W = 60` bars (1h) default; tune `W in {30, 60, 120}` (30m/1h/2h).
- **Selection criterion (the standardization target):** choose `lambda`/`W` so the **vol-scaled return** `ret_scaled` (§2) is as close to unit-variance and regime-stationary as possible. Concretely, on a held-out month and per symbol: (i) the cross-time std of `ret_scaled` should sit in `[0.8, 1.25]`; (ii) the **regime-conditioned** std — bucket bars into vol terciles by an *independent* slow estimator (e.g. 1-day rolling vol) and measure std of `ret_scaled` within each tercile — should be **flat across terciles** (max/min tercile-std ratio `< 1.3`). Flatness across vol regimes is the whole point: it certifies the scaling actually removed the regime dependence rather than just rescaling globally. Secondary check: clip-hit rate (§2) `< 0.5%`.
- `eps_vol = 1e-8` (variance-space floor; `sigma_t` is then `>= 1e-4`, preventing division blow-ups on dead, zero-return segments).

**Warm-up & segments (inherits §4 / §3.3).** The vol state resets at every contiguous-segment boundary (a price gap). Within a segment, until `n_warm_vol` bars are accumulated, `sigma_t` is unreliable:
- `n_warm_vol = max(30, [half-life])` for EWMA; `= W` for rolling.
- During vol warm-up: input-path vol-scaling (if enabled) is **not** applied and the same mask/warm-up flag bits are set (per §7's "undefined `sigma_t`" clause); for the **target path**, a training sample whose decision bar `t` falls in the vol warm-up region is **excluded from the loss** (its vol-relative target is undefined) — it still appears as context but contributes no gradient. This avoids polluting the loss with targets normalized by a garbage `sigma`.

---

### 2. What gets vol-scaled, and what does NOT

Vol-scaling applies **only to magnitude-of-move quantities measured in log-price units**, because those are the features whose meaning changes with regime. Everything already bounded or already relative is left alone — scaling it would be dimensionally meaningless and would destroy its calibrated range.

| Quantity | Vol-scaled? | Reason |
|---|---|---|
| **`ret_close` (target, always)** | **YES** | The headline move. Target is `ret_close_{t+1}` (the move being predicted); scaled by `sigma_t`. |
| **`ret_close` (input, optional)** | optional (`config.vol_scale_inputs`) | Same units as target; scaling it puts input and target in the **same vol-relative space**. Default **ON** (see §3 rationale). |
| **`range` (input)** | optional, tied to `vol_scale_inputs` | `range = log(H/L)` is also a per-bar move magnitude in log-price units; if inputs are vol-scaled, `range` is scaled by the **same** `sigma_t` for consistency. |
| `body`, `upper_wick_frac`, `lower_wick_frac` | **NO** | Already in `[-1,1]`/`[0,1]`; they encode bar *shape* (ratios within the bar), not absolute move size. Regime-invariant by construction. |
| `TFI`, `signed_count_imbalance`, `funding_rate` | **NO** | Bounded / natural-unit. Vol carries no meaning here. |
| `log_volume`, `log_amount`, `trade_count`, `mean_trade_size`, `trade_size_dispersion`, `large_trade_share` | **NO** | Liquidity/microstructure scale features, not price-move magnitudes. They are *inputs to* regime, not things normalized *by* it. (`large_trade_share` is already `[0,1]`.) |
| `log_oi`, `d_oi`, `funding_rate` | **NO** | Positioning channel; different physical dimension than price moves. |

So vol-scaling touches **at most three quantities**: the `ret_close` target (always), and the `ret_close`/`range` inputs (optional, coupled by one flag). The multi-horizon MTP targets (1/5/15/60 min) are the multi-bar generalization of the first row — see §4.

---

### 3. Exact transform and order of operations (no double-normalization)

The single source of confusion is that two normalizations touch `ret_close`: this section's vol-scale and the feature spec's §3.2 causal z-score. They compose in a **fixed, non-commuting order**, and the divisors are computed from disjoint state so there is no leakage and no redundancy.

**Order of operations — INPUT path** (per bar `t`, raw `float64`, in-segment):

```
1. raw           f_t        = ret_close_t  = log(C_t / C_{t-1})        # feature-spec §1 row 1, RAW
2. vol-scale     g_t        = f_t / (sigma_t + eps_vol)                # sigma_t from §1, uses bars <= t (i.e. <= t+1 close)
   (range)       g^range_t  = range_t / (sigma_t + eps_vol)           # same sigma_t, if vol_scale_inputs
3. z-score       z_t        = clip( (g_t - mu^g_{t-1}) / sqrt(var^g_{t-1} + eps_var), -5, 5 )
                                                                       # §3.2 EWMA, but stats accumulated on g, NOT f
4. emit          x_t[ret_close] = z_t  (float32)
```

**Critical reconciliation rule (prevents double-normalization):**
- When `vol_scale_inputs == True`, the §3.2 causal z-score for `ret_close` and `range` is computed on the **vol-scaled** series `g`, **not** the raw `f`. I.e. the EWMA mean/variance state `(mu^g, var^g)` is fed `g_t`, not `f_t`. Because `g_t` is already ~unit-vol and ~mean-zero, this second z-score is a **mild whitening** (it corrects residual cross-symbol mean/dispersion that survives vol-scaling), not a re-standardization of the same quantity twice. The two operations are **not** the same normalization applied twice: vol-scaling removes the *time-varying intra-symbol regime* component; the z-score removes the *static cross-symbol* mean/scale offset. They target orthogonal nuisance variation.
- When `vol_scale_inputs == False`, the input `ret_close`/`range` follow the **plain §3.2 path on raw `f`** (the feature spec's default), and *only the target* is vol-scaled (§4). This is the minimal-coupling configuration.
- **No feature is ever divided by `sigma_t` twice**, and no feature is z-scored on both `f` and `g`. The config flag `vol_scale_inputs` deterministically selects exactly one path; both are logged with the run.

**The §3.2 clip semantics are preserved:** the final `[-5,5]` clip still applies (step 3), now guarding the whitened vol-scaled value. Track its clip-hit rate separately under the `vol_scale_inputs=True` regime; a >0.5% rate signals `lambda`/`W` too fast (over-reactive `sigma_t` producing fat-tailed `g`).

**Why inputs default to vol-scaled (`vol_scale_inputs = True`).** The AR backbone consumes a token derived from the input vector and is asked to predict a token whose `ret_close` slot is the vol-scaled target. If the input move lives in raw-z space but the target lives in vol-relative space, the model must internally *infer and invert* `sigma_t` from the other features to map between them — wasting capacity and creating a train/inference mismatch when `sigma_t` is volatile. Keeping input and target in the **same vol-relative space** makes the next-bar prediction a stationary, regime-invariant problem: "given the recent vol-relative shape, what is the next vol-relative move." This is the principal benefit of the transform and the reason it is on by default.

**Tokenizer placement (critical).** Vol-scaling happens **before tokenization**. The 16-dim bar vector that the FSQ tokenizer encodes already contains the vol-scaled-then-z-scored `ret_close` (and `range`). The tokenizer therefore quantizes a regime-relative quantity; the same code means the same *vol-relative* move regardless of regime, which keeps the FSQ codebook well-utilized across calm and volatile regimes (no separate codes wasted re-encoding the unconditional vol level). The target the AR model predicts is the *token* of bar `t+1`, whose `ret_close` slot is likewise vol-relative — input and target share one tokenized space by construction.

---

### 4. Targets, multi-horizon (MTP), and distributional decode

**Single-horizon target (next bar).** The 1-step prediction target for decision bar `t` is the **vol-scaled** next-bar return:

```
y_{t->t+1} = ret_close_{t+1} / (sigma_t + eps_vol)
```

Note the divisor is `sigma_t` — the vol **known at decision time `t`** — applied to the *future* return `ret_close_{t+1}`. This is the one place a future-bar quantity (`ret_close_{t+1}`) appears, which is correct: it is the *label*, not an input, and it is normalized by a strictly-causal `sigma_t`. The model thus learns `y` = "how many current-regime sigmas does the next bar move," a stationary target.

**Multi-horizon MTP targets.** The DeepSeek-style MTP heads predict the causal chain of future bars at depths `h in {1,5,15,60}` min. Two design points:
- Each horizon's cumulative log-return target is scaled by a **horizon-appropriate** causal vol so all heads predict ~unit-variance quantities:
  ```
  R_{t->t+h} = log(C_{t+h} / C_t) = sum_{j=1}^{h} ret_close_{t+j}
  y^{(h)}_t  = R_{t->t+h} / ( sqrt(h) * sigma_t + eps_vol )
  ```
  The `sqrt(h)` is the IID random-walk vol-scaling of an `h`-bar sum; it is a **deterministic constant**, not a second estimator, so it introduces no leakage and keeps every head's target near unit variance using the single causal `sigma_t`. (If empirical `h`-bar vol departs from `sqrt(h)*sigma_t` due to autocorrelation/vol-clustering, optionally replace `sqrt(h)` with a per-horizon causal multiplier `c_h` calibrated on past data only — default off; `sqrt(h)` is the clean baseline.)
- All horizons use the **same** `sigma_t` (decision-time vol), so the de-normalization at decode is a single shared scalar per decision bar.

**Distributional decode and de-normalization (recovering absolute moves/prices/quantiles).** The model outputs a *distribution* in vol-relative space (via MTP heads + temperature/top-p sampling and/or Monte-Carlo trajectories). To recover absolute, tradeable quantities, multiply back by the **same causal `sigma_t`** the target was built with — `sigma_t` is fully known at decode time `t`, so de-normalization is exact and adds no lookahead:

```
# vol-relative predicted sample / quantile from the model:
y_hat                          (1-step, vol-relative)
y_hat^{(h)}                    (h-horizon, vol-relative)

# de-normalize to absolute log-returns:
ret_hat_{t+1}      = y_hat        * (sigma_t + eps_vol)
R_hat_{t->t+h}     = y_hat^{(h)}  * ( sqrt(h) * sigma_t + eps_vol )

# absolute price point estimate / sampled path:
C_hat_{t+1}        = C_t * exp(ret_hat_{t+1})
C_hat_{t+h}        = C_t * exp(R_hat_{t->t+h})

# quantiles: because x -> sigma_t * x is a strictly increasing affine map for sigma_t > 0,
# the q-quantile maps directly (quantiles are scale-equivariant):
Quantile_q[ ret_{t+1} ]   = sigma_t * Quantile_q[ y ]
Quantile_q[ C_{t+1} ]     = C_t * exp( sigma_t * Quantile_q[ y ] )
```

Because the transform is a fixed positive scalar at decision time, the **entire predicted distribution** (every sampled trajectory, every quantile, the predictive mean and variance) de-normalizes by the same `sigma_t`. This preserves the distributional-output-by-contract requirement end to end: the downstream v2 meta-labeling/bet-sizing layer receives absolute, cost-comparable moves *with* their full predicted spread, and `sigma_t` is recorded alongside each prediction for replay.

**Eval-harness handoff.** The cost-aware NET-IR execution filter (the headline metric) operates on **de-normalized absolute** predicted moves: the magnitude filter "trade only if `|predicted move| > transaction-cost threshold`" must compare a real basis-point move against a real basis-point cost (0.1%–0.3%), so it consumes `ret_hat`/`R_hat` **after** multiplying by `sigma_t`, never the vol-relative `y_hat`. This is the explicit contract between this section and the eval harness: train in vol-relative space, threshold/PnL in absolute space.

---

### 5. Lookahead safety (CI-enforced extension of §6)

The single testable invariant from feature-spec §6 is extended to cover `sigma_t` and every vol-scaled quantity:

- **`sigma_t` uses only data with effective timestamp `<= t+1`** (bars `<= t`). The EWMA recursion folds in `r_t` *before* emitting `sigma_t`, and never reads `r_{t+1}`. The rolling form sums `r_{t-W+1..t}`. Both are pure functions of past-inclusive raw returns.
- **The target `y_{t->t+1}` divides the future label `ret_close_{t+1}` by the strictly-past `sigma_t`** — the future quantity is a *label*, the divisor is causal; this is correct and not a leak (labels are allowed to be future; *inputs and divisors* are not).
- **Segment resets** for the vol state mirror §4 exactly (no carry across a price gap), and vol warm-up sets the same mask/flag bits.
- **Extended truncation/perturbation test (merge gate):** for sampled `(symbol, t)`, recompute `(x_t, m_t)` *including any input-path vol-scaling* and the target `y_{t->t+1}` on (a) the full stream and (b) a stream truncated/perturbed at every point `>= t+1`. Assert `sigma_t`, the vol-scaled inputs in `x_t`, and `y_{t->t+1}` are **bit-identical** across truncation and adversarial perturbation of the future. Any dependence of `sigma_t` on bar `t+1` (the classic off-by-one of including the predicted bar in its own scaler) fails this gate. A complementary per-batch assertion re-derives `sigma_t` and confirms its computation graph contains no node fed by `r_{>=t+1}`.

---

### 6. Reproducibility & inference modes

- **Streaming mode (live):** carry the live EWMA `sigma_sq` state per symbol alongside the §3.2 normalization state; checkpoint and content-hash it so any historical prediction (and its de-normalizing `sigma_t`) replays bit-exactly.
- **Frozen-stats mode (published leaderboard):** use the rolling form (§1b) with the frozen-boundary convention of §3.4 so the published eval is a pure function of `(weights, frozen-stats table, raw test bars)`; the `sigma_t` series over the test window is itself a deterministic function of the raw test returns and ships content-hashed with the model card.
- Every run logs: vol estimator type, `lambda`/`W`, `eps_vol`, `n_warm_vol`, `vol_scale_inputs` flag, the per-horizon `sqrt(h)` (or calibrated `c_h`) multipliers, and the dataset content-hash. Given these, both the vol-scaling and its de-normalization are byte-reproducible.

**Net benefit restated.** Identical raw moves carry different information in different regimes; dividing by causal `sigma_t` collapses them onto one stationary, regime-relative axis that the FSQ tokenizer and AR backbone can model with their full capacity — while the exact, fully-causal `sigma_t` lets the eval harness and v2 bet-sizer recover absolute, cost-aware moves and their complete predicted distribution at decode time without a shred of lookahead.

---

## 4. Microstructure-Aware FSQ Tokenizer

This subsystem maps each post-normalization bar vector `x_t ∈ R^16` (Feature Spec §1) to a single discrete token via a small Transformer autoencoder with a **Finite Scalar Quantization (FSQ)** bottleneck. It replaces Kronos's BSQ quantizer with FSQ — the headline contribution — while preserving the coarse→fine hierarchy, the single-fused-token-per-bar contract, and bits-per-token parity for the 2×2 ablation. All shapes assume batch `B`, context `L ≤ 512`, feature dim `F = 16`.

**This section is the single source of truth for the canonical FSQ configuration.** The Training, Backbone, MTP, and Evaluation sections reference the values below and must not restate divergent numbers.

> **Canonical FSQ config (owned here):** `fsq_levels = [11, 9, 9, 7, 7, 5, 5]` (`D = 7` dims, **all odd**). `bpt = Σ log2(L_i) = 20.06 bits` (matches BSQ `k = 20` within 0.06 bit). Coarse group `{11,9,9}` → `bpt_coarse = 9.80`, `V_c = 891`. Fine group `{7,7,5,5}` → `bpt_fine = 10.26`, `V_f = 1225`. Total implicit vocab `= 891 · 1225 = 1,091,475 ≈ 2^20.06` (BSQ ref `2^20 = 1,048,576`). Reconstruction loss = **Huber (smooth-L1, δ=1.0)** for both `L_coarse` and `L_fine`. Tokenizer dropout default = **0.0**. The coarse/fine dim split is **derived** by the §d rule (not hardcoded) and stored in the manifest.

### 0. End-to-end signal flow and tensor shapes

```
x      : R^{B×L×16}     # normalized bar vectors (Feature Spec §3)
mask   : {0,1}^{B×L×16} # 1 = imputed/missing (Feature Spec §1,§4)
ts     : Z^{B×L}        # bar-open timestamps (for AR temporal embeds; NOT used by tokenizer)
  │
  ▼  Encoder  E_θ  (3-layer Transformer, d_model=256)
h      : R^{B×L×256}    # per-bar latent (contextual within window)
  │
  ▼  Latent→FSQ proj  W_in : 256 → D
z      : R^{B×L×D}      # pre-quant FSQ vector, D = 7 (canonical)
  │
  ▼  FSQ quantize  (bounded squash → per-dim round, L_i levels)
ẑ      : R^{B×L×D}      # quantized (STE), each ẑ[...,i] ∈ one of L_i levels
codes  : Z^{B×L×D}      # integer level indices, codes[...,i] ∈ {0..L_i-1}
token  : Z^{B×L}        # mixed-radix flatten of codes (single token per bar)
(cidx, fidx) : Z^{B×L} each   # coarse / fine sub-token indices (hierarchy, §d)
  │
  ▼  FSQ→Latent proj  W_out : D → 256
g      : R^{B×L×256}
  │
  ▼  Decoder  D_φ  (3-layer Transformer, mirrors encoder)
x_hat  : R^{B×L×16}     # reconstruction
```

The token stream `token ∈ Z^{B×L}` (and its `(cidx, fidx)` factorization) is the **sole interface to the AR backbone**. The AR backbone never sees `x` or `z` — only discrete tokens — exactly as in Kronos. The tokenizer is trained **stage-1 standalone** (reconstruction only); the AR backbone is stage-2 over frozen tokens.

---

### (a) Encoder — 3-layer Transformer, d_model=256

Per Kronos's tokenizer scale (a *small* autoencoder, not the AR backbone), the encoder is a shallow bidirectional Transformer over the length-`L` window. It is **non-causal within the window** — the tokenizer's job is faithful per-bar compression, and bar `t`'s *token* depends only on bar `t`'s features after the bottleneck (see causal note below); intra-window attention only sharpens the latent and is acceptable because tokenization is a deterministic stage-1 artifact, not a forecast.

**Config (defaults, rationale):**

| Param | Default | Rationale |
|---|---|---|
| `d_model` | 256 | Kronos tokenizer scale; ample for a 16-dim input, keeps the AE ≪ the AR backbone. |
| `n_layers` | 3 | Kronos tokenizer depth; shallow AE avoids over-smoothing the per-bar signal. |
| `n_heads` | 4 | `d_head = 64`, standard. |
| `d_ff` | 512 | 2× `d_model`. |
| `norm` | RMSNorm, Pre-LN | Parity with backbone conventions. |
| `pos` | RoPE | Parity; relative positions inside window. |
| `dropout` | **0.0** | A reconstruction autoencoder wants fidelity; dropout hurts it. Tune range `{0.0, 0.1}`. Training Stage-1 uses the **same** 0.0 default. |
| `activation` | SwiGLU | Standard modern FFN (tokenizer AE only; the AR backbone uses plain SiLU FFN for Kronos parity — see Backbone §1). |

**Input embedding.** The 16-dim bar plus its 16-bit mask are linearly embedded and summed; the mask is embedded so the AE knows which channels are structurally absent (spot funding/OI) vs. momentarily zero — identical philosophy to Kronos's missing-channel flag. Note: only the model-visible mask `m_t` is fed here (Data Pipeline / Feature Spec); the QA-only `is_stale`/`dq_flags` columns are never an input:

```python
# x: R^{B×L×16}, mask: {0,1}^{B×L×16}
e = x @ W_x        # W_x: R^{16×256}
e = e + mask @ W_m # W_m: R^{16×256}, mask embedding (learned)
e = e + bias_in    # R^{256}
h = TransformerEncoder(e)   # 3 layers, RoPE, Pre-LN → R^{B×L×256}
```

**Latent → FSQ projection.**

```python
z = h @ W_in + b_in    # W_in: R^{256×D}, z: R^{B×L×D}, D = 7
```

`D = 7` is the FSQ dimensionality (canonical config; derived in §c). No nonlinearity between `W_in` and the FSQ squash — the squash (`tanh`) is the only nonlinearity at the bottleneck, so the projection learns the raw coordinate to be quantized.

**Causal-safety note (tokenizer ≠ forecast).** The tokenizer reads bar `t`'s features, which the Feature Spec lookahead invariant already guarantees use only data with effective timestamp `≤ t+1`. Intra-window bidirectional attention means the *latent* `h_t` is contextual, but this **does not** leak future into a forecast: the AR backbone is trained on the *discrete token sequence* and predicts `token_{t+1}` from `token_{≤t}` causally; the token for bar `t` is a label, not a prediction. To make the published artifact maximally defensible, we additionally expose a config flag `encoder_causal=True` that masks the encoder to be causal (bar `t`'s token depends only on bars `≤ t`); default `False` (Kronos-style bidirectional AE), but we run the FSQ ablation under both and report that headline metrics are insensitive. This forecloses any reviewer objection that the tokenizer smuggles lookahead.

---

### (b) FSQ mechanism — bounded squash + per-dim rounding + STE

FSQ quantizes each latent **coordinate independently** to a small, fixed grid of `L_i` levels. There is no learned codebook, no embedding lookup, no nearest-neighbor search — the "codebook" is the *implicit Cartesian product* of per-dimension level sets.

**Forward.** For dimension `i` with level count `L_i`:

```
# 1. Bounded squash into a fixed interval. half = (L_i - 1)/2
s_i      = half_i                                  # = (L_i-1)/2
z_sq[i]  = s_i * tanh(z[i])                         # ∈ (-s_i, s_i)  (bounded; no commitment needed)
# 2. Round to nearest integer level (grid points at -s_i, …, -1, 0, 1, …, s_i for odd L_i)
ẑ[i]     = round(z_sq[i])                           # ∈ {-s_i, …, s_i}
# 3. Integer code index in {0 … L_i-1}
codes[i] = ẑ[i] + s_i                               # shift to non-negative ints
```

For **odd** `L_i` the grid is symmetric and includes `0` (a natural "center"/level-0 used by the coarse-only decode in §d). **The canonical config uses all-odd levels for exactly this reason** — the coarse-only decode pins each fine dim to its level-0 center cell, which exists only when `L_i` is odd. **Even levels are forbidden everywhere in the project** (no 8s, no 6s): an even grid has no center cell and the hierarchical `L_coarse` reconstruction would be undefined.

**Straight-through estimator (exact forward/backward).** Rounding has zero gradient a.e., so we pass the gradient straight through the `round` while keeping the forward value exact:

```
forward:   ẑ[i] = z_sq[i] + stop_grad( round(z_sq[i]) − z_sq[i] )
backward:  ∂ẑ[i]/∂z_sq[i] = 1                       # round treated as identity in backward
```

The `tanh` squash **is** differentiated normally (it is not inside the stop-grad), so the full backward through the bottleneck is:

```
∂L/∂z[i] = ∂L/∂ẑ[i] · ∂z_sq[i]/∂z[i]
         = ∂L/∂ẑ[i] · s_i · (1 − tanh²(z[i]))      # STE: ∂ẑ/∂z_sq = 1, chain through tanh
```

In code:

```python
def fsq_quantize(z, levels):           # z: R^{B×L×D}, levels: list[int] len D, all odd
    half = (torch.tensor(levels) - 1) / 2          # R^{D}
    z_sq = half * torch.tanh(z)                    # bounded squash, R^{B×L×D}
    z_round = torch.round(z_sq)
    z_hat = z_sq + (z_round - z_sq).detach()       # STE: exact forward, identity backward
    codes = (z_round + half).to(torch.long)        # R^{B×L×D} in {0..L_i-1}
    return z_hat, codes
```

**Why FSQ needs NO commitment loss.** In VQ/BSQ the encoder output `z` and the chosen code `e` are *distinct learnable objects*: the commitment term `β‖z − sg(e)‖²` (and codebook term `‖sg(z) − e‖²`) exist solely to drag the continuous encoder output toward its discrete code and to move the codebook toward the encoder, because nothing else couples them. In FSQ **the code is a deterministic, parameter-free function of `z`** (`round(tanh(z))`): there is no separate codebook vector to pull toward, and the STE already routes the reconstruction gradient back into `z`. The only objective is reconstruction; the encoder is implicitly pushed to place `z_sq` near grid points purely because off-grid placement incurs the (rounding-induced) reconstruction error it must minimize. Hence **`L = L_coarse + L_fine`, with no `λ·L_quant` term** — the project's stated simplification holds exactly.

**Why FSQ has NO codebook-collapse mode.** Codebook collapse in VQ/BSQ is the failure where a learned embedding table has most entries never selected (dead codes) because nearest-neighbor assignment + the moving codebook concentrates usage on a few vectors; it requires tricks (EMA codebook, code reset, commitment tuning) to mitigate. FSQ has **no learned codebook to collapse**: every grid point in the Cartesian product `∏_i {0..L_i-1}` is reachable by construction, and the squash `tanh` guarantees the encoder *can* address the full bounded range. Usage may still be *non-uniform* (some grid cells rarely visited — a data property), but no code is structurally unreachable and there is no assignment/codebook tug-of-war to destabilize. We monitor usage as a *health diagnostic* (§f), not as a failure mode to be engineered around. **Bounded quantization error** is preserved exactly as Kronos valued in BSQ: since `|z_sq − ẑ| ≤ 0.5` per dimension after rounding, the per-dim quantization error is bounded by half a grid step, and the squash bounds the addressable range — giving FSQ the same outlier-robustness motivation (a bounded error) that drove Kronos to BSQ, with strictly fewer moving parts.

---

### (c) Dimension / level derivation — empirical sweep at fixed bits-per-token

**The control variable is bits-per-token (bpt), not integer vocab.** We do **not** force vocab `2^20`. FSQ is designed for *few* dimensions (`D < 10`) each with a *small* `L_i`. The bits-per-token is

```
bpt = Σ_i log2(L_i)          # = log2(|implicit codebook|)
```

We fix a **target bpt** and search the level configuration that minimizes reconstruction error on the 16-dim bar at that budget. Vocabulary is a *consequence* of the chosen levels and is **never** a parameter-padding lever — enlarging vocab to hit a round backbone parameter count is forbidden because it breaks the ablation's capacity control (see Backbone §2 and §g below).

**Target bpt.** Kronos BSQ uses `k = 20` bits (coarse 10 + fine 10). For ablation parity we set the **primary target `bpt* = 20`** so FSQ and BSQ are matched on information capacity. The empirical search restricts candidates to **bpt ∈ [19.5, 20.5]** (within the `±0.5`-bit BSQ tolerance). We additionally sweep coarser budgets `bpt ∈ {16, 18, 20}` only to chart the fidelity/capacity curve — these are diagnostic, not parity points.

**Candidate configs — every `bpt = Σ log2(L_i)` recomputed exactly; all odd-level:**

| Config (`L_1..L_D`) | `D` | `bpt = Σ log2 L_i` | Implicit vocab | Notes |
|---|---|---|---|---|
| `[7,5,5,5,5]` | 5 | 12.10 | 4,375 | Low-budget diagnostic anchor (off-parity). |
| `[9,7,7,5,5,5]` | 6 | 15.75 | 55,125 | Mid-budget diagnostic (off-parity). |
| `[7,7,7,7,7,5,5]` | 7 | 18.68 | 420,175 | Below parity band; flat (low-`L`, more dims). |
| `[9,9,9,7,7,5,5]` | 7 | **19.77** | 893,025 | **In-band; conservative alternate (FSQ ≤ BSQ).** |
| `[11,9,9,7,7,5,5]` | 7 | **20.06** | 1,091,475 | **PRIMARY PARITY POINT — canonical config.** |
| `[9,9,9,9,7,5,5]` | 7 | **20.13** | 1,148,175 | **In-band; valid alternate.** |
| `[7,7,7,7,5,5,5,5]` | 8 | 20.52 | 1,500,625 | At the upper edge of the band; flatter (8 dims). |

We restrict `D ∈ [5, 9]` (FSQ's "few dims" regime) and `L_i ∈ {5,7,9,11}` (odd, small). The number of in-band level-config candidates is small enough to grid-search exhaustively. **The primary parity point is `[11,9,9,7,7,5,5]` (`bpt = 20.06`, 0.06 bit above BSQ's 20 — within tolerance).** If a strictly-conservative FSQ ≤ BSQ comparison is desired, `[9,9,9,7,7,5,5]` (`19.77`) is the in-band fallback.

**Selection procedure (sweep):**

1. For each in-band candidate, train the AE to convergence (or a fixed short budget, e.g. 50k steps, identical for all candidates) on a held-out *fixed* train slice.
2. **Selection criterion = reconstruction error on the 16-dim bar**, reported as both **per-feature MAE** and **Huber loss** on a held-out month, *plus* a **finance-weighted MAE** that up-weights the features the downstream eval cares about most (`ret_close`, `range`, `TFI`, `funding_rate`, `d_oi`) by a config weight vector `w_feat` (default: 2× weight on those 5, 1× on the rest). Rationale: a config that reconstructs `log_volume` perfectly but smears `ret_close` is useless for forecasting.
3. **Tie-breakers, in order:** (i) lower per-feature MAE on the price/return block; (ii) better codebook health (§f) — higher perplexity / lower dead-cell rate; (iii) smaller `D` (fewer dims = simpler hierarchy, less residual decay risk).
4. **Per-stage residual check (gate):** the chosen config must NOT exhibit residual magnitude decay (§e) beyond threshold; if it does, either add learnable per-stage scaling (§e) or move bits from fine→coarse.

**Output of this stage:** a frozen `levels` list (canonical `[11,9,9,7,7,5,5]`, `D=7`, `bpt=20.06`) **and the derived coarse/fine grouping** (§d), both content-hashed into the dataset/config manifest so any token stream is reproducible. The coarse/fine split is the *derived* `{11,9,9}` / `{7,7,5,5}` — it is computed by the §d rule and stored, never hardcoded as a fixed dim count.

---

### (d) Coarse→fine hierarchy via dimension grouping

Kronos's hierarchy factorizes a 20-bit token into a coarse 10-bit + fine 10-bit subtoken and trains `L_coarse` (coarse-alone rough reconstruction) + `L_fine` (full-token high fidelity). FSQ has **no monolithic `2^k` codebook to bit-split**; instead we **partition the `D` FSQ dimensions into a coarse group `C` and a fine group `F`** (disjoint, `C ∪ F = {0..D-1}`). Coarse dims carry the high-`L_i` levels (most information), fine dims the residual detail.

**Grouping rule (derived, not hardcoded).** Sort dims by `L_i` descending; choose the split index `k` that **minimizes `|bpt_coarse − bpt_fine|`** (matching Kronos's balanced 10/10 split as closely as the integer levels allow). For the canonical config `[11,9,9,7,7,5,5]` this yields:

```
C = {11,9,9}   → bpt_coarse = log2(11)+log2(9)+log2(9) = 3.46+3.17+3.17 = 9.80
F = {7,7,5,5}  → bpt_fine   = log2(7)+log2(7)+log2(5)+log2(5) = 2.81+2.81+2.32+2.32 = 10.26
```

This yields a coarse sub-token vocab `V_c = ∏_{i∈C} L_i = 11·9·9 = 891` and fine sub-token vocab `V_f = ∏_{i∈F} L_i = 7·7·5·5 = 1225` — both close to Kronos's 1024-per-subtoken embedding tables (coarse slightly under, fine slightly over), so the AR backbone's hierarchical embeddings are sized `891 / 1225`. The derived split `(coarse_dims=3 {11,9,9}, fine_dims=4 {7,7,5,5})` is stored in the manifest; **YAML must not hardcode a 3/3 (or any fixed) dim count** — it records the rule-derived split alongside the frozen levels. The mixed-radix flatten producing `cidx`, `fidx`, and the global `token`:

```python
def split_codes(codes, coarse_idx, fine_idx, levels):
    # codes: Z^{B×L×D}
    cidx = mixed_radix_flatten(codes[..., coarse_idx], [levels[i] for i in coarse_idx])  # Z^{B×L}, in {0..890}
    fidx = mixed_radix_flatten(codes[..., fine_idx],   [levels[i] for i in fine_idx])    # Z^{B×L}, in {0..1224}
    return cidx, fidx     # consumed by the AR backbone's hierarchical heads
```

`(cidx, fidx)` map **directly** onto Kronos's `p(b_t|b_<t) = p(b^c_t|b_<t)·p(b^f_t|b_<t, b^c_t)` factorization and the intra-block cross-attention (query = embedding of the *sampled* coarse prediction). The tokenizer's only job is to make `cidx` a *coarse* summary and `fidx` a *refinement* — enforced by the hierarchical loss below.

**Coarse-only decode (fine dims pinned to center / level-0).** Because we use **odd levels**, every fine dim has a well-defined center grid point `ẑ_i = 0` (`codes_i = (L_i−1)/2`). Coarse-only reconstruction pins all fine dims to that center and decodes:

```python
def coarse_only_decode(z_hat, fine_idx):
    z_c = z_hat.clone()
    z_c[..., fine_idx] = 0.0          # fine dims → level-0 center (odd-level guarantee)
    return decoder(z_c @ W_out + b_out)   # x_hat_coarse : R^{B×L×16}
```

This is the FSQ analogue of "decode from the coarse subtoken alone" and is what forces an information hierarchy: the coarse dims must alone yield a usable rough bar.

**Hierarchical reconstruction loss (Huber; NO commitment term).**

```
x_hat_coarse = coarse_only_decode(ẑ, F)          # fine dims pinned to center
x_hat        = decoder(W_out·ẑ + b_out)          # full token (coarse + fine)

L_coarse = Huber( x_hat_coarse, x )              # smooth-L1, δ=1.0, masked
L_fine   = Huber( x_hat,        x )              # smooth-L1, δ=1.0, masked
L        = L_coarse + L_fine                     # NO  λ·L_quant  — FSQ commitment-free
```

- **Huber (smooth-L1, `δ=1.0`)** for **both** terms — chosen over MSE for outlier robustness (heavy-tailed crypto returns) on the `[-5,5]`-clipped normalized scale. Training Stage-1 uses the **identical** Huber objective.
- **Masking:** the per-feature loss is multiplied by `(1 − mask)` so structurally-absent channels (spot funding/OI, warm-up zeros) contribute zero gradient — the AE is not penalized for failing to reconstruct fields that were never present. A small `eps` guards all-masked bars.
- **Per-feature weighting `w_feat`** (same vector as §c) applied inside both Huber terms so price/return fidelity dominates.
- **Coarse weight schedule (optional):** start with `L = 2·L_coarse + L_fine` for the first ~10% of steps then anneal `→ L_coarse + L_fine`, to bootstrap a strong coarse code before the fine group specializes (prevents the fine group from greedily absorbing all signal early). Tune the warm weight `∈ {1.5, 2, 3}`.

**Forward-pass pseudocode (full tokenizer, stage-1).**

```python
def tokenizer_forward(x, mask, levels, coarse_idx, fine_idx, w_feat):
    # levels = [11,9,9,7,7,5,5] (canonical); coarse_idx → {11,9,9}, fine_idx → {7,7,5,5}
    # ---- ENCODE ----
    e = x @ W_x + mask @ W_m + bias_in            # R^{B×L×256}
    h = encoder(e)                                # 3-layer Transformer → R^{B×L×256}
    z = h @ W_in + b_in                           # R^{B×L×D}, D=7

    # ---- QUANTIZE (FSQ) ----
    z_hat, codes = fsq_quantize(z, levels)        # STE; R^{B×L×D}, Z^{B×L×D}
    cidx, fidx   = split_codes(codes, coarse_idx, fine_idx, levels)

    # ---- DECODE ----
    g          = z_hat @ W_out + b_out            # R^{B×L×256}
    x_hat      = decoder(g)                       # 3-layer Transformer → R^{B×L×16}
    z_c        = z_hat.clone(); z_c[..., fine_idx] = 0.0
    x_hat_crs  = decoder(z_c @ W_out + b_out)     # coarse-only reconstruction

    # ---- LOSS (Huber, no commitment) ----
    keep = 1.0 - mask
    L_fine   = weighted_huber(x_hat,     x, keep, w_feat, delta=1.0)
    L_coarse = weighted_huber(x_hat_crs, x, keep, w_feat, delta=1.0)
    L = L_coarse + L_fine

    return L, dict(x_hat=x_hat, codes=codes, cidx=cidx, fidx=fidx, z=z, z_hat=z_hat)
```

At inference / token-export the quantize step is run forward-only (no STE needed numerically — `round` is exact) and only `(token, cidx, fidx)` are emitted to the AR backbone.

---

### (e) Residual magnitude decay — monitoring + per-stage learnable scaling

**The pathology.** Naive hierarchical/residual FSQ can leave the **fine group underused**: after the coarse dims capture the bulk of the signal, the residual the fine dims must encode has small magnitude, so `tanh(z_fine)` operates in its near-linear small region and the rounded fine codes collapse toward level-0 (center) → fine dims become quasi-dead, the hierarchy degenerates to coarse-only, and effective bpt drops below the configured 20.06 budget.

**Monitoring (logged every eval):**

1. **Per-group reconstruction contribution.** Define the *marginal fine gain*:
   ```
   Δ_fine = MAE(x_hat_coarse, x) − MAE(x_hat, x)      # how much the fine group improves recon
   ```
   Report `Δ_fine` per feature and aggregate. A healthy hierarchy has `Δ_fine` clearly > 0 (fine group is doing work). `Δ_fine ≈ 0` ⇒ fine group is dead ⇒ residual decay.
2. **Per-dim pre-quant magnitude.** Track `RMS(z_sq[i])` and `RMS(z[i])` per dimension. Fine dims with `RMS(z_sq[i]) ≪ 0.5` (stuck inside one grid cell around the center) are decaying.
3. **Per-dim level entropy / occupancy** (the §f metrics, but split coarse vs fine). Fine dims with collapsed level distributions (entropy → 0, occupancy concentrated on level-0) are the smoking gun.
4. **Per-stage code-change rate.** Fraction of bars where any fine code ≠ its center level. If this → 0, fine is unused.

**Selection gate:** the §c config search rejects any config where `Δ_fine < τ_fine` (default `τ_fine = 0.05·MAE(x_hat,x)` — fine group must explain ≥5% of remaining error) or where any fine dim's level entropy `< 0.3·log(L_i)`.

**The fix — per-stage learnable scaling.** Insert a **learnable per-dimension scale** `γ_i > 0` between the projection and the squash, so the network can amplify a small-magnitude residual coordinate *into* the squash's nonlinear, multi-cell region before rounding:

```
z_sq[i] = half_i · tanh( γ_i · z[i] )            # γ_i learnable, init 1.0, softplus-parameterized > 0
```

Mechanism: if the fine residual is small, gradient descent grows the corresponding `γ_i`, stretching `z_fine` across the `tanh` range so rounding lands on distinct levels again — directly counteracting residual magnitude decay. `γ_i` is parameterized `γ_i = softplus(ρ_i)` (`ρ_i` the raw param, `γ_i > 0` always) and regularized lightly (`1e-4·Σ(log γ_i)²`) to discourage runaway scaling. A symmetric learnable scale on the **decoder side** is optional (`W_out` already absorbs per-dim scale, so a dedicated decoder `γ'_i` is off by default). Per-stage means coarse-group `{11,9,9}` and fine-group `{7,7,5,5}` dims each get their own `γ_i` vector; the fine group's `γ` is the one that grows under decay.

**Escalation if scaling is insufficient:** rebalance the bit budget — move one level/dim from coarse→fine (e.g. `[11,9,9,7,7,5,5]` → `[9,9,9,7,7,5,5]`, dropping coarse capacity from 9.80 to 9.51 bits so the coarse group leaves more residual for the fine group), then re-run the §c selection. The learnable `γ` is the cheap first-line fix; bit-rebalance is the structural fallback. Any rebalanced config must stay in the `[19.5, 20.5]` bpt band.

---

### (f) Codebook health metrics

Even without a learned codebook, FSQ usage statistics over the **implicit** code space are first-class diagnostics (logged per eval epoch over a fixed held-out batch of `N_bars`, default `1e6` bars). Computed both **per-dimension** (over each dim's `L_i` levels) and **globally** (over the product space, via the flattened `token`).

| Metric | Definition | Healthy target |
|---|---|---|
| **Per-dim usage %** | fraction of the `L_i` levels of dim `i` visited ≥ once | 100% for small `L_i`; flag any dim < 100% |
| **Per-dim entropy** | `H_i = −Σ_l p_{i,l} log p_{i,l}` (nats) over level frequencies | `> 0.5·log(L_i)`; near `log(L_i)` ideal |
| **Per-dim perplexity** | `PPL_i = exp(H_i)` | close to `L_i` |
| **Global perplexity** | `exp(−Σ_c p_c log p_c)` over observed tokens `c` | high fraction of `1,091,475` (capacity actually used) |
| **Effective bpt** | `log2(global perplexity)` | ≈ configured `20.06` |
| **Active-token count** | distinct `token` values observed in data | (reported; `∏L_i = 1,091,475` is the ceiling) |
| **Coarse vs fine entropy split** | `H_i` averaged over `C={11,9,9}` vs over `F={7,7,5,5}` | both groups non-degenerate (ties to §e) |
| **Center-cell occupancy (per fine dim)** | `p_{i, level-0}` | not → 1 (would mean fine dim dead) |

**Key reframing vs BSQ:** because FSQ cannot have *structurally* dead codes (every grid point is reachable — §b), the relevant health signal is **effective bpt** (`= log2(global perplexity)`): if it tracks the configured `20.06`, the tokenizer is using its full information budget; if it sits well below, the data simply doesn't fill the space (fine — bounded, expected) *unless* the gap is concentrated in the fine group, which then triggers the §e residual-decay fix. The dead-code rate that plagues BSQ is reported for the **BSQ ablation arm** and is expected to be **0 for FSQ by construction** — itself a result to report (the "no codebook-collapse mode" claim, empirically demonstrated).

---

### (g) Bits-per-token control — matching BSQ on information capacity

The 2×2 ablation `{BSQ, FSQ} × {OHLCV-only, +microstructure}` must be matched on **information capacity, not integer vocab**. **The control variable is bpt, and it is held fixed at ≈20 across both arms. Vocabulary is a consequence of the chosen levels — it is NOT a free parameter-padding lever**, because that would break the capacity control on which the entire FSQ-vs-BSQ comparison rests:

```
BSQ arm:  k = 20 bits  → bpt_BSQ = 20.00  (Kronos: 10 coarse + 10 fine; vocab 1024/1024)
FSQ arm:  bpt_FSQ = Σ_i log2(L_i) = 20.06 for [11,9,9,7,7,5,5]  (vocab 891/1225)
```

- We do **not** demand `bpt_FSQ == 20.000` (FSQ levels are integers, so bpt is a sum of `log2(odd)` and lands on non-integers). We demand `|bpt_FSQ − bpt_BSQ| ≤ 0.5 bit` and **report the exact bpt of each arm** in every table, so the comparison is explicitly capacity-controlled. The canonical config's surplus is `20.06 − 20.00 = 0.06 bit` — negligible, and well inside tolerance. A strictly-conservative FSQ ≤ BSQ run uses `[9,9,9,7,7,5,5]` (`19.77`, 0.23 bit *below* BSQ).
- **Coarse/fine split also matched:** BSQ is 10/10 (`bpt_coarse = bpt_fine = 10`); the FSQ coarse/fine groups are derived so `bpt_coarse = 9.80`, `bpt_fine = 10.26` (§d) — both within ~0.3 bit of 10, so the AR backbone's hierarchical factorization sees comparable sub-token capacities in both arms (`V_c = 891 ≈ 1024`, `V_f = 1225 ≈ 1024`).
- **The microstructure leg** (OHLCV-only vs +microstructure) is controlled **independently** by the input feature dimension at *identical bpt*: the OHLCV-only arm uses `F = 7` (5 price/shape + 2 volume/liquidity — the **Kronos-equivalent subset**, matching the Feature Spec and Eval §C.1), vs the full `F = 16` microstructure vector. Both arms quantize to the **same** `[11,9,9,7,7,5,5]` config (same bpt), so any fidelity/forecast delta is attributable to the extra 9 microstructure channels, not to capacity. This cleanly separates the two mechanically-independent legs of the single claim.
- **Report contract:** every ablation row carries `(quantizer, bpt_exact, F, D, levels, V_c, V_f)` so a reviewer can verify capacity parity at a glance — e.g. `(FSQ, 20.06, 16, 7, [11,9,9,7,7,5,5], 891, 1225)`. bpt and the levels are content-hashed into the run manifest.

---

### (h) Decoder — mirror of the encoder

The decoder is the structural mirror of the encoder: `D → 256` projection (`D = 7`), a 3-layer Transformer (same `d_model=256`, `n_heads=4`, `d_ff=512`, RMSNorm/Pre-LN/RoPE/SwiGLU), then a linear head `256 → 16`.

```python
def decoder(g):                     # g: R^{B×L×256}  (= W_out·ẑ + b_out)
    d = TransformerDecoderStack(g)  # 3 layers, SAME bidirectional config as encoder
    x_hat = d @ W_dec + b_dec       # W_dec: R^{256×16} → R^{B×L×16}
    return x_hat
```

- **Bidirectional within window**, mirroring the encoder (the decoder reconstructs the whole window jointly; this is reconstruction, not generation — the AR backbone is the generative model).
- **No output activation** — features are on the normalized `[-5,5]` scale; the linear head predicts them directly, the Huber loss handles the scale.
- **Bounded-feature heads (optional):** for features the Feature Spec bounds (`body, wicks ∈ [0,1]/[-1,1]`, `TFI, signed_count_imbalance ∈ [-1,1]`), an optional `tanh`/`sigmoid`-bounded output head can be enabled (config `bounded_heads=True`, default `False`) so reconstructions respect domain bounds; default off to keep parity with Kronos's plain linear decoder and avoid a second nonlinearity confounding the FSQ-vs-BSQ comparison.
- **Decoder per-dim scale `γ'_i`** (§e) is off by default; `W_out` already learns per-dim scaling.

**Symmetry guarantee for the ablation:** the BSQ arm reuses the **identical** encoder/decoder/3-layer config and the same `D→256` / `256→16` projections; only the bottleneck (`fsq_quantize` ↔ `bsq_quantize`) and its loss term (`L_coarse + L_fine`, both Huber ↔ `L_coarse + L_fine + λ·L_quant`) differ, at matched bpt (`20.06` vs `20.00`). This guarantees that any measured difference is attributable to the quantizer, not to AE capacity or information budget — the crux of the controlled comparison that proves the headline claim.

---

## 5. Autoregressive Backbone (decoder-only, Kronos_small class ~21–22M)

The backbone is a causal decoder-only Transformer over the per-bar token stream emitted by the FSQ tokenizer. Each bar `t` is represented by a hierarchical **(coarse, fine)** subtoken pair `(b^c_t, b^f_t)`; the backbone autoregresses over bars and factorizes the per-bar joint as `p(b_t | b_{<t}) = p(b^c_t | b_{<t}) · p(b^f_t | b_{<t}, b^c_t)`. Config is locked to **exact Kronos_small** for clean baseline parity: it is the *fixed substrate* against which the 2×2 {BSQ,FSQ}×{OHLCV,+microstructure} ablation isolates the tokenizer claim. This section owns the backbone trunk and the two same-timestep heads only; the **MTP depth heads are a separate section** and consume the published hidden state `h_t` defined in §9.

The two FSQ sub-vocab sizes are **owned by the Tokenizer section** (single source of truth). The canonical config is `fsq_levels = [11, 9, 9, 7, 7, 5, 5]` (D=7 dims, all odd), with the derived coarse group `{11,9,9}` → `V_c = 891` (9.80 bits) and fine group `{7,7,5,5}` → `V_f = 1225` (10.26 bits), total **20.06 bits-per-token** (matches the BSQ k=20 reference within 0.06 bit). This section references those numbers and does not restate divergent ones; the parameter count below is pinned to `V_c=891, V_f=1225`.

---

### 1. Locked configuration

| Symbol | Value | Meaning |
|---|---|---|
| `n_layers` | 8 | decoder blocks |
| `d_model` | 512 | residual-stream width |
| `d_ff` | 1024 | FFN inner width (2× `d_model`; Kronos_small uses a narrow 2× FFN, not 4×) |
| `n_heads` | 8 | attention heads |
| `d_head` | 64 | `d_model / n_heads` |
| `L_max` | 512 | context length in **bars** (one token per bar) |
| `act` | SiLU FFN (plain 2-matrix; SwiGLU optional, off by default) | gated SwiGLU is a config flag for Kronos parity (see §1.1) |
| `norm` | RMSNorm | Pre-LN placement |
| `pos` | RoPE | applied to Q,K per head; θ_base = 10000 |
| `attn` | FlashAttention-2 | causal, single triangular mask |
| dropouts | ffn 0.25, resid 0.25, attn 0.1, token 0.1 | Kronos_small values |

The vocab sizes are fixed by the tokenizer to `V_c = 891`, `V_f = 1225` (canonical config `[11,9,9,7,7,5,5]`). Let:
- `V_c` = coarse sub-vocab size (= product of coarse-group FSQ levels `{11,9,9}` = 891), `k_c = log2(V_c) = 9.80` bits.
- `V_f` = fine sub-vocab size (= product of fine-group FSQ levels `{7,7,5,5}` = 1225), `k_f = log2(V_f) = 10.26` bits.
- Bits-per-token `K = k_c + k_f = 20.06` is **the controlled variable** of the BSQ-vs-FSQ ablation, held fixed at ≈20 across both arms. The backbone code is written in terms of `V_c, V_f` so the same code serves any tokenizer config, but for v1 these are pinned to the canonical parity point. **Vocabulary is NOT a parameter-padding lever** (see §2): enlarging it to hit a round 27M would change capacity and break the ablation control, so it is forbidden.

#### 1.1 Plain SiLU FFN vs SwiGLU — note
Kronos_small as published uses a standard 2-matrix FFN with a SiLU/GELU pointwise nonlinearity. To hold exact Kronos parity, the backbone uses the same **plain 2-matrix FFN** (`W_in: 512→1024`, SiLU, `W_out: 1024→512`) by default — this is what the §5 block pseudocode and the §2 parameter count both assume. A SwiGLU variant (`d_ff` shrunk to `≈683` to stay param-matched) is a config flag, **off by default** to preserve baseline parity.

---

### 2. Parameter breakdown (honest count; lands ~21.3M)

All counts for `d_model=512`, `d_ff=1024`, `n_layers=8`, `V_c=891`, `V_f=1225`. Biases off on all projections except norms (RMSNorm has a learnable gain only). Counts in parameters.

**Per-block (×8):**

| Component | Formula | Count |
|---|---|---|
| Attn QKV proj | `3 · d_model²` | 786,432 |
| Attn out proj | `d_model²` | 262,144 |
| FFN in (`W_in`) | `d_model · d_ff` | 524,288 |
| FFN out (`W_out`) | `d_ff · d_model` | 524,288 |
| 2× RMSNorm gain | `2 · d_model` | 1,024 |
| **Block total** | | **2,098,176** |

`8 × 2,098,176 = 16,785,408` for the stack.

**Embeddings / heads / shared:**

| Component | Formula | Count |
|---|---|---|
| Coarse subtoken embed | `V_c · d_model = 891·512` | 456,192 |
| Fine subtoken embed | `V_f · d_model = 1225·512` | 627,200 |
| Fusion `W_fuse` | `2·d_model · d_model` | 524,288 |
| Temporal embeddings (5 tables, §4) | `(1440+24+7+31+12) · d_model` | 775,168 |
| Coarse head `W_c` (untied) | `d_model · V_c = 512·891` | 456,192 |
| Fine cross-attn block (§6) | `4·d_model² + RMSNorm gain = 1,048,576 + 512` | 1,049,088 |
| Fine head `W_f` (untied) | `d_model · V_f = 512·1225` | 627,200 |
| Final RMSNorm | `d_model` | 512 |
| **Subtotal** | | **4,515,840** |

**Grand total = 16,785,408 + 4,515,840 = 21,301,248 ≈ 21.30M.**

This is the realized count, reported truthfully. The cross-attention term is pinned exactly to `4·d_model² = 1,048,576` plus the fine-head RMSNorm gain `512` (no `≈`/`+small` fudges). The `~27M` figure is **nominal** — it denotes the "Kronos_small class," and landing at ~21–24M is fine and is stated honestly here. The backbone code must print the realized count at init and assert it falls in `[20M, 24M]`.

**Resolving the param-vs-bpt tension.** Bits-per-token is held **fixed at ≈20** (`V_c=891`, `V_f=1225`) because it is the controlled variable of the ablation; the vocabulary is therefore *not* a free knob for padding parameters toward a round number. If padding toward the upper band of the Kronos_small class is ever desired, it must be done with a **non-vocab knob** that does not change capacity — e.g. replacing the single linear `W_fuse` with a 2-layer MLP fusion (`2·d_model→d_model→d_model`, adds `512·512 = 262,144` per extra layer) — and this must be stated explicitly in the config. Enlarging the vocab to inflate params is forbidden.

> **Param-parity rule for the ablation:** BSQ and FSQ runs MUST be matched on total backbone params to ±2%. The BSQ arm uses `V_c=V_f` chosen so its `k_c=k_f=10` (`V_c=V_f=1024`, K=20.00); the FSQ arm uses the canonical `V_c=891, V_f=1225` (K=20.06). The only param-bearing difference between arms is `(V_c, V_f)` in embeds+heads; with K matched to within 0.06 bit the embedding/head param counts already agree to <2% (FSQ: 456,192+627,200+456,192+627,200 = 2,166,784 for the four `V×d` tables; BSQ: 4·1024·512 = 2,097,152 — a ~0.07M-of-21M difference (69,632 params), ~0.3% of total). Log the realized count for both arms; if either drifts past ±2%, equalize with the non-vocab `W_fuse` knob, never by retuning vocab.

---

### 3. Input embedding and fusion (`W_fuse`)

Inputs per training window: coarse ids `b^c ∈ Z^{B×L}`, fine ids `b^f ∈ Z^{B×L}`, open-timestamps `ts ∈ Z^{B×L}` (for temporal embeds, §4).

```
E_c = CoarseEmbed[b^c]                    # (B, L, d_model)
E_f = FineEmbed[b^f]                       # (B, L, d_model)
E_tok = W_fuse @ concat([E_c, E_f], -1)    # (B, L, d_model);  W_fuse: 2*d_model -> d_model
E_tok = TokenDropout(E_tok, p=0.10)        # Kronos token dropout, train only
```

`W_fuse` linearly mixes the two subtoken embeddings into one bar embedding so the backbone sees **one token per bar** (locked decision: single fused token per bar). Concatenation-then-projection (vs summation) lets the model learn an asymmetric coarse/fine weighting rather than forcing equal scale. **Token dropout** (p=0.10) zeros entire fused bar embeddings at random during training only — Kronos's input regularizer; disabled at inference/eval.

> **Shared with MTP (by reference).** This fused path — `CoarseEmbed`, `FineEmbed`, and `W_fuse` — **is** the per-bar token embedding `Emb(b)` that the MTP section consumes. MTP reuses these exact three modules by reference (no separate embedding table), which is why MTP carries **zero marginal embedding parameters** and why the §2 count above is the complete embedding cost for the whole model. The MTP section states the same.

**Temporal embeddings are summed in next (§4), then a residual dropout pass, then the stack.**

---

### 4. Learnable temporal embeddings

Five learnable lookup tables, keyed off the **bar-open timestamp** `ts` (deterministic, trivially causal — they are a pure function of `t`, never of any future datum). All five summed into the fused bar embedding:

| Table | Cardinality | Key derivation from `ts` (UTC) |
|---|---|---|
| minute-of-day | 1440 | `(ts // 60000) % 1440` |
| hour-of-day | 24 | `(ts // 3_600_000) % 24` |
| day-of-week | 7 | `weekday(ts)` (0=Mon) |
| day-of-month | 31 | `day(ts) - 1` |
| month-of-year | 12 | `month(ts) - 1` |

```
E_time = MinuteEmb[min_of_day(ts)] + HourEmb[hour(ts)]
       + DowEmb[dow(ts)] + DomEmb[dom(ts)] + MoyEmb[moy(ts)]   # (B, L, d_model)
x_0 = ResidDropout(E_tok + E_time, p=0.25)                      # (B, L, d_model) -> stack input
```

minute-of-day (1440) and hour-of-day (24) are intentionally **both** present (Kronos parity); they are not redundant — hour gives a coarse-grained bucket the optimizer can lean on early, minute the fine resolution. Crypto trades 24/7 so no market-session masking; UTC is canonical (Binance server time).

**Causal-safety:** temporal embeddings depend only on `ts` of bar `t`, which is known at bar-open. They introduce no lookahead.

---

### 5. Decoder block (Pre-LN, RMSNorm, RoPE, FlashAttention-2, causal)

Each of the 8 blocks, residual stream `x ∈ R^{B×L×d_model}`:

```
# --- Self-attention sublayer (Pre-LN) ---
h  = RMSNorm_attn(x)                                  # (B, L, d)
q  = h @ W_q;  k = h @ W_k;  v = h @ W_v              # each (B, L, d); reshape -> (B, n_heads, L, d_head)
q, k = RoPE(q, seq_pos), RoPE(k, seq_pos)            # rotary on Q,K per head; V untouched
a  = FlashAttention2(q, k, v, causal=True,            # (B, n_heads, L, d_head)
                     dropout_p=0.1 if training else 0)
a  = merge_heads(a) @ W_o                              # (B, L, d)
x  = x + ResidDropout(a, p=0.25)                      # residual add

# --- FFN sublayer (Pre-LN), plain 2-matrix SiLU FFN ---
h  = RMSNorm_ffn(x)                                   # (B, L, d)
f  = (silu(h @ W_in)) @ W_out                          # (B, L, d_ff) -> (B, L, d); FFN dropout 0.25 inside
x  = x + ResidDropout(f, p=0.25)
```

After the stack: `h_final = RMSNorm_final(x)` → this is the published hidden state (§9).

- **Pre-LN + RMSNorm:** normalization *before* each sublayer (stable gradients at depth-8), RMSNorm (no mean-centering, learnable gain only) for the small compute win and Kronos parity.
- **RoPE:** rotary applied to Q and K per head, base θ=10000; positions are the **in-window sequence index** `0..L-1` (relative order of bars within the sample), not absolute calendar time — absolute time enters via the §4 temporal embeddings. RoPE gives relative-position generalization and clean extrapolation toward `L_max=512`.
- **Causal masking:** strictly lower-triangular; bar `t` attends to `≤ t` only. FlashAttention-2 with `causal=True` materializes the triangular mask internally (no explicit `L×L` mask tensor → no O(L²) memory). This causal mask is the **architectural enforcement of the no-lookahead contract** at the sequence level (the feature pipeline enforces it at the per-bar level).
- **FlashAttention-2:** imported (allowed infra), used for the fused causal attention kernel; fp16/bf16 compute, fp32 softmax accumulation. Falls back to a reference `torch.scaled_dot_product_attention(is_causal=True)` path when FA2 is unavailable (CI/CPU), asserted bit-close in tests.

---

### 6. Hierarchical subtoken prediction — coarse head, then fine head via intra-block cross-attention

Per bar, given trunk output `h_t = h_final[:, t, :]` (already conditioned causally on all `b_{<t}`):

#### 6.1 Coarse head
```
logits_c[t] = h_t @ W_c            # (B, L, V_c=891)
p_c[t]      = softmax(logits_c[t]) # distribution over coarse sub-vocab
```
`p(b^c_t | b_{<t})` — the coarse subtoken depends only on the causal past. Plain linear head over the trunk state.

#### 6.2 Fine head via intra-block cross-attention (sampled-coarse query)
The fine subtoken is conditioned on the **same-timestep coarse subtoken**, realizing the factorization `p(b^f_t | b_{<t}, b^c_t)`. A small cross-attention block does this:

```
# 1. obtain the coarse token to condition on:
if training:
    b^c_sample[t] = sample(p_c[t])            # SAMPLED from the predicted dist (NOT teacher-forced)  [see 6.3]
else:
    b^c_sample[t] = sample(p_c[t]) or argmax  # inference: actually-emitted coarse token
q_fine[t] = CoarseEmbed_q[b^c_sample[t]]      # (B, L, d_model); query = embedding of the SAMPLED coarse pred
                                              # (separate small embedding table, or reuse CoarseEmbed)

# 2. cross-attend: query = sampled-coarse embedding, key/value = backbone hidden state h_t
#    single-head (or n_heads) cross-attention, query length 1 per position, K=V = h_t
q  = q_fine[t]            @ Wq_x               # (B, L, d)
k  = h_t                  @ Wk_x              # (B, L, d)   key/value = backbone hidden state
v  = h_t                  @ Wv_x              # (B, L, d)
c  = CrossAttn(q, k, v)   @ Wo_x              # (B, L, d)   per-position (no cross-bar mixing; q_t attends to its own h_t)
h_fine[t] = RMSNorm(h_t + c)                  # fuse coarse-conditioning into the fine state
logits_f[t] = h_fine[t] @ W_f                 # (B, L, V_f=1225)
p_f[t]      = softmax(logits_f[t])            # p(b^f_t | b_<t, b^c_t)
```

The cross-attention is **intra-block / per-position**: at position `t` the query (sampled-coarse embedding for bar `t`) attends to the backbone hidden state at bar `t` only — it injects "which coarse cell did we land in" into the fine prediction without leaking other timesteps (those are already in `h_t` causally). `Wq_x, Wk_x, Wv_x, Wo_x ∈ R^{d×d}` plus the fuse-step RMSNorm gain → the `4·d_model² + 512 = 1,049,088` params in §2.

#### 6.3 Exposure-bias rationale: sample, don't teacher-force
The fine query uses the **sampled** coarse prediction `b^c_sample ~ p_c`, **not** the ground-truth coarse token, during training. Rationale: at inference the fine head must condition on the coarse token the model *actually emitted*, which is sampled and can be wrong. If trained only on ground-truth coarse (teacher forcing), the fine head never sees the distribution of its own coarse mistakes → **exposure bias**: train/inference mismatch where errors compound. Feeding the sampled coarse token at train time matches the train and inference conditioning distributions, so the fine head learns to be robust to imperfect coarse predictions. (This mirrors scheduled-sampling / Kronos's exact choice.) The coarse **loss target** is still the ground-truth coarse id; only the *conditioning input to the fine head* is sampled. Gradient does not flow through the discrete `sample()` (stop-grad on the sampled id; the coarse head is trained purely by its own NLL).

> **Config knob:** `coarse_sample_prob` schedule — optionally anneal from teacher-forced (prob 0, early training stability) to fully-sampled (prob 1). Default: **fully sampled from step 0** for Kronos parity; the schedule is available if early-training instability appears (selection: monitor fine-head NLL divergence in first 2k steps).

---

### 7. Training loss (this section: coarse + fine NLL only)

Per bar, standard next-token NLL on both subtokens; the backbone trunk is shared:

```
L_coarse = - (1/N) Σ_t log p(b^c_t = y^c_t | b_<t)          # CE over V_c=891
L_fine   = - (1/N) Σ_t log p(b^f_t = y^f_t | b_<t, b^c_t)    # CE over V_f=1225, fine head conditioned on SAMPLED coarse
L_backbone = L_coarse + L_fine
```

- Equal weighting (1:1) by default — Kronos parity; the tokenizer's own coarse→fine *reconstruction* hierarchy (Huber loss) lives in the tokenizer section, not here. A `λ_fine` knob exists (search `{0.5,1,2}`, select by fine-token top-1 accuracy) but defaults to 1.
- Targets `y^c_t, y^f_t` are the tokenizer-emitted ground-truth subtoken ids for bar `t` (teacher-forced *targets*; the only place sampling enters is the fine head's *conditioning input*, §6.3).
- Label smoothing **off** by default (parity); optional `0.0–0.1`.
- **MTP loss is added by the MTP section** on top of this; `L_total = L_backbone + λ_mtp · L_mtp`. The backbone exposes `h_final` (§9) for that; `λ_mtp` and the depth-chain losses are owned there.
- Vol-scaling of *targets* does not touch the backbone token NLL — the targets here are discrete subtoken ids from the tokenizer. Vol-relative regression targets live in the MTP/target-builder section per the project contract.

**Optimizer (Kronos_small parity):** AdamW, LR `1e-3`, weight decay `0.01`, cosine decay with linear warmup over ~10% of steps (~15k warmup). β=(0.9, 0.95), grad-clip 1.0. bf16 autocast, DDP, seed-pinned. (Full schedule/DDP details owned by the training-loop section; restated here only for the backbone-relevant hyperparams.)

---

### 8. Inference: KV-cache; MLA explicitly NOT used

Autoregressive generation, one bar per step. Two-level per-bar sampling matches the factorization:

```
for t in range(prompt_len, prompt_len + horizon):
    h_t = backbone_step(token_{t-1}, ts_t, kv_cache)   # uses + updates KV-cache; O(1) attn per new bar
    b^c_t ~ softmax(h_t @ W_c / T)            with top-p     # sample coarse
    q     = CoarseEmbed_q[b^c_t]
    h_f   = RMSNorm(h_t + CrossAttn(q, K=h_t, V=h_t))
    b^f_t ~ softmax(h_f @ W_f / T)            with top-p     # sample fine, conditioned on the sampled coarse
    token_t = fuse(b^c_t, b^f_t)              # detokenize via tokenizer decoder downstream
```

- **KV-cache:** per layer, cache `K,V ∈ R^{B×n_heads×t×d_head}` (grown to `≤ L_max=512`); each new bar appends one row and attends over the cache → linear-time generation. RoPE applied to the new Q,K at the current position; cached K already carry their rotary phase. Standard, sufficient.
- **Sampling controls:** temperature `T` and top-p (nucleus) per Kronos; forecasting defaults `T≈0.6, top-p≈0.9`, with **Monte-Carlo averaging of N≈10 trajectories** for distributional output (every forecast is a full distribution, per project contract — point estimates forbidden). The 5/15/60-min forecasts are produced by **autoregressive rollout** of this loop; MTP heads provide native dense draft only for the first few bars (≤t+D) and serve as the speculative drafter that accelerates the rollout — they are a *separate* mechanism layered on this same `h_t`.

#### 8.1 MLA is NOT used — and why
**Multi-head Latent Attention is deliberately omitted.** MLA exists to compress the KV-cache when its memory dominates — a problem that appears at long context and large `d_model`. Here `L_max=512`, `n_heads=8`, `d_head=64`, `n_layers=8`: the full bf16 KV-cache is `2 · 8 · 512 · 8 · 64 · 2 bytes ≈ 8.4 MB` per sequence — negligible. MLA would add latent-projection complexity, break exact Kronos parity (contaminating the ablation), and buy nothing. **Decision: plain multi-head attention + standard KV-cache.** Revisit only at base-class scale / long-context (v3 roadmap). Likewise, no architecture-level latency micro-optimization here — sub-50ms serving is a serving-layer concern (TorchScript/ONNX/batching), kept out of the model.

---

### 9. Clean interface to MTP (the published hidden state `h_t`)

The backbone's contract to every downstream consumer (MTP depth heads, future meta-labeling, probes):

```
class BackboneOutput:
    h_final:   Tensor  # (B, L, d_model=512)  post-final-RMSNorm trunk state; h_t = causal summary of bars <= t
    logits_c:  Tensor  # (B, L, V_c=891)       coarse logits (this section's head)
    logits_f:  Tensor  # (B, L, V_f=1225)      fine logits  (this section's head, conditioned on sampled coarse)
    kv_cache:  Optional[list]  # per-layer (K,V) for incremental inference
```

- **`h_final[:, t, :]` is the single object the MTP section consumes.** It is the depth-0 trunk state — a causal summary of all bars `≤ t` — from which MTP's causal-chain depth heads predict bars `t+1, t+2, …` up to `t+D` (default D=4, DeepSeek-V3 style; each depth conditioning on the previous depth). The MTP depths therefore provide **native dense supervision/draft only for horizons ≤ t+D (≈5 min at D=4)**; the 5/15/60-min eval horizons are delivered by **autoregressive rollout** (§8) with MTP acting as speculative drafter, not by direct depth-60 heads (a non-starter). The MTP section owns: the depth-head parameters, the sequential depth chain, the rollout-target construction (incl. vol-scaling), the speculative-decoding path, and `λ_mtp`.
- **MTP reuses the backbone's embedding by reference.** MTP's per-bar token embedding `Emb(b)` is exactly this section's fused path (`CoarseEmbed + FineEmbed + W_fuse`, §3), shared by reference — zero marginal embedding parameters, as accounted in §2.
- **Invariant the backbone guarantees to MTP:** `h_final[:, t, :]` depends only on bars `≤ t` (causal mask, §5) — MTP heads may be attached without re-checking lookahead at the trunk level; their *own* multi-step rollout targets must satisfy the causal-target rules in the vol-scaling/target sections.
- The two same-timestep heads (`logits_c`, `logits_f`) stay in **this** section; MTP does not re-predict bar `t`, it predicts `t+1…t+D`. No parameter sharing is required between the fine cross-attn head and MTP heads (kept separate for a clean ablation: MTP can be toggled off and the backbone+heads remain a valid Kronos-parity AR model).

---

### 10. Tensor-shape summary (one forward pass, training)

| Stage | Shape |
|---|---|
| `b^c`, `b^f`, `ts` | `(B, L)` each |
| `E_c`, `E_f`, `E_time`, `x_0` | `(B, L, 512)` |
| per-block `q/k/v` | `(B, 8, L, 64)` |
| `h_final` | `(B, L, 512)` |
| `logits_c` | `(B, L, 891)` |
| `q_fine` (sampled-coarse embed) | `(B, L, 512)` |
| `logits_f` | `(B, L, 1225)` |
| `L_coarse`, `L_fine`, `L_backbone` | scalar |

`B` = batch (e.g. 64–256 windows), `L ≤ 512` bars. Compute is bf16 autocast in the stack, fp32 master weights, fp32 softmax/loss reduction.

---

## 6. Multi-Token Prediction (MTP) Heads for Multi-Horizon Forecasting

> **Status: SECONDARY contribution.** MTP is *not* the headline of this work — the microstructure-aware FSQ tokenizer is. MTP is folded in as the multi-horizon *training-signal densifier* and an inference accelerator, inherited-and-adapted from DeepSeek-V3, never claimed as novel. It is configured so the 2×2 `{BSQ, FSQ} × {OHLCV, +microstructure}` ablation can run with MTP **disabled** (`D=0`, pure next-token loss) for clean baseline parity against Kronos_small, and with MTP **enabled** as the production path. No paper claim rests on it.

> **Horizon scope up front (read this before §3).** The `D` MTP depths provide **native dense supervision/draft only for horizons up to `t+D` (default `D=4` → ≤ ~5 min)**. The 5/15/60-minute forecasts are **not** native MTP outputs — they are produced by **autoregressive rollout** of the AR model, with MTP acting as the speculative drafter (§3, §5–6). Direct depth-60 heads are a non-starter. Nothing in this section promises a native 60-minute (or 15-minute) head.

### 1. Setting and notation

Each bar `t` is one fused token produced by the FSQ tokenizer, factorized into a coarse subtoken `b^c_t` and a fine subtoken `b^f_t` (Kronos-inherited hierarchy; the FSQ section is the single source of truth for the levels, vocab sizes, and coarse/fine split — see §LOCKED FSQ config below). The AR backbone is the exact **Kronos_small** config: 8 layers, `d_model = 512`, `d_ff = 1024`, 8 heads, plain 2-matrix SiLU FFN (SwiGLU off by default for Kronos parity), RoPE + learnable temporal embeddings, 512-token context. Write the backbone's final hidden state at position `t` as

```
h_t ∈ R^{d_model}          d_model = 512
```

`h_t` is the standard causal representation that already predicts bar `t+1` in the baseline (Kronos) model. MTP adds `D` extra *depths* on top of this same `h_t`, each predicting one further future bar via a sequential chain. Depth `0` is the **main** next-token prediction (the existing Kronos head); depths `1 … D` are the auxiliary MTP heads.

**Canonical FSQ quantities (owned by the Tokenizer section; restated here only for shape bookkeeping, not redefined):** `fsq_levels = [11, 9, 9, 7, 7, 5, 5]` (D=7 dims, all odd), bits-per-token `= Σ log2(L_i) = 20.06` (the BSQ k=20 control point, within ±0.5 bit). Coarse group `{11,9,9}` → coarse vocab `V_c = 891`; fine group `{7,7,5,5}` → fine vocab `V_f = 1225`. MTP predicts these same `(b^c, b^f)` tokens at further offsets; it introduces **no new vocab** and is forbidden from being used as a vocab/param-padding lever.

Throughout: `B` = batch, `L` = sequence length (`≤ 512`), `V_c = 891`, `V_f = 1225`, `E` = shared token-embedding dim fed to the backbone (`E = d_model = 512`).

---

### 2. (a) Structure — `D` sequential MTP depths with a causal chain

#### 2.1 Per-depth module

Each MTP depth `k ∈ {1, …, D}` is a **single lightweight Transformer block** `MTPBlock_k` (one Pre-LN attention + one FFN sub-layer, RMSNorm, RoPE, same head count and same plain-SiLU `d_ff` as the backbone so it composes cleanly), plus a per-depth linear projection that fuses the previous depth's representation with the embedding of the bar that depth is conditioned on. This mirrors DeepSeek-V3 exactly: a shared trunk, then a per-depth `(RMSNorm‖concat‖linear‖Transformer block)` stack, with the **embedding layer and the output head shared across all depths and with the main model.**

Define the depth-`k` representation `H^{(k)} ∈ R^{B × L × d_model}`. Base case:

```
H^{(0)}_t = h_t                                  # backbone final hidden state, predicts bar t+1
```

Recursion (the **causal chain** — depth `k` conditions on depth `k-1`, *not* parallel-independent):

```
# 1. embed the ground-truth (teacher-forced, training) token of bar t+k, the bar depth k-1 just predicted
e^{(k)}_t   = Emb( b_{t+k} )            ∈ R^{B × L × E}     # SHARED backbone fused embedding (see §2.3)

# 2. fuse previous-depth representation with that embedding
z^{(k)}_t   = W_k · [ RMSNorm(H^{(k-1)}_t) ‖ RMSNorm(e^{(k)}_t) ]   # W_k ∈ R^{d_model × 2·d_model}, concat then project

# 3. one Transformer block over the sequence (causal mask, RoPE, temporal embeds reused)
H^{(k)}     = MTPBlock_k( z^{(k)} )    ∈ R^{B × L × d_model}

# 4. predict bar t+k+1 from H^{(k)}_t via the SHARED output head (§2.3)
logits^{(k)}_t = Head( H^{(k)}_t )                  # distribution over bar (t+k+1)
```

So at each position `t`, the model emits `D+1` predictions in **one forward pass**: depth 0 → bar `t+1`, depth 1 → bar `t+2`, …, depth `D` → bar `t+D+1`. The chain is causal in *depth*: `H^{(k)}` literally cannot be computed without `H^{(k-1)}`, which is the property that makes the auxiliary signal a coherent short rollout rather than `D` independent regressors. At the default `D=4` the depths span bars `t+1 … t+5` (i.e. native supervision out to ~5 min only).

> **Critical causal-safety rule (training).** The teacher-forced embedding `e^{(k)}_t = Emb(b_{t+k})` is fine in *training* (we have the labels). But the loss for depth `k` at position `t` predicts bar `t+k+1`, which means **position `t` may only be used as a training example if bars `t+1 … t+k+1` all lie inside the same contiguous segment** (the feature-spec segment discipline — no straddling a price-gap boundary). Positions within `D+1` bars of a segment's right edge are **masked out of the MTP loss** (their depth-`k` labels would either not exist or belong to a different segment). This is enforced per-segment, per-depth, and is part of the lookahead invariant (§6 below).

#### 2.2 Hierarchical (coarse→fine) prediction inside each depth

Every depth predicts a *full bar token*, which under our tokenizer is `(b^c, b^f)`. We reuse Kronos's intra-block factorization at **every** depth:

```
p^{(k)}(b_{t+k+1} | ·) = p(b^c_{t+k+1} | H^{(k)}_t) · p(b^f_{t+k+1} | H^{(k)}_t, b^c_{t+k+1})
```

The fine subtoken is conditioned on the **sampled** coarse prediction via the same small intra-block cross-attention used in the backbone (query = embedding of the sampled coarse code, key/value = `H^{(k)}_t`), preserving the exposure-bias fix. The cross-attention module is **shared** across depths (it is part of `Head`, see §2.3), so MTP does not duplicate it.

#### 2.3 What is shared vs. what is new

| Component | Shared with main model? | Per-depth copy? |
|---|---|---|
| Token embedding `Emb` — **the backbone's fused path** `CoarseEmbed[b^c] ⊕ FineEmbed[b^f] → W_fuse·concat(·)` | **Shared by reference** | No |
| Output head `Head` (coarse logits + fine cross-attn + fine logits) | **Shared** | No |
| Backbone (8-layer Kronos_small trunk) | Shared (produces `h_t`) | No |
| `MTPBlock_k` (1 Transformer block) | No | **Yes, `D` of them** |
| Fusion projection `W_k ∈ R^{d_model × 2d_model}` + 2 RMSNorms | No | **Yes, `D` of them** |

**`Emb(b)` is exactly the backbone's fused embedding pipeline**, used by reference: it is `E_c = CoarseEmbed[b^c]`, `E_f = FineEmbed[b^f]`, then `E_tok = W_fuse · concat([E_c, E_f])` ∈ R^{512} (the same `CoarseEmbed`, `FineEmbed`, `W_fuse` parameters defined in the Backbone section). MTP does **not** instantiate a second embedding table. Sharing `Emb` and `Head` by reference is the whole reason the overhead is small and stays aligned with the main objective (DeepSeek-V3's design intent), and it is what makes the **zero-marginal-embedding-param** accounting in §2.5 exact.

#### 2.4 Shapes (forward pass)

```
backbone:        x_tok ∈ Z^{B×L×2}  (coarse,fine ids) → Emb(=CoarseEmbed⊕FineEmbed→W_fuse) → R^{B×L×512}
                 → trunk → h ∈ R^{B×L×512}
per depth k:     e^{(k)} = Emb(b_{t+k}) ∈ R^{B×L×512}, H^{(k-1)} ∈ R^{B×L×512}
                 concat → R^{B×L×1024} → W_k → R^{B×L×512} → MTPBlock_k → H^{(k)} ∈ R^{B×L×512}
                 Head(H^{(k)}) → coarse logits R^{B×L×V_c}, fine logits R^{B×L×V_f}   # V_c=891, V_f=1225
outputs:         {logits^{(0)}, …, logits^{(D)}}, each (coarse R^{B×L×891}, fine R^{B×L×1225})
```

#### 2.5 Parameter overhead

Per MTP depth = one Transformer block + one fusion projection. With `d_model = 512`, `d_ff = 1024`, 8 heads, plain 2-matrix SiLU FFN (Kronos-parity, no gating):

- Attention (`W_q,W_k,W_v,W_o`): `4 · 512² = 1,048,576`
- FFN (`d_model→d_ff→d_model`, plain 2-matrix): `2 · 512 · 1024 = 1,048,576`
- Fusion `W_k` (`512 × 1024`): `524,288`
- RMSNorm gains / norms: `~1.5k` (4 × 512), negligible

**= 2,621,440 params per depth (≈ 2.62M, exact, no fudge).** Because `Emb` and `Head` are **shared by reference** (§2.3), they add **zero** marginal cost. The Kronos_small backbone itself realizes **~21.30M** params (honest re-count, owned by the Backbone section, with `V_c=891`/`V_f=1225`); "~27M" is *nominal* ("Kronos_small class") and the realized base landing ~21–24M is reported truthfully. MTP depths are an additive, separately-reported configuration on top:

| `D` | Added params (= `D · 2,621,440`) | Total (base 21.30M + added) | Overhead vs base |
|---|---|---|---|
| 0 (baseline / ablation parity) | 0 | ~21.30M | 0% |
| 2 | 5,242,880 (~5.24M) | ~26.54M | ~+25% |
| 4 (default) | 10,485,760 (~10.49M) | ~31.79M | ~+49% |
| 6 | 15,728,640 (~15.73M) | ~37.03M | ~+74% |
| 8 (max) | 20,971,520 (~20.97M) | ~42.27M | ~+99% |

> The added depths are **dropped at the end of training for the pure-forecasting / Kronos-parity baseline** (you keep only depth 0), so the published headline parameter count for the FSQ-vs-BSQ ablation is the clean ~21.30M base; the MTP depths are an optional, separately-reported configuration. **Vocabulary is the controlled variable of the ablation and is held fixed at `V_c=891`/`V_f=1225` (~20 bpt); enlarging vocab to pad params is forbidden.** If padding toward the upper Kronos_small band is ever desired, use a **non-vocab** knob (e.g. a 2-layer `W_fuse` MLP), not the codebook. Default **`D = 4`** (search range below) is the recommended trade-off: meaningful densification of the ≤5-min training signal and a usable speculative-decode draft length at ~+49% params.

---

### 3. (b) Resolving the horizon-granularity tension

**The tension.** Each MTP depth advances exactly **one 1-minute bar**. The project requires 1/5/15/60-minute horizons. Naively that would demand `D = 60` for the 60-minute horizon — `60 × 2.62M ≈ 157M` params of auxiliary heads on a ~21M backbone, with the well-documented MTP failure that signal quality decays sharply with depth (each depth conditions on an increasingly stale, increasingly self-generated representation). That is impractical and statistically weak. **Direct depth-60 heads are a non-starter.** We resolve it as follows.

**Primary scheme (chosen): modest dense MTP for training/decoding + AUTOREGRESSIVE ROLLOUT for the long horizons.**

1. **Set `D = 4` (default; `D ∈ {2,4,6,8}` tunable).** The MTP depths provide *dense next-bar supervision* (depth `k` is trained to predict bar `t+k+1`, a richer gradient than single-step next-token) **only out to ~5 min**, and a *draft of length `D`* for speculative decoding (§5). They are **not** the mechanism that delivers the 15- or 60-minute number.

2. **The 5/15/60-minute forecasts are produced by autoregressive rollout** of the depth-0 model (with the MTP depths optionally serving as the speculative drafter to make the rollout cheap, §5). To forecast horizon `H ∈ {5,15,60}` minutes, the model samples a trajectory of `H` consecutive 1-minute bars and the horizon target is the **aggregated** quantity over that path. Because every forecast is a full **distribution** (project contract), we roll out `N` Monte-Carlo trajectories (§6) and read the horizon distribution off the rollouts:

   ```
   target_H(t) = Σ_{i=1}^{H} ret_close(t+i)          # H-bar cumulative log-return (vol-scaled per the feature spec)
   ```

   The horizon-`H` predictive distribution is the empirical distribution of `target_H` over the `N` sampled rollouts. `H_max = 60` is the rollout/label look-forward (it is what drives the eval purge), **not** an MTP depth. This composes natively with the distributional-output contract and the cost-aware NET-IR eval (the execution filter thresholds on `|predicted move|` at the *horizon* the trade is held for).

3. **Why rollout is the primary mechanism (justification).**
   - **Statistical:** a single AR model trained on dense, vol-scaled 1m moves already encodes the conditional dynamics; the `H`-step distribution is the *correct* marginal of that process. Direct `H`-step heads (depth = H) would each see far fewer effective gradient updates per parameter, suffer the residual/representation decay MTP is known for at large depth, and cannot produce a *path* (only a smeared endpoint) — losing the within-horizon microstructure the tokenizer exists to model.
   - **Faithful to the eval contract:** the headline metric is cost-aware NET-IR under an execution filter; that needs a *distribution of the realized P&L path*, which rollout gives and a point `H`-step head does not.
   - **Parameter-economical:** `D=4` costs ~+49% params; `D=60` is a non-starter on a ~21M model.

4. **The trade-off (stated plainly).** Rollout pays in **inference compute** (an `H`-step, `N`-trajectory rollout is `H·N` forward steps) and is exposed to **compounding sampling error** over long horizons (errors accumulate over 60 steps). MTP **directly mitigates both**: it is the speculative drafter that collapses the `H·N` forward cost (§5), and the dense multi-depth (≤5-min) training signal regularizes the 1-step conditional so each rollout step is better-calibrated, reducing compounding. So the two mechanisms are complementary: **MTP makes the rollout fast and well-conditioned; rollout makes the long horizons reachable without `D=60`.**

**Secondary, optional scheme (config flag `aggregated_targets=True`, off by default): horizon-specific aggregated heads.** For the *specific* set `{5,15,60}` only, one may attach a tiny extra head off `h_t` whose label is the aggregated `target_H` directly (a regression-style auxiliary, or a quantized-`target_H` classification head). This gives a one-shot point/quantile readout for latency-critical serving without a full rollout. It is kept **off by default** because (i) it breaks the "always a full path-distribution" contract (it predicts the endpoint, not the path) and (ii) it adds horizon-specific heads that the clean ablation does not need. When enabled it is purely additive and never replaces the rollout distribution; it is reported separately. (Note: these are not "MTP depths" — they are auxiliary aggregate-target heads.)

> **Decision:** primary = **`D=4` dense MTP (native ≤5-min training signal + spec-decode draft) + autoregressive rollout for the 5/15/60-min horizons**. Direct horizon heads are an opt-in convenience, not the path; per-horizon depth heads at `t+15`/`t+60` are explicitly rejected.

---

### 4. (c) MTP loss and its combination with the main loss

#### 4.1 Per-depth loss

Each depth's loss is the **same hierarchical coarse→fine cross-entropy** used for the main token (so MTP supervision is identical in form to the main objective — only the prediction offset differs). For depth `k`, position `t`, predicting bar `τ = t+k+1`:

```
L^{(k)}_t = CE( p(b^c_τ | H^{(k)}_t),  b^c_τ )                          # coarse  (over V_c = 891)
          + CE( p(b^f_τ | H^{(k)}_t, b^c_τ),  b^f_τ )                   # fine, teacher-forced coarse for the loss term (over V_f = 1225)
```

(The coarse used to condition the fine *logits in the loss* is the ground-truth `b^c_τ`; the *sampled*-coarse cross-attention path is used at inference, matching Kronos's exposure-bias treatment. This is the standard teacher-forced-loss / sampled-inference split, applied per depth. This token-level CE is unrelated to the tokenizer's Stage-1 reconstruction loss, which is Huber and owned by the Tokenizer/Training sections.)

Mask: `L^{(k)}_t` contributes **only if** position `t` and target bar `τ=t+k+1` are valid (in-segment, not warm-up, not within `D+1` of segment edge — §2.1). Let `M^{(k)}_t ∈ {0,1}` be that validity mask.

#### 4.2 Depth aggregation and combination with the main next-token loss

Depth 0 is the main next-token loss `L_main = L^{(0)}` (this is *exactly* the Kronos training objective — backward-compatible). The auxiliary MTP loss is the **depth-weighted mean** of depths `1…D`:

```
L_MTP = (1/D) · Σ_{k=1}^{D}  λ_k · ( Σ_t M^{(k)}_t · L^{(k)}_t / Σ_t M^{(k)}_t )

L_total = L_main  +  μ · L_MTP
```

- **`μ`** = global MTP weight (DeepSeek-V3's `λ`). Default **`μ = 0.3`**; tune `μ ∈ {0.1, 0.3, 0.5}`. Selection criterion: largest `μ` that does **not** degrade depth-0 next-token validation perplexity vs. the `μ=0` baseline (MTP must remain an *auxiliary regularizer*, never trade away the primary objective — consistent with its SECONDARY status). DeepSeek-V3 itself anneals its MTP weight down late in training; we expose an optional linear decay `μ: 0.3 → 0.1` over the final 20% of steps (config flag, default off; turn on if late-training perplexity regresses).
- **`λ_k`** = per-depth weight. Default **uniform `λ_k = 1`** (the `1/D` already normalizes). Optional decaying schedule `λ_k = γ^{k-1}`, `γ ∈ {1.0, 0.8, 0.5}`, to down-weight the noisier deep heads; selection criterion: per-depth validation CE should be monotone non-decreasing in `k` (sanity that deeper = harder) and the chosen `λ_k` should not let `L_MTP` be dominated by the deepest, highest-variance head. The `1/D` normalization keeps `L_MTP` on a comparable scale to `L_main` so `μ` is interpretable.
- **Vol-scaling interaction:** the *targets* the MTP heads predict are the vol-scaled, regime-relative tokens (the tokenizer already encodes vol-scaled inputs/targets per the feature-spec causal hook); MTP inherits this with no change — it just predicts more of the same tokens at further offsets.

> **Ablation guard:** setting `D=0` (equivalently `μ=0`) recovers `L_total = L_main`, the bit-exact Kronos training objective, so the headline FSQ-vs-BSQ ablation is run in that mode for parity. MTP-on is a separate, clearly-labeled configuration.

---

### 5. (d) MTP as speculative decoding at inference

The trained MTP depths double as a **self-drafting speculative decoder** (DeepSeek-V3's stated secondary use), giving the autoregressive rollout (§3) its speed.

**Mechanism (one accept/verify cycle):**

1. **Draft.** From the current verified context ending at bar `t`, run one forward pass. The backbone + the `D` MTP depths emit a chain of `D+1` proposed bars `(b̂_{t+1}, b̂_{t+2}, …, b̂_{t+D+1})` — sampled greedily or at the rollout temperature. This is the *draft*, produced for free alongside the main prediction because the depths are already computed.
2. **Verify.** Run the backbone *once* over the drafted prefix (a single batched forward over `t+1 … t+D+1`) to get the model's *true* next-bar distribution at each drafted position. Compare each draft token to the verified distribution using the standard speculative-decoding acceptance rule (accept `b̂_{t+i}` with prob `min(1, p_verify/p_draft)`; on first rejection, resample that bar from the residual and stop).
3. **Commit.** Accept the longest verified prefix (length `1 … D+1`), append to context, repeat. Worst case = 1 bar/cycle (identical output distribution to plain AR — speculative decoding is **distribution-exact**, never changes what is sampled, only how fast).

**Why this matters here.** The `{5,15,60}`-minute rollout needs `H` sequential bars per trajectory. Speculative decoding amortizes that: with mean acceptance length `a ∈ [1, D+1]`, the expected forward passes drop from `H` to `≈ H/a` per trajectory. Empirically MTP draft acceptance is high on the short, locally-predictable 1m horizon, so `D=4` typically yields `a ≈ 2–3`, a 2–3× rollout speedup — directly serving the sub-50ms serving target *at the serving layer* (we do **not** touch the architecture for latency; speculative decoding is an inference-time decode policy, not an arch change).

**Causal-safety:** drafting/verifying only ever reads bars `≤` the position being predicted; the verify pass uses the same causal mask. No lookahead is introduced. Distribution-exactness means the published, reproducible eval is **identical** with or without speculative decoding (it's a pure speed optimization — we assert byte-equivalence of sampled trajectories under fixed seed with spec-decode on/off as a CI check).

---

### 6. (e) Monte-Carlo trajectory sampling and how the two interact

Full distributional output (project contract: *every forecast is a distribution, never a point*) is delivered by **Monte-Carlo trajectory sampling**, exactly as Kronos: sample `N` independent rollout trajectories with temperature `T` + top-p nucleus sampling at each bar, then read horizon statistics off the ensemble.

**Procedure for horizon `H`:**

```
for n in 1..N:
    traj_n = rollout(context, steps=H, temperature=T, top_p=p)     # H sampled 1m bars (spec-decoded)
    target_H[n] = aggregate(traj_n)                                # e.g. Σ vol-scaled ret_close over H bars
P_H = empirical_distribution({ target_H[n] : n=1..N })             # full predictive distribution at horizon H
```

From `P_H` we read the point forecast (mean/median), dispersion (the *confidence* a v2 bet-sizing layer will consume), and quantiles. **Defaults (Kronos-aligned):** `T ≈ 0.6`, `top_p ≈ 0.9`, `N ≈ 10` for forecasting; tune `T ∈ [0.4,1.0]`, `N ∈ {5,10,20,30}` with selection on held-out cost-aware NET-IR (the headline metric) — more trajectories sharpen the distribution at linear cost. Multi-scale (5m/15m) spine, if used, samples on its own token stream the same way.

**How MTP and Monte-Carlo interact (orthogonal axes):**

- **Monte-Carlo = the distribution axis (what).** It governs how many trajectories and how much exploration (`T`, `top_p`, `N`) — it *produces* the distributional output. This is **unchanged** by MTP and **always available**.
- **MTP/speculative decoding = the speed axis (how fast).** It accelerates the generation of *each individual trajectory* without altering its sampling distribution (distribution-exact, §5). So you run the **same `N` MC trajectories you would have run anyway**, each ~`a×` cheaper.

Concretely, the production inference loop is: *for each of `N` Monte-Carlo trajectories, roll out `H` bars using MTP speculative decoding to accept multiple bars per forward pass.* The two compose multiplicatively in cost (`N · H/a` forward passes) and **do not interfere statistically** — speculative decoding guarantees the accelerated trajectory is drawn from the identical distribution as a plain AR trajectory at the same `(T, top_p, seed)`. If maximal fidelity is wanted (or to validate), speculative decoding can be switched off and the same `N`-trajectory MC runs on plain AR with identical output distribution, only slower. The MTP-drafted bars are **never** used directly as the forecast distribution (drafts are biased toward the greedy chain); the forecast distribution is **always** the verified Monte-Carlo ensemble. MTP only decides *acceptance*, never the final sample.

---

### 7. Interfaces (implementation-ready)

```python
# Training forward
def forward_mtp(tok_ids:    LongTensor[B, L, 2],     # (coarse, fine) ids per bar; V_c=891, V_f=1225
                ts:         LongTensor[B, L],         # bar-open timestamps (for temporal embeds)
                seg_id:     LongTensor[B, L],         # segment id (for in-segment / edge masking)
                D:          int
               ) -> dict:
    # Emb is the backbone fused path (CoarseEmbed+FineEmbed+W_fuse), shared by reference.
    # returns logits per depth:
    #   logits[k] : (coarse: FloatTensor[B, L, 891], fine: FloatTensor[B, L, 1225]) for k in 0..D
    #   each predicts bar (t+k+1); depth 0 == main next-token head; native span = t+1 .. t+D+1
    ...

# Loss
def mtp_loss(logits, tok_ids, seg_id, D,
             mu: float = 0.3, lambda_k: Tensor = None  # default ones(D)
            ) -> dict:
    # builds per-(depth,position) validity mask M^{(k)} from seg_id (in-segment, non-warmup,
    # >= D+1 bars from segment right edge), computes L_main = L^{(0)} and
    # L_total = L_main + mu * mean_k( lambda_k * masked_CE^{(k)} )
    # returns {'total', 'main', 'mtp', 'per_depth_ce': [..]}
    ...

# Inference: distributional multi-horizon forecast
def forecast(context_ids: LongTensor[1, L0, 2],
             horizons:    list[int] = [1, 5, 15, 60],   # minutes == bars; produced by AR ROLLOUT
             N:           int = 10,                       # Monte-Carlo trajectories
             T:           float = 0.6, top_p: float = 0.9,
             use_spec_decode: bool = True                 # MTP speculative decoding on/off (draft span = D)
            ) -> dict:
    # rolls out N trajectories of max(horizons)=H_max bars (spec-decoded if enabled),
    # aggregates per horizon, returns per-horizon empirical distribution:
    #   {h: {'samples': FloatTensor[N], 'mean','median','q05','q95','std'} for h in horizons}
    # horizons 15 and 60 are NOT MTP depths; they are rollout aggregates.
    # distribution is IDENTICAL whether use_spec_decode is True or False (CI-asserted).
    ...
```

**Default config block:**

```yaml
mtp:
  enabled: true            # false -> D=0, bit-exact Kronos objective for the FSQ/BSQ ablation
  D: 4                     # search {2,4,6,8}; native span t+1..t+D+1 (<= ~5 min at D=4)
                           # selection: depth-0 val PPL must not regress
  mu: 0.3                  # search {0.1,0.3,0.5}; optional anneal 0.3->0.1 over last 20% steps
  lambda_schedule: uniform # {uniform, geometric:gamma in {0.8,0.5}}
  aggregated_targets: false  # opt-in direct {5,15,60} aggregate heads (NOT depths); off to keep path-distribution contract
inference:
  horizons_min: [1, 5, 15, 60]
  scheme: autoregressive_rollout   # 5/15/60 via rollout; MTP is the speculative drafter, not a per-horizon head
  H_max: 60                # rollout/label look-forward that drives the eval purge
  mc_trajectories_N: 10    # search {5,10,20,30}; selection on cost-aware NET-IR
  temperature: 0.6
  top_p: 0.9
  speculative_decoding: true   # distribution-exact; pure speed; serving-layer concern
```

**Lookahead/causal-safety summary (all enforced):** (1) depth-`k` training positions require bars `t+1…t+k+1` in the *same* segment, else masked; (2) all attention (backbone, `MTPBlock_k`, intra-block coarse→fine cross-attn) is causal; (3) speculative decoding is distribution-exact and reads no future data; (4) Monte-Carlo rollout only ever conditions on already-sampled bars; (5) vol-scaling of targets is inherited from the feature-spec causal hook unchanged. The feature-spec pipeline-wide purity invariant is extended to assert that MTP logits at position `t`, depth `k`, are a pure function of bars with effective timestamp `≤ t+1` (the conditioning context), independent of the teacher-forced labels beyond their role as supervision.

---

## 7. Training Plan (two-stage)

Two sequential stages with a hard freeze boundary between them. **STAGE 1** trains the FSQ tokenizer (encoder + FSQ quantizer + decoder) to a pure reconstruction objective until the codebook converges. **STAGE 2** freezes the tokenizer, materializes the token stream, and trains the decoder-only AR backbone (next-token NLL + MTP causal-chain loss). The two stages never co-train: Stage 2 reads tokens, never the encoder weights. This section is the single source of truth for every optimizer, schedule, regularizer, determinism flag, checkpoint rule, and the one-YAML-reproduces-a-run config schema. All starting hyperparameters are pinned to **Kronos_small Table 5** and explicitly flagged where crypto-1m statistics motivate a deviation.

> **Notation.** `B` = batch (windows), `L` = sequence length in bars (`L <= 512`), `F = 16` features/bar, `d_model = 512`. Tokenizer emits, per bar, a **coarse** subtoken `b^c` and a **fine** subtoken `b^f` (the FSQ analogue of Kronos's factorized 10+10 BSQ token). "Bits-per-token" `= sum of log2(L_i)` is the *controlled* quantity in the 2×2 ablation, **never** an integer-rounded vocab. **The Tokenizer section OWNS the canonical FSQ config; this section references it and never restates a divergent number.**

> **Canonical FSQ config (referenced from the Tokenizer section — single source of truth).** `fsq_levels = [11, 9, 9, 7, 7, 5, 5]` (`D = 7` dims, **all odd**). `bits_per_token = sum(log2(L_i)) = 20.06 bits`, which matches the BSQ baseline `k = 20` within `0.06` bit (tolerance `±0.5` OK). The coarse/fine dim split is **derived** by the §2 grouping rule, not hardcoded: `coarse_dims = 3 {11,9,9}` (`bpt_coarse = 9.80`, `V_c = 891`), `fine_dims = 4 {7,7,5,5}` (`bpt_fine = 10.26`, `V_f = 1225`). Total vocab `= 891 × 1225 = 1,091,475 ≈ 2^20.06` (BSQ ref `2^20 = 1,048,576`). All levels are odd because the coarse-only decode pins fine dims to the level-0 **center** cell, which exists only for odd `L_i`.

---

### 0. Pre-flight sanity gates (MANDATORY — block any full run)

No GPU-hours are spent until **all** gates below pass on the exact commit that will launch the run. These are CI/merge gates and pre-launch gates; the launcher refuses to start a full run if `preflight.passed != true` in the run manifest.

**G0 — Unit tests (both stages).**
- Feature pipeline: the lookahead **truncation + perturbation** tests must be green. These are **transform-agnostic** — they assert the ENTIRE feature + mask + segment_id + target pipeline (every gate, fill, clip, split, normalization, vol-scaling) is a pure function of raw data with effective timestamp `<= t+1`, and they sample test bars specifically from bad-tick-adjacent, segment-boundary, structural-break, and stale-run regions (not just random bars). The test fails if ANY transform reads beyond `t+1`.
- Loader causal-mask assertion: assert that `is_stale` and the packed `dq_flags` bitfield are **never** fed to the model and are **never** OR-ed into `m_t`; the model-visible missingness mask `m_t` is set only from strictly-causal per-bar conditions (bar-`t` volume==0, structurally-absent spot funding/OI, warm-up).
- Tokenizer shape/dtype contract: `encode(x: [B,L,16]) -> (b_c: [B,L], b_f: [B,L])`, `decode -> x_hat: [B,L,16]`; assert exact shapes and that `b_c` is a valid coarse index (`0 <= b_c < V_c = 891`) and `b_f` a valid fine index (`0 <= b_f < V_f = 1225`).
- FSQ round-trip determinism: same input + same seed → bit-identical codes (FSQ is a deterministic `round`-with-STE quantizer; no stochastic codebook).
- AR factorization: assert `p(b^f | ·)` is conditioned on the **sampled** `b^c` (not teacher-forced) via the intra-block cross-attention — a unit test that swaps the sampled coarse token and verifies the fine logits change.
- MTP chain: assert depth-`d` head input includes depth-`(d-1)` representation (causal chain, **not** parallel-independent), and that removing the depth-`(d-1)` tensor changes depth-`d` logits.
- Loss reductions ignore masked/padded positions (`m_t` and pad mask both honored).

**G1 — Overfit a single batch to ~0 loss (both stages, separately).**
- *Stage 1:* take one fixed batch (e.g. `B=8, L=512`), disable dropout/weight-decay/aug, train to convergence. **Gate:** reconstruction MAE → `< 1e-3` (z-scored feature units) within ≤ 2000 steps. If it floors above that, the FSQ levels are too coarse or the STE is mis-wired — fix before scaling.
- *Stage 2:* one fixed batch of pre-tokenized sequences, dropout off. **Gate:** train cross-entropy (coarse + fine) → `< 0.05` nats/token within ≤ 2000 steps; MTP depth-1 head matches the next-token head. Inability to overfit one batch ⇒ masking/causality/loss bug, not a capacity issue.

**G2 — Determinism smoke.** Two back-to-back 200-step runs with identical seed/config produce **bit-identical** loss curves (asserts the determinism flags in §7 are actually in force). Any divergence fails the gate.

**G3 — Throughput probe.** A 500-step timed run on the target GPU records tokens/s and step-time; the wall-clock estimate (§8) is regenerated from *measured* throughput, not the a-priori assumption. If measured throughput is < 60% of the assumed value, re-plan before committing the full run.

**Saturation watch (runs through both stages).** A Kronos_small-class model (~21–24M realized params) saturates well before 2B bars (Locked Decision). The val-loss early-stopping monitor (§3/§5) is the saturation detector: when val NLL improvement over a full eval interval `< delta_sat` (default `0.1%` relative) for `patience` consecutive intervals, the run is at the saturation point — stop, checkpoint, and do not feed more data expecting gains.

---

### 1. Shared training infrastructure (both stages)

- **Precision:** mixed `bf16` autocast on A100/H100 (no loss scaler needed for bf16; master weights stay fp32 under AdamW). `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`. **Carve-outs forced to fp32:** the FSQ bound/round op and the EWMA-derived normalization are upstream/cpu-side; inside the model, the **softmax + cross-entropy** and any **variance/normalization reduction** run in fp32 (`norm` layers compute in fp32 then cast back). FlashAttention-2 kernels run in bf16.
- **Single-GPU plan that is DDP-ready (explicit code path).** Default launch is 1×GPU, but the model is wrapped unconditionally so the multi-GPU path is a flag, not a rewrite:
  ```python
  # train.py — DDP-ready single-GPU. torchrun --nproc_per_node=N launches the SAME file.
  import os, torch, torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel as DDP

  def setup_dist():
      ws = int(os.environ.get("WORLD_SIZE", "1"))
      rank = int(os.environ.get("RANK", "0"))
      local = int(os.environ.get("LOCAL_RANK", "0"))
      ddp = ws > 1
      if ddp:
          dist.init_process_group(backend="nccl")  # NCCL for CUDA
      torch.cuda.set_device(local)
      return ddp, rank, local, ws

  ddp, rank, local, world_size = setup_dist()
  device = torch.device(f"cuda:{local}")
  model = build_model(cfg).to(device)
  if ddp:
      model = DDP(model, device_ids=[local], output_device=local,
                  gradient_as_bucket_view=True, find_unused_parameters=False)
  # IMPORTANT: find_unused_parameters MUST be False — MTP/cross-attn heads are all used
  # every step; True silently masks a wiring bug and slows AllReduce.
  ```
  - **Sampler:** `DistributedSampler(shuffle=True, seed=cfg.seed, drop_last=True)` when `ddp`, plain seeded `RandomSampler` otherwise. `drop_last=True` keeps step semantics identical across world sizes.
  - **Per-GPU vs global batch:** `cfg.batch_size` is **per-GPU**; global batch `= batch_size * world_size * grad_accum`. LR is set for the *global* batch (§2/§4); when scaling GPUs, either hold global batch fixed (reduce `grad_accum`) or apply linear LR scaling and re-warm. The config records both so a run is unambiguous.
  - **Grad-accum** (`cfg.grad_accum`, default 1) lets a single GPU emulate the target global batch; gradients are averaged, and DDP `no_sync()` is used on accumulation sub-steps to avoid premature AllReduce.
  - **Logging/checkpointing only on `rank == 0`.** `dist.barrier()` before checkpoint save; metrics are `all_reduce`-averaged before logging.

- **Optimizer (both stages): AdamW**, `betas=(0.9, 0.95)` (transformer-standard; tighter `beta2=0.95` over Adam's 0.999 stabilizes the noisy financial-token gradient), `eps=1e-8`, `weight_decay=0.01` (Kronos_small). **Decoupled decay groups:** weight decay applies to matmul/projection weights only; **no decay** on biases, all `LayerNorm`/`RMSNorm` gains, embedding tables (temporal + token embeddings), and the FSQ per-stage learnable scales (§2). Two param groups, built by name filter, recorded in the run manifest.
- **LR schedule (both stages): cosine decay with linear warmup.** Warmup = **10% of total scheduled steps** (Kronos uses ~10% / ~15k steps over the first phase). Cosine decays from peak to `min_lr = peak_lr * 0.1` (a 10% floor, not 0 — keeps the tokenizer codebook and the AR tail learning). One scheduler driver shared by both stages:
  ```python
  def lr_at(step, peak, warmup, total, floor_frac=0.1):
      if step < warmup:
          return peak * step / max(1, warmup)
      import math
      p = (step - warmup) / max(1, total - warmup)          # in [0,1]
      return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * p)))
  ```
- **Gradient clipping:** global-norm clip via `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`. Default `max_norm = 1.0`. Log the *pre-clip* grad norm every step; a sustained pre-clip norm ≫ 1 (e.g. > 5 for many steps) signals LR too high or a bad batch (outlier crypto bar) — auto-throttle hook optional, off by default.
- **Determinism & seed pinning:** see §7 (single source for both stages).
- **Checkpointing & resumability:** see §6 (single source for both stages).
- **Experiment tracking:** Weights & Biases (`wandb`), `project="trikaal"`, run group = stage tag, `config=` the full resolved YAML, `wandb.run.id` written into the checkpoint so a resumed run reattaches. Per-stage metric panels in §3 / §5.

---

### 2. STAGE 1 — FSQ tokenizer training (reconstruction)

**Objective.** Learn encoder `E`, FSQ quantizer `Q` (coarse + fine), decoder `D` so that `x_hat = D(Q(E(x)))` reconstructs the 16-dim bar, with a coarse→fine information hierarchy preserved from Kronos.

**Loss (hierarchical, FSQ-adapted — Huber/smooth-L1, δ=1.0 for BOTH terms).**
$$
\mathcal{L}_{\text{tok}} = \underbrace{\text{Huber}_{\delta=1}\big(D_c(\hat z^c),\, x\big)}_{\mathcal{L}_{\text{coarse}}} \;+\; \underbrace{\text{Huber}_{\delta=1}\big(D(\hat z^c, \hat z^f),\, x\big)}_{\mathcal{L}_{\text{fine}}}
$$
- Both `L_coarse` and `L_fine` use **Huber (smooth-L1, δ=1.0)**, not MSE — crypto 1-minute features are heavy-tailed even after the `[-5,5]` clip, and Huber's linear tail prevents a handful of residual outlier bars from dominating the reconstruction gradient. This matches the Tokenizer section exactly.
- `L_coarse` forces the **coarse subtoken alone** to give a rough reconstruction (Kronos hierarchy): the coarse-only decode pins the fine dims to their level-0 **center** cell (defined only because every `L_i` is odd). `L_fine` uses the full coarse+fine token for high fidelity.
- **No commitment / codebook loss term** — this is the whole point of FSQ over BSQ (Correction headline). FSQ has **no `lambda*L_quant`**; the bounded `round`-with-STE replaces it. Do **not** port Kronos's `lambda*L_quant`; its absence is a tested invariant (G0 asserts no commitment term in the Stage-1 graph).
- **Masked & per-feature weighting:** reductions skip imputed fields via `m_t` (don't penalize reconstructing a structurally-absent funding channel). Per-feature loss weights default to `1.0`; optionally upweight the 3 price/return-bearing features (`ret_close`, `range`, `body`) by `w_px` (default `1.0`, tune in `{1.0, 2.0}`) since downstream forecasting keys off them. All weights recorded in config.

**FSQ specifics (canonical config, enforced here).**
- The FSQ dims + per-dim level counts are the canonical `[11, 9, 9, 7, 7, 5, 5]` (`D = 7`, all odd, `bpt = 20.06`). They were **derived empirically** (in the Tokenizer section, which owns the search) by selecting, on a held-out month, the lowest val recon-MAE among configs whose `bpt = sum(log2(L_i))` lands in `[19.5, 20.5]` so it matches the BSQ baseline's `k = 20` bits within `±0.5` bit for ablation fairness. Valid alternates in that band (every entry's bpt computed as `sum(log2(L_i))`): `[9,9,9,9,7,5,5] = 20.13`; `[9,9,9,7,7,5,5] = 19.77` (conservative FSQ ≤ BSQ). **Primary parity point = `[11,9,9,7,7,5,5] = 20.06`** — the default. Training does **not** restate or re-derive these levels; it imports them from the Tokenizer §c table.
- **Derived coarse/fine split (NOT hardcoded).** The grouping rule sorts dims by `L_i` descending and splits so `bpt_coarse ≈ bpt_fine ≈ bpt/2`. For the canonical config that yields `coarse_dims = 3 {11,9,9}` (`bpt_coarse = 9.80`, `V_c = 891`) and `fine_dims = 4 {7,7,5,5}` (`bpt_fine = 10.26`, `V_f = 1225`). The split is computed by the rule and **stored in the run manifest** alongside the frozen levels — no `3/3` (or any fixed) split is written into the YAML.
- **Residual magnitude decay guard:** monitor **per-stage reconstruction contribution** = `MAE(coarse-only) - MAE(coarse+fine)`. If the fine stage's marginal contribution collapses (documented residual-decay failure), enable **per-stage learnable scaling** (`alpha_c`, `alpha_f` applied to each stage's residual before quantization; no-decay param group). Default: scaling **on** (cheap insurance), initialized to 1.0.

**Data / sampling (Stage 1).** Stage 1 needs reconstruction coverage, not long context — it can use **shorter windows** (`L_tok = 128`, segment-respecting) and a **subsample** of the corpus (≈ 100–150M bars is ample to converge a per-bar autoencoder; the full 0.5–1B is unnecessary and wasteful for a per-bar codebook). Stratify the subsample across symbols/regimes so the codebook sees fat-tail bars. Kronos volume/amount dropout (`p_drop=0.05`, feature-spec §4.4) is **on** in Stage 1 too (teaches robust reconstruction). Vol-scaling input hook state matches the feature-spec config.

**Hyperparameters (Stage 1).**

| Knob | Default | Source / rationale | Tune range + criterion |
|---|---|---|---|
| peak LR | `1e-3` | Kronos_small Table 5 | `{5e-4, 1e-3, 2e-3}`; pick lowest stable val-MAE, no grad-norm blowups |
| warmup | 10% of total steps | Kronos / §1 | fixed |
| min LR | `1e-4` (10% floor) | §1 | fixed |
| weight decay | `0.01` | Kronos_small | `{0.0, 0.01}`; autoencoders sometimes prefer 0 |
| batch (per-GPU) | `128` windows × `L=128` | A100 80GB headroom (tiny model) | raise to fill memory; LR re-tuned per global batch |
| reconstruction loss | **Huber (smooth-L1, δ=1.0)** | heavy-tailed crypto; matches Tokenizer §d | δ fixed at 1.0 |
| tokenizer dropout | **0.0** | reconstruction autoencoder wants fidelity; matches the Tokenizer §a config table (both say 0.0) | `{0.0, 0.1}` if val-MAE ≪ train-MAE (rare) |
| grad clip | `1.0` | §1 | fixed |
| epochs | up to `30` over the Stage-1 subsample | converges fast | early-stop below |
| early stop | patience `5` eval-intervals on val recon-MAE, `min_delta=1e-4` | codebook convergence | — |
| eval interval | every `2000` steps | — | — |

**Stage-1 convergence / report metrics (W&B).** Logged every eval interval: **recon MAE** and **Huber loss** (overall + per-feature, z-scored units), **per-stage contribution** (coarse-only MAE vs full MAE), **codebook usage** = fraction of FSQ codes used over the eval set (target ≥ 95% — FSQ rarely collapses, this confirms it), **perplexity** = `exp(H(code distribution))` per subtoken (coarse vs `V_c = 891`, fine vs `V_f = 1225`), **clip-hit / NaN-guard counters**, **grad-norm (pre-clip)**, **LR**, **throughput (bars/s)**. The Stage-1 "done" report = converged MAE/Huber table, usage ≥ 95%, both subtoken perplexities high (no dead-code collapse), and confirmation that the frozen levels are the canonical `[11,9,9,7,7,5,5]` with `bpt = 20.06` and the derived `{11,9,9}`/`{7,7,5,5}` split (the quantity Stage-2/ablation controls on).

**Freeze handoff.** On Stage-1 acceptance: freeze `E, Q` (and `alpha_c, alpha_f`), `model.eval()`, `requires_grad=False`, and **materialize the token stream** for the full corpus to a content-hashed Parquet/`.npy` cache (`b_c, b_f, ts, segment_id, mask` per bar). Stage 2 reads only this cache — the tokenizer is never in the Stage-2 graph. The tokenizer checkpoint hash + FSQ config (levels + derived coarse/fine split) are recorded in every Stage-2 run manifest.

---

### 3. STAGE 1 — early stopping, checkpoint, and saturation specifics

- **Checkpoint cadence:** every `2000` steps + every eval interval; keep `best` (lowest val recon-MAE) and `last`. Full resumability per §6.
- **Early stopping:** patience 5 eval-intervals, `min_delta=1e-4` on val recon-MAE; on trigger, stop and promote `best`.
- **Saturation:** Stage 1 saturates quickly (per-bar codebook); the watch is the same monitor — when marginal MAE gain < `delta_sat`, stop and lock the codebook.

---

### 4. STAGE 2 — AR pretraining on FROZEN tokenizer outputs

**Objective.** Train the decoder-only causal Transformer (exact Kronos_small config: 8 layers, `d_model=512`, `d_ff=1024`, 8 heads, RoPE, Pre-LN, RMSNorm, plain 2-matrix SiLU FFN — SwiGLU optional, off by default for Kronos parity — learnable temporal embeddings, 512-token context) over the materialized token stream, with the hierarchical factorization and MTP. **Bits-per-token is held FIXED at ≈20 (vocab `V_c=891`, `V_f=1225`)** — vocabulary is the controlled ablation variable, **not** a parameter-padding lever. The realized parameter count lands ~21–24M ("Kronos_small class"); if padding toward the upper band is ever desired it uses a non-vocab knob (e.g. a 2-layer `W_fuse` MLP), never an enlarged vocab.

**Loss.**
$$
\mathcal{L}_{\text{AR}} = \mathcal{L}_{\text{NLL}} + \beta \cdot \mathcal{L}_{\text{MTP}}
$$
- **Next-token NLL (Kronos factorization):**
  $$
  \mathcal{L}_{\text{NLL}} = -\sum_t \Big[\log p(b^c_t \mid b_{<t}) + \log p(b^f_t \mid b_{<t},\, b^c_t)\Big]
  $$
  where `p(b^f | ·)` conditions on the **sampled** (not teacher-forced) `b^c` via the intra-block cross-attention (query = embedding of the sampled coarse prediction; K/V = backbone hidden state) — fights exposure bias, per Kronos.
- **MTP loss (DeepSeek-V3 causal chain, Addition 1):** `D_mtp` auxiliary heads predict bars `t+1 .. t+D_mtp` as a **causal chain** — depth `d`'s head consumes depth `(d-1)`'s representation, **not** parallel-independent. Each depth carries the same coarse+fine factorized CE. **The MTP depths provide native dense supervision/draft only up to `t+D_mtp` (default `D_mtp = 4` ⇒ horizons ≤ ~5 min).** Direct depth-60 heads are a non-starter; the 5/15/60-min forecasts are produced by **autoregressive rollout** of the AR model, with MTP acting as the speculative drafter — they are *not* realized by choosing a depth or stride. The per-bar token embedding `Emb(b)` consumed by each MTP depth **is the backbone's fused embedding path** (`CoarseEmbed + FineEmbed + W_fuse`), shared **by reference** (zero marginal embedding params).
  $$
  \mathcal{L}_{\text{MTP}} = \frac{1}{D_{\text{mtp}}}\sum_{d=1}^{D_{\text{mtp}}} \text{CE}\big(\text{head}_d(h^{(d)}_t),\; b_{t+d}\big),\quad h^{(d)} \text{ depends on } h^{(d-1)}
  $$
  `beta` (MTP weight) default `0.3`, tune `{0.1, 0.3, 0.5}` — must stay **secondary** to the tokenizer headline; selection = best val NLL at depth-1 without degrading 1-bar forecast IC.
- **Vol-scaled, regime-relative targets (Addition 2):** the *token targets* are produced by the tokenizer applied to vol-scaled inputs per the feature-spec §7 hook (target path always on); Stage 2 trains on whatever token stream the (frozen) tokenizer emitted. No extra target transform lives in Stage 2 beyond consuming that cache. Distributional-by-contract output (Addition 3) is intrinsic: every head emits a full categorical distribution over subtokens.
- **Reductions** ignore pad positions and respect segment boundaries (windows never straddle, per feature-spec §4).

**Data / sampling (Stage 2).** Full materialized token corpus, `L = 512` (Kronos cap), segment-respecting sampling, `DistributedSampler` seeded. Token dropout (§ below) and temporal embeddings keyed off bar-open `ts` from the cache.

**Hyperparameters (Stage 2) — start from Kronos_small Table 5, deviations flagged.**

| Knob | Default | Source / rationale | Tune range + criterion |
|---|---|---|---|
| FFN dropout | `0.25` | **Kronos_small Table 5** | `{0.1, 0.25}` — crypto-1m note below |
| residual dropout | `0.25` | **Kronos_small Table 5** | `{0.1, 0.25}` |
| attention dropout | `0.1` | **Kronos_small Table 5** | `{0.0, 0.1}` |
| token dropout | `0.1` | **Kronos_small Table 5** (input-token masking regularizer) | `{0.05, 0.1}` |
| peak LR | `1e-3` | **Kronos_small Table 5** | `{5e-4, 1e-3}`; lower if grad-norm unstable on 0.5B+ tokens |
| weight decay | `0.01` | **Kronos_small Table 5** | fixed |
| AdamW betas | `(0.9, 0.95)` | §1 | fixed |
| warmup | 10% of total steps | Kronos | fixed |
| min LR | `1e-4` (10% floor) | §1 | fixed |
| grad clip | `1.0` | §1 | fixed |
| batch (per-GPU) | `64` windows × `L=512` (≈ 32k tokens/GPU) | A100 80GB fit at bf16 for a ~21–24M model | fill memory; keep global batch target via `grad_accum` |
| target global batch | ≈ `0.5M` tokens/step | stable LM pretraining scale at this size | hold fixed across GPU counts |
| epochs | `1–3` passes over ≤ 1B-bar corpus | a Kronos_small-class model saturates < 2B bars | early-stop + saturation watch |
| early stop | patience `4` eval-intervals on val NLL, `min_delta=0.1%` rel | saturation | — |
| eval interval | every `5000` steps | — | — |
| MTP weight `beta` | `0.3` | Addition 1 | `{0.1,0.3,0.5}` |
| MTP depths `D_mtp` | `4` | native dense ≤5 min; longer horizons via rollout | `{2,4}` |

> **Crypto-1m deviation note (explicit, per assignment).** Kronos's 0.25 FFN/residual dropout was tuned for equities/multi-frequency at base scale. Crypto 1-minute is **noisier and far more abundant** (≈ 0.5–1B bars for a ~21–24M model), so the overfitting pressure that justified 0.25 is weaker — heavy dropout may *underfit* here. Plan: **launch at the Kronos 0.25/0.1 defaults for baseline parity**, but run an early dropout sweep `{0.1, 0.25}` on a 50M-bar slice; **adopt the lower setting only if** it improves val NLL **and** the headline NET-IR diagnostic on a purged walk-forward fold without inflating train↔val gap. This deviation is a tuning decision, **not** a second research claim, and is logged in the run manifest. (The 2×2 tokenizer ablation holds dropout fixed across all four cells regardless.)

**Stage-2 metrics (W&B).** Every eval interval: **train/val NLL** (total, plus coarse-CE and fine-CE split), **per-token perplexity** (coarse vs `V_c=891`, fine vs `V_f=1225`), **MTP loss per depth** (1..`D_mtp`) and the depth-1 vs next-token agreement, **token accuracy** (top-1 coarse, top-1 fine | sampled coarse), **grad-norm (pre-clip)**, **LR**, **throughput (tokens/s)**, **GPU mem**, and a lightweight **proxy forecast IC** on a held-out slice (1-bar) as an online sanity signal (the *real* headline cost-aware NET-IR — defined on the portfolio period-return series, with funding debited for perps and a tuned `theta = kappa·c_total` execution filter — is computed by the eval harness offline, not in the train loop). The saturation watch (§0) drives early stop.

---

### 5. STAGE 2 — epochs, early stopping, saturation

- 1–3 corpus passes max; **early stop** patience 4 eval-intervals on val NLL (`min_delta=0.1%` relative). The **saturation point** (Kronos_small-class capacity) is the operative stopping criterion — when val NLL flattens, additional bars do not help; stop and hand the `best` checkpoint to the eval harness for the cost-aware NET-IR run. Do not chase the 2B mark.
- Checkpoint cadence + resumability per §6; `best` = lowest val NLL.

---

### 6. Checkpointing & full resumability (both stages)

- **Cadence:** every `ckpt_every` steps (Stage 1 `2000`, Stage 2 `5000`) and at every eval interval. Retain `last` (rolling, keep last `k=3`) + `best` (by val metric). Atomic write (`tmp` → `os.replace`).
- **A checkpoint fully resumes a run** — it contains: `model.state_dict()` (raw module, unwrapped from DDP), `optimizer.state_dict()`, `scheduler step counter`, `global_step`, `epoch`, **all RNG states** (`torch`, `torch.cuda` all devices, `numpy`, Python `random`), the **DataLoader/sampler position** (epoch + within-epoch index, so the data order resumes exactly under the same seed), `wandb.run.id`, the **resolved config**, the **dataset content-hash**, and (Stage 2) the **frozen tokenizer hash + FSQ config (canonical levels + derived coarse/fine split)**. Resume restores RNG and sampler state so the post-resume trajectory is bit-identical to an uninterrupted run (verified by a short overlap-replay test).
- **DDP note:** save only on `rank 0` (unwrapped `model.module`); on resume, every rank loads the same state dict, then `DDP` re-wraps. `dist.barrier()` brackets save/load.
- **Manifest:** each run writes `run_manifest.json` (config hash, code git SHA, dataset hash, tokenizer hash, FSQ levels + derived split, GPU/driver, preflight gate results). A prediction on any date is reproducible from manifest + checkpoint + content-hashed raw Parquet.

---

### 7. Seed pinning & determinism (both stages)

```python
def set_determinism(seed: int):
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # required for deterministic cuBLAS
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# DDP: per-rank seed = base_seed + rank for dropout/sampling decorrelation,
# but DistributedSampler uses the SHARED base seed so the global shuffle is consistent.
```
- **FlashAttention-2 caveat (flag explicitly):** FA-2's backward is **not bit-deterministic** by default. For the **determinism-gate runs (G2)** and any published-reproducibility run, set `cfg.deterministic_attention=True` to fall back to PyTorch SDPA math/`mem_efficient` deterministic path (slower); for throughput production runs FA-2 is allowed and the run manifest records `deterministic=false`. The choice is logged, never silent.
- Seeds (`seed`, derived data seed, dropout seed) are config fields; G2 asserts two same-seed runs match. Kronos volume/amount dropout (§4.4) and token dropout are seed-pinned and logged so any batch is replayable.

---

### 8. Wall-clock estimate — 1× A100/H100, ~0.5–1B bars, ~21–24M params

**Assumptions (stated, regenerated from G3 measurement before commit).** 1 bar → 1 fused token (single token stream). Corpus = **0.75B bars ≈ 0.75B tokens**. Stage 2 = **1.5 passes** ⇒ ≈ **1.1B training tokens**. A Kronos_small-class model with FA-2 in bf16 on a single A100-80GB sustains, conservatively, **≈ 60–90k tokens/s** end-to-end (small model, attention-light at `L=512`; H100 ≈ 1.6–2× faster). MTP adds ~`D_mtp` light heads → ~1.2× per-step cost; folded into the rate below.

| Stage | Tokens processed | Sustained tok/s (A100) | Wall-clock A100 | Wall-clock H100 (~1.8×) |
|---|---|---|---|---|
| **Stage 1** (tokenizer, ≤150M-bar subsample, ~20 epochs but converges early) | ≈ 1–2B bar-reconstructions | ≈ 120k bars/s (decoder is tiny) | **≈ 3–6 h** | **≈ 2–3 h** |
| **Stage 2** (AR, 1.5 passes × 0.75B) | ≈ 1.1B tokens | ≈ 75k tok/s (incl. MTP) | **≈ 4–5 h** → with grad-accum/eval overhead **≈ 6–9 h** | **≈ 3–5 h** |
| **Total (single GPU)** | — | — | **≈ 10–15 h** | **≈ 6–9 h** |

So a full Stage-1 + Stage-2 cycle fits comfortably in **<1 day on a single A100**, **half a day on an H100** — cheap enough to run the entire **2×2 {BSQ,FSQ}×{OHLCV-only,+microstructure} ablation** (4 Stage-1 + 4 Stage-2 cycles) in a few GPU-days. The OHLCV-only arm is the **`F = 7`** Kronos-equivalent subset (5 price/shape + 2 volume/liquidity) vs the full `F = 16` vector. **G3 replaces these a-priori numbers with measured tok/s before any full launch.** If measured throughput < 60% of assumed, re-plan (Correction-grade gate, not a guess).

---

### 9. One-YAML-reproduces-a-run config schema

A single YAML fully specifies a run; the launcher resolves it, content-hashes it into the manifest, and refuses to start if `preflight.passed != true`. Shared keys at top level; stage-specific blocks under `stage1` / `stage2`. `${...}` denotes interpolation/inheritance.

```yaml
run:
  name: trikaal_fsq_micro_v1
  stage: stage2                 # stage1 | stage2
  seed: 1337
  out_dir: ./runs/${run.name}
  wandb: { project: trikaal, group: ${run.stage}, mode: online }

determinism:
  enabled: true                 # set_determinism(seed)
  deterministic_attention: false # true forces SDPA deterministic path (G2 + published runs)
  cudnn_benchmark: false

dist:                           # single-GPU by default; torchrun flips world_size
  backend: nccl
  grad_accum: 1                 # raise to hit target_global_tokens on 1 GPU
  find_unused_parameters: false

precision: { amp_dtype: bf16, fp32_softmax: true, fp32_norm: true }

optim:                          # shared AdamW
  name: adamw
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.01            # Kronos_small Table 5
  no_decay_on: [bias, norm, embedding, fsq_scale]
  grad_clip_norm: 1.0

sched: { type: cosine_warmup, warmup_frac: 0.10, min_lr_frac: 0.10 }

checkpoint:
  every_steps: 5000             # stage1 override -> 2000
  keep_last: 3
  save_best_by: val_nll         # stage1 -> val_recon_mae (lower=better)
  resume_from: null             # path -> full RNG/optsched/sampler restore

data:
  parquet_root: /data/trikaal/features        # content-hashed
  dataset_hash: null            # filled+frozen by launcher
  seq_len: 512                  # stage1 override -> 128
  segment_respecting: true      # windows never straddle a gap (feature-spec §4)
  min_segment_len: 128
  num_workers: 8

features:                       # mirror of the authoritative feature spec (must match)
  norm_estimator: ewma          # ewma | rolling
  half_life: { fast: 1440, slow: 5760 }
  clip: [-5, 5]
  vol_scale_inputs: false
  vol_scale_targets: true       # project contract (always on)
  vol_amount_dropout_p: 0.05    # Kronos §4.4, train-only, seed-pinned

tokenizer:                      # FSQ config — canonical, OWNED by the Tokenizer section
  fsq_levels: [11, 9, 9, 7, 7, 5, 5]  # D=7, ALL ODD; bpt = sum(log2(L_i)) = 20.06
  # coarse/fine split is DERIVED (sort dims by L desc; bpt_coarse ~= bpt_fine ~= bpt/2)
  # and stored in the manifest — NOT hardcoded here.
  coarse_fine_split: derived          # launcher computes {11,9,9} / {7,7,5,5}; writes to manifest
  per_stage_learnable_scale: true     # residual-decay guard
  bits_per_token: null                # = sum(log2(L_i)) = 20.06; ablation control var (BSQ k=20)
  recon_loss: huber                   # smooth-L1, delta=1.0 (BOTH coarse & fine) — heavy-tailed
  huber_delta: 1.0
  quant_type: fsq                     # fsq | bsq  (the 2x2 ablation switch)
  use_microstructure: true            # OHLCV-only -> false (F=7 subset; other ablation axis)
  ckpt: null                          # stage2: REQUIRED frozen tokenizer path
  ckpt_hash: null

model:                          # EXACT Kronos_small
  n_layers: 8
  d_model: 512
  d_ff: 1024
  n_heads: 8
  rope: true
  norm: rmsnorm
  pre_ln: true
  ffn: silu_2matrix             # plain 2-matrix SiLU FFN; SwiGLU optional, OFF (Kronos parity)
  temporal_embeddings: [minute_of_day, hour_of_day, day_of_week, day_of_month, month_of_year]
  attn_impl: flash2
  mtp: { enabled: true, depths: 4, beta: 0.3, causal_chain: true }
  # MTP depths give NATIVE dense supervision <= t+D (<= ~5 min at D=4);
  # 5/15/60-min forecasts come from AUTOREGRESSIVE ROLLOUT (MTP = speculative drafter),
  # NOT from per-horizon depth/stride heads.
  # Emb(b) for every MTP depth IS the backbone fused path (CoarseEmbed+FineEmbed+W_fuse),
  # shared by reference (zero marginal embedding params).

preflight:                      # gates §0 — launcher enforces
  run_unit_tests: true
  overfit_single_batch: true
  overfit_target_loss: 0.05     # stage1 -> 1.0e-3 (recon mae)
  determinism_smoke_steps: 200
  throughput_probe_steps: 500
  passed: false                 # set true only after all gates green

# ---- stage-specific (one of these is active per run.stage) ----
stage1:
  loss: { coarse: true, fine: true, commitment: false }   # NO commitment term (FSQ); Huber both
  feature_weights: { default: 1.0, price_group: 1.0 }
  optim_peak_lr: 1.0e-3
  dropout: { ffn: 0.0, resid: 0.0, attn: 0.0, token: 0.0 } # autoencoder wants fidelity (tune {0.0,0.1})
  batch_size: 128
  max_epochs: 30
  early_stop: { metric: val_recon_mae, patience: 5, min_delta: 1.0e-4 }
  eval_every: 2000
  report: [recon_mae, recon_huber, per_feature_mae, coarse_only_mae,
           codebook_usage, perplexity_coarse, perplexity_fine]   # ppl_coarse vs 891, ppl_fine vs 1225

stage2:
  optim_peak_lr: 1.0e-3                                     # Kronos_small Table 5
  dropout: { ffn: 0.25, resid: 0.25, attn: 0.1, token: 0.1 } # Kronos defaults; sweep {0.1,0.25}
  batch_size: 64
  target_global_tokens: 500000
  max_epochs: 3
  early_stop: { metric: val_nll, patience: 4, min_delta_rel: 0.001 }
  eval_every: 5000
  saturation: { delta_sat_rel: 0.001, patience: 2 }
  report: [nll_total, nll_coarse, nll_fine, ppl_coarse, ppl_fine,
           mtp_loss_per_depth, tok_acc_coarse, tok_acc_fine, proxy_ic_1bar]
```

**Schema contract.** `quant_type ∈ {fsq,bsq}` × `use_microstructure ∈ {true,false}` is the **2×2 ablation switch** (four YAMLs, identical everything else, `bits_per_token` held equal at ≈20 across all four cells — vocab is the controlled variable, never a param-padding lever). `tokenizer.fsq_levels` is the canonical odd-level `[11,9,9,7,7,5,5]` (`bpt = 20.06`), owned by the Tokenizer section; the coarse/fine split is **derived** (`{11,9,9}` / `{7,7,5,5}`) and frozen into the manifest, not hardcoded. `stage2` runs **must** supply `tokenizer.ckpt` + `tokenizer.ckpt_hash` from an accepted Stage-1 run. `preflight.passed` is set to `true` only by the gate runner — a human cannot hand-set it; the launcher recomputes the gates and aborts on mismatch. Every field above lands in `run_manifest.json` with the config + code + dataset + tokenizer hashes (including the derived FSQ split), so one YAML + the content-hashed raw Parquet reproduces the run end-to-end.

---

**Relevant file paths.** The target project directory `/Users/lakshaybhati/Downloads/MY AI MODEL/` contains only an empty `docs/superpowers/specs/` tree — there is no existing training code, config, or convention to conform to, so the schema, file names (`train.py`, `run_manifest.json`, `run.yaml`), and module interfaces above are proposed greenfield and are the section's normative contract.

---

## 8. Evaluation Harness (first-class subsystem)

This subsystem is the *arbiter*: the tokenizer claim is proven or rejected here, so the harness is held to the same engineering bar as the model. It is config-driven, seed-pinned, and content-hash-aware: every number this section emits is a pure function of `(weights_hash, dataset_hash, frozen_stats_hash, eval_config_hash, seed)`. The harness is split into layers that compose strictly downstream of the §1–§7 feature contract: (A) **fold construction** (purged walk-forward + leading embargo over time *and* held-out symbols); (B) the **headline metric** (cost-aware net Information Ratio on a portfolio return series, under an execution filter and a perp-funding-inclusive cost model); (C) the **2×2 ablation protocol** that constitutes the actual experiment; (D) **secondary diagnostics**; (E) **honesty commitments** that bind every reported number.

All causal envelopes from the feature spec propagate into the harness unchanged: the **entire** feature + mask + `segment_id` + target pipeline (every gate, fill, clip, split, normalization, and vol-scaling) is a pure function of raw data with effective timestamp `≤ t+1`. A label window may never read a bar the model was not allowed to read, and a fold split may never let a training sample's label overlap a test bar.

---

#### A. Purged walk-forward + leading embargo (Lopez de Prado)

**Reference.** Lopez de Prado, *Advances in Financial Machine Learning* (2018), Ch. 7 (purged k-fold CV, embargo) and Ch. 11–12 (combinatorial purged CV, backtest overfitting). We use the **walk-forward** (anchored/rolling) specialization because the production use is one-directional forecasting and we want a *future* held-out block, not interpolated folds; the purge + embargo mechanics are taken from Ch. 7.

##### A.1 The label and its span (why purging is needed at all)

The forecasts are evaluated at horizons `h ∈ {1, 5, 15, 60}` minutes, **produced by autoregressive rollout of the AR model** (the MTP heads provide native dense supervision/drafting only out to `t+D` — `≤5` min at `D=4` — and act as the speculative drafter; the 15- and 60-minute forecasts are obtained by rolling the model forward, not by per-horizon MTP depth heads). A training sample *anchored* at decision bar `t` carries a label that reads **forward** up to `H_max = 60` bars: `H_max` is the rollout/label look-forward (the longest evaluated horizon), and it is `H_max` — not any "MTP depth" — that drives the purge. Its label window is

```
label_span(t) = [t+1, t + H_max]            # H_max = 60 (longest evaluated rollout horizon)
```

The sample's **feature window** is the preceding `L ≤ 512` bars `[t-L+1, t]`. Two samples whose `[feature_window ∪ label_span]` intervals overlap share information; if one is in train and the other in test, the test leaks. **Purge removes that overlap.**

##### A.2 Time folds (walk-forward, anchored expanding)

Partition the global time axis (UTC, 1m grid) into `K` contiguous, chronologically ordered blocks. v1 default `K = 6`, sized so each test block ≈ 2–3 months of data (a full quarter of regime variety per fold is the selection criterion). Fold `k` uses:

```
Train_k = all samples with decision bar t  in  [T_0,           T_k_start - 1]
Test_k  = all samples with decision bar t  in  [T_k_start,     T_k_end]
```

Anchored/expanding (train always starts at `T_0`) is the default because a foundation-scale model is data-hungry and we want monotone-growing train sets; a **rolling** variant (fixed-width train window) is available via config for a robustness check (report both if they disagree by > 1 IR unit). Only past data ever trains for a future test block — walk-forward is one-directional by construction.

##### A.3 PURGE

For each test block `Test_k` with bar range `[T_k_start, T_k_end]`, drop from `Train_k` **every training sample `t` whose feature-or-label interval intersects the test block's bar range:**

```
drop t from Train_k  if  [t - L + 1,  t + H_max]  ∩  [T_k_start, T_k_end]  ≠ ∅
```

This removes (i) training samples whose 60-bar label reaches into the test block (forward leak) and (ii) training samples whose feature window overlaps the test block (backward leak at the boundary). Purging is applied at the *sample* level, not the *bar* level: a bar may survive as context while the sample anchored there is dropped.

##### A.4 EMBARGO — LEADING gap on the train→test boundary (the binding guard for anchored mode)

Purging removes only samples whose feature-or-label interval *overlaps* the test block. But serial correlation in 1m crypto returns/microstructure leaks across the boundary **beyond** the overlap region: a train sample anchored at `t = T_k_start − (H_max+1)` has a label ending exactly at `T_k_start − 1` (no overlap, so it survives the purge) yet sits immediately adjacent to the test block and is autocorrelated with its leading edge. In the **default anchored walk-forward, `Train_k` lies entirely before the test block**, so the operative leakage guard is a **LEADING embargo** that inset the train block's right edge:

```
LEADING embargo band = [T_k_start - E,  T_k_start - 1]   # drop all training samples anchored here
                                                          # (equivalently: inset Train_k right edge by E)
```

The previously-specified *trailing* band `[T_k_end + 1, T_k_end + E]` is **inert in anchored mode** (no train samples ever lie after `T_k_end`) and applies **only to rolling and CPCV modes**, where post-test bars re-enter training; it is scoped explicitly to those modes.

**Concrete embargo length for 1m bars / ≤60m horizons.** The embargo must exceed the label horizon so no surviving train sample's label can overlap the test block, *plus* a serial-correlation margin:

```
E = H_max + L_corr        with   H_max = 60 bars,  L_corr = 60 bars (1h serial-corr margin)
  = 120 bars  (2 hours)   DEFAULT
```

Rationale: `H_max = 60` guarantees label non-overlap; `L_corr = 60` (one extra hour) absorbs the autocorrelation in 1m crypto returns and microstructure (TFI/OI mean-reversion timescales are well under an hour). The embargo (not just the purge) **brackets every train→test boundary on the side where training data actually sits** — the leading side in anchored mode, both sides in rolling/CPCV. Tune `E ∈ {60, 120, 240}` (1h/2h/4h); selection criterion: the headline IR must be **flat** as the *leading* embargo `E` grows past 120 — if IR drops materially from `E=60` to `E=120` but is stable `120→240`, leakage existed at `E=60` and `E=120` is the honest floor. Because the leading embargo is the binding constraint in the default configuration, this flatness gate now tests the band that actually does the work. Report IR at all three `E` in the appendix.

##### A.5 Held-out SYMBOLS and held-out FUTURE TIME (two-axis generalization)

A single time split tests "does it generalize forward"; it does **not** test "does it generalize to coins it never trained on." We hold out **both axes simultaneously**:

- **Symbol split.** Partition the ~100–200 USDT universe into `TRAIN_SYMBOLS` (≈80%) and `HELDOUT_SYMBOLS` (≈20%), stratified by liquidity decile and by spot-vs-perp so the held-out set spans the same liquidity/structure range. Symbol assignment is by a hash of the symbol string seeded by `eval_config_hash` (deterministic, documented). The model **never** sees a held-out symbol's bars in training — across *all* time folds.
- **Four evaluation quadrants** (report the headline metric in each; the diagonal is the honest generalization number):

| Quadrant | Symbols | Time | What it measures |
|---|---|---|---|
| Q1 (in-sample sanity) | TRAIN | TRAIN time | overfit check only — never headlined |
| Q2 (temporal OOS) | TRAIN | FUTURE block | generalization to unseen *future* on known coins |
| Q3 (cross-sectional OOS) | HELDOUT | TRAIN time | generalization to unseen *coins* |
| **Q4 (true OOS — HEADLINE)** | **HELDOUT** | **FUTURE block** | unseen coins *and* unseen future — the published number |

**The headline net-IR is always reported on Q4.** Q2/Q3 are diagnostics that localize failure (Q2≫Q4 ⇒ symbol-overfit; Q3≫Q4 ⇒ regime/time-overfit). The `K`-fold walk-forward of A.2 runs *within* the TRAIN_SYMBOLS set to produce Q2; Q3/Q4 evaluate the same trained checkpoints on the held-out symbols at, respectively, in-window and final-future time.

**Causal-safety hooks inherited.** Normalization stats for held-out symbols are computed in **frozen-stats mode** frozen at each fold's train/test boundary (no held-out-symbol future ever touches a statistic). The `tau` sketch and EWMA state for held-out symbols are warmed *only* on their own pre-test bars within-segment. The feature-spec pipeline-purity test (the transform-agnostic truncation/perturbation invariant — that the entire feature/mask/`segment_id`/target pipeline is a pure function of raw data with effective timestamp `≤ t+1`, with test bars sampled specifically from bad-tick-adjacent, segment-boundary, structural-break, and stale-run regions) is re-run on a sample of Q4 bars as a merge gate on the harness itself.

---

#### B. Headline metric — cost-aware NET Information Ratio on a portfolio return series, under a forecast-magnitude execution filter

The single number that proves or sinks the project. Raw RankIC is **not** headlined because a strategy can have positive IC yet lose money after costs; the headline must be the *tradeable, cost-net* quantity, and it must be measured as a **portfolio period-return series**, not a pool of individual trades.

##### B.1 Signal → position (the strategy)

At each decision bar `t` (for a chosen primary horizon `h`, default `h = 15` min — the horizon where 1m microstructure has decayed enough to be tradeable but signal still persists; report `h ∈ {1,5,15,60}` too, all produced by autoregressive rollout), the model emits a predicted **vol-scaled** move `ŷ_{t,h}` (the regime-relative target from the vol-scaling section) and, via the rollout MTP distribution, a point estimate `μ̂_{t,h}` for the *raw* expected log-return over `[t+1, t+h]`. The strategy:

```
threshold  θ = κ · c_total        # tuned execution filter: edge must clear cost WITH MARGIN κ>1
position   s_t =  +1   if  μ̂_{t,h}  >  +θ        (long)
                  -1   if  μ̂_{t,h}  <  -θ        (short)
                   0   otherwise                  (no trade — edge does not clear the margin)
```

- **Execution filter (tuned, not break-even):** we trade **only when `|μ̂_{t,h}| > θ = κ · c_total`** with the multiple `κ > 1` a **tuned hyperparameter**. A pure break-even filter (`κ = 1`, `θ = c_total`) admits marginal trades whose expected *net* return is ≈ 0 but whose variance is positive, mechanically depressing the very IR the filter is meant to maximize; the cost-optimal filter requires the predicted edge to clear cost **with a margin** that covers forecast error. We **search `κ ∈ {1, 1.5, 2, 3}`** and select the `κ` that maximizes **held-out net-IR on the VAL folds** (Q2-style temporal-OOS validation slices) — **never on the Q4 test set** — and report the chosen `κ` in the pre-registered config. **`θ = c_total` (i.e. `κ = 1`) is retained only as the break-even DIAGNOSTIC baseline**, reported alongside the tuned filter, never as the headline configuration. Optionally `θ` is made **σ-aware** (scaled by the predicted dispersion of the rollout distribution, which is already available) so the margin adapts to forecast confidence; this variant is reported as an ablation of the filter.
- **Uniform sizing (v1):** `|s_t| ∈ {0,1}` — a flat unit notional per active position. **Sizing against forecast confidence is explicitly v2** (meta-labeling); v1 publishes the *side*, not the *size*. Reported as `unit_notional` constant across all trades.
- **Long/short, both directions** (perps allow shorting; for spot-only symbols the short leg is disabled and that is flagged in the per-symbol report).
- **Non-overlapping, period-aligned execution accounting:** positions are held to horizon `h`; to avoid double-counting overlapping bars we rebalance on a **stride-`h` grid** of decision bars (decisions at `t, t+h, t+2h, …`). Each stride-`h` interval is one **rebalance period**; within a period a symbol holds at most one position, so per-period P&L windows are disjoint. The dense-grid (stride-1) IC diagnostics are reported separately in §D.

##### B.2 Transaction-cost model — including perp funding (cost from day one)

Per-side **execution** cost as a fraction of notional, with three additive components grounded in the crypto cost literature, **plus a funding-carry term for perpetual positions that span a settlement:**

```
c_side =  f_taker            (exchange taker fee)
        + 0.5 * spread_bps    (half-spread slippage: cross the book once)
        + k_impact * (Q / ADV_t)   (linear temporary impact)

c_total (round-trip execution) = 2 * c_side          # enter + exit
```

**Execution defaults and rationale (cite):**

- `f_taker = 0.04%` (4 bps) — Binance USDT-perp / spot taker fee at a representative VIP-0–1 tier. Conservative; many makers pay less but we assume **taker** (we cross the book, consistent with an aggressive 1m signal).
- `spread_bps`: BTC/ETH effective half-spread on Binance is ~0.01–0.05%; mid-cap USDT pairs 0.05–0.3%. We use a **per-symbol, per-bar effective spread** when reconstructible from data, else a liquidity-decile default: BTC/ETH `spread_bps = 1 bp`, top-20 `= 3 bp`, the long tail `= 10 bp`. (Effective Binance order-book spreads of 0.01–0.1% are well documented; see refs below.)
- `k_impact = 0.1` (10 bps per unit of `Q/ADV` participation) as a conservative linear temporary-impact coefficient; `Q` = order notional, `ADV_t` = causal trailing 24h average dollar volume (`log_amount`-derived, de-logged). With unit notional sized to ≤ a small fraction of ADV the impact term is small but **present from day one** so the model cannot learn to "trade" illiquid bars for free.

**Funding term (perpetuals only).** Perps pay/receive funding every 8h; any position held across a settlement boundary realizes a funding cash flow that is a first-order, recurring cost. Because `funding_rate` is in the feature set and causally available, the funding debit is a deterministic add-on to each trade's net P&L:

```
funding_cost(symbol, t, h, s_t) = s_t * Σ_{settlements g ∈ [t+1, t+h]}  funding_rate_g
```

i.e. a **long** position pays positive funding and a **short** receives it (and vice-versa for negative funding). The term is **zero** when the holding interval `[t+1, t+h]` spans no funding settlement (the common case for `h ≤ 60` min, but non-trivial for multi-bar held horizons and any position straddling an 8h timestamp). **Spot-pair trades have no funding leg** and the term is identically zero. Funding is included in the cost-stress sensitivity sweep (§E) alongside the execution components.

This yields an all-in **round-trip execution cost in the 0.1%–0.3% band** the crypto cost literature reports for taker execution on liquid pairs (≈8 bps for BTC up to ≈30+ bps for mid-caps), plus a position-dependent funding leg, matching the project mandate. The exact `c_total` and `funding_cost` per trade are computed per `(symbol, bar)`, **never** a flat universe constant, and the full cost-config is content-hashed with every reported number. Sensitivity is mandatory: re-report the headline IR at `c_total ∈ {0.10%, 0.20%, 0.30%}` and with funding stressed (the cost-stress curve, §E).

> Refs for the cost model: Lopez de Prado (2018) on cost-aware backtesting; documented Binance effective spreads of ~0.01–0.1% on liquid order-book pairs ([CoinMetrics/market-microstructure surveys](https://www.coinmetro.com/glossary/bid-ask-spread)); Bitcoin bid-ask spread ≈0.02–0.04% on majors ([market-microstructure references](https://markets.bitcoin.com/glossary/spread)). The paper cites the primary academic crypto-cost sources (e.g. Makarov & Schoar 2020 on crypto market frictions; Marshall, Nguyen & Visaltanachoti on crypto liquidity) alongside these.

##### B.3 Net P&L, the PORTFOLIO return series, and the net-IR formula

For each executed position `i` (symbol `n`, decision bar `t_i`, side `s_{t_i}`, horizon `h`), the **net** realized log-return is

```
r_i^net = s_{t_i} * log(C_{t_i + h} / C_{t_i + 1})
        - c_total(n, t_i)                      # round-trip execution cost
        - funding_cost(n, t_i, h, s_{t_i})     # perp funding carry (0 for spot / no-settlement spans)
```

(entry assumed at next-bar `C_{t_i+1}` to avoid same-bar lookahead; exit at `C_{t_i+h}`; execution cost subtracted once per round trip; funding subtracted for each settlement crossed.)

**The IR is defined on a PORTFOLIO RETURN SERIES, not on pooled trades.** Pooling all per-trade returns across the Q4 cross-section *and* time, then sqrt-annualizing by a single-symbol opportunity count, is dimensionally wrong: it conflates trade-level with period-level Sharpe, collapses `S` parallel positions into one variance estimate, and makes horizons non-comparable. Instead, at each stride-`h` **rebalance period** `p` we aggregate the equal-weighted net P&L across **all simultaneously-active symbol positions** into one portfolio period-return:

```
r_p  =  ( 1 / |A_p| ) * Σ_{n ∈ A_p}  r_{n,p}^net      # A_p = symbols with an active position in period p
                                                        # equal-weight; r_{n,p}^net is that symbol's net log-return for the period
```

The portfolio return series `{r_p}` over the Q4 horizon is the object on which the Information Ratio is computed:

```
mean_p      =  mean_p ( r_p )
std_p       =  std_p  ( r_p )                          # population std over PORTFOLIO PERIODS
IR_period   =  mean_p / (std_p + eps_ir)
IR_net      =  IR_period * sqrt( periods_per_year )    # annualization on the period series
periods_per_year = 525600 / h        # h=1 →525600, h=5 →105120, h=15 →35040, h=60 →8760
```

`eps_ir = 1e-12`. Because annualization now multiplies the std of a **portfolio period-return** by `sqrt(periods_per_year)`, the factor is dimensionally consistent and horizons are comparable.

- **Cross-sectional breadth is reported separately:** the distribution of `|A_p|` (number of concurrent positions per period) is published so the reader sees how many simultaneous bets back each portfolio return; a thin-breadth series with a high IR is flagged.
- **A per-trade IR, if reported at all, is labeled explicitly as a per-trade statistic and is NOT sqrt-annualized** (it is a descriptive trade-quality number, not an annualized portfolio ratio).
- The per-symbol net-return distribution is still published as a **box-plot** (so a single lucky coin cannot carry the result; see §E no-cherry-picking).
- We additionally report **net Sharpe** on the same portfolio series and gate it through the **Deflated Sharpe Ratio** (Bailey & Lopez de Prado) accounting for the number of ablation cells and hyperparameter trials, so an IR inflated by multiple-testing is discounted — see §E.

##### B.4 Baselines — the headline is INCREMENTAL net-IR over the best naive benchmark

Degenerate internal baselines alone (no-trade `s_t ≡ 0`, trade-everything `κ→0`) cannot distinguish genuine alpha from passive beta capture. We therefore report, **under the identical cost model (including funding), folds, and portfolio-series annualization:**

- **No-Trade baseline** (`s_t ≡ 0`, IR = 0) and **trade-everything baseline** (filter off) — the internal bracket; the tuned filter must beat trade-everything on net IR, which is the entire point of the cost-aware filter.
- **Passive buy-and-hold** — a long-only equal-weighted basket of the Q4 universe, rebalanced on the same stride-`h` grid, net of the same execution + funding cost model. This is the trivially-available beta.
- **Price-momentum baseline** — a simple naive rule (e.g. sign of the trailing `h`-bar return) under the same filter scaffold and cost model.

**The headline result is reported as INCREMENTAL net-IR over the best naive benchmark** (`IR_net(FSQ+micro) − IR_net(best of buy-and-hold / momentum)`), so a reader can see whether the model's net-IR is incremental over trivially-available exposure. This strengthens the controlled-comparison framing rather than weakening it.

---

#### C. The 2×2 ablation protocol {BSQ, FSQ} × {OHLCV, +micro} at matched bits-per-token

This is the experiment. The headline claim ("microstructure-aware FSQ tokenizer beats Kronos BSQ") is **only** established by this controlled 2×2, with everything but the manipulated variable held identical.

##### C.1 The four cells

| Cell | Quantizer | Input vector | Role |
|---|---|---|---|
| **1** | BSQ | OHLCV-only — `F = 7` (5 price/shape + 2 volume/liquidity = the Kronos-equivalent subset) | **Kronos reproduction / baseline** |
| **2** | BSQ | full `F = 16` (+micro) | isolates the *microstructure* leg |
| **3** | FSQ | OHLCV-only — `F = 7` | isolates the *FSQ* leg |
| **4** | **FSQ** | **full `F = 16` (+micro)** | **the proposed model** |

The OHLCV-only arm is **`F = 7`** (5 price/shape + 2 volume/liquidity), matching the feature spec and the Kronos-equivalent subset — not 6. Two main effects, cleanly separable: (4 vs 3)+(2 vs 1) attributes the **microstructure** contribution; (3 vs 1)+(4 vs 2) attributes the **FSQ** contribution; the interaction (4−2)−(3−1) tests whether FSQ and micro are synergistic. One claim, two mechanically-independent legs — exactly the design contract.

##### C.2 What is held FIXED across all four cells (parity contract)

Everything except {quantizer, input dims}:

- **AR backbone:** identical Kronos_small config (8 layers, `d_model = 512`, `d_ff = 1024`, 8 heads, plain 2-matrix SiLU FFN [SwiGLU off by default for Kronos parity], RoPE, learnable temporal embeddings, hierarchical coarse→fine + intra-block cross-attention, MTP heads). The realized parameter count is **~21–24M** (a "Kronos_small-class" budget — the nominal "~27M" is a class label, not a hard target); it is reported honestly and is **not** padded toward a round number by enlarging the vocabulary.
- **Bits-per-token (the critical control):** `bpt = Σ log2(L_i)` for FSQ must **match** the BSQ `k` (Kronos uses `k = 20`). The canonical FSQ config (owned by the Tokenizer section, referenced here without restatement of divergent numbers) is **`fsq_levels = [11, 9, 9, 7, 7, 5, 5]`** (`D = 7` dims, **all odd**), giving **`bpt = 20.06`**, which matches BSQ `k = 20` within `0.06` bit (tolerance `±0.5` bit). The derived coarse/fine split (sort dims by level descending; split so `bpt_coarse ≈ bpt_fine ≈ bpt/2`) is **coarse `{11,9,9}` → `bpt_coarse = 9.80`, `V_c = 891`** and **fine `{7,7,5,5}` → `bpt_fine = 10.26`, `V_f = 1225`** (total vocab `1,091,475 ≈ 2^20.06`; BSQ ref `2^20 = 1,048,576`). **Bits-per-token, not vocabulary size, is the controlled variable**: the vocab is *not* a free parameter-padding lever, because enlarging it (e.g. to a 22.7-bit FSQ) would break the capacity-parity that the FSQ-vs-BSQ comparison rests on. If the param budget is ever padded toward the upper Kronos_small band, it is done with a **non-vocab knob** (e.g. an optional 2-layer `W_fuse` MLP), stated explicitly.
- **Training budget:** identical optimizer (AdamW), identical dropout schedule, identical token/step count, identical data folds (A.5), identical seeds (run **≥3 seeds per cell**, report mean ± std — single-seed ablation numbers are not publishable). The Stage-1 tokenizer reconstruction loss is **Huber (smooth-L1, δ=1.0)** for both coarse and fine legs, with Stage-1 tokenizer dropout default `0.0` — both per the Tokenizer section's canonical config, referenced not restated.
- **Eval:** identical Q4 test set, identical cost model (including funding), identical execution filter (`θ = κ·c_total`, same tuned `κ`), identical horizons (all produced by autoregressive rollout). The *only* differences are the two manipulated factors.

For OHLCV-only cells (1, 3), the micro/perp dims are **removed from the tokenizer input entirely** (not zero-filled) so the tokenizer's capacity is spent only on the `F = 7` OHLCV subset — otherwise cell 1 is not a faithful Kronos reproduction. The mask convention still applies to the OHLCV fields.

##### C.3 Validating the Kronos reproduction (cell 1) against public weights

Cell 1 must be a *trustworthy* Kronos_small reproduction or the whole comparison is suspect. Validation protocol:

1. **Architecture parity check:** assert param count, layer shapes, and tensor flow match the published Kronos_small (≈24.7M, OHLCV input) within the param tolerance; diff the config against the released spec.
2. **Public-weights benchmark:** load the **official released Kronos_small weights** and run *our* eval harness on a *common public slice* (a fixed set of symbols/dates that overlaps Kronos's evaluation domain, e.g. the BTC/ETH 1m window in their setup). Record their IC/RankIC/MAE/R² under our metric code.
3. **Reproduction tolerance gate:** our from-scratch cell-1 checkpoint, trained on our data, must reach **IC/RankIC within a stated tolerance band of the public-weights numbers on that common slice** (target: within ~10–15% relative on RankIC; if our data domain differs, we report the gap and its cause rather than hiding it). If cell-1 underperforms public Kronos beyond tolerance, **the ablation is blocked** — we are not allowed to claim FSQ beats a *crippled* baseline.
4. **Metric-code cross-check:** compute IC/RankIC on the public weights with both our harness and (where feasible) the authors' reported protocol; agreement validates our metric implementations themselves.

This makes cell 1 simultaneously our baseline *and* an external calibration of the entire harness.

---

#### D. Secondary diagnostics (subordinate to the headline)

These localize *why* a cell wins or loses. Reported for every cell, never headlined over net-IR.

##### D.1 Price/return forecast — IC and RankIC

Over the dense (stride-1) decision grid, for horizon `h`, with predicted `μ̂_{t,h}` and realized `y_{t,h} = log(C_{t+h}/C_{t+1})`:

```
IC      = corr_Pearson( μ̂_{·,h},  y_{·,h} )            # linear forecast skill
RankIC  = corr_Spearman( μ̂_{·,h}, y_{·,h} )            # monotone/rank skill (robust to tails)
```

Computed **per (symbol, fold)** then aggregated; report **mean IC, ICIR = mean(IC)/std(IC)** across folds (the stability of skill), and the same for RankIC. Cross-sectional IC (rank across symbols at each timestamp) is reported additionally since the strategy trades a universe.

##### D.2 Realized-volatility forecast — MAE and R²

The model predicts next-window realized vol `σ̂_{t,h}` (derivable from the rollout distribution's dispersion); target `σ_{t,h}` = causal realized vol over `[t+1,t+h]` (same estimator as the vol-scaling section, applied *post-hoc* on realized bars):

```
MAE = mean_t | σ̂_{t,h} - σ_{t,h} |
R²  = 1 - Σ_t (σ_{t,h} - σ̂_{t,h})²  /  Σ_t (σ_{t,h} - σ̄)²
```

`R²` measured against the **realized-vol mean baseline** `σ̄` (causal) *and* against a **persistence/EWMA-vol baseline** (the honest bar to beat — predicting "vol = recent vol" is strong; we must beat it, and we report the gap).

##### D.3 Synthetic K-line generation — discriminative score + TSTR

Sampling full trajectories from the AR model (Monte-Carlo, §inference) produces synthetic K-line sequences; quality is two-pronged (Kronos protocol):

- **Discriminative score:** train a separate classifier to distinguish real vs synthetic length-`L` windows; score = `|accuracy − 0.5|` (0 = indistinguishable, 0.5 = trivially separable). Lower is better. Classifier architecture and train/test split are fixed and content-hashed.
- **TSTR (Train-Synthetic-Test-Real):** train the downstream forecaster (a fixed small probe model) on **synthetic** data, evaluate on **real** held-out data; report the probe's RankIC under TSTR vs the TRTR (train-real-test-real) ceiling. TSTR/TRTR ratio measures whether synthetic data carries the real predictive structure.

##### D.4 Multi-horizon quantile quality (pinball loss + PI coverage)

Because every forecast is distributional (rollout MTP distribution + sampling), we audit the *distribution*, not just the median, at each horizon `h ∈ {1,5,15,60}` (all produced by rollout):

- **Pinball (quantile) loss** at quantile levels `q ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}`:
  ```
  L_q(y, ŷ_q) =  q·(y - ŷ_q)        if y >= ŷ_q
                 (1-q)·(ŷ_q - y)    otherwise
  ```
  Report mean pinball averaged over levels and per-horizon; lower is better. This is the proper scoring rule rewarding calibrated *and* sharp distributions.
- **Prediction-interval coverage (PICP) + width (MPIW):** for the central `(1−α)` interval (e.g. 80%, 90%), empirical coverage
  ```
  PICP = (1/N) Σ_t  1[ ŷ_low_t  <=  y_t  <=  ŷ_high_t ]
  ```
  must track the nominal level (a calibration table: nominal 0.80 → measured ≈0.80). Report **calibration error `|PICP − nominal|`** at each horizon and the mean interval width (sharpness, lower better at equal coverage). A model that is right on the point but miscalibrated on the interval is flagged — v2 bet-sizing depends on this calibration being honest.

---

#### E. Backtest-honesty commitments (binding on every reported number)

These are not optional disclosures; they are CI-enforced conditions on what may be published.

1. **Provenance on every number.** Every metric in the paper, model card, and demo ships with its `(weights_hash, dataset_hash, frozen_stats_hash, eval_config_hash, seed_set)`. A reader can in principle reproduce any single number from the content-hashed raw Parquet. No number without its hash tuple is allowed in the paper.

2. **Publish the losses.** We report Q1/Q2/Q3/Q4 *all four quadrants* (not just the flattering one), the **per-symbol net-return distribution as a box-plot** (so a single lucky coin cannot carry the result), the **cross-sectional breadth** (concurrent-position count distribution behind each portfolio series), every horizon (`1/5/15/60` — including horizons where we lose), and the cost-stress curve (IR at `c_total = 0.10/0.20/0.30%`, with funding stressed). If the strategy is unprofitable after costs (execution + funding) on the long tail of illiquid symbols, that is stated plainly.

3. **No cherry-picking — pre-registered config.** The eval config (horizons, cost params incl. funding, filter multiple `κ`, leading-embargo `E`, `K`, symbol split, seeds) is **frozen before** the held-out Q4 evaluation and content-hashed. `κ` is tuned on the VAL folds only; **Q4 is touched once** per checkpoint. Any post-hoc config change forces a re-hash and is logged as a separate, labeled run; the paper distinguishes pre-registered from exploratory numbers.

4. **Multiple-testing discipline.** With 4 cells × ≥3 seeds × multiple horizons × the `κ ∈ {1,1.5,2,3}` filter search we are running many trials; we report the **Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014) and/or a **Probability of Backtest Overfitting (PBO)** via combinatorially-purged splits, discounting the headline for the number of effective trials (including the filter-tuning trials). A result that does not survive deflation is reported as not-significant, not dropped.

5. **Leakage gate.** The transform-agnostic pipeline-purity invariant (entire feature/mask/`segment_id`/target pipeline a pure function of raw data with effective timestamp `≤ t+1`, with test bars drawn from bad-tick-adjacent, segment-boundary, structural-break, and stale-run regions) is re-run on a Q4 sample as a *merge gate on the harness*; the **leading-embargo** flatness check (A.4) is reported. If IR is not flat in the leading embargo `E` past 120 bars, the result is quarantined as leaking until explained.

6. **Honest degradation forecast (mandatory, stated up front).** We publish, *before* any live deployment, an explicit expected gap between backtest Q4 net-IR and live net-IR, with its causes itemized: (i) **slippage realism** — our linear impact + half-spread understates true impact on the illiquid tail and during volatility spikes; expect cost realization worse than modeled. (ii) **funding drift** — realized funding can diverge from the recorded settlement series during stress; the funding leg may bite harder live. (iii) **non-stationarity / alpha decay** — 1m crypto microstructure edges decay as others trade them; expect signal half-life of weeks–months, not the static backtest. (iv) **fill uncertainty** — we assume taker fills at `C_{t+1}`; real fills miss, partially fill, or move the price. (v) **regime shift** — Q4 is one future block; a regime unseen in training degrades further. **Stated forecast: live net-IR is expected to realize at roughly 40–60% of backtest Q4 net-IR after these frictions, and a backtest Q4 net Sharpe above ~3 is treated as a red flag for overfitting (triggering the §E.4 investigation), not a success.** The paper's headline claim is framed as a *controlled relative* result — FSQ+micro vs BSQ baseline under the identical harness, and as **incremental** net-IR over the best naive benchmark (§B.4) — which is robust to this absolute degradation even though the absolute IR is not.

These six commitments are what make the harness co-equal with the model: they are the difference between a publishable research artifact and a curve-fit.

---

**Sources:**
- [Purged cross-validation — Wikipedia (López de Prado, 2017/2018)](https://en.wikipedia.org/wiki/Purged_cross-validation)
- [Deflated Sharpe Ratio — Wikipedia (Bailey & López de Prado)](https://en.wikipedia.org/wiki/Deflated_sharpe_ratio)
- [Bid-Ask Spread / crypto effective spreads — CoinMetro](https://www.coinmetro.com/glossary/bid-ask-spread)
- [Crypto spread (Bitcoin ≈0.02–0.04% on majors) — Bitcoin.com Markets](https://markets.bitcoin.com/glossary/spread)

---

## 9. Repository Layout, Tooling, Release Plan and Future-Work Roadmap

This section is the operational contract for the `trikaal` artifact: the directory structure, the engineering-hygiene gates (with the causal-safety test as a hard CI merge gate), the release surface (HuggingFace weights + model card + paper + live demo), and the firewalled future-work roadmap. Every path is relative to the repo root. The guiding discipline: **one research claim (the microstructure-aware FSQ tokenizer), everything else is reproducible harness.** The repo layout makes that separation legible to a reviewer at a glance.

---

### 1. Repository tree

```
trikaal/                                  # git root; PyPI-installable package `trikaal`
├── pyproject.toml                         # single source of build + deps + tool config (§2)
├── README.md                              # quickstart, claim, 2x2 result teaser, repro commands
├── LICENSE                                # Apache-2.0 (rationale §3.5)
├── NOTICE                                 # Apache-2.0 attribution; cites Kronos (arXiv:2508.02739)
├── CITATION.cff                           # machine-readable citation for the artifact
├── .pre-commit-config.yaml                # ruff + ruff-format + mypy + schema-check hooks (§2.3)
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml                         # lint + type + unit + schema + determinism (§2.4)
│       ├── causal-safety.yml              # MERGE-GATE: truncation/perturbation test (§2.5)
│       └── release.yml                    # tag -> build wheel + push HF weights/card (§3)
│
├── src/
│   └── trikaal/
│       ├── __init__.py                    # exports version, top-level API (load_model, forecast)
│       ├── _version.py                    # single version string (setuptools-scm or static)
│       │
│       ├── tokenizer/                     # ===== THE HEADLINE SUBSYSTEM =====
│       │   ├── __init__.py
│       │   ├── encoder.py                 # bar-vector -> latent: small Transformer AE encoder
│       │   ├── fsq.py                     # Finite Scalar Quantization (the novel leg)
│       │   ├── bsq.py                     # Kronos BSQ reimpl — ABLATION BASELINE ONLY
│       │   ├── decoder.py                 # latent -> reconstructed 16-d bar vector
│       │   ├── hierarchy.py               # coarse->fine split, per-stage learnable scaling (§Correction 1)
│       │   ├── model.py                   # TokenizerAE: encoder+quant+decoder, recon-loss heads
│       │   └── losses.py                  # L_coarse + L_fine + (BSQ-only) lambda*L_quant
│       │
│       ├── model/                         # AR backbone — EXACT Kronos_small parity
│       │   ├── __init__.py
│       │   ├── attention.py               # from-scratch MHA: RoPE, KV-cache, FlashAttn-2 path
│       │   ├── block.py                   # Pre-LN + RMSNorm transformer block
│       │   ├── embeddings.py              # token embed + learnable temporal embeds (min/hr/dow/dom/moy)
│       │   ├── cross_attn.py              # intra-block coarse->fine cross-attention
│       │   ├── predictor.py              # TrikaalAR: full decoder-only causal model
│       │   └── mtp.py                     # Multi-Token Prediction causal-chain aux heads (§Addition 1)
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingest.py                  # Binance CSV dumps (klines+aggTrades) + funding/OI API -> raw
│       │   ├── raw_store.py               # immutable raw layer + content-hash manifest
│       │   ├── reduce.py                  # aggTrades -> per-bar microstructure (feature-spec §2)
│       │   ├── normalize.py               # causal EWMA / rolling z-score (feature-spec §3)
│       │   ├── features.py                # assembles the canonical 16-vector + mask + ts
│       │   ├── quality.py                 # vol_recon_err, gap/segment detection, QA report (§4)
│       │   ├── segments.py                # price-gap -> contiguous-segment splitter (feature-spec §4.1)
│       │   ├── dataset.py                 # Parquet(part by symbol/freq/date) -> DuckDB -> torch Dataset
│       │   └── targets.py                 # vol-scaled regime-relative target builder (§Addition 2)
│       │
│       ├── train/
│       │   ├── __init__.py
│       │   ├── train_tokenizer.py         # stage-1 entrypoint
│       │   ├── train_predictor.py         # stage-2 entrypoint (consumes frozen tokenizer)
│       │   ├── ddp_utils.py               # DDP/NCCL init, rank-aware logging, grad-accum
│       │   ├── checkpoint.py              # atomic, content-hashed checkpoints + resume
│       │   ├── schedules.py               # AdamW + cosine-with-warmup, dropout schedule
│       │   ├── loop.py                    # shared train/val loop, W&B logging, seed pinning
│       │   └── config.py                  # dataclass config schema + loader/validator (§2.6)
│       │
│       ├── eval/                          # ===== FIRST-CLASS SUBSYSTEM (co-equal) =====
│       │   ├── __init__.py
│       │   ├── walkforward.py             # purged walk-forward + embargo splitter (MANDATORY)
│       │   ├── costs.py                   # transaction-cost model (0.1–0.3%/trade, cited)
│       │   ├── forecasting.py             # IC / RankIC (SECONDARY diagnostics)
│       │   ├── volatility.py              # realized-vol MAE / R^2 (secondary)
│       │   ├── generative.py              # synthetic K-line discriminative score, TSTR
│       │   ├── backtest.py                # execution-filtered backtest -> NET IR (HEADLINE)
│       │   ├── baselines.py               # naive / EWMA / Kronos-repro reference predictors
│       │   ├── metrics.py                 # shared metric primitives (bootstrapped CIs)
│       │   └── run_eval.py                # one config -> full eval table (incl. the 2x2)
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── sampler.py                 # temperature + top-p nucleus; MC trajectory averaging
│       │   ├── spec_decode.py             # MTP heads as speculative decoding (§Addition 1)
│       │   ├── kv_cache.py                # standard KV-cache (NO MLA — §skipped)
│       │   └── distribution.py            # assemble full predicted distribution (§Addition 3)
│       │
│       └── utils/
│           ├── __init__.py
│           ├── hashing.py                 # content-hash (dataset, config, checkpoint, stats table)
│           ├── seeding.py                 # global seed pin (torch/numpy/python/cudnn-deterministic)
│           ├── registry.py                # named-component registry for ablation matrix
│           └── logging.py                 # structured run manifest writer
│
├── configs/                               # YAML; every run is fully described by one file
│   ├── data/
│   │   ├── universe_top100_usdt.yaml      # symbol list, date range, freq spine
│   │   └── features.yaml                  # eps_*, H/W per group, W_tau, q, clip, p_drop
│   ├── tokenizer/
│   │   ├── fsq_base.yaml                  # FSQ dims+levels (empirically derived — §Correction 1)
│   │   └── bsq_base.yaml                  # Kronos BSQ baseline at matched bits-per-token
│   ├── predictor/
│   │   └── kronos_small.yaml             # 8L, d_model 512, d_ff 1024, 8 heads; dropouts; LR
│   ├── eval/
│   │   ├── walkforward.yaml               # fold boundaries, embargo, purge gap
│   │   └── costs.yaml                     # per-trade bps, taker/maker, slippage model
│   └── ablation/
│       └── matrix_2x2.yaml                # {BSQ,FSQ} x {OHLCV, +microstructure} at fixed bits
│
├── scripts/
│   ├── download_binance.py                # pull CSV dumps + funding/OI; verify checksums
│   ├── build_parquet.py                   # raw -> partitioned Parquet
│   ├── run_ablation_2x2.py                # launches the 4 cells, collects the central table
│   ├── export_hf.py                       # weights -> HF repo (tokenizer/AR/MTP) + card
│   ├── make_figures.py                    # paper figures from eval artifacts
│   └── reproduce_paper.sh                 # one command: data-hash -> table -> figures
│
├── tests/
│   ├── conftest.py                        # fixtures: tiny seeded synthetic stream
│   ├── data/
│   │   ├── test_causal_safety.py          # ★ TRUNCATION + PERTURBATION test (feature-spec §6)
│   │   ├── test_reduce.py                 # aggTrades buy/sell classification, bucketing
│   │   ├── test_normalize.py              # EWMA causality vs rolling-window reference
│   │   ├── test_segments.py               # gap-splitting, warm-up, no straddle
│   │   └── test_quality.py                # vol_recon_err, spot zero-fill flags
│   ├── tokenizer/
│   │   ├── test_fsq.py                    # straight-through grad, bounded error, level count
│   │   ├── test_hierarchy.py              # coarse-alone recon, residual-decay guard
│   │   └── test_roundtrip.py              # encode->decode shape + recon-fidelity threshold
│   ├── model/
│   │   ├── test_attention.py              # RoPE correctness, causal mask, KV-cache equivalence
│   │   ├── test_predictor.py              # forward shapes, param count == Kronos_small target
│   │   └── test_mtp.py                    # causal-chain conditioning, no future-leak across depths
│   ├── eval/
│   │   ├── test_walkforward.py            # purge+embargo: train/test disjoint, no leak
│   │   └── test_costs.py                  # cost model monotonicity, filter threshold logic
│   ├── test_determinism.py               # same seed+config -> bit-identical outputs
│   └── test_config_schema.py             # every shipped config validates against schema
│
├── paper/
│   ├── trikaal.tex                        # single-hook framing; 2x2 table = central result
│   ├── refs.bib                           # Kronos, FSQ, BTC-cost lit, AMH, DeepSeek-MTP
│   ├── figures/                           # generated by scripts/make_figures.py (git-ignored src)
│   └── tables/                            # 2x2 ablation, eval headline (auto-generated)
│
├── demo/
│   ├── app.py                             # live/replayed 1m stream -> multi-horizon distribution
│   ├── stream.py                          # Binance WS (live) or Parquet replay adapter
│   ├── plot.py                            # fan-chart prediction intervals (1/5/15/60 min)
│   └── README.md                          # how to run live vs replay; HF Space deploy notes
│
├── docs/
│   ├── architecture.md                    # subsystem map; links each module to a spec section
│   ├── data_recipe.md                     # the training-data recipe (mirrors model-card)
│   ├── reproducibility.md                 # content-hash chain; how to replay any prediction
│   └── model_card.md                      # source-of-truth model card (ships to HF — §3.2)
│
└── requirements/
    ├── base.txt                           # runtime deps (pinned, hash-locked)
    ├── train.txt                          # +flash-attn, +wandb, +ddp extras
    └── dev.txt                            # ruff, mypy, pytest, pre-commit, pytest-cov
```

**Layout rationale (what a reviewer should read off the tree):**
- `tokenizer/` is the only subsystem with both `fsq.py` and `bsq.py` present — BSQ exists *solely* as the ablation baseline, signaling the single controlled comparison. `hierarchy.py` is broken out because the coarse→fine residual-decay fix (per-stage learnable scaling, Correction 1) is the most error-prone part of the novel leg and deserves its own tested module.
- `eval/` is sized like a peer of `model/`, not a util folder — enforcing the "co-equal subsystem" mandate. `backtest.py` (NET IR) sits above `forecasting.py` (RankIC) in importance; the file ordering and the model card reflect that.
- `data/targets.py` (vol-scaled targets) and `model/mtp.py` are deliberately small, isolated files: they are SECONDARY additions and must not bloat the core. `inference/spec_decode.py` reuses `mtp.py` rather than duplicating it.
- `src/`-layout (not flat) so tests run against the *installed* package, catching packaging bugs before release.

---

### 2. Engineering hygiene

The repo is **deterministic, config-driven, and seed-pinned** end to end (locked decision). Every gate below is enforced in CI; nothing merges to `main` without all green.

#### 2.1 Build & dependency management — `pyproject.toml`
Single PEP 621 source of truth. `[project.optional-dependencies]` defines `train`, `dev`, `eval` extras mirroring `requirements/`. Runtime deps pinned with upper bounds; `requirements/*.txt` are hash-locked (`pip-compile`) for byte-reproducible environments. `setuptools-scm` derives version from git tags so the shipped wheel version and the HF model-card version cannot drift.

#### 2.2 Linting & formatting — ruff
`ruff` (lint) + `ruff format` (formatter) configured in `[tool.ruff]`. Enabled rule sets: `E,F,W` (pyflakes/pycodestyle), `I` (import sort), `B` (bugbear), `UP` (pyupgrade), `NPY` (numpy), `PD` (pandas-vet on `data/`), `RUF`. `line-length = 100`. The one project-specific custom lint we enforce by `flake8`-style grep-test in CI: **no bare `log(` / division without an `eps`** in `data/` (stability rule, feature-spec §5) — implemented as a `tests/data/test_no_unsafe_ops.py` AST check.

#### 2.3 pre-commit
`.pre-commit-config.yaml` runs on every commit: `ruff` + `ruff-format` + `mypy` (changed files) + `check-yaml` + `end-of-file-fixer` + `trailing-whitespace` + a **config-schema hook** that validates any touched `configs/**/*.yaml` against the dataclass schema (§2.6). Fast hooks only; the heavy causal-safety + determinism tests run in CI, not pre-commit.

#### 2.4 Typing — mypy
`mypy --strict` on `src/trikaal/` (tests excluded from strict). Public APIs (`load_model`, `forecast`, `TokenizerAE`, `TrikaalAR`) are fully typed including tensor-shape intent in docstrings. Numerical kernels may use `numpy.typing.NDArray`. Type-check is a CI gate.

#### 2.5 Testing — pytest
`pytest` with `pytest-cov`; coverage gate **≥ 85%** on `src/trikaal/data/` and `src/trikaal/eval/` (the leakage-sensitive subsystems get the strictest bar), ≥ 70% elsewhere. Tests are seeded and run on a tiny synthetic fixture (`conftest.py`) so the whole suite finishes in CI minutes. Markers: `@pytest.mark.slow` for anything touching real Parquet (excluded from PR CI, run nightly).

#### 2.6 Config-schema validation
Every run is one YAML validated against a frozen dataclass schema in `train/config.py` (and `eval`/`data` analogues). Validation checks: required keys present, types correct, value ranges (e.g. `top_p in (0,1]`, `q in {95,99,99.5}`, `clip == [-5,5]`), and **cross-field invariants** (e.g. FSQ vs BSQ configs must declare the *same* bits-per-token for an ablation cell — the fairness constraint from the locked decisions, asserted at load time, not discovered after a 10-hour run). Invalid config = immediate, descriptive failure. CI runs `test_config_schema.py` over all shipped configs.

#### 2.7 Determinism test
`tests/test_determinism.py`: run a 50-step micro-train of both tokenizer and predictor twice with identical seed+config on CPU (and CUDA if available with `deterministic=True`), assert **bit-identical** weights, losses, and emitted tokens. This guards the "reproduce any prediction on any date" promise. Seeding centralised in `utils/seeding.py` (python/numpy/torch + `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG`). The content-hash of (dataset, config) is asserted stable across the two runs.

#### 2.8 Causal-safety / lookahead test — the hard merge gate
This is a **first-class, separate CI workflow** (`.github/workflows/causal-safety.yml`) and a required status check on `main`. It executes the feature-spec §6 invariant:

- **Truncation test:** for a seeded sample of `(symbol, bar t)`, compute `(x_t, m_t)` on the full stream, then on a stream truncated to effective-timestamp `≤ t+1`; assert **bit-identical** (float64 pre-cast).
- **Perturbation test:** corrupt/delete/sign-flip all raw data strictly after `t+1`; re-run; assert `(x_t, m_t)` unchanged. Any future-dependence fails here.
- Coverage targets the highest-risk features explicitly: the rolling `tau` percentile (§2.3), the causal forward-fills of funding/OI (§1), the EWMA recursion (§3.2), and — if `vol_scale_inputs` is on — the vol-scaling input hook (§7).
- A complementary **per-batch fast assertion** lives in the training loop (`train/loop.py`): re-derive `(mu_{t-1}, var_{t-1})` used to normalize bar `t` and assert no contribution from `f_{≥t}`.

**This gate cannot be bypassed.** A green causal-safety check is a precondition for merge and for any release tag. It is the engineering embodiment of the project's central credibility risk: a single lookahead bug invalidates every backtest number in the paper.

#### 2.9 CI matrix (`ci.yml`)
On every push and PR to `main`: Python `{3.11, 3.12}` × OS `{ubuntu-latest}` → `ruff check` → `ruff format --check` → `mypy --strict` → `pytest` (fast markers) → config-schema → determinism. `causal-safety.yml` runs in parallel as its own required check. `release.yml` triggers only on a `v*` tag.

---

### 3. Release plan

The release is a **research artifact**, not a product: weights + model card + paper + live demo, all reproducible from content-hashed inputs. The single claim — microstructure-aware FSQ tokenizer — frames every release surface.

#### 3.1 HuggingFace weights
Three artifacts under one HF model repo `trikaal/trikaal-small-v1`, each a separately loadable checkpoint with its own `config.json` and content-hash:
1. **Tokenizer** (`tokenizer/`): encoder + FSQ quantizer + decoder. Ships the FSQ dims/levels table, per-stage learnable scales, and the BSQ-baseline checkpoint alongside (so reviewers can rerun the 2×2 themselves).
2. **AR predictor** (`predictor/`): the Kronos_small-config decoder-only backbone.
3. **MTP heads** (`mtp/`): the causal-chain auxiliary heads (loadable independently; the base AR runs without them for 1-step-only inference).

Also shipped in the repo: the **frozen-stats normalization table** (per symbol, per feature `(mu, var)` at the train/test boundary — feature-spec §3.4) and the **dataset content-hash manifest**, so the published eval is a pure function of `(weights, frozen-stats, raw test bars)`. `scripts/export_hf.py` builds and pushes this; `release.yml` automates it on tag.

#### 3.2 Model card (`docs/model_card.md` → HF)
Mandatory sections (a reviewer/user must find each):
- **Intended use & out-of-scope use:** research forecasting of crypto 1m K-lines; explicitly NOT a deployed trading system.
- **Training-data recipe:** Binance spot + USDT-perps, top ~100–200 USDT pairs, ~2019–present, 1m spine, ~500M–1B bars, multi-stage cleaning, segment-splitting on gaps. Mirrors `docs/data_recipe.md`; cites the content-hash manifest.
- **Architecture:** Kronos_small AR config + the FSQ tokenizer (the novel leg) + MTP heads, with the explicit statement that the backbone is held at Kronos parity *on purpose* for a clean comparison.
- **Eval results — INCLUDING LOSSES:** the 2×2 ablation table (the central result), the **headline cost-aware NET Information Ratio** under the execution filter with the stated transaction-cost band, then secondary IC/RankIC/MAE/R²/generative/backtest metrics. Report final tokenizer reconstruction losses (`L_coarse`, `L_fine`, per-stage residual contribution) and predictor train/val loss curves — not just downstream metrics — so the artifact is honestly auditable.
- **Limitations:** 27M params saturates ~1–2B bars; crypto-only, Binance-only; 512-bar context; spot pairs have structurally-absent funding/OI; backtest is purged walk-forward but still historical (no live-trading guarantee).
- **The explicit TFI-not-OFI caveat** (Correction 2): microstructure imbalance is **trade-flow imbalance (TFI)** computed from `aggTrades` taker side, NOT orderbook order-flow imbalance (OFI). True OFI requires L2 depth, which is **explicit v2 future work**. State plainly that a TFI signal is the weaker trade-based cousin of OFI.
- **Not-financial-advice notice:** prominent, unambiguous — outputs are model predictions for research, not investment advice; no warranty; crypto markets carry total-loss risk.
- **Reproducibility block:** the exact `reproduce_paper.sh` command, the dataset/config/weights hashes, and the frozen-stats table reference.

#### 3.3 Paper (`paper/`)
**Single-hook framing.** Abstract and contributions state exactly one claim: a microstructure-aware FSQ tokenizer for financial K-lines, with two mechanically-independent legs (FSQ-replaces-BSQ; free microstructure features in the per-bar vector) under one umbrella. **The 2×2 table {BSQ, FSQ} × {OHLCV-only, +microstructure}, at matched bits-per-token, is the central result/figure.** Everything else (Kronos-parity backbone, MTP, vol-scaled targets, eval harness) is positioned as inherited method or production harness, *not* a second research claim — keeping the paper honest and unrejectable on scope-creep grounds. The cost-aware NET-IR-under-execution-filter result is the headline downstream metric; RankIC etc. are reported as secondary diagnostics. Purged walk-forward + embargo and the cited transaction-cost model (0.1–0.3%/trade) are stated as method, not afterthoughts.

#### 3.4 Live demo (`demo/` → HF Space)
A Gradio/Streamlit app that consumes a **live Binance 1m WS stream (or a deterministic Parquet replay)** and renders a **multi-horizon distributional forecast** at 1/5/15/60 min using the MTP heads + nucleus sampling + MC trajectories. The visual is a **fan chart of prediction intervals** (e.g. 50/80/95% bands), never a single point line — directly demonstrating the "distributional output by contract" principle (Addition 3). The demo carries the same not-financial-advice banner and a "replay mode" toggle so the published artifact is reproducible offline.

#### 3.5 License
**Apache-2.0** (over MIT): the explicit patent grant and `NOTICE`-based attribution are the right fit for a model-weights research release that builds on Kronos, and it is the de-facto standard for HuggingFace model artifacts. `NOTICE` attributes Kronos (Shi et al. 2025, arXiv:2508.02739) and any vendored ideas; `CITATION.cff` makes citing Trikaal itself trivial.

---

### 4. Future-work roadmap — firewalled from v1

The roadmap is a **firewall, not a backlog**: nothing below is in v1, and the repo is structured so that a v1 reviewer can verify the v1 scope is closed and self-contained before any v2 work begins. v1 ships the **controlled comparison to Kronos_small with the FSQ-tokenizer claim proven by the 2×2 table** — and stops there.

#### 4.1 What is deliberately NOT in v1 (and why)
| Excluded | Why it's out of v1 |
|---|---|
| **MLA (Multi-head Latent Attention)** | Solves a KV-cache memory problem that does not exist at 27M params / 512 context. Plain efficient attention + standard KV-cache suffices. Revisit only at base-class scale / long context. |
| **Architecture-level latency micro-optimization** | Sub-50ms serving is a *serving-layer* concern (TorchScript/ONNX export, batching, KV-cache), not an architecture concern. Keep the model clean. |
| **Equities, Bybit/OKX, orderbook depth, base-class scale** | Parked to keep the v1 universe (crypto, Binance, 1m) and the comparison controlled. |
| **Orderbook OFI** | We have only free `aggTrades` → **TFI**, not OFI. Promising OFI without L2 data is a microstructure misrepresentation; deferred to v2. |
| **Meta-labeling / bet-sizing (the "size")** | v1 builds the **side** (a calibrated distributional forecast). **Size** is a separate learned, cost-aware layer — v2. The side/size split is deliberate. |

This discipline is the point: a 27M model with one sharp, well-controlled claim is more defensible than a sprawling one. The CI causal-safety gate + the 2×2 fairness constraint (matched bits-per-token) are what make the single claim credible.

#### 4.2 v2
- **Regime-conditioning (v2 headline candidate).** Detect and condition on the active market regime (calm/volatile, trending/mean-reverting), grounded in the **Adaptive Market Hypothesis** (markets as populations of behavioral species whose mix shifts over time). The psychology signal is *already in our 16-vector* — volume = attention, funding = positioning/greed, realized-vol = fear, TFI = herding — so no external sentiment feed is needed. This is the natural next research hook precisely because v1 deliberately kept the backbone regime-agnostic.
- **Meta-labeling / bet-sizing layer.** A learned, cost-aware execution filter that sits on Trikaal's v1 distributional output and decides position size against forecast confidence. This is where the real trading edge lives; v1's distributional-by-contract design is the substrate it plugs into.
- **Orderbook OFI.** Add L2 depth data and upgrade **TFI → true OFI**, closing the Correction-2 caveat. Requires a new ingest path (depth snapshots) outside v1's free-data envelope.

#### 4.3 v3
- **Cross-asset attention.** Joint modeling of correlated baskets (BTC/ETH/SOL) for lead-lag and contagion — a structural change to the attention pattern, intentionally out of the single-asset v1.
- **Base-class scale-up (~100M params).** Only **after** the v1 controlled comparison to Kronos_small is locked and published. Scaling before the controlled comparison would muddy the one claim v1 exists to prove.

**Firewall mechanism in the repo:** roadmap items have no stub modules, no feature flags, and no half-wired code paths in v1 (`liftor-fix`-style discipline — no dead routes/orphan handlers). They live only in `docs/architecture.md` (roadmap section) and this README section. The first v2 PR opens a `v2/` branch off a tagged, frozen `v1.0.0`; v1 `main` stays a clean, reproducible, single-claim artifact.

---

Relevant absolute path for this artifact: the repo root will be `/Users/lakshaybhati/Downloads/MY AI MODEL/trikaal/` (currently only `/Users/lakshaybhati/Downloads/MY AI MODEL/docs/` exists; this section defines the full tree to be scaffolded under that root).
