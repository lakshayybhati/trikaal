# Trikaal v1 — Execution Roadmap

The **design** blueprint is the spec (`docs/superpowers/specs/2026-06-18-trikaal-v1-design.md`, 9 subsystems). This file is the **build order**: the sequence of milestones from here to a shipped paper, each with a testable exit gate. The builder self-sequences against this — it is the answer to "which way next?".

**Standing discipline (overrides any tempt-to-expand).** v1 is **one** sharp, controlled, honestly-evaluated claim — the microstructure-aware FSQ tokenizer at ~27M (realized 21.3M) params — *not* a race to be "the best." No v2/v3 items pulled forward (regime-conditioning, OFI/L2 depth, meta-labeling/sizing, cross-asset, base-class scale). No new "novelty." **Honest over impressive; a clean null is a valid, publishable outcome.** Anything that tempts scope expansion: *parked, we build.*

**Critical path:** M2 ✅ → M3 ✅ → **M5** → **M4** → **M6** → M7 → M8, with two hard entry gates on M6 (below). M4/M5 are *independent* (parallelizable with two builders), but with one builder **M5 runs first** — it is local/cheap and the highest-risk-to-find-late subsystem — and **M4 crosses the cloud boundary**, deferred until just before M6 so we cross into paid compute once, with a proven harness.

**Independence framing (use this language everywhere):** Cell 1 is **our BSQ baseline, externally validated against published Kronos-small** — *not* a "Kronos reproduction." No Kronos code or weights are part of the model; the public weights appear only inside the eval harness as a validation target.

---

## Milestones

**M0 — Blueprint** ✅
- **Goal:** full v1 design (9 subsystems) frozen.
- **Exit gate:** spec adversarially audited (4-lens); CLAUDE.md invariants locked.
- **Compute:** —
- **De-risk:** nothing is built against assumptions; the spec wins every conflict.

**M1 — Synthetic vertical slice** ✅
- **Goal:** prove the *whole architecture* compiles and is leak-free before downloading a byte.
- **Exit gate:** G0 (exhaustive causal sweep, 100% bar coverage + planted leaks fail), G1 (tokenizer recon MAE < 1e-3; backbone+MTP CE < 0.05), G2 (bit-identical determinism); realized params == 21,301,248.
- **Compute:** local.
- **De-risk:** a wiring/causality bug found here costs minutes, not a multi-TB re-download.

**M2 — Single-symbol real slice (BTCUSDT 2023)** ✅
- **Goal:** prove the pipeline + tokenizer work on *real* signal, and measure whether microstructure carries standalone forward signal.
- **Exit gate:** real ingest→lake (SHA-256, content-hashed manifest), causal exhaustive sweep PASS on real bars (700/700), Stage-1 FSQ converges (val MAE 0.0756, no collapse, Δ_fine alive); per-feature causal IC screen + first-cut MDE delivered.
- **Result:** signed order-flow predicts standalone — TFI |RankIC| 0.027, signed_count_imbalance 0.020 @ h=5 (CI excludes 0); magnitude channels (trade_count/dispersion/large_share) live-no-signal; funding/OI masked. Two-cell temporal MDE 0.0038. **Univariate liveness only — NOT yet incremental-over-OHLCV (that is M6's 2×2 marginal + Cell 5 placebo). |RankIC| ~0.02–0.03 is a thin edge; IC ≠ net-IR after costs.**
- **Compute:** local MPS (dev-grade; *not* a determinism-gated artifact).
- **De-risk:** proves one coin-year before the multi-TB universe; the IC screen is the cheap early read that decides whether M6's microstructure arms are worth GPU-days (→ **M6 Gate B**).

**M3 — Stage-2 AR on the single symbol** ✅
- **Goal:** prove the backbone *learns to predict* real tokens (hierarchical coarse→fine + MTP) and the generation/KV-cache path runs end-to-end.
- **Exit gate:** AR train/val loss converges and **beats a trivial next-token baseline** on BTCUSDT-2023; the KV-cache `step` path is wired and bit-equivalent to the parallel forward; produces a usable single-symbol checkpoint.
- **Result (qualified PASS):** train NLL falls 13.38→11.93 (backbone fits real tokens); best-val (step 250) beats the train-marginal CE — **coarse +0.1195, fine +0.0419 nats** (coarse/regime token predictable, fine/residual near-noise); all 4 MTP depths drop (~0.5 nats each); **KV-cache rollout bit-exact, max|Δ| 3.3e-6**; checkpoint saved + content-hashed (`predictor_hash 6d118ede…`). Honest caveats: modest margin + fast single-coin-year overfit (val min ~step 250) — the expected single-regime limit; top-1 acc near the uniform floor for 891/1225-way stochastic targets (CE is the signal); MPS dev-grade (predictor weights not bit-reproducible). Write-up: `docs/milestone3_stage2_btcusdt.md`.
- **Compute:** local MPS (dev-grade checkpoint — the last cheap single-symbol de-risk).
- **De-risk:** confirms the predictor works before the universe and the 15-cycle ablation; **its checkpoint is the fixture M5 is validated against (→ M6 Gate A).**

**M4 — Full universe ingest + cloud transition** ☐ — ⚑ **CLOUD BOUNDARY (VPS kicks in)**
- **Goal:** §2 at scale — the full, delisting-aware universe lake.
- **Exit gate:** 100–200 USDT pairs ingested (SHA-256, content-hashed manifest, **listing + delisting dates recorded**), full Parquet/DuckDB lake, causal exhaustive sweep green on a sampled cross-section of real universe bars.
- **Compute:** **cloud** (storage + parallelism around the heavy aggTrades stream). **All intensive training from here is cloud CUDA A100/H100 — never MPS.**
- **De-risk:** the heavy data-engineering lift is isolated from modeling and is independent of M5, so the two overlap.

**M5 — Eval harness (co-equal subsystem)** ☐
- **Goal:** build the entire §8 harness — train-once-eval-forward folds, Q1–Q4 quadrants, cost-aware net-IR (funding + vol-scaled spread + break-even), secondary diagnostics, DSR/PBO, MDE/effective-N, the Cell-5 placebo machinery, and the Cell-1 external validation against published Kronos-small.
- **Exit gate (vertical-slice discipline applied to eval — see Gate A):** the **full harness runs end-to-end and leak-free on the M3 single-symbol checkpoint** — folds, Q1–Q4, net-IR, placebo, DSR/PBO all execute; the §8.A.5/§8.E.5 causal-safety exhaustive sweep is re-run on a Q4 sample and passes. **The numbers are meaningless (one coin, dev-grade model) and that is the point — we are testing the machine, not the result.** Plus a **metric-code cross-check**: run published Kronos-small weights through our harness on a common BTC slice to validate our IC/RankIC/MAE/R² *implementations* against Kronos's reported numbers (§8.C.3 steps 1–2, 4). *(The full "our Cell-1 reaches within ~10–15% of Kronos" reproduction gate needs a **universe-trained** Cell-1 and therefore lives at M6, not here — you cannot validate a baseline that has not been trained yet.)*
- **Compute:** cloud.
- **De-risk:** a metric-code bug found here costs one cheap run; the same bug found after M6 costs ~15 GPU-days. This is the single most expensive bug to discover late.

**M6 — The 2×2 + placebo ablation (THE experiment)** ☐
- **Goal:** run the controlled experiment — our FSQ vs our BSQ at matched bits-per-token, × {OHLCV-only, +micro}, plus the shuffled-microstructure placebo (Cell 5).
- **Entry gates (both binding — see below):** **Gate A** (M5 validated on the M3 checkpoint) **and Gate B** (microstructure kill-switch from the M2 IC screen). **Setup gate (§8.C.3 step 3):** once Cell-1 (our BSQ baseline) is trained on the universe, it must reach within ~10–15% RankIC of published Kronos-small on a common slice (fed Kronos's own input pipeline) — else the ablation is blocked; we do not claim FSQ beats a crippled baseline.
- **Exit gate:** G3 throughput check first; then 5 cells × ≥3 seeds = **15 train-once cycles** complete; headline net-IR @ 0.30% + break-even cost + all four quadrants + the placebo verdict, **DSR-deflated**.
- **Compute:** cloud CUDA, ~15 GPU-days.
- **De-risk:** gated so it runs *only* when the harness is proven trustworthy (A) and the cheap signal says the spend is justified (B).

**M7 — Honesty pass + write-up** ☐
- **Goal:** decide the claim's fate honestly; draft paper + model card + demo.
- **Exit gate:** all four quadrants, the placebo verdict, DSR-deflated numbers, the cost-stress curve + break-even cost, and the mandatory honest degradation forecast are published. **The microstructure leg is kept only if `IR_net(Cell 4) − IR_net(Cell 2)` exceeds the placebo `Cell 5` margin and survives DSR — else it is withdrawn as an honest null, with the FSQ leg standing alone.**
- **Compute:** —
- **De-risk:** the framing is chosen by the *result*, not in advance; a clean null is a valid outcome, not a failure.

**M8 — Release** ☐
- **Goal:** ship the artifact.
- **Exit gate:** HF weights (tokenizer + predictor + MTP + the BSQ baseline) + paper + **forecasts-only** demo; every number carries its `(weights, dataset, frozen-stats, eval-config, seed)` hash tuple; Apache-2.0 for code; pipeline + content-hashes for data (Binance data not redistributed).
- **Compute:** —
- **De-risk:** reproducibility from content hashes *is* the deliverable.

---

## The two binding M6 entry gates

**Gate A — Harness proven on the M3 checkpoint (vertical-slice discipline for eval).**
Before M6, the entire M5 harness must run end-to-end and leak-free on M3's single-symbol checkpoint: folds + Q1–Q4 + cost-aware net-IR + placebo + DSR/PBO all execute, and the §8.A.5/§8.E.5 causal-safety exhaustive sweep passes on a Q4 sample. The numbers are deliberately meaningless — this proves the *machine*, not the result. **No ablation until the harness is known-good.** (Prevents discovering a metric-code bug after 15 GPU-days.)

**Gate B — Microstructure kill-switch (pre-registered, bound to the M2 IC screen).**
- **Pre-registered threshold:** ≥1 microstructure channel must have (a) non-degenerate variance (not near-constant) **and** (b) a 95% moving-block-bootstrap RankIC CI **excluding 0** with **|RankIC| ≥ 2× the two-cell temporal MDE**, at a tradeable horizon `h ∈ {5, 15}`.
- **M2 verdict: PASS** — TFI |RankIC| 0.027 (7.1× the 0.0038 MDE) and signed_count_imbalance 0.020 (5.3×) @ h=5, both CIs exclude 0. (Magnitude channels fail; funding/OI N/A — masked in the single-symbol build, re-screened at M4 when the futures-API ingest lands.)
- **Branch on FAIL** (recorded for completeness): M6 does **not** spend GPU-days on the microstructure arms (Cells 2/4/5). M6 collapses to the **FSQ-vs-BSQ comparison only** (Cells 1 & 3, OHLCV-only), and the paper reframes to the honest finding *"trade-flow microstructure carries no tradable signal at 1m crypto."*
- **Branch on PASS (current):** M6 runs all five cells — **but univariate liveness only justifies the spend; it does not prove the claim.** The microstructure claim is decided at M6/M7 by incremental net-IR over OHLCV (Cell 4 − Cell 2) exceeding the placebo (Cell 5) and surviving DSR. A thin |RankIC|~0.02–0.03 can still vanish after costs — that is exactly what M6 measures.

---

*Update this file when a milestone closes (status marker + the realized exit-gate evidence). It is the single source of build order; the spec remains the single source of design.*
