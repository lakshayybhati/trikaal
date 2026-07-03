# M6 Readiness Audit (recon only — built nothing)

**Date:** 2026-06-23 · **Branch:** `real-data-slice` · **Lake:** 200 symbols / 304.6 M bars / 7,024-file compacted lake, anchored `sha256:5dfd667d05b97bda…`.

M6 = the 5-cell ablation **{BSQ,FSQ} × {OHLCV-only, +micro} + shuffled-micro placebo (Cell 5)**, trained on the universe, scored by the M5 harness cross-sectionally. This audit inventories what exists vs what M6 must build, verifies the eval path on universe symbols, benchmarks lake reads, and restates Gate A as a binding result. **No code, no training, no cloud.**

Method: an 8-agent read-only inventory over `src/trikaal/{tokenizer,model,train,eval,data}/` + runtime recon (Gate-A tests, a lake-read benchmark, a 3-symbol eval-path smoke, and symbol×regime cell counts from the lake).

---

## 1. Code inventory — EXISTS / PARTIAL / MISSING

| Component | Status | Evidence (file:line) | M6 gap |
|---|---|---|---|
| **FSQ tokenizer** (enc+FSQ+dec, hierarchical coarse/fine, no commitment loss) | ✅ **EXISTS** | `tokenizer/fsq.py:16`, `hierarchy.py:23`, `model.py:109`, `constants.py:80` (FSQ_LEVELS, bpt 20.06 in-band); tested | none (Cell 2/4/5 quantizer) |
| **BSQ tokenizer** (Cell-1/2 baseline at matched bpt) | ❌ **MISSING** | BSQ appears only in 4 docstrings, **zero executable code** | build BSQ bottleneck (k_c=k_f=10, V=1024, bpt 20.00) reusing enc/dec, +λ·L_quant; **Cells 1 & 3 cannot be instantiated today** |
| **AR backbone + MTP** (Kronos_small) | ✅ **EXISTS** | `model/predictor.py:38-47` (8L/512/1024/8h, **base params measured = 21,301,248**), RoPE/RMSNorm/Pre-LN, coarse→fine cross-attn `cross_attn.py`, DeepSeek MTP chain `mtp.py:43`; **symbol-agnostic** (0 hits for symbol/universe) — no universe blocker | none architecturally; MC-trajectory decode wrapper (spec §8) is adjacent-new |
| **OHLCV-only vs +micro arm switch** (the 2×2 input axis) | ❌ **MISSING** | `OHLCV_ONLY_IDX` (`constants.py:61`) **consumed nowhere**; tokenizer hardwired to 16 dims (`encoder.py:19,29`); no `use_microstructure` flag in src | a config switch that **removes** micro dims (F=7) from the tokenizer input — spec: "removed entirely, not zero-filled" — so Cell-1/3 capacity is spent only on OHLCV |
| **Cell-5 shuffled-micro TRAINING placebo** | ❌ **MISSING** | only the **eval-time** surrogate exists: `eval/placebo.py:31` `block_time_permute`, `:49` `phase_randomize`, wired at scoring `harness.py:254` | a Cell-5 **training arm** = Cell 4 with micro replaced by the seeded surrogate, applied **identically in train and eval**; the surrogate generators are reusable as-is |
| **Multi-symbol universe data-loader** | ❌ **MISSING** | entire data path is single-symbol: `assemble_window` is `WHERE symbol=?` (`lake.py:76`); trainers eat pre-materialized arrays (`train_tokenizer.py:70`, `train_predictor.py:75`) | a `MultiSymbolWindowDataset`: per-symbol segment catalog, symbol sampling ∝ n_windows^α (α≈0.5), split-aware windows, **thin-coin weighting** (the 43 thin coins) |
| **5-cell training runner** | ◑ **PARTIAL** | single-cell is solid + reusable: `train_stage1` (`train_tokenizer.py:70`), `train_stage2` (`train_predictor.py:75`), **content-hashed checkpoints** (`checkpoint.py:43`); `m3_stage2_btcusdt.py` hardcodes ONE cell | a 5-cell × ≥3-seed orchestrator (cell matrix, per-cell run dirs/artifacts) + the BSQ arm + the arm switch + **W&B logging (zero `wandb` in src; not yet a dep)** |
| **Cross-sectional eval driver** (M5 over 5 models × 200 symbols) | ❌ **MISSING** | harness is single-model/single-symbol: `run_harness(lake, model, tok)` (`harness.py:128`), `single_symbol_backtest` = "each decision its own period" (`harness.py:93`); `m5_eval_harness.py` hardcodes BTCUSDT | universe eval loader + multi-model orchestration + **cross-sectional pooling** (global stride-h rebalance index across symbols) + Cell-4-vs-1 + Cell-5 placebo verdict over purged walk-forward |
| **Determinism hook** (invariant 7) | ❌ **MISSING** | `set_determinism()` pins RNG but **records nothing** (`seeding.py:16`); checkpoints store only architecture config — no seed/device/**attention mode** (`checkpoint.py:53`, `predictor.py:54-69`); FlashAttention-2 not wired (SDPA math path only) | selectable attention backend (flash2 / sdpa-deterministic) + a per-run `RunManifest` recording attention_mode/seed/device/versions — **required before CUDA training** |

**Reusable leaf primitives (M5, already KAT'd, not single-symbol-bound):** `folds.make_fold_plan` + `folds.quadrant(symbol_in_train, time_in_train)` (the Q1–Q4 two-axis labels already exist), `metrics`/`costs`/`strategy`/`dsr`/`diagnostics`/`predict.predict_mu`, and the placebo surrogate generators. The cross-sectional **wiring** is the gap, not the leaves.

---

## 2. M5-on-universe verify (data path, 3 real symbols)

Loaded BTCUSDT (2,103,840 bars, 1 seg), ZROUSDT (279,930, 1 seg), AXSUSDT (2,103,840, **2 segs** — a real gap) from the compacted lake through the **existing** eval path:
- `lake.connect` + `assemble_window` → per-symbol `x[N,16]`/mask/ts/segment_id: **works**.
- `harness.forward_log_returns(raw_ret_close, segment_id, h=15)`: **works** (finite for all but the last h per segment).
- `folds.make_fold_plan(n, k=6)`: **works** (anchored train + 6 forward blocks).

**Needs adapting for 200-symbol cross-sectional eval:** `make_fold_plan` is **time-only** (single series); `single_symbol_backtest` treats each decision as its own period (no pooling). The Q1–Q4 **labels** exist (`quadrant`) but the **symbol-partition + cross-sectional pooling + frozen-stats-per-fold** are the M6 driver's job. Verdict: **data path ready; cross-sectional driver M6-new.**

---

## 3. Lake-read benchmark (compacted 7,024-file lake)

| Read | Throughput |
|---|---|
| Full-universe `count(*)` (parquet stats) | 304.6 M bars in 0.6 s → **472 M bars/s** |
| Full-universe 3-column aggregate **scan** | 1.2 s → **248 M bars/s** |
| Per-symbol `assemble_window` (x[N,16]+mask+ts → numpy) | 6 symbols / 4.65 M bars in 2.5 s → **~1.88 M bars/s** |

**Verdict: lake reads will NOT bottleneck the GPU.** Even the heaviest path (full feature materialization at ~1.88 M bars/s) feeds a 27 M-param model an order of magnitude faster than it can consume tokens; a columnar scan is ~250 M bars/s. With ordinary prefetch, a single 4090 or A100 is compute-bound, not I/O-bound — the **G3 throughput gate is satisfied on the read side**. (4090 vs A100 is a VRAM/compute choice, not a data-feed choice.)

---

## 4. Determinism hook (invariant 7)

**MISSING — M6-new before CUDA training.** `set_determinism()` exists but returns nothing and records no mode; content-hashed checkpoints persist only architecture kwargs (no seed/device/attention-mode); there is no `RunConfig`/`RunManifest` object in `src/`, and FlashAttention-2 isn't wired (the current path is deterministic SDPA-math). M6 must (a) add a selectable attention backend that resolves and exposes `{flash2, sdpa_deterministic}` at runtime, and (b) record the chosen mode + seed + device + torch/CUDA/flash-attn versions per run. (Corroborated by the supervisor's own draft notes `docs/m6_supervisor_directive.md` / `docs/drift_audit_2026-06-22.md`, which flag this as OW-4.)

---

## 5. Gate A — binding-gate RESULT (not just "M5 exists")

**YES — Gate A is satisfied, with fresh evidence.**
- `tests/eval/test_gate_a.py` — **BOTH checks PASS** (just re-run): `test_gate_a_causal_sweep_leakfree_on_q4_sample` (Q4-representative anomaly slice, exhaustive truncation sweep `passed=True`, **coverage_bars == total_bars = 100%**) **and** `test_gate_a_sweep_still_has_teeth` (a planted `global_zscore` lookahead is **caught** — anti-vacuity). `tests/eval/test_harness.py::test_run_harness_end_to_end_smoke` also PASSES.
- The **full M5 harness ran end-to-end on the M3 checkpoint** (folds → rollout → net-IR → cost-stress/break-even → DSR/PBO → Cell-5 placebo → diagnostics), content-hashed, per `docs/milestone5_eval_harness.md` ("the full harness runs end-to-end and leak-free"; "gate-A PASS ✓ (420/420 bars)"). **Numbers are meaningless by design** (dev-grade single-symbol M3 model) — Gate A is a *machine-validation* gate, and the machine is validated. (Not re-run here per instruction.)

---

## 6. Eval-power input (symbol × regime cell counts)

Per-symbol coverage/thin-flags are in **`docs/_m4b_dq_table.md`** (200 rows). Realized cells from the lake:

| Regime (year) | Active symbols | Bars |
|---|--:|--:|
| 2021 bull | 122 | 54,826,034 |
| 2022 bear+FTX | 138 | 67,594,835 |
| 2023 recovery | 182 | 83,931,043 |
| 2024 ETF | 194 | 98,273,269 |

**Per-symbol regime span:** 4 regimes → **116 symbols** (the deepest cross-section), 3 → 18, 2 → 52, 1 → 14 (youngest 2024 listings). Micro-availability is ~99% with **0 micro-starved coins** (§ M4b DQ), so the +micro/placebo contrast isn't diluted. These are the inputs for the design doc to **pre-register** the cross-sectional effective-N / MDE / placebo threshold (not computed here — that's the M6 design's job).

---

## 7. GAP-LIST — what M6 must build (dependency order)

1. **BSQ quantizer** (Cells 1 & 3) — k_c=k_f=10, V=1024, bpt 20.00, reuse the FSQ enc/dec, add λ·L_quant; + a **cross-arm bpt-parity check** (|Δbpt| ≤ 0.5, ±2% param parity).
2. **Feature-arm switch** (the 2×2 input axis) — OHLCV-only path that **drops** micro dims (F=7 input), not zero-fill.
3. **Cell-5 shuffled-micro training arm** — reuse `block_time_permute`/`phase_randomize`; train tokenizer **and** predictor on the surrogate, applied identically in train + eval.
4. **Multi-symbol universe data-loader** — `MultiSymbolWindowDataset` (segment catalog, symbol sampling ∝ n_windows^α, split/embargo-aware, thin-coin weighting).
5. **5-cell × ≥3-seed training orchestrator** — cell matrix, per-cell content-hashed artifacts, **W&B logging** (add `wandb` dep).
6. **Cross-sectional eval driver** — universe eval loader + frozen-stats per fold + multi-model load + **cross-sectional pooling** (global stride-h rebalance index) + Cell-4-vs-1 + Cell-5 placebo verdict over purged walk-forward.
7. **Determinism hook** — selectable attention backend + per-run `RunManifest` (attention_mode/seed/device/versions) — *gate before CUDA training*.
8. **(Adjacent) MC-trajectory decode** — temperature/top-p sampling + vol-scale de-normalization over the backbone's `step()` (spec §8), for the distributional forecasts.

**Already in hand (reuse, don't rebuild):** FSQ tokenizer; AR backbone + MTP (21.30 M, symbol-agnostic); single-cell `train_stage1`/`train_stage2` + content-hashed checkpoints; all M5 eval leaf primitives; the fast compacted lake + `assemble_window`.

**Bottom line:** the *modeling core* (tokenizer-FSQ, backbone, MTP, single-cell training, M5 metric leaves, the lake) is built and verified; M6 is mostly **orchestration + two missing arms** — BSQ, the OHLCV/+micro input switch, the Cell-5 training arm, the multi-symbol loader, the 5-cell runner, the cross-sectional eval driver, and the determinism/attention hook. No code, training, or cloud touched in this audit.
