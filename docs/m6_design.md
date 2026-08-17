# M6 — The 5-Cell Microstructure Ablation: Design + Pre-Registration

**Author:** Supervisor / research lead · **Date:** 2026-06-23
**Status:** design locked; the eval decision rule (§2) is a PRE-REGISTRATION — committed before any M6 training number exists.
**Reads against:** the blueprint spec (`docs/superpowers/specs/2026-06-18-trikaal-v1-design.md`) for the underlying architecture; this doc adds the M6 orchestration plan + the pre-registered eval. Where they conflict, the spec wins on architecture, this doc wins on M6 procedure.
**Inputs:** the M6-readiness audit (`docs/m6_readiness_audit.md`) and the drift audit (`docs/drift_audit_2026-06-22.md`).

---

## 0. The one question

Does the microstructure-aware **FSQ** tokenizer improve **cost-aware net Information Ratio** on 1-minute crypto K-lines by **more than a shuffled-microstructure placebo** — i.e., is it the microstructure *information* that helps, not merely the added input *capacity*?

**A NULL result is a valid, pre-committed, publishable outcome.** If the microstructure gain does not exceed the placebo, the microstructure leg is withdrawn and the contribution becomes FSQ-vs-BSQ on OHLCV alone. No post-hoc threshold moving.

## 1. The five cells

All cells share: Kronos_small backbone (8L / d512 / ff1024 / 8h; 21,301,248 params EXCLUDING the MTP heads — the realized total with MTP is 31,795,200 FSQ / 31,725,568 BSQ, §7 v1.6.34), matched bits-per-token (`FSQ_BPT_BAND` 19.5–20.5 ≈ BSQ k=20), the same training draw, the same ≥3 seeds, and the same eval. The **only** varied factors are {quantizer} × {input arm}.

| Cell | Quantizer | Input arm | Role |
|---|---|---|---|
| 1 | BSQ | OHLCV-only | our BSQ baseline. **NOT externally validated** — the §8.C.3 Kronos gate was found unexecutable and **dropped as binding** (2026-08-03, prereg §7 v1.6.22); no Kronos weights were ever pulled. The compensating control is a required verdict-manifest disclosure. **This cell is the only arm with scored artifacts.** |
| 2 | FSQ | OHLCV-only | isolates the FSQ-vs-BSQ tokenizer effect |
| 3 | BSQ | OHLCV+micro | micro under BSQ |
| 4 | **FSQ** | **OHLCV+micro** | **the hero** |
| 5 | FSQ | OHLCV+**shuffled**-micro | **placebo**: Cell-4 capacity, micro *information* destroyed |

- **OHLCV-only arm** = the `OHLCV_ONLY_IDX` channels (price-shape + log-volume/amount). The encoder *takes the reduced subset* — it does NOT zero-fill the micro dims (zero-fill leaks "micro absent" and mis-matches capacity).
- **+micro arm** = OHLCV + the aggTrades-derived microstructure channels (TFI / trade-flow / large-share — the bits the +micro arm adds over OHLCV-only). Funding/OI are carried-but-mostly-masked (historically sparse/zero-filled; not the contribution under test — invariant 1 scopes the claim to **TFI/trade-flow**).
- **Cell 5 (placebo)** temporally permutes the microstructure channels *within each symbol-segment*, at **both training and eval**, preserving each channel's marginal distribution (capacity) while destroying its temporal information. Deterministic under seed.

Matched fairness is at the **token** level (bits-per-token), not the input level: each arm's encoder maps its input subset → a latent → a quantizer at the same bpt. The arm difference is *what the encoder sees*, never the token budget (invariant 6).

## 2. Pre-registered primary test + decision rule (LOCKED before training)

> This section is the pre-registration. It is authored before any M6 model is trained. The threshold-defining quantities are functions of the realized data *covariance* structure and the design — never of the treatment comparison.

- **Headline metric:** cost-aware **net Information Ratio** at **0.30%/round-trip** transaction cost (break-even cost also reported), under the forecast-magnitude execution filter (θ = κ·c_total), **purged walk-forward + embargo (E = 120)**, **cross-sectionally pooled over symbols** within each fold. IC / RankIC / MAE / R² are secondary diagnostics only.
- **Primary test — microstructure *information*:**
  - ΔIR_info ≡ IR(Cell 4) − IR(Cell 5).  *(Algebraically = (+micro gain over OHLCV) − (placebo gain over OHLCV); same number either framing.)*
  - One-sided H₁: ΔIR_info > 0, at α = 0.05.
  - **The microstructure leg SURVIVES iff:** the lower bound of the (1−α) CI of ΔIR_info > 0 **AND** ΔIR_info ≥ MDE_prereg.
- **MDE_prereg:** computed from the realized **(symbol × regime) coverage** (122/138/182/194 active symbols across 2021→2024; 116 all-4-regime symbols), with effective-N **deflated by the realized cross-symbol return correlation** (estimated per fold from residuals after a market-factor removal — crypto has a strong common factor, so this deflation is mandatory, not optional), at α = 0.05, **power = 0.80**, via the cross-sectional + moving-block bootstrap (`ic_screen` extended to the cross-section). **Computed and COMMITTED as the first M6 task, before any cell trains.**
- **Secondary tests (reported with CIs, NOT gating the headline):** the 2×2 marginals — FSQ effect [IR(2)−IR(1), IR(4)−IR(3)], micro-marginal [IR(4)−IR(2)], BSQ-micro [IR(3)−IR(1)]; and the **per-regime** breakdown of ΔIR_info (2021 bull / 2022 bear+FTX / 2023 recovery / 2024 ETF — hypothesis: micro helps more under stress).
- **NULL is valid and pre-committed:** if the primary test fails, we report it and withdraw the microstructure leg. The threshold is not moved after seeing results.

## 3. Binding gates (all must pass before the headline is claimed)

- **G-instrument-live (Gate A — confirmed LIVE, not just documented):** the **FIRST M6 action**, before any build and before any cell trains — establish a live, pinned-environment Gate-A anchor. *History (2026-06/07): the original M5 anchor `75caa9ef` failed to reproduce under unpinned-dependency drift (→ `91ed9054` under numpy 2.3.3/torch 2.11.0); before any lockfile was committed the env drifted AGAIN (numpy 2.4.6/torch 2.12.1), and the 2026-07-04 full audit found instrument defects to fix (§6 item 10) — so **no historical hash is the target**.* The procedure, fail-closed at every step: **(1)** pin the environment exactly and **commit the lockfile** (uv.lock tracked, pyproject `==` pins — reverse the gitignore); **(2)** apply the §6-item-10 instrument fixes with extended KATs; **(3)** run `scripts/m5_eval_harness.py` on the frozen M3 checkpoint **≥2× under the committed env** — the runs must agree **bit-identically**, the eval KAT suite (`tests/eval`, currently 45 tests) + both Gate-A causal tests green, and the resulting `results_hash` recorded in a **committed, durable run-manifest artifact** (hash + numpy/torch/python versions + device + seeds + input content-hashes — never stdout-only again). That hash is THE anchor from then on; any later non-reproduction is a fail-closed STOP. The KATs are hand-computed, machine-independent and bit-exact — they carry logic-integrity across the re-anchoring; a KAT or causal-gate failure at any point is a logic regression and blocks everything. [inv. 7 — prediction replay]
- **G-parity (bpt):** BSQ and FSQ tokenizers within `FSQ_BPT_BAND`; verified pre-training. [inv. 6]
- **G-§8.C.3 (external validation) — ★ NO LONGER BINDING (Lakshay, 2026-08-03; prereg §7 v1.6.22; the clause below is retained as the pre-registered text and is NOT in force). Replaced by a REQUIRED disclosure carried in every verdict manifest (`external_validation.required_disclosure`), not by a substitute gate. Original text (QUANTIFIED in prereg v1.2, which governed):** Cell 1 (BSQ + OHLCV), universe-trained, **fails iff RankIC < 0.85 × published Kronos-small** on the pinned common slice (slice, transcription timing, and the pre-committed halt-before-any-Δ / Cell-1-only-fix / full-5-cell-same-seed-retrain failure protocol are all pinned in `docs/m6_prereg.md` §6 — the earlier "~10–15% / materially off / fix" wording was an unlimited re-run license and is superseded). [inv. 4]
- **G-causal:** the eval folds pass the §8.A.5 causal sweep (leak-free + anti-vacuity planted-leak catch), as in M5 / Gate A. [inv. 2]
- **G-determinism:** every run records its attention mode (deterministic-attention fallback vs FlashAttention-2); the headline numbers are produced and reproducible under the recorded mode. [inv. 7]

## 4. Training draw (data)

- Source: the compacted universe lake (200 symbols, 304.6M bars, anchored `5dfd667d`).
- **Symbol sampling ∝ n_windows^α, α ≈ 0.5 (sub-linear)** — so the 116 all-regime symbols carry the draw without the 14 single-regime / thin recent listings being drowned *or* over-weighted. The 43 thin-by-history coins STAY in the lake (eval breadth + survivorship) but are **down-weighted in the training draw** per their usable-window count. (Thinness here is by *history*, not micro-quality — micro-availability is ~99% with 0 starved coins.)
- **Budget ≈ compute-optimal:** ~1 effective pass over the quality-weighted draw (~270–300M effective bars). Per `[[training-saturation-budget]]` this is at/just-past Chinchilla-optimal for 21.3M params — **do NOT over-epoch** (a loss plateau here is expected, not a bug).
- Per-fold **frozen** causal normalization stats; **train-once-evaluate-forward** (no per-fold retrain — M5 fold design).
- **Identical draw + seeds across all 5 cells** — the only varied factor is quantizer × arm.

## 5. Seeds + reporting

- **≥3 seeds per cell** (15 training configurations: tokenizer + AR per cell × 3 seeds). Report mean ± across-seed CI for every IR; the ΔIR tests pool across seeds **and** the cross-section.

## 6. Build plan — the 11 gaps (LOCAL dev-grade first, THEN cloud; items 9–11 added by the 2026-07-04 audit)

**Reuse, do NOT rebuild:** FSQ tokenizer, backbone + MTP (21.3M, symbol-agnostic), single-cell `train_stage1/stage2`, content-hashed checkpoint *identity* (weights+config hashing — but see item 9: train-state **resume** does not exist and is new), all M5 metric leaves (as corrected by item 10), the compacted lake.

**Phase 1 — LOCAL, zero spend (build + unit-test + causal-gate each, then a tiny 5-cell SMOKE):**
**Step 0 — HARD PRECONDITION (before ANY build or train): confirm Gate A is LIVE** (G-instrument-live, procedure above) — pin+commit the env, apply the item-10 instrument fixes, then establish the new anchor: eval KAT suite + both Gate-A causal tests green AND `results_hash` bit-identical across ≥2 runs under the committed lockfile, recorded in a committed run manifest. Fail-closed: a non-reproducing hash STOPS Phase 1 until the regression is resolved. Only on a confirmed-live Gate A proceed to items 1–11.
1. **BSQ quantizer** + the bpt-parity check (G-parity). Self-written (no imported BSQ).
2. **Feature-arm switch** — OHLCV-only DROPS the micro dims (encoder takes the `OHLCV_ONLY_IDX` subset); NOT zero-fill. The tokenizer must accept the reduced input cleanly (it is currently hardwired to 16 dims).
3. **Cell-5 shuffled-micro TRAINING arm** — temporal-permute the micro channels within each symbol-segment (distribution preserved, information destroyed), deterministic under seed. (The eval-time surrogate `placebo.py` already exists — reuse its shuffle.)
4. **Multi-symbol universe data-loader** — segment catalog over the compacted lake, symbol sampling ∝ n_windows^0.5, thin-coin weighting, per-fold frozen stats. (The whole data path is currently single-symbol.)
5. **5-cell × ≥3-seed orchestrator + W&B** logging (W&B is currently absent from `src`).
6. **Cross-sectional eval driver** — universe loader + per-fold frozen stats + cross-sectional pooling over symbols + the 5-model + placebo verdict (extends `single_symbol_backtest`).
7. **Determinism / attention-mode hook** — record per run; wire FlashAttention-2 + the deterministic fallback. [inv. 7]
8. **MC-trajectory decode (spec §8)** — **DEFER unless the headline net-IR needs full distributions.** The MTP heads already give multi-horizon point/quantile output; keep v1 scope tight (out-of-scope creep otherwise).
9. **Resumable + atomic checkpointing** *(added by the 2026-07-04 audit — the earlier "reuse checkpointing" line covered only weights+config identity)* — a full `TrainState` checkpoint (model + **optimizer state + step counter/LR-schedule position + RNG states (torch/numpy/python) + data-sampler cursor**), **atomic writes** (tmp file + `os.replace`, never `torch.save` straight onto the best artifact), and a resume-from-checkpoint path in both trainers, proven by a local tiny-scale kill-and-resume test (the local twin of pre-flight Item 3). Also add the **bf16-autocast knob** (cloud target is bf16; currently absent) so the toy CUDA run exercises the real precision.
10. **Instrument fixes** *(added by the 2026-07-04 audit — must land BEFORE the Gate-A re-anchor and the pre-registration lock, each with a KAT):*
   - **Headline at the pre-registered cost:** `ir_headline` must be the net-IR with the **0.30%/round-trip netting** (§2); today it uses the modeled ~0.11% per-bar cost while `HEADLINE_COST` is dead code and the M5 runner mislabels the number "@0.30%". Keep the modeled-cost IR as a reported secondary; positions may still come from θ = κ·c_total(modeled).
   - **Flat-period convention:** the portfolio series must cover the **full stride-h calendar grid with flat periods as 0.0** before annualization (activity/breadth reported separately). Today flat periods are *excluded* then the IR is annualized at full calendar frequency — inflating IR by ~1/√(activity) and biasing cross-cell ΔIR whenever trade rates differ.
   - **Time-aligned PBO matrix:** build the CSCV input as `[T_grid, |κ|]` with per-κ flat periods as 0 — today each κ's active-only series is truncated to min-length and column-stacked, so rows compare different time periods (statistically invalid).
   - **Persist the per-κ VAL curve** in `HarnessResult` (today only `kappa_chosen` survives) so the κ*-knife-edge is visible and the pre-registered cost-stress-curve headline is auditable.
   - **σ̄-causality regression KAT** (the expanding-mean cost fix is correct but untested — cheap to pin) and a **placebo-dims tripwire** (assert the funding/OI dims are fully masked in any eval slice; if they ever activate, Cell 5's shuffle-dim set must fail loudly, not silently exclude them).
11. **CI gate (invariant 2's required enforcement, still absent):** `.github/workflows` running ruff + the fast suite on push with the causal-safety/Gate-A tests as a required check — must exist before paid compute.

**SMOKE gate (end of Phase 1, local):** all 5 cells instantiate, train a few steps on 2–3 symbols, checkpoint, and the cross-sectional eval driver emits a (meaningless-by-design) 5-model + placebo verdict. Full suite green + ruff clean + the causal sweep green on the **multi-symbol** path. **Compute + COMMIT `MDE_prereg` here — locked before any real training.**

**Between Phase 1 and Phase 2 sits the 8-item PRE-FLIGHT GATE (`docs/m6_preflight.md`)** — including the toy-CUDA rehearsal (full 5-cell toy run + kill-resume on the real GPU type, ~$5–20). The full-budget run is authorized only when all 8 items are green with committed evidence. `setup_cloud.sh` must install from the **committed lockfile** (done: it now runs `uv sync --locked` from the committed `uv.lock`; it previously installed latest-torch + floors — the exact drift failure mode).

**Phase 2 — CLOUD, M6 proper (the real run):** provision the GPU (cloud Phase B), transfer the compacted lake, run the 5-cell × ≥3-seed training + the cross-sectional eval → the headline, under recorded determinism mode. Verify G-§8.C.3 (Kronos parity) as the first cloud check.

## 7. Cost + hardware

- ~15 GPU-days (5 cells × 3 seeds × tokenizer+AR). Data-feed is NOT the bottleneck (audit: 248M bars/s scan, 1.88M/s materialize) → GPU is VRAM/compute-bound. For 21.3M params a single **RTX 4090 (24GB, ~$150)** likely suffices vs **A100 80GB (~$450)**; decide on the on-box throughput check at Phase B. **Spend is the human gate (Lakshay).**

## 8. What this does NOT do (firewall — invariant 3 + the v1 scope)

No second research claim; no base-class (~100M) scale-up (that firewalled word does not belong on a v1 milestone — M6 *is* the training; M7 = honesty pass + write-up; M8 = release); no orderbook/true-OFI; no MLA; no architecture latency micro-opt. One headline claim: the microstructure-aware FSQ tokenizer.
