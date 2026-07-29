# M6 Pre-Registration — MDE, Decision Thresholds, and the Headline Rule (LOCKED)

**Status:** LOCKED at commit time. **This file's git commit timestamp precedes any real M6
training run** — that timestamp is the proof the decision rule existed before any result did
(m6_preflight Item 6). No threshold below may be moved after seeing results; a NULL outcome is
valid and pre-committed (m6_design §0).

**Inputs (computed, not assumed):** `scripts/m6_prereg.py` over the compacted universe lake
(anchor `5dfd667d…`), 40 deepest symbols, full forward region 2023-10-20T16:48Z (the exact 0.7
calendar boundary) → 2025-01-01 — of which the §3a **primary** region is forward blocks 1–5
(≈ calendar 2024; block 0 = VAL, excluded). Raw inputs + all rows:
`runs_manifest/m6_mde_inputs.json` (committed; `basis` field states the region).

---

## 1. The realized covariance structure (why the deflation was mandatory)

- **Cross-symbol raw correlation ρ̄_raw ≈ 0.62** at every horizon — the crypto common factor is
  as strong as m6_design §2 warned: 40 symbols carry an **effective breadth of only ~1.6
  independent symbols** (N_eff = N / (1 + (N−1)·ρ̄)). Cross-sectional pooling adds robustness and
  coverage, NOT 40× statistical power. Any power claim ignoring this would be fiction.
- **Residual correlation after market-factor removal ρ̄_resid ≈ −0.02** — the equal-weight market
  factor captures essentially all commonality; per-symbol residuals are near-independent.
- **Pooled-series temporal autocorrelation ρ₁ ≈ 0** at h=5/15 (0.03 at h=60) — periods are
  near-independent at the stride-h grid; T_eff ≈ T (the moving-block deflation barely binds).

## 2. MDE_prereg (one-sided α = 0.05, power = 0.80)

Mechanics (per `scripts/m6_prereg.py`, locked): MDE = (z₀.₉₅ + z₀.₈₀) · √(2/T_eff) ·
√(periods_per_year(h)), with T_eff temporally deflated and ρ₄₅ conservatively set to 0 (the two
cells share the eval grid, so the true SE of ΔIR is smaller — this errs toward LARGER MDE).
**Basis (recomputed 2026-07-06 per §3a, audit item 5): the PINNED primary region = forward
blocks 1–5 of k=6 (VAL block 0 excluded), i.e. 2024-01-01T18:00Z → 2025-01-01.** The v1/v1.1
table (3.209 / 3.209 / 3.313) was computed on the full forward region including block 0 and is
superseded by this table — the nuisance-basis correction §3a disclosed (T shrinks ~1/6; git
history of `runs_manifest/m6_mde_inputs.json` preserves the old rows).

| Slice | h=5 | h=15 | h=60 |
|---|---|---|---|
| **Pooled blocks 1–5 (the primary)** | **3.515** | **3.518** | **3.533** |
| 2024 (OOS regime) | 3.515 | 3.518 | 3.533 |

(Annualized net-IR units. T at h=15: 35,063 periods, T_eff 35,002. Blocks 1–5 coincide almost
exactly with calendar 2024, so the 2024 regime row now equals the pooled primary; the 2023 OOS
tail (Oct–Dec) lies inside VAL block 0 and is therefore no longer a headline-region slice — it
remains VAL/κ territory only.)

**Honest reading:** ~1 year of headline-region OOS at these horizons can only power-detect a
ΔIR_info of **≥ ~3.5 annualized IR units** at the pre-registered α/power — under the
conservative unpaired (ρ₄₅ = 0) reference; the operative paired threshold is §3's MDE_paired.
Smaller true effects will read as CI-includes-zero → the pre-committed NULL. This is the cost of
the anchored train-once design and is accepted, not negotiated after the fact.

**Per-regime scope (recorded honestly):** under the anchored 0.7 split, only late-2023 + 2024 are
out-of-sample; 2021 (bull) and 2022 (bear+FTX) lie INSIDE the train region, so their per-regime
ΔIR reads are in-sample (Q1/Q2) secondary diagnostics — reported with that label, never as OOS
evidence.

## 3. The pre-registered primary test + decision rule (m6_design §2, restated as final)

- **Primary statistic:** ΔIR_info = IR(Cell 4) − IR(Cell 5) — pooled cross-sectional net-IR,
  netted at the flat **0.30 %/round-trip**, purged walk-forward + embargo E=120, computed by
  `eval/xsection.py` (the corrected §6-item-10 conventions: full calendar grid flat=0.0,
  time-aligned PBO, per-κ curve persisted).
- **CI (PAIRED — amendment v1.1):** moving-block bootstrap over the **paired per-period
  difference series** Δr_p = r_p(Cell 4) − r_p(Cell 5) on the shared calendar grid — the SAME
  resampled block indices apply to both cells (equivalently: resample blocks of Δr_p). Block
  length ⌈√T⌉. Seeds enter as matched pairs (seed s of Cell 4 vs seed s of Cell 5; the seed-mean
  Δ series is the resampled object). Cell 4 and Cell 5 share the draw, seeds, grid, and market
  exposure by construction — the common component cancels in Δr_p, and the CI must reflect that
  pairing; an unpaired CI would be wrongly wide.
- **THE MICROSTRUCTURE LEG SURVIVES iff ALL of (conjunctive — every clause must pass; where two
  clauses could disagree, both bind):**
  1. the one-sided (1−α) CI lower bound of ΔIR_info > 0 (the paired CI above), AND
  2. ΔIR_info ≥ **MDE_paired ≡ (z₀.₉₅ + z₀.₈₀) · SE_boot(ΔIR_info)** — where SE_boot is the
     standard deviation of the bootstrap ΔIR replicates from the SAME paired procedure as the CI
     (a variance/nuisance quantity, never a function of the effect's sign or magnitude). **No
     ceiling appeal:** if the realized SE_boot makes MDE_paired > the §2 tabled value, the
     larger bound applies, AND
  3. **placebo-validity (v1.2):** the one-sided paired CI lower bound of **IR(Cell 4) − IR(Cell 2)
     > 0** — micro must beat OHLCV-only *at all* for a micro-information claim; this clause is
     placebo-independent, so a Cell-5-harmed-below-its-counterfactual scenario ("the placebo is a
     victim, not a control") cannot manufacture survival on ΔIR(4−5) alone. The **placebo-health
     diagnostic** IR(Cell 5) − IR(Cell 2) is reported with its paired CI; if its upper bound < 0
     (the shuffle demonstrably *harmed* training below the OHLCV-only counterfactual), that is
     stated prominently and the capacity-vs-information interpretation of ΔIR(4−5) is qualified
     in the paper — the survival verdict itself is unaffected because clause 3 already carries
     the placebo-free requirement, AND
  4. ΔIR_info ≥ **0.5 annualized IR** (the economic-materiality floor, fixed pre-data: an
     after-cost gain below 0.5 is within seed-jitter scale at these horizons and would not
     justify the added aggTrades data dependency — too small to claim even if significant), AND
  5. **DSR (pinned recipe, v1.2):** DSR ≥ **0.95**, where the statistic is Cell 4's pooled
     headline series (0.30 % netting, seed-mean), SR₀ = expected-max-Sharpe under **N = 180
     enumerated trials** (5 cells × 3 seeds × 3 horizons {5,15,60} × 4 κ — **h=1 is not evaluated
     anywhere in M6**; the earlier "4 horizons / 240" budget text is corrected here), var_sr =
     the variance of the 180 recorded per-trial de-annualized IRs (every cell × seed × horizon ×
     κ VAL entry is persisted, so the trial set is enumerable, not assumed), higher moments from
     the same pooled headline series.
- **Otherwise: the pre-committed NULL** — the microstructure leg is withdrawn and the
  contribution is FSQ-vs-BSQ on OHLCV alone (Cells 1–2), governed by §5's fallback rule (v1.2).
  No re-thresholding, no horizon-shopping: h=15 is the primary; h=5/60 are reported as
  secondaries with their own MDEs above.
- **Relation to the §2 table:** the tabled value (3.518 at h=15 on the pinned blocks-1–5 basis)
  is exactly MDE_paired evaluated at ρ₄₅ = 0 (cells independent) — the conservative reference,
  reported for power honesty. The operative threshold uses the realized pairing, in **either**
  direction (see clause 2's no-ceiling rule).

### 3a. Pinned analysis surface (v1.2 — every previously-free choice, fixed before training)

- **Primary cross-section:** the **40-symbol set** in `runs_manifest/m6_mde_inputs.json`
  (`symbols_sampled`; sorted-list sha256 `60e24f598de96012…`) — the same set the §2 MDE is
  computed on. Equal-weight across symbols active at each pooled stride-h period; no per-symbol
  exclusion of any kind after this commit. The all-active-in-eval-region (~200-symbol) pool is a
  reported **secondary robustness** read, never the verdict.
- **Primary eval region:** forward blocks **1–5** of the k=6 plan (block 0 is VAL — κ territory —
  and is **excluded** from every headline/Δ statistic). The §2 MDE table was computed on the full
  forward region including block 0; it is **recomputed on blocks 1–5 by the same locked formula**
  (`scripts/m6_prereg.py`) and the refreshed table committed before training — a nuisance-basis
  correction, slightly raising MDE (T shrinks ~1/6), disclosed here first.
- **Seeds:** exactly **3**, literal values **{0, 1, 2}** (the committed orchestrator defaults).
  No seed added, dropped, or substituted after any eval statistic exists for any cell. A crashed
  run is re-run with the identical seed and config (§3a abort protocol below).
- **κ:** grid = **{1.0, 1.5, 2.0, 3.0}**. κ* is selected **per cell**, as the argmax of pooled
  net-IR **at the 0.30 % flat netting** on VAL (block 0) over the primary cross-section; ties →
  the smaller κ. If the selected κ* yields low forward activity, the verdict is still computed at
  that κ* — **no substitution**, ever. (The paired Δ series remains time-aligned by the flat=0
  calendar-grid convention regardless of per-cell activity patterns.)
- **Bootstrap recipe:** B = **10,000** replicates; bootstrap RNG seed = **20260704**; block
  length ⌈√T⌉; CI type = **percentile** (the one-sided lower bound is the α-quantile of the
  bootstrap Δ distribution); SE_boot = the standard deviation of the same B replicates. Rules 1
  and 2 are separate conjunctive clauses — no "which governs" question arises.
- **Instrument pinning:** the primary statistic is computed by the eval code
  (`src/trikaal/eval/{xsection,dsr,metrics,strategy,harness,placebo}.py` + the paired-bootstrap
  module) **at the training-start commit**. **Training-start ≔ the first non-SMOKE W&B run of any
  cell**; its commit hash + timestamp are appended to §7 at launch, and the eval driver verifies
  (by hash) that the prereg it runs under is this file at that commit. Any later change to those
  files requires a dated §7 entry BEFORE its output is used.
- **Attention mode:** ONE mode produces all 15 runs and the headline. It is fixed at the
  toy-CUDA rehearsal (flash2 if stable there, else the deterministic-SDPA fallback), **before any
  real cell trains**, and logged in §7. Two modes never coexist inside the trial set.
- **Aborts / exclusions:** tripwire criteria = `config/m6_tripwires.yaml` at the training-start
  commit; frozen once training starts. A tripped run is re-run with the identical seed + config;
  a tripped run's partial numbers are never inspected for any Δ statistic. There is no other
  exclusion pathway.
- **Declared simplifications (pre-data):** (i) funding is not subtracted in eval netting — at
  h ≤ 60 min holds rarely cross an 8-h settlement, the omission is symmetric across cells and
  ~cancels in every Δ; (ii) the 40-deepest primary set embeds survivorship — this inflates
  *absolute* IR levels, not the paired Δs; the paper states both, verbatim from here.

## 4. The κ* / cost-stress headline rule (the env-drift lesson, locked)

- κ* is selected **on the pooled VAL block (forward block 0), over the primary cross-section
  (§3a), per cell, as the argmax of net-IR at the 0.30 % netting; ties → smaller κ; no
  post-selection substitution (v1.2 pins)** — never on any headline block.
- **The headline is the COST-STRESS CURVE across flat nettings {0.10 %, 0.20 %, 0.30 %} at κ***
  (with break-even cost), NOT a single κ-point: the 2026-06 env-drift incident flipped κ* 2.0→3.0
  on a knife-edge, so a single-κ headline is too fragile to pre-register. The 0.30 % point of that
  curve is the primary number; the full per-κ VAL curve is persisted in every result for audit.
- Positions from θ = κ*·c_total(modeled); the modeled-cost IR is the named secondary.

## 5. Secondary analyses (reported with CIs; never gating) + the NULL-fallback rule (v1.2)

The 2×2 marginals — FSQ effect on OHLCV [IR(2)−IR(1)] and under micro [IR(4)−IR(3)]; the micro
marginal under FSQ [IR(4)−IR(2)] and BSQ [IR(3)−IR(1)]; the per-regime ΔIR_info breakdown (OOS:
2023-tail, 2024; in-sample-labeled: 2021, 2022); IC/RankIC/MAE/R² diagnostics.

**NULL-fallback rule (v1.2 — the fallback is not a free positive claim):** if the §3 primary
returns NULL, the FSQ-vs-BSQ leg [IR(2)−IR(1)] may be **claimed as a demonstrated improvement
only if it passes the IDENTICAL paired rule** (paired CI lower bound > 0, ≥ its own MDE_paired,
≥ the 0.5 floor, DSR ≥ 0.95 under the same N=180 budget) on the same pinned surface (§3a).
Otherwise it is reported **descriptively with CIs, claimed as nothing** — and **double-NULL
("neither leg survives") is a stated possible outcome of this experiment and of the paper.**

## 6. Binding gates (G-§8.C.3 quantified in v1.2; others unchanged)

G-instrument-live (anchor `3f86882a…`, re-anchored §7 v1.4.2 for the μ̂-mean estimator; `5eead7b6…` retired to history), G-parity, G-causal, G-determinism — per m6_design §3.

**G-§8.C.3 (Kronos external validation), quantified (v1.2 — was "~10–15% / materially off / fix",
three stacked judgment calls = an unlimited re-run license):**
- **Fails iff** Cell 1's RankIC < **0.85 × published Kronos-small RankIC** on the pinned common
  slice: **BTCUSDT, forward blocks 1–5 (§3a region), h=15 stride grid, Spearman RankIC per our
  `metrics`/`ic_screen` definition**, with Kronos-small run through **its own published
  preprocessing** on the same bars. The published comparison number is transcribed into §7 at
  weights-load time — **before any Δ statistic of any cell exists**.
- **Failure protocol (pre-committed):** a parity failure **halts the ablation before any Cell-2–5
  eval statistic is computed**. Only Cell-1-side fixes are permitted (its tokenizer/training
  config — never the eval, never the other cells). After the fix, **all five cells retrain from
  scratch with the same seeds {0,1,2}**, and the event + diff is a dated §7 entry. There is no
  other legitimate path to a re-run.

## 7. Amendment log

- **v1.4.3 (2026-07-29, money-leg adjudication + external-audit addenda; supervisor-ordered) —
  BEFORE any real training.** Ruling items 1–9 executed as ONE local ($0) pass; the ONE box
  re-run is STAGED with a pre-declared exit (item 4). Touches NO anchored M5-instrument file — the
  estimator (`eval/predict.py`) is UNCHANGED (item 1 is a DISCLOSURE, not a code change); the
  criterion, oracle, non-degeneracy, and recon changes all live in `scripts/m6_canary.py` (the
  fixture harness, never part of the eval instrument), so GATE-A anchor `3f86882a…` is unchanged
  and needs no re-anchor.
  1. **ESTIMATOR DISCLOSURE (item 1).** μ̂ is computed EXACTLY as a per-step MEAN-FIELD decode
     `decode_latent(E[z])`, `E[z] = softmax(logits_c)@grid_c ⊕ softmax(logits_f|ĉ)@grid_f`,
     accumulated along a GREEDY (argmax) token path — the cache/conditioning is fixed by the
     argmax chain, only each step's μ̂ CONTRIBUTION is the mean-field expectation. This equals the
     true conditional mean EXACTLY only under (i) a decoder LINEAR in z and (ii) the deterministic
     greedy path; it is a delta-method approximation otherwise. `"mc_mean"` (pinned n=32/seed
     20260721) is the UNBIASED-in-expectation Monte-Carlo reference, retained as the documented
     fallback. BIAS-BOUND RECEIPT (`runs_manifest/m6_estimator_bias_bound.json`): expectation vs
     mc_mean at n≥256 on a decision subsample of the local checkpoints. FINDINGS: (a) the gap is
     directionally COMMON-MODE (both cells +0.023..+0.032) and a LEVEL shift, NOT a sign shift —
     frac_neg is IDENTICAL between the two estimators for each cell (cell4 0.5=0.5, cell2 0.0=0.0),
     so positions (sign(μ̂) gated by |μ̂|>κ·c) and hence the paired ΔIR are robust to it; mc_mean
     (unbiased) is the reference, expectation understates the negative drift by a small delta-method
     level bias without flipping signs. (b) DECISIVE: on the ACCEPTANCE cell4 the expectation
     estimator gives BALANCED frac_neg=0.5, so the money-leg cell4's frac_neg=0.999 (mostly-short)
     is a MODEL property of that specific 3000-step cell, NOT an estimator artifact — and because
     training is SIGMA-invariant the re-run reproduces that cell4 bit-identically, so raising SIGMA
     cannot correct its sign bias (bears directly on the item-4 exit). Portability note recorded:
     `mc_mean` accumulates in float64 and is therefore CUDA/CPU-only (MPS lacks float64) —
     production is CUDA, so no impact; logged, not patched.
  2. **CANARY CRITERION REPLACED (item 2) — realigned to the LOCKED §3, not softened.** The
     Stage-2 canary verdict had DRIFTED to gating on placebo-neutrality; §3 (v1.2) already makes
     clause 3 = paired CI of IR(4)−IR(2) > 0 the placebo-INDEPENDENT validity clause with
     IR(5)−IR(2) a REPORTED diagnostic. The canary now PASSES iff clause 1 (paired CI ΔIR(4−5)>0)
     AND clause 3 (paired CI ΔIR(4−2)>0), the noise arm quiet on BOTH clauses, cell4 non-degenerate,
     recon + codebooks intact. Placebo-neutrality is demoted to a reported diagnostic
     (`placebo_health_diagnostic`). NON-DEGENERACY floor: a near-constant-μ̂ cell is not a strategy;
     the GATE is `std(μ̂_cell4) ≥ 0.5·SIGMA` (SIGMA-relative — μ̂ ∝ SIGMA exactly), per-cell
     reported; a degenerate cell4 makes clauses 1+3 UNINTERPRETABLE (differencing near-constant
     series — the exact pathology the money-leg's tight CIs on constant-sign cells exhibited).
  3. **FIXTURE ECONOMIC RECALIBRATION (item 3) — SIGMA 0.01→0.05, ECONOMIC-ONLY and
     TRAINING-INVARIANT.** SIGMA is the sole economic-magnitude knob and is provably
     training-invariant: `x[:,0]=r/SIGMA` and `x[:,9]=state` are SIGMA-free (the SIGMA in r
     cancels), so the tokenized features — every token, the tokenizer, the AR — are BIT-IDENTICAL
     across SIGMA (proven: `x_identical=True` both arms, raw_ret_close ratio exactly 5.0;
     `runs_manifest/m6_oracle_calibration.json`). Only the backtest changes (y ∝ SIGMA, μ̂ ∝ SIGMA,
     flat 0.30 % cost FIXED), so raising SIGMA shrinks fixed-cost DRAG without touching information
     content, plant, lag, or filler. ORACLE receipt (perfect causal state, same θ=κ·c filter, same
     0.30 % netting, EXACT canary scoring path): pinned SIGMA=0.05 → gross IR 18.89 / net IR 18.44
     / cost drag 0.45 / activity 0.23 / MDE_paired 2.75 → **oracle net IR = 6.71× MDE**, clearing
     the PRE-DECLARED margin (oracle net IR ≥ 5× fixture MDE_paired). **FINDING ON THE RECORD
     (premise revision):** the oracle ALREADY clears 0.30 % costs at the OLD SIGMA=0.01 (net IR
     17.12, 6.29× MDE, drag 2.17) — the fixture's planted edge survives costs at BOTH values. The
     money-leg diagnosis "the planted edge does not survive costs" holds only for the TRAINED cell4
     (net −3.43, frac-negative 0.999 — a sign-corrupted forecast), NOT for the oracle; recalibration
     is a fixed-cost-drag reduction (2.17→0.45 IR), not an oracle rescue. The binding gap is cell4
     EXTRACTION, which the canary MEASURES — not a fixture defect.
  4. **THE ONE RE-RUN (item 4) — pre-declared exit.** STAGED (box blocked on absent GPU
     credentials — the money/human-operator gate). Because training is SIGMA-invariant, the re-run
     re-trains the cells BIT-IDENTICALLY to the money-leg (same tokens; cell4 val ≈ 11.88 expected)
     and ONLY the backtest economics change. EXIT (pre-declared, so it can never read as a retreat):
     clauses 1+3 both fire, cell4 non-degenerate, noise quiet → canary GREEN, board 8/8. IF NOT →
     STOP, ship receipts, do NOT iterate the fixture; the supervisor adjudicates de-scoping the
     canary money leg against the rehearsal's real-data Item-2 discrimination receipts.
  5. **THROUGHPUT RESTATEMENT (item 5), LABELED.** Expectation is NOT a throughput regression:
     measured ratio expectation/argmax = 0.82 on mps at the real config (seq 512, h=15) —
     expectation ~1.2× FASTER, not slower, refuting the "expectation-decode is slower per rollout
     step" premise on the measured hardware. Real-scale h=15 headline decision count = 35,063
     periods × 40 symbols × 5 cells × 3 seeds = 21,037,800. LABELS (`runs_manifest/
     m6_eval_throughput_expectation.json`): training throughput = measured on the 4090 (prior
     rehearsal, 47.2k bars/s); eval leg = estimator not a regression (measured ratio 0.82), absolute
     rate chunk- and hardware-dependent (recipe pins chunk=512), PENDING the 4090 measurement in
     the re-run; dollar figure = spot-price dependent (~$0.26–0.40/hr). The banked "$20–30 / 2–3
     GPU-days" is NOT quoted without these labels.
  6. **PLACEBO-HARM, a paper-facing diagnostic (item 6).** Money-leg IR(5)−IR(2) = −2.59, CI
     [−5.21, −0.02] clears 0 — the shuffle demonstrably HARMED Cell 5 below the OHLCV-only
     counterfactual (a "placebo victim"). This VINDICATES §3's placebo-independence: shuffled
     channels consume tokenizer capacity and add noise, so they are strictly worse than omitting
     them, which is precisely why clause 3 (Cell 4 > Cell 2), not ΔIR(4−5), carries the
     micro-information claim. Diagnostic language, never the rule.
  7. **PAPER-FRAMING (item 7) — committed verbatim.** The supervisor WITHDRAWS the phrasing "the
     tokenizer-eviction finding is now the central argument of the paper" and commits instead: *"the
     eviction finding is a methodological contribution that stands INDEPENDENT of the M6 outcome;
     whether the re-specced tokenizer captures real tradable microstructure is the open question the
     run answers. Both outcomes — micro survives, or micro nulls after costs — are complete,
     publishable papers containing this finding."*
  8. **NON-CONTAMINATION (item 8) — the interface fix is GENERAL, not tuned to the plant.** (a) The
     weighting keys on `MICRO_DIMS_IDX = (7..12)` BY DIM INDEX (the aggTrades-derived channels;
     `constants.py`), never on the plant's shape/lag/functional form — landed c4cd082 (§7 v1.4,
     pointwise-fine + per-bar bottleneck) and 7da3dc0 (§7 v1.4.1). (b) λ\* = 3 was calibrated
     against CHANNEL-VALUE LEGIBILITY (`id_legibility_sign_acc`, gates.py:125 — the logistic
     sign-accuracy of the dim's value from bar t's OWN id), never against detection of the planted
     rule (7da3dc0). (c) Cell 5 receives IDENTICAL treatment — same fine_pointwise, same λ, same
     dims (forced in `build_cell_tokenizer`, cells.py:82-84) — so generic capacity effects are
     SHARED and ΔIR(4−5) still isolates information. ONE HONEST CAVEAT: λ's magnitude was tuned on a
     synthetic fixture whose micro-slot dim is iid unit-variance, while real TFI has different
     statistics; the already-armed STANDING real-data legibility gate (`micro_legibility_gate`,
     gates.py:151 / orchestrator hard-stop post-Stage-1/pre-Stage-2, six real micro dims, default
     0.9) is the NON-CIRCULAR check, and its first firing on real data is a NAMED adjudication
     checkpoint.
  9. **REALLOCATION-ASYMMETRY (item 9) — reported, not adjusted.** λ shifts Cell 4's per-bar
     bottleneck budget toward the six micro dims and away from OHLCV shape; Cell 2 (7 uniform dims)
     has no such reallocation. Per-cell recon on the SHARED OHLCV dims (0-6) is reported by
     `run_arm._shared_ohlcv_recon` in the re-run (AUTHORITATIVE, same-arm/same-budget). LOCAL
     INDICATIVE receipt (`runs_manifest/m6_reallocation_indicative.json`, cross-run/cross-arm): Cell
     4 is ~20-37 % worse than Cell 2 on 6 of 7 shared dims (dims 1-6 — suggestive of the asymmetry);
     dim 0 is dominated by the cross-run confound (the noise-arm Cell 2 reconstructs the planted
     return dim poorly, MAE 0.84), so the mean is not interpretable. If the authoritative receipt
     confirms Cell 4 materially worse on shared dims, that is a DISCLOSED asymmetry that makes
     clause 3 HARDER (a conservative bias on the micro claim) — flagged, not adjusted.

- **v1.4.2 (2026-07-21, acceptance-run adjudication; supervisor-ordered) — BEFORE any real
  training. THE μ̂ ESTIMATOR: conditional MEAN, not mode.** The acceptance run's Stage-2 money
  verdicts failed with all noise cells at a bit-identical IR across both quantizers; diagnosed
  as sign saturation — the greedy-argmax rollout is a conditional-MODE estimator, and under the
  skewed per-step return distribution the mode is biased low, a bias that COMPOUNDS over the h
  rollout steps (acceptance fixture, h=15: argmax μ̂ mean −0.051 with 92.9 % of decisions
  negative — every decision short → identical positions → identical IRs). μ̂ is pre-registered
  (§8) as the conditional MEAN. The decisive $0 comparison
  (runs_manifest/m6_mu_estimator_comparison.json) on the fixture confirms the named hypothesis:
  at h=15 the mean estimator removes ~91 % of the drift (argmax mean −0.051 → expectation
  −0.004, frac-negative 0.929 → 0.534) while RAISING corr(μ̂, planted-state) 0.858 → 0.898; at
  h=2 (few compounding steps) all estimators agree, the mode-bias signature. CHANGE: the
  default estimator in `eval/predict.py` is now `"expectation"` — the token CHAIN still advances
  greedily (deterministic cache, argmax conditioning) but each step's μ̂ contribution is the
  mean decode `decode_latent(E[z])`, `E[z] = softmax(logits_c) @ grid_c ⊕
  softmax(logits_f|ĉ) @ grid_f` (the mean-field/delta-method expectation — deterministic and
  seed-stable). `"argmax"` is retained for regression KATs and pre-v1.4.2 anchor reproduction;
  `"mc_mean"` (pinned n_samples=32/seed) is the unbiased-in-expectation fallback. **spec §8's
  deferred MC-decode is thereby PARTIALLY un-deferred as a CORRECTNESS need — mean only, no
  full-distribution scope.** Enforced by a skewed-toy KNOWN-ANSWER KAT (argmax provably biased
  to the mode value, expectation provably the true mean) plus regression KATs (k=1 exclusion,
  chunking, RoPE guard unchanged) and per-cell μ̂ mean/std/frac-negative receipts on
  `CellScore` + the `m6_cell_eval_v1` artifact. μ̂'s pre-registered meaning is the conditional
  mean; the estimator now matches it. GATE-A RE-ANCHOR: `predict.py` is inside the anchored M5
  instrument, so the estimator change re-derives the results_hash under the standing
  re-anchor procedure (≥2 bit-identical runs, new hash in the run-manifest, milestone5
  anchor-history extended); the KATs + causal gates carry logic-integrity across the transition.
- **v1.4.1 (2026-07-21, gate-2 final ruling execution; supervisor-signed) — BEFORE any real
  training.** The pre-authorized micro-weighted bottleneck fallback FIRED on a trigger
  strictly stronger than written: deterministic FIXTURE-gate failure with mechanism receipts
  (3-seed per-dim point-decoder arrays — seed 0: [0.98, .89, .91, .85, .83, .88, .91, .88,
  .83, **0.01**, .85, .91, .90]; seeds 1/2 near-identical; the recon objective buys variance
  and covariance, never independence — the independent state dim is priced out of the
  10.26-bit fine budget every time). AMENDMENTS: (a) the per-bar bottleneck leg weights THE
  SIX MICRO DIMS as a class by λ\* = 3 — the SMALLEST searched value whose three seeded
  calibration seeds clear the restated gate (0.9060/0.9142/0.9000: mean 0.9067, min 0.9000;
  full receipt in runs_manifest/m6_lambda_search_receipt.json). The adjudication's literal
  "λ\* = 2" was ratified from defective receipts (disclosure 3 below); the ruling's formula
  — smallest λ clearing the gate, calibrated not chosen — was executed on the corrected
  instruments, and λ=2's seeded triplet (0.8365/0.8594/0.8592, mean 0.8517) fails it; (b) OBJECTIVE SEPARATION: the window reconstruction
  losses receive the fine channels detached — the fine encoder is shaped exclusively by the
  per-bar objective (measured: without this the window pull re-creates the smearing
  incentive at any λ, and collateral recon is seed-unstable). DISCLOSURES (both incidents on
  the record): (1) a formatter re-wrap silently defeated the patches wiring
  ``w_feat_point``/``point_loss_coef`` into the bottleneck loss, so the first canonical
  λ-search ran entirely at λ=1 — those runs are retained as λ=1 replicates and the
  normalization-asymptote analysis derived from them is WITHDRAWN; (2) commit e8d2a06's
  "anchor re-proven" claim was FALSE — the unconditional ``w_feat_point`` buffer broke
  old-schema checkpoint loading and the M5 anchor run FAILED, but a ``| grep | tail``
  pipeline masked the exit-1, letting the unverified claim into the commit message;
  corrected in b55fffc (buffer registered only under ``fine_pointwise``; anchor genuinely
  re-proven exit-0, results_hash 5eead7b6 bit-identical; the exit-masking cause is now
  policed by the pipefail rider); (3) the calibration harness seeded torch only INSIDE the
  training loop, AFTER model construction, so every canonical calibration triplet carried
  unseeded-init variation and was unreproducible — exposed when the ordered formal
  cell-path re-run failed to reproduce the ruled-on triplet (0.8974/0.9152/0.9076 were
  init-lottery draws); fixed by seeding before construction, after which the direct and
  pinned-cell-path constructions produce BIT-IDENTICAL results per seed (the determinism
  the formal re-run was ordered to demonstrate), and the λ landscape was re-derived seeded. FAIRNESS: identical weighting and detach across both
  quantizers and all arms including Cell 5; OHLCV arms carry no micro dims and are
  unaffected; bits-per-token, the cell matrix, the decision rule, and all eval instruments
  are untouched. The fixture legibility gate is RESTATED AS RULED over the three calibration
  seeds: mean ≥ 0.9 AND every seed ≥ 0.85 (seeded receipts at λ\*=3: mean 0.9067, min
  0.9000, reproduced bit-identically through the pinned cell path); the STANDING real-data
  gate stays 0.9-on-all-six with its named-checkpoint semantics. Paper-facing mechanism
  sentence: *reconstruction-trained tokenizers allocate code capacity by variance and
  covariance, never by downstream value — microstructure state, being weakly covariant with
  price shape, is priced out unless the tokenizer is built to carry it;
  "microstructure-aware" is therefore an architectural property, not an emergent one.*
- **v1.4 (2026-07-20, supervisor adjudication of the token-control programme) —
  BEFORE any real training. Receipts (m6_canary_v6_stage1_manifest.json,
  m6_token_control_step0.json, m6_token_control_run_manifest.json) establish:
  the AR learns dense per-bar-legible token conditionals essentially perfectly
  at canonical scale (probe Spearman 0.9999, 94% of planted nats), but the
  causal contextual encoder smears each bar's feature state forward across
  later tokens' ids (per-bar id visibility ≈ chance: logistic 0.5135), so
  feature-space conditionals arrive per-bar-illegible and are never learned
  (zero nats from a ~1.15-nat plant). INSTRUMENT RE-SPEC: the fine subtoken is
  now a PER-BAR pointwise encoding of bar t's own features (micro dims included
  in +micro arms); the coarse subtoken remains causal-contextual. Identical for
  both quantizers and all arms — bits-per-token parity and the cell design
  unchanged. Enforced by the extended flip-KAT (fine invariant to all other
  bars), the per-bar legibility gate (logistic ≥ 0.9 on a 3σ planted state),
  and conformance pins. ACCEPTANCE: the v6 feature-space canary re-run under
  the new tokenizer must DETECT (and the noise arm stay quiet) before the
  canary gate may close. Comparability: 'microstructure-aware' is now an
  architectural property of the tokenizer; disclosed alongside the Kronos
  notes.**
- **Attention mode FIXED (2026-07-19, toy-CUDA rehearsal — the §3a one-mode rule; entry
  pre-authorized in the rehearsal instruction): sdpa_deterministic** for all 15 real runs and
  the headline. Evidence (`runs/m6_attention_bench.json`; RTX 4090, canonical 21,301,248-param
  backbone, bf16 autocast, batch 32 × seq 512): sdpa stable at 2.882 steps/s over the matched
  segment; flash2 UNAVAILABLE on the rehearsal box — no prebuilt wheel exists for torch 2.12
  (30 releases searched) and the source build is refused by the image's CUDA 12.4 toolkit vs
  torch's cu130 (compile-time version check, error captured in the rehearsal log).
  sdpa_deterministic additionally carries the bit-exact kill-resume claim, re-proven on CUDA in
  this same rehearsal (inv. 7). Switching modes would require a new dated entry BEFORE any real
  cell trains; the mode is otherwise fixed here.
- **v1.3 (2026-07-19, supervisor adjudication of the builder's halt-finding) —
  BEFORE any real training: the pre-tokenized eval context carried future-bar
  information (bidirectional encoder within fixed tokenization chunks; measured on
  the real lake: 41.8%/28.3% coarse/fine past-token flips under future-bar-only
  mutation; probe artifact runs_manifest/m6_token_causality_probe.json). Invariant
  2 binds eval INPUTS, not only training labels. DECISION: encoder_causal=True for
  ALL five cells, train and eval — symmetric across cells, so bits-per-token
  fairness and the paired design are unaffected; structurally closes the channel
  (measured 0.0%); enforced by a CI flip-KAT and a conformance pin. Rejected
  alternatives: eval-time re-tokenization (train/eval token-distribution mismatch);
  disclosure alone (indefensible at 41.8%). Comparability note, binding on the
  paper and on G-§8.C.3 reporting: Kronos-style tokenizers are bidirectional AEs;
  Trikaal's cells use the causal variant the blueprint pre-committed as 'maximally
  defensible' — stated wherever Cell 1 is compared to published Kronos-small.**
- **v1.2.1 (2026-07-06, supervisor — EDITORIAL ONLY, no rule changed):** after the audit item-5
  recompute of the §2 table on the pinned blocks-1–5 basis (MDE h=15: 3.209 → 3.518, T 42,076 →
  35,063), §3's two literal "3.209" mentions were updated to reference "the §2 tabled value"
  (clause 2's no-ceiling sentence and the Relation note), and §1's forward-region date was made
  exact (2023-10-20T16:48Z, the true 0.7 boundary; the earlier "2023-10-14" was imprecise
  prose — the operative region was always the formula, now conformance-gated). §1's "ρ̄ ≈"
  approximations remain approximate by design; exact values live in the committed JSON.
- **v1.2 (2026-07-06, supervisor/research lead) — BEFORE any real training** (still nothing
  beyond the meaningless-by-design SMOKE). Trigger: two independent external audits (a run
  pre-mortem and a prereg cold-read), each finding supervisor-verified against the repo before
  adoption. Changes: (i) §3 clause 3 **placebo-validity** added — survival now also requires the
  paired CI of IR(4)−IR(2) > 0, closing the "placebo is a victim, not a control" false-positive
  channel (ΔIR(4−5) alone can fire when the shuffle *harms* Cell 5; clause 3 is
  placebo-independent), with IR(5)−IR(2) reported as the placebo-health diagnostic; (ii) §3
  clause 5 DSR recipe pinned (statistic, SR₀ basis, **N=180 enumerated** — the "4 horizons/240"
  text was a bookkeeping error, h=1 is not evaluated; threshold 0.95); (iii) §3a added — every
  previously-free analysis choice pinned: primary cross-section (the hashed 40-symbol MDE set),
  primary region (forward blocks 1–5, VAL excluded; §2 MDE to be recomputed on that basis),
  exactly seeds {0,1,2}, the κ grid/criterion/no-substitution rule, the bootstrap recipe
  (B=10,000, seed 20260704, percentile CI) + the no-ceiling clause, instrument-pinning by
  training-start commit hash (training-start = first non-SMOKE W&B run), the single-attention-
  mode rule, the abort/re-run protocol (same seed, partials never read), and the declared
  funding/survivorship simplifications; (iv) §5 NULL-fallback rule — FSQ-vs-BSQ under NULL must
  pass the identical paired rule or is descriptive-only; double-NULL is a stated possible
  outcome; (v) §6 G-§8.C.3 quantified (0.85×, pinned slice, pre-committed halt-and-full-retrain
  failure protocol) — closing the qualitative re-run license. Nothing in §0–§2's commitments
  weakened; every new clause binds the analyst, not the data.
- **v1.1 (2026-07-04, supervisor/research lead) — BEFORE any real training** (only the
  meaningless-by-design SMOKE had run; no cell model existed beyond toy scale). Original v1.0
  (commit `2c72fff`) rule 2 read: *"ΔIR_info ≥ MDE_pooled = 3.209 (h=15)"* — the UNPAIRED
  (ρ₄₅ = 0) fixed threshold — and the CI clause did not specify pairing. Why amended: Cell 4 and
  Cell 5 are paired by construction (identical draw/seeds/grid; they differ only in the micro
  information), so the unpaired 3.209 as a decision threshold conflates a conservative power
  bound with the operative materiality bar and would pre-commit the experiment to NULL for any
  real-but-moderate effect — a detector that cannot detect. The amended rule keeps every
  anti-p-hack property: thresholds are formulas over nuisance quantities (paired variance /
  correlation) plus a fixed pre-data economic floor (0.5), none a function of the observed
  effect. The unpaired 3.209 table stays in §2 as the honest conservative bound. Nothing else
  changed. Locked from here; any further amendment requires its own dated log entry and is
  illegitimate once a real cell has trained.
