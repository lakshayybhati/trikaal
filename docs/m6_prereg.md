# M6 Pre-Registration — MDE, Decision Thresholds, and the Headline Rule (LOCKED)

**Status:** LOCKED at commit time. **This file's git commit timestamp precedes any real M6
training run** — that timestamp is the proof the decision rule existed before any result did
(m6_preflight Item 6). No threshold below may be moved after seeing results; a NULL outcome is
valid and pre-committed (m6_design §0).

**Inputs (computed, not assumed):** `scripts/m6_prereg.py` over the compacted universe lake
(anchor `5dfd667d…`), 40 deepest symbols, forward (eval) region 2023-10-14 → 2025-01-01 under the
anchored 0.7 calendar split. Raw inputs + all rows: `runs_manifest/m6_mde_inputs.json` (committed).

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

| Slice | h=5 | h=15 | h=60 |
|---|---|---|---|
| **Pooled (the primary)** | **3.209** | **3.209** | **3.313** |
| 2024 (OOS regime) | 3.532 | 3.512 | 3.669 |
| 2023 OOS tail (Oct–Dec) | 7.901 | 7.901 | 7.899 |

(Annualized net-IR units. T at h=15: pooled 42,076 periods; 2024 35,135; 2023-tail 6,941.)

**Honest reading:** ~1.2 years of OOS at these horizons can only power-detect a ΔIR_info of
**≥ ~3.2 annualized IR units** at the pre-registered α/power. Smaller true effects will read as
CI-includes-zero → the pre-committed NULL. This is the cost of the anchored train-once design and
is accepted, not negotiated after the fact.

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
     ceiling appeal:** if the realized SE_boot makes MDE_paired > the §2 table's 3.209, the
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
- **Relation to the §2 table:** the tabled 3.209 is exactly MDE_paired evaluated at ρ₄₅ = 0
  (cells independent) — the conservative reference, reported for power honesty. The operative
  threshold uses the realized pairing, in **either** direction (see clause 2's no-ceiling rule).

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

G-instrument-live (anchor `5eead7b6…`), G-parity, G-causal, G-determinism — per m6_design §3.

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
