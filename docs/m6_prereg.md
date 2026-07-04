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
- **THE MICROSTRUCTURE LEG SURVIVES iff ALL of:**
  1. the one-sided (1−α) CI lower bound of ΔIR_info > 0 (the paired CI above), AND
  2. ΔIR_info ≥ **MDE_paired ≡ (z₀.₉₅ + z₀.₈₀) · SE_boot(ΔIR_info)** — where SE_boot is the
     bootstrap SE from the SAME paired moving-block procedure as the CI (a variance/nuisance
     quantity, never a function of the effect's sign or magnitude), AND
  3. ΔIR_info ≥ **0.5 annualized IR** (the economic-materiality floor, fixed pre-data: an
     after-cost gain below 0.5 is within seed-jitter scale at these horizons and would not
     justify the added aggTrades data dependency — too small to claim even if significant), AND
  4. the DSR of Cell 4's headline survives the documented trial budget (5 cells × 3 seeds ×
     4 horizons × 4 κ).
- **Otherwise: the pre-committed NULL** — the microstructure leg is withdrawn and the
  contribution is FSQ-vs-BSQ on OHLCV alone (Cells 1–2). No re-thresholding, no horizon-shopping:
  h=15 is the primary; h=5/60 are reported as secondaries with their own MDEs above.
- **Relation to the §2 table:** the tabled 3.209 is exactly MDE_paired evaluated at ρ₄₅ = 0
  (cells independent) — the conservative ceiling, reported for power honesty. The operative
  threshold uses the realized pairing; if the cells turn out uncorrelated, the two coincide.

## 4. The κ* / cost-stress headline rule (the env-drift lesson, locked)

- κ* is selected **once, on the pooled VAL block (forward block 0), over the full cross-section**
  — never on any headline block.
- **The headline is the COST-STRESS CURVE across flat nettings {0.10 %, 0.20 %, 0.30 %} at κ***
  (with break-even cost), NOT a single κ-point: the 2026-06 env-drift incident flipped κ* 2.0→3.0
  on a knife-edge, so a single-κ headline is too fragile to pre-register. The 0.30 % point of that
  curve is the primary number; the full per-κ VAL curve is persisted in every result for audit.
- Positions from θ = κ*·c_total(modeled); the modeled-cost IR is the named secondary.

## 5. Secondary analyses (reported with CIs; never gating)

The 2×2 marginals — FSQ effect on OHLCV [IR(2)−IR(1)] and under micro [IR(4)−IR(3)]; the micro
marginal under FSQ [IR(4)−IR(2)] and BSQ [IR(3)−IR(1)]; the per-regime ΔIR_info breakdown (OOS:
2023-tail, 2024; in-sample-labeled: 2021, 2022); IC/RankIC/MAE/R² diagnostics.

## 6. Binding gates unchanged

G-instrument-live (anchor `5eead7b6…`), G-parity, G-§8.C.3 (Kronos external validation of Cell 1),
G-causal, G-determinism — all per m6_design §3. This document adds the numeric thresholds; it
changes no gate.

## 7. Amendment log

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
