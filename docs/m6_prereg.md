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

- **v1.6.3 (2026-08-01, RECORD CORRECTIONS — NO BEHAVIOURAL CHANGE TO ANY MEASUREMENT, NO
  CONCLUSION MOVED. Deliberately kept OUT of the C-6 commit: a wording fix must not ride along
  with a hard-failing assertion that can break callers.)** Local, $0.
  - **NEW STANDING NORM — CHECK A CLAIM ABOUT A FILE AGAINST THE COMMIT THE CLAIM WAS MADE
    ABOUT.** A **distinct** member of the "a check that cannot fail" family: not a check that
    cannot fail, but a check run against the **wrong baseline**, which looks identical from the
    inside because the check *does* fail — honestly — on the file you actually read. Line numbers
    drift under our own fixes. **The instance:** the audit cited a false rule string at
    `verdict.py:519-521`; I read the CURRENT file, found `_per_seed_delta` there, and reported the
    audit wrong. My own C-2 fix had added **64 lines** (758 → 822), moving the string from 520 to
    584. `git show 74b6094:src/trikaal/eval/verdict.py | sed -n '520p'` returns it verbatim. **The
    auditor's citation was accurate; my correction is WITHDRAWN** (struck in the v1.6.2 entry
    below). The supervisor caught it before it propagated. Landed in `CLAUDE.md` beside the other
    five.
  - **A SUPERVISOR CLAIM CORRECTED AND OWNED, recorded with attribution as directed.** The C-2
    ruling stated *"both HALT guards were silently non-functional."* **That is false.**
    `power_guard` reads `headline_series` and never touches `mu_diag`; in the blind arm it
    produced a real `ir_range = 19.56`. Only the degeneracy guard was disarmed — blast radius one
    guard, not two. The supervisor took the auditor's phrasing and asserted it without checking,
    and records it as the third propagated-unverified-claim of the week. **The more important half
    was the builder's extension, which the audit missed:** `tests/eval/test_verdict.py:74` also
    omitted `mu_diag`, so the entire verdict KAT suite was exercising the blind path — the tests
    could not have caught the defect they exist to catch, one layer deeper than where the auditor
    found it.
  - **RECEIPT CORRECTED — `m6_prefill_decision_stability`.** The n=256 receipt persisted
    `fixture_can_discriminate: false` **and** the verdict "CONDITION (a) SATISFIED" from an
    observed flip rate of 0. At that size the spacing between adjacent |μ̂| order statistics near
    the median is ~1e-4 against a max|δ| of 3.5e-06, so no decision could straddle a threshold:
    zero flips was the only attainable outcome. Fixed **in the script and regenerated** — not by
    hand-patching a committed artifact, which is a process error already on this record. The flag
    is now BINDING: when it is false the verdict is `PROBE INDETERMINATE` and no conclusion is
    drawn from the flip rate. Verified by diff: **every measured field is bit-identical**; only
    `verdict` and two new documentation keys changed. **The ruling is unaffected** — condition (a)
    was closed on the ANALYTIC impact bound, which never used the observed count; the flip
    observation was corroboration and was never able to carry the conclusion. The new branch was
    discrimination-checked both ways before regenerating (real n=256 spacing → INDETERMINATE;
    decisions placed on the boundary → a conclusion permitted).
  - **THE Z1 CRITERION ERROR, recorded as the control-arm norm working.** In the (now paused)
    zero-mean probe I set the allowed flip-direction imbalance at 1/√2231 — the flip count
    **pooled** across all 25 (cell, seed) units — when IR is computed **per unit**, making the
    correct half-width 1/√89.2449 = 0.10585, a 5× over-tightening. Confirmed by reconstructing the
    bound: `worst/rms = 9.496 = √89.2449`. It surfaced because my own sign-randomised control
    **failed**, and the probe emitted `PROBE INVALID` rather than a conclusion about the real
    perturbation. That is the control-arm rule doing exactly its job.

- **v1.6.2 (2026-08-01, AUDIT TIER-1 C-7 — THE THRESHOLDS THAT DECIDE THE VERDICT WERE UNDER NO
  GATE. DEFECT FIX, NOT A SPECIFICATION CHANGE: every pinned VALUE is unchanged; what changed is
  that they are now READ. Gate-A anchor `3f86882a` re-proven byte-identical ×2, exit 0.)** Local,
  $0.
  - **THE DEFECT.** `ECON_FLOOR_IR` (§3 clause 4's materiality floor), `BOOT_POWER` (the power
    point inside MDE_paired) and `PLACEBO_DISPERSION_TRIPWIRE` (§7 v1.5 A.4.3) appeared **nowhere**
    in `conformance.py`. Each was a bare module constant in the file that consumes it, so editing
    any one of them would have moved a §3 clause with nothing objecting. This is C-2 one level up:
    not a gate that examines nothing, but a **pinned surface that pins nothing**.
  - **THE FOUR DECORATIVE `PINNED_DSR` KEYS, resolved by wiring rather than deletion** — each was
    carried in the dict and read by no code path:
    - `n_trials_factorization` was the prose *"cells x horizons x kappas (seeds are replicates,
      not trials)"*. It is now the **axis tuple the multiplicity count is computed from**, so
      re-introducing `seeds` changes `n_trials` and fails the gate instead of merely describing
      the rule it was supposed to protect.
    - the `var_sr` prose became `var_sr_ddof: 0`, **read** by the expected-variance re-derivation
      and **separately asserted to be 0** — a drift to a sample variance cannot pass by quietly
      moving the expectation along with the computation.
    - `placebo_dispersion_tripwire` was worse than unread: a **second independent copy** of the
      1.5 that `verdict.PLACEBO_DISPERSION_TRIPWIRE` also holds — the same two-files-own-the-truth
      shape that produced C-6. The two are now cross-checked. `DSR_VAR_SR_BASIS_CELL` was
      audited for the same shape and given an explicit equality check (it had been enforced only
      indirectly, via a confusing downstream variance mismatch).
    - `statistic` is now **read by the shipped manifest**: the clause-5 `rule` string is BUILT
      from the pins.
  - **A TIER-3 SITE CLOSED AS A SIDE EFFECT, AND ITS LOCATION CORRECTED.** The clause-5 `rule`
    string persisted into the verdict manifest read *"var_sr over the 180 de-annualized VAL IRs
    (ddof=0)"* — wrong on **both** counts after v1.5 (N is 60; the basis is the cell-5 placebo
    subset, not all arms). The shipped artifact would have carried a false account of how its own
    `var_sr` was computed. Building the string from the pins makes that disagreement structurally
    impossible. The rest of C-17 remains open Tier 3.
    - **WITHDRAWN (2026-08-01) — MY CORRECTION TO THE AUDIT'S CITATION WAS ITSELF WRONG.** This
      entry first read: *"The audit located this at `verdict.py:519-521`; the actual site is the
      `5_dsr` block (pre-fix `583-585`) — `519-521` is inside `_per_seed_delta` and carries no rule
      string."* **That is false and is withdrawn.** `git show 74b6094:src/trikaal/eval/verdict.py |
      sed -n '520p'` returns exactly
      `f"series; SR0 from N={DSR_N_TRIALS} enumerated trials; var_sr over the 180 "`. The
      auditor's citation was accurate against the commit they read. I checked it against the
      CURRENT file, which **my own C-2 fix had lengthened by 64 lines** (758 → 822), moving the
      string from 520 to 584 — and then declared the audit wrong rather than diffing against the
      audited commit. The supervisor caught it before it propagated.
  - **NEW GATE.** `PINNED_THRESHOLDS` + `pinned_threshold_failures()`, an **independent second
    statement** of each value rather than an alias, called from **both** entry points — the
    money-run gate (`assert_conformance`) and the verdict gate (`verdict_dsr_failures`) — so a run
    cannot start under an edited floor and discover it only at the verdict. Imported locally
    because `verdict` imports `conformance`; the dependency runs one way.
  - **EVERY KAT IS A MUTATION TEST** (`tests/eval/test_pinned_surface.py`, 14). Asserting a pin
    equals itself is circular and would pass against a gate that does nothing, so each case
    perturbs the live constant and requires the gate to name it: halving `ECON_FLOOR_IR`,
    weakening `BOOT_POWER` to 0.5, loosening the tripwire to 3.0, desynchronising the two
    tripwire copies, moving the basis cell, re-adding `seeds` to the factorization, dropping an
    axis, and switching to a sample variance. Plus a clean-surface pre-check so no mutation can
    pass vacuously, and an unread-key detector that is **itself** discrimination-tested.
  - **ONE OF MY OWN CHECKS FAILED ITS FIRST RUN AND WAS FIXED AT THE RIGHT LEVEL.** The
    `statistic` KAT initially asserted on `verdict.py`'s SOURCE TEXT and flagged this very file's
    comment explaining the string's removal — a source-substring check cannot distinguish a live
    rule string from a comment about a dead one. Replaced with an assertion on the **emitted
    manifest**, via a real 25-artifact fixture through the shipped path, plus a sentinel mutation
    proving the pin is read. Suite 365 green, ruff clean.

- **v1.6.1 (2026-08-01, AUDIT TIER-1 C-2 — A BINDING HALT GATE WAS SILENTLY DEAD IN THE DECISION
  PATH. DEFECT FIX, NOT A SPECIFICATION CHANGE: no pin, threshold, seed, enumeration or gate VALUE
  moved; the degeneracy guard stays HALT-only and per-seed. Gate-A anchor `3f86882a` re-proven
  byte-identical ×2, exit 0, after the `src/` edit.)** Local, $0. Found by an independent audit,
  reproduced here before being touched.
  - **THE DEFECT.** `write_cell_eval_artifact` took `mu_diag: dict | None = None` and persisted
    `mu_diag or {}`; `_validate_artifact` never inspected it; `degeneracy_guard` read its only two
    inputs via `.get(..., nan)`, and since every NaN comparison is False it returned a **hardcoded**
    `"armed": True` with `halted: False` having examined nothing. `m6_toy_rehearsal.py:445` — the
    driver the end-to-end validation run goes through — had `head_score.mu_diag` available and
    never passed it. That run would have delivered exactly its stated deliverable, a verdict word
    with `dual_specification`, while a binding HALT gate was inert. This is the project's own
    *"a check that cannot fail is not a check"* pattern, in the decision path, on the very run
    meant to validate the guards. **Recorded as instance seven.**
  - **REPRODUCED BEFORE FIXING.** Same 15-artifact fixture, cell 4 planted at
    `frac_negative = 0.999`: with `mu_diag` present → `halted: True, degenerate: [4]`; with
    `mu_diag` absent → `armed: True, halted: False`, per-cell values `None`, `reasons: []`.
  - **ONE PART OF THE FINDING AS STATED IS REFUTED.** It was reported that *both* HALT guards were
    non-functional. `power_guard` reads `headline_series` and computes IRs from it — it never
    touches `mu_diag` — and in the blind arm it still produced a real `ir_range = 19.56`. Only the
    degeneracy guard was disarmed. The blast radius is one guard, not two.
  - **A SECOND SITE, NOT IN THE ORIGINAL FINDING.** `tests/eval/test_verdict.py:74` also omitted
    `mu_diag`, so every clause KAT in the main verdict suite was exercising the blind path too.
    Fixed with an explicit benign payload.
  - **THE FIX, at all four layers.** (i) `mu_diag` is a REQUIRED keyword and is validated at the
    single emission path — empty, missing-key and non-finite payloads raise `VerdictInputError`
    and no file reaches disk; (ii) `_validate_artifact` rejects an on-disk artifact whose
    `mu_diag` the guard could not read, so a hand-edited or pre-v1.6 file is refused by the
    loader; (iii) `armed` is now COMPUTED from whether every `(cell, seed)` yielded finite values,
    `unreadable_inputs` names what was missing, and the guard **fails closed** — unreadable inputs
    HALT instead of passing; (iv) the driver forwards `head_score.mu_diag`, and the eval-resume
    gate now compares against the `ARTIFACT_SCHEMA` constant instead of a hardcoded
    `"m6_cell_eval_v1"` literal, which would have kept resuming artifacts written under the
    superseded contract. `ARTIFACT_SCHEMA` → `m6_cell_eval_v2`; the bump is what makes a v1
    artifact fail loudly rather than resume into a blind guard.
  - **THE KATs THAT WERE MISSING** (`tests/eval/test_mu_diag_required.py`, 14 tests): writer
    refuses; loader refuses; guard reports `armed: False` + `halted: True` and never the historical
    `armed ∧ ¬halted ∧ nothing-examined` signature; a single unreadable seed still halts; and the
    guard **still discriminates** benign from degenerate, so fail-closed has not become
    halt-on-everything. Call-site coverage is checked on the AST — the "flag accepted but never
    forwarded" class that actually bit here — and **that checker is itself mutation-tested**
    against a synthetic call omitting `mu_diag`, per the fixture-discrimination norm.
  - **THE FIX IS MUTATION-PROVEN.** Reverting `armed` to a literal → the guard KAT fails;
    removing the `_validate_artifact` leg → the loader KAT fails; deleting the driver's
    `mu_diag=` line (the original defect, restored exactly) → the call-site KAT fails. Suite
    351 green, ruff clean.

- **v1.6 (2026-07-31, PRE-RUN INSTRUCTIONS — NOT A SPECIFICATION CHANGE. The design stays FROZEN:
  no pin, threshold, seed, enumeration, guard or gate value moved; the Gate-A anchor `3f86882a` was
  re-proven byte-identical after `src/` and `pyproject.toml` edits.)** Local, $0, no spend.
  - **FINDING 0 — THE COST BASIS IS UNBACKED (the priority item).** Receipt:
    `runs_manifest/m6_cost_basis_forensics.json`; full write-up as entry 0 of
    `docs/m6_claims_audit_appendix.md`. **(i) Was it ever measured?** The logs **cannot settle it,
    and it is NOT reconstructed**: `runs_cloud/box_logs/item2.log` (the training invocation) has
    **zero non-wandb lines** — only wandb's stderr was captured, so the script's own
    `[throughput]` stdout is absent entirely. What IS settled: training happened; the manifest on
    disk is the `--eval-only` re-run (`config.eval_only_rerun=true`) writing to the SAME path, so
    the training invocation's manifest was **OVERWRITTEN**; `item2b.log` shows that re-run printing
    `nan steps/s → nan bars/s`. **A lost measurement, not a failed one.**
    **CORRECTION TO THE RULING'S DICHOTOMY:** there is a third option, and it is what happened —
    **47.2k is a REAL measurement of a DIFFERENT quantity, persisted numerically in a DIFFERENT
    artifact.** It is exactly reproducible from `m6_attention_bench.json`
    (`steps_per_s` 2.8819876814331926 × 32 × 512 = **47,218.5**). That bench is **compute-only**
    (pre-resident batches; no loader, no Stage-1, no tokenize pass, no eval, no checkpointing) — an
    **upper bound**, not an end-to-end training rate. **NO artifact in the repo carries an
    end-to-end training rate at the canonical geometry**: 20 real end-to-end records exist, all at
    `seq_len=32` (the canary fixture), none at the pinned 512.
    **A SECOND DEFECT, NOT ASKED FOR:** the bench ran under **FORCED determinism** and never
    recorded it (`set_determinism`'s default is `True`) while production
    (`orchestrator.py:159`) runs **UNFORCED** — so 47.2k is a *forced* rate and the
    "1.3× penalty on top of 47.2k → $43–65" arithmetic **DOUBLE-COUNTS**.
    **(ii) INSTRUMENTATION FIXED** — `src/trikaal/utils/throughput.py`: numeric records only, a
    **closed `basis` vocabulary** (`compute_only_step` / `end_to_end_training` / `unmeasured`) so an
    upper bound can never again be quoted as a cost basis, `measured=False` + NaN when a rate is
    unsupported (never 0), and `is_costable()` which accepts **only** end-to-end. Wired into
    `orchestrator.run_cell` (per-stage end-to-end records — the real run now produces the datum as
    a by-product), the attention bench, the determinism probe, and `m6_toy_rehearsal` — which now
    **carries a prior finite record forward instead of clobbering it**. 12 KATs.
    **(iii) EVERY DOWNSTREAM COST CLAIM RE-LABELLED ESTIMATED-NOT-MEASURED** ($20–30, $33–50,
    $43–65, "2–3 GPU-days", the 1.3×) — see the claims-audit appendix §2.
    **(iv) THE STAGED CUDA PROBE NOW CARRIES BOTH JOBS.** It also had a **broken command**: it
    invoked `--forced-determinism`, a flag `m6_attention_bench.py` **did not have** — it would have
    failed at the box. The flag now exists, with `--no-forced-determinism` for the other arm, and
    the staged command measures the pair.
    **ATTRIBUTION, ON THE RECORD:** the supervisor propagated the unchecked figure twice; **so did
    I**, in every prior report that repeated it. Neither of us opened the receipt.
  - **FINDING 1 — THE PROPOSED FLOOR IS NOT A FLOOR (verified before use, as instructed).** Receipt:
    `runs_manifest/m6_mde_floor_verification.json`. **(a)** 3.518 is **not** `2.4865 × SE_boot`; it
    is an **ANALYTIC** iid-normal MDE, `z_sum·√(2/T_eff)·√(periods_per_year(h))`
    (`scripts/m6_prereg.py:74`), reproduced to 3dp (**3.5183**) from the receipt's own `T_eff`.
    Inverting it yields the **analytic** SE (1.4149), not the bootstrap `SE_boot` v1.5 multiplies —
    an **estimate of an estimate**. **(b)** `3.0728 × 1.4149 = 4.3476` is **NOT a lower bound**: the
    3.0728 multiplier is the **ν→4 limit**, which requires the training term to DOMINATE, while the
    product is taken against an SE that IGNORES it — mutually exclusive limits. Verified through the
    repo's own `paired_bootstrap` arithmetic: the MDE is **monotone increasing** in `se_train`, so
    its infimum is at `se_train=0`, where Welch–Satterthwaite returns ν→ν_boot=9,999, the multiplier
    returns to 2.4865 and the MDE returns to the v1.2 number. **The pre-run computable floor is
    3.518**; 4.3475 is reached only once `se_train ≥ 0.685×se_boot`, unknowable pre-run.
  - **ITEM 1 — REFRAME (done; the anchor rule fired and was honoured).** "Foundation model" removed
    at all six sites the ruling named plus a re-grep of my own (`pyproject.toml`, `README.md`,
    `CLAUDE.md`, the design spec ×2, `src/trikaal/__init__.py`); repo-wide re-grep now returns only
    the prohibition itself. **Trikaal is a TOKENIZER STUDY; the backbone is the measurement
    vehicle.** Param count corrected `~27M` → **21,301,248** at `CLAUDE.md:7`, `CLAUDE.md:33`, the
    design spec's "~27M-class" and its §3 body text — **and at `m6_prereg.md`'s R3 text**, which is a
    *publishable outcome sentence* and would have shipped the wrong number. Because `src/` and
    `pyproject.toml` changed, **Gate-A was re-proven: exit-0 ×2, manifest byte-identical to the
    pre-change baseline both times, `results_hash 3f86882a`, causal sweep 420/420.**
  - **ITEM 2 — POWER STATEMENT drafted PRE-DATA** (`docs/paper_skeleton.md` §7): states the **floor
    (≥ 3.518)** and the **structure** (two-term, t + Welch–Satterthwaite, training term supplied by
    the run), **never a point number**, and makes INCONCLUSIVE read as instrument-reporting. Every
    number verified against the live pinned surface. **The skeleton locks NO title and NO abstract**
    — the legibility gate decides which paper this is.
  - **ITEM 3 — TAXONOMY AUDIT: TWO HOLES FOUND, FLAGGED NOT FILLED.** Receipt:
    `runs_manifest/m6_taxonomy_audit.json`. **R1 and R3 are determinate.** **R2/R2b are NOT:**
    (1) *"degenerate at a MAJORITY of seeds"* does not say majority over **one leg** or over the
    **union** — and the guard deliberately persists the two legs (sign-lock, activity-never-binds)
    as separate seed lists; a fixture with 2 seeds locked and 2 *different* seeds never-binding is
    majority-by-union (R2) and minority-by-leg (R2b) simultaneously. (2) R2b's headline test
    (*strict minority of seeds*) and its parenthetical (*per-seed leg fires while the seed-mean is
    in band*) are **different conditions presented as one**, and can disagree. Both demonstrated on
    concrete fixtures. **Filling them is a specification choice and the design is frozen.**
    **S=5 COLLAPSE CONFIRMED:** R1 and R2b prescribe the same remedy (add seeds within the cap), the
    cap is exhausted at 5, so every INCONCLUSIVE path terminates at **R3 with zero re-runs** — R3 is
    the operative rule. **The holes do not change the OUTCOME at S=5, but they do change what is
    REPORTED** (R2 requires the degeneracy to be written up as a PRIMARY mechanism finding; R2b
    treats the same halt as a training lottery), so the ambiguity is live for the paper's claims.
  - **ITEM 4 — CLAIMS-AUDIT APPENDIX BUILT** (`docs/m6_claims_audit_appendix.md`), every substantive
    claim tagged MEASURED (artifact + field + commit) or ESTIMATED. Sweep receipt:
    `runs_manifest/m6_claims_sweep.json`, **idempotent (re-run byte-identical)**. Repo-wide the
    Finding-0 signature occurs **exactly once** (P3=1, the entry-0 string). P2-HIGH = 5: **3 true
    positives** (all `m6_rehearsal_manifest.json::throughput.*`) and **2 FALSE POSITIVES OF MY OWN
    AUDIT** (`token_causality_probe`'s flip rates matched on the substring "rate"; 0.0 there is the
    genuine and desired causal-safety result, corroborated by the companion 20-position array).
    35 P2-LOW are per-step `rollout_h2_corr` NaNs — undefined by construction, not defects.
    **ALSO FOUND, LISTED NOT FIXED:** `m6_eval_throughput_expectation.json`'s
    `real_scale_decision_count_h15_headline.seeds = 3` is stale under v1.5 (S=5), understating
    `total_decisions` by 5/3.
    **TWO SELF-INFLICTED AUDIT DEFECTS DISCLOSED:** the sweep first scanned **its own output** and
    the forensics receipt (self-inclusion — the same 49→98 bug), fixed with a skip set and
    **idempotence asserted, not assumed**; and the first severity pass **buried the 3 real hits
    under 35 benign NaNs**, which is how the original defect survived in the first place.
  - **ITEM 5 — final pre-flight board re-run; see the v1.6 board section of the report.**
  - **THE STAGE-1 STEP BUDGET — A GAP THAT EXISTED UNNOTICED (documentation of an existing default;
    NOT a specification change, so the freeze is untouched).** The §4 budget line — "~1 effective
    pass over the quality-weighted draw (~270–300M effective bars)" — is a **Chinchilla derivation
    for the 21.3M AR** (540M tokens ÷ ~2 subtokens/bar ≈ 270M bars, per the
    `training-saturation-budget` note), so it pins **STAGE-2 ONLY**: 16,479–18,311 steps at batch
    32 × seq 512. **"Effective bars" = raw bar-visits**, confirmed three independent ways: the
    sampler draws windows **with replacement** (`universe_loader.py:229`) so bars are revisited;
    270–300M **exceeds** the ~213M distinct train-region bars, which is coherent only as a visit
    count; and the canary recipe already computes `bar_visits = steps × batch × seq_len` against
    distinct `train_bars`.
    **READ FROM CODE, NOT CHOSEN:** `OrchestratorConfig.steps_stage1` defaults to **2000** (as does
    `steps_stage2`), and **every driver in the repo** — `m6_smoke.py`, `m6_toy_rehearsal.py`,
    `m6_cuda_probe.py` — sets `steps_stage1 == steps_stage2` from a **single `--steps` flag**. No
    driver sets them independently; no separate Stage-1 budget exists anywhere; no real-run driver
    exists yet. So Stage-1 **inherits** the Stage-2 count **by code convention, not by written
    specification**.
    **WHY THIS MATTERS AND WHY IT DOES NOT.** Ablation **VALIDITY is UNAFFECTED** — all five cells
    take the same Stage-1 budget from the same flag, so the comparison stays clean and the only
    varied factor is still quantizer × arm. **REPRODUCIBILITY IS AFFECTED**: the prereg pins the AR
    budget and is silent on Stage-1, so a replicator would have to read three driver scripts to
    discover the coupling. Recorded here and in `runs_manifest/m6_cuda_validation.json`
    (`stage1_budget_from_code`) as a documented operational parameter.
  - **STANDING CREDENTIAL RULE (§7 v1.6, landed in `docs/cloud_runbook.md` §3 as a hard
    precondition on any cloud transfer leg).** **Never ship a write-capable or non-fine-grained
    credential to rented third-party infrastructure.** The risk is **INTEGRITY, not
    confidentiality**: the lake derives from public Binance data, so read exposure is low-stakes,
    but a write-capable token can **overwrite or delete the Merkle-`5dfd667d` anchor** the entire
    reproducibility claim rests on — on a machine rented by the hour from an anonymous host. A
    fine-grained read-scoped token reduces the residual to acceptable read exposure. The available
    HF token is `role: write`, `fineGrained: false`, so the **HF pull is DEFERRED, not cancelled**;
    the rsync fallback is proven, so the real run is **not blocked** — HF is a transfer-time
    optimisation, to be proven as a cheap **standalone** step before the run, never at hour zero of
    a ~50-hour rental.
  - **VERDICT ON THE RUN: NO-GO, unchanged.** Three blockers: **B1** (invariant 7 — needs the
    probe), **GPU credentials**, and now **the absence of a receipt-backed cost**. No rental
    instruction issued; the probe stays STAGED.

- **v1.5 (2026-07-30, the AMENDMENT WINDOW — SIGNED OFF AND IN FORCE). THE LAST SPECIFICATION
  CHANGE BEFORE THE RUN; after this the design is FROZEN.** Full text: `docs/m6_prereg_v1_5_final.md`
  (working drafts: `docs/m6_prereg_v1_5_drafts.md`). Time-boxed on purpose: several central numbers
  were derived assuming training variance is negligible, which v1.4.5 **measured** to be false, and
  they could be revised legitimately **only while no real Cell-4 number existed**.
  - **⚠ CORRECTION TO THE WINDOW'S OWN PREMISE — the ruling's framing was WITHDRAWN, and the
    correction ORIGINATED WITH THE BUILDER** (recorded here at the supervisor's instruction, the same
    treatment C1 received). The ruling stated *"every amendment in this window loosens the effective
    bar."* **That is false.** Clause 5 (deflation) loosens by `M(60)/M(180) = 0.859×` times an
    unmeasured `√f`; **clause 2 — the clause that most directly tests the headline contrast —
    TIGHTENS twice over**: the quantile multiplier goes `2.4865 → 3.9806` at S=3 (1.601×) or
    `→ 3.0728` at S=5 (1.236×), **and** `SE_total = √(SE_boot² + SE_train²) ≥ SE_boot` always. The
    net across the conjunctive gate is **indeterminate pre-run and NOT monotonically loosening.**
    Had the original framing survived into the paper it would have been a false statement about our
    own amendments, in the section a referee reads most carefully. **No cosmetic offset was invented
    anywhere** — the dual report is the safeguard.
  - **A.5 — N = 60 (`PINNED_DSR["n_trials"]` 180 → 60; seeds REMOVED from the cross-product).**
    N counts **distinct configurations evaluated and displayed**; it does **not** count **replicates**
    of a configuration. A multiple-testing adjustment cannot legitimately get stricter because you
    replicated the same configuration more times. Seeds are the only replicate axis, so seeds are the
    only axis removed — cells, horizons and κ are distinct configurations and stay. **The chain stops
    there and we declined the further 0.385× loosening** to κ-only N=4 that the ruling half-offered:
    removing cells/horizons would claim credit for a discipline (pre-registration of what to *report*)
    that DSR does not measure — DSR discounts the **search performed and shown**, and we show all 60.
    *Disclosed limitation, arguing the other way:* N=60 omits development-time search, so a referee
    could argue for more — a reason not to go lower.
  - **A.4 — `var_sr` basis → the PLACEBO arm (cell 5).** Principle: **a null dispersion estimate must
    not contain the treatment contrast.** The old all-arms basis meant that if the microstructure
    claim were TRUE, cell 4 would separate, the cross-trial spread would grow, `var_sr` would grow and
    **clause 5 would get HARDER** — the clause was **anti-correlated with the hypothesis it tests**,
    and at realistic effect sizes was likely unpassable (required cell-4 annualized IR 3.01/4.38/7.11/
    9.84 at VAL-IR spreads 0.5/1.0/2.0/3.0, against an economic floor of 0.5). Cell 5 is signal-free
    **by construction** and matched to cell 4 on quantizer and input dimensionality. **NOT cell 2** —
    §5's NULL-fallback can CLAIM `IR(2)−IR(1)`, which makes cell 2 a *treatment* arm, and 7 vs 16 dims
    is a dispersion mismatch. **Placebo-victim concern resolved:** `pb(5−2) = −2.59` is a **LEVEL**
    statement and `var_sr` is a **DISPERSION** statistic — a level shift does not move a variance. The
    second-order risk (degraded training → more variable models) errs **conservative** and is
    **measured** by a pre-data **1.5× tripwire** against the median dispersion of cells 1–3;
    if it fires, report clause 5 under **both** cell-5 and within-cell-4 bases, never cell 2.
  - **B — MDE carries an explicit training-variance term.** Law of total variance:
    `Var(ΔÎR) = E[Var_scoring|models] + Var_models(E[ΔÎR|models])`. Term 1 stays `SE_boot` (and seeds
    do **not** reduce it — all seeds are scored on the same data and grid, so scoring noise is
    common). Term 2 is **clean** — cross-seed variation is purely model-induced. `σ̂²_train =
    var({ΔIR_s}, ddof=1)`, `SE_train = σ̂_train/√S`, `SE_total = √(SE_boot²+SE_train²)`,
    `MDE = (t_{0.95,ν} + t_{0.80,ν})·SE_total` with Welch–Satterthwaite ν. **t, not z**: at S=3 the
    term has 2 df and `t_{0.95,2} = 2.9200` vs `z = 1.6449`, so z is not conservative once the term
    enters. **Direction recorded pre-data: this makes the primary test HARDER.** The tabled
    `MDE_paired` is superseded as a **number**; its scoring component is retained as a **term**.
  - **C — OUTCOME TAXONOMY: three pre-committed, publishable outcomes.** SURVIVES / NULL /
    **INCONCLUSIVE**. The guards could already emit `HALT_ADJUDICATE` while the prereg described the
    experiment as binary; that gap is closed. INCONCLUSIVE response is selected **BY RULE**, first
    match governing: **R1** power-halt only → add seeds to the cap, re-run ONLY the named cells;
    **R2** degeneracy at a MAJORITY of seeds → no re-run, the mechanism becomes a primary finding;
    **R2b** degeneracy at a strict MINORITY (per-seed leg, seed-mean in band) → that signature IS the
    training lottery, treat as R1; **R3** report INCONCLUSIVE. **Bound: ≤1 re-run round ever, ≤5 total
    distinct seeds.**
    - **BECAUSE C.1 RULED S=5 UP FRONT, R1 AND R2b COLLAPSE INTO R3 — the design is COMMITTED to R3
      as the INCONCLUSIVE response, with no seed headroom. R3 is therefore written out in full as an
      acceptable, signed-off-in-advance answer, and must not read as a dead end when it arrives:**
      > **R3 — INCONCLUSIVE, fully specified.** Report: the measured ΔIR(4−5) and ΔIR(4−2) **point
      > estimates with their CIs**; the realized `MDE_paired` including the training term; the
      > **per-cell across-seed IR distribution** (per-seed values, range, std) that defeated the
      > claim; the guard reason and which rule fired; the per-seed `frac_negative` and
      > `activity_decisions`; and the `dual_specification` block. The stated conclusion is: **"the
      > effect, if any, is smaller than the training variance of our own design and is unresolvable
      > at this scale."** The ablation headline is **WITHDRAWN, not softened**. The mechanism result
      > is reported separately and only at §F strength. This is a **complete, publishable finding**
      > about the detectability limit of a 21.3M-parameter two-stage design at this data scale — it is
      > pre-committed as an acceptable answer, not a failure to report one.
  - **D — `PINNED_SEEDS` (0,1,2) → (0,1,2,3,4); S = 5 UP FRONT (C.1 Option 2).** Decided on the
    substantive asymmetry, not optics: a power halt at S=3 is plausibly a fixable sample-size
    artifact; at S=5 the effect is genuinely small relative to training variance, so more seeds would
    be marginal and having no remedy is an **honest stopping point**. The √(5/3) framing applies
    **only** to the training term (`SE_boot` is untouched by seed count), so the MDE gain is 0–22.5%
    and **not quantifiable pre-run**; **the statable benefit is B's degrees of freedom 2 → 4**,
    cutting `t_{0.95}` 2.9200 → 2.1318 (**−27.0%**) and the combined multiplier 3.9806 → 3.0728
    (**−22.8%**). Under A.5, N = 60 independently of S, so **there is no DSR tightening at all** —
    clause 5 unchanged, clause 2 improved, **net sign positive**.
    - **SPEND CONTINGENCY, BOTH PATHS PRE-REGISTERED NOW so neither becomes post-hoc.** S=5 raises
      the run to **$33–50** (~**$43–65** with forced determinism at the illustrative 1.3×). **That
      increase is LAKSHAY'S to approve and he has not yet** *(true as written on 2026-07-30;
      SUPERSEDED 2026-07-31 — see the OPERATOR DECISION immediately below. Left standing rather than
      rewritten: this is a dated record, and the pre-registration is only worth anything if the
      state before the approval is still legible)*. **S=5 is the pre-registered primary**;
      **if the operator declines the cost, the design falls back to S=3 under the SAME v1.5
      specification in all other respects**, with the S=3 multiplier **3.9806** already tabled here.
      The choice between them is then a **budget** decision, never a **specification** one.
    - **OPERATOR DECISION — 2026-07-31: LAKSHAY APPROVED S=5. The contingency is RESOLVED in favour
      of the pre-registered primary; the S=3 fallback is NOT exercised.** Recorded here as what it
      is: a **BUDGET approval of a pre-registered primary**, taken with **no M6 result in hand and
      nothing seen** — it selects which of two paths already written down on 2026-07-30 gets funded,
      and it is **not a specification choice**. Nothing about the v1.5 specification moved with it:
      `PINNED_SEEDS = (0,1,2,3,4)`, `n_trials = 60` (N is independent of S under A.5), the
      Welch–Satterthwaite multiplier **3.0728** at ν=4, and the R3 commitment all stand exactly as
      signed off. The approved figure is **$33–50**, or ~**$43–65** if invariant 7 closes on
      amendment A (forced determinism, illustrative 1.3×) — the invariant-7 A/B ruling is a
      **separate, still-open** operator gate and this approval does **not** pre-decide it.
  - **E — legibility-gate adjudication, pre-written.** Reframe: real TFI is low-variance and weakly
    price-correlated — exactly the eviction profile — and the λ fix was validated on a **synthetic**
    tuned signal, so **a firing is the MODAL prediction of our own finding**, and simultaneously the
    ablation's blocker and the mechanism's strongest real-data evidence. Rules: gate fires → **STOP
    and report** (unchanged); **the PRIMARY becomes the mechanism finding**; λ may be re-derived **at
    most ONCE**, by the pinned formula, **on a slice carved from the END of the TRAIN region — never
    block 0** (verified: block 0 already carries κ* **and** every clause-5 trial entry, and λ is a
    *training* hyperparameter set before any scoring; the real run had **no** in-train holdout, so
    the slice is created by this amendment — `gates.lambda_calibration_boundary_ms`, inert unless the
    gate fires); the re-derivation must emit **the pinned formula's calibration receipt on REAL
    data**; **the re-derived λ counts as an additional configuration in the clause-5 multiplicity**
    (N 60 → 120 for the affected report) — widening the search without paying for it is exactly what
    DSR exists to prevent; and **the branch structure is biased and is disclosed as such** — the
    contingency can only ever *help* the ablation, so any ablation under a re-derived λ is reported
    **SECONDARY/exploratory**, naming which λ was used and that it was the second.
  - **SAFEGUARDS (binding).** (1) **Direction-blind test answered per amendment.** A.5's evidence is
    hard rather than asserted: the same amendment **deliberately keeps `var_sr` on the full
    individual-trial set**, the tightening half of the available choice — pure seeds-as-replicates
    offered 0.45–0.86× and we took 0.859× and declined the rest; *a motivated actor takes both.*
    A.4's direction was **genuinely unknown at adoption**. B is itself a tightening. (2) **The
    aggregate** is reported as one number (above). (3) **`dual_specification` is a REQUIRED manifest
    field**: the headline is computed from the SAME artifacts under BOTH the v1.2 original spec
    (seeds-as-trials, all-arms `var_sr`, z-quantile scoring-only MDE) and the v1.5 amended spec.
    **v1.5 is the pre-registered primary; any disagreement is a FIRST-CLASS FINDING reported in the
    abstract and results, never resolved in our favour or relegated to an appendix.** A manifest
    lacking the field **cannot be quoted as the M6 outcome** — the same footing as `grid_pinned`.
  - **A NON-DISCRIMINATING TEST FIXTURE, FOUND AND FIXED (disclosed).** The A.4 mutation check —
    "the gate rejects the superseded all-arms `var_sr` basis" — could not fire on the existing
    conformance fixture, because that fixture's trial values were a modular function that made the
    placebo-only and all-arms dispersions **numerically identical**. The assertion would have passed
    vacuously on a gate that did nothing. Caught only because the mutation test failed to fail. The
    fixture is now cell-dependent, **and the test asserts `_var_all_arms(full) != _var0(full)`
    before using it** — a mutation check must first prove its own fixture can tell the two cases
    apart. Same family as the control-arm norm: a check that cannot fail is not a check.
    - **LANDED AS A STANDING NORM (2026-07-31, `CLAUDE.md` "Tooling & commands", beside the
      control-arm rule).** **A mutation or negative check must first prove that its own fixture can
      discriminate the cases it claims to separate; an assertion that would pass equally against a
      gate that does nothing is not a check.** It is filed there as the third costume of a single
      defect — with the **pipefail rider** (v1.4.1: a `| grep | tail` masked an exit-1 and put a
      false anchor claim in a commit message) and the **control-arm rule** (v1.4.7: a probe emitted
      a verdict while both arms died of the same `TypeError`). All three are a harness concealing
      its own failure and letting an unverified claim through. **No new KAT was manufactured for
      it**: the norm's own enforcement already exists as `tests/eval/test_conformance.py:252`, the
      discrimination assertion that the mutation test now makes before relying on its fixture.
      **This is a PROCESS norm, not an experimental parameter — it does not break the v1.5 freeze**;
      no pin, threshold, seed, enumeration, or gate value moved.
  - **VERIFICATION.** Mutation KATs prove the conformance gate **REJECTS** the pre-v1.5 values
    (N=180, the seeds-as-trials 300, and the all-arms `var_sr` basis) and still rejects a missing
    seed in the enumeration; the Student-t implementation is KAT'd against published tables and the
    ν→∞ normal limit; the guards' seed loops and every fixture are now **S-agnostic**. **`harness.py`
    is NOT touched** — its `5*3*4*len(KAPPAS)` = 240 is the M5 instrument frozen under Gate-A,
    re-verified this pass. **NOTHING PINNED HAS
  CHANGED:** `conformance.PINNED_DSR` (N=180), `verdict.DSR_N_TRIALS`, `PINNED_SEEDS` (3),
  `PINNED_MICRO_POINT_WEIGHT` (3.0) and the tabled `MDE_paired` are all as ruled. The window is
  time-boxed on purpose: several central pre-registered numbers were derived assuming training
  variance is negligible, which v1.4.5 **measured** to be false, and they can be revised
  legitimately **only while no real Cell-4 number exists**. Nothing in the drafts is quotable as the
  pre-registration until it is ruled on and landed here. **Contains one flagged finding
  (draft §A.4) that may outrank the item it was found inside: clause 5's `var_sr` basis makes the DSR
  bar rise when the ablation succeeds, so the clause is anti-correlated with the hypothesis it
  tests, and may be unpassable at realistic effect sizes.**

- **v1.4.7 (2026-07-30, post-v1.4.6 ruling: bit-exactness sufficiency defect scoped, a NEW named
  statistical-power risk mitigated, item 4 resolved; supervisor-ordered) — BEFORE any real training.
  Local $0 CPU only; no spend authorized, none taken.**
  - **CANARY GATE: CLOSED — PASS. Pre-flight item 2 is GREEN and the board is 8 of 8 — AND 8/8 DOES
    NOT MEAN CLEARED TO RUN, because a new blocking item (B1) postdates the board.** Both halves of
    this belong in one paragraph and must never be split: **anyone quoting "8/8" without B1 attached
    is quoting it wrong.** The close is at exactly this strength: legs (a), (b) and (c) are all
    demonstrated; leg (c)'s single-checkpoint residual is **ATTRIBUTED** to unconstrained CUDA
    reduction order (not hardware, not a mystery) and **MITIGATED** by two armed HALT-only guards;
    the control arm is clean at **0 of 16** noise cell-horizon combinations; and no remaining
    affordable test adds information.
    - **B1 — DETERMINISM POSTURE (BLOCKING; OUTSIDE the 8-item board).** Blocking for **two
      independent** reasons: **(i)** reproducibility is a stated deliverable in invariant 7 and a
      paper claim, and **49 of 49** records currently assert it falsely; **(ii)** the measured
      same-seed basin-hopping leaves the 3-seed design's statistical power **unquantified**. B1
      closes on **three** things: the STAGED CUDA feasibility + throughput probe (~$1–2, awaiting
      credentials); **Lakshay's** choice between amendments A and B
      (`docs/invariant7_amendment_decision.md`); and — **under A only** — the same-seed twice-run
      **weight-hash comparison**, proving that forcing DELIVERS bit-identity rather than merely
      claiming it. **The power guard stays ARMED under EITHER amendment**, because A removes the
      same-seed component and NOT the across-seed one.
  - **STANDING NORM (generalizes; sits alongside the PIPEFAIL rider).** **A probe or gate MUST
    REFUSE to emit a verdict when its own CONTROL ARM fails.** "Both arms failed identically and the
    script printed a conclusion" is a recurring failure mode, not a one-off: it produced a false
    "FORCED DETERMINISM RAISES" verdict in this session's first feasibility probe, where a
    `TypeError` killed the forced *and* unforced runs alike. Every probe that compares a treatment
    against a control now reports **PROBE INVALID** unless the control completes, and must
    distinguish the failure signature it is testing for from an unrelated fault. Implemented in
    `scripts/m6_determinism_probe.py` (`probe_valid`, keyed on the unforced baseline). Companion to
    the pipefail rider — same class of defect: a pipeline that masks its own failure and lets an
    unverified claim through.
  - **END-TO-END VERDICT INTEGRATION — PASS (`m6_verdict_integration_check.json`).** Both guards
    were added to the ONLY path that may emit SURVIVES/NULL, and the pre-flight verdict dry run
    (`m6_toy_verdict_manifest.json`) **predates both** — it carries neither guard. Unit KATs prove
    the guard FUNCTIONS behave; they do not prove the DRIVER still emits. `scripts/m6_verdict.py` was
    therefore invoked **as a subprocess** (real §3a conformance gate, real content-hash load, real
    manifest write) with `--allow-toy-grid` on three fixtures. All three PASS: **healthy** → still
    EMITS (SURVIVES, no HALT, `emitted == primary`); **one-locked-seed degenerate** →
    `HALT_ADJUDICATE` end-to-end with `degenerate_cells [4]` from the v1.4.6 **per-seed leg only**
    (its seed-mean 0.6997 is in-band); **wide-seed-spread low-power** → `HALT_ADJUDICATE` with BOTH
    clause-1 and clause-3 flagged underpowered. Every run confirmed `grid_pinned: false` and loud,
    both guards `armed`, `frac_negative_by_seed` / `activity_decisions_by_seed` / `ir_by_seed` /
    `ir_range_across_seeds` present for all five cells and keyed by the pinned seeds, `primary`
    preserved as a clause word, and the HALT surfaced on stdout (the driver now prints the
    power-guard HALT and each guard's per-cell reasons, which it previously did not). The script
    exits non-zero on any failed assertion and is CI-wired.
  - **THE `bit_exact_claim` SUFFICIENCY DEFECT — scoped exactly (`m6_bit_exact_claim_correction.json`).**
    Root cause `attention_mode.py`: `claim = attention_mode == MODE_SDPA`, i.e. **invariant 7's
    sufficiency premise encoded in code**. It is false — deterministic attention is NECESSARY, not
    SUFFICIENT. **49 of 49** determinism records claim bit-exactness with forced determinism OFF on
    `device: cuda` (a 100% rate, not a sample). **Count composition, stated so neither number can be
    quoted misleadingly:** 46 run records over **16 distinct training runs** (each run appears in a
    per-run manifest AND a rollup duplicated across `runs_cloud/` and `runs_manifest/`, so the record
    count over-counts runs ~3×) plus 3 derived copies inside this session's own receipts.
    `orchestrator.py:159` — the **PRODUCTION** path — passes `deterministic_algorithms=False`,
    overriding a default of `True` on a function whose docstring calls it "required for the G2
    determinism gate and any published run".
    - **SCOPE — neither broader nor narrower.** **FALSE:** the GPU-TRAINING bit-exactness clause, for
      the enumerated runs. **INTACT:** invariant 7's data-pipeline / frozen-stats / prediction-replay
      clause (independently evidenced: M4a `dataset_hash` reproduction, token-stream content hashing,
      fixed-chunk rollout replay KATs). **INTACT:** the **Gate-A anchor** — it anchors the CPU
      eval-harness replay on a frozen M3 checkpoint, is genuinely bit-identical (`results_hash
      3f86882a`, re-proven with a byte-identical run manifest) and does not depend on GPU training
      determinism. **NO prior result is invalidated and no number changes**; what changes is that a
      run stamping `bit_exact_claim: true` may not be QUOTED as a bit-exact GPU training run. And it
      is **not** a documentation nit: the same-seed divergence is its observable consequence and is
      BLOCKING for the 15-day run until adjudicated.
    - **NOT REDEFINED.** `bit_exact_claim` keeps its historical meaning — 49 records carry it and
      silently redefining a recorded field rewrites history. `bit_exact_preconditions_met` is the
      field any published claim must cite; the caveat string stays; the corrective receipt enumerates
      every affected record so the correction is discoverable from the ARTIFACTS, not only from here.
      (The receipt skips its own output — without that, each re-run would scan the enumeration it
      wrote last time and the count would inflate; verified idempotent at 49.)
    - **$0 FEASIBILITY PROBE — FORCED DETERMINISM COMPLETES ON CPU**
      (`m6_determinism_feasibility_probe.json`). The real Stage-1 + Stage-2 objectives run under
      `use_deterministic_algorithms(True, warn_only=False)` without raising, so no op in the M6
      training path lacks a deterministic kernel there and "make the claim true" reduces to a GPU
      THROUGHPUT question. **CUDA is NOT yet measured** and has its own non-deterministic kernels.
      **BUILDER ERROR, DISCLOSED:** the probe's first version called a non-existent
      `TrikaalAR.forward(targets=...)`, so BOTH arms failed on a `TypeError` while the script
      printed a determinism verdict. It now (a) uses the orchestrator's own `compute_loss` path and
      (b) **refuses to conclude anything unless the UNFORCED baseline completes** — a forced-run
      failure is only interpretable against a passing baseline, and it distinguishes a
      missing-deterministic-kernel signature from a probe/API fault.
    - **STAGED, NOT RUN (PENDING):** the CUDA throughput delta against the measured 47.2k bars/s.
      `scripts/m6_determinism_probe.py --device cuda --steps 50`; ~$1–2, < 2 min for the probe and
      ~25 min of box time for a quotable forced-vs-unforced pair. To be conclusive it also needs a
      same-seed twice-run weight-hash comparison — confirming forcing DELIVERS bit-identity rather
      than merely claiming it.
    - **POSTURE NOT CHANGED, AND NOT THE SUPERVISOR'S CALL EITHER.** CLAUDE.md invariant 7 embeds the
      false premise in its own text, so amending it is **Lakshay's sign-off**. Both candidate
      amendments are drafted with the measured facts in `docs/invariant7_amendment_decision.md` —
      **(A)** force deterministic algorithms (invariant text unchanged, cost PENDING, degrades to
      `warn_only` + an op allowlist if any CUDA op raises); **(B)** amend the invariant to state GPU
      training is not bit-reproducible, scope the deliverable to pipeline/stats/replay, and report
      across-seed spread instead ($0). **Neither is recommended** until the CUDA probe lands — it is
      the discriminating input.
  - **NEW NAMED RISK: STATISTICAL POWER — mitigated HALT-only at $0 (`verdict.power_guard`).** The
    consequence larger than reproducibility: same recipe, **SAME SEED**, different run gave
    frac_negative 0.5662 vs 0.99883 — **basin-hopping, not float jitter** — and same-seed divergence
    is a **LOWER BOUND on across-seed variance**. The pre-registered MDE was derived from **SCORING
    noise only** (paired moving-block bootstrap over time) and contains **no training-variance
    term**, so it understates the true detection threshold and the 3-seed design may be underpowered
    against its own effect size. **Forcing determinism removes the same-seed component and NOT the
    across-seed component**, so amendment A does not fix this.
    - **THE GUARD.** Per-cell across-seed IR (per seed, range, std) is persisted. Pre-declared: if
      the WITHIN-cell across-seed IR range of the cells a claimed between-cell ΔIR is built from
      meets or exceeds |ΔIR|, the claim is smaller than the seed-to-seed wobble of its own inputs
      and the verdict is **HALT_ADJUDICATE** rather than a reported SURVIVES. Same semantics as the
      degeneracy guard: a PURE post-hoc read that never touches a clause and can **never flip
      SURVIVES↔NULL** — it only refuses to report an outcome its own inputs cannot support. It halts
      symmetrically on a NULL too, because an underpowered NULL is equally uninterpretable (the
      symmetry the degeneracy guard already has). Applied to clause 1 (ΔIR 4−5) and clause 3
      (ΔIR 4−2). **KATs both directions** plus a HALT-only proof that two fixtures with the same
      seed-MEAN but different seed SPREAD give identical clauses and differ only in `emitted`
      (`tests/eval/test_degeneracy_guard.py`, now 11 KATs). This measures the power question on real
      data at zero cost rather than arguing about it now.
  - **ITEM 4 — EXPECTATION STANDS, and the record states WHY.** *Expectation is retained **NOT**
    because it passed the 0.97 traded-set test — **it failed, at 0.9102** — but because at feasible
    cost it is the closest available estimator to the unbiased reference; the pre-registered mc_mean
    at its pinned n=32 is measurably farther.* The supervisor's fallback was defective (specified
    without checking `MC_DEFAULT_SAMPLES = 32`) and the measurement retires it: mc@32's own noise
    floor (agreement with mc@256 = **0.8164**) EXCEEDS the disagreement it was meant to remove
    (**0.9102**), at a **32×** rollout cost — the branch degrades the property it protects. Measured
    CPU cost 0.3402 s/decision at n=32 (2.9228 at n=256); GPU PENDING. Pinned money surface for
    scale: 35,064 grid periods × 40 symbols = **1,402,560** headline decisions per (cell, seed),
    uncapped by pin, ×15 (cell, seed) pairs.
    - **M4a — THE SPLIT-HALF FLOOR, AND IT REFRAMES THE ITEM (`m6_estimator_forensics.json`).**
      M4a as specified was impossible: `_rollout_mc_mean` accumulates and returns only the mean, so
      the 256 per-decision samples were **never materialized** and cannot be partitioned after the
      fact (retaining them would mean editing the anchored `predict.py`). Instead the SAME 256-sample
      budget was spent as **two independent n=128 halves** at different `mc_seed`s — giving the
      split-half floor **directly, measured rather than 1/√n-extrapolated**, at no extra rollout cost
      over the ordered M4b. Result: **agree(half_a, half_b) = 0.8906** and
      **agree(expectation, combined n=256) = 0.8906**.
      - **M4a CORRECTION (v1.4.7, supervisor-issued; BOTH earlier readings WITHDRAWN).** The two
        quantities have **different error budgets**, so their equality is *not* the null expectation
        and "indistinguishable" was wrong. `agree(half_a, half_b)` compares two independent n=128
        estimates → combined scale √2·σ/√128 = **σ/8 = 0.1250σ**. `agree(expectation, combined)`
        compares expectation to ONE n=256 estimate → scale √(s_exp² + (σ/16)²). Equal disagreement
        therefore *implies* s_exp² = σ²/64 − σ²/256 = 3σ²/256, i.e. **s_exp ≈ 0.1083σ** against the
        reference's own **0.0625σ**. So expectation's deviation is **REAL and ≈1.73× the reference
        noise** — NOT indistinguishable from it — and only **25%** of that squared error budget is
        reference variance. **Withdrawn: the builder's "statistically indistinguishable from
        reference variance"; equally withdrawn: the ruling's "most of the gap is REFERENCE
        VARIANCE".** Both were too strong in the same direction. The builder's instinct that the
        228/256 tie is coincidence was right and is now stronger than stated: under this model the
        tie should **not** be expected even in principle.
      - **THE DECISION-RELEVANT STATEMENT (same arithmetic, and it is what carries item 4).**
        mc@32's noise scale is σ/√32 = **0.1768σ**, so **expectation's error is ≈0.61× the PINNED
        estimator's, at 1/32 the cost.** That, not "indistinguishable", is why expectation stands.
      - **SELF-CONSISTENCY across FOUR independent measurements under ONE error-scale model**
        (verified numerically): D(exp,mc256) 0.1094 @ 0.1250σ · D(half_a,half_b) 0.1094 @ 0.1250σ ·
        D(mc32,mc256) 0.1836 @ 0.1875σ · D(exp,mc32) 0.1875 @ 0.2073σ — **monotone**, same rank
        order in observed and predicted. Four measurements, one model, no free parameters after
        s_exp is fixed by the first pair.
    - **M4b — THE PERTURBATION IS BENIGN, MEASURED NOT ASSERTED.** The premise holds: |μ̂| on the
      disagreement set is **0.00859** vs **0.02790** on the agreement set — **3.25× smaller**, i.e.
      the flips sit exactly where the forecast is least informative. Of 256 traded decisions, 28
      disagree. On those 28: realized-return `frac_positive` is **exactly 0.500**, median −0.0071,
      mean −0.0288 at SE 0.0237 (**1.22σ** — not distinguishable from zero); and
      **corr(flip direction, realized return) = −0.0935** with |t| ≈ **0.47** — uncorrelated.
      Decisively, the mean signed return on the disagreement bars is **+0.000665** under
      expectation's positions and **−0.000665** under mc's, versus **+0.029664** on the agreement
      set — a **44.6×** magnitude gap. Those bars earn essentially nothing either way, so the
      induced position noise **dilutes measured IR toward zero, biasing toward NULL — the
      conservative direction — and cannot manufacture a false SURVIVES.** Symmetric and
      uncorrelated: **the argument HOLDS and enters the prereg as MEASURED.**
    - **Caveats (unchanged and binding):** n=256 traded decisions on ONE fixture-planted cell; point
      estimates, **not** significance claims; no paired CI computed on any difference; the σ and |t|
      figures above are derived from the receipt's own mean/std/n fields, not from a formal test. The
      per-decision vectors ARE now persisted (`m6_estimator_forensics_vectors.npz`) so any further
      question about them costs no rollouts.

- **v1.4.6 (2026-07-30, post-v1.4.5 ruling: leg (c) UPGRADED, guard hole closed, divergence
  attributed; supervisor-ordered) — BEFORE any real training. Local $0 CPU only; no spend
  authorized, none taken.** Legs (a), (b) and (c) are now all demonstrated and the canary is a
  **CONDITIONAL PASS — explicitly NOT 8/8**, and does not become 8/8 until items 1 and 2 below are
  accepted.
  - **LEG (c) UPGRADES — recorded at exactly this strength and no more. This sentence is the
    permitted claim; nothing stronger may be derived from it:**
    > *"A trained model on planted data converts microstructure into net IR through the money path,
    > demonstrated at the pinned h=15 with a clean negative control, on ONE checkpoint whose
    > byte-identical sibling is degenerate."*

    Control arm verified independently by the supervisor: **0 of 16** noise cell-horizon combinations
    pass the conjunction, and `moneyleg_noise_cell2` @ h=5 (activity 0.9318, binding) is correctly
    rejected by the sign-balance leg. Legs (a), (b) and (c) are now all demonstrated; **the canary is
    a CONDITIONAL PASS, explicitly NOT 8/8**, and does not become 8/8 until the v1.4.6 items 1 and 2
    are accepted. **v1.4.7 note:** the "byte-identical sibling is degenerate" clause is now
    ATTRIBUTED — see v1.4.7, the divergence is unforced-determinism, not hardware — and the
    single-checkpoint weakness in this sentence is exactly what the v1.4.7 power guard exists to
    catch on real data.
  - **ITEM 1 — THE GUARD WAS DEFEATED BY THE THING IT NOW HAS TO CATCH (fixed).** v1.4.4's
    `degeneracy_guard()` tested only the seed-MEAN and **discarded the per-seed lists it had already
    computed**. At the measured 1-in-2 lock rate a realistic triple is (0.999, 0.55, 0.55): mean
    **0.700**, comfortably inside [0.05, 0.95], **no HALT** — while the locked seed's
    constant-direction book still enters the seed-mean headline series the §3 clauses are computed
    on. Averaging is the wrong reduction for a degeneracy test: one poisoned seed is not diluted by
    two healthy ones, it contaminates the aggregate the verdict reads. **FIX:** HALT if **ANY** seed
    is degenerate, as a **union** with the existing seed-mean condition, on both the sign and the
    activity leg; per-seed `frac_negative` and `activity_decisions` are now **persisted** in the
    verdict manifest (`frac_negative_by_seed`, `activity_decisions_by_seed`,
    `degenerate_seeds_frac_negative`, `degenerate_seeds_activity`) so an adjudication has the
    distribution rather than an aggregate. **HALT-only semantics unchanged and non-negotiable** — the
    guard is still a pure post-hoc read that can never flip SURVIVES↔NULL. KATs: the exact
    (0.999, 0.55, 0.55) pattern proving **the old rule passes it and the new rule halts** (asserting
    both halves), the same for one non-binding seed, and a no-false-positive case with three
    differing in-band seeds. `tests/eval/test_degeneracy_guard.py` now 8 KATs.
  - **ITEM 2 — THE DIVERGENCE, ATTRIBUTED FROM CODE (`m6_divergence_attribution.json`).** The
    pre-declared reading could not be applied as written, and I say so plainly rather than inferring:
    - **The GPU comparison CANNOT be settled from the receipts.** Acceptance: RTX 4090, torch
      2.12.1+cu130, numpy 2.4.6 (from `runs_cloud/acceptance/setup.log`); driver/CUDA version and
      attention mode absent. Money-leg: **no setup log exists at all** — nothing recorded. wandb
      covers only the 2026-07-19 toy rehearsal (and records `gpu: None`). Checkpoints carry only
      `{format, config, hash, state_dict}`. Both Stage-2 manifests record `device` and `seed` and
      nothing else. So branch A/B cannot be adjudicated on hardware.
    - **But the question is answered from CODE, not logs.** **Forced determinism is NEVER engaged**:
      `orchestrator.py:159` — the REAL 15-day run path — and `m6_canary.py:482,557` both call
      `set_determinism(..., deterministic_algorithms=False)`, leaving
      `torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG` and `cudnn.deterministic`
      unset. CUDA reduction order is unconstrained, so **run-to-run divergence is EXPECTED even on
      identical hardware** — the GPU identity is not needed to attribute it.
    - **This lands on branch B's DISPOSITION by a different mechanism than branch B assumed.** SDPA
      attention was never the weak link; the missing `use_deterministic_algorithms` is. It is a
      determinism-deliverable defect and **BLOCKING for the 15-day run**.
    - **PROVEN, not argued:** **15 of 15** committed toy-rehearsal run records stamp
      `bit_exact_claim: true` beside `deterministic_algorithms: false` on `device: cuda`. The
      recorded bit-exactness claim is unsound as written.
    - **NOT CHANGED BY THE BUILDER.** The determinism posture stands: forcing deterministic
      algorithms slows CUDA training and the 2–3 GPU-day / \$20–30 budget was measured **without**
      it, so the throughput/cost trade is a supervisor decision. `bit_exact_claim` was likewise NOT
      redefined (that would silently restate a recorded claim and break its KAT).
  - **ITEM 3 — RUN PROVENANCE, where it was absent and matters most.** `determinism_record()` —
    the single point feeding **both** the cloud `run_manifest.json` and the checkpoint metadata — now
    records GPU model / capability / count / total memory, driver version, CUDA build, cuDNN version,
    torch / numpy / python, attention mode, **all** seeds (`seeds`, via an `extra_seeds` passthrough),
    and **the determinism flags actually in force** (`torch_use_deterministic_algorithms`,
    `cudnn_deterministic`, `cudnn_benchmark`, `cublas_workspace_config`). Every probe is guarded —
    provenance must never be the thing that crashes a 15-day run. It also adds
    **`bit_exact_preconditions_met`** (SDPA **and** forced algorithms **and** cuDNN determinism **and**
    autotune off) plus a `bit_exact_caveat` string whenever that disagrees with the historical
    `bit_exact_claim`, so the contradiction is **self-announcing** instead of silently wrong. KATs in
    `tests/model/test_attention.py` assert the provenance fields and both directions of the
    precondition predicate (flags set explicitly and restored, so the assertion is not
    order-dependent on the shared global torch state).

- **v1.4.5 (2026-07-29, C1/C2 corrections + the FINAL canary fixture iteration; supervisor-ordered)
  — BEFORE any real training. Local $0 CPU only; no spend authorized, none taken.** The supervisor
  confirmed gate legs (a) and (b), restated (c) at full strength, and ordered ONE pass: two wording/
  epistemic corrections, a lattice observation to verify independently, an approved horizon sweep, a
  BLOCKING band-stability check, and regeneration of every diagnostic receipt on the pinned `.venv`.
  This is **pre-declared as the LAST fixture iteration** — there is no seventh.
  - **C1 — MEASURED, and it overturns the premise (including the builder's first attempt at this
    correction).** The ordered correction was to withdraw a "bit-exact |Δ| = 0.00e+00" claim as a
    6-decimal rounding artifact whose true value was inferred to be ≈4.9e-7. **That inference is
    wrong, and the original reported value was right.** The v1.4.4 receipt's *display* field was
    rounded (`-7.240329`), which made the reported zero *look* like it could be an artifact.
    Whether that receipt computed `L1_abs_diff` from rounded or full-precision values can no longer
    be established — the ad-hoc script was not saved, itself a lesson now fixed by making the
    rescore a committed, re-runnable mode — but it does not matter: the true |Δ| IS 0.0, so the
    reported value was correct either way. Regenerated at full float64 on the pinned `.venv`
    (`m6_moneyleg_local_rescore.json`): local == cloud **EXACTLY** for all three cells —
    cell1 `-7.240328510778865`, cells 2,3 `-7.563377128272953`, worst |Δ| = **0.0**, not 4.9e-7.
    The inferred discrepancy does not exist.
    - **WHERE THE INFERENCE ORIGINATED (recorded at the supervisor's instruction, v1.4.6).** The
      ≈4.9e-7 figure was inferred **in the v1.4.5 ruling itself**, from a rounded DISPLAY field, and
      issued as a correction without being measured — the same failure this project's rules exist to
      prevent, committed first at the ruling stage and then propagated into this prereg by the
      builder. Both halves are on the record; neither is softened on the other's behalf.
    - **BUILDER ERROR, DISCLOSED.** The C1 correction was written into this document **before** the
      receipt was regenerated — acting on the ruling's inference instead of on a measurement, which
      is precisely what "claims must match receipts" forbids. Receiving a wrong premise does not
      excuse propagating it unmeasured; the builder's job was to measure first and did not. That
      text is withdrawn and replaced by the measured result above. The annotations it introduced
      into the v1.4.4 entry are corrected in place. The term-of-art discipline still stands on its
      own merits: **"bit-identical" is what Gate-A means** and is not spent loosely — here it
      happens to be literally true, for the reason C2 gives.
  - **C2 — L1's epistemic status REFRAMED, and the C1 measurement STRENGTHENS it.** The agreement
    is **exact, not merely small**, and that is the point: with traded == scored at every κ the book
    is CONSTANT, so μ̂'s *magnitudes* drop out of the IR entirely and the computation reduces to the
    same return series under the same fixed positions. The arithmetic never touches the numbers that
    differ between environments, which is why |Δ| is 0.0 across CUDA/numpy 2.4.6, CPU/numpy 2.3.3
    and CPU/numpy 2.4.6. On a NON-degenerate cell those same differences would flip
    marginal bars and reproduction would be looser. **L1 is therefore a SYMPTOM of the degeneracy,
    not evidence of instrument health.** It validates the SIGN PATTERN and the scoring arithmetic —
    which is all L2/L4 needed, so their conclusions stand unchanged — and nothing more.
  - **CONSTANT-BOOK LATTICE — recorded as an OBSERVATION (`m6_constant_book_lattice.json`), n=2
    runs, no error bars, NOT a hypothesis test.** Verified independently from the two Stage-2
    manifests (recipes confirmed byte-identical). Across 20 (run, arm, cell) observations there are
    **5 measured constant-book points**; a constant-direction book makes the pooled net series a
    function of the RETURN BLOCK and the direction alone, so its IR is model-independent. This is
    **measured, not inferred**: the three money-leg noise cells that are all-long span BOTH
    quantizers and BOTH feature arms (cell2 FSQ+ohlcv, cell3 BSQ+micro, cell4 FSQ+micro) and return
    the IDENTICAL float64 −7.563377128272953. Planted: all-short −3.638854395025198, all-long
    −1.050127274973947, flat (never trades) 0.0. **Off-lattice = 4 of 20**, ALL in the planted arm
    and ALL micro-fed: acceptance planted cells 3, 4, 5 and money-leg planted cell4. **0 of 8
    OHLCV-only observations and 0 of 10 noise-arm observations are off-lattice.**
    - **CORRECTION to the posted enumeration (verified before acting on it).** The ruling listed
      "cell4 in both runs, acceptance cell3 once". **Acceptance planted cell5 is ALSO off-lattice** —
      its −1.0308263783219347 is NOT the planted all-long constant −1.050127274973947. Conversely
      money-leg planted cells 1 and 2 are NOT off-lattice: cell1 is the flat never-trades point and
      cell2 IS the all-long point (a singleton value is not an off-lattice value). Cell4 remains the
      only cell off-lattice in BOTH runs.
    - **What it does NOT support.** Not evidence for the microstructure claim. Cell 5 is the
      SHUFFLED-micro placebo and is off-lattice, so the pattern is equally consistent with extra
      input CAPACITY widening μ̂ dispersion as with micro INFORMATION. "Off-lattice" also means
      only "not perfectly constant": the one off-lattice cell with a measured direction is
      money-leg planted cell4 at frac_negative 0.99883 — **14 long decisions out of 12,000**.
  - **H-SWEEP — the approved horizon diagnostic (`m6_h_sweep.json`).** 5 checkpoints × h ∈
    {1,2,3,5,15}, 12,000 decisions each (cap 4,000/symbol × 3), CPU, expectation estimator,
    42,000,000 bars total. **h=1 is STRUCTURALLY VOID and is not a measurement**: μ̂ covers
    [t+1, t+h] = log(C_{t+h}/C_{t+1}), so `predict.py` excludes rollout step k=1 (entry-next-bar,
    spec §8.B.3) and μ̂ ≡ 0 identically for **every** model — confirmed on all five checkpoints.
    Its all-zero row must never be read as "the filter did not bind".
    - **PRE-DECLARED TEST → UPGRADE. All three legs hold on the acceptance PLANTED cell4 at
      h ∈ {2,3,5,15}:** IR@κ* +65.098 / +43.461 / +28.878 / **+15.008**; activity@κ* 0.9152 /
      0.9127 / 0.8862 / 0.8874 (all strictly inside (0,1)); frac_negative 0.4987 / 0.5198 / 0.6100 /
      0.5662 (all inside [0.05, 0.95]).
    - **MECHANISM CONFIRMED for the planted arm.** μ̂ std is flat across h (0.0267 → 0.0293) while
      the mean drifts −0.00018 → −0.00576, so dispersion/|mean| collapses 146 → ~5. The offset
      accumulates with h; the state-driven spread does not.
    - **NOISE ARM QUIET AT EVERY h** on all four noise checkpoints — IR ≤ 0 throughout,
      frac_negative pinned at 0.0 or 1.0, μ̂ std ~1e-5..5e-4. So the planted cell's binding activity
      is NOT a generic short-horizon artifact of the backtest. At h=2 the two arms differ by ~2,700×
      in μ̂ dispersion (0.02673 vs 0.00001).
    - **A SECOND, DISTINCT DEGENERACY MECHANISM, separated by measurement.** Noise cells obey
      μ̂ = (h−1)·c with std scaling identically (exact: 0.01730 / 0.03459 / 0.06918 / 0.24216 at
      h = 2/3/5/15) — the model emits a fixed, context-independent per-step value. Their books are
      constant **by absence of signal, not by drift**, so no horizon rescues them; lowering h merely
      flips them from all-trade (activity 1) to no-trade (activity 0) without passing through a
      discriminating regime. This must NOT be filed under the same mechanism as the planted arm's
      h=15 sign-lock.
    - **THE v1.4.4 CONJUNCTION FLOOR IS VALIDATED BY A COUNTEREXAMPLE.** `moneyleg_noise_cell2` at
      h=5 has decision-activity **0.9318** — strictly inside (0,1) — with μ̂ std 0.00016 and NO
      signal: θ merely happened to land inside a very narrow μ̂ spread. **Partial activity is
      necessary, not sufficient.** That cell is caught because it fails the dispersion leg
      (0.00016 < the 0.005 floor) AND the sign-balance leg (frac_negative 0.0). A filter-only guard
      would have passed it. Each leg covers a case the others miss — now an instance, not an
      argument.
    - **INDEPENDENT REPRODUCTION OF THE LATTICE.** At h=15 the noise checkpoints land exactly on
      the constant-book points: −7.563377128272953 (all-long) and −7.240328510778865 (all-short),
      reproduced here by a different estimator, a different device and an independent code path
      from the CUDA runs that established them.
    - **THE FINDING THAT NEEDS ADJUDICATION — RUN-TO-RUN MODEL DIVERGENCE.** The acceptance cell4
      is **not degenerate at h=15** (activity 0.8874, frac_negative 0.5662, IR +15.008 against the
      oracle's 17.12 — ~88% of the analytic ceiling), while the money-leg cell4 at the same h, the
      same cap and the same estimator gives frac_negative 0.99883, activity 1.0, IR −3.4258.
      Recipes are byte-identical and the seed is the same. The sign-lock is therefore **run-to-run
      model divergence, not a property of h=15 and not grid/cap dependence** (the latter is
      excluded by the band-stability check below). Lowering h helps by shrinking a drift offset
      whose magnitude is itself unstable across runs (~15× different here).
      **CAVEAT, stated so the upgrade is not over-read:** the demonstration rests on ONE checkpoint,
      is contradicted by its sibling, and comes from a re-decode with the corrected estimator rather
      than from a run whose verdict path executed end-to-end (the acceptance run's own recorded IRs
      are pre-v1.4.2 **argmax**, which is why its manifest reads −3.4709 for this cell and only the
      money-leg comparison is apples-to-apples). Leg (c) upgrades **per the letter of the
      pre-declared rule**; whether a single-checkpoint demonstration contradicted by its sibling is
      sufficient to close the leg is the supervisor's call, not the builder's.
  - **BLOCKING BAND-STABILITY CHECK — PASSED (`m6_guard_band_stability.json`).** Acceptance planted
    cell4 re-decoded at h=15 across three decision-set sizes: n=4,000 → frac_negative 0.5557
    (drift 0.0028); n=12,000 → 0.5662 (drift **0.0077**, the worst); n=**140,013** → 0.5638
    (drift 0.0053). Against the pre-declared 0.15 limit that is inside by a factor of ~20, across a
    35× range of set sizes. **The [0.05, 0.95] band IS grid-stable; no STOP is triggered and the
    band is NOT adjusted.** This is what excludes grid/cap dependence as the explanation for the
    acceptance-vs-money-leg gap, leaving model divergence.
    - **PREMISE CORRECTED.** The ruling described the money-leg's n=12,000 set as UNCAPPED. It is
      not: 12,000 = 3 symbols × the 4,000 per-symbol cap, and the genuinely uncapped block-3 grid is
      46,671/symbol = **140,013**. Both are scored here, so cap dependence is settled by measurement
      rather than by the label — had only the set labelled "uncapped" been run, cap dependence would
      have been tested over no range at all.
    - **CAPPING DISTORTS FIXTURE IR ~3×, but not the guard statistic.** The same cell reads +15.008
      capped and +52.067 uncapped: the headline series is a full-calendar grid with untraded periods
      entered as 0.0, so capping to 4,000 of 46,671 periods leaves ~91% of the series flat and
      inflates the IR denominator. `frac_negative` and decision-activity are ratios over SCORED
      decisions and are unaffected — which is why the guard reads stable while IR moves 3.5×. A
      fixture-reporting caveat only: the real §3a money surface is uncapped by pin.
  - **RECEIPT REGENERATION ON THE PINNED `.venv`** (numpy 2.4.6 / torch 2.12.1), closing the v1.4.4
    environment disclosure. `m6_moneyleg_local_rescore.json` (L1/L2/L4 at full float64 — L2/L4
    reproduce exactly: traded == scored == 12,000 at every κ at both sigmas, `binds=False`);
    `m6_oracle_calibration.json`; `m6_estimator_bias_bound.json`. The L3 statistic is reproduced
    inside the band-stability check (n=4,000 → 0.5557 vs L3's 0.5585). **A STALE FIELD WAS FOUND AND
    FIXED:** the oracle receipt still recorded `pinned_sigma: 0.05` after the v1.4.4 revert; it now
    reads 0.01, with `pinned_equals_baseline_after_v1_4_4_revert: true` making the (deliberately
    retained) 0.05-vs-0.01 comparison legible as the receipt for the WITHDRAWN recalibration. Oracle
    unchanged on the pinned env: net IR 17.1221 = **6.286× MDE**, activity 0.216, Δ vs placebo
    20.7686 (CI lower 19.0507), margin met.
    - **THE BIAS-BOUND REGENERATION WITHDRAWS A v1.4.3 CONCLUSION.** The order read "at n ≥ 256";
      v1.4.3 used mc_samples=256 but **n_dec = 24**, and the decision count was the leg the
      supervisor challenged. Re-run at n_dec = 256 AND mc_samples = 256 (both readings satisfied).
      The level bias is confirmed **common-mode, and far more tightly than before**: gap means
      +0.00498 and +0.00477, spread **0.000204** (v1.4.3 reported 0.00965 at n=24). **But the
      inference drawn from that is wrong.** Sign agreement between expectation and mc_mean is
      **0.8398** on the non-degenerate acceptance cell4 (|mean|/std = 0.150) versus **0.9883** on
      the degenerate money-leg noise cell2 (|mean|/std = 27.4); frac_negative differs 0.5430 vs
      0.6406, not identically as at n=24. A common-mode level shift of ≈+0.0049 preserves positions
      ONLY when |mean| ≫ std. On a cell centred near zero — which is what a healthy, discriminating
      cell looks like — **it flips ~16% of signs.** v1.4.3's "level shift ⇒ positions preserved ⇒
      paired ΔIR robust" therefore holds for the DEGENERATE cells and FAILS for exactly the cells
      the verdict depends on; the n_dec=24 equality was a small-sample artifact. That wording is
      withdrawn in the v1.4.3 entry.
    - **CONSEQUENCE — PROPOSED, NOT EXECUTED.** Expectation is biased ≈ +0.0049 against the
      unbiased mc_mean, ~17% of the planted cell's μ̂ std (0.029), and it moves ~1 position in 6 on
      a non-degenerate cell. `predict_mu`'s default is a v1.4.2 PIN inside the Gate-A anchored
      instrument, so the builder changes nothing. Flagged for the supervisor: the estimator choice
      is not free on the cells that matter, and if the real run's cells are non-degenerate (which
      the guard now requires) the expectation-vs-mc_mean choice is position-relevant. No change is
      made and none is implied.
    - **PROVENANCE GAP, DISCLOSED.** `m6_h_sweep.json` and `m6_guard_band_stability.json` were
      written BEFORE the `bars_total`/`environment` stamping was added, so they carry `device` and
      `eval_cap_per_symbol` but not the stream size or interpreter versions. Both ran on the pinned
      `.venv` at the committed default `--bars-total 42_000_000` (14,000,000 bars/symbol) with no
      CLI overrides — evidenced by the run log and the script's own default. A new `_env()` helper
      now stamps numpy/torch/device into every receipt this script writes (the v1.4.4 environment
      slip was caught only by chance from an unrelated manifest diff; this makes it
      self-announcing). **The two artifacts are NOT patched after the fact** — editing a receipt
      post-hoc is exactly what receipts exist to prevent, and a ~2.5 CPU-hour re-run to add a
      provenance string is not proportionate.
  - **TOOLING (`scripts/m6_h_sweep.py`, `scripts/m6_lattice_observation.py`).** The sweep needs the
    token stream hoisted out of the h-loop (25 scorings; three 14M-bar feature matrices do not fit
    in this box's RAM), so `score_pretokenized` is a line-for-line mirror of `xsection.score_cell`
    that imports the pooling/cost/position/netting/IR arithmetic from the instrument modules. **The
    mirror is PROVEN, not asserted:** `tests/eval/test_h_sweep_mirror.py` runs both on one fixture
    and requires bit-equality of κ*, headline IR, the VAL per-κ curve, the decision count and every
    μ̂ diagnostic the guard reads. Tokenization is restricted to the aligned chunks containing a
    needed bar — exact because `tokenize_features` encodes NON-OVERLAPPING windows, so a bar's
    token depends on exactly one chunk — and that equality is its own KAT. Decisions occupy only
    eval blocks 0 and 3, so this touches 10.0 % of the stream (1,400,174 / 14,000,000 bars per
    symbol) and removed a swap-thrash that had made the full-stream version unusable on 16 GB.
    **No file in the Gate-A anchored path is touched by this amendment.**
  - **SCOPE GUARDRAIL.** Everything in v1.4.5 is CANARY-ONLY. The real run's h=15 and the entire
    pinned §3a money surface are NOT in scope and are unchanged. If a fixture result argues for
    moving them, that is a PROPOSAL for the supervisor, never an executed change.

- **v1.4.4 (2026-07-29, money-leg sign-lock adjudication; supervisor-ordered) — BEFORE any real
  training. Spend WITHDRAWN; local $0 CPU work (L1–L4) + the degeneracy guard.** The staged $0.60
  re-run's authorization was withdrawn: the money-leg manifest carries a stronger fact than the
  v1.4.3 report drew out. VERIFIED locally from `runs_manifest/m6_moneyleg_rerun_manifest.json`:
  the noise arm has exactly TWO distinct IRs partitioned PERFECTLY by frac_negative (cells 2,3,4
  all-long share −7.563377128272953; cells 1,5 all-short share −7.240328510778865); planted cell3
  and cell5 share −3.638854395025198 to 15 dp despite μ̂ means 8.48× and stds 237× apart (both
  all-short); planted cell1 IR is exactly 0.0 (the single sub-threshold cell); cell4 frac_neg
  0.99883 = 14/12000 long. Two cells produce a bit-identical net series ONLY if they trade the
  SAME bars with the SAME signs — so EVERY bar trades, the θ=κ·c forecast-magnitude filter NEVER
  BINDS, and the money-leg IR is a pure function of sign(μ̂). κ is an all-or-nothing switch, not a
  per-bar conviction filter.
  - **L1–L4 (`runs_manifest/m6_moneyleg_local_rescore.json`, CPU, full 14M, saved checkpoints).**
    L2/L4: decision-activity (traded/scored) at EVERY κ ∈ {1,1.5,2,3} = **1.0** for noise cells
    1,2,3 at BOTH SIGMA=0.01 and 0.05 — the filter never binds, and raising SIGMA does NOT put
    activity inside (0,1) (it moves the fixture FURTHER from the binding regime — larger μ̂ vs the
    fixed threshold → more saturated; at 0.05 the noise |IR| merely shrinks, −7.24→−1.32 and
    −7.56→−1.65, activity still 1.0). L1 (the fidelity precondition): reproduced ALL three cloud
    IRs EXACTLY, |Δ| = 0.0, inside the declared 1e-3 criterion: cell1 −7.240329, cells 2,3
    −7.563377. **[v1.4.5 NOTE: these display values are rounded to 6dp, which made the reported
    zero LOOK like a rounding artifact; re-measured at full float64 on the pinned `.venv` the
    agreement is exact (cell1 −7.240328510778865, cells 2,3 −7.563377128272953). But see v1.4.5
    C2: the exactness is a CONSEQUENCE of the constant book — μ̂'s magnitudes never enter the IR —
    and is NOT evidence of instrument health.]**
  - **SIGMA RECALIBRATION WITHDRAWN — a documented NON-FIX.** Per the pre-declared
    decision rule (L2 activity 0/1 everywhere → revert), SIGMA is reverted 0.05 → 0.01: the v1.4.3
    recalibration reduced fixed-cost DRAG but did NOT address the actual pathology (the filter
    never binds), and raising SIGMA moved the fixture away from the filter-binding regime. Recorded
    as a MEASURED NEGATIVE RESULT, not deleted. (The oracle margin still holds at 0.01: net IR
    17.12 = 6.29× MDE, clears the pre-declared 5×; `m6_oracle_calibration.json` retained.)
  - **DEGENERACY GUARD (HALT-only), armed on the real run (mitigation built now).** Every
    `CellScore`/eval artifact now records per-cell `frac_negative` and decision-activity at κ*
    (`activity_decisions`, traded/scored — the FILTER-BINDING ratio, distinct from the
    cap-diluted grid-activity). `verdict.degeneracy_guard`: a cell whose seed-mean frac_negative
    is outside **[0.05, 0.95]** (sign-locked) OR whose seed-mean decision-activity at κ* is
    exactly 0 or 1 (filter never binds) is a constant-direction book; any clause involving it is
    uninterpretable, so the verdict is **HALT_ADJUDICATE**. The guard is HALT-ONLY: it is computed
    POST-HOC, never mutates the clauses or the clause-derived `primary`, and can NEVER flip
    SURVIVES↔NULL — it can only refuse to emit a verdict. That asymmetry is what makes it
    legitimate to add pre-run (it can manufacture neither a positive nor a negative claim). KATs:
    HALT on each boundary leg, no-HALT just inside both, a proof that the guard cannot alter a
    clause outcome (byte-identical clauses/primary with vs without a degenerate cell), and NULL-
    side symmetry (`tests/eval/test_degeneracy_guard.py`). The canary's std-only non-degeneracy
    floor is replaced by the CONJUNCTION dispersion ∧ sign-balance ∧ binding-filter (the std leg
    is KEPT — it correctly flags the near-constant cells 2,5; it MISSED the sign-lock, a
    99.88%-one-sided book that clears std ≥ 0.5·SIGMA). GATE-A: xsection.py (CellScore) and
    verdict.py are edited; the M5 anchor replay uses `harness.single_symbol_backtest`, not
    score_cell/verdict, so 3f86882a is RE-PROVEN bit-identical (results_hash 3f86882a, Gate-A
    420/420 PASS, ×2 confirming runs) — the edit is anchor-neutral, fail-closed rule satisfied.
  - **ENVIRONMENT DISCLOSURE (builder error, on the record).** This session's local receipts
    (v1.4.4 L1–L4/L3 and the v1.4.3 oracle/bias-bound/throughput) were computed on the SYSTEM
    interpreter (numpy 2.3.3 / torch 2.11.0), NOT the pinned uv-locked `.venv` (numpy 2.4.6 /
    torch 2.12.1). The LOAD-BEARING gates are re-verified on the pinned `.venv`: the Gate-A anchor
    re-proves 3f86882a BIT-IDENTICALLY on BOTH the system env and the pinned `.venv`, and the full
    suite is re-run green on the pinned `.venv`. Independently, L1 reproduced the pinned-env cloud
    IRs (the cloud box carried the lockfile) EXACTLY across the env gap — the sign-lock /
    filter-never-binds findings are structural (signs and all-trade), not ULP-sensitive.
    **[v1.4.5 C2 REFRAMES WHAT THIS SHOWS: that agreement is a SYMPTOM of the degeneracy — a
    constant book makes IR independent of μ̂'s magnitudes — not evidence of instrument health.]** All diagnostic receipts are
    REGENERATED on the pinned `.venv` in v1.4.5, so the "every receipt on the pinned env"
    invariant holds without relying on this label.
  - **ITEM-1 CORRECTION (on the record).** The v1.4.3 bias-bound's "acceptance cell4 gives balanced
    frac_neg=0.5 → the money-leg cell4 sign-lock is a MODEL property not an estimator artifact"
    OVERREACHED: that result is n_dec=24 on a FIXTURE-PLANTED cell; it supports ONLY that
    expectation and mc_mean agree on the same cell, and is NOT evidence that the AR avoids
    sign-lock — we have NO real-data evidence either way. Withdrawn in the receipt. L3
    (`m6_L3_acceptance_cell4_fullset.json`) re-decodes that cell over the FULL decision set (n=4000):
    expectation frac_negative 0.5585 (balanced, NOT the near-0.999 sign-lock) with mc_mean 0.6016,
    so the n=24 balance was not a small-sample artifact — but this still only shows the two
    estimators agree on a FIXTURE-PLANTED cell, never that the AR avoids sign-lock on real data.
  - **ARTIFACT-RETENTION FAILURE + standing rule (`m6_artifact_retention_disclosure.json`).** The
    $0.60 run's PRIMARY artifacts did not survive teardown: moneyleg planted cells 1–5 and noise
    cells 4,5 are EMPTY; only noise cells 1,2,3 retain checkpoints (verified). Summary diagnostics
    survived but not the per-cell μ̂ SERIES. STANDING RULE: no box is destroyed until, for every
    cell, the per-cell μ̂ SERIES + checkpoints + manifest are pulled AND sha256-verified locally;
    for the real 15-day run this is a HARD PRE-TEARDOWN GATE, fail-closed.
  - **GATE DISPOSITION (stated; the verdict-logic change is NOT implemented — awaiting the
    supervisor's read of L1–L4).** Item 2 closes as a CONDITIONAL PASS with components stated
    separately, never a clean green: **(a)** the tokenizer→AR interface transmits microstructure —
    PASSED on-box (legibility 0.8990 enforced, tf_corr 0.9407, Δval −1.3381, noise quiet); **(b)**
    the backtest layer converts a genuine per-bar edge into placebo-separated net IR under 0.30 %
    costs — PASSED by the ORACLE receipt, the right receipt for this leg (activity 0.216 means the
    filter genuinely binds; net IR 17.12 at 6.29× MDE; ΔIR vs placebo 20.77, CI lower 19.05,
    through the identical grid_series/positions/net_trade_returns/information_ratio path); **(c)**
    the trained-model → backtest handoff on PLANTED data — **NOT DEMONSTRATED**: the fixture cells
    are sign-degenerate, so we would enter the real run never having watched a trained model turn
    planted microstructure into net IR. This is a NAMED RESIDUAL, not a pass; an uncaught pathology
    here fails toward NULL (conservative), never toward a false SURVIVES — the degeneracy guard is
    the pre-run insurance for it.
  - **CANARY-ONLY PROPOSAL (not executed).** Re-scoring the canary at the plant's own lag-2 horizon
    (h=2) would concentrate the signal into one forward bar instead of diluting it across the h=15
    window, so μ̂ magnitudes would vary enough for the θ=κ·c filter to BIND — the missing (c)-leg
    demonstration on planted data. Proposed as a CANARY-ONLY fixture diagnostic; NOT executed; the
    real run's h=15 and the entire §3a surface are untouched and out of scope.
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
     level bias without flipping signs.
     **[FINDING (a) IS WITHDRAWN BY THE §7 v1.4.5 REGENERATION — see v1.4.5. At n_dec=256 the
     identical-frac_neg result does NOT survive: sign agreement is 0.8398 on the non-degenerate
     cell. The "level shift ⇒ positions preserved ⇒ paired ΔIR robust" inference holds only for
     DEGENERATE cells and FAILS for exactly the cells the verdict depends on. The n_dec=24 equality
     was a small-sample artifact.]** (b) DECISIVE: on the ACCEPTANCE cell4 the expectation
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
