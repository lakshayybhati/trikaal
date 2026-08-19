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
     headline series (0.30 % netting, seed-mean), SR₀ = expected-max-Sharpe under **N = 60
     enumerated trials** (5 cells × 3 horizons {5,15,60} × 4 κ — **seeds are REPLICATES, not
     configurations, and are excluded from the multiplicity count** (§7 v1.5 A.5) — **h=1 is not evaluated
     anywhere in M6**; the earlier "4 horizons / 240" budget text is corrected here), var_sr =
     the **ddof=0 variance of the CELL-5 (placebo) per-trial de-annualized IRs** — 5 seeds × 3
     horizons × 4 κ = **60 values**, seeds retained inside the basis (§7 v1.5 A.4). The full
     5 × 5 × 3 × 4 = **300**-entry cross product is still persisted as the audit trail, so the
     trial set is enumerable, not assumed. Higher moments come from the same pooled headline
     series.
     *(§7 v1.6.13, disclosed: this sentence read "the variance of the 180 recorded per-trial
     de-annualized IRs (every cell × seed × horizon × κ …)" — the superseded all-arms basis at
     the superseded 3-seed count — until the ruling-(a) sweep. The C-9 fix corrected `N = 180 →
     60` two lines above and stopped at the clause boundary it had been pointed at. **Both halves
     of one sentence, one edit apart.** The var_sr basis has been the cell-5 placebo in code since
     v1.5 (`DSR_VAR_SR_BASIS_CELL`, mutation-KAT'd); only this description was stale.)*
     *(§7 v1.6.13, AUDIT C-3, REPORTED NOT FIXED: "de-annualized" divides each trial by
     `sqrt(periods_per_year(h))`, so the basis mixes per-5, per-15 and per-60-minute Sharpes —
     a 3.4641× span — while SR₀ is compared against a per-15-minute Sharpe. Measured, receipted,
     and awaiting ruling; the recipe is frozen and unchanged.)*
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
- **Seeds:** exactly **5**, literal values **{0, 1, 2, 3, 4}** (§7 v1.5 item D — S=5 UP FRONT;
  `conformance.PINNED_SEEDS`, which the orchestrator now REFERENCES rather than restating, §7
  v1.6 C-6).
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
- **Attention mode:** ONE mode produces all 25 runs and the headline. It is fixed at the
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
≥ the 0.5 floor, DSR ≥ 0.95 under the same N=60 budget) on the same pinned surface (§3a).
Otherwise it is reported **descriptively with CIs, claimed as nothing** — and **double-NULL
("neither leg survives") is a stated possible outcome of this experiment and of the paper.**

## 6. Binding gates (G-§8.C.3 quantified in v1.2; others unchanged)

G-instrument-live (anchor `3f86882a…`, re-anchored §7 v1.4.2 for the μ̂-mean estimator; `5eead7b6…` retired to history), G-parity, G-causal, G-determinism — per m6_design §3.

> **★ SUPERSEDED — G-§8.C.3 IS NO LONGER BINDING (Lakshay's ruling, 2026-08-03, §7 v1.6.22).**
> The clause below is retained VERBATIM as the pre-registered text and is **not in force**: it is
> unexecutable on three counts (two published Kronos-small RankICs 2.4× apart, both on SSE 15-min
> equities, and running the weights needs Kronos model code invariant 8 forbids). What replaces it
> is a **required disclosure** — *"We therefore cannot exclude that our BSQ baseline is weaker than
> a reference BSQ implementation, which would inflate the FSQ-vs-BSQ comparison reported in §5"* —
> carried in every verdict manifest as `external_validation.required_disclosure` and refused by
> `verdict.load_verdict_manifest` if absent. **§7 v1.6.25 R8: this banner was missing here, at
> `m6_design.md:50` and at `ROADMAP.md:58` — three documents describing a live entry gate that
> had been withdrawn, which is the "fix the class, not the instance" lesson recurring.**

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
  scratch with the same seeds {0,1,2,3,4}**, and the event + diff is a dated §7 entry. There is no
  other legitimate path to a re-run.

## 7. Amendment log

- **v1.6.51 (2026-08-19, THE RETIRED FILENAME IS GONE — AND A COMMIT WHOSE MESSAGE CLAIMS THAT
  LANDED A CENSUS CORRECTION FOR A SOURCE STATE A LATER HISTORY REWRITE DESTROYED.)**
  - **★★ PERMANENT ANNOTATION ON `a96fbb1`, WHICH CANNOT BE AMENDED AND WILL NOT BE REWRITTEN.**
    Its message carries the rephrasing and read-back findings verbatim; **its tree touched ZERO
    prereg files.** The rephrasing it describes was destroyed with the rest of the writer's later
    work in the history rewrite that followed, but the paper-side census change it also carried
    **survived**. A third rewrite is not acceptable, so the divergence is recorded here instead:
    **`a96fbb1`'s message describes prose edits that its tree does not contain.**
  - **★ THE DEFECT THAT LEFT BEHIND: THE PAPER OVER-COUNTED ITS OWN LOG BY ONE, IN FOUR PLACES.**
    `a96fbb1` moved the paper from 64→65 tags and 61→62 entries. Those were the CORRECT values for
    a tree containing v1.6.51 — measured while that entry existed. The entry was destroyed; the
    correction was not. **A derived artifact was corrected to match a source state that a later
    history operation then removed**, which is a new member of the propagation class: previously a
    correction failed to reach a derived artifact; here it reached one and the source went away.
  - **AND NO GUARD IN THE SUITE COUNTS THE LOG, so the battery was green throughout.** `tectonic`
    exit 0, `check_claim_drift` CLEAN, `check_submission_build` PASS and `ruff` clean are all true
    of a paper asserting a tag count its own log contradicts. **A green battery is evidence about
    what is checked, not about what is right** — and nothing checked this.
  - **THE REPHRASING, REDONE.** Seven occurrences in five passages. Pointers became *"the
    engineering guide"*; subjects became descriptions rather than quotations; v1.6.48's restored
    sentence reads *"the file it would replace is still in place"*; and the reconciliation line
    states the file **was renamed on 2026-08-19**, so a path cited in an earlier entry reflects the
    current filename rather than the one in force on that entry's date. **Where the rule needs the
    name to stay legible, the bracketed form `[the retired filename]` is used** — the audit
    region's own device, applied to the rule that governs it.
  - **★ AND READING THE RESULT BACK CAUGHT THE SAME TWO DEFECTS TWICE, WHICH IS THE FINDING.** The
    redo reproduced both, because I reapplied the FIRST-draft replacement text rather than the
    corrected one: a collided sentence — *"rewrote v1.6.48's own sentence its assertion that…"* —
    and a heading reading **"ONE LINE RECORDS THE OLD NAME"** above a line that by design no longer
    did. **Recovering destroyed work from memory recovers the draft, not the fix**; the corrections
    made after a passage was written are exactly what a redo loses. Both fixed again, and the
    bullet rewrapped.
  - **TWO EARLIER FINDINGS, KEPT BECAUSE THEY ARE TRUE OF ANY TREE.** (1) **The census sync
    silently did nothing**: the `perl` substitution matched none of its patterns — the targets wrap
    across lines in two of four files — and exited cleanly having changed nothing. **A substitution
    reports success by not erroring, which is not the same as having substituted.** (2) **A write
    reported success without writing**: a script printed *"§7 v1.6.51 written"* and the entry was
    absent, and I committed a message describing it. **A script's own success message is not
    evidence that the file changed** — only re-reading the file is.
  - **AND THE ENTRY'S WRITE-GUARD FIRED CORRECTLY**, then and now: the first attempt refused
    because the rephrased text above already cites `v1.6.51`, so a naive *"tag not present"* check
    saw a duplicate. It tests for the **entry header** instead — a citation to an entry is not the
    entry, the same reference-versus-subject distinction one level up.

- **v1.6.50 (2026-08-19, THE COMMIT MAP APPLIED — 36 POINTERS RE-STAMPED, 6 SUBJECTS LEFT, AND THE
  POINTER/SUBJECT RULE TRANSFERRED FROM FILENAMES TO SHAs UNCHANGED.)**
  - **36 OCCURRENCES RE-STAMPED FROM THE MAP, NEVER FROM A PATTERN**, across 19 in-map tokens: 16
    in `paper/` (including both `git show` commands in `make_fig4_legibility.py`) and 20 here.
    Every substitution came from `restamp_map.json`; a token absent from the map was never touched.
  - **★ 6 OCCURRENCES DELIBERATELY LEFT, ON THE RULE THIS LOG DERIVED ONE ENTRY AGO.** A SHA cited
    as a **pointer** — *"landed at `c4cd082`"* — is re-stamped, and the reference stays followable.
    A SHA that is the **subject** of its sentence is not, because re-stamping it destroys the
    statement:
    - **line 244** — v1.6.49's *"`448e4fe`, `2c72fff`, `46c6a9d` and `2012acc` … no longer
      resolve"*. Re-stamped, that sentence becomes **false**: the new SHAs resolve. The whole point
      of the sentence is that the old ones do not.
    - **lines 355 and 358** — v1.6.46's record that *"`845106d` and `96c1f3b` carried identical
      subjects"* and *"`845106d` → `f050236`"*. Both sides are the subject; re-stamping either
      turns a history record into a false claim about commits that never had those names.
    - **`845106d` was excluded by the supervisor** and independently is not in the map: unreachable
      from any ref, so `filter-repo` never mapped it. It is cited as the record of an amendment,
      not as a live pointer.
  - **VERIFIED IN THREE DIRECTIONS, because one direction would not have been a check.** All **17
    distinct new SHAs resolve** as commits; all **7 left-alone SHAs still dangle**, which is
    precisely what their sentences assert; and every **content hash is untouched** — `3f86882a`
    (Gate-A anchor, 18 occurrences), `5eead7b`, `5dfd667d` (lake Merkle), `1d3ec4f8`, `6d118ede`,
    `05b97bda`. **A pattern-based re-stamp would have destroyed all six**, which is why the map is
    the only admissible instrument and why v1.6.49 flagged that my own probe could not tell them
    apart.
  - **THE VERBATIM CASE DOES NOT ARISE IN THIS DOMAIN, CHECKED RATHER THAN ASSUMED.** The builder's
    `tests/test_audit_findings_complete.py` guards `docs/m6_readiness_audit_findings.md` only, and
    the one retained-VERBATIM block here (the superseded G-§8.C.3 clause, line ~194) contains no
    in-map token. No revert-inside-annotate-beside was needed.
  - **THE TWO FILENAME MENTIONS WERE ALREADY DONE IN v1.6.49** — `06_results.tex:7` and
    `make_fig5_design.py:11` both read `docs/ENGINEERING.md`. The mentions of the retired filename
    that survived in this log were all **subjects** in the entries defining the distinction; v1.6.51
    rephrased them so the facts survive without the retired string.
  - **NOT PUSHED, AND NOT MINE TO PUSH.** `origin` is absent from this clone: `git remote -v` is
    empty and `git ls-remote` fails. The work is committed locally and the tree is clean.

- **v1.6.49 (2026-08-19, THE RENAME LANDED AND WAS APPLIED — AND MY OWN BLANKET `sed` CORRUPTED A
  DATED ENTRY BY OVERWRITING THE NAME THAT ENTRY WAS ABOUT.)**
  - **THE HOLD IN v1.6.48 IS RELEASED: `docs/ENGINEERING.md` NOW EXISTS** (the builder renamed it
    in `b964428`, correctly leaving `paper/` and this log alone). **23 references updated** — 2 in
    `paper/` and 22 occurrences across 21 lines here. Zero dangling document citations remain:
    every `docs/*.md` path cited from my files resolves to a file that exists.
  - **★★ AND THE RENAME ITSELF PRODUCED A NEW INSTANCE OF THE CLASS, IN THE ONE FILE WHERE IT DOES
    THE MOST DAMAGE.** A blanket `sed` over a **document of record** rewrote v1.6.48's own assertion
    that the old file was still present into one naming the NEW path instead — sitting beside
    *"`docs/ENGINEERING.md` exists neither locally nor at origin"*. **The entry now asserted that a
    file both did not exist and was still there.** The sentence was not CITING the old name, it was
    ABOUT it, and a mechanical rename cannot tell those apart. Restored, and marked with its date.
  - **THE DISTINCTION THAT SURVIVES: A RENAME IS SAFE FOR A CITATION AND UNSAFE FOR A MENTION.**
    Where the old name is a **pointer** — *"per `[the retired filename]`'s standing rules"* —
    renaming it is correct and keeps the pointer followable. Where the old name is the **subject**
    — *"`[the retired filename]` is still there"*, or the rename statement itself — renaming
    destroys the claim.
    **In a log whose purpose is recording what was true when, the second kind is common**, which is
    why a blanket rename is more dangerous here than anywhere else in the tree. Every remaining
    occurrence was re-checked and all are pointers.
  - **ONE LINE KEEPS DATED ENTRIES RECONCILABLE:** the standing-rules file **was renamed on
    2026-08-19**, so a path cited in any entry dated before then reflects the current filename
    rather than the one in force on that entry's date. That is what lets a reader square an entry
    dated 2026-07-04 with a path created six weeks later, at the cost of one line rather than
    twenty-three annotations.
  - **★ AND THE SHA RE-STAMP IS NOW A MEASURED FACT RATHER THAN A PENDING ASSUMPTION — THE OLD
    COMMITS ARE GENUINELY ORPHANED.** `b964428` re-stamped 329 SHAs from the commit map, which
    means it rewrote history. Checked directly: `448e4fe`, `2c72fff`, `46c6a9d` and `2012acc` —
    all cited in my files — **no longer resolve** as commits. **The probe was controlled before it
    was believed**: it first reported *41 of 41 dangling*, which is a red flag rather than a
    result, because the token set includes **content hashes that are not commits and must not
    resolve** — the lake Merkle root `05b97bda` and the Gate-A anchor `3f86882a` among them. With
    live commits `310b234` and `b964428` as the positive control and those hashes as the negative,
    the orphaning is confirmed for commits and the false positives are explained.
  - **THE RE-STAMP IS HELD, PER INSTRUCTION AND FOR THE REASON v1.6.48 GAVE.** A wrong SHA is worse
    than a stale one: a stale SHA fails loudly on `git show`, a wrong one resolves to the wrong
    commit and lies quietly. The map decides these, not me — **and my probe cannot cleanly separate
    commit SHAs from content hashes, which is a second reason the map is needed rather than a
    heuristic.**

- **v1.6.48 (2026-08-18, THE PARAMETER CITATION NOW POINTS AT WHAT A READER CAN RECOMPUTE. THE
  RENAME IS HELD, BECAUSE THE TARGET DOES NOT EXIST YET.)**
  - **★ THE RECOMMENDATION IS RIGHT AND IT IS INDEPENDENT OF THE RENAME, WHICH IS WHY IT WAS DONE
    FIRST.** `04_design.tex:66` cited an instruction document for **realized parameter counts**.
    An instruction document **ASSERTS** a measured quantity; a manifest lets a reader **VERIFY**
    one. The citation now points at the artifacts, and the norms file is cited only for the norms
    it is actually authoritative for.
  - **AND THE MANIFEST COVERS HALF THE SENTENCE, NOT ALL OF IT — CHECKED RATHER THAN ASSUMED.**
    `runs_manifest/m6_weights_release.json` carries **31,725,568 / 10,493,952 / 21,231,616** and a
    `how_to_verify` field printing the recipe verbatim: *"load predictor.pt and sum numel() over
    state_dict to reproduce n_params_realized_total"*. **Those are the BSQ figures, because the
    released units are cell 1.** The manifest does **NOT** contain **31,795,200** — the FSQ total
    the same sentence quotes. That half is summed from `runs_cloud/ckpt_cell4_seed0/predictor.pt`
    and asserted at render time by `make_fig5_design.py::_verify_params()`, which raises on a wrong
    constant and has been observed to. **Both halves now cite the thing that can check them; a
    single citation to the manifest would have covered one and implied two.**
  - **★ THE RENAME IS HELD AND THIS IS THE REASON.** *(As of this entry's date, 2026-08-18:)*
    `docs/ENGINEERING.md` exists **neither locally nor at origin**; the file it would replace is
    still in place.
    Repointing 23 citations — 3 in `paper/`,
    20 here — at a file that does not exist would replace a correct citation with a **dangling**
    one, which is the defect class this project has spent two days on, in a new costume: *a
    citation whose artifact is not there*. The references stay pointed at what exists. **The moment
    the rename lands the change is mechanical and I will make it in one pass**, and for this log it
    is a straight rename rather than the document-of-record form — the referent is being renamed,
    not retired, and **nothing here expired**, which is the distinction v1.6.47 turned on.
  - **THE COMMIT RE-STAMPING IS PENDING THE BUILDER'S MAP AND WILL NOT BE GUESSED.** Twenty
    commits cited across my files need new SHAs. A re-stamped SHA that is wrong is worse than a
    stale one, because a stale SHA fails loudly on `git show` and a wrong one resolves to the
    wrong commit — which is exactly the failure mode the baseline rule was written for.

- **v1.6.47 (2026-08-18, THE REPO WENT PUBLIC. TWO DATED PREMISES EXPIRED AND NEITHER WAS
  REWRITTEN — THIS IS THE ONE FILE WHERE THE CORRECTION FORM IS DIFFERENT.)**
  - **VERIFIED FIRST, UNAUTHENTICATED:** `lakshayybhati/trikaal` returns **HTTP 200**,
    `"private": false`, `"visibility": "public"`.
  - **★ WHY NOTHING WAS OVERWRITTEN, AND WHY THAT IS THE POINT.** §7 v1.6.43 split the correction
    form by artifact: **this log records what we believed and when**, so a correction here must
    SHOW the belief it replaced; **`paper/` teaches a finding**, so a correction there is just the
    finding stated correctly. R1's premise — *"the repo is private, so scp a tarball"* — sits in
    the entry dated **2026-08-03** and **was true on that date**. It is not a false statement to
    overwrite; it is a **superseded operational premise**, and silently rewriting it would make
    this log the one thing it cannot afford to be: a record whose entries change after the fact.
    Both sites therefore keep their original wording and carry a dated addendum beneath it.
  - **R1 ITSELF IS UNCHANGED AND IS NOW EASIER TO SATISFY.** A public repo clones with **no
    credential at all**, so *"no token ever reaches a rented box"* holds trivially rather than by
    the tarball workaround. **The mechanism got simpler; the rule did not move.** Verified — not
    relayed — that the builder made the same correction in `docs/cloud_runbook.md` and
    `docs/m6_fanout_runbook.md`, in the same struck-premise-plus-new-fact form.
  - **★ AND THE SWEEP FOUND A SECOND EXPIRED PREMISE THAT NOBODY ROUTED, where the conclusion
    survives and one supporting fact does not.** Line 1361 argued the M6 lake is not publicly
    readable from two observations: a 401 on `trikaal-m6-snapshot`, **and** that the account lists
    *"zero public repos"*. Re-checked: the **401 still holds on both `api/datasets/…` and
    `api/models/…`, so the conclusion is unchanged** — but the account now lists
    `lakshayybhati/trikaal-v1-baseline` with `"private": false`, created by our own weights
    release. **One of the two supporting observations expired; the other and the conclusion did
    not.** Recorded that way rather than as a blanket correction, because collapsing "one premise
    expired" into "the claim was wrong" would overstate it — the same defect as the "artifacts
    disagree" disclosure of v1.6.39.
  - **THE CLASS, ONE MORE TIME AND FROM A NEW DIRECTION.** Every previous instance was a claim
    that went stale because **we** changed something. This one went stale because **the world**
    did: a repository's visibility is not a fact about our text, and no grep over our own diffs
    would ever have surfaced it. It was found by an agent re-reading a runbook with the current
    state in mind — which is the third clause of the rule (v1.6.46) arriving from the opposite
    side: **an instrument scoped to what you edited cannot see a premise invalidated by something
    you did not edit.**

- **v1.6.46 (2026-08-17, ★★ FOURTH INSTRUMENT FAILURE OF THE DAY, AND IT WAS THE INSTRUMENT BUILT
  TO CATCH THE OTHER THREE. THE READING IS WHAT FOUND ALL FOUR.)**
  - **THE SITES.** §6.5 said the class weight *"takes nothing from the PRICE channels"* and *"costs
    the PRICE channels nothing"*. The measured quantity is OHLCV reconstruction MAE over **seven**
    dimensions — `ret_close, range, body, upper_wick_frac, lower_wick_frac, log_volume,
    log_amount` — of which **two are volume**. So the noun named five of the seven that were
    measured, in the subsection carrying our answer to the ceiling objection.
  - **★ AND THE SUPERVISOR'S READING SHARPENS MY OWN DISTINCTION, SO I AM REPLACING IT.** I had
    split these sentences into OUTCOME statements (where "price" is simply wrong) and MECHANISM
    statements (where it was not false). These are neither: they describe what the weight COSTS.
    The right test is not which class a sentence belongs to — it is **does the noun name exactly
    what was measured?** Here it named a SUBSET, which is an error of coverage whatever the
    sentence is doing. That test would have caught all six of yesterday's sites and both of these;
    my two-class split caught six and missed two.
  - **THE FIX IS ALSO STRONGER THAN THE SENTENCE IT REPLACES.** The weight costs nothing across
    **all seven** dimensions including the volume channels — and volume is precisely what the
    SURVIVING microstructure co-varies with, so the volume channels are the ones a sceptic would
    most expect to pay. Saying "price" understated our own result.
  - **★★ THE FINDING THAT OUTLASTS THE FIX: EVERY INSTRUMENT BUILT FOR THIS CLASS HAS BEEN
    NARROWER THAN THE CLAIM IT CHECKED — INCLUDING BOTH BUILT SPECIFICALLY TO CATCH IT.**
    (1) my sweep, scoped to `sections/`, missed `main.tex`; (2) the supervisor's grep matched
    *"duplicates price"* and missed *"duplicates what price already carries"*; (3) the builder's
    matched *"vs"* and missed *"against"*; (4) **`check_claim_drift.py` reported CLEAN while
    "price channels" survived twice** — its `\bprice\b` rule fires only alongside eviction
    vocabulary, and these sentences use cost vocabulary instead.
  - **SO THE RULE NEEDS A THIRD CLAUSE: A CLAIM CAN BE RESTATED IN A VOCABULARY THE INSTRUMENT DOES
    NOT SHARE.** Grep the superseded string → catches verbatim survivals. Grep the claim's
    distinctive terms → catches restatements in the vocabulary you anticipated. **Neither catches a
    restatement in vocabulary you did not.** The only defence that has worked all day is **a human
    or agent reading with the claim in mind**: all four of these were found that way, and none by
    the instrument that preceded it. **Keep the instruments; do not let them retire the reading.**
    A sixth rule now covers cost vocabulary, and it is observed to fail on the pre-fix text — but
    it was added *after* a reading found what it could not.
  - **HISTORY HYGIENE.** `845106d` and `96c1f3b` carried identical subjects, one mine and one the
    builder's, which makes the log unreadable at the moment strangers start reading it. Mine was
    the tip and matched origin exactly, so its subject was amended and force-pushed **under
    `--force-with-lease`**, which refuses if anyone has pushed since. `845106d` → `f050236`;
    **the builder should rebase rather than merge if it holds the old SHA.** The distinction from
    the earlier incident this week is deliberate: that one amended ANOTHER author's commit without
    re-checking the tree; this is my own commit, at the tip, with a lease guard, at the
    coordinator's request.

- **v1.6.45 (2026-08-17, ★★ THE PAPER WAS RIGHT AND THE FOUR DOCUMENTS A READER MEETS FIRST WERE
  WRONG — AND THREE HAND-GREPS ALL MISSED THE ABSTRACT BECAUSE EACH WAS NARROWER THAN THE CLAIM.)**
  - **★ THE SECOND HALF OF THE RULE, WHICH IS THE PART THAT MATTERS.** "Grep the superseded string"
    catches **verbatim** survivals. It does not catch **the claim restated in different words**,
    and that is what defeated all three of our instruments at once: my sweep ran over `sections/`
    and **not `main.tex`, where the abstract lives**; the supervisor's matched the exact phrase
    *"duplicates price"* where the abstract said *"duplicates what price already carries"*; and the
    builder's matched *"vs"* where its own text said *"against"*. **Each was narrower than the
    claim it was checking.** So: grep the claim's DISTINCTIVE TERMS — here, any sentence pairing
    the eviction claim with "price" rather than "OHLCV".
  - **SIX SITES CORRECTED, IN TWO CLASSES.** Where the sentence describes the OUTCOME — which
    channels survive, what the shuffle destroys — *price* is simply **wrong**, because the
    survivors co-vary with **volume**: the abstract (main.tex:103), §6.5 twice, §7.3's placebo
    sentence. Where it states the general MECHANISM — *"a low-variance channel weakly covariant
    with price"* — it was **not false**, and the builder was right to decline to call it a defect;
    the referent is made explicit anyway (*"the high-variance channels already being coded"*),
    because that exact ambiguity is what produced the wrong title in the first place.
  - **★ AND THE INSTRUMENT FOUND ONE THE ROUTING DID NOT.** §6.5's closing sentence still read
    *"the placebo is the one comparison in the design where capacity is constant and structure is
    not"* — **the capacity-neutral claim withdrawn two commits ago, restated in the summary
    sentence four paragraphs below the step that withdrew it.** The class again, on my own fix,
    inside the same subsection.
  - **`paper/check_claim_drift.py`: FIVE RULES, EACH PAIRING A CLAIM PATTERN WITH A FORBIDDEN ONE**
    — price-referent, ladder-necessity, "essentially pass", the firing attributed to 200
    instruments, and placebo-capacity-neutrality — over `sections/*.tex` **and `main.tex`**.
    Exemptions carry the reason they are allowed, so a silence cannot outlive its justification.
  - **ITS OWN FALSE-POSITIVE RATE IS A MEASURED PROPERTY, AND THE FIRST DRAFT HAD ONE OF TWO.** The
    sentence splitter broke on **decimals** (`0.90` ended a sentence) and merged `%` comment lines
    into surrounding prose once newlines collapsed, so it fired twice on a clean tree. **A gate
    with false positives becomes decoration.** Fixed: comments stripped before flattening, and a
    boundary requires a period followed by whitespace and a capital or a control sequence.
  - **★★ AND THE MUTATION HARNESS LIED TO ME A THIRD TIME.** Rule 4 reported `EXIT 0` — a
    failure to fail — and the rule was fine: **my `sed` targeted a phrase the rewritten abstract no
    longer contains, so the mutation never landed.** Caught only by asserting the mutation was
    present before believing the pass. A separate one silently `git checkout`-ed a file and
    **reverted a fix I had made minutes earlier**, which the control run then surfaced. All five
    rules are now observed to fail on landed mutations, with a comment-only no-op as negative
    control.
  - **★★ THE HIGHEST-VALUE ITEM IS STILL OUTSIDE THE WRITER'S DOMAIN AND IS NOW MORE URGENT.**
    `README.md`, `docs/MODEL_CARD.md`, `scripts/m6_weights_release.py` and
    `runs_manifest/m6_weights_release.json` all still say the tokenizer *"keeps microstructure that
    duplicates PRICE"*. **The paper is now correct in all six places and the four documents a
    reader meets FIRST are wrong** — the model card is the HuggingFace landing text and the release
    receipt is what a downstream user quotes. Flagged, not touched.

- **v1.6.44 (2026-08-17, THE THIRD FORM OF THE SAME CLASS — THE SWEEP COVERED SOURCES AND
  STOPPED AT EXPORTS — AND THE INSTRUMENT FOUND FOUR SHIPPING ARTIFACTS THAT ARE NOT OURS.)**
  - **THE CLASS, NOW COMPLETE AT THREE INSTANCES: A CORRECTION PROPAGATES TO WHAT YOU EDIT, NOT TO
    WHAT DERIVES FROM IT.** Prose → generators (fig1 kept `21.3M` through the parameter sweep);
    title → figures that quote it (fig12 kept "duplicates price" through the title fix); **sources
    → exports** (`paper/main_full.txt`, four days stale, carrying the superseded title). Each time
    the edit was correct and the derived artifact was not swept. **The only reliable instrument is
    a grep of the superseded STRING across the WHOLE tree**, not a review of the files you touched.
  - **DELETED, because their purpose is spent and regenerating buys nothing:** `paper/main_full.txt`,
    `paper/03_mechanism.md`, and the whole `review_s3`–`s7` family (`.tex`, `.pdf`, `.blg`, `.log`,
    `.txt`) — exports built 2026-08-09/10 for reviews that are finished, describing an experiment
    that had not yet stopped. Also `paper/main.blg` and `paper/main.log`, which are build detritus
    that was tracked. **No regeneration step was added, deliberately: "remember to regenerate" is
    the shape of the defect, and the paper builds from source in one command.**
  - **THE SWEEP FOUND TWO THINGS IN paper/ THAT THE AUDIT DID NOT NAME.** §6.3 still read *"the four
    magnitude dimensions essentially pass"* — the same defect corrected in the abstract two commits
    earlier and missed one section away, which is the class again at the smallest scale. And §8.9
    still narrated the figure's draft history in the present tense; the methodological point
    survives, stated as a property rather than as a story, per the prereg/paper split.
  - **★★ AND THE HIGHEST-VALUE FIND IS NOT IN MY DOMAIN: FOUR ARTIFACTS THAT SHIP WITH THE WEIGHTS
    CARRY THE SUPERSEDED AND FACTUALLY WRONG FRAMING.** `README.md:21`, `docs/MODEL_CARD.md:37`,
    `scripts/m6_weights_release.py:135` and `runs_manifest/m6_weights_release.json` all state that
    the tokenizer *"keeps microstructure that duplicates PRICE"*. **The surviving channels co-vary
    with VOLUME, not price** — that is the reviewer's first quibble and the reason the title was
    changed. These are the release surfaces: the model card is what a HuggingFace reader sees
    first, and the release receipt is what a downstream user quotes. **Flagged, not touched** —
    outside the writer's domain. The correction is the title's: *keeps the microstructure that
    duplicates OHLCV and drops the signed channels.*

- **v1.6.43 (2026-08-17, ★★ THE AUDIT VOICE LEAKED INTO THE PAPER VOICE. 54 PP → 42 INTERNAL /
  39 SUBMISSION, AND ALL TWELVE PROTECTED PASSAGES SURVIVED.)**
  - **★ MY RULING ON THE AUDIT-VOICE DIAGNOSIS: THE REVIEWER IS RIGHT, AND THE ROOT CAUSE IS MINE
    AND NAMEABLE.** *"A paper is not its changelog"* is correct, and the reason the changelog got
    into the paper is that **I applied the document-of-record norm to the wrong document.** That
    norm — quote the false wording, name why it was false, never silently rewrite — is right for
    THIS log, which exists to record what we believed and when. Applied to `paper/` it produced a
    paper that argues with its own previous drafts in the present tense. **The corrections are all
    still recorded; they are recorded where they belong.** Removed from the paper: "This subsection
    asserted through several drafts…", "One admission belongs with this…", "an earlier draft of
    this sentence quoted…", "This entry described §6 before the run. It is corrected rather than
    replaced…", and the parameter-count revision history.
  - **THE CUT, EXECUTED: 54 → 42 pp internal, 50 → 39 pp submission.** §7 from twelve subsections
    and 609 source lines to eleven and 277, each limitation stated ONCE at its first natural home
    (−4 pp). §4.4–4.7 — a pre-registration document embedded in a paper — compressed into one
    subsection with the machinery pointed at Appendix D (−2 pp). **Figures 8, 9 and 10 DELETED**
    with their generator and orphaned PDFs (−3 pp): three consecutive pages of empty watermarked
    panels read, in the reviewer's words, as *"the paper performing its own honesty"*, and a
    sentence does what an empty plot does. §8.5's GPU-rental narrative to one paragraph, §5.6's
    storage layout to one sentence, the abstract 476 → 377 words with its three-negation ending
    gone, and the one-line widow closed.
  - **★ WHAT I REFUSED, AND WHY.** (1) The reviewer wanted §7.2, the dropped external anchor, in
    *"two sentences"*. It is a paragraph: a reader needs it once, but a **referee** checks the
    withdrawal where it is used, and the required-disclosure field shipped in every verdict
    manifest has to be traceable to prose. (2) The **placebo handicap** stays in §4.3 *and* §7.3
    rather than once — §4.3 states the design fact, §7.3 states what it costs the claim, and C-12
    is what makes that the difference between a disclosure and a live objection. (3) **§7.6 was
    not cut to limitation size.** The reviewer itself called break-even *"the most trading-relevant
    number in the paper"* and complained it was *"filed as a limitation"*; it stays in §7, because
    that is where a bounded negative result belongs, but it now opens by saying it is the number to
    take away, and Figure 11 draws it.
  - **ALL TWELVE PROTECTED PASSAGES SURVIVE, verified in the render — and one FALSE ALARM is worth
    more than the confirmation.** My check reported *"Both outcomes are reportable"* missing; the
    phrase wraps across a line in the PDF and my grep required it whole. **A verification that
    fails on its own formatting is a false positive, and I came within one step of "fixing" a
    passage that was never broken.** The instrument, not the paper, was wrong.
  - **THREE OF THE REVIEWER'S FOUR "MISSING FIGURES" ALREADY EXIST**, built before the review
    arrived: the argument-chain schematic is Figure 12, the economics figure is Figure 11, the seed
    divergence is Figure 13. The two still absent — the smearing curve, and a real-data per-dim
    legibility figure across all thirteen live dims — are buildable from existing receipts and are
    named as future work rather than quietly dropped.
  - **SUBSTANTIVE FIXES BEYOND ITEMS 1–4.** The title said *"Duplicates Price"* when the surviving
    channels co-vary with **volume** — now *"Duplicates OHLCV and Drops the Signed Channels"*. §1
    carries the signed-vs-magnitude split on **page one** as the result rather than on page 24. The
    smearing mechanism gets its gloss where it is first used instead of seven pages later. §3.2
    states the fixture↔real mapping (why eleven fillers for a 7+6 vector). Figure 2's caption title
    said *"reconstruction quality"* for what is **per-bar recoverability**. And §7.1's *"no
    experiment here varies the objective on market data"* contradicted §6.4, which sweeps λ on
    market data — corrected to name the sweep as the one real-data intervention we have.

- **v1.6.42 (2026-08-17, ★ MY OWN ITEM-1 STEP 2 WAS RIGHT FOR THE WRONG REASON — THE PLACEBO IS
  NOT CAPACITY-NEUTRAL, AND WE MEASURED THAT OURSELVES.)**
  - **WHAT I WROTE AND WHY IT WAS FALSE.** §6.5 step 2 claimed the placebo *"holds capacity fixed
    and changes only structure"* and concluded that *"a comparison in which capacity is held
    constant cannot have its differential explained by capacity."* **The premise is false.**
    Shuffled microstructure is incompressible, so cell 5 faces a HARDER compression problem at the
    same budget: **same BUDGET, less EFFECTIVE capacity.**
  - **VERIFIED AT ENTRY-TIME FROM OUR OWN RECEIPT, NOT FROM THE CORRECTION.**
    `runs_manifest/m6_ohlcv_recon_ratio.json` (C-12): cell 5 reconstructs the **byte-identical**
    OHLCV targets **1.512811× worse** than cell 4 — worse on **all seven** price dimensions, with
    per-dimension ratios **1.2426 to 1.8654**. The placebo permutation never touches the OHLCV
    columns, which is exactly what makes the comparison like-for-like and the finding unavoidable.
  - **★ AND THE DANGER WAS SPECIFIC: WE DISCLOSE C-12 OURSELVES.** A referee who read our own
    disclosure would have found the paper asserting capacity equality across the placebo while a
    committed receipt of ours measures a 1.51× capacity gap. **We would have handed them our own
    measurement as the weapon.** This is the recurring class — an argument that is right for a
    slightly different reason than the one stated — and it is the second time in two days that the
    stated reason, not the conclusion, was the defect.
  - **THE ARGUMENT SURVIVES, RE-FOUNDED ON THE WITHIN-ARM PATTERN.** A capacity penalty is
    indiscriminate: a harder problem at a fixed budget degrades what the tokenizer encodes roughly
    TOGETHER. Measured under the shuffle: the two signed channels fall **−0.2088** and **−0.1632**,
    collapsing to within 0.10 and 0.06 of their majority-class base rates, while the four magnitude
    channels fall **−0.0059, −0.0551, −0.0271, −0.0635**. **The smallest signed drop is 2.6× the
    largest magnitude drop** — a separation between two disjoint groups, along exactly the
    signed/unsigned line. **A uniform capacity penalty cannot produce that.** The paper now states
    what the placebo is NOT before stating what it shows, and quotes the false framing rather than
    deleting it, per the document-of-record norm.
  - **STEP 1 KEPT VERBATIM AS INSTRUCTED:** *"a budget constraint does not know which dimensions
    are signed."*
  - **STILL HELD: items 5, 6 and 7.** The full review has not arrived.

- **v1.6.41 (2026-08-17, ★★ THE CEILING-VS-ALLOCATION OBJECTION IS VALID AND IS NOW CONFRONTED
  WHERE IT ARISES — AND THE ABSTRACT WAS ATTRIBUTING THE GATE'S FIRING TO 200 INSTRUMENTS WHEN THE
  PROBE SAMPLED 40.)**
  - **★ THE RULING ON ITEM 1, WHICH IS THE ONLY ITEM THAT CHANGES WHAT THE PAPER CLAIMS.** A
    reader-reviewer objected that §6.4's readings — the class weight barely moves TFI and costs
    OHLCV nothing — describe a **CEILING**, not an **ALLOCATION**, since nothing is traded off; and
    that §3.1's own bit arithmetic may over-determine the real-data failure without any allocation
    story. **I accept the objection as valid, reject that it is the whole story, and adopt the
    both-framing.** New §6.5 makes the argument where the objection arises instead of labelling the
    readings "strengthens the primary finding" and moving on.
  - **THE THREE-STEP ARGUMENT, EACH STEP VERIFIED AT ENTRY-TIME.**
    (1) **A ceiling predicts a UNIFORM shortfall; ours is selective.** 97.3% of cell 4's shortfall
    is in two of six dims — recomputed from the stop receipt and reproduced exactly (my own first
    subtraction was wrong; the paper's figure is right). A budget constraint does not know which
    dimensions are signed.
    (2) **The placebo holds capacity CONSTANT and changes only structure.** Cell 5 has the same
    bits, dims, budget and architecture; only the covariance with price and the block's own serial
    structure are destroyed. Signed dims collapse 0.8223→0.6135 and 0.7528→0.5896 while the four
    magnitude dims hold. **A comparison in which capacity is held constant cannot have its
    differential explained by capacity.** This is the discriminator and it is measured on real data.
    (3) **The channel is ABSENT, not merely per-bar illegible.** The fixture's escape — state
    encoded but smeared across the window — does not apply here.
  - **★ AND THE DECISIVE NUMBER WAS ALREADY MEASURED AND ABSENT FROM THE PAPER.**
    `runs_manifest/m6_window_context_probe.json`: reading TFI from the whole 512-bar window rather
    than bar *t*'s token moves sign accuracy **0.8399 → 0.8385, Δ = −0.0014**, across 19 log-spaced
    lags to 511 with the gate's own estimator, split and seed. **The window recovers NOTHING.** The
    reviewer named this as the sharpest question a referee will ask (~0.95 → the fixture's story;
    ~0.82 → the tokenizer cannot encode the channel at this budget) and the answer is the second.
    Its restricted basis can only UNDER-state recoverability, so the null is conservative.
  - **ITEM 2, A LOGICAL OVERCLAIM, CORRECTED IN THE ABSTRACT AND §3.6.** "Each was necessary; none
    was sufficient alone" is not what a CUMULATIVE ladder establishes. (i), (i)+(ii), (i)+(ii)+(iii)
    establishes **insufficiency of the prefixes and sufficiency of the triple** — necessity of (i)
    or (ii) needs each REMOVED from the final triple, which was never done. Only (iii) has that
    evidence. The claim is weakened to what the ladder supports rather than the leave-one-out being
    run, and the abstract's "necessary and not sufficient" becomes "not sufficient".
  - **ITEM 3, THE HIGHEST VALUE PER WORD AND THE REVIEWER IS RIGHT.** §3.2 presents Figure 2 as
    measuring "the tokenizer" and a reader takes that as the ORIGINAL contextual design; it was
    produced by a tokenizer that ALREADY carries interventions (i) and (ii), which do not exist in
    the reader's world for another three pages. One sentence at the head of §3.2 now says so, and
    says the original fails the same test MORE severely, not less.
  - **★★ ITEM 4 CONTAINED ONE DEFECT WORSE THAN THE REVIEWER GRADED IT, IN THE MOST-QUOTED SENTENCE
    OF THE PAPER.** It flagged "the scored universe is 40 instruments, not 200" as a buried lead.
    It is not buried, it is **misattributed**: the abstract read *"On 304,625,181 one-minute bars
    across 200 instruments, that gate fired"*, and the gate's probe measured **150,000 rows across
    40 symbols** (`n_symbols_in_sample = 40`, both arms). The lake is the training draw; the probe
    is not the lake. Now: *"Trained on ... 200 instruments and probed on 150,000 decisions drawn
    from 40, that gate fired."* **Neither audit caught this.**
  - **THE REST OF ITEM 4, ALL VERIFIED BEFORE FIXING.** Figure 3's caption described a **different
    figure** — "nats of validation cross-entropy ... dashed rule marking the information planted"
    against a generator that draws **probe correlation with a 0.30 detection threshold**; corrected
    to what is plotted, with the nats named as receipt values quoted beside it. **β was never
    defined** anywhere (`(λ, β)` at §3.6) — now defined as the weight on the bottleneck leg's own
    reconstruction term. **§3.2 contradicted itself in one paragraph** — "every dimension is
    marginally standard normal" against "the return carries 3.15× the marginal sd of the rest";
    the twelve compared dims are matched, the thirteenth return dim is deliberately not, and that
    is the point. **"The four magnitude channels essentially pass"** hid that two of four fail the
    letter of the gate; the abstract now prints 0.0025 and 0.0038. **§6.3 and §7.1 disagreed** —
    "observed on real market data" against "carried over by argument"; §6.3 now separates the
    CONSEQUENCE (observed) from the CAUSAL claim (carried over), and points at §6.5.
    **The λ table's note rendered to the RIGHT of the table** rather than beneath it; fixed and
    confirmed by rendering.
  - **HELD, NOT REFUSED: items 5, 6 and 7** — the audit-voice diagnosis, the ~14-page cut list, and
    promoting the signed-vs-magnitude split to the headline. The supervisor's instruction was to
    read the full review before acting and it has not arrived; these three are restructuring
    decisions whose whole value is in the reviewer's detail, and acting on a relay would be the
    thing this project exists not to do. **The one-line widow is deferred with them**, since the
    document re-flows wholesale under any cut.

- **v1.6.40 (2026-08-17, THREE NEW FIGURES, ALL RECEIPT-BACKED — AND THE FIGURE CHECKER I BUILT
  TO CATCH CHECKS-THAT-CANNOT-FAIL WAS ITSELF ONE, FOR EVERY TITLE IN THE PAPER.)**
  - **THE NON-NEGOTIABLE HELD: every quantity in every new figure loads from a named receipt at
    render time, and each new generator carries a `_verify()` that REFUSES TO DRAW if the receipt
    stops supporting the figure's own headline.** Fig 11 raises if any unit's break-even ever
    clears the fee; Fig 12 raises if the stop receipt stops recording a halt or if a gated arm
    clears the threshold; Fig 13 raises if the two receipts disagree on any unit's lean or if a
    short-leaning unit's lean stops losing money.
  - **BUILT 3 OF THE 7 CANDIDATES.** (1) **Fig 12, the decision path** — the sequence a reader
    otherwise assembles from prose in three sections; schematic boxes, every number loaded.
    (2) **Fig 11, the economics** — measured break-even against realistic execution on a LOG axis,
    because the finding is the SIZE of the gap and a linear axis puts every measured value on the
    origin. (3) **Fig 13, three seeds** — candidates 3, 4 and 7 MERGED, because they are three
    views of the same three units and the same 1,402,520 decisions each; split into three figures a
    reader has to reassemble the causal chain that is the actual point.
  - **REFUSED, WITH GROUNDS.** **Candidate 5 (three estimators): its own receipt records the input
    as `/Users/.../BTCUSDT_1m.csv`, NOT_IN_THE_REPOSITORY** — a figure no reader could regenerate,
    and the receipt itself calls it *"illustrative of a measured property, not a new measurement"*.
    **Candidate 6 (data and fold design): IT ALREADY EXISTS** — fig6's three panels ARE the
    purged-walk-forward/embargo/ACF diagram and fig7 is the 200-symbol coverage figure; adding a
    third would be duplication. **Candidate 3 standalone: its receipt is 2 SYMBOLS AND 80
    DECISIONS**, and its own header says the raw disagreement rate must not be quoted alone — the
    finding is kept but drawn at MONEY scale (1.4M decisions per unit) inside Fig 13 instead.
  - **AND ONE QUANTITY IN THE BRIEF WAS NOT DRAWN BECAUSE I COULD NOT SOURCE IT.** The "25–446×
    over-dispersion" figure appears in no receipt I could locate. Fig 13 uses the two ratios that
    ARE sourced — sd(μ̂) against the return sd it forecasts (1.5–3.0×), and against a calibrated
    forecast at our own measured prior (56–110×, §7.7). Stated rather than quietly substituted.
  - **★★ THE CHECKER WAS EXTENDED FOUR TIMES, EACH TIME BECAUSE A RENDER WAS LOOKED AT AND SHOWED
    SOMETHING IT MISSED — AND THE FOURTH IS THE ONE WORTH KEEPING.** v1 containment only → passed a
    legend sitting on a data row. v2 + overlap → passed a scope line running **232px off the
    canvas**. v3 + figure-level text → passed a panel label overlapping its own title. v4 + titles
    and axis labels, overlap made GLOBAL → a title spilling into the NEXT panel was invisible while
    overlap was computed one axes at a time.
  - **★ AND THEN v4 DID NOT WORK, WHICH IS THE REAL FINDING. `ax.title` IS THE *CENTRE* TITLE AND
    IS EMPTY WHENEVER `loc="left"` IS USED — WHICH IS EVERY PANEL TITLE IN THIS PAPER.** The check
    "now covers titles" was reading an empty artist and could not fail. It was caught only because
    a deliberate over-long-title mutation **failed to fail**, and only because I checked that the
    mutation had actually landed before believing the pass. Matplotlib keeps three title artists;
    all three are read now. **A check-that-cannot-fail, inside the checker built to catch
    checks-that-cannot-fail.**
  - **WHAT THE HARDENED CHECKER THEN FOUND IN FIGURES THAT HAD BEEN PASSING:** fig4's right-panel
    title ran **94px** past the canvas and fig9's x-axis label **26.6px**. Stated precisely rather
    than dramatically: under `bbox="tight"` such text is **not clipped** — it widens the saved
    bounding box, so the figure is scaled down to fit `\textwidth` and its type renders SMALLER
    than its siblings'. **That is the mechanism behind a figure set that reads as separate
    efforts**, which is exactly the complaint this pass started from.
  - **THE SYSTEM WORK IS A FIVE-STEP TYPE LADDER plus three shared helpers** (`panel_title`,
    `finding`, `scope`), on top of the palette that already existed and was already Tol-based and
    colourblind-checked. `panel_title` folds the panel letter INTO the title because a floating
    "(a)" collides with a left-aligned title on any narrow panel — removing the collision class
    rather than tuning around it. `finding` and `scope` put the finding and the qualifier INSIDE
    the raster, per the context-stripping rule.
  - **COUPLING CONFIRMED, EXACTLY AS WARNED.** Setting an explicit left margin on fig4 to stop the
    x-label overflowing narrowed the right panel and **collided its legend** — a change in one
    place breaking text in another, caught by the checker rather than by me.

- **v1.6.39 (2026-08-17, THE 35,064/35,063 "DISAGREEMENT" DID NOT EXIST; THE WATERMARK OUTLIVED
  THE RUN; AND APPENDIX F'S REMOVAL IS NOW A MECHANISM WITH A TWO-SIDED CHECK.)**
  - **★ MY OWN DISCLOSURE OVERSTATED A PROBLEM, WHICH IS ITS OWN DEFECT.** v1.6.38 recorded that
    "the artifacts disagree" on 35,064 vs 35,063 and declined to explain the off-by-one. Declining
    was right; **"disagree" was wrong.** Measured: they are **two named quantities**, each
    internally consistent — `n_periods` (21) and `n_periods_h15` (1) carry **35,064**, the period
    GRID and the length of the headline series; `T` (2), `T_periods` (1) and
    `decisions_per_symbol` (1) carry **35,063**, the SCORED count, and **1,402,520 / 40 = 35,063
    exactly**, so it is self-consistent with the exact A_p run's own total. **A disclosure that
    describes a defect which does not exist is a defect.** Corrected; the mechanism for the
    off-by-one is still NOT asserted, because it is still a guess.
  - **AND I RECONCILED THE COUNTS RATHER THAN ACCEPTING EITHER SIDE'S.** The audit said "across two
    receipts"; the supervisor said 35,063 appears at "one site"; my first grep said 18. **All three
    were wrong.** A raw grep is inflated by substrings inside longer floats (`35063987737`,
    `350639224`); matching on JSON *values* gives **4 occurrences across 3 receipts under 3 names**.
    This is the distrust-greps norm working: re-run the counterpart grep unanchored and reconcile
    against the tool's own total.
  - **THE WATERMARK OUTLIVED THE EVENT IT NAMED.** `make_fig8_10_stubs.py:93` still rendered
    **"PLACEHOLDER — awaiting the run"** onto figures 8, 9 and 10. "PLACEHOLDER" remains correct —
    those panels carry no result and never will — but **nothing is awaited**: the run is over and
    was stopped. Now **"PLACEHOLDER — THE RUN WAS STOPPED"**, verified on the rendered fig10. The
    two other hits (`make_fig4_legibility.py:26`, `08_repro.tex:317`) are **historical record of
    what the figure used to say** and correctly stay.
  - **★ APPENDIX F'S REMOVAL IS A MECHANISM NOW, NOT AN INTENTION.** F.5 said the appendix "is
    removed before submission", which is a promise. The mechanism: **`paper/submission.tex`** sets
    `\SUBMISSIONBUILD` and `main.tex` gates the appendix on it — build `submission.tex` for
    anything that leaves the project (48 pp), `main.tex` for the internal read-through (52 pp).
  - **AND THE CHECK IS TWO-SIDED ON PURPOSE, because a one-sided one is the defect class.**
    "F is absent from the submission build" passes just as happily if the appendix is **deleted
    outright**, if the gate is **inverted**, or if the target stops compiling the real document.
    `paper/check_submission_build.py` compiles BOTH and asserts four things: F present in the
    internal build, F absent from the submission build, **every other section still present** (a
    written-out list, not one derived from the build, so an over-broad gate fails instead of
    agreeing), and the submission build strictly shorter. **Observed to fail on all three
    mutations** — gate inverted, appendix deleted, gate widened to drop §7 — with a comment-only
    no-op as negative control.

- **v1.6.38 (2026-08-17, ★★ PRE-PUBLICATION AUDIT: THE MEASUREMENT SURVIVED, THE MOST-READ
  SURFACES DID NOT. THE PRIMARY FIGURE DID NOT CONTAIN THE PRIMARY RESULT.)**
  - **THE AUDIT'S OWN VERDICT IS PART OF THE RECORD: *publish with named changes, not as-is*.** An
    independent auditor re-ran the Gate-A anchor bit-for-bit, traced ~90 quoted values to the
    digit, and attacked five gates finding all five bite; its assessment was that nothing traced
    was fabricated, inflated or unreproducible. **Every defect below is on a surface a reader sees
    FIRST — contributions list, abstract, primary figure, limitations opening — and every one of
    them described the world BEFORE the gate fired.** The pattern is the point: the measurement
    layer was maintained and the presentation layer was not.
  - **★★ THE PRIMARY FIGURE DID NOT CONTAIN THE PRIMARY RESULT, AND NOTHING COULD HAVE CAUGHT IT.**
    `fig4_legibility`'s right panel rendered **six empty dashed boxes watermarked "NO VALUES /
    awaiting the run"** for every draft after 2026-08-12, while the six values it was waiting for
    were printed in §6.1 and quoted in the abstract. Root cause verified by grep: the generator
    contained **ZERO references to `m6_micro_legibility_stop.json`** — it loaded four fixture-era
    receipts, found everything it asked for, and rendered a watermark over the result. **Nothing
    failed because there was nothing to fail.**
  - **★ AND THAT FALSIFIED §8.9's CLAIM ABOUT OUR OWN TOOLING** — *"a figure that disagrees with
    its artifact fails to build rather than rendering a stale value."* This figure disagreed with
    its artifact and built happily. **Loading from a receipt prevents a figure CONTRADICTING the
    receipt it reads; it does nothing about a figure that reads NO receipt.** §8.9 now states the
    weaker true claim, names this figure as the instance, and points at the two explicit checks
    that now exist (`fig4` against the stop receipt, `fig5` against the checkpoints), **each
    observed to fail under a wrong constant.**
  - **THE OTHER SURFACE DEFECTS, ALL VERIFIED AGAINST SOURCE BEFORE FIXING.** (1) Contributions
    bullet 4 read *"We RUN a pre-registered ... ablation ... SCORED by cost-aware net IR"* — it was
    neither run nor scored, twelve lines above §1's own *"It did not run."* Rewritten to
    specify/pre-register with the gate outcome named. (2) *"This paper reports no information
    ratio"* is false as written — §7.9 prints gross IRs of +1.2665/+2.1715/+1.1446 and §7.8 reports
    break-even, which our own text calls the same statement as a positive gross IR; now *"no
    ablation outcome — no cost-aware net information ratio"*. (3) §7.2 carried the **RankIC label
    over the plain-IC pair**: "differ by 2.4× (0.0431 against 0.0665)", internally inconsistent
    since 0.0665/0.0431 = 1.54. The RankIC pair is **0.0254/0.0622 = 2.45×**; both pairs now
    printed with the earlier error named. (4) §6.5 said item E was pre-written *"months earlier"*;
    **git says 13 days** — dcdc50e 2026-07-30 against a 2026-08-12 firing — and the error ran in
    the FLATTERING direction in the one sentence whose job is provenance. (5) §7.1 still denied the
    title; rewritten around the distinction that actually holds — the fixture establishes the
    CAUSAL statement, the gate establishes its CONSEQUENCE on markets, and the causal direction on
    real data is carried by argument rather than by a real-data intervention.
  - **★ ITEM 8, RULED. `fig9_cost_stress`'s placeholder curve crossed zero EXACTLY on the pinned
    0.30% rule** — the most quotable point on the axis — reading, once cropped, as a break-even of
    0.30% against a measured 0.0023–0.0208%: **wrong by 14–130×**. Its caption's own defence was
    also false: it claimed magnitudes "two orders above anything this design could produce", and
    the design produced net IRs of −28.5 to −146.5, so ±45 sat **INSIDE** the measured range.
    **I took none of the three options offered.** Moving the crossing (auditor) yields a different
    wrong number, not no number; removing the figure (supervisor's first) contradicts §6.4's own
    stated position that pre-registered-but-unreached analyses are published unfilled as part of
    the record; relocating it to §4 (supervisor's preferred) changes the section but not the
    defect, because **any crossing at any x is quotable**. **I applied Figure 8's solution instead:
    draw the specified apparatus and NO data at all** — axes, pinned band, economic floor, no
    curve, so there is no crossing to quote — and then went further than any option asked, by
    drawing the **one break-even we have actually measured** in the panel with its cell-1-only
    scope IN THE RASTER. The quantity a reader tries to read off is now true rather than absent,
    which is the context-stripping rule discharged **positively** rather than only by omission.
  - **THE RENDER RULE, APPLIED TO THE POPULATION AND NOT TO THE HYPOTHESIS.** All ten figures were
    regenerated and **looked at**. Two defects surfaced that NEITHER audit named: `fig1_architecture`
    still printed **"21.3M params"**, a site the v1.6.34 parameter sweep missed because that sweep
    covered the `.tex` files and one generator but not the other eight — **the class rule failing
    on the class rule's own fix**; and `fig7_coverage`'s two annotations were drawn directly on the
    filled mass and are unreadable at print size. **A third suspected defect was WITHDRAWN after
    checking**: I read "must not not see" in fig6 at 95 dpi, and the PDF text says "must not see" —
    a misreading of my own low-resolution render, not a defect.
  - **THE INSTRUMENT THIS PASS ADDED, because three of the figure defects were invisible to every
    test we own:** `figstyle.assert_text_legible` measures EVERY text node on the RENDERED figure
    via the renderer's own extents — the matplotlib analogue of `getBBox` — and asserts nothing is
    clipped by its axes and nothing overlaps anything else. **Its first version checked containment
    only and passed a legend that was sitting on top of a data row**, which is the same
    check-that-cannot-fail defect one level up; overlap detection was added after that was seen.
    Artists tagged `gid="overlay"` are exempt so the DRAFT watermark can lie across a panel by
    design. Proven capable of failing on three separate mutations with a semantic no-op as negative
    control.
  - **SMALLER, ALL VERIFIED:** §6.6's unexplained `---` at λ=8 hid **0.8465**, three seeds present
    and monotone between the printed endpoints — an unexplained dash where a benign number exists
    reads as selective omission, and it is now printed. §6.0's "confined to §7.5, §7.6" undercounted
    — §7.9 and §8.1 also rest on those three units. `fig8`'s caption claimed its rule strings were
    "persisted into the verdict manifest, not a restatement" when they are hardcoded in the
    generator, and the `--real` swap its docstring and Appendix F.5 describe **does not exist**
    (no argv handling at all). Test count 727 → **911 collected**; census figures 11 → 10, `\PH`
    6 → 5. `fig4`'s caption called the fixture rule "pre-registered" where §3 honestly says
    "restated". A vestigial comment cited "the three comments" where two exist. And 03_mechanism's
    0.3044/0.8216 sat above a receipt that does not contain them.
  - **ONE THING I DECLINED TO ASSERT.** The 35,064-vs-35,063 period discrepancy: every scored money
    artifact carries `grid.n_periods = 35,064` with a headline series of that length, while the
    break-even receipt's costing block says 35,063 and the prose follows it. **I first wrote an
    explanation — "35,064 boundaries over 35,063 scored periods" — and then deleted it, because I
    had invented it rather than measured it.** The comment now states both values, that the
    artifacts disagree, and that we could not establish which is canonical. No quantity depends on
    it.

- **v1.6.37 (2026-08-15, ★ WITHDRAWN: THE "PERCENTILE CLEARS, NORMAL DOES NOT" DISAGREEMENT
  DOES NOT EXIST. IT COMPARED A ONE-SIDED BOUND AGAINST A TWO-SIDED ONE.)**
  - **WHAT WAS PUBLISHED AND IS NOW WITHDRAWN.** §7 v1.6.36 and the paper stated that *"the
    pre-registered percentile interval clears zero at +0.0454; a symmetric normal interval would
    not (−0.1545)"*, and presented that disagreement as the measure of the effect's strength.
    **There is no disagreement.**
  - **VERIFIED FROM SOURCE, NOT FROM THE CORRECTION.** `src/trikaal/eval/paired_bootstrap.py:36`
    reads `BOOT_ALPHA = 0.05  # §3a: one-sided level (the CI lower bound is the α-quantile)`. **The
    pre-registered bound is ONE-SIDED at α = 0.05.** At that level the percentile bound (+0.0454)
    and its normal counterpart (**+0.0307**, = Δ − 1.6449·se) **BOTH CLEAR AND AGREE**. The
    −0.1545 is a **two-sided 95%** bound — a STRICTER test than the one pre-registered — and
    comparing it to a one-sided quantity **manufactured a conflict that does not exist**.
  - **WHAT SHIPS NOW, EVERY BOUND WITH ITS LEVEL BESIDE IT.** Δ = **+0.9972**, se_boot **0.5876**,
    **t = 1.6971**. Pre-registered **one-sided α=.05**: percentile **+0.0454** / normal **+0.0307**
    — **CLEARS**. **Two-sided 95%**: lower **−0.1545**, **p = 0.0897** — **DOES NOT**. The honest
    sentence, and the one in the paper: *the effect is real on the instrument declared in advance,
    and it is not comfortably large.*
  - **THE RULE THIS INSTANTIATES: A BOUND WITHOUT ITS LEVEL IS THE DETACHABLE-QUALIFIER FAILURE,
    IN NUMERIC FORM.** The context-stripping rule has so far been about captions and figures; this
    is the same defect inside a statistic. "+0.0454 clears zero" and "−0.1545 does not" are both
    true and describe **the same effect at two different levels**; quoted without their levels they
    read as a contradiction, and one of them will be the one that travels. Every bound in §7 and
    Appendix E now names its level and its sidedness.
  - **★ MY SHARE OF THIS, STATED PLAINLY.** The instruction to publish the comparison came from the
    supervisor and was wrong, but **I published it with the source open in the same session** —
    `BOOT_ALPHA` is documented as one-sided on the line that defines it, and I had already read
    that file's neighbours to verify the bootstrap pins. I verified the NUMBERS (t and −0.1545 both
    reproduce exactly) and did not verify the **LEVEL THEY WERE COMPUTED AT**, which is the one
    thing that made them incomparable. *Checking that a number is correct is not checking that it
    is the right number.* That is a distinct failure mode from the ones already in docs/ENGINEERING.md and it
    is why the level now travels with every bound.
  - **UNCHANGED FROM v1.6.36 AND RE-CONFIRMED:** 3 of 3 seeds; the bias explanation predicting the
    WRONG SIGN for seeds 0 and 2; the V_p substitution caveat dropping for seed 4 ONLY; the exact
    benchmark having flattered the deflationary explanation rather than the model; and the
    ~$4.48 spend correction.
  - **THE CROSS-ENVIRONMENT REPRODUCTION NOW HAS ITS OWN RECEIPT**
    (`runs_manifest/m6_cross_environment_reproduction.json`), harvested out of the bias test's
    control arm because reproducibility evidence buried inside a bias test is invisible where it
    matters. §8.1 cites it and states its limits. **THE RECEIPT NUMBERS THREE; THE PAPER STATES
    FOUR.** Numbered: (1) not bit-exactness — the 2.587e-04 residual is the float32
    reduction-order effect invariant 7 already concedes deterministic attention does not remove;
    (2) nothing about TRAINING reproducibility, where the same-seed basin-hopping finding lives;
    (3) one arm, one seed. **The fourth is readable from the receipt's own `environments` block and
    is not in its numbered list: BOTH sides ran `attention_mode: sdpa_deterministic` with
    `deterministic_algorithms: true`, so the result establishes NOTHING about the default path.**

- **v1.6.36 (2026-08-15, ★ SEED 4 RE-RUN ON THE SPECIFIED BENCHMARK: THE CLAIM BROADENS TO 3 OF 3
  AND THE MARGIN IS 0.05 IR UNITS. BOTH FACTS TRAVEL TOGETHER OR NEITHER SHIPS.)**
  - **VERIFIED AT ENTRY-TIME AGAINST `runs_manifest/m6_skill_vs_bias_exact.json`.** On the names
    seed 4 ACTUALLY TRADED: benchmark **+0.1474** (was +0.4282 on the evaluated set), bias-only
    share **37.4% → 12.9%**, Δ **+0.9972**, 95% lower bound **+0.0454**. The V_p benchmark had been
    **flattering the deflationary explanation, not the model** — seed 4 traded LOWER-drift names
    than the average symbol. Both outcome branches were pre-declared by the supervisor BEFORE the
    run (`.PRE_DECLARED_READING` names the consequence for excludes-zero, straddles-zero AND
    negative), so the reading was not chosen after the number.
  - **★ THE MARGIN IS THE FIRST THING STATED, NOT THE LAST — REPORTING RULE.** Δ is **t = 1.70**
    (0.9972 / 0.5876). The pre-registered **percentile** interval clears zero at +0.0454; a
    **symmetric normal** interval at the same SE would **NOT** (−0.1545). **BOTH ARE IN THE PAPER,
    the adverse one first.** A referee computes t in ten seconds; a paper that already reported it
    is in a different position from one that did not. "Excludes zero" is TRUE and "comfortably
    positive" is FALSE, and quoting +0.9972 without the bound would misrepresent it.
  - **THE PERMITTED CLAIM, BROADENED AND STILL BOUNDED:** *directional bias times sample drift does
    not explain the gross performance of ANY of the three seeds — for two the bias alone LOSES
    money while the models earn (wrong sign), and for the third the exact benchmark reproduces
    12.9% with the remainder excluding zero.* It remains **NOT** a claim of economically useful
    skill: §7's break-even subsection puts the gross edge an order of magnitude short of fees, and
    the paper says so in the same breath.
  - **THE SUBSTITUTION CAVEAT NOW SPLITS, AND THE TABLE CARRIES THE SPLIT INSIDE ITSELF.** It
    **DROPS for seed 4** (that row is the specified benchmark) and **STANDS for seeds 0 and 2**,
    not re-run by ruling because their inversion is **structural**: a short-biased model in a
    rising sample cannot have its positive edge explained by drift on ANY benchmark. Verified
    rendered on the same page (31) as the numbers, per the context-stripping rule.
  - **A CORRECTION TO MY OWN PRIOR ENTRY.** §7 v1.6.35 and the §7 prose shipped *"a GPU pass would
    cost about $3.57 per seed ... and was not run"*. **It was then run**, for the one seed whose
    answer was open, at **~$4.48** (box 47715433, RTX 4090 @ $0.3322/h, 13.5 h, of an $8 cap). The
    CPU-infeasibility finding (193 h / 579 h) **stands and is the reason** the GPU path was taken.
    Corrected in place rather than left standing.
  - **AND A REPRODUCIBILITY RESULT THAT QUALIFIES AN INVARIANT-7 CLAIM RATHER THAN MERELY
    SUPPORTING IT.** The exact run's control arm re-scored a banked predictor on a **different
    torch build (2.5.1+cu124 vs 2.12.1+cu130) and a different physical 4090, weeks later**: active
    set **IDENTICAL (34,989)**, gross series max|diff| **2.59e-04**, corr **0.99999986**, IR within
    **3.5e-04**. **This is a CLOSE reproduction, NOT a bit-exact one.** It does not contradict the
    bit-exactness claim — a different build is a different provenance identity by construction,
    since torch version and platform ABI are both identity keys — but it **bounds what that claim
    never covered and had never been tested**: cross-environment replay reproduces to about four
    decimal places, not to the bit. Written into §8.1 and cross-referenced from §7's determinism
    subsection, in that shape, rather than quoted as evidence of bit-exactness.

- **v1.6.35 (2026-08-14, THE GROSS EDGE SURVIVES AN ADVERSARIAL BIAS TEST ON TWO OF THREE SEEDS —
  AND THE CONCERN IS *INVERTED* THERE, NOT MERELY ABSENT. THE SKILL CLAIM NARROWS AND HOLDS.)**
  - **WHY THE TEST EXISTS.** `c_break > 0` on every unit is the same statement as a positive gross
    IR. The cheapest deflationary explanation is **directional bias × sample drift**, and an edge
    never tested against it is an assertion. `runs_manifest/m6_skill_vs_bias.json` runs it: each
    model's own **modal direction held constant on the same active periods**, paired block
    bootstrap (B = 10,000, block 188, seed 20260704).
  - **THE SAMPLE DRIFTED UP**, +7.900e-06 per 15-min period; always-long returns annualized IR
    **+0.3704**. So a model that merely leaned the right way would post a positive gross edge with
    no timing content.
  - **★ SEEDS 0 AND 2 INVERT THE CONCERN — THIS IS THE STRONG HALF AND IT LEADS THE PROSE.** Both
    lean **SHORT** (67.7%, 65.5%) in a sample that **rose**, so their lean **alone LOSES**
    (−0.7607, −1.2640) while the models earn **+1.2665** and **+2.1715**. Drift capture does not
    merely fail to explain them — **it predicts the OPPOSITE SIGN**. A harder test than the one
    asked for, passed.
  - **SEED 4 IS LIVE AND IS WRITTEN AS SUCH.** Long on **92.7%** in a rising sample; holding long
    reproduces **37.4%** of its +1.1443, and the remainder (+0.7161) has a 95% lower bound of
    **−0.2410 — INCLUDES ZERO**. **We cannot exclude that most of seed 4's gross edge is drift.**
  - **ONLY SEED 2 CLEARS ZERO AT 95%** (Δ lower bound +1.5574). Seeds 0 and 4 do not (−0.5530,
    −0.2410).
  - **THE PERMITTED CLAIM, NARROWED TO THE EVIDENCE:** *directional bias times sample drift does
    not explain the gross performance of two of three seeds, where that bias would have LOST
    money; on the third it accounts for 37% and the remainder is not distinguishable from zero;
    ONE of three demonstrates timing content at 95%.* **Any sentence of the form "the models have
    real predictive skill" is too broad to ship** and none appears in the paper.
  - **★ THE CAVEAT IS BLOCKING AND TRAVELS INSIDE THE TABLE, NOT BESIDE IT.** The benchmark
    averages over **V_p — every symbol with a valid decision — NOT over A_p, the names each model
    actually traded.** A_p is unrecoverable: `write_cell_eval_artifact` persists only the pooled
    net series, and the execution filter **does** bind (activity 0.0926/0.0193/0.4702, moving with
    κ), so traded ≠ scored. **The direction of that difference is unknown and NO bound is claimed.**
    Placed inside the table's own `minipage` and verified on the same rendered page (31).
  - **THE EXACT A_p VERSION LANDED WHILE THIS WAS BEING WRITTEN — AS AN INFEASIBILITY RESULT.**
    Authorized as "$0, local CPU, hours" under the standing bound *ship V_p rather than delay*.
    **Measured: it does not complete.** 494.9 ms/decision on CPU (best observed; threading does not
    help — the work is compute-bound over 15 rollout steps at seq_len 512) → **193 h for seed 4
    alone, 579 h for all three.** Subsampling does not rescue it: a 1,000-period window costs ~6 h
    and widens the bootstrap SE ~5.9×, on an interval that already straddles zero. **GPU would cost
    ~$3.57/seed (~$10.7 all three) at the money run's measured marginal rate — NOT RUN**, economics
    spend deferred; reported so the omission is **costed rather than assumed impossible**. V_p ships
    per the bound.
  - **AND THE BUILDER'S OWN COSTING ERROR IS CARRIED, NOT DROPPED.** The first receipt priced
    tokenization (~1 h) and left "plus the rollouts" unquantified; **the rollouts are the entire
    cost.** The cheap term was priced and the dominant one implicitly set to zero — *the same error
    class as the 7.80 h eval estimate that omitted 47.2% of the leg*, made again, in a receipt.
    This is why the paper states the measured rate rather than an estimate.

- **v1.6.34 (2026-08-14, ★ THE PUBLISHED PARAMETER COUNT EXCLUDES A THIRD OF THE MODEL. THE
  HEADLINE FIGURE BECOMES THE REALIZED TOTAL, AND C-19'S DISCLOSED CONFOUND SHRINKS.)**
  - **VERIFIED AT ENTRY-TIME FROM BOTH CHECKPOINTS, NOT FROM THE SUMMARY.** Summing every tensor
    in the shipped `predictor.pt`: **BSQ = 31,725,568** (`runs_cloud/rescue/r0/cell1_bsq_ohlcv_seed0`)
    and **FSQ = 31,795,200** (`runs_cloud/ckpt_cell4_seed0`). The `mtp.*` group is **10,493,952 in
    BOTH**, and `TOTAL − MTP` reproduces **21,231,616 / 21,301,248 exactly**. The quoted figure was
    never the model; it was the model **minus its multi-token-prediction heads**, which are trained,
    checkpointed and shipped. MTP is **33.08%** of the BSQ total and **33.00%** of the FSQ total —
    a third of the artifact, absent from every document.
  - **THE CODE IS NOT AT FAULT AND THAT MATTERS FOR THE FIX.** `cells.py:134` gates
    `model.num_backbone_params()` — backbone, excluding MTP — against
    `PINNED_BACKBONE_PARAMS = 21,301,248` (`conformance.py:48`), and the assertion message says
    *"backbone base params"*. **The pin is correctly named and correctly scoped.** The defect is
    that the PAPER, docs/ENGINEERING.md and the design docs promoted a BACKBONE count to a MODEL count. So
    the pin stays exactly as it is; what changes is what we call it.
  - **THE RULING (supervisor, 2026-08-14): quote the REALIZED TOTAL as the model size, with the
    backbone-excluding-MTP figure as a NAMED SECONDARY.** docs/ENGINEERING.md's own instruction is *"quote
    the realized number"*, and 21.2M is not it. Applied across §2, §4, §7, both appendices and the
    conformance table; the pin's description is corrected to *"backbone excluding the MTP heads"*
    rather than *"the realized parameter count"*.
  - **★ C-19 SURVIVES AND IMPROVES, CHECKED BEFORE IT WAS WRITTEN.** MTP is IDENTICAL across arms
    (10,493,952 each), so the arm gap is **69,632 either way** — but as a fraction it is
    **0.219% OF THE REALIZED MODEL**, not the 0.327% of the backbone we have been disclosing. **The
    confound is SMALLER than disclosed**, and it is now stated with its denominator, because a
    percentage travelling without one is the context-stripping shape.
  - **★★ AND THE SAME OMISSION PRODUCED TWO WRONG CORRECTIONS IN THE SAME DIRECTION — this is the
    part worth keeping.** Both figures involved were themselves corrections, and both OVERSHOT:
    (1) `~27M` → `21,301,248`, justified as "a target must give way to a realized count" — but the
    realized count is 31,795,200, so **|27.0 − 31.80| = 4.80M against |27.0 − 21.30| = 5.70M: the
    figure we discarded as imprecise was CLOSER**;
    (2) the hedge "the training budget is a small fraction of compute-optimal" was withdrawn for
    "20.00 tokens per parameter, which is compute-optimal" — but **426,033,152 / 31,795,200 =
    13.40**, about two-thirds of the reference ratio, so **the withdrawn hedge was again the closer
    statement**. One root cause, two corrections, both moving the number FURTHER from the truth
    while making it sound MORE precise. Recorded in §7 rather than quietly fixed.
  - **WHY IT MUST CLOSE BEFORE THE HF RELEASE.** Publishing the weights makes the total checkable
    with one line of torch. A reader who runs it must find the discrepancy **already described by
    us**, not discover it.
  - **OUT OF SCOPE FOR THIS ENTRY AND FLAGGED, NOT TOUCHED:** `docs/ENGINEERING.md` (invariant text and the
    architecture section), `docs/m6_design.md`, `docs/paper_skeleton.md`, `docs/v2_and_limitations.md`
    and the audit docs all still carry 21.2M/21.3M as the model size. They are outside the writer's
    domain (`paper/` and `docs/m6_prereg.md`); the class is named here so the sweep is not mistaken
    for complete.

- **v1.6.33 (2026-08-14, THE PRE-REGISTERED BREAK-EVEN INSTRUMENT CANNOT ANSWER THE QUESTION IT
  WAS BUILT FOR — ITS GRID FLOOR IS EXACTLY THE FEE IT IS COMPARED AGAINST — AND THE PAPER SAID
  IN FOUR PLACES THAT A QUANTITY WAS UNMEASURED WHICH THREE COMMITTED ARTIFACTS RECORD.)**
  - **THE ADMISSIBILITY RULING.** `runs_manifest/m6_horizon_break_even.json` recomputes a
    pre-registered §4 headline component (break-even cost, named at prereg:170-171 inside the
    cost-stress curve) with a **NON-PRE-REGISTERED instrument**, and its own `status` field states
    that admissibility is the supervisor's ruling and not the builder's. **The supervisor ruled it
    admissible by directing it into §7 Limitations (2026-08-14), and it enters the paper as a
    limitation rather than as a result.** Invariant 5 is untouched: the headline metric is still
    cost-aware net IR, and no gross quantity replaces it. Both values ship per unit — the
    pre-registered `-inf` beside the bisected root.
  - **DEFECT 1, THE FLOOR.** `harness.py:45` pins `FLAT_COSTS = (0.0010, 0.0020, 0.0030)` and
    `metrics.break_even_cost` returns `-inf` when IR ≤ 0 across the whole grid. On all three
    real-lake units it returns exactly that. **The grid's floor, 0.10%, is precisely the realistic
    round trip the comparison is against**, so the pre-registered instrument bottoms out at the
    threshold where the question begins. It cannot distinguish "break-even is just under a round
    trip" from "break-even is forty-three times under it" — and the true answer is the second.
    This is a limitation of our own design, found in our own instrument.
  - **DEFECT 2, AND IT IS THE WORSE ONE.** `CellScore.break_even` **is** computed at eval time
    (`harness.py:221`, stored at `:280`) and `verdict.write_cell_eval_artifact` **does not persist
    it**. Verified: the string `break_even` appears nowhere in any of the three shipped money
    artifacts. So the design computes the quantity, discards it, and falls back to a grid that
    bottoms out at the comparison threshold. Two defects on one quantity, in the same pass.
  - **THE MEASUREMENT, VERIFIED AT ENTRY-TIME AGAINST THE RECEIPT.** At the pinned h=15 on the
    headline money grid, c_break = **0.00232%, 0.00538%, 0.02082%** across seeds 4, 0, 2 — **4.8x
    to 43x short** of a 0.10% round trip. It survives sampling error: the best 95% upper bound is
    0.0396%, **still 2.5x short**, and two of the three intervals contain zero.
  - **THREE THINGS THE SUMMARY HANDED TO ME DID NOT CARRY, ALL FOUND BY READING THE RECEIPT.**
    (1) **IT IS NOT A HORIZON RESULT.** The receipt's own `scoping_correction` retracts the
    "three points" framing: there is **ONE measured point and TWO exact bounds**. At h=5 and h=60
    the only exact statement is c_break < 0.30%, which sits **above** the 0.10% fee and therefore
    carries no constraint on the fee question; the inference was attempted and **REJECTED as
    UNINFORMATIVE** (admissible bands 66%-1140% wide). The file is named `m6_horizon_break_even`
    and that name invites exactly the reading its own contents forbid.
    (2) **THE INTERVAL EVIDENCE IS NOT INDEPENDENT OF THE IR_gross EVIDENCE — IT IS THE SAME
    STATEMENT.** IR_gross **is** the t-statistic of c_break (35,064 periods against
    periods_per_year(15) = 35,040), so "two of three intervals contain zero" is the same fact as
    the frozen predeclaration's block-bootstrap IR_gross intervals containing zero for seeds 0
    and 4. Reporting them as two corroborating findings would be double-counting one.
    (3) **THE 0.10% THRESHOLD IS SUPERVISOR-SUPPLIED AND EXPLICITLY NOT VERIFIED IN THE RECEIPT**
    (`fee_reference.source`). The direction saves it: a taker round trip **excludes** spread,
    impact, funding and slippage, so the true threshold is **above** 0.10% and the finding is
    conservative. Stated, not assumed.
  - **THE STRONGEST FORM IS ASSUMPTION-FREE AND IS THE ONE THE PAPER USES.** c_break = mean gross
    return per active period, **identically** — proven numerically per unit, with the check proven
    capable of failing first (fixture-discrimination rule). So clearing a round trip of c requires
    gross edge per active period ≥ c, and **that bar is the same at every horizon**; what changes
    with h is how much edge a period can contain, not the bar. Under linear scaling the measured
    h=15 edge reaches 0.10% only after **72-646 minutes** of holding — a LOWER bound, since real
    decay is sub-linear.
  - **THE TRANSFER LIMIT IS ENFORCED, NOT ASSERTED.** Micro-arm artifacts exist (`micro`,
    `micro_shuffled`) but every non-money artifact set is REFUSED by the production
    `load_cell_evals` — no `index.json`, wrong schema, or incomplete matrix. No transfer check is
    possible at $0. That is a stronger statement than "cell 1 only" as a claim, and it is why the
    finding is **WEAK IN BOTH DIRECTIONS**: a horizon that cleared fees would be suggestive, not
    proof, and finding none does not license "no horizon is tradeable" — the arm the headline
    claim is about was never scored.
  - **★ THE CORRECTION THIS FORCED, AND IT IS THE REASON THE ENTRY IS LONG.** Verifying the above
    surfaced a **FALSE CLAIM SHIPPED IN FOUR PLACES**: §7's subsection title *"Whether the
    execution filter binds on real data is unmeasured"*, the identical sentence in §4 and §7 that
    *"no evaluation artifact from a real-lake run retains a decision-activity value"*, and
    Appendix E.3's row *"filter activity on real data — pending — no artifact records it"*. **All
    three real-lake artifacts record it**, at `mu_diag.activity_decisions`: **0.0926, 0.0193,
    0.4702** at κ*=3, and 0.339/0.176/0.916 at κ=1. The whole class is swept and corrected, not
    the site that was pointed at (class rule).
  - **AND THE MEASUREMENT REFUTES THE BRANCH WE PLANNED FOR.** §7 pre-committed that "the
    near-zero activity branch is therefore the one to plan for", on the argument that a calibrated
    forecast has std(μ̂) = IC·std(y) = 0.027 × 0.003504 = 9.46e-05. Measured std(μ̂) is
    **0.005376, 0.005331, 0.010373 — 56x to 110x that value**. The models are not calibrated; they
    are grossly over-dispersed, they trade, and they lose. That is the *other* branch, the one §7
    argued was less likely, and it is now the measured one on cell 1. The prediction was wrong in
    the direction we said was safer to plan for, and §7 now records that rather than the
    prediction.
  - **THE COMPARABILITY PROHIBITION TRAVELS WITH ALL OF IT** and the cell-1-only scope goes INSIDE
    any caption or table note that carries a number from this entry, never beside it
    (context-stripping rule).

- **v1.6.32 (2026-08-12, ITEM E's ONCE-ONLY λ RE-DERIVATION IS SPENT AND RETURNED NOTHING.
  `clearing_lambda: null`, 0 of 18 configs. THE DISPOSITION IS FINAL.)**
  - **THE SPEND IS RECORDED BECAUSE THE ALLOWANCE WAS ONCE-ONLY.** §7 v1.5 item E permitted λ to be
    re-derived **at most ONCE**, by the pinned formula, on a slice carved from the END of the TRAIN
    region. That allowance is now **used**. `runs_manifest/m6_lambda_sweep.json`: λ ∈ {5, 8, 12} ×
    3 seeds × 2 arms = **18 configs, 0 clearing**, criterion *"0.9 on all six dims, both arms, all
    three seeds"*, `disposition: ITEM E DISPOSITION IS FINAL — no further attempt`. The receipt
    carries its own stopping rule: *"If no listed config clears, item E's disposition is FINAL. No
    fourth lambda, no widening after a near-miss."* The money pin is untouched —
    `PINNED_MICRO_POINT_WEIGHT` remains 3.0, and `run_cell` raises if a calibration λ is set on a
    money run.
  - **THIS IS SECONDARY/EXPLORATORY BY PRE-REGISTRATION AND IS NOT WRITTEN AS ANYTHING ELSE.**
    Item E binds it in advance: even a *clearing* config would have returned the ablation only as
    SECONDARY/exploratory at N 60 → 120. Nothing cleared, so no ablation returns at all. **The
    primary remains the mechanism finding.**
  - **★ THE COMPARABILITY PROHIBITION, WHICH TRAVELS WITH EVERY ONE OF THESE NUMBERS.** These
    configs were trained with the last 10% of the train region held out; the λ=3 values in the
    gate-firing run were not. The receipt carries this as a required field,
    `NOT_COMPARABLE_TO_THE_GATE_FIRING_RUN`: *"The lambda=3.0 values 0.8223/0.7528 were trained on
    the full train region INCLUDING this slice and measured off-slice-held-out. No delta against
    them is a measured quantity."* **No difference between the two sets is a measurement.** The
    criterion here is ABSOLUTE (0.9) and needs no baseline, which is why the sweep is readable at
    all. Where any of this reaches a figure or table, the prohibition goes **in** the caption.
  - **WHAT THE SWEEP MEASURED — three results, each re-derived from the artifact at entry-time
    rather than transcribed, and two of the briefed figures did not survive unchanged.**
    1. **THE KNOB DOES NOT REACH THE FAILING CHANNEL.** On cell 4, dim 8
       (`signed_count_imbalance`), the seed mean moves **0.7140 → 0.7219, +0.0079**, across a
       **2.4×** increase in λ. Stated without mixing bases: **0.1781 of the gap to 0.90 still
       remains at the top of the swept range**, and the movement is 4.2% of what was required from
       the bottom of it. *(The brief's "4.4%" divides the λ=5→12 movement by the requirement
       measured from λ=12; the two bases differ. Both underlying numbers are right.)*
    2. **λ REDISTRIBUTES WITHIN THE MICRO BLOCK; IT DOES NOT BUY FROM PRICE.** Cell 4 OHLCV
       reconstruction MAE across λ = **0.12814 / 0.12678 / 0.12530** — flat to slightly *better*
       as λ rises — while inside the micro block TFI rises **+0.0307** and the magnitude dims give
       ground: **−0.0045, −0.0063, −0.0012** on dims 10, 11, 12, while dim 9 rises +0.0025.
       *(Correction to the brief, since confirmed by the supervisor: it is THREE of the four
       magnitude dims that fall, not four.)*
       **AND CELL 5 IS THE CLEANER DEMONSTRATION THAT PRICE IS UNTOUCHED, verified at
       entry-time.** Its OHLCV MAE is **0.17442 → 0.17444, +0.00002 end to end** (the λ=8 midpoint
       dips to 0.17313, so this is endpoint flatness, not a monotone line) while its two signed
       dims rise **MORE** than cell 4's: **+0.0405 and +0.0248**.
       **BUT THE "REDISTRIBUTION WITHIN THE BLOCK" FRAMING DOES NOT TRANSFER TO CELL 5 — on that
       arm ALL SIX micro dims RISE.** There is no internal trade-off there to observe. The honest
       joint statement is sharper than either half: **the weight takes nothing from price on
       EITHER arm**, and the internal trade-off appears only on the arm with no headroom (cell 4,
       TFI already at 0.8268) and vanishes on the arm that has it (cell 5, TFI at 0.7387). That is
       what saturation predicts, and it is now stated that way rather than as a blanket
       redistribution claim.
    3. **THE PLACEBO IS ITSELF AN EVICTION EXPERIMENT.** Cell 5's micro block is shuffled and so is
       independent of its own bar's price. The cell4-minus-cell5 legibility gap is **0.0655 on the
       two SIGNED dims against 0.0179 on the four MAGNITUDE dims — 3.66×**, averaged over all 18
       configs. *(The brief's 0.0657 / 0.0259 / 2.5× is the **λ=5 slice only**; I reproduce it
       exactly at that λ. Across the full sweep the separation is larger, so the sweep-wide figure
       is the one recorded.)* A channel stripped of covariance with its neighbours is
       systematically harder to encode — **the paper's own mechanism, reproduced inside the control
       arm.**
  - **DIRECTION: this amendment records a SPEND, not a change.** No threshold, pin, seed,
    enumeration or gate value moves. The once-only allowance is consumed and the disposition it
    was attached to is unchanged.

- **v1.6.31 (2026-08-12, ★★ THE MICRO-LEGIBILITY GATE FIRED ON REAL DATA. ITEM E TAKES EFFECT.
  THE PRIMARY IS NOW THE MECHANISM FINDING. ★★ No threshold moved. No rescope. No re-run.)**
  - **THE FIRING.** `runs_manifest/m6_micro_legibility_stop.json`, git_commit `e307309`.
    `stage2_entered: false`, `artifacts_produced: 0`, `ALL_FILES_VERIFIED: true` over 6 files with
    byte totals and sha256 reconciled against on-box production time. Both gated arms refused
    before any Stage-2 spend, at `min_acc = 0.90`, n = 150,000 per dimension, 40 symbols,
    stratified by symbol with an 80/20 blocked-in-time split within each symbol.

    | dim | feature | cell 4 acc | cell 5 acc | base rate + |
    |---|---|---|---|---|
    | 7 | `TFI` | **0.8223** | **0.6135** | 0.4869 |
    | 8 | `signed_count_imbalance` | **0.7528** | **0.5896** | 0.4665 |
    | 9 | `trade_count` | 0.8975 | 0.8916 | 0.4725 |
    | 10 | `mean_trade_size` | 0.9076 | 0.8525 | 0.5462 |
    | 11 | `trade_size_dispersion` | 0.8962 | 0.8691 | 0.4073 |
    | 12 | `large_trade_share` | 0.9320 | 0.8685 | 0.8577 |

  - **★ ITEM E, WRITTEN PRE-DATA, PRE-WROTE THIS DISPOSITION.** §7 v1.5 item E reads: *"a firing
    is the **MODAL prediction of our own finding**, and simultaneously the ablation's blocker and
    the mechanism's strongest real-data evidence. Rules: gate fires → **STOP and report**
    (unchanged); **the PRIMARY becomes the mechanism finding**"*. **The primary is now the
    mechanism finding.** λ may be re-derived at most ONCE, by the pinned formula, on a slice
    carved from the END of the TRAIN region — never block 0 — and item E binds the consequence in
    advance: the re-derived λ counts as an additional configuration in the clause-5 multiplicity
    (**N 60 → 120**), and **any ablation under a re-derived λ is reported SECONDARY/exploratory,
    never a primary**, because the contingency can only ever help the ablation.
  - **★ WHY THE FIRING WAS PREDICTABLE RATHER THAN SURPRISING — the unflattering fact, stated
    plainly because it was in the repository from the day the gate was pinned.** The pinned
    λ = 3.0 was calibrated on **ONE dimension**: `scripts/m6_canary.py:531` probes
    `d0["x"][:, 9]` and nothing else. Dim 9 is `trade_count` — **a magnitude channel, not a
    signed one** — so the calibration dimension is not representative of the dimensions the gate
    governs. It was calibrated under a **restated** rule (mean ≥ 0.9 AND min ≥ 0.85), and its own
    receipt records the shortfall verbatim: *"no searched (lambda, beta) clears 0.9 on all 3
    seeds; the gate constant sits inside the instrument's seed-noise band (~±0.03) at the
    achievable ceiling"*. The standing gate demands 0.9 on **all six** dims in **one** run. That
    gap between what was calibrated and what is demanded sat unexamined in this repository, and
    nobody named it until the gate fired.
  - **★ THE SCIENTIFIC CONTENT, WHICH IS NOW THE PAPER'S SPINE.** Measured on the shortfall
    against 0.90: **97.3% of cell 4's total shortfall and 83.5% of cell 5's sit in dims 7 and 8** —
    `TFI` and `signed_count_imbalance`, **the two SIGNED channels**, which are exactly what
    invariant 1 scopes the claim to. The four magnitude dims essentially pass (9 and 11 miss by
    0.0025 and 0.0038; 10 and 12 clear). **THE TOKENIZER PRESERVES MICROSTRUCTURE THAT DUPLICATES
    OHLCV AND LOSES MICROSTRUCTURE INDEPENDENT OF IT.** That is *reconstruction buys variance and
    covariance, never independence* — on real data, with the evicted features named.
  - **THE PLACEBO DIFFERENTIAL IS THE ROBUST FORM OF THAT CLAIM, AND IS STRONGER THAN THE RAW
    SHORTFALL.** Against the majority-class baseline `max(p, 1−p)`, cell 4's lifts are: dim 7
    **+0.3092**, dim 8 **+0.2193**, dim 9 +0.3700, dim 10 +0.3614, dim 11 +0.3035, dim 12
    **+0.0743**. Under the shuffle, dims 7 and 8 **collapse to +0.1004 and +0.0561** while the
    magnitude dims hold at +0.3641, +0.3063 and +0.2764. **Shuffling destroys the signed channels'
    recoverability and leaves the magnitude channels intact** — a placebo-controlled statement of
    the same finding that does not depend on where the 0.90 line happens to fall.
  - **A CAVEAT THE RAW READING HIDES, RECORDED BECAUSE THE DEGENERACY RULE DEMANDS IT.** Dim 12
    `large_trade_share` has a base rate of **0.8577**, so its majority-class baseline is 0.8577 and
    its 0.9320 is a lift of only **+0.0743** — the weakest recovery of any dimension in cell 4, yet
    one of only two that clear the threshold. **The gate reads raw sign accuracy, not lift, so an
    imbalanced dimension passes cheaply.** This does not change the disposition — the gate fired on
    other dimensions — but any downstream reading of "dim 12 passed" must carry it.
  - **NO DRY-RUN PRE-FLIGHT COULD HAVE CAUGHT THIS.** `scripts/m6_money_run.py:423` sets
    `micro_legibility_min=None if args.dry_run`, with the comment *"A dry run cannot clear a gate
    that needs a trained tokenizer, and must not pretend to."* Neither P1 nor P2 reached a gated
    arm. **Pre-flight green implied nothing about this gate**, and the execution rule's own claim —
    that a component whose first execution is on rented hardware is untested — is confirmed here at
    full cost.
  - **SCOPE, as the paper already states it.** One budget point; λ fitted on a synthetic proxy and
    on a single unrepresentative dimension; this reconstruction objective and this architecture.
    **It does NOT establish that no tokenizer can make microstructure per-bar legible** — only that
    this one, built to and weighted for it, does not at this budget.
  - **WHAT THIS ENTRY DOES NOT CITE, FOR THE SECOND CONSECUTIVE AMENDMENT.** The brief directed
    citing an independent auditor who reached this reading unprompted and who located §7 v1.5
    item E when neither the builder nor the supervisor had. **No such report is committed.** An
    exhaustive search returns item E in exactly three places — `docs/m6_prereg_v1_5_drafts.md`,
    `src/trikaal/train/gates.py` and `tests/train/test_micro_legibility_gate.py` — and no auditor
    document, no independence statement and no conditions anywhere. The reading above stands on
    the receipt and on item E's own text, neither of which needs corroboration to be checkable.
    **If the report lands it is added as a dated addendum, with its conditions.**

- **v1.6.30 (2026-08-12, ★ THE CODEBOOK GATE IS RESCOPED TO FSQ. ★ THE FIRST POST-DATA
  AMENDMENT IN THIS LOG — read the disclosure below before the justification.)**
  - **★ WHAT WAS KNOWN WHEN THIS WAS MADE, STATED FIRST BECAUSE A HOSTILE REVIEWER OPENS HERE.**
    The run had started. **Two cell-1 evaluations existed at the pinned budget** —
    `runs_cloud/results/r0/cell1_seed0_eval.json` and `runs_cloud/results/r2/cell1_seed4_eval.json`,
    both `steps_stage2 = 26003`, both `dry_run: false`, both stamped `git_commit 95fc017`. Every
    prior entry in this log was written before any unit existed; this one was not. That difference
    is the reason the disclosure leads.
  - **THE INFORMATION THOSE TWO ARTIFACTS LEAK, AND IN WHICH DIRECTION.** They are not bare
    diagnostics: each carries a full `headline_series` of 35,064 periods and twelve
    `val_ir_by_kappa_by_h` readings, **and every one of those readings is negative**. The §5
    fallback claims **IR(2) − IR(1)**, FSQ minus BSQ (§7, prereg :774). A cell 1 that looks poor
    makes that difference **more likely to be positive**, which makes a fallback claim **more
    likely to be attempted and more likely to succeed**. We record the direction and deliberately
    do not transcribe the magnitudes here, because repeating them would spread the leak further
    than disclosing it requires.
  - **★ PRE-COMMITMENT, MADE NOW AND BINDING: any §5 fallback claim arising from this run is
    flagged, in the paper and in the verdict manifest, as EVALUATED UNDER A POST-DATA AMENDMENT.**
    That flag is not contingent on the fallback's outcome and may not be dropped if the fallback
    fails to fire.
  - **THE COUNTERFACTUAL, WHICH IS WHAT SEPARATES THIS FROM GOALPOST-MOVING. Had BSQ returned
    0.96 and passed the gate untouched, restoring the spec's scope would still have been
    correct**, because the justification is a reading of the specification and not a reading of
    the data. The measurement below determined the *timing* of this amendment. It did not
    determine its *content*.
  - **THE CHANGE.** The codebook block stays **REQUIRED and finite on all 25 artifacts** — a
    missing, empty or non-finite block still refuses the whole verdict. The **0.95 threshold is
    retained, scoped to the FSQ cells 2, 4 and 5**. The **BSQ cells 1 and 3 become
    REQUIRED-AND-REPORTED**: the number must be present and finite, is published, and does not
    gate. **Fail-closed on identity:** the quantizer is read from the **pinned cell registry**,
    never from the artifact's own `quantizer` field, and a cell id absent from the registry or an
    artifact whose self-declaration disagrees with the registry is refused. A self-declared field
    would be a lever for an artifact to relabel itself out of the threshold.
  - **WHY THIS IS SPEC RESTORATION, NOT RELAXATION — each verified against the source at
    entry-time rather than taken from the brief.**
    - **design `:1859`** — the ≥95% target appears inside the **FSQ Stage-1 report-metrics**
      paragraph and is scoped in its own words: *"codebook usage = fraction of **FSQ** codes used
      over the eval set (target ≥ 95% — FSQ rarely collapses, this confirms it)"*. The target was
      never written as a cross-quantizer minimum.
    - **design `:1154`** — *"The dead-code rate that plagues BSQ is **reported** for the **BSQ
      ablation arm** … itself **a result to report**."* Reported, not gated, and named as a result.
    - **design `:986`** — *"We monitor usage as a **health diagnostic** (§f), **not as a failure
      mode** to be engineered around."*
    - **`docs/ENGINEERING.md:13`** — *"If anything here conflicts with the spec, **the spec wins**."* The
      conflict is pre-committed in our favour, and not by this entry.
    - **THE ORIGIN OF THE OVER-BROAD GATE IS IN THIS LOG.** §7 v1.6.22(e) reads *"enforces the
      spec's own **≥95% utilization**"* — **with no quantizer scope**. That transcription dropped
      the scope the spec carried. The defect is ours and it is a transcription defect, not a
      design decision that is now being reversed.
    - **THE CODE CITES THE LINE THAT CONTRADICTS IT.** `verdict.py:302` emits *"dead-code
      collapse; **spec :1859**"* on every cell — citing, as authority for an all-cells threshold,
      the one line that scopes the target to FSQ.
    - **NO BSQ-SCOPED MINIMUM EXISTS ANYWHERE.** Verified across the design spec, `m6_design.md`
      and this pre-registration: every utilization mention is FSQ-scoped or concerns another
      gate. There is nothing to relax, because nothing was ever set.
  - **THE MEASUREMENT.** Same seed, same OHLCV arm, same 84,153,600 tokens per leg, same 26,003
    steps — only the quantizer differs. FSQ (cell 2 probe): coarse 1.0000 (891/891), **fine
    0.9951** (1219/1225), fine perplexity 795.88. BSQ at the pinned budget: **fine 0.8672**
    (888/1024, perplexity 255.75) on r0/seed0 and **fine 0.939453125** (perplexity
    302.81498823353724) on r2/seed4, both with **coarse utilization 1.0000**. **Neither BSQ unit
    is collapsed** — a collapsed codebook has perplexity near 1 against a 1024 vocabulary, and
    these sit at 256 and 303 with every coarse code in use. The gate would have refused the
    entire verdict over a non-collapse.
  - **THE GATE'S PASS PATH HAS NEVER PASSED A REALISTIC ARTIFACT.** Of the 25 evaluation
    artifacts on disk carrying a codebook block, **5 would pass both legs and 20 would refuse**.
    All five that pass are **FSQ cells (4 and 5) from the toy rehearsal at a non-pinned budget**.
    **No BSQ artifact has ever passed it, and no artifact at the pinned budget has ever passed
    it.** Its fixtures are invented: the healthy cases are 0.98 and 0.99, the collapsed case is
    0.01, and **no realistic BSQ reading — the 0.87 to 0.94 band the hardware actually
    produces — is exercised anywhere in its tests.** A gate whose pass path is reachable only by
    values no measurement has produced has not been tested against the world it governs.
  - **DIRECTION, RECORDED BEFORE THE OUTCOME IS KNOWN, per the standing direction-blind rule:
    this amendment LOOSENS** the artifact-admission surface for two of five cells. It is recorded
    as a loosening even though we hold it to be a restoration, because the reader is entitled to
    the mechanical fact independent of our reading of it.
  - **WHAT THIS ENTRY DOES NOT CITE, AND WHY.** The brief directed that an independent auditor
    reached the same reading without being told ours, and that the entry cite that finding and its
    conditions. **No such report is committed to this repository** — the only document describing
    the rescope is `runs_manifest/m6_codebook_gate_fix_proposal.md`, which is the builder's own
    proposal and contains no auditor citation, no independence statement and no conditions.
    Citing a corroboration whose artifact does not exist is the v1.6.28 failure exactly, and the
    justification above does not need it: it stands on four spec lines and a transcription defect
    in this log. **If the auditor's report lands, it is added here as a dated addendum, with its
    conditions.**
  - **CONSEQUENCE FOR THE PAPER'S TIMING CLAIMS.** §7 v1.6.29 and the paper's Appendix F both
    state that no evaluation artifact carries `steps_stage2 == 26003`. That was true when it was
    written and is **now false** — two do. Those claims are to be read as of their own date, and
    the paper is corrected accordingly rather than left to be discovered.

- **v1.6.29 (2026-08-11, ★ THE `platform` IDENTITY KEY IS SPLIT — WE CHANGED WHAT WE REFUSE ON,
  NEVER WHAT WE RECORD. Docs + a bounded src change. THIS IS A SCHEDULE DECISION.)**
  - **★ THE MOTIVE, IN PLAIN WORDS, AHEAD OF THE TECHNICAL ARGUMENT: WE LOOSENED A GATE BECAUSE
    THE RUN WAS TOO SLOW.** With the kernel inside the identity key, no two rentable boxes matched,
    so the fan-out was impossible and the run was a single box: **270.4 GPU-h serial = 11.3 days
    wall-clock.** Splitting the key makes the five-box fan-out purchasable and takes compute
    wall-clock to **54.1 h ≈ 2.3 days**, a 5.0× reduction. A reader should see that ordering, judge
    it, and find the technical argument sitting BESIDE it rather than IN PLACE OF it. The
    technical argument is real and is stated below; it is not why this was done now.
  - **THE TECHNICAL CLAIM, STATED AND NOT OVERCLAIMED.** A Linux kernel patch release does not
    participate in GPU floating-point computation: with the same GPU model, the same driver, the
    same CUDA build, the same torch and deterministic algorithms forced, 5.15.0-179 against
    5.15.0-186 changes no float operation, no CUDA kernel and no torch code path. **This is an
    argument about GPU arithmetic. It is not a proof that kernels never matter for anything** —
    scheduling, page-cache behaviour and driver-kernel interaction are all real, and none of them
    is what the identity key was defending.
  - **THE MEASUREMENT THAT FORCED IT.** vast.ai exposes no kernel filter; a box's kernel is
    knowable only by renting it. Across two batches and a probe: the **probe rented 6 boxes**
    driver-filtered and got **one driver and THREE kernels** (6.8.0-124 ×2, 5.15.0-186 ×2,
    5.15.0-179 ×2); **batch 1 rented 8**, stamped 7, and got **SEVEN distinct 16-key tuples**;
    **batch 2 rented 4** across two driver cohorts and **both cohorts split**, largest matched
    group = 1. **TOTAL: 14 boxes rented, 14 distinct kernels, ZERO matched pairs.** Driver and
    kernel are independent axes, and filtering on driver — the only one exposed — collapses one
    axis and determines nothing about the other. A matched pool was not purchasable at any price
    found.
  - **THE CHANGE, PRECISELY. The key count stays 16.** `platform_abi` — the ABI-bearing part,
    e.g. `x86_64-with-glibc2.35` — REPLACES `platform` in `PROVENANCE_IDENTITY_KEYS` and still
    refuses the whole verdict on mismatch. `platform` — the full string, kernel included — is
    still stamped on EVERY artifact as a recorded, non-compared field. Nothing else moved: GPU
    model, driver, CUDA build, torch, numpy, python, `git_commit`, image, `lockfile_sha256`,
    `attention_mode` and all three determinism flags still must match exactly.
  - **IT FAILS CLOSED.** The kernel is stripped by removing the exact `"{system}-{release}-"`
    prefix; where that prefix does not match, **the FULL string is returned** and the key goes on
    comparing everything it compared before. Not hypothetical: macOS reports `platform()` as
    `macOS-26.5.1-arm64-arm-64bit` while `system()`/`release()` say `Darwin`/`25.5.0`. A parser
    that silently mangled an unrecognised format would be the fail-open class wearing a success
    mask.
  - **VERIFIED INDEPENDENTLY AT ENTRY-TIME, on the literal strings, not adopted on the brief's
    word.** All **seven** kernels named in the builder's test as actually rented collapse to the
    single value `x86_64-with-glibc2.35`; `glibc2.31` still separates; `aarch64` still separates;
    the macOS prefix mismatch returns the full string. `PROVENANCE_IDENTITY_KEYS` reads 16 with
    `platform_abi` at position 12 and `platform` absent from the comparison set.
    `tests/run/test_platform_abi_split.py` and `tests/run/test_fanout_refusal.py`: **40 passed.**
  - **THE TIMING, AND HOW A READER CHECKS IT.** Decided and landed **before any unit was trained
    at the pinned budget**, so it cannot have been motivated by seeing a result. The check is
    mechanical: no eval artifact anywhere under `runs_cloud/` carries
    `provenance.steps_stage2 == 26003`. **Stated precisely, because the loose form is falsifiable:
    23 eval artifacts DO exist on disk** — 15 from the toy rehearsal, 7 from the CUDA validation
    probe, 1 from P2 — and every one of them is a probe at a non-pinned budget. "Zero units
    existed" is true of M6 money units and false of eval artifacts; the entry says which.
  - **WHO — the sequence, because two of these are easy to misattribute.**
    1. **The builder** made the technical observation — that the platform key's kernel component is
       stricter than the invariant it defends — and **explicitly declined to propose acting on
       it**, writing *"I am not proposing to weaken it, and I have changed nothing. Your call."*
    2. **The supervisor** ruled the key STAYS, on the premise that a gate satisfiable for ~$2 by
       filtered rental is never weakened. **That premise was later refuted by measurement** — 14
       boxes, 14 kernels, no purchasable pool at any price found.
    3. **The supervisor** then converted the builder's observation into a live option, presented it
       to the operator against the 11.3-day wall-clock, **and recommended against it.**
    4. **The operator** ruled to take it.
    The builder originated the technical claim and declined to act on it; the supervisor
    originated the PROPOSAL and argued against his own proposal; the operator decided. Recorded at
    the builder's specific request, so that observing a gate is stricter than necessary is not
    filed as proposing to weaken a scientific gate.
  - **WHAT IS PRESERVED: the relaxation is AUDITABLE, not invisible.** The kernel is still stamped
    per artifact, so a reader can read off exactly which kernels ran and check for themselves
    whether any result correlates with one. Had we dropped the field instead of demoting it, that
    check would be impossible.
  - **COST — THE BRIEF'S FIGURE DOES NOT SURVIVE, AND THE CORRECTION STRENGTHENS THE POINT.** The
    brief stated the change "raises cost from ~$96 to ~$121". **$96 appears in no artifact in this
    repository**, and the runbook's own measured table contradicts the direction: one box is 270.7
    billed GPU-h at $79/$108, five boxes is 271.8 at $79/$109 — **a delta of +1.1 GPU-h and about
    +$1**, because the work is identical and only the 0.28 h/box setup multiplies. The $121 figure
    is real but is the worst-case total for *determinism on the eval leg at $0.40/hr spot*, not a
    fan-out delta.
    **THE ACCURATE STATEMENT, which is about WORK and not about DOLLARS.** The fan-out adds
    **~1.1 GPU-h of work — about a dollar**. Any dollar difference beyond that comes from the
    POOL'S MEAN HOURLY RATE against the single cheapest box: total cost is
    `(270.4 + 0.28×N) GPU-h × mean hourly rate`, and at N=1 you buy the single cheapest machine
    while at N=5 you buy the five cheapest, so the mean rises — the cheapest matched box measured
    $0.3556/h and the next ones $0.58–$0.74 in an earlier cohort. **That is a market condition,
    not a property of the change.** "Roughly one dollar" is true of HOURS and can be false of
    DOLLARS, and the distinction is kept here because collapsing it would understate what the
    operator actually pays. The conclusion survives and is stronger for being exact: **there was
    no WORK-versus-time trade to make**, which removes the only reading under which the motive
    could have been cost.
  - **NO GATE VALUE MOVES elsewhere; the freeze is otherwise untouched.** The identity surface
    still refuses on 16 keys, and a deleted key now fails a test rather than deleting its own test
    case (`test_the_identity_surface_IS_the_literal_list_and_nothing_drifted`).

- **v1.6.28 (2026-08-11, ★ RULING — THE DEGENERACY GUARD'S ACTIVITY LEG IS **NOT** BANDED.
  $0, documentation only. NO GATE VALUE MOVES.)**
  - **Ruling of record:** `docs/degeneracy_guard_activity_leg_RULING.md`, commit `a13bed3`. This
    entry is the log's copy; the ruling document is the long form. **The ruling previously existed
    only as that side document and named a tag that was not in this log** — caught by the paper's
    own count, which reported what the log CONTAINED rather than what the ruling ASSERTED. The
    discipline that caught it is the one being recorded here.
  - **THE RULING: keep as is. The activity leg tests the exact endpoints 0 and 1, with no band.**
    The reason is **INSUFFICIENCY OF THE STATISTIC, NOT ACCEPTANCE OF THE RISK.** Verified against
    `runs_manifest/m6_h_sweep.json :: cells.moneyleg_noise_cell2.by_h.5.mu_diag`:
    `activity_decisions_by_kappa = {1: 1.0, 1.5: 1.0, 2: 1.0, 3: 0.9318333333333333}` — **the
    filter is FULLY INERT at three of the four pinned grid values while the κ\* scalar the guard
    reads sits at 0.9318**, interior to `[0.05, 0.95]` and to any plausible widening. A perfect
    band on that scalar would not have seen it. The κ\* scalar answers *"did the filter bind at the
    selected κ"*; the degeneracy is *"did the filter do per-bar work"*. Banding a provably
    insufficient statistic buys the APPEARANCE of coverage. Secondary reasons: activity is a
    scale-dependent FRACTION (9 trades is 7e-6 at n = 1.3M and 7.5e-4 at n = 12k — the same disease
    three orders of magnitude apart, and no single band expresses it at both scales); the endpoints
    target the two fixed points of the argmax-over-κ map rather than a rounding boundary; and a
    false HALT is not cheap, since C.1 ruled S=5 up front so R1/R2b collapse into R3 and
    `grep -c "HALT\|degenerac" docs/m6_fanout_runbook.md` returns **0** — no recovery procedure
    exists.
  - **THE BLIND SPOT IS REAL, WIDE, AND DISCLOSED RATHER THAN CLOSED.** An endpoint-only test does
    not see a book trading 0.1% of bars, and that is the EXPECTED real-data regime rather than a
    corner case. This ruling does **not** find the exposure acceptable; it finds that **banding
    this scalar does not close it**. The closure is the disclosure plus the three reads below.
  - **TWO CORRECTIONS THE VERIFICATION PRODUCED AGAINST `docs/degeneracy_guard_band_decision.md`
    (`cefca3b`), BOTH CONSERVATIVE — the memo understated its own finding.**
    1. **`std(y_15)` = 0.003504 over n = 2,103,825**, not 0.004238 over n = 1,051,199. Measured on
       `processed/universe_bars/symbol=BTCUSDT` over the FULL 48 symbol-months, respecting segment
       boundaries and counting only complete 15-bar windows. **Independently reproduced at
       v1.6.28-time to six decimals and to the exact bar count under that convention** (a naive
       window that bridges segment gaps gives n = 2,103,839, a 14-bar difference that does not move
       the standard deviation). Direction: a SMALLER std(y) puts the execution threshold FURTHER
       out — **8.5 / 11.6 / 22.2 sd** at IC = 0.027, κ=1 across the modeled-cost range rather than
       7.0 / 9.6 / 18.4 — and raises the IC a calibrated forecast needs to trade 5% of bars from
       0.096–0.253 to **0.116–0.306**, four to eleven times our own prior.
    2. **The memo's "7.7% of compute-optimal" hedge is STALE**, superseded by `0b9c804`
       (2026-08-03). The pinned budget is 26,003 steps = **426,033,152 tokens = 20.00
       tokens/param**, independently recorded at `docs/BUILD_RECORD.md:422`. **A model trained to
       compute-optimal is MORE likely to be calibrated, not less**, so the no-trades branch is MORE
       likely than the memo states and the over-dispersion escape it leaned on is WEAKER.
       *Sourcing note recorded at v1.6.28-time:* the token figure needs three inputs, and only two
       are pins — `PINNED_STEPS_STAGE2` (26,003) and `PINNED_MONEY_SEQ_LEN` (512). The batch size
       of 32 is a **dataclass default** at `src/trikaal/train/orchestrator.py:90`, not a
       conformance pin, so it can drift where a pin cannot. The arithmetic and the conclusion hold
       — 20.00 tok/param is independently recorded in the build record — but the ruling document's
       claim that the figure was "verified from `PINNED_STEPS_STAGE2` at HEAD" is **incomplete as
       stated**, and is corrected here rather than in the side document.
  - **THREE $0 REPORTING CHANGES ADOPTED**, all reads of quantities the code already computes, all
    taking the same required-field treatment as `mu_diag`: (i) persist
    `activity_decisions_by_kappa` over the FULL grid per (cell, seed) — exactly the quantity the
    κ\* scalar hides; (ii) persist the trade **COUNT** beside the fraction per (cell, seed), since
    a fraction is scale-dependent and a count is not; (iii) persist `std(μ̂)/θ` at κ\* per
    (cell, seed) — the dimensionless margin, which tells an adjudicator immediately which branch
    the run landed in.
  - **NO GATE VALUE MOVES.** No pin, threshold, seed, enumeration or gate value is altered, so the
    v1.5 freeze is untouched, **no direction-blind test is engaged, and no mutation test is
    required** — there is no changed gate for a mutation to falsify. The three additions are
    manifest fields, not gates.
  - **DOES NOT BLOCK THE RUN**, under the pre-committed bar (K21, 2026-08-03, fixed before its
    first application): a finding delays the run **iff** it would cause a FALSE VERDICT that
    disclosure cannot neutralize. The exposure produces `HALT_ADJUDICATE`, which is the guard
    working rather than a false verdict, and HALT is one of the five pre-written, pre-dated
    interpretation branches in the paper's §6.5. The κ grid is **unchanged**: θ = κ·c with κ ≥ 1
    means *trade only when the forecast exceeds its own cost*, and κ < 1 is economically incoherent
    for a cost-aware filter. If the filter admits nothing, **that is a result** — at a measured
    RankIC of 0.027 no forecast clears a 0.30% round trip on any of 192 instruments — and a study
    unable to reach that conclusion would be the defective one.
  - **PAPER TOUCH-POINTS:** §7.5 carries both `std(y)` measurements with both sample windows named
    and the direction stated; Appendix D.2 records the leg as endpoint-only **by ruling, not by
    omission**, cites `a13bed3`, and names the three reads above. The paper's amendment-tag count
    moves 41 → 42 **only now that this entry exists**, which is the ordering the discipline
    requires.

- **v1.6.26 (2026-08-04, ★ P2 EXECUTED ON REAL CUDA — $0.426, three defects, primary question
  still OPEN. Boxes destroyed, re-list confirms 0.)**
  - **P2 DID NOT ANSWER ITS QUESTION AND IS REPORTED AS A RESULT.** Eval under forced determinism
    on CUDA remains **UNMEASURED**; the run stopped at the lake gate. Nothing below is dressed up
    as the answer.
  - **F1 — `torch 2.12.1+cu130` REQUIRES DRIVER ≥ 580, AND THE OFFER FILTER DID NOT SAY SO.** On
    driver 565.77 (CUDA 12.7) the pinned wheel installs cleanly and reports
    `cuda.is_available() == False`. The runbook filtered `reliability` and `disk_space` and nothing
    about the driver, so the fan-out would rent an unusable box and pay a full setup to find out.
    Filter is now `cuda_max_good>=13.0`; 44 of the 4090 offers qualify, so it costs nothing.
  - **F2 — ★ THE RUNBOOK HAD NO LAKE-PROVISIONING STEP. The money run cannot execute on any rented
    box.** `LAKE MISSING at processed/universe_bars — refusing to invent data`. The §2 tarball
    ships 550 KB of source; `--lake …` was a placeholder. **CORRECTED, AND THE CORRECTION IS
    AGAINST MY OWN PROSE:** I wrote that this "would have hit shard 0 after TRAINING SPEND". It
    would not — the lake gate is `m6_money_run.py:250` and `train_matrix` is called at `:363`, 113
    lines later (verified by me at HEAD). **The cost of the miss is five boxes' setup, ~$0.50, not
    a training run.** That is prose running HARSHER than its artifact, the third time in that
    direction, and the rule is symmetric: I took the softer version as a standing correction two
    passes ago.
  - **F3 — `driver_version` STAMPED `unavailable` ON A REAL 4090 REPORTING 580.159.03, WHICH IS
    THE R6 CLASS RECURRING INSIDE THE FIX FOR R6.** `torch._C._cuda_getDriverVersion` no longer
    behaves as assumed at 2.12.1 and the lookup failed into its own placeholder. R6's whole
    argument is that an identity key exists to make shards DISAGREE; a key reading the same
    `"unavailable"` on all 25 units can never disagree, so it contributed nothing while being
    counted among the 16 that made the surface look complete. Two halves: the lookup now asks
    `nvidia-smi` FIRST (the driver's own reporter, not a binding to it), and
    `identity_placeholder_failures` REFUSES any CUDA unit carrying a placeholder — at the money
    driver **before any spend** and again at `write_cell_eval_artifact`, the durable gate. CPU is
    exempt by construction: there those keys are genuinely unresolvable and `"unavailable"` is the
    honest value.
  - **★ THE MUTATION HARNESS REJECTED MY OWN FIRST F3 TEST, AND IT WAS RIGHT.** `F3a` came back
    **NOT CLOSED**: the test stubbed `subprocess.run` and asserted only on the stub's return value,
    so corrupting the query field to `--query-gpu=NOTHING` left it green. **A test that mocks the
    call it is verifying, and checks only the mock's own answer, verifies the mock.** The argv is
    the contract — a real `nvidia-smi` asked the wrong field prints an error and the lookup falls
    back to the placeholder, which IS the F3 defect. Rewritten to assert the argv; **12/12 CLOSED**
    with the negative control still passing.
  - **TWO OPERATIONAL FINDINGS, BOTH INTO THE RUNBOOK.** (1) **Setup is a HOST PROPERTY WITH A LONG
    TAIL, not a constant** — measured the same day: **45 min at reliability 0.992** (produced
    nothing, **$0.254**) versus **29 s at 0.997**, a 90× spread. Filter `reliability>=0.995` and
    **destroy any box not `running` inside 15 min** rather than waiting. My estimate missed by 1.4×
    and this is its entire cause; setup must be costed with a tail, not a mean. (2) **`vastai
    create` PRINTED NOTHING AND CREATED THE BOX ANYWAY** — the exact mirror of the `destroy` trap,
    opposite sign: destroy lies by returning 0 having done nothing, create lies by saying nothing
    having done everything. **RE-LIST AFTER EVERY `create` AND EVERY `destroy`** (rule R5b).
  - **WHAT P2 DID ESTABLISH, on real hardware, for $0.426:** torch 2.12.1+cu130 installs and sees a
    4090 on driver 580.159.03; the **conformance gate passes ON THE BOX**;
    `use_deterministic_algorithms(True)` is settable on CUDA; `attention_mode` reads
    `sdpa_deterministic`; **15 of 16 identity keys stamp real values**, the 16th being F3.
  - **THE LAKE ROUTE IS OPTION B AND IT NEEDS ONE OPERATOR ACTION.** `lakshayybhati/trikaal-m6-
    snapshot` returns **HTTP 401 unauthenticated** on both `api/datasets/…` and `api/models/…`, and
    the account lists **zero public repos** — so the lake is **not publicly readable**. 401
    deliberately does not distinguish *private* from *absent*, and that is the strongest statement
    the evidence supports; I am not claiming more.
    - **★ ADDENDUM 2026-08-18 — THE CONCLUSION SURVIVES, ONE SUPPORTING FACT DOES NOT, AND NOBODY
      ROUTED THIS ONE.** Re-checked unauthenticated: `trikaal-m6-snapshot` still returns **HTTP
      401** on both `api/datasets/…` and `api/models/…`, so **the lake is still not publicly
      readable and the conclusion above stands unchanged**. But *"the account lists zero public
      repos"* is **no longer true** — the weights release created
      `lakshayybhati/trikaal-v1-baseline`, which returns `"private": false`. The dated sentence is
      left as written; what is corrected is that one of its two supporting observations has since
      expired while the other, and the conclusion, have not. **Only the 40 pinned symbols are needed: 4.08 of
    14.59 GiB, 27.9 % — 72 % of the lake never moves.** The existing token is write-scoped and NOT
    fine-grained and must never reach a rented box: the risk is INTEGRITY, not confidentiality —
    it can overwrite or delete the Merkle-`5dfd667d` anchor the whole reproducibility claim rests
    on.

- **v1.6.25 (2026-08-03, ★ THE RE-AUDIT'S BLOCKING FIXES. Local, $0. The re-audit landed FIVE DAYS
  EARLY and returned NOT READY; the run date does not move.)**
  - **★ THE HEADLINE, AND IT IS THE WORST DEFECT THIS PROJECT HAS PRODUCED (R1). The v1.6.22
    codebook gate — the fourth instance of "specified but not enforced", the one whose closure was
    the compensating control for dropping the C-4 external gate — SHIPPED BELOW ITS FUNCTION'S
    `return bad` AND WAS UNREACHABLE FROM THE MOMENT IT WAS WRITTEN.** `verdict.py:270`. It was
    born dead in `0b9c804`, the commit that implemented all four approved decisions in one pass,
    and 601 tests stayed green because not one of them fed the gate an artifact it was supposed to
    reject. **The class was declared closed by a fix that never executed once.** The `return` now
    sits below the block, and `tests/eval/test_codebook_gate.py` rejects a collapsed codebook
    (0.01), an empty payload, a missing key (tampered on disk and RE-INDEXED so the content hash
    cannot be what refuses it), a half payload, and a NaN utilization — through the real
    write → load → assemble path, with the healthy set asserted to assemble first.
    **THE LESSON IS NOT "READ MORE CAREFULLY": a gate is closed only when something has watched it
    fail.** Hence `scripts/m6_reaudit_mutations.py` (below), which is the deliverable, not the fix.
  - **R2 — THE C-3 AMENDMENT SHIPPED WITHOUT THE PIN AND THE MUTATION KAT ITS OWN SIGNED PACKAGE
    SPECIFIED.** Reverting one identifier (`PRIMARY_H` → `h` at `enumerate_dsr_trials`) passed all
    601 tests and silently restored the EASIER mixed-unit basis — outcome-material by the
    amendment's own witness (DSR 0.9868 PASS → 0.8577 FAIL). Now `PINNED_DSR
    ["de_annualization_horizon"]`, cross-checked against `verdict.PRIMARY_H`, **and**
    `verdict.dsr_unit_convention_failures` which RE-DERIVES all 300 trial values from the
    artifacts **on the money path** — so the revert fails the RUN, not merely a test. 300 divisions
    on a 270-GPU-hour experiment.
  - **R3 — THE `v1_2_original` DUAL-SPECIFICATION LEG WAS A HYBRID CALLING ITSELF FAITHFUL, AND
    THE BIAS HAD A DIRECTION.** It read `trials`, which C-3 had re-based onto PRIMARY_H units, so
    the shipped leg was {all-arms basis (v1.2)} × {PRIMARY_H units (v1.5)} — a combination no
    specification ever had. Mixing horizons INFLATES dispersion, so the faithful leg has the
    larger `var_sr`, larger SR₀ and LOWER DSR: **the hybrid agreed with the v1.5 primary more
    readily than v1.2 would have, i.e. it under-reported disagreement exactly when the primary
    says SURVIVES** — the one direction a dual-specification report exists to guard.
    `own_h_trial_values` is now a function and both superseded legs call it; the direction is
    asserted, not argued (`test_the_hybrid_leg_was_biased_TOWARD_agreement`).
  - **R4 — THE G-§8.C.3 VERDICT AND ITS REQUIRED BSQ DISCLOSURE WERE EVALUATED AND DISCARDED.**
    The gate result was passed straight to `assert_clear_to_compute_deltas` and never bound, so
    the one compensating record for dropping a binding entry gate existed for the duration of one
    expression. It is evaluated UNCONDITIONALLY now and persisted as `external_validation`;
    `REQUIRED_MANIFEST_FIELDS` + `verdict_manifest_failures` refuse a manifest without it,
    `assemble_verdict` validates its own output, and `load_verdict_manifest` re-validates **from
    the file** — the case emission-side validation structurally cannot see. **A MISSING DISCLOSURE
    IS THE ONE DEFECT DISCLOSURE CANNOT NEUTRALIZE**, which is why this cleared the bar.
    Both "the gate is BLOCKED today and will HALT" comments (`verdict.py`, `m6_verdict.py`) were
    TRUE when written and FALSE from v1.6.22; corrected. **A comment describing a control that
    cannot fire is worse than no comment — it tells the next reader the control is live.**
  - **R6 — THE IDENTITY SURFACE RECORDED WHICH MACHINE AND SAID NOTHING ABOUT WHICH CODE**, and
    the tests that proved it worked PARAMETRIZED OVER THE LIVE TUPLE, so deleting a key deleted
    its own test case. Two halves of one defect. `git_commit`, `steps_stage1`, `steps_stage2` are
    identity keys now (**16**); the commit is stamped from `TRIKAAL_GIT_COMMIT` because the money
    boxes have no `.git` (we ship a tarball precisely so no credential reaches them), with a local
    `git rev-parse` fallback. The refusal tests parametrize over a **written-out literal list**,
    plus `test_the_stamper_POPULATES_every_identity_key` — because a key in the surface that
    `run_provenance` never writes is absent from all 25 units EQUALLY, so it protects nothing
    while appearing to. The auditor's named path: a stale-payload shard training at the old
    2,000-step budget assembling silently beside four 26,003-step siblings.
  - **R12e — CELL 5'S TWO REQUIRED DISCLOSURES DESCRIBED A TENSOR THE CELL WAS NEVER FED.**
    `score_cell` shuffled into LOCAL variables and then rebuilt the diagnostic window from
    `se.x`, the RAW matrix. So on the placebo arm — and only there — `ohlcv_recon` and
    `decode_agreement` measured a tokenizer trained on shuffled micro against UNSHUFFLED micro:
    out-of-distribution input, not "the capacity handicap". **The C-12 disclosure's entire
    argument is about what Cell 5 spends its bits on, so it was wrong in exactly the arm it exists
    to characterise.** Every other arm was unaffected, which is why nothing looked odd. The
    arm-transformed tensors now travel from the loop that built them; the regression test spies on
    `tok.latent` (which only the diagnostics call) and asserts the unshuffled window **never
    reaches the tokenizer at all** on the placebo arm.
  - **FREE, SAME PASS: R7** — the money manifest shipped *"the default 2000 … not a spec
    constant"* beside a values dict reading 26,003, **and `test_unpinned_parameters_are_recorded_
    with_their_reasoning` ASSERTED THE STALE SENTENCE — the suite was defending the false half.**
    The budgets moved out of `unpinned_parameters` entirely (they are pinned and gated) and the
    test now asserts against the live conformance pins. **R10a** — `forward_log_returns`, the
    money label every net IR in the study is built from, had ZERO tests; a same-bar-entry
    lookahead (`clp[p+h] - clp[p]`) passed everything **including the Gate-A causal file**, which
    tests that FEATURES do not see the future and says nothing about which bar the LABEL enters
    on. Two different lookaheads, one unguarded. 12 tests, keyed on the sharp invariant: y_t is
    invariant to r_t AND r_{t+1} and responds to every bar in (t+1, t+h].
  - **★ THE DELIVERABLE IS THE MUTATION HARNESS, NOT THE EIGHT FIXES —
    `scripts/m6_reaudit_mutations.py` → `runs_manifest/m6_reaudit_mutations.json`, 9/9 CLOSED.**
    Each fix's PRE-FIX source is restored in an isolated copy of the tree and its tests must FAIL
    there. Three ordered stages, because three of my probes have manufactured false findings from
    their own bugs: BASELINE (the unmutated copy must PASS, else `HARNESS_BROKEN` — never
    evidence), APPLY (the string must occur EXACTLY ONCE, else `NOT_APPLIED`), MUTANT (must FAIL,
    else `★ NOT CLOSED`). **Plus a NEGATIVE CONTROL: a comment-only edit that MUST still pass** —
    without it, "9/9 CLOSED" is itself a check that has never been seen to fail. The real
    repository is never mutated.
  - **R8 AND R9 ARE RECURRENCES OF CLASSES WE DECLARED CLOSED, AND ARE RECORDED AS SUCH.** R8: §6
    here, `m6_design.md:50` and `ROADMAP.md:58` each described G-§8.C.3 as a **live binding entry
    gate** after it was dropped; `enumerate_dsr_trials`'s docstring said C-3 was *"REPORTED, NOT
    FIXED"* and *"this function is unchanged"* **one line above the line that changed**; and
    `attention_mode.py` still carried invariant 7's false SUFFICIENCY premise. These were on the
    supervisor's move-to-limitations list and I **PROMOTED** them: they are protocol claims that
    are false, in the four documents the paper's methods will be written from, and one of them is
    a standing invitation to revert R2. R9: five sweep receipts stale at HEAD, regenerated —
    "stale" is indistinguishable from "never re-run after the change that mattered".
  - **AND R9 IMMEDIATELY EARNED ITS PROMOTION, WHICH IS THE ARGUMENT FOR TREATING A STALE RECEIPT
    AS A FINDING.** `m6_manifest_prose_sweep.json` was committed reading **CLEAN over 162
    strings** — but that run PREDATED v1.6.22's mixed-unit leg, so
    `/dual_specification/v1_5_mixed_unit_basis_superseded/why_superseded` **had never been swept
    while the receipt asserted the manifest was clean.** Regenerating at HEAD surfaced 4 strings
    for adjudication over 178; all four are legitimate (a derived √12 forced by `DSR_HORIZONS`, a
    DATE the extractor cannot distinguish from a number, a rhetorical "~100%" ceiling in a caveat,
    and two EXTERNAL Kronos figures plus their ratio) and are now in the sweep's `ADJUDICATED`
    registry, which binds to the exact number set so a changed number re-opens them. **"Clean" was
    a statement about a repository that no longer existed.**
  - **A SECOND INSTANCE OF THE v1.6.24 SETUP-COST DEFECT, FOUND BY THE REGENERATED CLAIMS SWEEP AND
    IT IS MINE.** `runs_manifest/m6_integrated_price.json`'s `top_up_reasoning` claimed *"~30 min
    per box observed on the probe"*. The measurement from that same probe day is **~17 min** (10
    min pull + 6m41s install + <5s scp = 0.28 GPU-h). Written from impression rather than from the
    receipt — **the same defect as the runbook's 0.6 h/box row, in a cost manifest Lakshay funds
    against, and the runbook fix was the instance while this was its sibling.** Corrected in place
    with the derivation. Direction conservative both times, so **$150 is untouched**; that is luck,
    not method.
  - **★ AND THEN P1 FIRED, WHICH IS THE POINT OF P1 — R5 WAS NOT A HYGIENE ITEM, IT WAS HIDING A
    BLOCKING DEFECT.** The first `--dry-run` at the fixed HEAD reached the eval stage — further
    than any previous run of this driver has ever reached — and **REFUSED EVERY ARTIFACT**:
    > `refusing to write an eval artifact the degeneracy guard could not read:`
    > `cell1_seed0: decode_agreement missing 'sign_agreement_dim0' …`
    > `cell1_seed0: ohlcv_recon missing 'ohlcv_recon_mae' …`
    **`score_cell` has TWO return sites.** The `val_only` one passed both diagnostics; **the FULL
    one — the return the verdict artifact is built from — passed neither**, so they defaulted to
    `{}` and `write_cell_eval_artifact` refused. Verified pre-existing at `HEAD~1`
    (`git show HEAD~1:src/trikaal/eval/xsection.py` — the full return never mentions them), so it
    is **not** collateral from the R12e fix. **THE MONEY DRIVER COULD NEVER HAVE WRITTEN A SINGLE
    EVAL ARTIFACT**, and it would have discovered that AFTER paying for a shard's training.
    It is the R4 shape in a different file: two REQUIRED disclosures computed thirty lines earlier
    and dropped on the exact path that feeds the decision. It survived 655 tests because
    `test_score_cell_and_ablation_verdict_end_to_end` asserts on `s.codebook` and **never on these
    two**, and because the driver had never been run to completion at this configuration — which
    is exactly what R5 said and why I ran it. Fixed; the new test is parametrized over **both**
    returns so repairing one and leaving the other cannot pass. Mutation row R5 added: **10/10**.
  - **DISCLOSED, MINE, AND THE THIRD COSTUME OF THE PIPEFAIL RIDER IN TWO PASSES.** I launched P1
    as `<cmd> > log 2>&1; echo "EXIT=$?"`. The shell's status for that compound is **`echo`'s**, so
    the harness reported the job as **exit 0** while the run had exited **1**. I caught it only by
    reading the log. **An exit code taken from a compound whose last statement is `echo` is not the
    command's exit code**, the same way one taken through `| head` is `head`'s. The v1.6.24 norm
    said to state exit codes explicitly; it now also has to say WHERE the exit code came from.
  - **★ P1 THEN COMPLETED, EXIT 0 — AND IT IS THE FIRST TIME THIS DRIVER HAS EVER RUN TO THE END.**
    25 units trained, 25 checkpoints reloaded, **25 eval artifacts + index written**, and
    `money_run_manifest.json` produced — the file the re-audit observed did not exist anywhere.
    Verified on the REAL artifacts, not a fixture: **16/16 identity keys stamped, ONE distinct
    stamp across all 25**, `git_commit = a9743938…` (the exact commit — R6's stamping proven
    end-to-end), `steps_stage1/2 = 26,003`, and `load_cell_evals` **REFUSED** the set with
    *"DRY-RUN artifact (money_run=False, grid_pinned=False)"*, which is the C-5 A9 contract
    working. Four keys read `unavailable` — `image`, `gpu_name`, `cuda_build`, `driver_version` —
    exactly as they must on a laptop with no CUDA and no `TRIKAAL_IMAGE`, **which is what P3 exists
    to catch on the box**.
  - **A MEASUREMENT I GOT BADLY WRONG, RECORDED BECAUSE THE ERROR WAS AVOIDABLE.** Asked how long
    P1 would take I said the eval leg would be "single-digit minutes total"; artifact timestamps
    give **11.2 min PER UNIT** (00:00:24 → 00:12:38 → 00:23:22 → 00:34:06), ~25x out, ~4 h for the
    matrix. I estimated from DECISION COUNT (32) and ignored that **~97 % of rollout wall-clock is
    context prefill at seq_len 512** — which is item C6 in our own limitations register. **I had
    the number and did not apply it.** The datum stands as the local-MPS calibration: 32 decisions
    x 3 horizons is ~11 min/unit here, and it says nothing about the 4090 (the standing "local
    timings carry ~8x variance and MPS is not CUDA" rule applies).
  - **★ NEW STANDING NORM, SUPERVISOR-DIRECTED — "HAS NEVER BEEN EXECUTED" IS NOT AN OPERATIONAL
    STATE; IT IS AN UNMEASURED RISK, AND EXECUTION IS THE ONLY INSTRUMENT THAT MEASURES IT.**
    Landed in `docs/ENGINEERING.md` beside the remediation rule. The supervisor recorded it as their own
    ninth error and the pair with R1 is exact: **R1 — the SYMBOL was confirmed to exist, not that
    the GATE FIRED. R5 — the driver was confirmed to be BUILT, not that it RAN.** Both are
    verifying EXISTENCE instead of BEHAVIOUR, the class this project has spent three weeks
    enforcing against. Operational corollary: **any component whose first execution is on rented
    hardware is an untested component, whatever its coverage says.**
  - **P1's COMPLETION EVIDENCE IS RETAINED — `runs_manifest/m6_p1_dry_run_completion.json`
    (`scripts/m6_p1_retain.py`).** The supervisor asked for the artifact and was right to; their
    accompanying claim that `money_run_manifest.json` "is not on disk anywhere — it went to
    args.out and was cleaned" is **the one thing in that message that is wrong**: it survived at
    the scratch `--out` path and was lifted from there intact (sha256 `20202b71e0df91e4…`). The
    receipt carries the manifest verbatim, the 16-key identity surface, the measured timings, and
    — re-executed rather than asserted — **the loader's refusal of its own artifact set**
    (*"DRY-RUN artifact (money_run=False, grid_pinned=False)"*), so it ships the proof of its own
    un-quotability instead of a promise of it. Measured: **5.65 h wall, 4.43 h of it eval, 638 s
    mean per unit** — the datum behind the estimate error disclosed above, and it bounds nothing
    about the 4090.
  - **NARROWED, AND IT IS WHAT MAKES P2 A ONE-QUESTION PROBE:** every P1 artifact carries
    `deterministic_algorithms: true`, so **the eval path completed end-to-end under FORCED
    DETERMINISM on CPU**. Training under forced determinism on CUDA was already measured (v1.6.22
    B1, 13.175 % penalty). The single uncovered leg is therefore **EVAL under forced determinism
    on CUDA** — an op with no deterministic CUDA kernel raising *inside eval, after training
    spend*. That is exactly one binary question and it does not need 26,003 steps to answer.
  - **R5 DOES NOT DELAY THE DATE; IT IS AN ORDERED PRECONDITION**, now `docs/m6_fanout_runbook.md`
    §1a: **P1** local `--dry-run` at the shipped HEAD ($0), **P2** shard 0 ALONE including at least
    one eval decision under FORCED DETERMINISM (a first-time code path — the flag is process-global
    and was armed for training), **P3** all 16 identity keys present and populated. Only then do
    the other four boxes launch. Discovering a first-time path at P2 costs one shard, not five.
  - **WHAT I DO NOT KNOW, STATED:** I hold the supervisor's summary of R10b–d, R11 and R12a–d and
    my own reproductions — **not the auditor's verbatim text**. Those rows are deferred to
    `docs/v2_and_limitations.md` **on the supervisor's triage, not on mine**, and the register says
    so. If one of them is false-verdict-capable, that is a promotion I did not make, and the
    reason will be that I never read it.
  - **MY SUPERVISOR'S OWN FINDING, WHICH IS THE REASON R1 SHIPPED, RECORDED BECAUSE IT GENERALISES:
    they verified the codebook fix by GREPPING FOR `PINNED_CODEBOOK_MIN_UTILIZATION` AND FINDING
    IT.** Confirming that a symbol exists is not confirming that a gate fires. Everyone audited the
    findings; nobody audited the fixes — and `0b9c804`, which implemented four approved decisions
    in one pass, got the least scrutiny in the repository because both of us were busy verifying
    its report. **The new code in a remediation pass is written fast, by the person whose blind
    spots caused the defects, and is the least-reviewed code in the project.**

- **v1.6.23 (2026-08-03, RUN PREPARATION — fan-out refusal, runbook, assembly dry run. No new
  investigation; the re-audit is external and none of it is the builder's.)** Local, $0.
  - **★ A CORRECTION IN A COMMIT MESSAGE, WHICH MAKES IT PERMANENT — AND IT RUNS HARSHER THAN ITS
    OWN RECEIPT.** `0b9c804` says the measured eval leg was *"up to 2.5x the banked figure"*. That
    **mixes endpoints**: measured-at-$0.40 against inferred-at-the-bottom. **At matched rates it is
    1.60–1.64×** ($50.67/$30.98 = 1.64; $77.95/$48.69 = 1.60). Still a serious underestimate and
    still the right catch — but the number as written overstates it. **This is the same defect as
    prose running SOFTER than its receipt, which was taken as a standing correction two tranches
    ago, and it is recorded on identical terms: THE ARTIFACT WINS IN BOTH DIRECTIONS.** The
    receipt `m6_integrated_price.json` carries the honest ratio; the commit message cannot be
    edited, so the correction lives here.
  - **THE FAN-OUT REFUSAL IS WIRED, AND TWO IDENTITY KEYS WERE MISSING.** `provenance_failures`
    was already a hard refusal via `load_cell_evals` → `VerdictInputError`, but
    `PROVENANCE_IDENTITY_KEYS` did not include the **container image** or the **lockfile hash** —
    two boxes can carry identical torch/numpy/driver strings and still differ in CUDA userspace or
    resolved dependency set. Both added (13 keys). `TRIKAAL_IMAGE` is stamped by the launcher and
    `"unavailable"` is the honest value when unset — which, being an identity key, means a run
    where SOME shards stamp it and others do not is a **refusal**, the case that would otherwise
    slip through.
  - **EVERY KEY MUTATION-PROVEN.** `tests/run/test_fanout_refusal.py` — 18 tests. Discrimination
    first (a uniform fan-out MUST assemble, or every refusal below is vacuous), then one drifted
    unit on **each of the 13 keys**, each required to refuse and to NAME the key. The drifted unit
    is #7 (shard 2) deliberately: an off-by-one in the partition would make unit 0 a special case
    and hide the general failure. Plus the absence case — a shard whose launcher never stamped.
  - **THE ASSEMBLY DRY RUN — `runs_manifest/m6_fanout_dry_run.json`, GREEN, $0.** 25 fabricated
    units across 5 simulated shards: exact disjoint cover ✓, empty matrix refused ✓, uniform
    fan-out assembles to a verdict word with **one** distinct provenance stamp ✓, **13/13 keys
    refuse** ✓. Scope stated in the receipt: it proves the ASSEMBLY contract, not that the launcher
    stamps correctly on a real box — **that is verified by shard 0's artifact before the other four
    launch.**
  - **THE RUNBOOK IS EXECUTABLE AND COSTED — `docs/m6_fanout_runbook.md`.** Five boxes: 54.1 h
    wall, **271.8** billed GPU-h, **$79–109**, a preemption costs 1/5. Setup measured at **~17 min
    per box** (10 min image pull + **6m41s pinned-torch install**, because the image ships torch
    2.5.1 and we pin 2.12.1) — **90% of the probe's cost, and it must be in the plan rather than
    discovered again.** Standing lesson recorded: **cost a rental as `setup + compute`, never
    `compute`.** Credential rule R1 is explicit: the repo is private, so scp a tarball; no token
    ever reaches a rented box. R5 records the `destroy` trap verbatim.
    - **★ ADDENDUM 2026-08-18 — THE PREMISE MOVED; THE RULE DID NOT.** `lakshayybhati/trikaal`
      **WENT PUBLIC** (verified unauthenticated: HTTP 200, `"private": false`,
      `"visibility": "public"`). The sentence above is **left standing because it was true when
      written and this log records what we believed and when** — the entry is dated 2026-08-03.
      **R1 itself is UNCHANGED and is now easier to satisfy, not harder**: a public repo clones
      with no credential at all, so "no token ever reaches a rented box" holds trivially rather
      than by the tarball workaround. The mechanism got simpler; the rule did not move. The
      builder made the same correction in `docs/cloud_runbook.md` and `docs/m6_fanout_runbook.md`,
      in the same form.
  - **CORRECTED 2026-08-03 (same day, supervisor-caught) — THE RUNBOOK CONTRADICTED ITSELF ON A
    ROW LABELLED "MEASURED".** The cost table's setup row read **0.6 h/box *(measured)*** — double
    the 0.28 GPU-h measured and stated two sections below in the same file, and supported by
    nothing. It fed billed-GPU-h, so **271.0 / 273.4 / 285.4 all inherited it**; corrected to
    **270.7 / 271.8 / 277.4** and $ rows to **$79–108 / $79–109 / $80–111**. The direction was
    conservative, so **the $150 top-up is untouched and no funding claim changes.** It **halves the
    setup tax (5 boxes = 0.5 %, not 1 %)**, which *strengthens* the five-box choice — but the
    choice was, and remains, made on the identity-mismatch argument (one drifted key refuses all
    25), not on the tax. **A self-contradicting artifact carrying a "measured" label is the class
    of `docs/m6_c3_amendment_decision.md` §C-17 and the prose sweep: same document, two numbers,
    one of them authored rather than measured.**
  - **PROCESS, AND IT IS THE LARGER HALF OF THIS ENTRY — AN OMITTED CLAIM IS HARDER TO CATCH THAN
    A FALSE ONE.** The v1.6.23 report omitted the `ruff` line that every prior report carried, and
    `ruff check` was **exit 1** (6 errors, in files that very commit created). Nothing false was
    stated; the *absence* is what let it through, because a missing green line has to be noticed
    while a wrong one only has to be read. **Standing from here: every report states `ruff`'s exit
    code explicitly, pass or fail, on the same terms as the Gate-A anchor's.** The supervisor
    disclosed the symmetric defect on their side — the exit code they first read came from
    `ruff … | head`, i.e. `head`'s status, the **pipefail rider for the third time this session**.
    **Both halves are the same lesson: a verification you did not actually see is not a
    verification.**

- **v1.6.22 (2026-08-03, ★ THE FOUR DECISIONS IMPLEMENTED + THE EVAL LEG MEASURED. The project
  now has a TERMINATION CONDITION and a pre-committed run-blocking bar.)** Probe cost **$0.14**.
  - **THE TERMINATION CONDITION.** Lakshay's diagnosis, accepted: the audit-fix-reaudit loop had
    become comfortable — real catches, no spend, asymptotically approaching the run. **The
    run-blocking bar is pre-committed BEFORE the re-audit exists and may not be adjusted in
    response to its findings:** *a finding delays the run iff it would cause a FALSE VERDICT —
    SURVIVES when the truth is NULL or vice versa — AND cannot be neutralized by disclosure.*
    **THE RUN FIRES 2026-08-13 WHATEVER THE STATE.** No new tranches;
    `docs/v2_and_limitations.md` is the terminal register and is not a queue.
  - **★ THE EVAL LEG IS MEASURED AT LAST, AND IT WAS UNDERSTATED.** Receipt
    `runs_manifest/m6_eval_throughput_probe.json` (sha256 verified on the box and locally before
    teardown), integrated in `runs_manifest/m6_integrated_price.json`.
    - **NO OOM AT `chunk=512`.** Peak memory **8.91 GiB of 24** on the 4090. **The local OOM
      (19.69 GiB) was an MPS artifact, not a property of the recipe** — the pinned configuration
      runs on the hardware class the whole experiment was costed against. That was the outcome
      that would have cleared the bar, and it did not fire.
    - **50.0 decisions/s** at the exact money surface (`chunk=512`, `seq_len=512`, h=15,
      expectation, torch 2.12.1+cu130 pinned on the box). Marginal **0.020009 s/decision** from
      the two tight points (n=1024 std 0.031 s, n=2048 std 0.064 s, agreeing to 0.13%).
    - **THE n=512 POINT IS DISCARDED AND THE PROBE'S OWN FIELD WITH IT.** Its std was 9.17 s on a
      15.5 s mean — one outlier repeat, and NOT warm-up (a warm-up discard was added and the std
      did not move). The receipt's `marginal_seconds_per_decision` computes from `runs[:2]` and is
      therefore contaminated (0.0096 vs the true 0.0200). **Recorded as a probe defect; the number
      used here comes from the clean pair.**
    - **EVAL = 1,402,560 decisions × 25 units × 0.020009 s = 194.9 GPU-h = $50.67–77.95.** The
      banked figure — **inferred by subtraction** as $30.98–48.69 — was **up to 2.5× too low.**
  - **★ THE SINGLE INTEGRATED PRICE: 270.4 GPU-h (eval 194.9 + forced-determinism training 75.5).
    $78 at the $0.29/hr we actually paid; $108 at $0.40; $120 if forced determinism also slows
    eval (unmeasured). ★ TOP UP AGAINST $150 ★** — carrying spot variance, the ~30 min of
    setup/staging observed per box, and one re-launch.
  - **(a) BUDGET → 26,003 STEPS, MATCHED AND FIXED.** 426,033,152 tokens = **20.00 tokens/param =
    1.399 passes** over the lake — the spec's own "1–3 passes", i.e. **implementing the blueprint**
    rather than exercising discretion. Per-cell val-NLL saturation is MEASURED AND REPORTED, never
    the stopping rule: data-dependent stopping gives 25 different budgets and confounds ΔIR(4−5)
    with *"cell 4 trained longer"* — the C-12 class, in the PRIMARY. **The spec-vs-`m6_design.md:18`
    conflict is reconciled here, dated, in favour of MATCHED.**
  - **(b) C-4 DROPPED AS BINDING.** `GATE_IS_BINDING = False`, with the three reasons, the
    resolvability numbers, and the required disclosure **encoded in code**, not prose. The binding
    path is retained and a KAT proves it still HALTs if re-armed — a dropped gate whose halt path
    had rotted would be undetectable. The withdrawal covers **all three marginals with a BSQ arm**
    (IR(2)−IR(1), IR(4)−IR(3), IR(3)−IR(1)); only IR(4)−IR(2) is clean; ΔIR(4−5) is unaffected.
    **The supervisor's direction nuance is recorded:** a weak BSQ *inflates* the first two in our
    favour, while IR(3)−IR(1) is BSQ-minus-BSQ and probably *attenuating* — so the
    "we-do-not-keep-a-favourable-defect" argument applies only to the first two, and claiming
    otherwise would overstate our own scrupulousness.
  - **(c) C-3 AMENDED, PRE-DATA.** `enumerate_dsr_trials` de-annualizes at `PRIMARY_H`. **The
    superseded mixed-unit basis is RETAINED and REPORTED** as a third `dual_specification` leg, so
    nobody takes our unit judgement on trust; disagreement is a first-class finding.
  - **(d) B1 PAID.** `set_determinism(seed, deterministic_algorithms=True)` on the production
    path. 13.175% measured, not the double-counted 1.3×.
  - **(e) THE CODEBOOK DIAGNOSTIC IS NOW REQUIRED AND GATED** — `codebook: dict` (was
    `dict | None = None` → `{}`), `_validate_artifact` rejects missing/empty and enforces the
    spec's own **≥95% utilization**, which existed nowhere in code. **C-2 shape, fourth instance,
    closed before we lean on it for the BSQ disclosure.** What it does and does not do is written
    at the pin: it discriminates COLLAPSED from NON-COLLAPSED and **not** crippled from competent —
    utilization is a property of the marginal distribution of code ids, so a tokenizer assigning
    codes by hashing noise would score ~100%.
  - **A BUILDER DEFECT, DISCLOSED:** my first fixture sweep matched only `ast.Name` call sites and
    missed `V.write_cell_eval_artifact` (an `ast.Attribute` call) — the same class as the
    manifest-sweep extractor hole. Caught by the suite, fixed by widening the matcher to both
    forms. An earlier line-heuristic version of the same edit corrupted a multi-line import and was
    reverted; the AST version re-parses every file before writing.
  - **THE `destroy` TRAP FIRED AND THE STANDING RULE CAUGHT IT:** `vastai destroy instance <id>`
    prompted, aborted, and **returned while the box was still running**. The re-list is why it was
    caught. Destroyed with `-y`; re-list `[]`.
  - Suite **583 → 583** (contracts changed, count unchanged). Ruff clean. Gate-A anchor
    **3f86882a re-proven byte-identical ×2, exit code 0 both**.

- **v1.6.20 (2026-08-03, THE CONSOLIDATED DECISION BRIEF + one banked cost figure WITHDRAWN.
  Nothing implemented; no budget set, no gate dropped, no invariant amended, no weights pulled.)**
  Local, $0.
  - **`docs/m6_decisions_pending.md`** — the four pending rulings in one document, because they
    interact. Receipt `runs_manifest/m6_recost_worstcase.json`.
  - **★ THE SATURATION SUBSTITUTION SURVIVES FOR ONE FAILURE MODE AND FAILS FOR THE OTHER, AND
    THE OTHER IS THE ONE THAT TOUCHES OUR SECOND CLAIM.** Attacked before agreeing, as ordered.
    It *does* cover "Cell 1 is weak because it was under-trained" — the most likely cause, and
    saturation answers it self-certifyingly. It does **NOT** cover *"our BSQ tokenizer is worse
    than the reference BSQ tokenizer"* — **and Kronos-small IS a BSQ model.** Cell 1 is BSQ+OHLCV;
    the §5 fallback claims **IR(2) − IR(1)**, i.e. FSQ minus BSQ. **The external gate was in effect
    the only control anchoring our BSQ implementation to a reference BSQ one, and a
    saturated-but-weak BSQ tokenizer inflates our own FSQ claim by exactly the amount it is weak.**
    Saturation cannot see it: a poorer token stream converges to a worse val NLL and saturation
    certifies "converged" — true, and useless. **Not hypothetical here:** the v6 canary measured
    the AR extracting ZERO nats from a planted conditional its tokenizer provably encoded. So the
    word *substitute* is withdrawn — it is a **narrower, different control**, and the disclosure
    sentence naming the uncontrolled confound must ship with it.
  - **★ A BANKED COST FIGURE IS WITHDRAWN, AND IT IS OURS.** The C-4 retrain contingency was
    banked as *"+1× the training leg, i.e. roughly doubling worst-case spend"* → **~$66–100 /
    ~$86–130 forced**, and has been quoted in every budget discussion since. **That holds only if
    training is ~half the spend; measured, it is 2.6–6.1%.** Corrected: **+$1.51–2.32** at the
    current budget, **+$19.62–30.19** at the raised one. **Not a doubling.**
  - **THE SINGLE RECOSTED WORST CASE**, on measured 4090 throughput at the money surface and the
    measured 13.175% determinism penalty (applied to TRAINING only, because that is where it was
    measured): **recommended pair (budget raised, B1 forced, C-4 dropped) = $51–79 → top up
    against $79**; if C-4 stays binding, **$70–109 → $109**. The **eval leg is INFERRED**, not
    measured (approved band minus measured training = $30.98–48.69), and cannot be measured
    without spend — the recipe's `chunk=512` OOM'd locally and the only local datum is a `chunk=64`
    floor on other hardware.
  - **★ THE INTERACTION THAT MAKES IT ONE DOCUMENT: approving the cheap budget silently forecloses
    the recommended C-4 resolution.** The saturation criterion needs eval-interval 5,000 ×
    patience 4 = **≥25,000 steps** to fire; at 2,000 steps **the spec's first evaluation never
    happens**, so the early-stop machinery would be **inert even if written**. Order: C-3 (free,
    time-critical) → budget (unlocks C-4) → C-4 → B1 (price then known).
  - **THE AUTHORITY CORRECTION IS RECORDED AS THE SUPERVISOR'S**, accepted by them: "no prereg
    constant states it, therefore a budget decision" was right about the freeze and **wrong about
    the authority** — the design spec states the budget and docs/ENGINEERING.md makes the spec a source of
    truth, so **raising it IMPLEMENTS the blueprint and leaving 2,000 is a standing undisclosed
    DEVIATION.** Also theirs, found while verifying: the spec's Stage-2 **eval interval is every
    5,000 steps against a 2,000-step budget — the designed schedule's first evaluation never
    fires.**

- **v1.6.19 (2026-08-03, THE TRAINING BUDGET IS A SMOKE-TEST CONSTANT — CONFIRMED, COSTED, HELD +
  G-§8.C.3 IS ILL-POSED. Nothing set, nothing implemented; one provenance warning landed.)**
  Local, $0. No weights pulled.
  - **★ THE SUPERVISOR'S FINDING IS CONFIRMED, AND MY REFUTATION ATTEMPT MADE IT STRONGER.** I
    tried three ways to refute it and all three failed.
    - `steps_stage1 = steps_stage2 = 2000` is the design spec's **G1 overfit-a-single-batch SMOKE
      GATE** threshold (`:1762-1763`, *"within ≤ 2000 steps"*). No sizing argument exists anywhere
      in design, prereg, roadmap or commit history.
    - **It entered in commit `64f728c` in the SAME dataclass literal as `seeds = (0,1,2)` and
      `seq_len = 128`** — the two values C-6 later identified as the train/eval split-brain. Those
      two had pinned counterparts and were corrected; the step budget had none, so it survived, and
      **§7 v1.6.15 C-18 then pinned it.** There is now a mutation KAT defending a rehearsal value.
      **This is the C-6 mechanism, third instance.**
    - **AND THE HONEST VERSION IS SHARPER THAN "NEVER DESIGNED": IT WAS DESIGNED AND THE CODE DOES
      NOT IMPLEMENT IT.** The spec specifies Stage-2 *"1–3 passes over a ≤1B-bar corpus"* at
      *"≈0.5M tokens/step"* with early-stop on val-NLL saturation (`:1912`, `:1924`); the code runs
      **0.108 epochs at 16,384 tokens/step** — **9–28× short on epochs, 30× on batch**. Stage 1 is
      30–60× short of its own designed 1–2B bar-reconstructions. That is worse than an unexamined
      default, because a reader of the spec has every reason to believe the budget exists.
    - **CORROBORATION:** the designed 1–3 passes over our 304,625,181-bar lake is **14.3–42.9
      tokens/param**, which BRACKETS the ~20 compute-optimal point. **The design's budget IS
      compute-optimal; the code's is 7.69% of it.**
  - **★ THE COST SPLIT, MEASURED — THE SUPERVISOR'S INTUITION WAS RIGHT AND IS NOW A NUMBER.**
    Throughput is measured on a **real 4090 at the exact money surface** (`seq_len=512`,
    `batch=32`; `m6_cuda_probe_cell_manifest.json`): Stage 1 **16.7859 steps/s**, Stage 2
    **3.2967 steps/s**. **Training all 25 units costs 5.04 GPU-hours = $1.31–2.02 — 2.6–6.1% of
    the approved $33–50. Eval is the other ~94–97%. We are spending ~95% of the budget evaluating
    a model we trained for 3% of it.** The eval leg **cannot** be costed without spend, precisely:
    the recipe pins `chunk=512`, which **OOM'd locally** (19.69 GiB KV cache > 20.13 GiB), and the
    only local datum is a `chunk=64` FLOOR on `mps` labelled *"PENDING the 4090"*.
  - **THE CURVE, FOR LAKSHAY TO CHOOSE A POINT ON** (`docs/m6_training_budget_decision.md`,
    receipt `runs_manifest/m6_training_budget.json`): **13× the training budget — 2,000 → 26,003
    steps, i.e. 7.7% → 100% of compute-optimal and 1.40 passes over the lake — costs +$15.73 to
    +$24.20**, taking the run from $33–50 to ~$49–74. **Because eval dominates, the training
    budget is cheap to fix.** Builder's view, on a decision that is not his: take it, because at
    0.108 epochs **a flat ΔIR(4−5) cannot distinguish "microstructure does not help" from "we
    stopped before it could"** — the false-NULL mode ruled Tier-1 blocking for C-10, at the scale
    of the whole experiment. **HELD. No budget set: no prereg constant states it, so this is a
    BUDGET decision, not a specification change, and it does not reopen the freeze.**
  - **ONLY VISIBILITY WAS CHANGED, AS AUTHORISED:** `conformance.PINNED_STEPS_STAGE1/2` now carries
    a provenance warning naming the smoke-gate origin, the commit, the designed budget it
    displaced and the 7.69% figure. The value, the pin and the KAT are untouched.
  - **★ G-§8.C.3 — RESOLVABILITY SETTLED, AND THE GATE IS ILL-POSED FOR A MORE BASIC REASON.**
    Receipt `runs_manifest/m6_c4_resolvability.json`.
    - **WHICH FIGURE: there are TWO, differing by 2.4×.** Paper Table 2, `Kronos_small`: RankIC
      **0.0254** (price-series) and **0.0622** (return-forecasting) → thresholds 0.02159 vs
      0.05287. **The spread is far larger than the 15% band the gate is made of**, and the prereg
      does not say which.
    - **AND BOTH ARE ON SHANGHAI STOCK EXCHANGE, 15-MINUTE BARS** — equities, not crypto; 15-minute
      bars, not our 1-minute bars. **The gate's two requirements cannot both hold:** run Kronos on
      our slice and you get a NEW number, not the published one; use the published number and you
      are comparing crypto-1m against equities-15m.
    - **THIS ALSO KILLS STEPS 1–2/4 AS A METRIC CROSS-CHECK, INDEPENDENTLY.** Validating our
      metrics by reproducing Kronos's published numbers would need **SSE 15-minute equity bars**,
      which docs/ENGINEERING.md firewalls out of v1. **The external metric check as specified is unavailable
      at any price** — not merely unscheduled.
    - **THE BAND ITSELF IS RESOLVABLE, AND MY OWN SPECULATION IS WITHDRAWN.** Measured with the
      project's `ic_screen.effective_sample_size` on the pinned 40 symbols over the pinned region:
      raw 1,402,560 stride-15 periods, **N_eff 1,401,637 (deflation ratio 0.999)**, SE(RankIC)
      **0.00084**, band/SE **4.51** (price-series) and **11.05** (return). Autocorrelation is not
      the binding term. Caveat stated because it cuts against the conclusion: summing per-symbol
      N_eff assumes cross-sectional independence, so this is an UPPER BOUND — at ~3–5 independent
      factors the band is 1.3–1.6 SE (marginal) / 3.1–3.9 SE (comfortable). **I flagged sampling
      noise as possibly decisive in v1.6.18; measured, it is not.**
    - **Ruling 2's destination reached by a different route: no further C-4 implementation work
      should proceed until the gate is re-specified**, which is a prereg/ROADMAP change and
      Lakshay's, alongside the invariant-8 question. Four candidate re-specifications are tabled.

- **v1.6.18 (2026-08-03, THE AUDIT PERSISTED VERBATIM + C-4: THE BINDING ENTRY GATE THAT HAD NO
  IMPLEMENTATION. One gate written and wired, fail-closed; nothing decided that is Lakshay's.)**
  Local, $0. **No weights were pulled** — metadata and licence text only, per the hard stop.
  - **THE AUDIT IS NOW IN THE REPO, VERBATIM AND WHOLE.** The supervisor recovered all 24 findings
    from transcript `780b64e7` and it is folded into `docs/m6_readiness_audit_findings.md`:
    **PART 1 the auditor's text unedited, PART 2 our disposition**, with the boundary
    machine-enforced rather than merely marked. `tests/test_audit_findings_complete.py` pins the
    **sha256 of the verbatim region**, so tidying a typo, correcting a drifted line number, or
    letting our disposition leak inside the markers all fail the suite. **Proven capable of
    failing four ways**: deleting a body (3 failures), a one-word tidy (1), a line-number
    "correction" (1), and smuggling a disposition heading inside the markers (2). The recovered
    scratch file was deleted. **The supervisor's own defect — never committing it at the time — is
    recorded in PART 1's provenance header, on the same terms as the builder's.**
  - **★ C-4 — THE GATE IS NOW IMPLEMENTED AND `BLOCKED`, WHERE IT WAS ABSENT.**
    `src/trikaal/eval/external_validation.py`. It fires inside `assemble_verdict` **before any
    between-cell Δ exists**, and `money_verdict` defaults to **True** so the dangerous direction is
    the one that must be declared (the C-6 rule); `scripts/m6_verdict.py` reuses the existing
    `--allow-toy-grid` fixture declaration rather than inventing a second one that could drift.
    The three pre-committed responses (`HALT_BEFORE_ANY_DELTA`, `CELL1_ONLY_FIX`,
    `FULL_5_CELL_SAME_SEED_RETRAIN`) are **encoded, not described**. **Demonstrated: a money
    verdict HALTS today; a declared fixture assembles; a supplied passing reference clears it** —
    so the HALT is not vacuous.
  - **LICENCE (read, not recalled): Kronos is MIT, `Copyright (c) 2025 ShiYu`**, quoted verbatim in
    the requirements doc. The HF model repos are **public and ungated**; **an unauthenticated pull
    works** (HTTP 200, `gated: False`), weights are 94.4 MiB + 15.1 MiB. **NO TOKEN IS NEEDED and
    none should be issued.** Cost: **$0, local, no rental** — Kronos-small is `d512/8L/8H/ff1024`,
    the same class as our backbone. **The licence is not the constraint; invariant 8 is, and it is
    stricter than MIT.**
  - **★ AND THE BLOCKER NOBODY HAD NAMED, WHICH IS WHY THIS IS NOT A SCHEDULING FAILURE.** The
    published weights are a **bare `state_dict`** — the model card's own loader is
    `from model import Kronos, KronosTokenizer`. **Running them requires Kronos's model code, and
    invariant 8 permits Kronos WEIGHTS in the eval harness but Kronos CODE nowhere.** The prereg
    further requires *"fed Kronos's own input pipeline"*, which is more of the same code. **Steps
    1–2 and 4 are therefore not merely unscheduled — they are unexecutable as specified**, and the
    four options (vendor into a quarantined path / reimplement from the paper / compare published
    numbers only / re-specify the gate) are **Lakshay's**, like B1. The gate surfaces this as its
    own `blocking_question` rather than reporting a vague "not configured", and a KAT asserts it.
  - **STEP 3's PROBABILITY QUESTION — ANSWERED WITH A REFUSAL AND THE REASON.** The hard number
    from our own pins: Stage-2 sees **32,768,000 tokens = 1.54 tokens/param = 7.69% of
    compute-optimal**, **0.108 epochs** over the lake. Three forces pull in different directions:
    the budget (**against us, and it dominates**), domain specialization (**for us, and not
    small** — Kronos-small is multi-market, Cell 1 is crypto-1m and is scored on crypto), and
    **resolvability**: at RankIC 0.02–0.03 the 15% band is 0.0030–0.0045, needing **n_eff ≈
    49,000–111,000**, against **35,064 stride-15 periods per symbol** in a correlated 40-symbol
    cross-section. **A gate whose band may sit inside its own sampling error is not yet a gate.**
    I decline to give a calibrated probability and name what would make it knowable — (i) identify
    *which* published figure and slice, $0; (ii) compute the deflated effective N and ask whether
    the band is resolvable at all, $0; (iii) a scaling probe at 2–3 reduced budgets, a few dollars.
    **The consequence, stated plainly for Lakshay BEFORE he funds anything: on the budget term
    alone, a miss is more likely than not, which makes the ~$66–100 (~$86–130 forced-deterministic)
    retrain the EXPECTED PATH rather than a contingency.**
  - Suite **543 → 583**: +28 audit-completeness, +12 external-validation. Coverage, not fixtures.

- **v1.6.17 (2026-08-03, TRANCHE 3 — THE CLAIMS THE PAPER WILL MAKE. Prose corrections at named
  sites, one numerical routine rewritten for provenance, two values pinned that were only ever
  described. No clause touched; the MDE is BIT-IDENTICAL across the rewrite.)** Local, $0.
  - **★ THE MICROSTRUCTURE CLAIM WAS WRONG IN THE TWO MOST-READ SENTENCES IN THE PROJECT.** The
    design spec's contribution sentence and `docs/ENGINEERING.md`'s architecture line both named *"trade-flow
    imbalance (TFI), funding, open interest"* as what the per-bar vector carries. **Funding and
    open interest are constant-zero and masked on 100% of the 304,625,181 bars we publish from**
    (§7 v1.6.16). The masking DECISION is documented and sound (the OI retention trap,
    `milestone4b_universe_ingest.md:129`); the CLAIM built on top of it was not. Corrected at all
    three named sites — design spec §contribution, `docs/ENGINEERING.md` (both the tokenizer line and the
    data-pipeline line), and `paper_skeleton.md` §2 as a **binding** note. What is now stated: the
    v1 microstructure leg is **TFI plus the aggTrades trade-flow statistics, six live dims (7–12)**;
    funding/OI are **specified and wired at dims 13–15 with a tripwire, structurally absent**; the
    **input is 16-wide with 13 live dims**.
    - **THE BUILDER AGREES IT MAKES THE CONTRIBUTION CLEANER, NOT WEAKER**, and the reasoning is
      recorded rather than asserted: *free microstructure* becomes **literally** true (aggTrades
      dumps are free; funding/OI need the futures API), the claim now matches the evidence that
      exists, and a referee who checks the data finds the description accurate instead of finding
      two of three named channels empty. **The one thing that IS narrower** is the contribution's
      breadth — "microstructure including derivatives-market state" becomes "trade-flow
      microstructure". But the narrowing is in the DESCRIPTION only: funding/OI were never
      measured, so no evidence is lost.
    - **AND THE MEASUREMENT THAT COULD HAVE MOVED THE PRIMARY CAME BACK CLEAN.**
      `block_time_permute` permutes **exactly dims 7–12** — verified empirically, not read:
      `MICRO_DIMS = (7,…,12)`, `PERP_DIMS = (13,14,15)` excluded, and `assert_perp_dims_masked`
      fires if funding/OI ever activate (confirmed by planting an active bit). **Cell 5 is NOT
      shuffling constants, and the C-12 capacity disclosure STANDS unchanged** — its "six dims of
      independent noise" wording is exactly right, where a nine-dim story would have been wrong.
  - **THE CLAIMS SWEEP IS NOW REPRODUCIBLE FROM A CLEAN CHECKOUT — a supervisor-found defect.**
    The 2026-08-02 regeneration globbed the directory and swept `m6_prefill_zero_mean.json`, which
    is UNTRACKED, so a fresh clone produced different counts: **we replaced a STALE receipt with a
    NON-REPRODUCIBLE one, same class, one turn later.** Now `git ls-files runs_manifest/*.json`.
    **Restricting to tracked files FIXES reproducibility; persisting the file list only DOCUMENTS
    it** — so both are done, in that priority: the input set is a function of the commit, and the
    receipt additionally carries the **sha256 of all 63 swept files** so a reader can verify the
    input rather than trust it. **Corrected counts: P1 33 (was 34), P2_total 76 (was 80),
    P2_HIGH 41 (was 43), P2_MEDIUM 0 (was 2).** The real P2_HIGH is **41**.
  - **S-4 — `tdist._betacf` REWRITTEN FROM THE PUBLISHED DEFINITION.** Receipt
    `runs_manifest/m6_s4_betacf_rewrite.json`. The predecessor was *Numerical Recipes* `betacf` in
    structure (`qab/qap/qam/c/d/h/aa/m2`, forward modified-Lentz). No licence violation is
    asserted; a released artifact should not carry a transcription when the mathematics is public.
    The replacement takes its coefficients from **A&S 26.5.8 / DLMF 8.17.22** via a named
    `beta_cf_coefficient(a,b,x,k)` and sums the fraction by **BACKWARD recurrence at doubling
    depth to convergence** — a different algorithm for the same quantity.
    - **EQUIVALENCE PROVEN, NOT ASSUMED:** `betainc` worst relative difference vs the predecessor
      over a 252-point grid **3.275e-15**; **the MDE clause is BIT-IDENTICAL** (`delta_ir`,
      `mde_paired 5.013646441007843`, `pass`), and **both pinned MDE multipliers are bit-identical**
      (S=3 → 3.980645752134, S=5 → 3.072811363562).
    - **A PRE-EXISTING RESIDUAL, ATTRIBUTED AND NOT INTRODUCED:** `student_t_ppf` sits ~1.1e-8 from
      the published value at (0.99, df=30) and ~6.4e-7 at (0.95, df=1000). **The predecessor
      produced 2.457261542401 at (0.99, df=30) — digit-for-digit the rewrite's value**, so this is
      the ROOT FINDER, not the continued fraction. The design operates at Welch df 2–4 where
      agreement is ~1e-12. Recorded, not fixed.
    - **85 KATs** at named reference points (closed forms `I_x(1,1)=x`, `I_x(a,1)=x^a`,
      `I_x(1,b)=1−(1−x)^b`, `I_½(a,a)=½`, the reflection identity; published t-quantiles at
      df 1–10), with the public callables named so an INDEPENDENT implementation can be compared
      against the same set.
    - **RoPE attribution added** on `_rotate_half`/`apply_rope` — the standard LLaMA/HF
      formulation (Su et al., RoFormer, arXiv:2104.09864), effectively forced for interoperability,
      recorded because invariant 8 is stated strongly.
  - **C-19 — THE BSQ COUNT IS NOW AN ASSERTION.** `test_orchestrator_m6.py` bounded it only by the
    ±2% band, so **21,231,616 appeared in the audit and in reports while no artifact pinned it** —
    in the very pass whose headline was about things that pass without measuring. Now asserted
    exactly, with `21_301_248 − 21_231_616 == 69_632`. `docs/ENGINEERING.md` quotes both numbers.
  - **S-3 — the non-independence is now IN THE SHIPPED DISCLOSURE**, not only in an adjudication:
    `decode_agreement_disclosure` carries `not_independent_of_the_expectation_estimator`, recording
    that the expectation decode's delta-method error is evaluated in the **same** out-of-distribution
    single-bar regime C-1 measures, so the two channels compound.
  - **TWO LIMITATIONS PROMOTED FROM RECEIPT TO PAPER** (`paper_skeleton.md` §10, binding): the
    embargo's signed-return-only justification **with the residual in the builder's own words**
    (*"a leakage channel that isn't signed-return autocorrelation wouldn't be caught"*), and the
    one-calendar-year headline with no cross-year replication. A third was added: that two of the
    eight causal surfaces were never exercised on the published lake until after it was built.
  - **THE ACF SCOPE WAS A BUILDER DEFECT, DISCLOSED AND FIXED.** The v1.6.16 probe took
    `all_symbols[:40]` — the first forty **alphabetically** — and the report called them "the
    pinned primary cross-section size". They were not the pinned set. Now read from
    `m6_mde_inputs.json` **and** measured across all 200. **It was never a cost decision: all 200
    takes 7 seconds.** Signed ACF at lag 60: **pinned 40 → 0.00717 mean / 0.01366 worst; all 200 →
    0.00667 / 0.01908.** Premise supported on both.
  - **THE AUDIT IS PERSISTED, WITH A NAMED GAP** — `docs/m6_readiness_audit_findings.md`, marked
    external and provenance-tagged per entry (VERBATIM / SUMMARY-ONLY / PARTIAL / MISSING). **The
    C-1..C-9 and C-11..C-20 bodies never reached the builder and are NOT reconstructed**, because
    a fuller-sounding entry would be the builder writing the audit — the exact failure the file
    exists to prevent. Slots are open; the disposition table closes all 24 findings by ID.
  - Suite **458 → 543**, the entire +85 being `tests/eval/test_tdist_reference.py` — coverage, not
    fixtures. Ruff clean. Gate-A anchor re-proven byte-identical ×2, exit code 0 (`src/` changed:
    `tdist.py`, `verdict.py`, `attention.py`).

- **v1.6.16 (2026-08-02, TRANCHE 2 — THE TWO LEAK DETECTORS THAT HAD NEVER BEEN POINTED AT THE
  REAL LAKE, plus the VACUITY class. One function DELETED, one guard fixed, three receipted
  measurements. No pinned value moved, no clause touched.)** Local, $0.
  - **★ C-15 (b) — THE HEADLINE FINDING. THE SWEEP THAT ADMITTED THE LAKE COULD NOT FAIL ON TWO OF
    ITS EIGHT SURFACES.** Receipt `runs_manifest/m6_c15b_lake_surface_check.json`. Measured over
    **ALL 200 symbols and all 304,625,181 bars — no sampling**, because the density read did the
    whole lake in 2.2 s and a sample would have been a choice to know less for no saving.
    - `m4b_universe_ingest.py` sweeps an **800-bar head slice** (`--sweep-sample-bars`, default
      800) under production `FeatureConfig()`, whose `effective_n_warm_vol()` is **1440**.
      `target_valid[t]` is True only past that warm-up. **Reconstructed from the persisted
      `segment_id`/`bar_open_ms`: `target_valid` is True for 304,174,569 of 304,625,181 bars
      (99.85%) across the lake — and for ZERO bars inside the sweep's own 800-bar slice, on
      0 of 200 symbols.** So `target` and `target_valid` compared equal under every truncation and
      perturbation: two of the eight output surfaces were structurally unable to fail, and the
      `n_checks > 0` guard passed on the other six.
    - **And the scope is narrower than the guard implies:** the sweep runs on **exactly two
      symbols** (`if label not in sweeps` — one "live", one delisting-"tail"), 800 bars each. That
      is **1,600 bars of 304,625,181 = 0.00053% of the lake**, and the results are **printed to
      stdout and never persisted** — the ingest ledger records only checksums, dataset_hash,
      event, n_bars, symbol.
    - **THE LAKE DATA ITSELF IS CLEAN, and that is a different statement.** Every surface the
      sweep names is non-degenerate over the full lake. At finer per-COLUMN granularity than the
      sweep uses, `x_13/14/15` have stddev exactly 0.0 and `m_13/14/15` are masked on 100% of
      bars — **the documented funding/OI "carried-but-masked" design** (`features.py:86-89`,
      `arms.py:36`), since M4b ingested klines+aggTrades only. Not a defect, but it means the
      **16-dim micro arm is 13 live dims plus 3 constant-zero masked ones**, and the paper must
      say 16 with that qualification. `m_2/3/4` are never masked anywhere (frac 0.0).
    - **BOUNDARY STATED IN THE RECEIPT:** a non-degenerate surface is a PRECONDITION for a leak
      check to mean anything, not evidence that the surface is leak-free. The lake CAN be checked;
      it had not been, on those two surfaces.
  - **★ C-15 (b) PART 2 — THE GAP IS NOW CLOSED, ON REAL BARS, AT PRODUCTION PARAMETERS. GREEN.**
    Receipt `runs_manifest/m6_c15b_production_sweep.json`, `$0`, exit code 0. Finding the gap and
    leaving it open would have been half the work, and the raw material was on disk: a real
    BTCUSDT month from `processed/shards/` (the same reduction the ingest itself used), swept at
    production `FeatureConfig()` on **1,600 bars — above the 1,440 warm-up**, so every surface is
    live (**159 valid targets**, vs 0 in the ingest's 800-bar slice).
    - **Clean baseline PASSES: 1,600 anchors, 12,798 checks, coverage 1.0, zero failures.**
    - **CONTROL ARM: both localized leaks are CAUGHT** — `localized_validity_next` flags
      `target_valid`, `localized_sigma_next` flags `sigma`. **`target_valid` is precisely the
      surface that could not fire in the sweep which admitted the lake**, so the check is now
      demonstrated to discriminate exactly where it previously could not.
    - **AND THE PROBE CAUGHT ITS OWN BUG FIRST, WHICH IS THE THIRD TIME THIS TRANCHE.** The first
      run planted at `warm + 200` = bar **1640 on a 1600-bar slice** — outside the swept range —
      and duly reported both leaks "MISSED" with a ★ FINDING verdict. That was the probe failing,
      not the gate. The plant is now asserted to sit inside the slice and past the warm-up before
      anything is read. Had it been reported as written it would have been a false
      invariant-2 alarm on a lake that is fine.
    - **SCOPE, STATED NOT IMPLIED:** one symbol, one month, 1,600 bars. The four GLOBAL planted
      variants were dropped from this run (each is a separate O(T²) sweep and they are already
      exercised non-vacuously by the CI gate at reduced parameters) — a bounded scope, recorded
      rather than silently trimmed. A DATA-DEPENDENT leak on some other symbol remains out of
      scope; the structural argument (a lookahead defect in `compute_features` is transform-level,
      not symbol-level) is what carries the rest, and it is an argument, not a measurement.
  - **C-20 leg 1 — `embargo_flatness` DELETED. My ruling, with the argument and the evidence.**
    - **Wiring it in costs 3× the M6 run.** It requires the headline at E ∈ {60,120,240}, and the
      embargo binds `fold_valid_starts`, which gates which TRAINING windows are legal — so each E
      needs its own training. ~$99–150 against a design with no such leg and an approved $33–50.
    - **The premise it defends is now MEASURED rather than assumed**
      (`runs_manifest/m6_c20_embargo_premise.json`, 40 symbols, control arm recovers a known
      AR(1)): signed-return autocorrelation is **+0.0061 mean / 0.0127 worst-symbol at lag 60**
      and already ≈0 by lag 5, so `L_corr = 60` carries roughly a **24× margin** over the channel
      a purge/embargo must outrun. |return| ACF is 0.20 at lag 60 and 0.15 at lag 240 — long
      memory, but that is volatility clustering, not a label-leakage channel, and an embargo sized
      to it would run to days. Both reported so the distinction is visible rather than assumed.
    - **What remains is STRUCTURAL and binds every window:** `fold_valid_starts` admits a window
      only if its last bar opens strictly before `boundary − 120 bars`, at load time, not sampled.
      Its test was replaced: the old `test_embargo_flatness_gate` asserted against a hand-written
      dict of IRs no run ever produced; the new one asserts the bound on real windows, proves the
      fixture yields some legal windows first, and proves a tighter embargo admits strictly more —
      so the constant is demonstrably READ.
    - **WHAT IS LOST, STATED PLAINLY:** end-to-end flatness of the headline IR in E is now
      ASSERTED from the premise rather than DEMONSTRATED. A leakage channel that is not
      signed-return autocorrelation would not be caught by this argument. The instrument is
      recoverable from git history if the 3× run is ever funded.
  - **C-20 leg 2 — CONFIRMED, and it is a disclosure not a bug.** The headline primary region is
    **2024-01-01T18:00 → 2025-01-01T00:00 = 365 days, 0.9993 years**, i.e. exactly one calendar
    year: the 4-year window at `train_frac` 0.7 leaves ~1.2 years, block 0 is VAL, blocks 1–5 are
    the headline. **The cost-aware net IR therefore has no cross-year replication**, which belongs
    in the limitations beside the 40-symbol cross-section.
  - **C-20 leg 3 — CONFIRMED STALE.** Committed receipt: P1 **23**, P2_total **40**, P2_HIGH **5**.
    Re-run: P1 **34**, P2_total **80**, P2_HIGH **43**. **Zero hits disappeared** — nothing
    previously flagged was silently resolved; 51 appeared because manifests were added after the
    receipt was taken. Regenerated. Triage of the delta: 25 of the new HIGH hits are `stddev: 0.0`
    / `frac_masked: 0.0` in the C-15(b) receipt **written this same pass** (real measured zeros —
    the funding/OI dims), 8 are flip-rate zeros in the paused prefill manifests, 3 in the CUDA
    probe. **No new hit is a claim on the verdict path.**
  - **THE VACUITY SWEEP — `scripts/m6_vacuity_sweep.py`, receipt
    `runs_manifest/m6_vacuity_sweep.json`.** The supervisor's framing, which is right: *"deleting
    a pin deleted its own check"* and *"the empty matrix was certified a perfect cover"* are one
    shape — **an assertion that iterates a collection is vacuously satisfied when the collection
    is empty.** `for x in []` never runs; `all([])` is True.
    - **Coverage is AST-DERIVED, not recalled.** 35 candidate functions across
      `eval/run/train/data` that both iterate a collection and return a verdict-shaped value. Each
      is either probed (14) or excluded with a stated reason (27 entries incl. the newly
      accounted); **a candidate in neither is a hard error**, so a sibling cannot be dropped by
      forgetting to list it. First run had 4 unaccounted; now 0.
    - **Mutation control FIRED** (`micro_legibility_gate` wrapped to return `pass: True` on empty
      input is named by the sweep).
    - **1 genuine vacuity found: `provenance_failures({})` returned `[]`** — a divergence check
      over an empty set finds no divergence, so the gate certified "one instrument" over zero
      units. Fixed: zero units is now a named failure. **0 of 14 vacuous now.**
    - **AND THE SWEEP CAUGHT ITS OWN MIS-MODELLING AGAIN.** The first run also flagged
      `enumerate_dsr_trials({}) → {}`. That is a PRODUCER, not a check — `{}` for `{}` is correct,
      and its guard reports "300 missing" on the same input. Counting it would have inflated the
      finding with the probe's own error, exactly as three harness bugs did in the fail-open
      sweep. Reclassified with the reason recorded.
  - Suite **458 → 458**: one test deleted (`test_embargo_flatness_gate`, whose function is gone)
    and one added (the structural embargo test). Ruff clean.

- **v1.6.15 (2026-08-02, TRANCHE 1 — THE FAIL-OPEN CLASS SWEPT AND CLOSED. Guard/gate fixes, two
  values pinned that were already pre-registered, one test rewritten. No pinned VALUE moved and no
  clause was touched; every change makes a check harder to pass, never easier.)** Local, $0.
  - **THE SWEEP IS THE DELIVERABLE — `scripts/m6_failopen_sweep.py`, receipt
    `runs_manifest/m6_failopen_sweep.json`.** 14 functions × 6 degenerate inputs
    (absent/empty/None/NaN/inf/wrong-type) = **84 rows**, each recording the value actually
    returned and whether it FAILS OPEN (a quiet pass) or CLOSED (raised, or reported). Coverage is
    listed explicitly in the receipt, because *"we swept and found one"* and *"our sweep only
    looked at one"* produce the same table otherwise.
    - **MUTATION CONTROL FIRED.** A known-CLOSED guard — `degeneracy_guard`, the C-2 site, the one
      we most believe is fixed — is wrapped to return a quiet pass, and the sweep must name it. It
      named it on 6/6 rows. A sweep that cannot catch a regression at C-2's own site would be
      decoration, so the script exits non-zero rather than printing a table nobody should trust.
    - **12 FAIL-OPEN ROWS FOUND → 0 REMAIN.**
    - **THE SWEEP FOUND THREE OF ITS OWN HARNESS BUGS FIRST, AND THEY WOULD HAVE BEEN FALSE
      FINDINGS.** `provenance_failures` reads `meta.provenance`; my first registry mutated a
      top-level `provenance`, so the guard never saw a degenerate input and all six rows read
      FAIL OPEN. `artifact_reuse_failures` and `shard_partition_failures` were called with wrong
      signatures, so every case died of the same `TypeError` — both arms failing alike, which is
      `PROBE INVALID` by the control-arm rule, not a verdict. And
      `pinned_threshold_failures` takes no arguments, so six no-op rows inflated the count.
      **21 rows before the harness was correct; 12 after.** A defect count is worthless until the
      probe can tell the failure it is testing for from its own.
  - **S-2 CONFIRMED — `power_guard` carried a HARDCODED `armed: True`.** With non-finite per-seed
    IRs, `finite` is empty, every range is `None`, `worst` is `None`, `trips` is False, and the
    guard reported armed/not-halted having evaluated nothing. **This is C-2 verbatim, in the
    sibling guard, and it survived because the C-2 fix was applied at C-2's site.** `armed` is now
    a measurement, the unreadable units are NAMED, a non-finite CLAIM is treated as unreadable
    too, and the guard HALTs. Still HALT-only: it can never flip SURVIVES↔NULL.
  - **`decode_agreement_disclosure` silently dropped unreadable units** — a disclosure computed
    from nothing was indistinguishable from one computed from five seeds. It now reports
    `unmeasurable` with the named units, and `assemble_verdict` raises that to HALT_ADJUDICATE.
    `_validate_artifact` already refuses such artifacts at load, so this sat behind a closed
    door — **which is exactly how C-2 read until the rehearsal path bypassed it.**
  - **`pinned_threshold_failures` iterated the PIN dict, so DELETING a pin deleted its own check**
    and the gate returned clean. It now iterates the LIVE constants: every live threshold must
    have a pin, and a pin with no live constant is also a failure.
  - **`shard_partition_failures` certified the EMPTY matrix** — a partition of nothing is a
    perfect disjoint cover. In fan-out that is the dangerous case, not the trivial one.
  - **C-10 FIXED, BOTH LEGS, AND THE SUPERVISOR'S PARTIAL-SKIP CLAIM IS CONFIRMED NOT REFUTED.**
    Reproduced with a control: **5 thin + 1 dense passing dim returned `pass: True`**, while a
    failing dense dim still raised (so the fixture discriminates). The `continue` sat BEFORE
    `ok = ok and acc >= min_acc`, so a skipped dim could not lower `ok`. **Skipped is now a third
    state that HALTS**, naming the unmeasured dims.
    - **MATERIALITY MEASURED ON THE REAL LAKE (`runs_manifest/m6_c10_micro_density.json`, $0,
      DuckDB-aggregated under a 4 GB cap).** All 200 symbols, 304,625,181 bars — the recorded lake
      total, which is also a cross-check. **Every micro dim 7–12 clears the 10,000-bar floor by
      orders of magnitude: minimum ACROSS SYMBOLS is 277,758 and the totals are ~300.3–300.8M.**
      So the defect is real but **cannot fire on this lake** — which is what makes the halt-on-skip
      fix safe: it cannot halt a legitimate run at the universe we have.
    - **LEG 2 IS THE MATERIAL ONE AND IT IS WORSE THAN SUSPECTED.** `id_legibility_sign_acc` read
      the FIRST 150,000 rows of a symbol-ordered concatenation with a CONTIGUOUS 80/20 split.
      Measured: **that window spans 1 of 200 symbols.** The standing gate that decides whether
      Stage-2 spend proceeds was reading **one symbol** and calling it *"the run's real training
      stream"*. Now stratified by symbol — every symbol contributes — while keeping the split
      **BLOCKED IN TIME within each symbol**, because 1-minute bars are autocorrelated and an
      interleaved split would put near-duplicate neighbours on both sides and inflate accuracy.
      The contiguous cut had that right; it simply did not cover the universe. **Direction: the
      gate now measures 200× more of the universe and must hold across all of it — the TIGHTENING
      direction, the same argument as C-3.**
  - **C-18 — THE TWO OPEN LEGS PINNED. This does NOT reopen the freeze:** the μ estimator is the
    v1.4.2 pre-registration and the step budget is the recipe every receipt already reports. The
    code simply failed to enforce what was already pre-registered — the C-7 shape.
    `PINNED_MU_ESTIMATOR` and `PINNED_STEPS_STAGE1/2` now exist; `XSectionConfig.mu_estimator`
    is asserted in money mode AND at the conformance gate (defence in depth — a config built
    before the field existed, or mutated after construction, still reaches the gate), and
    `xsection.MU_ESTIMATOR` is cross-checked against the pin so the two copies cannot drift.
    `OrchestratorConfig` now REFERENCES the step pins and asserts them on a money run.
  - **A PRE-EXISTING TEST ASSERTED THE FAIL-OPEN BEHAVIOUR AS CORRECT, AND THAT IS WHY C-10
    SURVIVED.** `test_gate_skips_masked_everywhere_dims` asserted only that the dim was recorded
    as skipped, under the docstring *"never trivially passed"* — and passed while the gate
    returned `pass: True` on exactly that input. **The defect was not an oversight; it was written
    down as the contract.** Rewritten to assert the halt, to check the diagnostic survives the
    halt, and to prove the same fixture still passes unmasked.
  - **EVERY NEW KAT PROVEN CAPABLE OF FAILING** — `scripts/m6_failopen_kat_mutations.py`, receipt
    `runs_manifest/m6_failopen_kat_mutations.json`: 8 fixes, each run against a predecessor-
    equivalent implementation, each KAT required to FAIL there and PASS here. 8/8 discriminate.
  - Suite **439 → 458**, the entire delta being the 19 tests in the new
    `tests/eval/test_failopen_class.py` — coverage, not fixtures. Ruff clean. Gate-A anchor
    **3f86882a re-proven byte-identical ×2, exit code 0 both** (`src/` changed).

- **v1.6.14 (2026-08-02, THE C-3 AMENDMENT DRAFTED AND HELD + AUDIT TIER-4 TRIAGE + one standing
  norm. No pinned value moved, no clause touched, no code changed except two new verify-only
  probes.)** Local, $0. *(Numbering note: if the C-3 amendment is approved it lands as **v1.6.15**;
  this entry deliberately does not consume the number, so the amendment's own entry is the record
  of the change rather than a back-reference.)*
  - **C-3 RULED CORRECT BY THE SUPERVISOR AND THE FIX IS PRIMARY — DRAFTED, NOT IMPLEMENTED.**
    `docs/m6_c3_amendment_decision.md`. The unit-consistent basis (every trial de-annualized at
    `PRIMARY_H`) becomes primary; the pinned mixed-unit basis is **retained and reported alongside
    in `dual_specification`**, on the same footing as the v1.2/v1.5 legs, disagreement a
    first-class finding in the abstract. **It stops here** because this is the third reopening of a
    design twice declared frozen and it changes a clause threshold — that combination is Lakshay's.
    The decisive argument is recorded in the draft: **keeping the pinned basis is not neutral
    fidelity to the freeze, because the pinned basis is the EASIER bar** (SR₀ understated 8.70%).
    A freeze that protects an error in our own favour protects nothing worth protecting. The
    direction-blind test is not hypothetical here — **the fix TIGHTENS, it is proposed PRE-DATA,
    and it corrects a unit error rather than exercising a judgement.** Implementation is one token
    (`h` → `PRIMARY_H`), a new pinned convention key, a `dual_specification` leg, and a mutation
    KAT; $0.
  - **THE 5.8056e-4 IS RECORDED AS IRREPRODUCIBLE AND NOT CHASED**, on the supervisor's ruling.
    The finding rests on the span (bit-exact), the mechanism (from code) and outcome-materiality
    (proven by construction with a witness); the auditor's figure illustrated those rather than
    founding them. Closest construction agrees to **0.026% at |IR| = 7.56**; no repo value sits in
    the interval that would display as 5.8056e-4.
  - **NEW STANDING NORM — FIX THE CLASS, NOT THE INSTANCE** (docs/ENGINEERING.md, ninth in the family).
    When a defect is found at a named site, sweep every site of that class before closing it. Both
    citing instances came from the C-17 pass itself: §3 clause 5's body, half a sentence from an
    edit I made, and `enumerate_dsr_trials`'s docstring, in the very file whose module docstring
    that pass corrected. Neither was visible from re-reading the fix; both were found by building
    the sweep and running it. **Corollary: when the class needs a tool, the tool is the
    deliverable — and the tool is suspect until shown capable of failing.**
  - **TIER-4 TRIAGE — `docs/m6_tier4_triage.md`, receipt
    `runs_manifest/m6_tier4_vacuous_gates.json`.** Verdicts per finding, nothing fixed:
    - **C-10 CONFIRMED — the micro-legibility HARD STOP returns `pass: True` having measured
      nothing.** Every micro dim below the 10k-unmasked-bar floor is `continue`d BEFORE
      `ok = ok and acc >= min_acc`, so with all six thin the initial `True` survives and no
      `RuntimeError` fires. Reproduced: 6/6 dims `"skipped"`, `pass: True`, `raised: False`. The
      docstring's *"never trivially passed"* is the opposite of the behaviour. Second leg: the
      150k sample is the **head** of a symbol-ordered concatenation with a contiguous 80/20 split.
    - **C-15 CONFIRMED, and the obvious remedy is wrong.** The invariant-2 CI gate runs only at
      reduced parameters; at production `FeatureConfig()` the 1440-bar volatility warm-up exceeds
      the 600-bar fixture, so `target_valid` is **True for 0/569 bars** and the planted validity
      leak flips nothing at **0/20** bars — versus **375/569** and **14/20** at the reduced config
      (the control arm that makes the comparison mean anything). **Swapping the config would make
      the gate vacuous, not stronger**; a production-parameter gate needs a fixture longer than the
      warm-up, which is a test-cost decision and is NOT proposed here. Mitigation recorded:
      `m4b_universe_ingest.py:292` does sweep at production parameters on real shards, but its
      `n_checks > 0` guard proves the sweep RAN, not that each surface was non-degenerate.
    - **C-10 AND C-15 ARE THE C-2 PATTERN, WHICH WAS RULED TIER-1 BLOCKING.** A gate reporting a
      verdict while examining nothing. I am not re-tiering them — that is the supervisor's — but
      the tier looks wrong and it is said plainly rather than left in a table.
    - **C-19 CONFIRMED EXACTLY.** BSQ cells realize **21,231,616** backbone params vs FSQ's
      **21,301,248** — **−69,632 (−0.327%)**, the auditor's figure to the digit. The code knows
      (`build_cell_backbone` asserts a ±2% band for non-FSQ vocabs); **the prose does not** — we
      quote one number while invariant 4 requires a backbone "matched across every ablation arm".
      **Where it bites: the §5 fallback claim is IR(2) − IR(1), i.e. FSQ vs BSQ, so that claim is
      confounded with the parameter difference. The primary ΔIR(4−5) is unaffected** — both cells
      are FSQ at identical vocab.
    - **C-18: one leg CLOSED (interpreter, `provenance.py:68`), two CONFIRMED.** The Stage-1/2 step
      budget is a bare dataclass default — recorded in the manifest but **pinned nowhere**, and
      recording is not pinning. **The μ estimator is the sharp one:** `predict_mu(...,
      estimator="expectation")` is the v1.4.2 pin existing ONLY as a Python default, with no
      `PINNED_MU_ESTIMATOR` and no assertion anywhere — a caller passing `"argmax"` restores the
      biased decode v1.4.2 was written to remove and no gate notices. **The C-7 shape.**
    - **C-16 CONFIRMED, bounded:** `harness.py:83` hardcodes `q_over_adv=1e-3`, unpinned and
      unreceipted — but it feeds the **modelled-cost secondary**, never the 0.30%-netting headline,
      and `harness.py` is frozen, so it is report-only either way.
    - **C-14 CLOSED** by C-5/A6 (`matrix.py:201-211` binds `meta.checkpoints` plus seven more
      fields). **C-11 CONFIRMED but is B1**, already drafted and awaiting Lakshay.
    - **C-20: 1 of 3 CONFIRMED** — `embargo_flatness` (`folds.py:88`) has no caller outside its own
      test, i.e. a purge/embargo leak diagnostic that never runs. The calendar-year and
      claims-sweep legs were **not verified this pass and carry no verdict**.
    - **S-1 was PROMOTED into C-1** and is now a required real-cell disclosure. **S-2..S-4: I do
      not have their text and will not report verdicts on my reconstruction of them.** Answering
      "were any silently dropped?" from a paraphrase is precisely the failure that question exists
      to catch. Requested as written.

- **v1.6.13 (2026-08-02, AUDIT TIER-4 C-3 MEASURED AND REPORTED (nothing fixed, ruling pending) +
  the ruling-(a) MANIFEST PROSE SWEEP + the ruling-(b) OPERATIONAL CONSTRAINT. Two prose strings
  now derive from constants and render byte-identically; no pinned value moved, no clause touched,
  no measurement changed.)** Local, $0.
  - **C-3 — CONFIRMED IN PART, AND IT IS OUTCOME-MATERIAL. REPORTED, NOT FIXED.** Receipt
    `runs_manifest/m6_c3_dsr_units.json`, probe `scripts/m6_c3_dsr_units.py`.
    - **CONTROL ARM FIRST** (standing norm): the probe re-derives the toy manifest's own all-arms
      `var_sr` from its per-cell artifacts and requires a bit-exact match before comparing
      anything. `0.052821905816472746` re-derived == recorded. Had it differed, the probe emits
      `PROBE INVALID` and no conclusion about units.
    - **THE 3.46× SPAN IS CONFIRMED, EXACTLY.** `enumerate_dsr_trials` divides each annualized VAL
      IR by `sqrt(periods_per_year(h))`, and those divisors are 324.2221…/187.1897…/93.5948… for
      h = 5/15/60. Their ratio is **√12 = 3.4641016151377544**, bit-equal to `math.sqrt(12)`. One
      and the same annualized IR therefore enters the trial set **3.4641× larger at h=60 than at
      h=5**, purely from the horizon it was measured at. The trial set is not in one unit; it is a
      blend of three.
    - **THE UNIT MISMATCH IS REAL, FROM THE CODE.** `deflated_sharpe_ratio` turns `var_sr` into SR₀
      via `expected_max_sharpe`, and `probabilistic_sharpe_ratio` compares SR₀ against
      `_sharpe(headline_series)` — whose grid `_validate_artifact` pins to `PRIMARY_H = 15`. So
      **SR₀ is compared against a per-15-minute Sharpe while it is built from a per-5/15/60-minute
      blend.**
    - **THE 5.8056e-4 HAND-CALCULATION IS *NOT* REPRODUCED.** The closest construction I can
      identify is the pure unit-mismatch variance — every trial carrying the SAME annualized IR, so
      that all dispersion is the artifact alone: `var = IR²·Var({1/√ppy(h)})` with
      `Var = 1.0155267159845607e-05`. At |IR| = 7.56 (this project's own sign-saturated IR, quoted
      to 3sf) that gives **5.804100771469519e-04**, agreeing to **0.026%**; at the exact saturated
      value 7.563377128272953 it gives 5.809287430421818e-04. To *display* as 5.8056e-4 the
      construction needs |IR| ∈ [7.560944, 7.561009], and no value in the repo sits there. **I am
      not asserting the auditor is wrong** — the agreement is far too close for coincidence and the
      residual is intermediate-rounding scale. I am reporting that **I cannot reproduce the exact
      figure without the auditor's own arithmetic**, and the sensible next step is to ask for it
      rather than to adopt a number I could not rebuild.
    - **FIXTURE DISCRIMINATION, DECLARED (standing norm).** The only real multi-cell artifact set in
      the repo (the 3-seed toy) **cannot discriminate the clause-5 OUTCOME**: cell 4's DSR
      saturates at **0.0 under all four unit bases**, so its pass/fail column is uninformative and
      must not be read as "the mismatch does not matter". What it does discriminate is the INPUT —
      `var_sr` **0.0270932** as pinned vs **0.0325023** with every trial at h=15, i.e. the pinned
      basis is **0.8336×** the unit-consistent one and SR₀ is **understated by 8.70%**.
    - **SO THE OUTCOME QUESTION IS ANSWERED BY CONSTRUCTION INSTEAD.** With a deterministic n=1128
      unit-sd series whose mean is exactly SR̂, the minimum passing SR̂ is **0.43840** under the
      pinned basis and **0.47569** under the h=15-consistent one. **Witness SR̂ = 0.45705 gives
      DSR = 0.9868 (PASS) as pinned and 0.8577 (FAIL) unit-consistent.** The mismatch is therefore
      **outcome-material, not merely input-material**: a non-empty band of cell-4 Sharpes passes
      clause 5 under the frozen recipe and fails under a unit-consistent one.
    - **THE DIRECTION IS NOT SIGNED PRE-DATA, AND THAT IS THE WORST PROPERTY.** Converting to the
      h=15 unit rescales the h=5 group by 0.5774 and the h=60 group by 2.0 **about zero**, so the
      sign depends on the realized per-horizon means and spreads. On the toy it is
      **anti-conservative** (bar lowered). It could be conservative on the real cells. **An
      unknown-direction distortion of a gating threshold cannot be defended as "errs safe"** — the
      defence available for the C-12 tripwire is not available here.
    - **NOTHING WAS CHANGED.** §3 clause 5, `PINNED_DSR` and `enumerate_dsr_trials` are frozen and
      untouched; only the docstring gained the disclosure. **This is a NEEDS-RULING item**, and it
      is the one Tier-4 finding that moves a threshold rather than a description.
  - **RULING (a) — THE MANIFEST PROSE SWEEP. Receipt `runs_manifest/m6_manifest_prose_sweep.json`,
    probe `scripts/m6_manifest_prose_sweep.py`.** It sweeps the **emitted artifact**, not the
    source, across **four verdict branches** (planted-SURVIVES, NULL, placebo-harmed,
    fallback-claimed) — **162 strings** — because a string emitted only on the NULL path is still a
    shipped string. Every number in every string must resolve to a live pin, a value derived from
    pins, or an adjudicated non-recipe number.
    - **TWO STRINGS STATED A RECIPE NUMBER AS A LITERAL AND NOW DERIVE.** `PINNED_DSR["statistic"]`
      hardcoded *"the 0.30% flat netting"* → now `f"{PINNED_HEADLINE_COST:.2%}"`; the tripwire's
      `action_if_fired` hardcoded *"7 vs 16 dims"* → now `arm_n_features(ARM_OHLCV)` /
      `arm_n_features(ARM_MICRO)`. **Both render byte-identically**, so no manifest content moved —
      the change is that they can no longer drift from the cost the money path charges or from the
      widths the cells actually have.
    - **TWO NUMBERS ARE HISTORY, NOT RECIPE, AND ARE ADJUDICATED WITH A BINDING THAT CAN FAIL.**
      The decode disclosure's 1.10 / 1.82 / 2.62 (verified: 0.1071 ÷ 0.0973; t = 1.821; crit
      2.6226) and the degeneracy guard's worked triple (verified: (0.999+0.55+0.55)/3 = 0.69967 →
      0.700). The adjudication is bound to the exact number set read, so **any change to those
      strings re-opens them** — mutation-proven both ways.
    - **THE SWEEP FOUND A HOLE IN ITSELF FIRST, AND IT IS MINE.** My initial pattern ended in
      `(?![\w.])`, which **silently skipped every number glued to a letter** — `3x` (the micro
      weight) and `1.5x` (the dispersion tripwire), i.e. **the two recipe numbers most worth
      checking**. A sweep that cannot see the strings it exists to check is the *"a check that
      cannot fail"* shape again, in the tool built to enforce it. Fixed, disclosed in the source,
      and both values then verified against `PINNED_MICRO_POINT_WEIGHT` and
      `PLACEBO_DISPERSION_TRIPWIRE`. The probe also carries its own control arm (a known-stale
      string must flag, a current one must not) and emits `PROBE INVALID` if it cannot separate
      them.
    - **RESULT: 0 unverified prose numbers across 162 strings on 4 branches.**
  - **TWO STALE STRINGS OUTSIDE THE MANIFEST, BOTH MISSED BY MY OWN TIER-3 PASS. DISCLOSED.**
    - **§3 clause 5's own body** still read *"var_sr = the variance of the **180** recorded
      per-trial de-annualized IRs (**every cell** × seed × horizon × κ …)"* — the superseded
      all-arms basis at the superseded 3-seed count. **The C-9 fix corrected `N = 180 → 60` two
      lines above and stopped.** Both halves of one sentence, one edit apart. Corrected to the
      cell-5 placebo basis (60 values) with the 300-entry enumeration named as the audit trail.
    - **`verdict.py:509`'s `enumerate_dsr_trials` docstring** read *"exactly the 5 × 3 × 3 × 4 =
      180 cross product"* — the pre-v1.5 3-seed count — and it survived the C-17 pass **that
      corrected this very file's module docstring**. The lesson is the one the sweep exists to
      encode: **C-17 was fixed at the strings the audit named, not at the file.**
    - Neither was ever read by code; both were false accounts of the recipe in the two documents a
      referee reads first.
  - **RULING (b) — OC-1, THE S=3 CONSTRAINT, RECORDED AT ITS SITE (§7 v1.5 item D).** Not "closed":
    if the budget runs short mid-run the options are **fund it** or **declare INCONCLUSIVE per
    R3** — never drop to three seeds, which would now mean moving `PINNED_SEEDS` post-hoc under
    freeze. **Recorded together with the compounding**, because separated each reads as survivable:
    C-4's retrain contingency takes worst case to **~$66–100 / ~$86–130 forced-deterministic**, and
    that is precisely the scenario in which someone would reach for the cheaper seed budget. **No
    reachability restored** — the C-6 assertion is correct and the freeze is worth more than the
    hatch.
  - **C-8 stands REFUTED as stated** (v1.6.12), recorded as a refuted finding with the reasoning
    rather than as a silent correction. The audit was right 19 times and wrong once; the once is
    recorded in both directions.

- **v1.6.12 (2026-08-02, AUDIT TIER-3 — PRE-REGISTRATION INTEGRITY. Documentation corrections; NO
  pinned value moved, NO clause touched, NO behaviour changed except one manifest string that now
  derives from the pins instead of restating them.)** Local, $0.
  - **C-9 — THE §3/§3a BODY STILL CARRIED THE SUPERSEDED NUMERICS, FIXED.** The amendment log said
    N=60 / S=5 while the body it amends still read **N = 180**, **"Seeds: exactly 3, {0,1,2}"**,
    **"ONE mode produces all 15 runs"**, and an N=180 §5 fallback budget. A reader reaching §3
    first would have taken the superseded design as the specification. All corrected to N=60,
    seeds {0,1,2,3,4}, 25 runs, with the seeds-are-replicates reason stated at the point of use.
  - **C-17 — STALE NUMERICS IN SOURCE, INCLUDING ONE THAT SHIPS IN THE MANIFEST.**
    `verdict.py`'s module docstring described N=180, 15 artifacts, seeds {0,1,2}, and an all-arms
    `var_sr`; `harness.py:213` likewise. **The one that mattered most:** the §5 fallback `rule`
    string persisted into the verdict manifest hardcoded *"same N=180 budget"* — the shipped
    artifact would have carried a false description of its own recipe, the **same defect class as
    the clause-5 string fixed in v1.6.2, in the sibling clause.** It is now built from
    `ECON_FLOOR_IR`, `DSR_THRESHOLD` and `DSR_N_TRIALS`, so it cannot disagree with the recipe.
  - **C-13 — `harness.py` IS INSIDE THE M6 PATH, CONFIRMED AND RE-DESCRIBED.** `xsection.py:36`
    imports `HEADLINE_COST`, `KAPPAS`, `_per_bar_cost` and `forward_log_returns` from it, and
    `xsection` is the money eval path — so four symbols are load-bearing for the headline while the
    file described itself as the M5-only instrument. **The freeze STANDS and nothing was edited to
    change behaviour**; only the scope description, because a wrong scope description is exactly
    how a "frozen, M5-only" file gets edited by someone who does not know the money path depends
    on it. Also recorded, not renamed: `_per_bar_cost` is imported across module boundaries despite
    its underscore — renaming it would touch the anchored instrument for cosmetic reasons.
  - **C-8 — REFUTED AS STATED, with a different and real finding underneath.** The audit reported
    the v1.5 entry as *"asserting both N=60/S=5 and N=180/S=3"*. It does not: it states N=60/S=5 as
    the pins (items A.5 and D), cites 180/3 only as the values being changed FROM and as what the
    mutation KATs must reject, and separately pre-registers an S=3 **budget** fallback that was
    RESOLVED on 2026-07-31 in favour of S=5. That is coherent, not contradictory.
    **What IS wrong:** the S=3 fallback still read as a live option, and it is now **unreachable in
    code** — the C-6 money assertion hard-fails any seed set other than `PINNED_SEEDS`, and the
    verdict requires 25 artifacts, so S=3 can be neither executed nor assembled without moving a
    pin. Marked **CLOSED** at its site; re-opening it would be a specification change needing its
    own dated entry. Costs nothing today because the contingency was already resolved, but a
    pre-registered contingency the code forbids is a contradiction of a different kind than the one
    reported.

- **v1.6.11 (2026-08-02, C-1 RECORDED AS INDETERMINATE + the DECODE-AGREEMENT disclosure wired to
  the real cells + the handicap interpretation draft extended to BOTH channels. No pinned value
  moved, no clause touched; both disclosures are NON-GATING.)** Local, $0.
  - **THE STATISTIC WAS THE SUPERVISOR'S AND IT WAS THE WRONG ONE — verified independently before
    acting on it.** Their instruction, quoted: *"Express asymmetry RELATIVE TO THE CLAIMED EFFECT —
    self-scaling, the same shape as the power guard, which refuses to report a dIR smaller than its
    own inputs' spread."* The power guard's form is a **REFUSAL rule**; it was applied to an
    **INFERENCE question**. Comparing two means needs the SE of their DIFFERENCE, which combines
    both arms and is √n smaller.

    | form | value | reads |
    |---|---|---|
    | as specified: \|diff\| ÷ one arm's sd | 1.10 | "asymmetric" |
    | **correct: \|diff\| ÷ SE(diff)** = 0.1071 ÷ 0.0588 | **t = 1.821**, Welch df 2.385, crit **2.6226**, p = 0.0947 | **not distinguishable** |

    Reproduced with the repo's own `eval/tdist.py`. **One refinement to the ruling's arithmetic:**
    the quoted critical value 2.92 is the df=2 figure; at the actual Welch df 2.385 it is
    **2.6226**. Same conclusion either way.
  - **THE PATTERN, WHICH IS MORE USEFUL THAN THE INSTANCE.** This is the **third** statistical error
    of the same shape in this project — after the MDE floor (mutually exclusive limits of one
    formula combined) and the scaled-over-unscaled ratio: **a formula used outside the regime it
    was derived for.**
  - **EVERY C-1 COMPARISON IS NOW INDETERMINATE AT n=3**, including the cross-width one previously
    called "symmetric" (t = 0.47 vs crit 2.15). **Failing to reject is not evidence of equality** —
    so "symmetric" was never established either. The only durable finding is that the single-seed
    reading was an artifact of `micro_shuffled` seed 0 at the top of its spread.
  - **DO NOT BUY MORE SEEDS ON THE PROXY (ruling).** n ≈ 13 per arm would be needed, on
    briefly-trained tokenizers at reduced seq_len that may not transfer.
  - **MEASURED ON THE REAL CELLS INSTEAD — `single_bar_decode_diagnostic`**, wired
    `score_cell` → `CellScore.decode_agreement` → the artifact (**REQUIRED**, per the C-2 rule) →
    `verdict.decode_agreement_disclosure` in the manifest. Per-arm sign agreement with its
    **variance-ratio degeneracy flag**, seed-mean and per-seed values carried, and a proper
    **two-sample Welch test** on the cell-4 vs cell-5 pair. 5 seeds already paid for.
  - **THE SECOND DIAGNOSTIC WIRED TO BOUND A HANDICAP CHANNEL, and BOTH POINT THE SAME WAY IF
    REAL.** C-12's capacity handicap and this decode-noise handicap each degrade the PLACEBO arm
    more than the treatment arm, so each **inflates ΔIR(4−5)** — and they **compound**. Neither can
    deflate it.
  - **THE INTERPRETATION DRAFT IS RE-DRAFTED OVER BOTH CHANNELS** (`docs/m6_m1_interpretation_draft.md`):
    band membership is a function of the TOTAL measured handicap H = H₁ (capacity) + H₂ (decode),
    with a channel counting only if distinguishable on its own statistic and a degenerate arm
    contributing nothing. Boundaries remain the supervisor's, to be set once both instruments exist
    and BEFORE either produces a number.
  - **THREE DEFECTS OF MINE IN THIS PASS, ALL CAUGHT BY THE TESTS.** (1) A no-threshold KAT sliced
    `verdict.py` between two NAMED functions and silently scanned a neighbour once a new function
    was inserted between them — a fixture whose boundaries move when unrelated code moves is not
    measuring what it claims; now AST-bounded. (2) The decode disclosure lost its degeneracy flag
    when an arm had zero across-seed spread, falling through to "INSUFFICIENT REPLICATES" — the
    check has to survive the very case it exists to catch; degeneracy is now reported first and
    always. (3) A fixture assumed n=3 while `DSR_SEEDS` is 5, so the two statistics **agreed** and
    the test failed — which is itself the point: **the wrong-vs-right disagreement is n-dependent**,
    and the regression test now pins the actual recorded n=3 case directly.
  - Suite **439 green**, ruff clean.

- **v1.6.10 (2026-08-02, TWO STANDING NORMS + A SUPERVISOR FRAMING ERROR CORRECTED BY THE BUILDER
  + the C-4 retrain contingency budgeted. No pinned value moved; no clause touched.)** Local, $0.
  - **A SUPERVISOR FRAMING ERROR, VERIFIED AND RECORDED AS SUCH.** The C-1 leg (ii) ruling asked
    whether an arm-dependent decode gap would fail to cancel in ΔIR(4−5), specifying an
    **ohlcv-vs-micro** comparison. **Cells 4 and 5 are BOTH fsq / 16-dim** (`micro` vs
    `micro_shuffled`), so that comparison cannot touch the primary at all — it lands on clause 3's
    ΔIR(4−2) and the §5 fallback ΔIR(2−1), which do compare across widths. The builder substituted
    **micro vs micro_shuffled at matched width**, which is the leg (ii) that was meant. Recorded
    because the failure mode is instructive: **had the specified comparison been run and reported
    "symmetric", we would have been reassured about a contrast that was never at risk.**
  - **NEW STANDING NORM — A PERFECT AGREEMENT SCORE IS AS SUSPICIOUS AS A FAILING ONE.** Any
    agreement, correlation or match statistic must carry a **degeneracy check on its own inputs
    before its value is read**; a collapsed input scores perfectly for free. Instance:
    `micro_shuffled/pre_v1_4` scored **1.0000 sign agreement with `variance_ratio = 0.000`** and
    |Δ| at **1.246× the in-window sd** — a decode collapsed to a constant. Without the flag the
    primary would have read `|diff| = 0.1338 → ASYMMETRIC` on an artifact. This is the project's
    own pathology (frac_neg lock, activity pinned at 0/1) reappearing **inside a metric** rather
    than inside a result. Landed in `docs/ENGINEERING.md`.
  - **NEW STANDING NORM — DO NOT INVENT A BAND; SCALE AGAINST THE CLAIM OR THE STATISTIC'S OWN
    SPREAD.** My first C-1 symmetry reading used a hand-chosen **0.05** band and the value landed
    at **0.0469** — the conclusion rested on a convention I had invented. Replaced with the
    self-scaling form: the between-arm gap is compared to the **across-seed spread of the same
    statistic**, the shape `verdict.power_guard` already uses when it refuses to report a ΔIR
    smaller than its own inputs' wobble. A difference inside its own measurement noise is not a
    measured difference. Landed in `docs/ENGINEERING.md`.
  - **★ THE SELF-SCALING RULING REVERSED A CONCLUSION I HAD REPORTED — AND THEN THE STATISTIC IT
    SPECIFIED TURNED OUT TO BE THE WRONG ONE. C-1 LEG (ii) IS RECORDED AS INDETERMINATE. ★**
    Re-measured at 3 seeds (receipt `runs_manifest/m6_c1_pinset_decode.json`):

    | arm / config | mean | across-seed sd | per-seed |
    |---|---|---|---|
    | micro / m6_pin_set | 0.9512 | 0.0303 | 0.9199, 0.9805, 0.9531 |
    | micro_shuffled / m6_pin_set | 0.8441 | 0.0973 | 0.9541, 0.7695, 0.8086 |

    | form | value | reads |
    |---|---|---|
    | as specified: \|diff\| ÷ one arm's sd | 1.10 | "asymmetric" |
    | **correct: \|diff\| ÷ SE(diff)** = 0.1071 ÷ 0.0588 | **t = 1.821**, Welch df 2.385, crit **2.6226**, p = 0.0947 | **not distinguishable** |

  - **WHAT SURVIVES, AND IT IS THE REAL RESULT.** The single-seed "SYMMETRIC" at |diff| 0.0342 was
    an artifact of `micro_shuffled` seed 0 landing at 0.9541, the top of a spread reaching 0.7695.
    **That withdrawal stands.** What replaces it is **INDETERMINATE AT n=3**, not a reversal:
    *both* readings were unsupported, and the second was produced by a statistic the supervisor
    specified incorrectly.
  - **THE SUPERVISOR'S ERROR, RECORDED AS A PATTERN RATHER THAN AN INSTANCE — their instruction,
    quoted:** *"Express asymmetry RELATIVE TO THE CLAIMED EFFECT — self-scaling, the same shape as
    the power guard, which refuses to report a dIR smaller than its own inputs' spread."* The power
    guard's form is a **REFUSAL rule** (never report a claim smaller than its own inputs' spread);
    it was applied to an **INFERENCE question** (are these two arms different?). Comparing two
    means needs the SE of their *difference*, which combines both arms and is √n smaller.
    **This is the third statistical error of the same shape in this project** — after the MDE floor
    (mutually exclusive limits of one formula combined) and the scaled-over-unscaled ratio: **a
    formula used outside the regime it was derived for.** The builder executed the instruction
    faithfully and it produced a wrong conclusion; the builder's own caveat at the time —
    *"this establishes that symmetric was unsupported, not that asymmetric is established"* — was
    more right than the headline placed above it.
  - **DO NOT BUY MORE SEEDS ON THIS PROXY (supervisor ruling).** Resolving it needs **n ≈ 13 per
    arm**, not 3 more, and the scope limit is the reason not to spend even that: briefly-trained
    tokenizers at reduced seq_len, where the contrast between rows is the result and absolute
    levels are not a claim about a trained M6 cell. **Powering a proxy that may not transfer is
    poor value.** It is measured on the REAL cells instead — see §7 v1.6.11.
  - **TWO DEFECTS OF MINE IN THAT RUN.** A leftover `args.seed` after the multi-seed change (the
    "flag accepted but never forwarded" family, caught by the crash rather than by me); and worse,
    a **MIXED ROW** — `sign_agree` printed as the 3-seed mean beside `var_ratio` from seed 0 only.
    A row whose columns come from different populations is unreadable and would have misled the
    same way the degenerate 1.0000 did. Every reported column is now the seed-mean with its
    per-seed values carried.
  - **THE M1 INTERPRETATION RULE IS DRAFTED, NOT ADOPTED** —
    `docs/m6_m1_interpretation_draft.md`, put to the supervisor for ruling. Written **now, before
    any real cell runs**, because "how do we read M1's number" cannot be decided after the number
    exists without choosing an interpretation from the result. Three bands (A: handicap
    indistinguishable from zero → headline stands; B: measurable but small against the claimed
    effect → headline stands, magnitude **in the abstract**, claim bounded as "inflated by at most
    this"; C: comparable to or larger than the claimed effect → **the microstructure information
    result is not claimed**, and the C-12 mechanism becomes the substantive contribution), each
    expressed **relative to the claimed ΔIR(4−5)**, never against an absolute. It is explicitly a
    **CLAIM rule, not a verdict rule**: M1 is non-gating, the verdict still emits
    SURVIVES/NULL/INCONCLUSIVE from the five clauses alone, and the freeze is untouched. **Two
    limits are stated inside the draft rather than papered over:** H and Δ are in different units
    (reconstruction MAE vs annualized IR), so "comparable to" cannot be a direct numeric
    comparison and none was written; and M1 measures the handicap's size in reconstruction space,
    not its transmission to IR — treating H as an upper bound on inflation is the conservative
    direction and the draft says so. Band boundaries are deliberately left to the ruling, because
    proposing them would re-introduce the invented constant the ruling forbade.
  - **C-4 RETRAIN CONTINGENCY, COSTED AND PRE-REGISTERED WITH ITS TRIGGER**
    (`docs/m6_c4_kronos_gate_requirements.md` §4a). Trigger: Cell 1 scores **RankIC < 0.85 ×
    published Kronos-small** on the pinned slice → halt before any Δ → fix Cell 1 only → **full
    5-cell same-seed retrain**. Cost: **+1× the training leg**, taking the worst case from the
    recorded **$33–50** (S=5) / **$43–65** (forced determinism) to **~$66–100 / ~$86–130**. Every
    figure inherits the §7 v1.6 ESTIMATED-NOT-MEASURED labelling; the ×2 is arithmetic on the
    protocol, not a measurement. Named because a doubling discovered when it fires is a crisis and
    a doubling written down in advance is a budget line — and it had appeared in **no** estimate
    the builder produced to date.

- **v1.6.9 (2026-08-01, C-12 M1 ADOPTED — the placebo CAPACITY disclosure, REQUIRED in the
  results. A non-gating disclosure: no clause, no threshold, cannot flip SURVIVES↔NULL.)** Local,
  $0.
  - **WHAT IT BOUNDS.** The Cell-5 permutation destroys the contemporaneous micro↔OHLCV dependence
    as well as the intended temporal alignment (**measured 0.2037 → 0.0005**), so Cell 5's micro
    channels are six dims of **independent noise** rather than "Cell 4 minus information". Noise
    with a preserved marginal is incompressible and shares no structure with OHLCV, so bits spent
    on it are bits taken from OHLCV — and `micro_point_weight = 3.0` aims triple gradient pressure
    at fitting it. ΔIR(4−5) therefore carries (information) + (capacity handicap), inseparable
    from the contrast alone. **★ §7 v1.6.27 — THE CLAUSE THAT ENDED THIS SENTENCE WAS FALSE.** It
    read *"It **inflates rather than manufactures**: Cell 4 must still clear the MDE and the 0.5
    economic floor on its own."* **NO CLAUSE TESTS CELL 4 ON ITS OWN.** Clause 4 is
    `pb45.delta_ir >= ECON_FLOOR_IR`, clause 2 is `pb45.delta_ir >= mde_paired` — **both test the
    DIFFERENCE**, so the handicap sits inside the quantity compared against the floor and a large
    enough handicap can carry ΔIR over 0.5 with no information at all.
  - **WHY THE COMPARISON IS LIKE-FOR-LIKE.** `block_time_permute` never touches the OHLCV columns
    — they come back **byte-identical** — so both arms reconstruct the SAME OHLCV targets. Cell 5
    doing it worse is capacity diverted, and the ratio is a measured magnitude where there was
    previously an unbounded confound.
  - **WHERE IT LIVES.** `diagnostics.ohlcv_recon_diagnostic` → `score_cell` (`CellScore.ohlcv_recon`)
    → the per-(cell, seed) artifact, **REQUIRED** like `mu_diag` for the C-2 reason (a disclosure
    that may be absent is a disclosure that will be absent) → `verdict.placebo_capacity_disclosure`
    → emitted in the manifest **beside `dual_specification`**, i.e. in the results, not an
    appendix.
  - **TWO PROPERTIES PINNED, AND THEY PULL OPPOSITE WAYS — which is the point.** Required (writer
    refuses, loader refuses, both mutation-tested) **and NON-GATING**: a dedicated KAT assembles
    two verdicts differing ONLY in the OHLCV-recon numbers — benign vs a 10× placebo handicap —
    and asserts the emitted word, the primary, the failing-clause list and every clause's `pass`
    are IDENTICAL, while separately asserting the two disclosures DID differ (fixture
    discrimination — otherwise the test proves nothing). Plus a source check that the function
    contains no comparison to any constant: **a disclosure with a bar is a clause in disguise**.
  - **COMPUTED, NOT RESTATED.** Mutation KAT: moving Cell 5's OHLCV recon from 0.10 to 0.30 moves
    the reported ratio 1.0 → 3.0 and the excess 0.0 → 0.20.
  - **A BUILDER PROCESS SLIP, DISCLOSED.** My first pass inserted the fixture constant with a
    line-heuristic that matched a `@pytest.mark.parametrize(` closing paren, landing a module
    constant **inside a function body** in two test files. Caught by ruff, then fixed properly by
    locating the end of the import block with the **AST** and re-parsing each file to prove it
    before moving on — the same "don't guess structure from text" lesson as the C-1 source-scrape.

- **v1.6.8 (2026-08-01, AUDIT TIER-2 C-1 LEGS (i)+(ii) — the single-bar decode under the FULL M6
  PIN SET, per arm. REPORT ONLY.)** Local, $0. Receipt:
  `runs_manifest/m6_c1_pinset_decode.json`.
  - **WHY LEG 1 DID NOT SETTLE IT.** Neither trained checkpoint carries `fine_pointwise=True`.
    Under §7 v1.4 the fine subtoken becomes a PER-BAR encoding of bar *t*'s own features
    (`model.py:83-86`), which is exactly the mechanism that could make a single-bar decode
    meaningful — and it is pinned for every M6 cell. The 0.507 measured the **pre-v1.4**
    architecture.
  - **THE COMPARISON IS MATCHED, WHICH LEG 1'S WAS NOT.** Data, seed, steps, optimizer, dims and
    quantizer identical; only the pin set varies. d256/3L, real BTCUSDT, 400 steps, seq_len 128.

    | arm | pre-v1.4 | **M6 pin set** |
    |---|---|---|
    | ohlcv (7d) | 0.9629 | 0.9668 |
    | micro (16d) | 0.8662 | **0.9199** |
    | micro_shuffled (16d) | 1.0000 ⚠ degenerate | **0.9541** |

    Reference points: auditor's d64/2L toy **0.838**; leg-1 checkpoint **0.507**.
  - **LEG (i) — THE PIN SET HELPS, AND THE HONEST READING IS NARROWER THAN THE HEADLINE.** Under
    the full pin set sign agreement is **0.92–0.97**, decisively toward 1.0, so C-1 is
    substantially mitigated by a fix already in the design. **But the matched contrast isolates
    the pin at +0.054 (0.8662 → 0.9199), not +0.41.** The leg-1 **0.507 was largely a property of
    THAT CHECKPOINT's training, not of `fine_pointwise=False` as such** — an identically-trained
    pre-v1.4 tokenizer here scores 0.8662. Crediting the pin with closing the 0.507 gap would be
    over-claiming, and this entry does not.
  - **LEG (ii) — SEVERITY, AND A CORRECTION TO THE FRAMING OF THE QUESTION.** The ruling asked
    whether an arm-dependent gap would fail to cancel in ΔIR(4−5). **Cells 4 and 5 are BOTH
    16-dim micro-shaped** (`micro` vs `micro_shuffled`), so an ohlcv-vs-micro asymmetry cannot
    reach the primary at all — it lands on clause 3's ΔIR(4−2) and the §5 fallback ΔIR(2−1),
    which do compare across widths. The comparison that decides the PRIMARY is real micro vs
    shuffled micro at the same width, and it was added for that reason.

    | | PRIMARY (4 vs 5) | cross-width (4 vs 2) |
    |---|---|---|
    | pre-v1.4 | **INCONCLUSIVE** (degenerate arm) | 0.0967 — width-dependent |
    | **M6 pin set** | **0.0342 — SYMMETRIC** | 0.0469 — symmetric |

    ~~**Under the pinned configuration the gap is symmetric on the primary pair**, so on this
    evidence it costs POWER and cannot bias ΔIR(4−5). The cross-width figure 0.0469 sits just
    under the 0.05 band and **that band is arbitrary**.~~
  - **★ WITHDRAWN 2026-08-02 — THE SINGLE-SEED READING WAS AN ARTIFACT. ★** The reassuring 0.0342
    came from `micro_shuffled` seed 0 landing at 0.9541, the high end of a spread running down to
    0.7695. **That withdrawal stands.**
  - **★ AND THE REPLACEMENT WAS ALSO WRONG — "ASYMMETRIC" IS WITHDRAWN TOO. ★** I briefly recorded
    the primary pair as ASYMMETRIC on |diff| 0.1071 ÷ one arm's sd 0.0973 = 1.10. **That statistic
    is not a two-sample test.** The correct comparison uses the SE of the *difference*:
    `SE = √(sa²/n + sb²/n) = 0.0588`, giving **t = 1.821** on **Welch df 2.385** against a
    critical **2.6226** (self-written `eval/tdist.py`; one-sided p = **0.0947**). **NOT
    SIGNIFICANT.** The correct record is **INDETERMINATE AT n=3** — *both* "symmetric" and
    "asymmetric" are unsupported. See §7 v1.6.11.
  - **A DEGENERACY GUARD ADDED MID-EXPERIMENT, WHICH CHANGED A CONCLUSION.**
    `micro_shuffled/pre_v1_4` scored a **perfect 1.0000 sign agreement with `variance_ratio =
    0.000`** — the single-bar decode collapsed to a CONSTANT, whose sign trivially agrees
    wherever the window sign is also constant, while |Δ| was **1.246× the in-window sd**. Without
    the flag the primary comparison would have read `|diff| = 0.1338 → ASYMMETRIC` **on the basis
    of an artifact**. The probe now refuses to conclude from a degenerate arm (control-arm rule).
    This is the project's own pathology — the frac_neg lock, activity pinned at 0/1 — reappearing
    inside a *metric*, and it is the reason sign agreement must never be read without the
    variance ratio beside it.
  - **THE `point_decoder` REGIME, MEASURED BECAUSE IT EXISTS AND NOT PROPOSED.** With
    `fine_pointwise` on, `model.py:95-102` builds a genuinely per-bar zero-cross-bar head whose
    own comment says *"Train-time only; the AR interface and the main decode path are
    unchanged."* Its sign agreement is 0.9463 / 0.9453 / 0.8477 — **comparable to, and for the
    placebo arm WORSE than, the window decoder**. So "the fix exists but is not wired in" is
    NOT supported by measurement, and that expectation of mine is withdrawn.
  - **STANDING FACT, unchanged by any of this.** `model.py:104-105` constructs
    `TokenizerDecoder(..., False, ...)` — the decoder is hardcoded **bidirectional and
    window-trained** in every row above, so a length-1 decode remains out of distribution.
    `fine_pointwise` changes what the ENCODER puts in the fine dims, not what the DECODER does.
  - **SCOPE LIMIT.** These tokenizers are trained briefly at reduced seq_len for a $0 local
    comparison. **The CONTRAST between rows is the result; the absolute level is not a claim
    about a fully-trained M6 cell.** Sensitivity is real: at 20 steps micro read 0.5352/0.7207,
    at 400 steps 0.8662/0.9199 — the ordering held, the levels moved a lot.

- **v1.6.7 (2026-08-01, AUDIT TIER-2 C-1 — THE SINGLE-BAR-DECODE GAP AT CANONICAL DIMS. REPORT
  ONLY: nothing proposed, nothing changed, no design touched.)** Local, $0.
  - **THE STRUCTURE (not in doubt, S-1).** The tokenizer decoder is a Transformer over the
    SEQUENCE dimension and its reconstruction objective is optimised on FULL WINDOWS. The M6
    decision path does not do that: `predict._rollout` calls `decode_latent(z)` with `z` of shape
    `[n, 1, dim]` — **a length-1 sequence, no context** — and takes `[:, 0, 0]`, i.e. feature 0,
    which is exactly the quantity μ̂ accumulates. The decoder is evaluated in a regime its
    objective never trained.
  - **MEASURED: canonical d256 / 3 layers, trained checkpoints, real lake, 3 symbols × 16 windows
    × seq_len 512.** The auditor's four metrics, held identical for comparability:

    | | auditor d64/2L toy | **cell4 ckpt** (`encoder_causal=True`) | **M3 ckpt** (`encoder_causal=False`) |
    |---|---|---|---|
    | variance ratio (dim 0) | 1.44 | **0.509** [0.445, 0.571] | **1.205** [1.145, 1.291] |
    | sign agreement (dim 0) | 0.838 | **0.507** [0.498, 0.517] | **0.780** [0.766, 0.794] |
    | recon MAE ratio | — | **2.27×** | **4.00×** |
    | recon MAE dim 0 | — | 0.1269 → **0.5258** | 0.1085 → **0.4095** |
    | mean \|Δ\| / sd(in-window) | — | 0.465 | 0.888 |

  - **THE HEADLINE.** On the checkpoint carrying the **causal-encoder pin** — the v1.3 M6
    requirement — **sign agreement is 0.507, indistinguishable from a coin flip**, consistent
    across all three symbols (0.4982–0.5167). The toy's 16.2% flip rate **understated it**: the
    real figure on the pinned-encoder instrument is ~49%. This is not a near-zero artifact:
    mean|Δ| is 0.47× the in-window standard deviation and dim-0 reconstruction MAE is 4.1× worse.
  - **THE TWO CHECKPOINTS DISAGREE ON THE VARIANCE RATIO'S DIRECTION** (0.51 deflation vs 1.21
    inflation) while agreeing that MAE degrades severely. The causal checkpoint has the SMALLER
    mean|Δ| but the WORSE sign agreement — a smaller, more centred single-bar output whose sign
    is then essentially random. Reported, not explained.
  - **WHAT THIS DOES AND DOES NOT IMPLY, stated because the distinction is load-bearing.** All
    five cells share the decode regime, so by the same argument that makes the unpinned
    hyperparameters harmless (§ C-5 A8) this cannot by itself manufacture or destroy ΔIR(4−5) —
    it degrades every arm. What it plausibly does is destroy μ̂ QUALITY and therefore the design's
    POWER. Whether it is symmetric across arms is **open and not investigated here**: the cells
    differ in feature set, so a decode gap that differs by arm would not cancel. Naming it;
    proposing nothing.
  - **DISCLOSURE THAT BOUNDS THE CLAIM.** No trained tokenizer in the repo carries the FULL M6 pin
    set — both canonical-dims checkpoints predate `fine_pointwise=True` and
    `micro_point_weight=3.0`. Both were measured and both are reported rather than one being
    chosen. Receipt: `runs_manifest/m6_c1_single_bar_decode.json`.

- **v1.6.6 (2026-08-01, C-5 A9 CLOSED + RESOURCE PRECONDITIONS. Defect fixes; no pinned value
  moved.)** Local, $0.
  - **A9 AND A4 WERE CONFLATED — the reusable lesson, now written into the driver's docstring.**
    A9 is LOCAL, $0, COMPLETE PATH COVERAGE (does every stage execute and hand off — a WIRING
    proof, in minutes). A4 is the GPU validation at reduced scale (does it produce the right
    numbers — a NUMERICAL proof, and it costs money). The first `--dry-run` zeroed training steps
    but ran the REAL money eval, 1,402,560 decisions per unit — A4's job on A9's budget.
  - **THE FIX USES THE MECHANISM C-6 ALREADY BUILT.** A dry run DECLARES `money_run=False`, which
    is what lifts `xsection.py:71`'s cap prohibition — legitimate precisely because `money_run` is
    a declaration and never an inference. Option **(a)**, shortened work with REAL scoring, not
    (b): a synthetic score would skip the path A9 exists to prove. **Conformance still asserts the
    FULL pinned 40-symbol surface first** — only the subsequent LOAD is bounded, and a KAT asserts
    on the AST that the gated object is the untouched `cfg`.
  - **STRUCTURALLY UN-QUOTABLE, not merely labelled:** every dry-run artifact carries
    `money_run: false` / `grid_pinned: false` / `dry_run: true`, and `_validate_artifact` REFUSES
    any of the three, so `load_cell_evals` cannot assemble one — including the dangerous case of
    24 real units and one dry-run unit.
  - **A REAL MONEY-CONFIGURATION DEFECT THE DRY RUN FOUND.** The backbone was built at the default
    `max_len = seq_len`, but the KV-cached rollout addresses `seq_len - 1 + k` for k up to h. The
    money run would have **TRAINED EVERY UNIT AND THEN RAISED AT THE FIRST EVAL**. Fixed as a
    money assertion plus a pin-derived default (`seq_len + max(horizons)`); `max_len` sizes RoPE
    buffers only and the realized count is 21,301,248 either way (verified), so it moves no number.
  - **A COSMETIC KNOB OF MINE, REMOVED.** My first bounding also sliced the recorded grid. That
    bounded nothing — `score_cell` builds its own grids from `cfg`, so the real bound is
    `cap_per_symbol` — and it would have made the artifact's `n_periods` disagree with its own
    `headline_series`, which `_validate_artifact` requires to match. `n_periods` is now DERIVED
    from the scored series (true by construction) and is therefore no longer a resume-binding
    field: it is a deterministic function of fields that ARE bound. Recorded so the omission reads
    as a decision.
  - **RESOURCE PRECONDITIONS, before any compute (`utils/preflight.py`).** The argument is the
    measured one: 27.8 GiB on a 16.0 GiB host did NOT fail — it swapped, and one zero-step unit
    took 2,581 s. **An OOM stops the meter; swap-thrash keeps billing while looking like ordinary
    slowness.** Memory strategy is a HOST property (`--mem-strategy`, `auto` by default,
    recorded in the manifest); both strategies are byte-identical and differ only in residency and
    rebuild count, so the 3×/5× recompute is NOT paid on the 251 GB box that does not need it.
    Disk is checked too; VRAM is recorded.

- **v1.6.5 (2026-08-01, AUDIT TIER-1 C-5 — THE MONEY DRIVER. New code, no pinned value moved.)**
  Local, $0. Built to the supervisor's A1–A9 scope; the A9 dry run is what proved it.
  - **WHY IT EXISTS.** Pre-flight Item 2 promises ZERO first-time code paths on rented hardware,
    and the money configuration had **no driver at all**. `m6_toy_rehearsal.py` differs from it on
    symbols, window, seeds, step budget, grid, conformance and the legibility gate — so validating
    the rehearsal validates the wrong code. The outstanding CUDA validation is re-scoped to run
    THROUGH `scripts/m6_money_run.py` at reduced scale.
  - **A1 — ONE PATH, NOT TWO.** The shared orchestration moved into `trikaal/run/matrix.py`
    (train loop, reload-proof, resume, per-unit scoring, artifact emission, lake I/O, window
    build) and **both** drivers call it. `m6_toy_rehearsal.py` was refactored onto it rather than
    left as a second copy — duplication is exactly how C-6 happened. The rehearsal thereby
    inherits the C-14 resume binding and the A7 provenance stamp for free.
  - **A2 — NO FLAG MAY MOVE A PIN.** The driver's argparse is `--out --shard --device --no-resume
    --dry-run --lake --wandb`. There is no `--seeds`, `--seq-len`, `--steps`, `--symbols`,
    `--cost`. Enforced by a KAT that reads the parser and rejects any flag name containing a
    pin-shaped token; the filter is itself discrimination-tested.
  - **A3 — CONFORMANCE FIRST, asserted on the AST**, not merely written first: a KAT parses
    `main()` and requires `assert_conformance` to precede every lake/train/eval call. Verified
    live in the dry run — `conformance PASS over 40 symbols, seeds (0,1,2,3,4)`, symbols sha256
    `60e24f598de96012…`, before any compute.
  - **A5 — FAN-OUT.** `--shard i/N` strided over the 25 units, with `shard_partition_failures`
    proving the shards are an exact disjoint cover (tested at N ∈ {1,2,3,5,7,25}) and balanced to
    ±1. **The verdict now REFUSES to assemble across mixed hardware**: `provenance_failures`
    compares eleven identity fields across artifacts and `load_cell_evals` raises on divergence
    or on any unattributable unit. The index is withheld unless the matrix is complete — a partial
    index looks authoritative while describing a fraction of the matrix.
  - **A6 / C-14 — RESUME BINDS TO THE CHECKPOINTS.** Schema equality alone let a rerun *after
    retraining* silently mix generations: the stale artifact parses, carries the right schema, and
    hashes consistently with itself, so `load_cell_evals` content-verifies the mixture happily. An
    artifact is now honoured only if `meta.checkpoints` matches the checkpoints on disk **and**
    symbols, window, h, n_periods and start_ms all match. Eight binding fields, one KAT each.
  - **A7 / C-18 — PROVENANCE PER UNIT**, auto-filled at the single emission path so an
    unattributable artifact cannot be written rather than merely being discouraged: GPU name,
    driver, CUDA build, torch, numpy, **Python**, platform, attention mode and every determinism
    flag. The interpreter is called out because `uv.lock` pins packages, not python — `>=3.11`
    admits 3.14 and a rented box silently ran 3.14.6.
  - **A8 — NOTHING INVENTED.** `unpinned_parameters()` records the values READ from code
    (`steps_stage1/2`, `peak_lr_*`, `warmup_frac`, `batch_size`, `alpha`) **with the reasoning
    beside them**: ablation validity does not depend on them because all five cells share them, so
    a too-small budget weakens every arm identically and cannot manufacture or destroy ΔIR(4−5);
    only reproducibility depends on them. Stage-2's budget is recorded as derivable from
    m6_design.md:59 (16,479–18,311 steps/stage at canonical geometry) and Stage-1's as genuinely
    free.
  - **A9 — THE DRY RUN FOUND A BLOCKER, WHICH IS WHAT IT IS FOR.** The first invocation died
    **SIGKILL (exit 137) with zero output**. Two defects, both fixed:
    - **MEMORY, MEASURED NOT GUESSED.** The money configuration is 40 symbols × **84,153,600
      bars**. The raw arrays alone are **8.78 GiB**; building all 5 seeds × 3 arms of
      arm-transformed windows up front adds **95.22 GiB** for a **~104 GiB peak**. Window
      construction is now LAZY in the shared path — one seed resident, built inside the loop and
      released — which changes *when* windows are built and never what they contain, taking the
      peak to **~27.8 GiB**. The rehearsal never hit this because it slices to 200k bars × 5
      symbols (≈1M bars) against the money configuration's 84.15M.
    - **OUTPUT LOST ON KILL.** Every `print` was block-buffered to the redirect, so a SIGKILL
      discarded the entire progress log — a run that dies must still show how far it got. The
      driver's prints now flush.
    - **CORRECTION TO MY OWN PREDICTION, ON THE RECORD.** I reported the lazy run would also die
      of memory. **It did not.** It cleared training and the checkpoint reload. What actually
      happens is worse than a clean failure: macOS absorbed the overflow in **swap**, and the
      arithmetic is confirmed almost exactly by the outcome — **16.0 GiB RAM + ~11.96 GiB swap in
      use ≈ 28 GiB, against the predicted ~27.8 GiB peak**. The visible cost is that ONE unit with
      **ZERO training steps** took **2,581 s (43 min)** of wall-clock, essentially all of it
      swap-thrashing through the lake load and window build. On rented hardware this failure mode
      is the dangerous one: the run does not stop, it silently burns money at a fraction of the
      expected rate. The memory FINDING stands and is now measured rather than predicted; my
      "it will be killed" prediction is withdrawn.
    - **THE RESIDUAL IS A SUPERVISOR DECISION, NOT A BUILDER ONE.** ~27.8 GiB against 16 GiB of
      RAM means A9's "prove the whole thing locally at $0" is satisfiable here only by swapping,
      which makes the proof take hours per unit rather than failing honestly. A `--dry-run-symbols` flag
      would resolve it and is deliberately **NOT** added: symbols are pinned and A2 forbids a flag
      that can move a pinned value. Options are the operator's — authorise an explicitly
      operational reduced-symbol dry run, run the dry run on the rented box (which forfeits the
      "$0 before it sees a GPU" property), or reduce the driver's footprint further.
    - **A FURTHER REDUCTION IS AVAILABLE AND IS NOT TAKEN UNILATERALLY, BECAUSE IT IS A
      TRADE-OFF.** `run_cell` uses exactly ONE arm (`orchestrator.py:238`,
      `per_symbol = per_symbol_by_arm[spec.arm]`), while `windows_by_arm` builds all three.
      Building only the arm a unit needs takes the peak from **~27.8 GiB to ~16.3 GiB**
      (8.78 raw + 7.52 for the `micro` arm) — but it buys that with RECOMPUTE, and how much
      depends on unit ordering: the current cell-major order would rebuild 25 times instead of 5
      (5×), a seed-major order with a within-seed arm cache would rebuild 15 times (3×) at the
      same 7.52 GiB peak. Rebuilds are byte-identical (`shuffle_micro` is seeded on
      `(symbol, seed)`), so nothing numerical moves either way — this is purely memory vs
      wall-clock on rented hardware, i.e. a **cost decision, not a builder decision**. Reported
      with the numbers; not implemented.
    - **A HYPOTHESIS FORMED AND REFUTED RATHER THAN REPORTED.** The 40-symbol load ran at roughly
      4.3 min/symbol, and I suspected the per-symbol query was scanning all 7,024 parquet files
      instead of pruning. `EXPLAIN` shows `File Filters: (symbol = '…')` for both parameterised
      and literal predicates — **pruning works, the hypothesis is wrong.** The load cost is real
      but unexplained, was measured under contention from a concurrent test run, and must be
      re-measured cleanly before it is quoted as a cost input.

- **v1.6.4 (2026-08-01, AUDIT TIER-1 C-6 — THE TRAINING SIDE AND THE EVAL SIDE EACH HELD THEIR OWN
  COPY OF THE TRUTH. DEFECT FIX: no pinned VALUE moved; the eval side was already authoritative and
  is now the only statement. Gate-A anchor `3f86882a` re-proven byte-identical ×2, exit 0.)**
  Local, $0.
  - **THE DEFECT, AND THE PART THAT IS LARGER THAN THE FINDING AS STATED.**
    `OrchestratorConfig` carried `seeds = (0, 1, 2)` and `seq_len = 128` against a pinned
    `PINNED_SEEDS = (0, 1, 2, 3, 4)` and a money eval surface of **512**. Two consequences:
    (1) the S=5 the operator approved and funded on 2026-07-31 never reached the training side;
    (2) a **TRAIN/EVAL REGIME MISMATCH** — a model optimised over 128 bars of context and then
    scored on 512. That is not a stale default, it is a different experiment, and it is the same
    class as C-1's single-bar decode through a window-trained decoder; the two compound.
  - **PROVENANCE — WAS ANY PRIOR RESULT PRODUCED UNDER A 128/512 SPLIT? NO.** Exhaustive sweep of
    every manifest and eval artifact on disk: `seq_len = 32` in 103 files (toy/dryrun shells),
    `512` in 45 (every real cell run, including the CUDA validation), `1` in 2 (a decode probe),
    and `128` in **exactly one** — `runs_manifest/gate_a_run_manifest.json`, the **M5 Gate-A
    anchor**, which is a different and earlier instrument running 128 *consistently* for both
    rollout and scoring. **No M6 result was produced under a mismatched split.** It stayed
    invisible exactly as predicted: all four `OrchestratorConfig` call sites override both fields,
    and the money driver that would have inherited the dataclass defaults (C-5) does not exist.
  - **IMPLEMENTED AS AN ASSERTION, PLUS THE REMOVAL OF THE SECOND COPY.** The fields now
    **reference** `PINNED_SEEDS` / `PINNED_MONEY_SEQ_LEN` rather than restating them — a
    right-looking literal fixes today and guarantees nothing about tomorrow, and the defect
    existed *because* two files each restated one truth. On top of that, `__post_init__`
    hard-fails (`ConformanceError`) on divergence, which additionally catches a caller passing
    bad values **by argument** — something no default can do. `money_config`'s `seq_len` default
    now reads the same constant, so eval and training are one statement.
  - **`money_run` IS A DECLARATION, NEVER AN INFERENCE, and defaults to TRUE.** Deciding toy-ness
    from the values would re-create the hole precisely: a 128-context money run would be
    classified as a toy and skip the check. Defaulting to True makes the **dangerous** direction
    the one that must be declared — a toy shell announcing itself is safe; a money run silently
    inheriting toy values is the defect. The four toy/smoke/probe call sites now declare
    `money_run=False`.
  - **THE SHADOWED-FIELD SWEEP, as ordered.** Two more fields held standing commitments with **no
    pin at all**, both now pinned and asserted: `micro_legibility_min = 0.9` (the §7 v1.4 standing
    six-micro-dims gate — a **HARD STOP** whose threshold nothing guarded) and
    `expect_backbone_params = 21_301_248` (the realized count, a `docs/ENGINEERING.md` invariant).
    `enforce_parity` must be True on a money run. **REPORTED, NOT PINNED:** `steps_stage1`/
    `steps_stage2` (2000), `peak_lr_stage1/2`, `warmup_frac`, `batch_size` and `alpha` shadow
    **nothing** — no prereg constant states them. §7 v1.6 records the step budget as *read from
    code*, not as a pin. Whether any of these should become pins is a **ruling**, not a builder
    decision, and none was invented here.
  - **A STALE DOC IN THE GATE'S OWN DESCRIPTION**, found while editing it and corrected:
    `conformance.py`'s module docstring advertised "seeds exactly {0, 1, 2}" — the gate describing
    itself by the superseded value it exists to enforce against. (C-9 remains open for the §3/§3a
    body text.)
  - **KATs** (`tests/train/test_money_surface_assertion.py`, 14): a clean-surface pre-check so no
    mutation can pass vacuously; seven divergence mutations including **both exact historical
    values** `(0,1,2)` and `128` plus a near-miss `(0,1,2,3)`; a declared-toy exemption proving
    fail-closed did not become fail-on-everything; a toy-ness-is-not-inferred case; a
    one-constant check tying `money_config().seq_len` to `OrchestratorConfig().seq_len`; an AST
    check that no dataclass default **re-types** a pinned literal; and an AST sweep of every
    caller — itself discrimination-tested. Suite 379 green, ruff clean.

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
    584. `git show 7a4d13b:src/trikaal/eval/verdict.py | sed -n '520p'` returns it verbatim. **The
    auditor's citation was accurate; my correction is WITHDRAWN** (struck in the v1.6.2 entry
    below). The supervisor caught it before it propagated. Landed in `docs/ENGINEERING.md` beside the other
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
      string."* **That is false and is withdrawn.** `git show 7a4d13b:src/trikaal/eval/verdict.py |
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
    `docs/ENGINEERING.md`, the design spec ×2, `src/trikaal/__init__.py`); repo-wide re-grep now returns only
    the prohibition itself. **Trikaal is a TOKENIZER STUDY; the backbone is the measurement
    vehicle.** Param count corrected `~27M` → **21,301,248** at `docs/ENGINEERING.md:7`, `docs/ENGINEERING.md:33`, the
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
      of the pre-registered primary; the S=3 fallback is NOT exercised.**
      **★ AND IT IS NOW UNREACHABLE IN CODE (§7 v1.6.12, C-8).** The C-6 money assertion
      hard-fails any seed set other than `PINNED_SEEDS`, and the verdict requires 25 artifacts,
      so an S=3 run cannot be executed or assembled without moving a pin. Since the contingency was
      RESOLVED in favour of S=5 this costs nothing — but the fallback must be read as **CLOSED**,
      not as a live option, and re-opening it would be a specification change requiring its own
      dated entry.**

      **★ NAMED OPERATIONAL CONSTRAINT OC-1 (§7 v1.6.13) — THE CHEAP EXIT FROM A BUDGET OVERRUN IS
      SHUT, DELIBERATELY.** "Resolved" and "unavailable" are different statements and only the
      second binds what happens next, so it is recorded as a constraint rather than as a closed
      item:
      - **If the budget runs short mid-run the options are exactly two: (i) fund it, or
        (ii) declare INCONCLUSIVE per R3. Dropping to three seeds is NOT among them** — it would
        require moving `PINNED_SEEDS` after seeing part of the data, i.e. a post-hoc specification
        change under freeze, which is the precise thing the pre-registration exists to forbid.
      - **The compounding is the part that can actually bite.** C-4's retrain contingency
        (`docs/m6_c4_kronos_gate_requirements.md`) takes worst-case spend to **~$66–100**, or
        **~$86–130 under forced determinism** — and that is exactly the scenario in which someone
        reaches for a cheaper seed budget. The hatch is shut in that scenario too. Both facts are
        recorded together here on purpose; separated, each reads as survivable.
      - **This is NOT a request to restore reachability.** The C-6 assertion is correct and the
        freeze is worth more than the hatch. The constraint is written down so it is discovered
        now rather than at the worst moment.

      Recorded here as what it
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
    - **LANDED AS A STANDING NORM (2026-07-31, `docs/ENGINEERING.md` "Tooling & commands", beside the
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
    - **POSTURE NOT CHANGED, AND NOT THE SUPERVISOR'S CALL EITHER.** docs/ENGINEERING.md invariant 7 embeds the
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
     `constants.py`), never on the plant's shape/lag/functional form — landed a1729b8 (§7 v1.4,
     pointwise-fine + per-bar bottleneck) and 39be8ba (§7 v1.4.1). (b) λ\* = 3 was calibrated
     against CHANNEL-VALUE LEGIBILITY (`id_legibility_sign_acc`, gates.py:125 — the logistic
     sign-accuracy of the dim's value from bar t's OWN id), never against detection of the planted
     rule (39be8ba). (c) Cell 5 receives IDENTICAL treatment — same fine_pointwise, same λ, same
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
  normalization-asymptote analysis derived from them is WITHDRAWN; (2) commit d9e1b53's
  "anchor re-proven" claim was FALSE — the unconditional ``w_feat_point`` buffer broke
  old-schema checkpoint loading and the M5 anchor run FAILED, but a ``| grep | tail``
  pipeline masked the exit-1, letting the unverified claim into the commit message;
  corrected in bb1e17c (buffer registered only under ``fine_pointwise``; anchor genuinely
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
  (commit `d5c6aad`) rule 2 read: *"ΔIR_info ≥ MDE_pooled = 3.209 (h=15)"* — the UNPAIRED
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
