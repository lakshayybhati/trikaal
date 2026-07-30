# M6 prereg v1.5 — AMENDMENT DRAFTS (nothing in force; every item needs a supervisor ruling)

**STATUS: DRAFTS AND RECOMMENDATIONS ONLY.** No pin, threshold, seed count, or enumeration has been
changed. `conformance.PINNED_DSR`, `verdict.DSR_N_TRIALS`, `PINNED_SEEDS`, `PINNED_MICRO_POINT_WEIGHT`
and the tabled MDE are all untouched. Nothing here is quotable as the pre-registration until ruled on
and landed in `docs/m6_prereg.md §7`.

**Why time-boxed.** Several central pre-registered numbers were derived under an assumption we have
since **measured** to be false — that training variance is negligible. They can be revised
legitimately **only while no real Cell-4 number exists**. After the first real number every item here
becomes post-hoc and fatal to the pre-registration's purpose.

**Constraint (F), honored throughout.** The eviction finding is measured on a **synthetic 13-dim
fixture**, not on real microstructure. Nothing below elevates it to a headline, and item E's reframe
is written so it cannot leak into an early claim of generality. The outcome-independent framing
committed at v1.4.3 stands unchanged.

---

# A. The DSR trial enumeration — should seeds count as trials?

## A.0 Anchor check (done first, as ordered)

**The M6 DSR pin lives entirely in `conformance.py` + `verdict.py`. `harness.py` is NOT touched by
any amendment here, and must not be.**

| location | value | role |
|---|---|---|
| `conformance.py` `PINNED_DSR["n_trials"]` | 180 | **the M6 pin**; cross-product hard-asserted against `cells × seeds × horizons × kappas` |
| `verdict.py` `DSR_N_TRIALS` | 180 | the verdict path's independent literal; must agree with the pin |
| `harness.py:216` | `5 * 3 * 4 * len(KAPPAS)` = **240** | the **M5 instrument**, frozen under the Gate-A anchor |

`harness.py:216`'s own comment already states this: *"n_trials=240 here is the HISTORICAL M5 budget,
kept numerically because run_harness is the M5 machine-validation instrument frozen under the Gate-A
anchor … it is NOT the M6 decision path."* **Confirmed: an amendment to `PINNED_DSR` leaves
`harness.py` untouched.** The two numbers are deliberately different and the conformance gate
explicitly rejects the M5 convention as an M6 input (its docstring names *"a 4-κ-only var_sr basis
(the M5 run_harness convention, which is NOT the M6 decision path)"* as a failure mode). If any
future amendment appears to require editing `harness.py`, that is a STOP-and-report, not an edit —
the fail-closed anchor rule applies.

**Mechanical consequence of a naive seed bump, confirmed by reading the gate:** `PINNED_DSR` is a
cross-product over `PINNED_SEEDS`, so 3 → 5 seeds takes N to **300** automatically, the conformance
gate **refuses the run** until the pin is amended, and a higher N makes the DSR bar **stricter**. A
seed bump therefore cannot be made silently, which is the gate working as designed.

## A.1 The argument that seeds ARE trials

1. **Conservatism is cheap and referee-proof.** A referee can always ask "you had three shots — did
   you report the best?" Counting seeds pre-empts that question without requiring anyone to trust us.
2. **var_sr benefits.** With seeds in the trial set, `var_sr` (the dispersion of per-trial VAL IRs)
   contains a **training-variance component**. Given that we have now measured basin-hopping, having
   training variance represented *somewhere* in the deflation is defensible. Excluding seeds removes
   it entirely from `var_sr`.
3. **The κ* search is per-(cell, seed).** κ* is selected on VAL *within* each (cell, seed), so each
   seed does carry its own small selection event. Under a strict reading that is a trial.
4. **Precedent/inertia.** 180 is the number the MDE table, the conformance pin and every KAT were
   built around. Changing it touches a decision surface for a reason that is arguable, not forced.

## A.2 The argument that seeds are REPLICATES, not trials

1. **The decisive one: a correct multiple-testing adjustment cannot get stricter because you
   replicated the same configuration more times.** DSR's N is the number of configurations
   **selected among** — the multiplicity of the search. Replicates of one configuration are not
   additional hypotheses. That our current specification *penalises* collecting more data on the
   same hypothesis is diagnostic of a misspecification, not of appropriate caution. A statistical
   adjustment that makes better evidence look worse is measuring the wrong thing.
2. **We do not select over seeds — we average over them, and we pre-registered that.** The clause-5
   statistic is pinned as *"Cell 4's **seed-mean** pooled headline series."* There is exactly ONE
   statistic. Nothing is maximised over seeds at any point.
3. **Bailey & López de Prado's construction is a max-of-N.** `SR₀ = √var_sr · [(1−γ)Φ⁻¹(1−1/N) +
   γΦ⁻¹(1−1/(Ne))]` is the **expected maximum** Sharpe across N trials. It is the right benchmark for
   "I searched N configurations and report the best". It is the wrong benchmark for a
   pre-specified contrast, and counting replicates into N compounds that mismatch.
4. **Point A.1.3 does not survive scrutiny.** κ* *is* selected per (cell, seed) — but κ is already in
   the cross-product as its own factor (4 κ). Counting the κ search once via the κ factor and again
   via the seed factor double-counts the same selection event.

## A.3 The numbers (computed, not asserted)

`SR₀ = √var_sr · M(N)` where `M(N) = (1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))`.

| scheme | N | M(N) | SR₀ vs status quo |
|---|---|---|---|
| (i) status quo — seeds-as-trials, S=3 | 180 | 2.7309 | 1.000× |
| (ii) seeds-as-trials, S=5 | 300 | 2.8955 | **1.060×** |
| (iii) seeds-as-replicates (S=3 **or** 5) | 60 | 2.3453 | **0.859×** |

**The N-dependence is logarithmic** (`M ~ √(2 ln N)`), so a **67% increase in N costs only 6.0%** in
SR₀. This is the single most decision-relevant number in item A, and it largely settles D.

### What `var_sr` does in each case

`var_sr` is pinned as the population variance of the per-trial de-annualized VAL IRs, so its basis
moves with the trial set. Decomposing across-config vs within-config (seed) variance:

- seeds-as-**trials**: `var_sr = V_between + V_within` (180 or 300 individual values)
- seeds-as-**replicates**: per-trial value becomes the seed-**mean** → `var_sr = V_between + V_within/S` (60 values)

SR₀ relative to status quo, as a function of `ρ = V_within/(V_between+V_within)`:

| ρ | (ii) N=300, S=5 | (iii) N=60, S=3 | (iii) N=60, S=5 |
|---|---|---|---|
| 0.00 | 1.060× | 0.859× | 0.859× |
| 0.25 | 1.060× | 0.784× | 0.768× |
| 0.50 | 1.060× | 0.701× | 0.665× |
| 0.75 | 1.060× | 0.607× | 0.543× |
| 0.90 | 1.060× | 0.543× | 0.454× |

**Pure option (iii) is a 14–55% loosening of the deflation benchmark**, because it lowers N *and*
shrinks `var_sr`. I flag that plainly: it is the principled option **and** the self-serving one, and
that combination deserves more scrutiny, not less.

## A.4 ⚠ A FINDING inside item A that may matter more than the seed question

The supervisor asked what `var_sr` does. Working that through surfaced a specification problem in
clause 5 itself.

**`var_sr` is the dispersion across ALL trials, and the trial set spans structurally different arms**
(BSQ/FSQ × OHLCV/micro/shuffled-micro). Therefore:

> **If the microstructure claim is TRUE, cell 4 separates from cells 1/2/3/5 → the cross-trial spread
> is LARGE → `var_sr` is large → SR₀ is large → clause 5 gets HARDER. If the claim is FALSE, all
> cells look alike → `var_sr` is small → clause 5 gets EASIER.**
>
> **Clause 5 as specified is anti-correlated with the hypothesis it is meant to test. A successful
> ablation raises its own deflation bar.**

Quantified. Required cell-4 **annualized** IR to pass DSR ≥ 0.95, as a function of the spread of the
annualized per-trial VAL IRs (n = 35,064 money periods, denom ≈ 1):

| sd of VAL IRs (annualized) | SR₀ (annualized) | required cell-4 IR, N=180 | N=60 |
|---|---|---|---|
| 0.5 | 1.37 | **3.01** | 2.82 |
| 1.0 | 2.73 | **4.38** | 3.99 |
| 2.0 | 5.46 | **7.11** | 6.33 |
| 3.0 | 8.19 | **9.84** | 8.68 |

For reference: **clause 4's economic floor is 0.5** annualized IR and the tabled MDE_paired is ~3.5.
The passing unit-test fixture clears clause 5 only because its cell-4 annualized IR is **11.37** with
`var_sr = 6e-6` — an effect roughly an order of magnitude above anything credible for 1-minute crypto
after 0.30% round-trip costs.

**So clause 5 is very likely the binding constraint, and may be unpassable at any realistic effect
size.** A NULL could then be a statement about the deflation calibration rather than about
microstructure — which would defeat the purpose of pre-registering it. **This needs a ruling.**
Options, none of which I am adopting:

- **(a) Keep it as-is** and pre-register explicitly that a clause-5-driven NULL is a possible outcome
  whose interpretation is "not separable from the deflation calibration". Honest, but concedes that
  the headline may be undecidable by construction.
- **(b) Re-specify the `var_sr` basis** to the dispersion of the search *within the arm being tested*
  (e.g. across (seed, h, κ) within cell 4), which is closer to Bailey & LdP's intent — the variance
  of Sharpe estimates across **the search performed for that strategy** — and removes the
  anti-correlation. Requires justifying a new basis pre-run.
- **(c) Keep DSR as a reported diagnostic** and move the multiple-testing defence onto the paired
  bootstrap + MDE (clauses 1–2), which already test the pre-registered contrast directly. Weakens
  the multiple-testing story; must not be chosen merely because clause 5 is hard.

## A.5 Recommendation for A

**Adopt a hybrid, and adopt it for the principle in A.2.1, not for the loosening:**

- **N = 60** — seeds are replicates, not trials. `N = 5 cells × 3 horizons × 4 κ`, independent of S.
- **`var_sr` computed over the FULL individual-trial set** (all 5 × S × 3 × 4 values, seeds
  included), *not* over the 60 seed-means.

Effect: `SR₀ × 0.859` — a **14.1% loosening, invariant in ρ**. This deliberately declines the second
loosening available under pure (iii): it keeps training variance inside the dispersion estimate, where
it belongs and where we now know it is real, while removing it from the multiplicity count, where it
was a category error. It is the only option on the table that is defensible on **both** axes rather
than convenient on one.

**And A.4 needs a separate ruling regardless of which scheme is chosen** — it is orthogonal to the
seed question and larger than it.

---

# B. MDE with an explicit training-variance term

## B.1 What is superseded, and what is retained

The tabled `MDE_paired` was derived from **scoring noise only** — a paired moving-block bootstrap over
time, conditional on the realized trained models. It carries **no training-variance term**. We have
since measured same-seed basin-hopping, which is a **lower bound** on across-seed variance.

**The tabled MDE is therefore superseded as a NUMBER. Its scoring-noise component is retained as a
TERM.** No number can be supplied now: the training term requires trained cells on real data, which
is the run itself. What is pre-registered now is **the formula and the estimator**; the run supplies
the value. That is legitimate pre-registration.

## B.2 The algebra

By the law of total variance, for the estimator ΔÎR computed from S seeds:

```
Var(ΔÎR) = E_models[ Var_scoring(ΔÎR | models) ]  +  Var_models( E[ΔÎR | models] )
           \_________ term 1: SCORING _________/     \______ term 2: TRAINING ______/
```

**Term 1 — scoring.** Estimated as today: the paired moving-block bootstrap **on the seed-mean
series** (B=10,000, seed 20260704, ⌈√T⌉ blocks, percentile). This already correctly reflects the
seed-mean's scoring noise, because it resamples the actual seed-mean series. → `SE_boot`.

*Note the seeds do NOT reduce term 1.* Every seed is scored on the **same** eval data and the **same**
decision grid, so their scoring noise is common, not independent. Averaging seeds does not shrink
scoring noise at all.

**Term 2 — training.** Conversely, because all seeds share the same eval data, the variation *across*
seeds in the per-seed contrast is **purely model-induced** — there is no scoring-noise contamination
to subtract. Define, per seed s:

```
ΔIR_s = IR(series_{4,s}) − IR(series_{5,s})            (and likewise 4−2 for clause 3)
σ̂²_train = sample variance of {ΔIR_s}_{s=1..S},  ddof = 1
SE_train = σ̂_train / √S
```

**Combined:**

```
SE_total   = √( SE_boot² + SE_train² )
MDE_paired = ( t_{0.95, ν} + t_{0.80, ν} ) · SE_total
```

with Welch–Satterthwaite effective degrees of freedom

```
ν = ( SE_boot² + SE_train² )²  /  ( SE_boot⁴/ν_boot  +  SE_train⁴/(S−1) )
```

## B.3 Two things this exposes

1. **Use t, not z.** With S = 3, `σ̂²_train` has **2 degrees of freedom** — `t_{0.95,2} = 2.92` vs
   `z_{0.95} = 1.645`. Continuing to use z would understate the MDE precisely in the regime where the
   new term is least well estimated. The z-based table is not conservative once a 2-df variance
   estimate enters it.
2. **S = 3 can barely estimate the term we now know we need.** 2 df vs 4 df at S = 5. This is a
   stronger argument for item D than the standard-error argument, and it is an argument the
   pre-v1.4.5 design could not have made.

## B.4 Recommendation for B

Pre-register the formula, the estimator, and the t-quantile with Welch–Satterthwaite ν exactly as
above. Record in the entry that the tabled `MDE_paired` is superseded as a number while its
scoring-noise component is retained as a term, and that the realized MDE will be **larger** than the
tabled value by construction — so this amendment makes the primary test **harder**, not easier. State
that plainly, so the direction of the change is on the record before any data exists.

---

# C. The outcome taxonomy — naming the third outcome

## C.1 The gap

`HALT_ADJUDICATE` was built into the only path that may declare a result, while the prereg continued
to describe the experiment as binary. The guards' **mechanism** is documented thoroughly; **INCONCLUSIVE
is nowhere named as a reported, publishable outcome.** A verdict word the instrument can emit but the
pre-registration does not recognise is a design gap, not an implementation detail.

## C2 Draft §-new: OUTCOME TAXONOMY (three pre-committed, publishable outcomes)

| outcome | emitted when | what is reported |
|---|---|---|
| **SURVIVES** | all five §3 clauses pass and neither guard halts | the ablation as the headline: ΔIR(4−5) with CI, MDE (incl. the B training term), all five clauses, per-seed distributions, the placebo-health diagnostic |
| **NULL** | ≥1 clause fails, neither guard halts | the ablation as a NULL with the failing clauses **named**, the point estimate and CI, and the §5 fallback verdict (CLAIMED / DESCRIPTIVE_ONLY, incl. double-NULL) |
| **INCONCLUSIVE** | either guard halts (`HALT_ADJUDICATE`) | the guard reason, the clause-derived `primary` **as computed but explicitly not claimed**, per-seed `frac_negative` / `activity_decisions` / IR, the measured effect size with CI, the realized MDE, and the mechanism result under the (F) framing |

All three are **publishable outcomes committed before the run.** INCONCLUSIVE is not a failure to
report; it is a report that the instrument's own preconditions were not met, which is more
informative than a verdict computed through a degenerate or underpowered input.

## C.3 Draft rule set for INCONCLUSIVE — selected BY RULE, never after seeing which way it went

Applied in order; the first matching rule governs. The guards produce structurally different HALT
causes and they have different remedies, which is what makes a rule possible at all.

- **R1 — POWER halt only** (`power_guard` halted, `degeneracy_guard` did not).
  Seeds directly attack this (`SE_train ∝ 1/√S`). → **Add seeds to the pre-declared cap and re-run
  ONLY the cells named in the tripped claim(s).** Re-emit the verdict. If it still halts → R3.
- **R2 — DEGENERACY halt with a cell degenerate at a MAJORITY of seeds.**
  A structurally degenerate cell is a mechanism result, not a sample-size problem, and seeds cannot
  fix it. → **No re-run. Go to R3**, and report the degeneracy as a **primary finding** (it is the
  eviction/interface mechanism appearing on real data — see E, and under the (F) framing).
- **R2b — DEGENERACY halt with a cell degenerate at a strict MINORITY of seeds** (the per-seed leg
  fires while the seed-mean is in band). That signature *is* the training lottery, which replicates
  do address. → **Treat as R1** (shares the same cap).
- **R3 — REPORT INCONCLUSIVE.** Publish per the taxonomy row above. The headline ablation claim is
  **withdrawn**, not softened; the mechanism result is reported separately and only at the strength
  (F) permits.

**BOUND (this is the part that stops it becoming a loop):**

- **At most ONE re-run round for the entire experiment**, ever.
- **Total distinct seeds ≤ 5**, whatever the starting S.
- Therefore: **if D adopts S = 5 up front, INCONCLUSIVE gets ZERO re-runs and goes straight to R3.**
- A re-run may re-train **only** the cells named by the tripped guard, at the added seeds only.
- The re-run's existence, trigger, and which rule fired are reported whatever the outcome.

**A trade-off the supervisor should decide explicitly, because the worst-case cost is identical:**
*5 seeds up front* (better first estimate always; no re-run available) versus *3 up front + 2 held in
reserve* (spends only if needed; but a rule-bound re-run still **looks** adaptive to a referee, and
the reserve is exactly the seeds that would have improved the first estimate). **I recommend 5 up
front** — same cap, strictly better estimate, and no adaptive appearance to defend.

---

# D. Seed count — conditional on A

## D.1 Cost

| | cell-runs | cost |
|---|---|---|
| S = 3 (status quo) | 5 × 3 = 15 | $20–30 (2–3 GPU-days, measured) |
| S = 5 | 5 × 5 = 25 | **$33–50**, i.e. **+$13–20** |

**Compounding risk to flag:** if invariant-7 amendment **A** (force deterministic algorithms) is also
adopted, its throughput penalty multiplies the seed cost. At an illustrative 1.3× determinism penalty,
S=5 + forced determinism ≈ **2.17× baseline ≈ $43–65**. The penalty is **PENDING** (the CUDA probe is
staged, not run). **D and B1's amendment A should be decided together, because their costs multiply
rather than add.**

## D.2 The power gain is smaller than a bare √(5/3) — and its size is unknowable pre-run

The `√(5/3) ≈ 1.29` figure applies **only to the training term.** Under the B formula
`SE_total = √(SE_boot² + σ²_train/S)`, raising S shrinks the second term only:

- if training variance **dominates** → MDE improves by the full **22.5%** (`1 − 1/1.291`)
- if scoring noise **dominates** → MDE improvement ≈ **0**
- the true position depends on the very variance ratio the run will measure

So **the headline power gain from 5 seeds cannot be quantified before the run.** That is not an
argument against it — it is an argument for stating the benefit correctly.

**The benefit that CAN be stated now is B.3.2: 4 df instead of 2 for `σ̂²_train`.** With S=3 the
training-variance term the B amendment introduces is estimated from 2 degrees of freedom, and the
t-quantile penalty for that (`t_{0.95,2}=2.92` vs `1.645`) is severe. **S=5 is what makes B's term
usable rather than merely present.** That is the real case for D.

## D.3 Does the stricter DSR bar cancel the gain?

**Under the A.5 hybrid recommendation: no, because there is no stricter bar.** N = 60 independently of
S, so 3 → 5 seeds changes N **not at all**. `var_sr` is computed over more values (a better estimate
of the same population variance, with no systematic direction). **Clause 5 is unchanged; clause 2
improves. Net sign: positive.**

**Under status quo seeds-as-trials:** N 180 → 300 tightens SR₀ by **6.0%** while the MDE improves by
somewhere in 0–22.5%. Net sign is **probably** positive but genuinely not determinable pre-run,
because the two effects land on **different conjunctive clauses** and which one binds is unknown. I
would not describe that as a change that "nets positive" with confidence.

## D.4 Recommendation for D

**Adopt S = 5, conditional on the A.5 hybrid being adopted first.** With the hybrid, D is
unambiguously non-negative on clause 5 and positive on clause 2, at +$13–20. **If A.5 is rejected and
seeds remain trials, I do not recommend the bump** on power grounds alone — the 6% DSR tightening
against an unquantifiable MDE gain is exactly the "net sign non-obvious" trade the supervisor said he
would not buy. In that branch the *only* remaining argument for S=5 is B's degrees of freedom, which
is real but should be weighed against the coupled cost in D.1 rather than treated as free.

---

# E. Legibility-gate adjudication — pre-written

## E.1 The reframe (state this in the prereg, because it changes what a firing MEANS)

`gates.py` already commits to stop-and-report with per-dim receipts and forbids a silent threshold
move. **That is correct and stays.** What is missing is the *content* of the adjudication, and the
framing that makes a firing interpretable:

> Our own measured mechanism says reconstruction-trained tokenizers allocate code capacity by variance
> and covariance, never by downstream value. **Real TFI is low-variance and weakly correlated with
> price shape — exactly the eviction profile.** The λ fix was validated on a **synthetic planted
> signal with tuned properties**, never on real microstructure. **So the gate firing is the MODAL
> prediction of our own finding, not an edge case.**
>
> And if it fires, **that firing is the real-data confirmation of the eviction mechanism** — the
> measurement that would upgrade it from a synthetic result to a general one. It is simultaneously the
> ablation's blocker and the mechanism's strongest evidence.

**(F) compliance:** this is written as *what a firing would license*, conditional and unearned. It
does **not** assert generality now. Until the gate actually fires on real microstructure, the finding
remains **a mechanism demonstrated in a controlled setting**, stated exactly that way, and it is not
elevated to headline anywhere.

## E.2 Critique of the supervisor's proposed rule (asked for, so: where it is wrong)

Proposed: *λ may be re-derived ONCE by the already-pinned formula, on VAL data (block 0) ONLY, with
criterion and fold fixed in advance, no iteration; if it still fails, proceed under the (C) taxonomy
with the gate failure as a primary result. Forbid tuning λ on any scored block; forbid a second
attempt.*

The shape is right — one shot, pinned formula, no iteration, failure is a result. **Five problems:**

1. **Block 0 would be doing a third job, and the wrong kind of job.** Block 0 already supplies κ*
   selection **and** all 180 clause-5 DSR trial entries. Adding λ makes three uses of one block, and
   repeated reuse of a validation set is a known overfitting channel. Worse, **λ is a TRAINING
   hyperparameter and the legibility gate fires during training, before any scoring.** Using
   forward-region data to set a training hyperparameter is not lookahead in the invariant-2 sense
   (block 0 is excluded from the headline), but it inverts the train/eval separation for no reason.
   **I checked: the real run has NO in-train holdout** — `build_symbol_windows(boundary_ms=boundary)`
   sets the boundary at the train/eval calendar split, so training draws from the entire train
   region. The canary's `HOLD_FRAC` is canary-only. **Better alternative: carve a held-out slice from
   the END of the TRAIN region and re-derive λ there.** It costs nothing but a boundary, keeps every
   eval block untouched by any training decision, and puts a training hyperparameter where training
   hyperparameters belong. This does not currently exist and would need creating.
2. **The pinned formula is the instrument that just failed.** λ=3.0 was calibrated on a fixture tuned
   to lake entropy. If the gate fires, the premise is that real micro does not behave like that
   fixture — which is precisely the assumption the formula encodes. **Re-deriving by the same formula
   may be re-applying a broken instrument.** The rule should require the re-derivation to emit the
   formula's own calibration receipt on **real** data (the per-dim variance/covariance profile) so a
   reader can see whether the formula's premise held, rather than only its output.
3. **The re-derivation is a second configuration and the DSR does not pay for it.** Even bounded to
   one attempt, two λ values will have been tried and the reported one chosen after seeing a failure.
   Under seeds-as-trials logic that is a genuine additional trial; under the A.5 hybrid the honest
   accounting is that the **configuration count** rises. **Whatever A rules, the λ re-derivation must
   be added to the multiplicity count** — otherwise the search has been widened without paying.
   *(This is the one place where E and A interact and it is easy to miss.)*
4. **The branch structure is biased in one direction.** If the re-derivation succeeds → full ablation.
   If it fails → report the failure. So the contingency can only ever *help* the ablation's chances,
   never hurt them. That must be disclosed, and the paper must state which λ was used and that it was
   the second one. An ablation under a re-derived λ is **weaker evidence** than one under the original
   pin, and the write-up has to say so rather than presenting them as equivalent.
5. **A cleaner alternative that removes problem 4 entirely.** Pre-declare that **if the gate fires,
   the PRIMARY result becomes the mechanism finding** and any ablation under a re-derived λ is
   **explicitly secondary/exploratory**. This is cheaper (no bias to defend), it matches E.1's own
   logic — a firing is the mechanism's strongest evidence — and it removes the incentive to hope the
   re-derivation succeeds.

## E.3 Draft amended rule for E (for ruling)

1. **The gate fires → STOP and report**, with per-dim receipts, unchanged from `gates.py`. Silent
   threshold moves remain forbidden.
2. **The PRIMARY result becomes the mechanism finding**, reported at (F) strength only. The ablation
   headline is not the primary in this branch.
3. **λ may be re-derived at most ONCE**, by the pinned formula, with criterion and fold fixed in
   advance, **on a held-out slice carved from the END of the TRAIN region** — never on block 0, never
   on any scored block. Creating that slice is part of this amendment. No iteration; no second
   attempt; the search is exhausted after one.
4. The re-derivation **must emit the formula's calibration receipt on real data** (per-dim variance
   and covariance profile vs the fixture's), so its premise is auditable, not just its output.
5. The re-derived λ **counts as an additional configuration in the clause-5 multiplicity**, per
   whatever A rules.
6. Any ablation computed under a re-derived λ is reported as **SECONDARY/exploratory**, labelled with
   which λ was used and that it was the second, and never as the pre-registered primary.
7. If the gate still fails after the single re-derivation → the (C) taxonomy governs, with the gate
   failure as a primary result.

---

# F. Compliance statement

The eviction/mechanism finding is **measured on a synthetic 13-dim fixture, not on real
microstructure**. As it stands it is a claim about tokenizers trained on data we generated. It is
**not** elevated to headline in this document or any other; E.1's reframe is written strictly as
*what a real-data firing would license*, and item E is the thing that would earn the stronger claim.
Until then it is **a mechanism demonstrated in a controlled setting**, stated exactly that way. The
outcome-independent framing committed at v1.4.3 stands unchanged.

---

# Summary of recommendations (each needs a ruling; none is in force)

| item | recommendation | confidence / note |
|---|---|---|
| **A** | Hybrid: **N = 60** (seeds are replicates) with **`var_sr` over the full individual-trial set** | Strong on principle; declines the second loosening available; SR₀ × 0.859 |
| **A.4** | ⚠ **Separate ruling needed** — clause 5 is anti-correlated with its own hypothesis and may be unpassable at realistic effect sizes | Highest-consequence item in the window |
| **B** | Pre-register the two-term MDE formula + estimator, **t-quantile with Welch–Satterthwaite ν** | Makes the primary test HARDER; direction on the record pre-data |
| **C** | Three-outcome taxonomy; INCONCLUSIVE response by rules R1/R2/R2b/R3; **≤1 re-run round, ≤5 total seeds** | — |
| **D** | **S = 5, conditional on A.5.** If A.5 is rejected, do **not** bump on power grounds | Real benefit is B's df (2→4), not the √(5/3) SE story |
| **E** | Amended rule E.3 — mechanism becomes primary; λ re-derived on an **in-train** holdout (must be created), counted in multiplicity, ablation labelled secondary | Supervisor's rule is right in shape, wrong in 5 specifics |

**Nothing pinned was changed. No spend. The staged CUDA probe stays staged.**
