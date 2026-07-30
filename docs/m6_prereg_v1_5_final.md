# M6 prereg v1.5 — FINAL AMENDMENT TEXT (awaiting sign-off; NO pin changed)

**STATUS: final text for ruling. Nothing is in force.** `conformance.PINNED_DSR` (N=180),
`verdict.DSR_N_TRIALS`, `PINNED_SEEDS` (0,1,2), `PINNED_MICRO_POINT_WEIGHT` (3.0) and the tabled
`MDE_paired` are untouched and verified so. On sign-off this text lands in `docs/m6_prereg.md §7` and
the code changes it names are implemented in one pass.

Supersedes `docs/m6_prereg_v1_5_drafts.md`, which remains as the working record.

---

# 0. THE SAFEGUARD (binding on A.4, A.5, B and D together)

## 0.1 Direction-blind test — answered explicitly per amendment

> *"Would we adopt this if it TIGHTENED the bar?"* Any amendment failing this does not land.

| amendment | answer | evidence that this is not post-hoc |
|---|---|---|
| **A.5** (N=60) | **YES** | The principle — replicates are not hypotheses — is direction-free. **Decisive evidence: the same amendment DELIBERATELY KEEPS `var_sr` on the full individual-trial set**, which is the tightening half of the available choice. Pure seeds-as-replicates would have given 0.45–0.86×; we took 0.859× and declined the rest. A motivated actor takes both. |
| **A.4** (placebo `var_sr` basis) | **YES** | The supervisor stated it first: *"a specification error I would fix if it made the gate harder."* **And the direction is genuinely UNKNOWN at adoption**: §A.4.3 identifies a mechanism (shuffle-induced training degradation) by which the placebo basis could **inflate** `var_sr` and make clause 5 **harder**. We are adopting it without knowing which way it moves. |
| **B** (training-variance MDE) | **N/A — it IS a tightening** | Raises the MDE twice over (§0.2). Adopted anyway, with the direction recorded pre-data. |
| **D** (S=5) | **YES** | More replicates is better evidence regardless of which way the bar moves — the exact A.5 argument run in reverse. If collecting more data made the bar harder we would still collect it. |

## 0.2 ⚠ THE AGGREGATE — and a correction to the ruling's premise

The ruling states *"Every amendment in this window loosens the effective bar."* **That is not
correct, and the aggregate must not be reported that way.** B is a substantial **tightening** of the
clause that most directly tests the headline contrast.

**Clause 5 (deflation) — LOOSENS.** `SR₀` scales by `[M(60)/M(180)] · √f`, where
`f = var_sr(cell 5) / var_sr(all trials)`:

| f | 1.0 | 0.75 | 0.50 | 0.25 | 0.10 |
|---|---|---|---|---|---|
| combined A.5 + A.4 | **0.859×** | 0.744× | 0.607× | 0.429× | 0.272× |

`f` is unmeasurable pre-run. A.5 alone is a known **0.859×**; A.4's contribution is unknown in
magnitude **and** direction (§A.4.3).

**Clause 2 (the primary contrast test) — TIGHTENS, twice over.**

1. Quantile multiplier `z → t`: `2.4865 → 3.9806` at S=3 (**1.601× harder**), `→ 3.0728` at S=5
   (**1.236× harder**).
2. `SE_total = √(SE_boot² + SE_train²) ≥ SE_boot` **always** — a second, strictly-positive increase.

**The honest aggregate statement, for the paper and for any referee:**

> The v1.5 amendments **loosen the deflation clause by a factor of 0.859 times an unmeasured √f, and
> tighten the primary contrast clause by at least 1.24–1.60× on the quantile multiplier alone plus a
> strictly positive variance term.** The net effect across the conjunctive gate is **indeterminate
> pre-run and is NOT monotonically loosening.** No cosmetic offset has been applied anywhere.

## 0.3 DUAL-SPECIFICATION REPORT — binding, wired as a required manifest field

The verdict path computes and reports the headline outcome under **both** specifications from the
**same artifacts**. ~$0 (one extra pass over data already in memory).

Required manifest field `dual_specification` (schema drafted; implemented on sign-off):

```
"dual_specification": {
  "v1_2_original":  {"n_trials": 180, "var_sr_basis": "all_trials",
                     "mde_rule": "z-quantiles, scoring-noise SE only",
                     "sr0":…, "dsr":…, "mde_paired":…, "clauses":{…}, "primary": "SURVIVES|NULL"},
  "v1_5_amended":   {"n_trials": 60,  "var_sr_basis": "cell5_placebo",
                     "mde_rule": "t-quantiles (Welch–Satterthwaite), SE_boot ⊕ SE_train",
                     "sr0":…, "dsr":…, "mde_paired":…, "clauses":{…}, "primary": "SURVIVES|NULL"},
  "agree": true|false,
  "disagreement_is_a_first_class_finding": true
}
```

**Rules.** The **v1.5** result is the pre-registered primary. If the two disagree, the disagreement
is a **FIRST-CLASS FINDING**: reported prominently in the abstract and results, never resolved in our
favour and never relegated to an appendix. A `SURVIVES` under v1.5 with `NULL` under v1.2 must be
reported as *"survives only under the amended specification"*, with both numbers. The field is
**required** — a manifest lacking it cannot be quoted as the M6 outcome, on the same footing as
`grid_pinned`.

**No compensating tightening has been invented elsewhere.** An arbitrary offset would be its own
unprincipled adjustment; the dual report is the safeguard.

---

# A.4 — `var_sr` basis: ADOPTED, option (b), basis = THE PLACEBO ARM (cell 5)

## A.4.1 The principle (as ruled)

> **A null dispersion estimate must not contain the treatment contrast.**

`var_sr` spanning structurally different arms is a specification error independent of direction.
Options (a) and (c) are rejected as ruled; (c) is retained **only** as a secondary framing note, never
as the fix.

## A.4.2 Recommendation: cell 5, and I disagree with the proposed fallback

**Primary basis: cell 5 (the shuffled-microstructure placebo).** `var_sr` = population variance of
the de-annualized VAL IRs across cell 5's `(seed, h, κ)` trials.

Why cell 5 over my own draft proposal (within-cell-4): **cell 4 is the treated arm.** Its own
dispersion can still be shaped by the treatment working — a weaker echo of the very anti-correlation
we are removing. Cell 5 is the only arm that is **signal-free by construction** (the shuffle destroys
temporal alignment while preserving the marginal distribution), and it is matched to cell 4 on
quantizer (FSQ) **and** input dimensionality (16 dims). Using the placebo to calibrate the null is
what a placebo is for.

**But I do not accept cell 2 as the fallback, and this is the one place I push back on the ruling.**

> **Cell 2 is not signal-free.** It is FSQ + OHLCV-only — the baseline arm — and §5's NULL-fallback
> explicitly contemplates `IR(2) − IR(1)` being **CLAIMED**. An arm we have pre-registered as
> potentially carrying a claimable edge cannot serve as the null dispersion. Its spread would reflect
> a real OHLCV signal's search dispersion, not a null's. It is also mismatched on dimensionality
> (7 vs 16), which affects capacity and therefore dispersion.

**Fallback if the §A.4.3 tripwire fires: report clause 5 under BOTH cell 5 and within-cell-4** — the
two principled bases — and disclose both. Never cell 2.

## A.4.3 The placebo-victim concern, worked through (as ordered)

The fixture measured `pb(5−2) = −2.59`: the shuffle harmed cell 5 below the OHLCV-only counterfactual.
Does that disqualify cell 5 as a null?

1. **Not directly, and the reason is dispositive: `pb(5−2)` is a statement about the LEVEL (a
   difference of means); `var_sr` is a DISPERSION statistic.** A systematic level shift does not move
   a variance. The placebo-victim finding therefore does not impugn cell 5's dispersion at first
   order.
2. **There is a genuine second-order concern.** If the shuffle actively degrades training, cell 5's
   models may be **more variable across seeds**, inflating its dispersion.
3. **That error direction is CONSERVATIVE.** Inflated `var_sr` → higher `SR₀` → clause 5 **harder** →
   errs toward NULL. It fails safe, and it is why A.4 passes the direction-blind test with the
   direction genuinely unknown at adoption.
4. **It is measurable, not assumed.** Pre-registered **tripwire**: report the across-seed IR
   dispersion of **all five arms** side by side. If cell 5's dispersion exceeds **1.5×** the median
   dispersion of cells 1, 2 and 3, disclose it as a possible shuffle-degradation artifact **and**
   report clause 5 under both bases per §A.4.2. The 1.5× threshold is fixed here, pre-data.

---

# A.5 — N = 60, and this is where the chain correctly stops

## A.5.1 The principle, restated precisely

The ruling asks whether 60 is principled or a convenient stopping point, and pushes the chain toward
κ alone. **The chain stops immediately after seeds, and it stops on a principle, not on arithmetic.**

> **N counts DISTINCT CONFIGURATIONS evaluated. It does not count REPLICATES of a configuration.**
> Seeds are replicates of one configuration. Cells, horizons and κ are distinct configurations.
> **The seed axis is the only replicate axis, so it is the only axis the principle removes.**

## A.5.2 Why the chain does NOT continue to κ alone

The ruling's extension rests on *"we could not have reported cell 2 as the headline, so cells are not
selected."* That conflates two different things:

- **What we pre-registered ourselves to REPORT** — a constraint on us, ex post.
- **The multiplicity of what was COMPUTED and is shown** — which is what DSR discounts.

`SR₀` is the **expected maximum** across N trials. It asks: *given a search of this size, how large
would the best result be by chance?* We do compute all 5 cells × 3 horizons × 4 κ, and **we report
them** — the ablation table and the secondary horizons are in the paper. A referee sees 60
configurations. Pre-registering which one is the headline reduces **selection** risk; it does not
reduce the **search** that was performed and displayed. Removing cells and horizons from N would
claim credit for a discipline (pre-registration) that DSR is not measuring.

The numbers, so the size of the declined loosening is explicit:

| stopping point | N | M(N) | vs status quo |
|---|---|---|---|
| κ only | 4 | 1.0521 | 0.385× |
| κ × horizons | 12 | 1.6648 | 0.610× |
| **cells × horizons × κ (recommended)** | **60** | **2.3453** | **0.859×** |
| status quo | 180 | 2.7309 | 1.000× |

**We are declining a further 0.385× loosening that the ruling half-offered.** That is the answer to
"is 60 convenient": a convenient stopping point would have been 4.

**Disclosed limitation, arguing the other way:** N=60 counts only the final pre-registered evaluation
grid. It does **not** count development-time search (tokenizer variants, λ calibration, estimator
choice, canary horizons). A referee could argue the effective N exceeds 180. This is a reason **not**
to go below 60 and is stated in the amendment text rather than left implicit.

## A.5.3 The amendment

- `PINNED_DSR["n_trials"]`: 180 → **60**, enumerated as `cells × horizons × κ` (**seeds removed from
  the cross-product**).
- `var_sr` basis: per **A.4** — cell 5's `(seed, h, κ)` trials. **Seeds remain in the `var_sr`
  computation** (dispersion), while being absent from N (multiplicity). These are deliberately
  different sets and the amendment text says why.
- `verdict.DSR_N_TRIALS` mirrors it; the conformance cross-product assert is updated to the new
  factorization and continues to hard-fail on any divergence.
- **`harness.py` is NOT touched.** Its `5*3*4*len(KAPPAS)` = 240 is the M5 instrument frozen under
  Gate-A. Re-verified in this pass; the fail-closed rule stands.

---

# B — MDE with a training-variance term: ADOPTED AS DRAFTED

Unchanged from the draft, which the ruling accepted. Restated for the final text:

```
Var(ΔÎR) = E_models[Var_scoring(ΔÎR | models)]  +  Var_models(E[ΔÎR | models])
ΔIR_s      = IR(series_{4,s}) − IR(series_{5,s})        (and 4−2 for clause 3)
σ̂²_train   = var({ΔIR_s}, ddof=1)                        [2 df at S=3; 4 df at S=5]
SE_train   = σ̂_train / √S
SE_total   = √(SE_boot² + SE_train²)
MDE_paired = (t_{0.95,ν} + t_{0.80,ν}) · SE_total
ν          = (SE_boot² + SE_train²)² / (SE_boot⁴/ν_boot + SE_train⁴/(S−1))
```

Seeds do **not** reduce `SE_boot` (all seeds are scored on the same data and grid, so scoring noise is
common). Conversely `σ̂²_train` is **clean** — cross-seed variation is purely model-induced with no
scoring contamination to subtract.

**Direction recorded pre-data: this makes the primary test HARDER**, by 1.24–1.60× on the quantile
multiplier plus a strictly positive variance term. The tabled `MDE_paired` is superseded as a
**number**; its scoring-noise component is retained as a **term**.

---

# C — Outcome taxonomy ADOPTED; the seed tension put to the supervisor

Taxonomy (SURVIVES / NULL / INCONCLUSIVE), rules R1 / R2 / R2b / R3, and the bounds — **≤1 re-run
round ever, ≤5 total distinct seeds, re-train only the cells named by the tripped guard** — are
adopted as drafted.

## C.1 The tension, stated not absorbed

**S=5 up front spends the re-run budget in advance.** Both options, with consequences:

| | **Option 1 — S=3, 2 in reserve** | **Option 2 — S=5 up front** |
|---|---|---|
| first-estimate quality | σ̂²_train on **2 df**; quantile multiplier **3.9806** | σ̂²_train on **4 df**; multiplier **3.0728** (**22.8% lower**) |
| INCONCLUSIVE remedy | **available** — R1/R2b can fire | **none** — R1/R2b collapse into R3 |
| cost | $20–30, conditional +$13–20 | $33–50 always |
| referee optics | a rule-bound re-run still **looks** adaptive | nothing adaptive to defend |
| worst-case cost | identical | identical |

**The substantive asymmetry, which is not about optics:** if the power guard fires at **S=3**, that is
plausibly a sample-size artifact two more seeds would fix. If it fires at **S=5**, the effect is
genuinely small relative to training variance and more seeds would be marginal — so having no remedy
at S=5 is an **honest stopping point**, not a lost option.

**My recommendation: Option 2 (S=5 up front)** — the df gain applies to the first and only estimate,
the missing remedy is one we would rarely want to exercise, and there is no adaptive appearance to
defend. **But this is the supervisor's call and I am not treating it as settled.**

---

# D — S=5: CONDITIONALLY APPROVED (as ruled)

Conditional on A.4 landing and C.1 being resolved. Recorded facts:

- Cost: 25 cell-runs, **$33–50** (+$13–20).
- The √(5/3) framing applies **only** to the training term; `SE_boot` is untouched, so the MDE gain is
  **0–22.5% and not quantifiable pre-run**.
- **The statable, quantified, pre-run benefit is B's degrees of freedom, 2 → 4**: `t_{0.95}` falls
  **2.9200 → 2.1318 (−27.0%)**, and the combined multiplier **3.9806 → 3.0728 (−22.8%)**. This is the
  strongest argument for more seeds on the table.
- Under **A.5**, N = 60 independently of S, so **there is no DSR tightening at all** — clause 5
  unchanged, clause 2 improved. **Net sign: positive.**
- **FOR LAKSHAY, WITH THE SEED DECISION, NOT SEPARATELY:** D compounds **multiplicatively** with
  invariant-7 amendment A. At an illustrative 1.3× determinism penalty, S=5 + forced determinism ≈
  **2.17× baseline ≈ $43–65**. The penalty is PENDING the staged CUDA probe.

---

# E — Legibility-gate adjudication: REVISED per the adopted corrections

The reframe (§E.1 of the drafts) stands, written strictly as what a real-data firing **would**
license. The final rule:

1. **The gate fires → STOP and report**, with per-dim receipts. `gates.py`'s existing behaviour is
   unchanged; silent threshold moves remain forbidden.
2. **The PRIMARY result becomes the mechanism finding**, reported at §F strength only. The ablation
   headline is not the primary in this branch. *(This removes the branch-structure bias in item 4
   below at its source.)*
3. **λ may be re-derived at most ONCE**, by the pinned formula, criterion and fold fixed in advance,
   **on a held-out slice carved from the END of the TRAIN region — never from block 0, never from any
   scored block.** Creating that slice is part of this amendment; it does not currently exist
   (verified: `HOLD_FRAC` appears only in `scripts/m6_canary.py`; the real path's `boundary_ms` is the
   train/eval calendar split, and `val_block` 0 sits **after** `calendar_boundary_ms` — it is forward
   eval data, not a train holdout). No iteration; the search is exhausted after one attempt.
4. **The re-derivation must emit the pinned formula's calibration receipt on REAL data** (per-dim
   variance and covariance profile vs the fixture's), so its premise is auditable and not merely its
   output — the formula is, after all, the instrument that just failed.
5. **The re-derived λ counts as an additional configuration in the clause-5 multiplicity.** Under
   A.5's factorization this takes N from 60 to **120** for the affected report. **DISCLOSED
   EXPLICITLY IN THE AMENDMENT TEXT, not in passing:** widening the search without paying for it is
   exactly the failure DSR exists to prevent.
6. **The branch structure is biased and this is disclosed, not just noted.** If the re-derivation
   succeeds → full ablation; if it fails → report the failure. The contingency can therefore **only
   ever help the ablation's chances, never hurt them.** Any ablation computed under a re-derived λ is
   reported as **SECONDARY/exploratory**, labelled with which λ was used and that it was the second,
   and never as the pre-registered primary. Rule 2 makes this mostly moot by demoting the ablation in
   this branch anyway.
7. If the gate still fails after the single re-derivation → the **C** taxonomy governs, with the gate
   failure as a primary result.

---

# F — Constraint honored

The eviction/mechanism finding is measured on a **synthetic 13-dim fixture**, not on real
microstructure — a claim about tokenizers trained on data we generated. It is **not** elevated to
headline anywhere in this text. E's reframe is written strictly as what a real-data firing **would**
license: conditional and unearned. Until then it is **a mechanism demonstrated in a controlled
setting**, stated exactly that way. The v1.4.3 outcome-independent framing stands unchanged.

---

# Implementation manifest (on sign-off only — nothing done yet)

| change | file | note |
|---|---|---|
| `n_trials` 180 → 60; drop seeds from the cross-product | `conformance.py` `PINNED_DSR` | assert updated to the new factorization |
| `DSR_N_TRIALS` 180 → 60 | `verdict.py` | independent literal; must agree |
| `var_sr` basis → cell 5 trials | `verdict.py`, `conformance.py` | seeds retained in `var_sr`, absent from N |
| two-term MDE + t/Welch–Satterthwaite | `paired_bootstrap.py`, `verdict.py` | direction: harder |
| `dual_specification` required manifest field | `verdict.py`, `m6_verdict.py` | refuse-to-quote if absent, like `grid_pinned` |
| per-arm dispersion table + 1.5× tripwire | `verdict.py` | A.4.3 |
| `PINNED_SEEDS` → 5 seeds | `conformance.py` | **only if C.1 Option 2 is ruled** |
| train-region calibration slice | `orchestrator.py`, loader | **only if the legibility gate fires** |
| **`harness.py`** | — | **NOT TOUCHED. Gate-A anchored. Fail-closed.** |

Every one of these is gated on sign-off. **No pin has been changed. No spend. The staged CUDA probe
stays staged.**
