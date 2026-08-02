# C-3 — the clause-5 unit fix: a decision for Lakshay (drafted 2026-08-02, §7 v1.6.14 DRAFT)

**STATUS: DRAFTED AND HELD. NOT IMPLEMENTED.** The supervisor has ruled the fix correct and
primary. It is written up here and stops, because **this is the third reopening of a design twice
declared frozen and it changes a clause threshold.** That combination is Lakshay's call, not the
builder's and not the supervisor's.

---

## 1. In plain terms, before any notation

One of the five conditions the result must pass (clause 5, the deflated Sharpe ratio) builds its
bar by averaging numbers **measured in three different time units** — per-5-minute, per-15-minute
and per-60-minute — and then compares that bar against a number measured in **one** unit
(per-15-minute). It is a unit error, of the same kind as averaging metres with feet and inches.

**The bar it produces is 8.70% too low on the fixture we can measure.** Fixing it makes our own
headline result **harder** to claim.

---

## 2. What is wrong, from the code

`src/trikaal/eval/verdict.py`, `enumerate_dsr_trials`:

```python
trials[(cid, seed, h, k)] = ir_ann / float(np.sqrt(periods_per_year(h)))
#                                                              ^ each trial's OWN horizon
```

Each trial's **annualized** VAL IR is divided by `sqrt(periods_per_year(h))` **at that trial's own
horizon**, so the value is a per-**h**-minute Sharpe. With `DSR_HORIZONS = (5, 15, 60)` the divisors
are `324.2221460665511 / 187.18974330876145 / 93.59487165438073`.

Then:

- `deflated_sharpe_ratio` → `expected_max_sharpe(var_sr, N)` → **SR₀ ∝ √var_sr**, and
- `probabilistic_sharpe_ratio` compares SR₀ against `_sharpe(headline_series)`, whose grid
  `_validate_artifact` pins to **`PRIMARY_H = 15`**.

**So SR₀ is compared against a per-15-minute Sharpe while being built from a per-5/15/60 blend.**

### The span, verified exactly

`324.2221460665511 / 93.59487165438073` = **3.4641016151377544**, which is `math.sqrt(12)` —
**bit-equal**, asserted as `span == math.sqrt(12.0)` in the probe, not merely equal to displayed
precision. One and the same annualized IR enters the trial set **3.4641× larger at h=60 than at
h=5**, purely from the horizon it was measured at.

---

## 3. The fix

**One token.** `h` → `PRIMARY_H`:

```python
trials[(cid, seed, h, k)] = ir_ann / float(np.sqrt(periods_per_year(PRIMARY_H)))
```

Every trial is then expressed in the unit the tested statistic is actually in. `var_sr` becomes
`Var(annualized IRs) / periods_per_year(15)` — a single common rescaling, so the trial set carries
one unit instead of three.

### Implementation surface (for scale, not for approval-by-detail)

| site | change |
|---|---|
| `verdict.enumerate_dsr_trials` | the one token above |
| `conformance.PINNED_DSR` | a new pinned key recording the de-annualization convention, so the convention is pinned rather than implicit |
| `verdict.assemble_verdict` | `dual_specification` gains the mixed-unit leg (below) |
| KATs | a mutation test proving the gate REJECTS the per-trial-horizon convention, in the shape every other pin already has |
| prose | the clause-5 rule string and §3 body, which now derive from the pins |

`verdict_dsr_failures` needs no change: it recomputes `var_sr` from the trial dict it is handed and
asserts bit-equality, making no unit assumption of its own.

**Nothing else moves.** The trial set feeds clause 5 and the v1.2 leg only; clauses 1–4, the MDE,
the bootstraps, both guards and the money path are untouched.

---

## 4. Before / after

The M6 values do not exist yet, so these are the **measured illustration** from the only real
multi-cell artifact set in the repo (the 3-seed toy, `runs_cloud/runs/m6_toy/eval`), plus the
**exact algebra**, which is data-independent.

| quantity | as pinned (mixed units) | unit-consistent (all at h=15) | change |
|---|---|---|---|
| `var_sr` (cell-5 basis) | 0.0270932216941985 | 0.032502330107555405 | ×1.1996 |
| **SR₀ (the clause-5 bar)** | **0.38604302499386645** | **0.4228269234318554** | **+9.53%** |
| minimum passing SR̂ | 0.43839918320399485 | 0.47569310299032647 | **+0.0373** |

Read the other way: **the pinned basis understates SR₀ by 8.70%.**

### It is outcome-material, proven by construction

The toy fixture **cannot** answer this — cell 4's DSR saturates at **0.0 under all four unit
bases**, so its pass/fail column is uninformative and is declared as such. The outcome question is
answered instead with a deterministic n=1128 unit-sd series whose mean is exactly SR̂:

> **Witness SR̂ = 0.45704614309716063 → DSR 0.9868 (PASS) as pinned, 0.8577 (FAIL)
> unit-consistent.**

A non-empty band of cell-4 Sharpes passes clause 5 under the frozen recipe and fails under a
unit-consistent one. This is not a rounding-scale difference.

Receipt: `runs_manifest/m6_c3_dsr_units.json`. Probe: `scripts/m6_c3_dsr_units.py` (control-armed —
it re-derives the toy manifest's own `var_sr` bit-exactly before comparing anything, and emits
`PROBE INVALID` if it cannot).

---

## 5. Why this could not wait for the run

**The direction is not signed pre-data, and that is what makes it urgent rather than academic.**

Converting to the h=15 unit rescales the h=5 group by `sqrt(5/15) = 0.5774` and the h=60 group by
`sqrt(60/15) = 2.0` **about zero**, so whether the mixed basis lands above or below the correct one
depends on the realized per-horizon means and spreads. On the toy it is **anti-conservative**. On
the real cells it is **unknown**.

The C-12 placebo-dispersion tripwire had an *errs-conservative* defence — it could only make the
gate harder, so leaving it was safe. **C-3 has no such defence.** Discovering the direction after
the data exists is exactly the situation pre-registration is built to prevent.

---

### 5a. TIMING — the window is open now and closes when we start spending

**If the outcome data does not yet exist, the amended specification inherits FULL pre-registration
status.** Amended after the run, the identical change is a post-hoc correction *forever*, and no
amount of correctness recovers that: a referee cannot distinguish a principled unit fix from one
that happened to help, once the numbers are visible.

**No M6 cell has been trained.** The window is open today and closes the moment we spend.

### 5b. TWO COMPETENT ANALYSES DISAGREE ON THE SIGN — which is the evidence §5 was missing

§5 asserts the direction is **not knowable pre-data**. Until now that was an argument from the
algebra (the h=5 group rescales by 0.5774 and the h=60 group by 2.0 *about zero*, so the sign
depends on realized per-horizon means and spreads). It now has evidence:

| analysis | predicted direction of the pinned mixed basis |
|---|---|
| this document's toy-fixture measurement | **ANTI-conservative** — var_sr 0.8336× the h=15-consistent basis, SR₀ understated 8.70% |
| an independent second model's null-model analysis | **CONSERVATIVE by ~4/3** |

**Two competent analyses reaching opposite signs is not a problem with either — it is the
proposition being demonstrated.** §5 claims the sign is unknowable before the data exists; two
capable attempts disagreeing is exactly what that claim predicts. The disagreement is therefore
*supporting evidence for the amendment*, not an argument to wait for a tiebreak.

Neither number is load-bearing for the fix. The fix rests on the span (bit-exact √12), the
mechanism (from code), and outcome-materiality (proven by construction with a witness). **The two
sign predictions are load-bearing only for the claim that the direction cannot be signed in
advance — and on that they agree by disagreeing.**

### 5c. A SUPERVISOR CORRECTION, ON THE RECORD

The supervisor quoted the 8.70% figure to Lakshay as *"the deciding fact"* and stated that the
error makes the test easier to pass. **This document says twice that the direction is
data-dependent and unknown on the real cells, and the receipt labels the fixture illustrative.**
The qualifier was dropped in the summary. The draft is correct as written; the summary of it was
not. Recorded here on the same terms as the builder's own withdrawn claims — the log is not a
one-way ledger.

## 6. The direction-blind test, applied

The v1.5 amendment window fixed a standing test: *would we adopt this amendment if it moved the bar
the other way?* Here it is not hypothetical:

- **The fix TIGHTENS.** +9.53% on SR₀ on the only fixture that can be measured.
- **It is proposed PRE-DATA.** No M6 cell has been trained. Nobody has seen a result.
- **It corrects an unambiguous unit error, not a judgement call.** There is no defensible reading
  under which a Sharpe measured per 60 minutes and a Sharpe measured per 5 minutes belong in one
  variance that is then compared against a Sharpe measured per 15 minutes.

**And the inverse matters more than the tightening.** Keeping the pinned basis is *not* a neutral
act of fidelity to the freeze: it retains a threshold that is simultaneously **wrong** and
**favourable to us**. A freeze that protects an error in our own favour protects nothing worth
protecting.

---

## 7. What is retained, so nobody takes our unit judgement on trust

**The mixed-unit basis is not deleted.** It is computed from the same artifacts and reported in
`dual_specification`, on the same footing as the v1.2 and v1.5 legs, and **a disagreement between
legs is a first-class finding stated in the abstract** — never resolved in our favour, never
relegated to an appendix.

That machinery already exists and is a REQUIRED manifest field. This adds a computation and **no
new convention, no new threshold, and no new degree of freedom.**

**The v1.2 leg keeps its own mixed units.** It exists to reconstruct the pre-amendment rule
faithfully; unit-correcting it would defeat its only purpose.

---

## 8. What the paper has to say either way

Whichever way this goes, it is disclosed. The difference is which sentence we write:

- **If amended:** *"A unit inconsistency in the pre-registered deflation basis was found and
  corrected pre-data; the correction raises the bar by 9.53% on the calibration fixture and both
  bases are reported."*
- **If not amended:** *"The pre-registered deflation basis mixes Sharpe estimates measured at three
  rebalance frequencies; we report it as specified alongside a unit-consistent basis, and note that
  the specified basis is the more permissive of the two on our calibration fixture."*

The second sentence is publishable and honest. It is also a sentence a referee is entitled to read
as us keeping the easier bar because it was written down first.

---

## 9. The decision requested

**Amend, or decline and report as specified.** Both are defensible and both are pre-committed to
disclosure. What is not available is silence.

If amended, the builder implements §3 above, re-proves the Gate-A anchor twice, adds the mutation
KAT, and dates the change into `docs/m6_prereg.md` §7 as **v1.6.14** with this document as its
record. Estimated cost: **$0** — it is local and touches no training.

---

## 10. Standing limits on the record

1. **The illustration is a 3-seed toy fixture**, not the M6 cells. The ratios above carry (SR₀
   scales as √var_sr, so the shift is a pure ratio at any scale) but the **sign** is data-dependent
   and could be conservative on the real cells.
2. **The auditor's own 5.8056e-4 figure was NOT reproduced** and is not load-bearing here. The
   finding rests on the span (bit-exact), the mechanism (from code), and outcome-materiality
   (proven by construction). Closest construction agrees to 0.026% at |IR| = 7.56; no repo value
   sits in the interval that would display as 5.8056e-4. Recorded as irreproducible rather than
   adopted.
3. **This is the third reopening of the freeze.** That is a cost in itself, and it belongs in the
   decision alongside the statistics.
