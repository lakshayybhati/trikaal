# DRAFT for ruling — the HANDICAP interpretation rule (M1 + decode agreement)

**STATUS: DRAFT. NOT ADOPTED.** Written by the builder, put to the supervisor for ruling. Nothing
in this document binds anything until it is ruled on and dated into `docs/m6_prereg.md` §7.

---

## 0. Why this is written now, before any real cell runs

The question "does ΔIR(4−5) remain the right headline given M1's number" **cannot be answered after
M1's number exists**. Deciding how to read a measurement after seeing it is exactly what a
pre-registration exists to prevent. So the reading rule is fixed first, dated, while the value is
still unknown to everyone.

## 1. What this is NOT

**This is not a specification change and it does not touch the freeze.**

- M1 is **non-gating**. It has no threshold, appears in no clause, and a dedicated KAT asserts that
  two verdicts differing only in the M1 numbers emit an identical word, primary, failing-clause
  list and per-clause `pass`.
- The **verdict** continues to emit SURVIVES / NULL / INCONCLUSIVE **from the five clauses alone**,
  computed exactly as pinned in §3. No band below can change a verdict word.
- What the bands constrain is **only what we are permitted to CLAIM in the paper**. It is a claim
  rule, not a verdict rule.

Stated plainly because the distinction is load-bearing: **the verdict is unaffected; the claim is
bounded.**

## 2. EXISTENCE and MATERIALITY are different questions, judged differently

**Re-specified 2026-08-02 on the supervisor's ruling.** An earlier draft of mine wrote
`H = H₁ + H₂`. **That was wrong and is withdrawn:** my own "different units" objection applies
*between the channels*, not only between H and Δ — a reconstruction-MAE ratio plus a
sign-agreement difference is dimensionless nonsense. The same draft's band C ("handicap comparable
to or larger than the claimed effect") required precisely the reconstruction-to-IR mapping I had
said does not exist, i.e. **it specified a band that could never be entered.**

### EXISTENCE — per channel, on its own statistic, in its own units

- **H₁, the CAPACITY channel** (C-12 M1, `verdict.placebo_capacity_disclosure`): the placebo's
  excess OHLCV reconstruction error. Mechanism: the permutation leaves six dims of independent
  noise (contemporaneous micro↔OHLCV coupling measured **0.2037 → 0.0005**), incompressible and
  weighted 3× by `micro_point_weight`, so bits spent on it are bits taken from OHLCV.
- **H₂, the DECODE channel** (C-1, `verdict.decode_agreement_disclosure`): the cell-4 vs cell-5
  difference in single-bar-vs-in-window decode sign agreement, tested with the SE of the
  **difference** (Welch), degeneracy flag honoured. Mechanism: the decision path decodes ONE bar
  through a decoder trained on FULL WINDOWS, so lower agreement on the placebo arm makes cell 5's
  μ̂ noisier.

A channel **counts only if distinguishable from zero on its own test**. INDETERMINATE contributes
nothing; a degenerate arm contributes nothing and is reported as unmeasurable. **The channels are
never summed or converted.**

### MATERIALITY — one quantity, already in IR units, already computed

**pb(5−2)** — the paired bootstrap of IR(cell 5) − IR(cell 2) — computed at
`verdict.py:743`, flagged at `:766` as `shuffle_harmed = pb52.ci_upper < 0.0`, and reported at
`:787-790` and `:997`.

The reasoning needs no mapping and no invented constant: **removing information should make the
placebo no worse than having no microstructure at all.** Cell 2 *is* "no microstructure". So cell 5
scoring **below** cell 2 is direct evidence, in the right units, that the shuffle harmed it beyond
information removal — whatever the mechanism.

**Band C's condition is therefore identical to the existing `shuffle_harmed` flag**, which is
already computed, already reported, and already documented as "disclosure, never gating". No new
instrument, no new threshold.

## 3. The three bands

### Band A — no channel distinguishable AND pb(5−2) CI contains zero
No evidence of handicap. **ΔIR(4−5) stands as the headline**; the disclosure is reported as a null
result. One sentence in the results.

### Band B — one or both channels distinguishable, pb(5−2) CI containing zero or positive
**ΔIR(4−5) stands as the headline**, with **both channel magnitudes stated IN THE ABSTRACT in
their own units**, and the claim explicitly bounded as *inflated in these named channels by an
amount **not quantifiable in IR units***.

That last phrase is deliberate. We may not write "inflated by at most H" — we cannot compute that
bound, and implying one we do not have would be worse than stating the limit plainly.

### Band C — pb(5−2) upper CI BELOW ZERO
The placebo scored **worse than having no microstructure at all**.

**WE DO NOT CLAIM THE MICROSTRUCTURE INFORMATION RESULT.** The finding becomes **C-12's mechanism
as the substantive contribution** — a block permutation preserving marginals destroys
contemporaneous cross-channel dependence as well as temporal alignment (0.2037 → 0.0005), and a
reconstruction-trained tokenizer under a weighted per-bar objective pays a measurable capacity
price for the resulting independent noise — with **ΔIR(4−2) reported as a capacity-mismatched
secondary** (cell 2 is information-matched but carries 7 dims, not 16, so it never pays the
noise-encoding cost either; it is not a clean substitute and must not be presented as one).

**BAND C IS NOT HYPOTHETICAL.** The canary fixture already measured **pb(5−2) = −2.59**, so this
condition has fired once on synthetic data. A rule that has already triggered in rehearsal is not
a formality.

## 4. Boundaries

**There are none to choose.** Band membership is determined by **significance tests on quantities
that already exist**, not by constants anyone selects. The only judgement left is the CI
convention, and it **inherits the pre-registered one** rather than getting a new one.

Two methodological notes carried forward:

- **"Not distinguishable from zero" is NOT "zero".** Every C-1 comparison at n=3 returned
  INDETERMINATE — including the one briefly called "symmetric" — because failing to reject is not
  evidence of equality. Band A is read as *"no channel was measurably distinguishable at the run's
  n"*, not as "no handicap exists".
- **Comparing two arms requires the SE of their DIFFERENCE**, not one arm's spread. The spread
  form is the power guard's REFUSAL rule and does not answer an inference question; applied to
  C-1 it read 1.10 → "asymmetric" where the correct statistic read t = 1.82 vs crit 2.62 →
  indeterminate.

## 5. Limits that remain on the record

0. **BOTH CHANNELS ARE MEASURED ON THE REAL CELLS, not on a proxy.** The toy attempt at H₂ came
   back INDETERMINATE at n=3 (t = 1.821, Welch df 2.385, crit 2.6226) and resolving it there would
   need n ≈ 13 per arm on briefly-trained tokenizers that may not transfer. Both disclosures are
   now REQUIRED, NON-GATING per-(cell, seed) fields on the run's own artifacts, with 5 seeds
   already paid for.
1. **The channels are in different units from each other AND from Δ.** This is why they are never
   summed and why materiality is judged on pb(5−2) instead. The missing reconstruction-to-IR
   mapping **still does not exist** — the re-specification routes around it rather than pretending
   otherwise.
2. **M1 measures the handicap's existence and size in reconstruction space, not its transmission
   to IR.** A large H with no IR consequence is possible in principle. The rule above treats H as
   an upper bound on the inflation, which is the conservative direction, and that conservatism
   should be explicit in whatever is adopted.

## 6. Requested ruling

- Adopt / amend / reject the three bands.
- Fix the A/B and B/C boundaries, or rule that the ordinal reading in §5.1 is sufficient.
- Confirm the §1 framing — non-gating, claim-rule-only, freeze untouched — is stated as intended.
- Confirm that "total handicap" means the two channels are considered **jointly** (they compound
  and neither can deflate Δ), rather than each judged against its own band.
