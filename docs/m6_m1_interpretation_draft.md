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

## 2. The quantities — TWO CHANNELS, NOT ONE

**Amended 2026-08-02 on the supervisor's ruling: band membership is a function of the TOTAL
measured handicap, because the two channels compound.**

Let

- **Δ** = the claimed ΔIR(4−5), the pre-registered headline;
- **H₁ — the CAPACITY channel** (C-12 M1), from `verdict.placebo_capacity_disclosure`: the
  placebo's excess OHLCV reconstruction error, `ohlcv_recon_mae_cell5 − ohlcv_recon_mae_cell4`,
  with the ratio alongside. Mechanism: the permutation leaves the micro dims as six dims of
  independent noise (contemporaneous micro↔OHLCV coupling measured **0.2037 → 0.0005**),
  incompressible and weighted 3× by `micro_point_weight`, so bits spent on it are bits taken
  from OHLCV.
- **H₂ — the DECODE channel** (C-1), from `verdict.decode_agreement_disclosure`: the cell-4 vs
  cell-5 difference in single-bar-vs-in-window decode sign agreement, reported with the SE of the
  **difference**, Welch df, and a degeneracy flag. Mechanism: the decision path decodes ONE bar
  through a decoder trained on FULL WINDOWS; lower agreement on the placebo arm makes cell 5's
  μ̂ noisier.
- **H = the total handicap.** Both channels degrade the PLACEBO arm more than the treatment arm,
  so **both inflate ΔIR(4−5) in the same direction, and they compound.** Neither can deflate it.

**A channel counts toward H only if it is measurably distinguishable from zero on its own
statistic** — H₁ against its across-seed spread, H₂ against the SE of its difference (Welch, with
the degeneracy flag honoured). A channel whose test returns INDETERMINATE contributes nothing;
a channel whose arm is degenerate contributes nothing and is reported as unmeasurable.

Every band below is expressed **relative to Δ**, never against an absolute. There is no invented
constant anywhere in this rule, by design: an asymmetry that is small against the effect being
claimed is immaterial, and one comparable to it is not, and that comparison needs no convention.

**The uncertainty in H is its own across-seed spread**, the same self-scaling shape the power guard
uses — it refuses to report a ΔIR smaller than its own inputs' seed-to-seed wobble. H is
"measurably distinguishable from zero" when its across-seed mean exceeds its across-seed spread;
below that it is measurement noise.

## 3. The three bands

### Band A — NEITHER channel measurably distinguishable from zero

**ΔIR(4−5) stands as the headline, unqualified.** The disclosure is reported as a null result: the
placebo's OHLCV reconstruction is indistinguishable from the treatment's, so no capacity was
measurably diverted and the contrast carries information alone. One sentence in the results; no
change to the abstract's claim.

### Band B — at least one channel measurable, TOTAL small against Δ

**ΔIR(4−5) stands as the headline, with the magnitude stated IN THE ABSTRACT** and the claim
explicitly bounded: *"this estimate is inflated by at most H, measured."* Not a footnote, not an
appendix — the abstract carries both the effect and its bound, because a reader who takes the
headline without the bound has taken a number we know to be an overestimate.

### Band C — TOTAL comparable to or larger than Δ

**WE DO NOT CLAIM THE MICROSTRUCTURE INFORMATION RESULT.**

The finding becomes: *the design cannot separate information from capacity at this effect size.*
Reported with ΔIR(4−2) as a **capacity-mismatched secondary** (Cell 2 is information-matched but
carries 7 dims, not 16 — it never pays the noise-encoding cost either, so it is not a clean
substitute and must not be presented as one), and **the C-12 mechanism as the substantive
contribution**: that a block permutation preserving marginals destroys contemporaneous
cross-channel dependence as well as temporal alignment, measured 0.2037 → 0.0005, and that a
reconstruction-trained tokenizer under a weighted per-bar objective pays a measurable capacity
price for encoding the resulting independent noise.

**This is a claim rule, not a verdict rule.** If the clauses emit SURVIVES, the manifest still says
SURVIVES; band C constrains what the paper is permitted to conclude from it, and the divergence
between the verdict word and the permitted claim is itself reported prominently.

## 4. Boundaries between the bands

**Deliberately left to the ruling**, and the supervisor has said they will be set **once both
instruments exist and BEFORE either produces a number**. Proposing cut-points here would
re-introduce exactly the invented constant the ruling forbade.

One methodological note that the C-1 episode makes concrete: **"not distinguishable from zero" is
NOT the same as "zero".** Every C-1 comparison at n=3 returned INDETERMINATE — including the one
I had briefly called "symmetric" — because failing to reject is not evidence of equality. Band A
should therefore be read as *"no channel was measurably distinguishable at the run's n"*, and
whether that is strong enough to license an unqualified headline is part of what the ruling fixes.

## 5. Two things I want on the record before this is ruled on

0. **BOTH CHANNELS ARE MEASURED ON THE REAL CELLS, not on a proxy.** The toy attempt at H₂ came
   back INDETERMINATE at n=3 (t = 1.821, Welch df 2.385, crit 2.6226) and resolving it there would
   need n ≈ 13 per arm on briefly-trained tokenizers that may not transfer. Both disclosures are
   now REQUIRED, NON-GATING per-(cell, seed) fields on the run's own artifacts, with 5 seeds
   already paid for.
1. **H and Δ are in different units.** H is a reconstruction MAE in feature space; Δ is an
   annualized information ratio. "Comparable to" therefore cannot be a direct numeric comparison
   and I have not written one. The honest available comparison is **ordinal and relative**: does
   the measured handicap plausibly account for a material share of the measured effect? If the
   ruling wants a *numeric* band, the missing piece is a mapping from OHLCV-reconstruction
   degradation to IR degradation, which **does not exist and would have to be measured** — that is
   a separate experiment, not a threshold choice.
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
