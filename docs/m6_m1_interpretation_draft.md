# DRAFT for ruling — the M1 interpretation rule

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

## 2. The quantity

Let

- **Δ** = the claimed ΔIR(4−5), the pre-registered headline;
- **H** = the M1 capacity handicap, read from
  `verdict.placebo_capacity_disclosure` — specifically the placebo's **excess OHLCV
  reconstruction error**, `ohlcv_recon_mae_cell5 − ohlcv_recon_mae_cell4`, with the ratio reported
  alongside.

Every band below is expressed **relative to Δ**, never against an absolute. There is no invented
constant anywhere in this rule, by design: an asymmetry that is small against the effect being
claimed is immaterial, and one comparable to it is not, and that comparison needs no convention.

**The uncertainty in H is its own across-seed spread**, the same self-scaling shape the power guard
uses — it refuses to report a ΔIR smaller than its own inputs' seed-to-seed wobble. H is
"measurably distinguishable from zero" when its across-seed mean exceeds its across-seed spread;
below that it is measurement noise.

## 3. The three bands

### Band A — H not measurably distinguishable from zero
*(the across-seed mean of H is within its own across-seed spread)*

**ΔIR(4−5) stands as the headline, unqualified.** The disclosure is reported as a null result: the
placebo's OHLCV reconstruction is indistinguishable from the treatment's, so no capacity was
measurably diverted and the contrast carries information alone. One sentence in the results; no
change to the abstract's claim.

### Band B — H measurable but small against Δ

**ΔIR(4−5) stands as the headline, with the magnitude stated IN THE ABSTRACT** and the claim
explicitly bounded: *"this estimate is inflated by at most H, measured."* Not a footnote, not an
appendix — the abstract carries both the effect and its bound, because a reader who takes the
headline without the bound has taken a number we know to be an overestimate.

### Band C — H comparable to or larger than Δ

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

**Deliberately left to the ruling.** I have not proposed numeric cut-points between A/B/C, because
proposing them would re-introduce exactly the invented constant the ruling forbade. The natural
self-scaling forms are: A when `mean(H) ≤ sd(H)`; C when `H` is of the same order as `Δ`; B
otherwise. Whether "same order" means `H ≥ Δ`, `H ≥ Δ/2`, or something else is a judgement about
how much inflation makes a claim untellable, and that is the supervisor's or Lakshay's, not the
builder's.

## 5. Two things I want on the record before this is ruled on

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
