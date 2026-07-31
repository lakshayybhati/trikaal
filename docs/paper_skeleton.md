# Paper skeleton — mechanism-led, framing PROVISIONAL

**Status:** drafted 2026-07-31, pre-run (prereg §7 v1.6 items 1–2).

> **NO TITLE AND NO ABSTRACT ARE LOCKED.** The real-data legibility gate decides which paper this
> is. Everything below is structure and pre-committed content; the framing stays provisional until
> the gate has spoken. Candidate titles are deliberately absent — a title written now would be a
> prediction of the result.

**Spine: capacity eviction.** The ablation is *supporting evidence*, not the thesis. This ordering
is chosen pre-data precisely because it is the one ordering that is honest under **every** outcome
in the taxonomy — SURVIVES, NULL and INCONCLUSIVE each leave the mechanism section standing.

---

## §1 Introduction

Reconstruction-trained tokenizers allocate code capacity by **variance and covariance, never by
downstream value**. A low-variance, weakly-price-correlated channel — which is exactly what
microstructure is — is **silently evicted**, and no amount of downstream modelling recovers what
the tokenizer discarded. That is the claim the paper is organised around.

## §2 Background

Kronos as cited prior art (two-stage tokenizer → AR). BSQ vs FSQ. What "microstructure" means here
and — **binding** — why it is **TFI** (signed executed-volume imbalance from free aggTrades) and
never OFI: true order-flow imbalance needs orderbook depth, which is explicit v2 work.

## §3 The object under study

**A tokenizer study, not a foundation model.** The 21,301,248-parameter decoder-only backbone
(Kronos_small class: 8L / d512 / ff1024 / 8h) is the **measurement vehicle** — held fixed and
matched across every arm so the only varied factor is quantizer × input arm. Bits-per-token parity
(FSQ 20.058 / BSQ 20.000) is the controlled variable, not vocabulary size.

## §4 The eviction mechanism *(the spine)*

Statement, derivation, and the fixture evidence: per-dimension point-decoder correlations —
return **0.98**, correlated fillers **0.82–0.92**, the independent state dimension
**0.001–0.014**, stable across 3 seeds. Reconstruction buys variance and covariance; it never buys
independence.

> **BINDING FRAMING COMMITMENT.** This is measured on a **SYNTHETIC fixture**, not on real
> microstructure. It is **not** elevated to headline anywhere. It stands independent of the M6
> outcome: both "micro survives" and "micro nulls after costs" are complete papers containing it.

## §5 Method

Per-bar feature vector and causal rules; FSQ tokenizer + the pointwise-fine / per-bar-bottleneck
interface; the AR backbone and MTP; the standing micro-legibility gate.

## §6 Experimental design

The five cells, purged walk-forward + embargo, cost-aware net IR at 0.30%, the pinned h=15 money
surface, S=5 seeds as **replicates**, and the pre-registration itself — including the amendment log
with its withdrawals.

## §7 Power and detectability *(pre-registered, pre-data — see below)*

## §8 Results

Structured by the taxonomy. **All three outcomes are pre-committed and publishable**; R3
(INCONCLUSIVE) is written out in full in the prereg and is not a failure to report.

## §9 The legibility gate on real data

Pre-written adjudication. **A firing is the modal prediction of our own mechanism** — real TFI has
exactly the eviction profile, and the λ fix was validated on a synthetic plant. If it fires, that
*is* the real-data confirmation of eviction: simultaneously the ablation's blocker and the
mechanism's strongest evidence.

## §10 Limitations and threats to validity

Determinism posture (below); training-variance / basin-hopping; single asset class; the
fixture-vs-real gap in §4; the cost basis (§ appendix entry 0).

## §11 Reproducibility

Content-hashed lake, the Gate-A anchor, the pinned lockfile — **with the determinism scope stated
exactly** (see below).

### Appendices
A. Pre-registration + full amendment log · B. **Claims audit** (`docs/m6_claims_audit_appendix.md`)
· C. Dual-specification report (v1.2 vs v1.5) · D. Guard receipts and per-seed distributions

---

# §7 — THE POWER STATEMENT (drafted 2026-07-31, PRE-DATA)

Every number below was verified against the live pinned surface
(`runs_manifest/m6_mde_floor_verification.json`), not carried over from any prior statement.

> **Draft, for the abstract.** *We pre-registered the detectability limit of this design before
> observing any outcome. The test statistic is a paired difference in cost-aware annualized
> information ratio, and its minimum detectable effect combines two independent sources of noise:
> scoring noise, estimated by a paired moving-block bootstrap over time, and training noise,
> estimated across S = 5 seed replicates, combined as* `SE_total = √(SE_boot² + SE_train²)` *with
> Student-t quantiles at Welch–Satterthwaite effective degrees of freedom. Under the analytic
> scoring-noise estimate available before the run, the minimum detectable effect is* **at least
> 3.518 annualized IR** *at the pinned 15-minute horizon, and strictly greater once training
> variance is non-zero — by an amount the run itself supplies. We therefore state the floor and the
> structure, not a point estimate: an effect smaller than this design's own training variance is
> reported as unresolvable at this scale, which is a measurement of the instrument and not a
> failure of the experiment.*

**What that sentence deliberately does and does not say.**

- It states a **floor** (**≥ 3.518**) and a **structure** (two-term, t-quantiles, Welch–
  Satterthwaite, training term supplied by the run). It states **no point number** for the
  realized MDE, because none is knowable pre-data.
- **It does not quote 4.35.** That figure combines the ν→4 multiplier (which requires the training
  term to dominate) with an SE that ignores the training term — two mutually exclusive limits. The
  MDE is monotone increasing in `se_train` with its infimum at `se_train = 0`, where the multiplier
  returns to 2.4865 and the MDE returns to the v1.2 value. 4.3475 is reached only once
  `se_train ≥ 0.685 × se_boot`, which is not knowable before the run.
- The floor itself rests on the **analytic** scoring SE (1.4149) standing in for the realized
  bootstrap `SE_boot`. That substitution is labelled **ESTIMATED** in the claims audit; the
  realized value replaces it in the final text.
- **INCONCLUSIVE reads as instrument-reporting, not failure.** The last clause is load-bearing and
  is not softened later: reporting that an effect is smaller than the design's own training
  variance is a finding about the detectability limit of a 21.3M-parameter two-stage design at this
  data scale.

---

# Two things that must appear, and must not be separated from their qualifiers

**1. The determinism posture (§10/§11).** The reproducibility deliverable is true for the data
pipeline, frozen statistics, prediction replay and the Gate-A anchor. It is **not** currently true
for GPU training: 49 of 49 determinism records assert `bit_exact_claim` while
`deterministic_algorithms=False`, because `attention_mode.py` encodes the premise that
deterministic attention is *sufficient* when it is only *necessary*. Scope this precisely; do not
let the intact clause carry the false one.

**2. The cost basis (§10 + appendix).** ESTIMATED, not measured — see the claims audit, entry 0.

**Board discipline, verbatim:** the pre-flight board stands at 8 of 8 with the canary gate closed,
**and 8/8 must never be quoted apart from B1** (the determinism posture), which postdates the board
and is blocking. Anyone quoting one without the other is quoting it wrong.
