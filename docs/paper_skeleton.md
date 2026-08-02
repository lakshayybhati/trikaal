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

**BINDING (§7 v1.6.17) — say what the v1 microstructure leg ACTUALLY is.** It is TFI plus the
aggTrades trade-flow statistics: **six live dims (7–12)**. Funding and open interest are
specified and wired at dims 13–15 with a tripwire, but are **constant-zero and masked on 100%
of the 304,625,181 bars we publish from** — they were never ingested in v1 (the documented
OI-retention reason). **The input is 16-wide with 13 live dims**, and the paper must not name
funding or OI as carried information. Measured: `runs_manifest/m6_c15b_lake_surface_check.json`.

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

**BINDING ADDITIONS (§7 v1.6.17) — these must reach the paper, not only a receipt.**

1. **The embargo is justified by the SIGNED-return channel only.** `E = H_max + L_corr = 120` bars
   rests on serial correlation dying within 60 bars. Measured on the real lake (all 200 symbols;
   the pinned 40 separately): **signed-return ACF at lag 60 is +0.0072 mean / 0.0137 worst-symbol
   on the pinned 40, and +0.0067 / 0.0191 across all 200** — already ≈0 by lag 5, a ~24× margin.
   **But |return| ACF at lag 60 is 0.232 mean / 0.360 worst and is still 0.15 at lag 240**:
   volatility clustering is long-memory and an embargo sized to it would run to days. We argue
   signed-return autocorrelation is the label-leakage channel for a signed target, and we accept
   that argument — but state the residual in these words: **a leakage channel that isn't
   signed-return autocorrelation wouldn't be caught by it.** The empirical end-to-end alternative
   (headline IR flat as E grows) was costed at 3× the run and not funded; it is asserted from the
   premise, not demonstrated. Receipt: `runs_manifest/m6_c20_embargo_premise.json`.
2. **The headline is ONE CALENDAR YEAR with no cross-year replication.** The primary region is
   **2024-01-01T18:00 → 2025-01-01T00:00 = 365 days (0.9993 years)**: the 4-year window at
   `train_frac` 0.7 leaves ~1.2 years, block 0 is VAL, blocks 1–5 are the headline. The cost-aware
   net IR therefore carries one year's regime, alongside a 40-symbol cross-section. This is a
   design property, not a defect — and it bounds external validity, so it is stated rather than
   left for a referee to derive.
3. **Two of the eight causal-safety surfaces were never exercised on the published lake until
   after it was built.** The ingest sweep ran an 800-bar head slice under a 1440-bar volatility
   warm-up, so `target`/`target_valid` compared equal trivially, on 2 of 200 symbols, unpersisted.
   Closed retrospectively at production parameters on real bars past the warm-up (1,600 anchors,
   12,798 checks, coverage 1.0, both planted leaks caught). Stated because the *original*
   demonstration did not cover what it appeared to. Receipts:
   `m6_c15b_lake_surface_check.json`, `m6_c15b_production_sweep.json`.

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

## The three layers of the floor, and the withdrawal that produced them

Recorded here in full because the superseded figure **was stated to the operator**, so the
correction has to be visible rather than quietly absorbed — the same treatment C1 got.

**Layer 1 — what 3.518 actually is.** Not `(z_.95+z_.80) × SE_boot`. It is an **analytic
iid-normal** MDE, `z_sum · √(2/T_eff) · √(periods_per_year(h))` (`scripts/m6_prereg.py:74`),
reproduced to 3dp (**3.5183**) from the receipt's own `T_eff` = 35,002.3. Inverting the multiplier
therefore recovers the **analytic** SE (1.4149), **not** the paired-bootstrap `SE_boot` that the
v1.5 MDE multiplies. Standing one in for the other is a defensible pre-run assumption but it is
**an estimate of an estimate, not an identity** — so the floor is tagged ESTIMATED in the claims
audit and is replaced by the realized value in the final text.

**Layer 2 — why 4.3475 is not a floor.** The 3.0728 multiplier is the **ν→4 limit**, which obtains
only when the **training term dominates**; the product was taken against an SE that **ignores** the
training term. Those are mutually exclusive states of the same formula. Verified through the repo's
own arithmetic (`paired_bootstrap.py:156-160`): at `se_train = 0` the code takes the
scoring-only branch, ν = B−1 = 9,999, and the multiplier collapses back to z_sum = 2.4865.

**Layer 3 — where the true infimum is.** The MDE is **monotone increasing** in `se_train`, so its
infimum sits at `se_train = 0` and equals the v1.2 value, **3.518**. The figure 4.3475 is reached
only once `se_train ≥ 0.685 × se_boot` — a threshold that is **not knowable before the run**.

> **WITHDRAWAL, ATTRIBUTED.** The "≥ 4.35 annualized IR at S=5" floor originated with the
> **supervisor**, was stated to the operator, and is **withdrawn in full**. The supervisor verified
> the correction independently in `paired_bootstrap.py:156-160` and adopted all three layers. The
> builder's further point — that 3.518 is analytic rather than a bootstrap product — was adopted in
> the same ruling.

**What the abstract sentence deliberately does and does not say.**

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
