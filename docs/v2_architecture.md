# v2 architecture — what the v1 failures say the next model should be

**Status: DESIGN NOTES. Nothing here touches v1.** No pin moves, no gate changes, no
amendment. v1 is frozen at the §7 v1.5 item-E disposition: the legibility gate fired on real
data, the primary is the mechanism finding, and the ablation does not run as designed.

**Purpose.** v1 produced an unusually well-instrumented set of failures. This document turns
them into a specification for a successor, so the diagnoses survive the paper and are not
re-derived from scratch later.

**Epistemic discipline, applied to this document as to everything else.** Every diagnosis
below is traced to a receipt. Every proposed fix is marked `[INFERRED]` and carries **zero**
supporting evidence that it works — the evidence is for the *problem*, never for the remedy.
Two of the diagnoses have been measured three independent times; one has been measured once.
The distinction is kept explicit because collapsing it is how a design note becomes folklore.

---

## The four problems are separate and get blurred constantly

| # | problem | layer | evidence | fixes v1? |
|---|---|---|---|---|
| 1 | the tokenizer evicts independent information | representation | 3 independent measurements | no |
| 2 | the encoder smears per-bar state across the window | representation | 1 measurement, canonical scale | partially |
| 3 | transaction costs are 14–129× the gross edge | economics | 3 seeds, robust across the cost band | no |
| 4 | the headline metric measures turnover, not skill | evaluation | 3 seeds, mechanism fully closed | no |

**Problems 1 and 2 decide whether the science can be done. Problem 3 decides whether it is
ever profitable. Problem 4 decides whether either is measurable.** They are routinely
discussed as one thing. They are not one thing, and a fix to any of them leaves the others
untouched.

---

## Problem 1 — the tokenizer evicts information that is independent of what it reconstructs

### Measured, three times, on three different setups

**(a) Synthetic fixture, 3 seeds** (`runs_manifest/m6_lambda_search_receipt.json`). A
reconstruction-trained tokenizer given a return dim, eleven correlated filler dims and one
independent state dim allocated capacity as: return **0.98**, correlated fillers
**0.82–0.92**, the independent dim **0.001–0.014**. Reproduced across seeds with corrs
"≈identical", i.e. deterministic rather than noisy.

**(b) Canary, canonical scale** (prereg §7 v1.4; paper §3). A ~1.15-nat signal planted in
*feature* space produced **zero nats** of extraction by the AR, while per-bar id visibility of
the planted dim measured **0.5135 — chance**. The identical rule planted directly in *token*
space was learned to **Spearman 0.9999, 94% of the planted nats**. The tokenizer was the
bottleneck, not the backbone.

**(c) Real data, both arms, 40 symbols, n=150,000/dim**
(`runs_manifest/m6_micro_legibility_stop.json`):

| dim | feature | cell4 (real micro) | cell5 (shuffled) | base rate |
|---|---|---|---|---|
| 7 | **TFI** | 0.8223 | 0.6135 | 0.5131 |
| 8 | **signed_count_imbalance** | 0.7528 | 0.5896 | 0.5335 |
| 9 | trade_count | 0.8975 | 0.8916 | 0.5275 |
| 10 | mean_trade_size | 0.9076 | 0.8525 | 0.5462 |
| 11 | trade_size_dispersion | 0.8962 | 0.8691 | 0.5927 |
| 12 | large_trade_share | 0.9320 | 0.8685 | 0.8577 |

**97.3% of cell 4's shortfall and 83.5% of cell 5's sit in dims 7 and 8 — the two *signed*
channels.** The four *magnitude* dims essentially pass. Magnitude features correlate with
volume and volatility, which OHLCV already carries, so they cost almost nothing to keep.
The signed imbalances are the ones carrying information OHLCV does not, and they are the ones
evicted.

### Root cause — and it is not a bug

The objective is reconstruction MSE. **MSE allocates capacity by variance explained.** A
dimension independent of the others contributes its own variance but no covariance, so per bit
it is the worst available investment. The encoder is doing exactly what it was asked to do.

**We required a compressor to preserve something a compressor has no reason to preserve, and
then weighted the loss and hoped.**

### Candidate fixes `[INFERRED — none tested]`

**(a) Weight the reconstruction loss (λ). CLOSED BY EVIDENCE — do not retry.** λ was searched;
the landscape is non-monotone (seeded: λ=2 → 0.8517, λ=3 → 0.9067, λ=4 → 0.8752, λ=6 → 0.8999)
with a ceiling near 0.92–0.93 **on a single dim**, and the receipt's own summary reads *"no
searched (lambda, beta) clears 0.9 on all 3 seeds."* Weighting fights the objective instead of
changing it.

**(b) ★ Put the gated quantity into the loss.** v1 *gates* on per-bar legibility and *trains*
on reconstruction, hoping one implies the other. Add an auxiliary head predicting the micro
dims from the token, with cross-entropy, as a training term. **Train the metric; stop proxying
it.** Highest expected value, cheapest to test, and not architectural — it is a loss term.

**(c) Reserve capacity structurally.** One token currently encodes all 16 dims jointly and they
compete for a fixed budget. Give microstructure its own subtoken with reserved bits. The
allocation stops being learned and OHLCV *cannot* crowd it out. Costs vocabulary and a
re-derived bits-per-token parity argument.

**(d) ★ Do not quantize microstructure at all.** The bottleneck exists so the AR has a discrete
vocabulary; nothing requires *every* input to pass through it. Feed micro as continuous
side-information alongside the discrete token. **Probably the correct answer** — it removes the
competition rather than managing it.

---

## Problem 2 — the encoder smears per-bar state across window ids

### Measured once, at canonical scale

Canary: per-bar id visibility **0.5135** (chance) while window-level reconstruction was
non-degenerate. A causal *windowed* encoder distributes bar *t*'s information across tokens
*t…T*, so no single token carries it.

v1's response was the `fine_pointwise` per-bar bottleneck leg (§7 v1.4). It helped; the real
data shows it did not solve the problem.

### Root cause `[INFERRED]`

**The token is being asked to do two jobs at once** — represent this bar, and carry window
context. The AR already models the sequence. Conflating the two is the original design error,
and the pointwise leg patched a symptom of it.

### Candidate fix `[INFERRED]`

**Encode bar *t* from bar *t* alone**, plus causal normalization statistics. Per-bar legibility
then becomes structural rather than something to be measured and hoped for. The cost is that
the token carries no cross-bar context — which is the AR's job, not the tokenizer's.

### ★ THIS QUESTION IS NOW ANSWERED — $0, and it answers against the gate being over-strict

**Asked:** does a *windowed* read of the token stream recover dims 7–8, or are they gone
entirely? If a 512-bar probe also fails, the eviction is complete and per-bar encoding is
**necessary**. If it succeeds, the gate is measuring a per-bar proxy for something the AR
could still reach, and the gate itself is **over-strict**.

**Run 2026-08-13** (`runs_manifest/m6_window_context_probe.json`), same estimator, same split,
same features as the gate, with 19 log-spaced lags 0–511 added (53 features/bar × 19 = 1,007):

| dim | arm | per-bar | windowed | Δ |
|---|---|---|---|---|
| 7 (TFI) | cell 4 | 0.8399 | 0.8385 | **−0.0014** |
| 8 (signed count) | cell 4 | 0.7516 | 0.7562 | **+0.0046** |
| 7 (TFI) | cell 5 | 0.6432 | 0.6403 | **−0.0028** |
| 8 (signed count) | cell 5 | 0.6012 | 0.5955 | **−0.0057** |

**A 512-bar read recovers nothing beyond a single bar.** The largest gain anywhere in the
twelve cells is dim 12 at +0.0297, and the two signed channels move by less than half a
percentage point in *either* direction. **So this is eviction, not smearing** — a different
failure from the canary's, and per-bar encoding would not fix it. **Only the allocation would.**

**Two asymmetries that must travel with this result.** (i) The basis is a *subset* — 1,007 of
the 27,136 features an unrestricted 512-lag read would use — and a subset can only *under*-state
recoverability, so **finding signal would have been conclusive; finding none is suggestive**.
(ii) 2 of the 12 cells (`cell5` dims 9 and 11) returned `READABLE=False` — the control fired and
the probe refused to score them, per the control-arm rule.

---

## Problem 3 — costs are 14–129× the gross edge. No tokenizer fixes this.

### Measured on three seeds, robust across the pre-registered cost band

Gross performance is exactly recoverable from banked artifacts (`eval/diagnostics.py ::
gross_series`), because the headline pass applies a constant 0.30% to active periods:

| seed | gross edge/trade | **IR gross** | IR @0.10% | @0.20% | @0.30% | cost ÷ edge |
|---|---|---|---|---|---|---|
| 0 | +0.0054% | **+1.2665** | −22.20 | −45.12 | −67.02 | **55.8×** |
| 2 | +0.0208% | **+2.1715** | −8.24 | −18.51 | −28.47 | **14.4×** |
| 4 | +0.0023% | **+1.1443** | −48.10 | −97.32 | −146.49 | **129.1×** |

Gross IR of 1.1–2.2 is meaningful, and **the cost model destroys it at every cost in the band we
pre-registered** — including the cheap end.

**★ BUT "REAL PREDICTIVE SKILL" IS A NARROWER CLAIM THAN IT LOOKS, AND WAS TESTED 2026-08-14**
(`runs_manifest/m6_skill_vs_bias.json`). The three seeds carry structurally different directional
biases — seed 0 SHORT 67.7%, seed 2 SHORT 65.5%, **seed 4 LONG 92.7%** — so a positive gross IR
could be bias × sample drift rather than timing. Against a constant-direction benchmark at each
seed's own modal sign, same periods, same pinned bootstrap:

| seed | bias | (a) model | (b) bias-only | Δ = a−b | CI95 lower |
|---|---|---|---|---|---|
| 0 | SHORT | +1.2665 | **−0.7607** | +2.0271 | −0.5530 |
| 2 | SHORT | +2.1715 | **−1.2640** | +3.4355 | **+1.5574** |
| 4 | **LONG** | +1.1443 | **+0.4282** | +0.7161 | −0.2410 |

The sample drifted **up** (+7.900e-06 per period; always-long IR **+0.3704**). **So the concern is
INVERTED for seeds 0 and 2** — they are short-biased in a rising market, their bias alone *loses*
money, and their gross IR is earned *against* their own directional lean. Drift capture cannot
explain it. **The concern is LIVE for seed 4**: holding long reproduces **37.4%** of its gross IR
and the remainder's interval **includes zero**.

**★ SEED 4 RE-RUN ON THE EXACT BENCHMARK 2026-08-14** (`m6_skill_vs_bias_exact.json`, GPU, $4.48).
Averaging over the names seed 4 *actually traded* rather than the whole cross-section **lowered**
the benchmark to **+0.1474** — it traded lower-drift names than average. Bias-only share falls
from 37.4% to **12.9%**, Δ rises to **+0.9972** with a 95% lower bound of **+0.0454**.

**So the claim is: bias × drift does not explain any of the three seeds, and all three show a
positive timing contribution — but seed 4's is MARGINAL and that must travel with it.** At
`se_boot 0.5876` the point estimate is **t = 1.6971**. The pre-registered bound is **one-sided at
α = 0.05** (`paired_bootstrap.py:36`), and at that level the percentile bound (+0.0454) and its
normal counterpart (+0.0307) **agree — both clear.** A *two-sided* 95% test, which is stricter
than the one pre-registered, would not (lower bound −0.1545; two-sided p = 0.0897). **Both numbers
should be visible: the effect is real on the instrument declared in advance, and it is not
comfortably large.** Seeds 0 and 2 remain the strong half — their bias alone *loses* money.

**Caveat still live for seeds 0 and 2 only:** their benchmark averages over every symbol with a
valid decision rather than the names each traded; those positions are not persisted and the
direction of the difference is unknown. It **drops for seed 4**, which was re-run exactly.

**Free by-product worth more than the test itself:** seed 4's gross series was reproduced on a
**different torch (2.5.1+cu124 vs 2.12.1+cu130), a different physical 4090, weeks later** — active
set identical at 34,989, IR within 3.5e-04, correlation 0.99999986. That is independent evidence
for the reproducibility deliverable, obtained as a control arm rather than bought.

### This is economics, not architecture `[INFERRED]`

- **Longer horizons.** Edge accumulates with horizon; cost is per-trade. h=15 pays 0.30% for a
  fifteen-minute move. **The largest single lever and nearly free to test on banked artifacts.**
- **Realistic costs.** 0.30% is conservative. Binance perp taker is ~0.04–0.05% per side;
  maker is lower. ~4× on the ratio.
- **Cross-sectional rather than directional.** Similar costs, often stronger and
  market-neutral signal.

**Kept separate deliberately:** problems 1 and 2 make the *science* work. They do not make it
profitable. Blurring the two is how one builds a beautiful model that loses money.

### ★ MEASURED 2026-08-13, $0 — HOW FAR SHORT, EXACTLY

`runs_manifest/m6_horizon_break_even.json`. With positions held fixed, the break-even cost is an
identity: **`c_break` = mean gross return per ACTIVE period.** So the question "what cost could
this model survive" has a single answer per unit, and it is directly comparable to a real fee.

| seed | `c_break` | 95% CI | t | vs ~0.10% round trip |
|---|---|---|---|---|
| 0 | **0.00538%** | [−0.00294, +0.01370]% | 1.27 | **0.054×** |
| 2 | **0.02082%** | [+0.00204, +0.03961]% | 2.17 | **0.208×** |
| 4 | **0.00232%** | [−0.00166, +0.00630]% | 1.14 | **0.023×** |

**Break-even is 4.8× to 43× below realistic execution cost, and the gap survives sampling error** —
even the best seed's 95% upper bound (0.0396%) is **2.5× short**. Two of three intervals contain
zero. Binance perp taker (~0.05%/side) is a *lower* bound on true cost, excluding spread, impact,
funding and slippage, so the real gap is wider.

**The horizon-independent form, and the reason this is the section that matters.** Since
`c_break = ḡ_A`, clearing a 0.10% round trip requires **0.10% of gross edge per active period at
every horizon**. The bar does not move with *h*; only how much edge a period can contain does. A
naive linear extrapolation puts the required horizon at **72–646 minutes** — an order of magnitude
wide, and explicitly **refuted 3/3 as a bound** (sampling error crosses it on all three seeds; it
overshoots its own model by 5.6–7.0%; entry drag makes edge/*h* increasing; θ is horizon-invariant
while μ̂ is not). It is a labelled point extrapolation and nothing more.

**★ THE CONSEQUENCE FOR THIS DOCUMENT'S OWN THESIS.** Problems 1 and 2 are worth fixing and the
evidence for them is strong. **But a tokenizer that perfectly preserved microstructure would still
have to multiply gross edge per active period by 5–43× to break even at realistic fees, and
nothing in problems 1 or 2 delivers a multiple of that size.** So: **v2-as-a-better-tokenizer is a
scientific project, not a profitable one.** Profitability requires moving on the horizon axis, the
execution-cost axis, or the cross-sectional axis — problem 3's levers, not problem 1's. Any v2
that fixes the representation and leaves the economics alone will produce a measurably better
model that still loses money, which is the exact failure this document was written to name.

**Two limits, stated in both directions.** Everything above is **cell 1 only** (BSQ + OHLCV-only,
one arm) — finding a clearing horizon would be *suggestive*, and finding none is *also only
suggestive*. Micro-arm artifacts exist but are **structurally unquotable**: fed to the production
`load_cell_evals`, every one is refused (schema v1≠v2, missing index, incomplete matrix), so the
limit is enforced by the loader rather than by discipline. And *h*=5/60 cannot be measured at $0 —
the val passes banked four IR **scalars** per horizon, not a series, and a scalar IR at one cost
cannot be re-netted. The curve has **one measured point and two bounds** (`c_break < 0.30%`).

**Found while measuring it, and it is a finding about our own design:** the pre-registered
instrument `metrics.break_even_cost` sweeps `FLAT_COSTS = (0.0010, 0.0020, 0.0030)` and returns
**−inf** when net IR is negative across the grid — which it is, on all three units. **Its floor is
0.10%, exactly the number the interesting comparison is against.** The pre-registration therefore
**cannot answer "does break-even clear realistic fees" by construction**, and the finite values
above come from bisecting below that floor. Both are shipped per unit. Separately,
`CellScore.break_even` is computed at eval time and **never persisted** by
`write_cell_eval_artifact` — a pre-registered headline component, paid for in GPU time and
discarded.

---

## Problem 4 — the headline metric measures turnover, and the mechanism is fully closed

### The IRs are cost drag, not skill

A zero-edge model paying 0.30% on a fraction *a* of periods yields
`IR ≈ −√(a/(1−a))·√(525600/15)`. Against measured:

| seed | activity | IR measured | IR from drag alone | error |
|---|---|---|---|---|
| 0 | 0.0926 | −67.02 | −59.81 | −12.0% |
| 2 | 0.0193 | −28.47 | −26.24 | −8.5% |
| 4 | 0.4702 | −146.49 | −176.35 | +16.9% |

**A model with literally no skill reproduces all three headline numbers within 8–17%, across a
24× spread in activity.**

### And turnover varies because μ̂'s scale is uncalibrated and the threshold is absolute

| seed | μ̂ std | gross/trade | **over-dispersion** | threshold in μ̂ sd | predicted activity | observed |
|---|---|---|---|---|---|---|
| 0 | 5.376e-3 | 5.379e-5 | **99.9×** | 1.674 | 0.0941 | 0.0926 |
| 2 | 5.331e-3 | 2.082e-4 | **25.6×** | 1.688 | 0.0914 | 0.0193 |
| 4 | 1.037e-2 | 2.324e-5 | **446.4×** | 0.868 | 0.3856 | 0.4702 |

**The forecast distribution is 25–446× wider than the returns it forecasts.** The execution
filter trades when |μ̂| > κ·c — an *absolute* threshold — so where it lands depends entirely on
μ̂'s uncalibrated, seed-varying scale. Seed 4's μ̂ is 1.93× wider than seed 0's, the same
absolute threshold sits at half the z-score, and it trades 5× more. **That is the entire 24×
activity spread and, through the drag identity, the entire 118-point IR range.**

Confirming: at seed 0 the threshold sits at 1.674 sd and a normal predicts 9.41% activity
against 9.26% observed — **0.15 percentage points.**

### The instability is in the execution layer, not the model

**sd(IR net) = 60.18. sd(IR gross) = 0.561. A 107× reduction.** The three seeds' *gross* IRs
are statistically indistinguishable — their bootstrap SEs (0.958–1.089) exceed their spread.

### Candidate fixes `[INFERRED]`

- **Calibrate μ̂** against realized returns before any threshold is applied. A forecast 100×
  over-dispersed is not selecting; it is trading noise its own scale mislabelled as signal.
- **Make the threshold relative** — trade the top *q*% by |μ̂| rank, or standardize before
  thresholding. Turnover becomes controlled by construction, and seed variance in scale stops
  leaking into the headline.
- **Report gross alongside net as a first-class diagnostic**, so a metric dominated by
  execution can never again be mistaken for one dominated by the model.

**Also recorded:** κ\* = 3.0 at the **maximum of the pinned grid** on every horizon and every
seed. The unconstrained optimum plausibly sits outside the grid. Untouched in v1 by rule;
a v2 grid should extend until an interior optimum appears.

---

## What the successor architecture looks like

**Kronos's two-stage design is correct for its purpose and wrong for ours.** Compress bars,
model tokens — that works when the information you care about *is* the dominant variance. It
fails structurally when the signal is a small, independent fraction of it. Which is precisely
what microstructure is, and we now have that measured three ways.

Derived from the failures rather than from taste:

1. **Split the representation by information type, not by compression convenience.** Price and
   volume shape are high-variance, compressible, and benefit from a discrete AR. Microstructure
   is low-variance, independent, and does not. **Stop putting them through one bottleneck.**
2. **Train the tokenizer against what the downstream model needs**, not against reconstruction.
   Reconstruction was adopted because it is easy; we have three measurements that it is the
   wrong proxy for this class of signal.
3. **Per-bar encoding wherever per-bar recovery matters** — structural, not weighted.
4. **Calibrated forecasts and rank-relative execution**, so the money metric measures skill
   rather than turnover.
5. **Horizons and costs matched to the edge that exists**, rather than a 0.30% cost fighting a
   0.005% edge.

**That is a genuinely different model, and it is specified by our own failures.** "Here is why
the standard design fails on this class of signal, and here is the architecture that follows
from the failure" is a stronger contribution than another ablation would have been.

---

## What is measured and what is not — stated plainly

**Strongly measured (independent replications):** the eviction mechanism (3), the
cost-to-edge ratio (3 seeds, whole cost band), the drag identity (3 seeds), the
over-dispersion mechanism (3 seeds, activity predicted to 0.15pp at seed 0).

**Measured once:** the smearing mechanism (canary, n=1 architecture/config).

**Not measured at all:** *every proposed fix in this document.* The evidence is for the
diagnoses. **Nothing here is evidence that any remedy works**, and the one I would bet on —
putting the gated quantity into the loss — is also the cheapest to test, which is convenient
enough that it should be checked by someone who did not propose it.

**Cheapest open questions, in order of value per dollar:**
1. ~~Does a *windowed* read recover dims 7–8?~~ — **ANSWERED 2026-08-13, $0. No.** See
   Problem 2. Eviction is complete at every read width tested; the gate is not over-strict.
2. ~~Does the cost/edge ratio improve with horizon?~~ — **ANSWERED 2026-08-13, $0. The economics
   do not close at h=15, and no tokenizer fix closes them.** See the section below.
3. Does an auxiliary legibility head change the allocation? — one Stage-1 run, ~$0.20–0.50.

---

*Written 2026-08-13 from v1's receipts. Every measured figure above is traceable to a named
artifact under `runs_manifest/` or `runs_cloud/`. No v1 pin, gate, threshold or claim is
altered by this document.*
