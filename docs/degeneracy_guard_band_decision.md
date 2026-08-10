# The degeneracy guard's activity leg — and the larger thing found while checking it

**Decision staged for Lakshay. Recommendation only — no code changed, nothing committed to the
frozen guard.** Written 2026-08-10. Companion to `docs/invariant7_amendment_decision.md`.

> **This memo reversed twice before it settled, and both reversals are recorded in §2 rather than
> quietly overwritten.** The final answer agrees with my first instinct — do not band it — but
> every reason I first gave for it was wrong, and the reasons that survive are not mine. What the
> checking actually produced is §1, which matters more than the question I was asked.

---

## 0. Recommendations, in priority order

1. **§1 first.** At our own measured prior for forecast skill, the pre-registered execution filter
   admits essentially **no trades on real data**: at $\text{IC}=0.027$ the threshold sits **7 to 28
   standard deviations** out in the forecast distribution, and **1 of 192 instruments** would trade
   even 5% of bars under the most favourable modeled cost. This is a property of the *design*,
   measured on the lake. It should be resolved before the guard question is worth a decision.
2. **Do not band the activity leg — at either pole.** Not because the blind spot isn't real (it is,
   and it is wide), but because `activity_decisions` at $\kappa^\ast$ is the wrong statistic to
   band: it is a scale-dependent fraction, and it is provably not a sufficient statistic for the
   condition it stands for (§3).
3. **Instead, make three $0 reporting changes** that move quantities the code *already computes*
   into the manifest, so the risk becomes visible without moving any gate value (§4).
4. Record the caveats in §5. Fix the one misleading comment in §6.

---

## 1. The larger finding: the filter is unreachable at our own prior

Binding compares the forecast $|\hat{\mu}|$ — a raw cumulative log-return, per `predict.py:203`,
which multiplies the vol-relative output back to raw units — against
$\theta = \kappa\, c_{\text{modeled}}$ (`xsection.py:335`). For a **calibrated** conditional mean,
$\operatorname{std}(\hat{\mu}) = \text{IC}\cdot\operatorname{std}(y)$.

Measured on the lake rather than assumed: BTCUSDT's 15-bar forward log-return has
$\operatorname{std} = 0.004238$ over $1{,}051{,}199$ bars; across 192 instruments in the headline
year the median is $0.00536$ (p5 $0.00372$, p95 $0.00887$).

At $|\text{RankIC}| = 0.027$, over the modeled-cost range the harness's own KATs pin
($8\times10^{-4}$ pure-taker to $2.1\times10^{-3}$ at 0.5% participation):

| $c_{\text{total}}$ | $\kappa=1$ | $\kappa=1.5$ | $\kappa=2$ | $\kappa=3$ |
|---|---|---|---|---|
| $8.0\times10^{-4}$ | 7.0 sd → $3\times10^{-12}$ | 10.5 sd | 14.0 sd | 21.0 sd |
| $1.1\times10^{-3}$ | 9.6 sd → $7\times10^{-22}$ | 14.4 sd | 19.2 sd | 28.8 sd |
| $2.1\times10^{-3}$ | 18.4 sd → $3\times10^{-75}$ | 27.5 sd | 36.7 sd | 55.1 sd |

Inverted: a calibrated forecast needs $\text{IC}\ge 0.096$–$0.253$ to trade 5% of bars at
$\kappa=1$ — **4× to 9× our measured prior**. Across the 192 instruments at the most favourable
cost, **one** clears 5% activity and four clear 1%.

**The fixture cannot warn us about this, and its own numbers show why.** Planted cell 4 runs at
$\operatorname{std}(\hat{\mu})/\theta = 6.97$, which for a calibrated forecast would require
$\text{IC} > 1$. Every observation we have of the filter behaving sensibly comes from an
over-dispersed forecast in a regime real data cannot reach.

**Two branches, both informative.** If the trained models are calibrated, activity rounds to
exactly $0$ (expected trades $\approx 0.001$ of $35{,}063$), the guard fires on every cell, and the
run ends in `HALT_ADJUDICATE` rather than a verdict — the guard working, but the outcome taxonomy
never exercised. If they are over-dispersed (plausible at 7.7% of compute-optimal) they trade, but
the realized IR then partly reflects miscalibration rather than skill.

I am not proposing a design change. I am saying $\kappa$'s grid — pinned at
$\{1.0,1.5,2.0,3.0\}$, every value $\ge 1$ — is worth looking at before ~195 GPU-hours are spent
discovering this.

**Assumptions, so this can be attacked precisely.** (i) Calibration; this is load-bearing, and the
second branch is what happens if it fails. (ii) Gaussian $\hat{\mu}$ — fat tails raise tail mass
but nothing rescues $7\sigma$. (iii) $\operatorname{std}(y)$ measured on 2024-06 for the
cross-section, BTCUSDT full history for the headline figure. (iv) Arithmetic over measured inputs;
the harness was **not** run end to end at these values.

## 2. Two wrong answers, recorded

**First answer — "do not band, the leg is redundant."** It rested on: *across every measured pair,
the activity leg has never fired without the sign leg also firing.* **False.**
`m6_moneyleg_rerun_manifest.json`, planted cell 1: `frac_negative = 0.1375`, comfortably **inside**
$[0.05,0.95]$ so the sign leg is silent, with `cell_ir` exactly $0.0` — the prereg names it *"the
single sub-threshold cell"*, i.e. it never traded, so activity is $0$ and **the activity leg fires
alone**. The arithmetic is consistent: `mean` $6.51\times10^{-4}$ against `std`
$6.05\times10^{-4}$, so $\hat{\mu}$ straddles zero (balanced signs) while sitting below
$\theta\approx1.1\times10^{-3}$ almost everywhere.

How I missed it is the project's own pathology: my enumeration read `activity_decisions` from every
receipt, and *this* receipt does not carry that field. The row returned `None`, dropped out of the
table, and I asserted a universal over a set my own probe had silently truncated. **Separate your
probe's bugs from its findings** — this was the probe's limitation wearing a conclusion's clothes.

The structural argument was one-sided too. It explains the activity $\to 1$ pole correctly; the
activity $\to 0$ pole is the opposite mechanism ($\hat{\mu}$ too small to clear $\theta$), fully
compatible with balanced signs — and §1 shows that pole is the *expected* real-data regime, not a
corner case.

**Second answer — "band the lower pole."** Also wrong, for a reason I had not considered: it bands
the wrong object. §3.

## 3. Why banding `activity_decisions` is the wrong instrument

**It is not a sufficient statistic for the condition it stands for.** Measured, in our own
receipts: `m6_h_sweep.json`, `moneyleg_noise_cell2` at $h=5$ —

```
activity_decisions_by_kappa = {1: 1.0, 1.5: 1.0, 2: 1.0, 3: 0.9318}
activity_decisions (at kappa*) = 0.9318
```

The filter is **fully inert at three of the four grid values** while the scalar the guard reads
sits at $0.9318$, comfortably interior to any plausible band. A perfect band on the $\kappa^\ast$
scalar would not have seen it. The κ\* scalar answers "did the filter bind at the selected κ", not
"did the filter do per-bar work", and only the second is the degeneracy.

**It is a scale-dependent fraction, so no single band is correctly shaped.** Nine trades is
$7\times10^{-6}$ at $n=1.3$M decisions and $7.5\times10^{-4}$ at $n=12$k — the same disease three
orders of magnitude apart. What determines whether an IR is a measurement is the **count** of
trades, not their share.

**A false HALT is not cheap, so widening on a badly-shaped statistic is not free.** I checked the
premise that made widening look costless and it does not hold here: because C.1 ruled $S=5$ up
front, R1 and R2b collapse into R3 — a HALT is a rule-selected, non-recoverable INCONCLUSIVE with
zero seed headroom, and OC-1 forbids dropping to three seeds. `grep -c "HALT\|degenerac"
docs/m6_fanout_runbook.md` returns **0**: there is no recovery procedure in the runbook at all.

**And the endpoints are principled rather than arbitrary.** When the filter does no per-bar work
the argmax-over-$\kappa$ map has exactly two fixed points: all-inert → ties → `cfg.kappas[0]` with
activity $1.0$; or an all-flat book scoring IR exactly $0.0$, which beats any losing book, with
activity $0.0$. Both are already caught. The exact-endpoint test is aimed at the two attractors,
not at a rounding boundary.

**One claim I checked and will not pass on.** The stress test asserted that IR scales as
$\sqrt{\text{activity}}$, making a cross-cell activity gap a first-order confound in
$\Delta\IR$. Within the one healthy cell we have, that is **not** supported: across $\kappa$,
activity falls $0.962\to0.887$ while IR stays flat at $14.83$–$15.02$, and the
$\sqrt{\cdot}$ prediction under-shoots by up to $0.72$. The filter's selection offsets the
frequency effect, as designed. The cross-variant ratio the claim rested on was confounded with a
change in the decision cap.

## 4. What to do instead — three $0 reporting changes

None moves a pin, threshold, seed, enumeration or gate value, so none is a v1.5-class amendment.
All three surface quantities the code already computes. All three still touch the anchored
instrument, so each needs a Gate-A re-proof and your sign-off.

1. **Persist `n_trades`.** It is computed at `harness.py:132` (`n_trades=int(active.sum())`) and
   discarded — a repo-wide grep finds it only in the dataclass field, that assignment, and one
   test. It never reaches `xsection.py` or `verdict.py`. This is the quantity that decides whether
   an IR is a measurement, and we throw it away.
2. **Persist the full per-$\kappa$ activity vector**, not only the $\kappa^\ast$ scalar. The
   h-sweep receipt already has the shape (`activity_decisions_by_kappa`); the eval artifact does
   not. §3's counterexample is invisible without it.
3. **Surface both in the verdict manifest** beside the existing per-seed activity values, so an
   adjudication reads the distribution rather than one scalar.

With those three in place, the §1 regime becomes visible in the artifact the moment it occurs,
and a later decision to band — on trade count, with a derivation — has the inputs it needs. Today
it does not: no across-seed spread of `activity_decisions` has ever been measured anywhere in this
repository, so the self-scaling denominator a principled band would require does not exist pre-run.

## 5. Caveats to record whichever way this goes

1. **Clause 5's multiple-testing correction is deflated by a near-inert filter, and the deflation
   is anti-conservative.** `var_sr` is the variance of the **cell-5 (placebo)** trial set —
   5 seeds × 3 horizons × 4 $\kappa$ — and $SR_0=\sqrt{\text{var\_sr}}\cdot f(N)$ with $N$ pinned
   at 60. Measured: when the filter binds, the $\kappa$ axis contributes real dispersion (planted
   cell 4 spans $0.19$–$0.96$ IR across $\kappa$); when it does not, the $\kappa$ axis contributes
   **exactly** $0.0000$ at every horizon. An inert placebo filter therefore shrinks `var_sr`
   relative to what it would otherwise be while $N$ stays 60, lowering $SR_0$ and making clause 5
   **easier to pass**. Neither guard sees this below the exact endpoints; the magnitude on real
   data is unmeasured. This belongs in §4.7 of the paper, not in a limitations list — it affects a
   clause, not a scope boundary.
2. **The verdict guard has no dispersion leg, by specification.** `MU_DIAG_REQUIRED_KEYS` is
   `("frac_negative", "activity_decisions")`; `std` is not required, so a dispersion floor could
   not be evaluated on the real run. The canary has one (`NONDEGEN_MU_STD_FRAC * SIGMA`) but it is
   defined against SIGMA, a fixture-only parameter, so it cannot be ported as written.
3. **The two legs are correlated at the upper pole and separate at the lower one.** §7 v1.4.4
   describes a conjunction where "each leg covers a case the others miss". At activity $\to 1$ they
   detect the same condition; at activity $\to 0$ they genuinely separate, which is what §2's
   counterexample shows. There is less independent redundancy than the conjunction language
   suggests.
4. **Every measured activity/frac-negative pair in the repository is synthetic-fixture.** Zero come
   from the lake. Any statement of the form "banding changes no measured outcome" is true of a
   regime, not of the run.

§4.4 of the paper already discloses the endpoint asymmetry. Caveats 1–4 are not written anywhere
yet and should be.

## 6. The one change I recommend that is not a guard change

`scripts/m6_canary.py:883` says the canary's three-leg conjunction **"Mirrors the real-run verdict
guard."** It does not: the canary has a dispersion leg the verdict guard lacks, and the verdict
guard does not require `std` at all. A reader takes that comment as assurance of coverage that does
not exist. One-line comment fix, no behaviour change, no sign-off needed.

Per the class rule I swept for siblings rather than patching the instance — every
`mirror`/`mirrors`/`same as the real`/`matches the verdict` claim across `src/trikaal/`,
`scripts/*.py`, `docs/*.md`, `paper/sections/*.tex`. Twelve hits, eleven accurate; the two
neighbouring canary claims (`:862` and `:1213`) are correct, since both name exactly which clauses
they mirror. Line 883 is a single instance, not a class.

## 7. What I did not do

No file under `src/` or `scripts/` was modified. No test was added. Nothing was committed to the
guard. No band value was computed — §4 argues the band should be on a different statistic and that
its denominator does not exist pre-run. §1's arithmetic is over measured inputs; the harness was
not run end to end at those values.
