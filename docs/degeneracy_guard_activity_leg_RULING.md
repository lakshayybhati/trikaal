# RULING — the degeneracy guard's activity leg

**Dated 2026-08-11. Prereg §7 amendment tag: v1.6.28 (the 42nd).**
**Ruling: KEEP AS IS. The activity leg is not banded.**
**Verified independently against receipts, not adopted on either recommendation's authority.**

Two proposals were before me and they conflict:

- **External integration review:** band the activity leg `[0.05, 0.95]` to match its sibling —
  HALT-only widening, direction-safe, cannot manufacture a positive or a negative.
- **`docs/degeneracy_guard_band_decision.md` (`5c9ab3f`, 2026-08-10):** do not band, because
  `activity_decisions` at κ\* is the wrong statistic to band.

I verified the load-bearing evidence for both. **The memo is right and the review's recommendation
is wrong on the instrument — but the review's *finding* is right, is the same as the memo's §1, and
my verification makes it worse than either party states.**

---

## 1. Why the leg is not banded

### 1.1 The κ\* scalar is not a sufficient statistic for the condition — verified

`runs_manifest/m6_h_sweep.json`, `moneyleg_noise_cell2` at h=5. I read the receipt directly:

```
activity_decisions_by_kappa = {"1": 1.0, "1.5": 1.0, "2": 1.0, "3": 0.9318333333333333}
```

**The filter is fully inert at three of the four pinned grid values while the scalar the guard reads
sits at 0.9318 — comfortably interior to `[0.05, 0.95]` and to any plausible widening.** A perfect
band on this scalar would not have seen it.

The κ\* scalar answers *"did the filter bind at the selected κ"*. The degeneracy is *"did the filter
do per-bar work"*. Only the second is the condition, and the scalar is not a sufficient statistic
for it. Banding a statistic that is provably insufficient buys the appearance of coverage.

### 1.2 It is a scale-dependent fraction, so no single band is correctly shaped

Nine trades is 7×10⁻⁶ at n = 1.3M decisions and 7.5×10⁻⁴ at n = 12k — the same disease three orders
of magnitude apart. **What makes an IR a measurement is the count of trades, not their share.** A
fraction band cannot express that condition at two scales at once.

### 1.3 The exact endpoints are principled, not a rounding boundary

When the filter does no per-bar work the argmax-over-κ map has exactly two fixed points: all-inert
→ ties → `cfg.kappas[0]` at activity 1.0; or an all-flat book scoring IR exactly 0.0, which beats
any losing book, at activity 0.0. **The endpoint test is aimed at the two attractors.** Both are
already caught.

### 1.4 A false HALT is not cheap, so widening is not costless

Because C.1 ruled S=5 up front, **R1 and R2b collapse into R3**: a HALT is a rule-selected,
non-recoverable INCONCLUSIVE with zero seed headroom, and OC-1 forbids dropping to three seeds.
`grep -c "HALT\|degenerac" docs/m6_fanout_runbook.md` returns **0** — there is no recovery procedure
in the runbook at all. The review's "direction-safe by construction" is true of the *verdict* and
false of the *run*.

### 1.5 What the review is right about

The blind spot is real and it is wide. **An endpoint-only test does not see a book trading 0.1% of
bars**, and §2 shows that is the expected real-data regime rather than a corner case. I am not
ruling that the exposure is acceptable. I am ruling that **banding this scalar does not close it**,
and that the closure is §3's reporting changes plus §2's disclosure.

---

## 2. The larger finding, and two corrections that make it worse

The memo's §1 — the execution filter may admit essentially no trades on real data — is the same
finding the external review reached from the other direction. I verified its arithmetic and found
**two independent errors, both conservative.**

### 2.1 The identity and the table reproduce exactly

For a calibrated conditional mean, `Cov(μ̂, y) = Var(μ̂)`, so `IC = std(μ̂)/std(y)` and
**`std(μ̂) = IC · std(y)`**. The identity is correct.

Using the memo's `std(y) = 0.004238`, its table reproduces to the digit: at IC = 0.027 the
threshold sits **7.0 / 9.6 / 18.4 sd** out at κ=1 for c_total ∈ {8.0e-4, 1.1e-3, 2.1e-3}, and
**28.8 sd** at κ=3. The inverted form — IC ≥ 0.096–0.253 to trade 5% of bars at κ=1 — also
reproduces. Arithmetic confirmed.

### 2.2 Correction 1 — my own lake measurement makes the threshold further out

Computed directly from `processed/universe_bars/symbol=BTCUSDT`, cumulative 15-bar forward
log-return over the **full** history:

| | memo | my measurement |
|---|---|---|
| n | 1,051,199 | **2,103,825** |
| std(y) | 0.004238 | **0.003504** |
| sd-out at IC=0.027, κ=1 | 7.0 / 9.6 / 18.4 | **8.5 / 11.6 / 22.2** |
| IC needed for 5% activity | 0.096 / 0.253 | **0.116 / 0.306** |

Different sample windows, so this is not a contradiction. **The direction is what matters: a smaller
std(y) puts the threshold further out and raises the required IC. The memo understates its own
finding.**

### 2.3 Correction 2 — the memo's mitigating branch rests on a superseded number

The memo offers two branches and hedges with: *"If they are over-dispersed (**plausible at 7.7% of
compute-optimal**) they trade."*

**That figure was superseded on 2026-08-03 by `591ec44`.** The pinned budget is now 26,003 steps =
426,033,152 tokens = **20.00 tokens/param = 100% of compute-optimal** — verified from
`PINNED_STEPS_STAGE2` at HEAD.

**A model trained to compute-optimal is more likely to be calibrated, not less.** So branch (a) —
activity ≈ 0, the guard fires on every cell, the run ends in HALT_ADJUDICATE — is **more** likely
than the memo states, and the over-dispersion escape it relies on is **weaker**. The memo was
written 2026-08-10, a week after the budget was raised; the stale figure is the only place its
reasoning drifts optimistic.

---

## 3. What this does *not* block, and why

**Applying the pre-committed run-blocking bar** (2026-08-03, fixed before its first application):

> A finding delays the run **iff** it would cause us to publish a FALSE VERDICT — SURVIVES when the
> truth is NULL, or NULL when SURVIVES — **and cannot be neutralized by disclosure.**

**§2 produces `HALT_ADJUDICATE`, which is not a false verdict.** It is the guard working. And
**HALT is one of the five §6.5 interpretation branches, written and dated before the run** — the
paper's own §6.5 enumerates *gate-fail, SURVIVES, NULL, INCONCLUSIVE, HALT*. **The paper is already
correct under this outcome.**

**It does not block. It is disclosed, and it changes what is being funded — which is the operator's
call, not a gate's.**

### The κ grid stays, and the reason is not inertia

θ = κ·c with κ ≥ 1 means *"trade only when the forecast exceeds its own cost."* **κ < 1 is
economically incoherent for a cost-aware filter** — it would mean acting on forecasts known not to
cover their costs. The grid is principled.

So if the filter admits nothing, **that is a result, not a defect**: *at a measured RankIC of 0.027,
no forecast clears a 0.30% round-trip cost by any margin, on any of 192 instruments.* That is a
publishable, economically meaningful negative finding, and it is precisely what a cost-aware study
exists to be able to say. A study that could not reach that conclusion would be the defective one.

---

## 4. What is adopted — three $0 reporting changes

The exposure in §1.5 is closed by making the risk **visible in the manifest** rather than
discovered in it. These move quantities the code already computes; **no gate value moves**, so the
freeze is untouched and no direction-blind test is engaged.

1. **Persist `activity_decisions_by_kappa`** (the full grid, not only the κ\* scalar) per
   (cell, seed). This is exactly the quantity §1.1 shows the scalar hides, and it is already
   computed — `m6_h_sweep.json` carries it today.
2. **Persist the trade COUNT alongside the fraction** per (cell, seed). §1.2's objection is that a
   fraction is scale-dependent; the count is not, and an adjudicator needs it to know whether an IR
   is a measurement.
3. **Persist `std(μ̂)/θ` at κ\*** per (cell, seed) — the dimensionless margin §2 is about. Planted
   cell 4 ran at 6.97; a calibrated real forecast at our prior would be ~0.05. **One number tells an
   adjudicator immediately which branch of §2 the run landed in.**

None gates anything. All three are `[recollection — no artifact]`-free: they are reads of existing
computations.

---

## 5. The keep-as-is rationale, stated so it survives the question

*"You knew the endpoint-only test was blind at the low pole, and you shipped it anyway."*

**Yes — and here is the record.** The blind spot is real, wide, and expected to be the operating
regime. It is not banded because the statistic available to band is provably insufficient for the
condition (§1.1, verified against a receipt), is scale-dependent so no single band is correctly
shaped (§1.2), and because its endpoints target the two attractors of the selection map rather than
a rounding boundary (§1.3). Widening it would have produced the *appearance* of coverage over a
statistic that demonstrably misses the case it is meant to catch, at the cost of a non-recoverable
HALT path with no runbook procedure behind it (§1.4).

**What was done instead:** the exposure is disclosed, the three quantities that actually express the
condition are persisted (§4), and the outcome it leads to — HALT_ADJUDICATE — is one of five
pre-written, pre-dated interpretation branches the paper already handles.

**What would change this ruling:** a statistic that is sufficient for "the filter did per-bar work"
— the count, or the full-grid activity vector — being available to band. §4 makes both available.
**Banding them is a v2 question, and it is now well-posed rather than blocked.**

---

## 6. Disposition

| item | ruling |
|---|---|
| Band the activity leg `[0.05, 0.95]` | **DECLINED** — insufficient statistic (§1.1) |
| The blind spot is real | **ACCEPTED** — disclosed, not closed by banding |
| The filter may admit no trades (§2) | **ACCEPTED, does not block** — HALT is a pre-written branch |
| Memo's `std(y) = 0.004238` | **CORRECTED** to 0.003504 on my measurement; finding strengthens |
| Memo's "7.7% of compute-optimal" | **STALE** — superseded by `591ec44`; branch (a) more likely |
| The κ grid | **UNCHANGED** — κ ≥ 1 is principled; an empty filter is a result |
| Three reporting changes (§4) | **ADOPTED** — $0, no gate value moves |
| Paper touch-points D.2 and §7.5 | **UPDATE REQUIRED** — see §7 |

## 7. Required follow-through

- **§7.5** — carry the corrected std(y) (0.003504, n = 2,103,825) beside the memo's 0.004238 with
  both sample windows named, and state the direction: the correction makes the threshold further
  out. Remove or re-date any reliance on "7.7% of compute-optimal."
- **D.2** — record that the activity leg is endpoint-only **by ruling, not by omission**, cite this
  document, and name the three §4 quantities as the compensating disclosure.
- **No mutation test is required**, because no gate value moves. The three §4 additions are
  manifest fields and take the same required-field treatment as `mu_diag`.

*Ruled by the supervisor, 2026-08-11, against `runs_manifest/m6_h_sweep.json`,
`runs_manifest/m6_moneyleg_rerun_manifest.json`, `processed/universe_bars/symbol=BTCUSDT`,
`PINNED_STEPS_STAGE2` at HEAD, and `paper/main_full.txt` §6.5.*
