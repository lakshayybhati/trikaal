# PRE-DECLARATION — UNITS 1 AND 2 (cell 4 seed 0, cell 5 seed 0)

**Written BEFORE the units run. HEAD `3e86e22`. A reading declared after the number is a story.**

---

## 1. ★ THE STANDALONE RESULT, TRUE REGARDLESS OF HOW THE RUN ENDS

> **Three seeds of one cell, identical data and configuration, produced headline IRs of −28.47,
> −67.02 and −146.49 — a range of 118.03, which is 57.9× the scoring standard error the
> pre-registered MDE was constructed from. Activity spans 24× across those seeds.**

Measured, not argued. Provenance of every number in it:

| quantity | value | source |
|---|---|---|
| IR(cell1, seed0) | −67.0160 | `runs_cloud/results/r0/cell1_seed0_eval.json` |
| IR(cell1, seed2) | −28.4690 | `runs_cloud/results/r1/cell1_seed2_eval.json` |
| IR(cell1, seed4) | −146.4945 | `runs_cloud/results/r2/cell1_seed4_eval.json` |
| range | 118.0255 | max − min |
| SE_scoring | 2.0378 | moving-block bootstrap on cell1_seed0's own `headline_series`, PINNED recipe: B=10,000, seed 20260704, L=⌈√35064⌉=188, 187 blocks |
| ratio | **57.9×** | 118.0255 / 2.0378 |
| activity @κ=3 | 0.0193 / 0.0926 / 0.4702 | `mu_diag.activity_decisions`, seeds 2 / 0 / 4 → **24.4×** |
| tabled MDE | 3.518 | `runs_manifest/m6_mde_inputs.json:h15_pooled` |

All three artifacts sha256-verified against their on-box production-time hashes, all three
produced at `git_commit 055d14f`, all 19 provenance keys identical except the recorded kernel on
r2. **All three now validate CLEAN under the shipped v1.6.30 gate.**

This belongs in the paper's methods whatever the verdict word turns out to be.

---

## 2. THE PRE-DECLARED READING OF ΔIR₀ — BOTH SCALES MEASURED, NOT CHOSEN

Units 1 and 2 are **cell 4 seed 0** (machine 19883) and **cell 5 seed 0** (r1, machine 13450), run
**in PARALLEL on two physical machines verified identical on ALL 19 RECORDED KEYS including the
kernel** — `Linux-5.15.0-52-generic-x86_64-with-glibc2.35`, driver `590.48.01`, at
`git_commit 448e4fe`, both passing the pre-flight identity gate on all 16 compared keys.

*(This paragraph originally read "sequentially on r1", written before machine 19883 — r0's own
machine, on the host that supplied both r0 and r1 — was found available and verified. Corrected
rather than left stale.)*

**THE STATED LIMITATION, RECORDED WHICHEVER WAY THE NUMBER FALLS:** the pair ran on two physical
machines identical on all 19 provenance keys; **any cross-card term is unquantified** — nobody has
ever verified that two distinct physical 4090s produce bit-identical results — and the reading is
robust only to a term small relative to the 29.5× separation between the two hypotheses. A
card-to-card difference of order 85 would be needed to flip the net reading, which is not credible
under forced determinism with every software key matched, but it is not zero and it is not
measured.

Two hypotheses, quantitatively different predictions, both derived from measurements already in
hand (verified independently — the supervisor's figures reproduce exactly):

    PERFECT PAIRING   |ΔIR_0| ~ sqrt(2) x SE_scoring  = sqrt(2) x 2.0378  =  2.8819
    NO CANCELLATION   |ΔIR_0| ~ sqrt(2) x sd(level)   = sqrt(2) x 60.1841 = 85.1131
    separation                                                            = 29.5x
    DIVIDING LINE = geometric midpoint sqrt(2.8819 x 85.1131)             = 15.6616

The line is **derived from the two hypotheses, not invented** — the self-scaling rule, the same
one that killed the hand-chosen 0.05 band on the C-1 symmetry statistic.

**DECISION, FIXED IN ADVANCE:**

- **|ΔIR₀| < 15.66** → closer to the cancellation world. **BUY PAIR 2** (cell 4 + cell 5 at
  **seed 1**) to obtain the first actual `sd(ΔIR_s)`.
- **|ΔIR₀| > 15.66** → closer to the independence world. **STOP AND REPORT.** On that reading the
  5-seed design cannot resolve an effect at the 0.5 economic floor, and the remaining ~$180 buys a
  HALT_ADJUDICATE.

### ★ THE LIMIT, STATED PLAINLY: ONE PAIR IS A SCREEN, NOT A DISPERSION

One pair gives **one number**. It **cannot prove cancellation** — a single small |ΔIR₀| is equally
consistent with luck, because one draw from a wide distribution can land anywhere including near
zero. It **can falsify** cancellation: a large |ΔIR₀| is hard to explain under a hypothesis that
predicts ~2.9. The test is therefore **asymmetric by construction**, and the `< 15.66` branch buys
a second pair precisely because the first cannot settle it.

### ONE REFINEMENT, AND IT CUTS AGAINST THE CHEAPER CONCLUSION

`sqrt(2) × SE_scoring` treats the two cells' scoring noises as INDEPENDENT. They are not:
`paired_delta_ir_bootstrap` shares one set of moving-block indices across both series precisely so
the common market component cancels (`paired_bootstrap.py:6-7`). So the true scoring noise on a
paired ΔIR is **smaller** than 2.8819, which makes 2.8819 an **upper bound** on the pairing
prediction — and therefore makes the true dividing line **lower** than 15.6616.

Using 15.6616 is thus biased toward the **BUY PAIR 2** branch, i.e. toward spending more rather
than stopping. Recorded because a bias that favours the more expensive path is exactly the one
worth declaring in advance. **I am not moving the line** — it is the supervisor's, declared before
the data, and shifting it after this observation would be the defect the whole exercise exists to
prevent.

---

## 3. WHAT IS REPORTED, WITHOUT INTERPRETATION

`IR(4,0)`, `IR(5,0)`, the paired `ΔIR₀ = IR(4,0) − IR(5,0)`, both codebook utilizations, and the
`dispersion` block. **The power arithmetic and what follows are the supervisor's and the
operator's.**

---

# ADDENDUM — THE GROSS READING, PRE-DECLARED 2026-08-12T20:12:58Z

**Written BEFORE units 1 and 2 landed. Boxes untouched, $0, local, from the three banked cell-1
artifacts only.**

## 1. THE THREE REQUESTED QUANTITIES

Identical pinned recipe to the net derivation: B=10,000, seed 20260704, L=⌈√35064⌉=188, on
`gross_series(headline_series, headline_cost=0.0030)`.

| seed | IR_gross | SE_gross | 95% bootstrap interval |
|---|---|---|---|
| 0 | 1.2665 | **0.9581** | [−0.6080, 3.1722] |
| 2 | 2.1715 | 1.0277 | [0.2375, 4.2539] |
| 4 | 1.1443 | 1.0889 | [−0.9345, 3.2862] |

    SE_scoring_GROSS (seed 0, matching how SE_scoring = 2.0378 was derived) = 0.9581
    sd(IR_gross) across 3 seeds                                             = 0.5611

    PERFECT PAIRING   sqrt(2) x SE_gross = 1.3550
    NO CANCELLATION   sqrt(2) x sd_gross = 0.7936     (supervisor's figure 0.7935 — reproduced)
    geometric midpoint                    = 1.0370

## 2. ★ THE CONSTRUCTION DOES NOT SEPARATE ON THE GROSS BASIS. REPORTED, NOT PATCHED.

On the **net** basis the two hypotheses are 2.8819 and 85.1131 — **29.5× apart and correctly
ordered**, pairing below independence. One measurement discriminates them, which is what makes
15.6616 a real test.

On the **gross** basis they are **1.3550 and 0.7936 — only 1.7× apart, and INVERTED**: the
"perfect pairing" prediction is LARGER than the "no cancellation" prediction.

**The cause is arithmetic, not opinion.** The construction assumes `sd(level) >> SE_scoring`. Net:
60.1840 vs 2.0378. Gross: **0.5611 vs 0.9581 — the across-seed spread is SMALLER than the
bootstrap noise of a single measurement.** Once that inequality flips, the two predictions cross
and the geometric midpoint no longer sits between two separated hypotheses; it sits inside their
overlap.

**So 1.0370 is recorded as the arithmetic answer to the question asked, and it is NOT a usable
dividing line.** A threshold whose two hypotheses predict values on the wrong side of each other
cannot be failed in either direction — the same family as "a check that cannot fail is not a
check". **I am not inventing a substitute construction after seeing this**; the numbers are above
and the reading is the supervisor's.

The same caveat as the net line applies and is likewise **NOT adjusted for**: the paired bootstrap
shares block indices, so the true paired scoring noise is below `sqrt(2) x SE_gross`, making the
pairing prediction an upper bound.

## 3. ★ THE DIRECTIONAL EXPECTATION — THE ACTUAL HYPOTHESIS, STATED IN ADVANCE

**IF MICROSTRUCTURE CARRIES INFORMATION, THEN `IR_gross(cell 4, seed 0) > IR_gross(cell 5, seed 0)`.**

Cell 5 is the same architecture, same seed, same arm geometry, on **shuffled** microstructure. A
gross contrast that is **negative or zero at seed 0 is evidence AGAINST the microstructure leg**,
and it is evidence that does not depend on any power or pairing question.

**ONE SEED IS WEAK AND THAT IS SAID HERE, NOT DISCOVERED LATER.** This is DIRECTIONAL, NOT
DECISIVE. The bootstrap intervals above show why: on the banked cell-1 units the 95% interval for
IR_gross contains 0 for two of three seeds. A single-seed gross contrast is one draw from a
distribution of comparable width, so a positive result does not establish the leg and a negative
result does not kill it — it moves the prior, and it does so before any data was seen.

## 4. WHAT DECIDES WHAT — EXPLICIT

- **THE NET LINE AT 15.6616 DECIDES BUY PAIR 2 / STOP. Unchanged. Gross does not vote.**
- The gross numbers decide **what we have learned** on either branch, and whether the tokenizer
  question is answerable on a basis the execution filter does not dominate.

Reported on landing, without interpretation: net IR(4,0), IR(5,0), ΔIR₀ vs 15.6616 · gross
IR_gross(4,0), IR_gross(5,0), ΔIR_gross₀ · a₄, a₅ · and the drag-predicted ΔIR from those
activities alone.
