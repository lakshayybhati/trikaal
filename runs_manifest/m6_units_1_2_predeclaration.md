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

Units 1 and 2 are **cell 4 seed 0** and **cell 5 seed 0**, run **sequentially on r1** — one
instrument, agreeing with itself on all 19 keys including the recorded kernel, so no box effect
can enter the paired difference the design's power rests on.

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
