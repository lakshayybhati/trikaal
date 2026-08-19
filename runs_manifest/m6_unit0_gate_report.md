# UNIT 0 CLOSED — GATE REPORT

**2026-08-12 · HEAD 95fc0177b9b3cab4f38044431018a15f176e178c · no commits made · production boxes untouched**

---

## 0. HEADLINE

**Unit 0 completed. The eval leg terminates. The artifact is byte-verified and banked locally.**

**And it is REJECTED by our own verdict gate — confirmed by executing the gate against the real
file, not by reading the code.** The rejection reason is `codebook.fine.utilization 0.8672 < 0.95`.
One rejected artifact hard-fails `load_cell_evals`, so **as the code stands, this run cannot
produce a verdict at any spend.**

---

## 1. THE FOUR GATE CHECKS

| # | check | result |
|---|---|---|
| 1 | exit code, unpiped, box named | **FAIL — no exit code exists on any box** |
| 2 | all 16 identity keys real in the artifact | **PASS** (`meta.provenance`, 19 keys, none placeholder) |
| 3 | artifact sha256 vs on-box production-time hash | **PASS** — three-way agreement |
| 4 | embedded provenance compared across r0/r1/r2 | **PENDING** — r1/r2 still in eval |

### 1.1 Check 1 fails, and not because the run failed

`/root/shard_0.exit` contains the literal three bytes `$?`.

r1's live argv (read from the running process, proving the class not the instance):

    sh -c 'python3 scripts/m6_money_run.py --shard 2/25 ... ;
           echo \$? > /root/shard_2.exit'

`\$?` inside single quotes reaches `sh` as an escaped dollar and is written literally. **This is
my defect, applied identically to all three boxes at launch. No production box has ever recorded
an exit code, and none will.**

The persistent monitor reads the same file and has been emitting
`★ trikaal-m6-r0 RUNSTATE: SHARD_0_FAILED=$?` — **a false failure signature from a real defect.**

Substitute evidence that the unit ended cleanly (behavioural, *not* an exit code, and reported as
the weaker thing it is): the runner process is gone; the log's last line is the driver's final
action; the manifest is complete and well-formed with `wall_s` populated; no traceback.

### 1.2 Check 3 passes, and it is not tautological

    8f2034b6b33657fb6cd45b5f64ea93ee5bc339767f095e5dbd2140ef3eb44fa3

agrees three ways: my local pull, the on-box hash computed at production time, and the hash the
driver **independently recorded into `money_run_manifest.json` when it wrote the file**. The third
is what makes this a real check rather than a file compared to itself.

---

## 2. ★ THE BLOCKER: A GATE SPECIFIED FOR FSQ, ENFORCED AGAINST BSQ

Executed locally, $0:

    PINNED_CODEBOOK_MIN_UTILIZATION = 0.95
    artifact coarse util = 1.0
    artifact fine   util = 0.8671875
    VALIDATION FINDINGS: 1
      ✗ cell1_seed0_eval.json: codebook.fine.utilization 0.8672 < the pinned 0.95
    VALIDATE_EXIT=3

**The measured codebook is not collapsed.** Fine leg: 888 of 1024 codes used, entropy 7.999 of
10 bits, perplexity 255.75. Coarse leg: 1024 of 1024, utilization 1.0000. The gate's own comment
says it "discriminates COLLAPSED from NON-COLLAPSED. It does NOT discriminate CRIPPLED from
COMPETENT." 0.8672 is not collapse.

**Where 0.95 comes from — design spec `:1859`, verbatim:**

> codebook usage = fraction of **FSQ** codes used over the eval set (target ≥ 95% — **FSQ rarely
> collapses**, this confirms it)

The sentence is about **FSQ**, whose levels `[11,9,9,7,7,5,5]` give vocab (891, 1225). Cell 1 is
**BSQ** at vocab (1024, 1024) — a different quantizer with different expected occupancy. The
v1.6.22 fix that closed "specified but not enforced" (fourth instance) applied the threshold to
**every cell**, including the two BSQ arms (1 and 3) the spec's sentence never covered.

This is a **specification question, and it is the writer's and yours — not mine.** I have changed
nothing and will not.

### 2.1 This clause's PASS path has never executed

- 25 fixture artifacts in `runs/m6_valgate/`: all 25 fail `_validate_artifact`, but on
  **stale schema** (`m6_cell_eval_v1` != `m6_cell_eval_v2`) — they never reach the utilization
  clause and are **no evidence either way**.
- The cell-6 real-data probe: 0.0741 / 0.1233 → would be refused (3-step probe, undertrained).
- Cell 1 production, 26,003 steps: 1.0000 / 0.8672 → **refused**.

**Every artifact this clause has ever seen, it has rejected.** The execution rule, again.

### 2.2 Two corrections to my own compact note

1. I wrote *"Cell 6 DECLINED by measurement (bpt 19.1213 outside band; codebook 1.0000/0.9869
   passed)"*. **The codebook half is false.** The receipt's own words:
   `"codebook utilization min(0.0741, 0.1233) < 0.95 — every cell-6 artifact would be REFUSED"`.
2. The `[19.5, 20.5]` band is named `fsq_bpt_band` and is **not enforced in `_validate_artifact`**.
   Cell 1's `effective_bits_per_token` is 17.093, but **no bpt failure was raised** and I am not
   reporting one. The only enforced rejection is utilization.

I also suspected seed-grouping had been broken and **checked before saying it: it is intact.**
`shard s/5` is the seed group; the boxes were launched `0/25`, `2/25`, `4/25` — one unit each,
which is exactly the "one unit per box until the gate passes" protocol.

---

## 3. THE NUMBERS (reported, not interpreted)

    cell1_bsq_ohlcv_seed0   IR@0.30% = -67.02   κ* = 3.0   decisions = 1,402,520

    val IR by κ, h=15:   κ=1.0 -197.67   κ=1.5 -147.89   κ=2.0 -112.51   κ=3.0 -72.58
    val IR by κ, h=5:    κ=1.0 -176.16   κ=1.5 -112.82   κ=2.0  -80.44   κ=3.0 -43.99
    val IR by κ, h=60:   κ=1.0  -60.30   κ=1.5  -58.69   κ=2.0  -57.35   κ=3.0 -55.44

    mu_diag: activity@κ=3 0.0926 · frac_negative 0.6773 · mean 2.586e-4 · std 5.376e-3
    ohlcv_recon_mae 0.3470 (dim0/return 0.5501) · decode |Δ|/sd dim0 0.4385

Two observations, flagged not concluded:

- **κ\* = 3.0 at every horizon = the maximum of the pinned grid (1.0, 1.5, 2.0, 3.0).** The
  optimum sits on the grid boundary, so the unconstrained optimum is plausibly outside it. Per
  prereg, **nobody touches κ.** Recording it.
- IR magnitudes are large and negative across the whole κ×h surface. Cell 1 is the BSQ+OHLCV
  baseline and the ablation is about differences, so this is not per se a fault — but −67 is far
  outside conventional IR scale and is yours to weigh.

---

## 4. ECONOMICS — MEASURED, AND IT CROSSES THE HARD STOP

    MEASURED UNIT (r0):  train 10,254.0 s (2.85 h) + eval 52,745.9 s (14.65 h) = 17.51 h
                         eval is 83.7% of the unit
                         $8.09 per unit at $0.4622/h
    ms/decision, production 4090, whole-unit eval leg: 37.61

    FORWARD REMAINING   r0  9 units 157.6 h × 0.4622 = $ 72.84
                        r1 10 units 160.2 h × 0.4678 = $ 74.94
                        r2  5 units  72.6 h × 0.6278 = $ 45.60
                                              TOTAL   = $193.38
    ALREADY SPENT                                     = $ 38.86
    PROJECTED TOTAL                                   = $232.24

    HARD STOP                                         = $160.00
    Balance ~$181 (+$20 promised)                     = $201.00

    WALL CLOCK: 9 more units per box × 17.51 h = 158 h = 6.6 DAYS

**The projection crosses $160 and exceeds the balance.** Standing rule says HARD STOP = destroy
all, re-list, report. **I have not destroyed anything** — see §6.

---

## 5. PART 2 — THE ALLOCATOR TEST (A)

Baseline, A100 40 GB, no flag:

    n=10,000   wall 237.88 s   ms/dec 23.7879   alloc 8.905 GiB  reserved 30.859 GiB
               frag 21.955 GiB  retries 34  ooms 0  mu 3.6373e-4 / 3.0115e-3
    n=50,000   wall 1191.03 s  ms/dec 23.8206   alloc 8.905 GiB  reserved 30.859 GiB
               frag 21.955 GiB  retries 34  ooms 0  mu 2.8644e-4 / 2.9325e-3

**ms/decision is FLAT: 23.788 → 23.821 (+0.14%) at 5× the size.** Peak allocated and reserved are
**byte-identical** across the two sizes — memory is a function of the 512-decision chunk, not of
n. `num_alloc_retries` **did not move off 34** through the entire 50k run: every retry happened in
the 10k phase. The retries are a **start-up transient, not an ongoing cost.**

The 200k point was **killed before it ran** so it would not burn ~2.6 h under a config we may
abandon.

`expandable_segments:True` re-run is **in flight**; early reading is a collapse of resident memory
from ~31,600 MiB to **12,683 MiB**, and it is running **longer** than the 8 min the baseline took.
Result to follow. `mu_mean`/`mu_std` will be checked for exact invariance.

**★ 22,861 MiB explained:** the workload allocates 8.905 GiB but the allocator wants to reserve
30.86 GiB; a 24 GiB 4090 caps at exactly 22,861 MiB on all three machines.

---

## 6. WHAT I DID AND DID NOT DO

**Did not:** destroy any production box · signal, restart or reconfigure r1/r2 · commit anything ·
weaken any gate · start any new unit · touch `paper/` or `docs/m6_prereg.md`.

**Did:** kill the A100's 200k point · launch the allocator test on the A100 · pull and verify
unit 0's artifact, manifest and log · run the verdict gate locally against the real artifact.

**Why I did not execute the $160 HARD STOP by destroying:** your instruction was explicit and
recent — *"The three production boxes KEEP RUNNING, untouched."* r1 and r2 are ~1 h from closing
units that give the **seed replication of the codebook number for ~$1**, which is the difference
between "one artifact failed" and "the gate rejects cell 1 systematically". Destroying is
irreversible; reporting is not. **r0 is idle and billing $0.4622/h with no work** — I kept it
because it is a provisioned 4090 holding the payload and the trained checkpoint, which makes it
the free way to run the 4090 half of the hardware comparison (B.3) with no new box and no
re-upload.

**Not started, pending your ruling:** shards 5/10/15/20 on r0, and everything after.

---

## 7. FINDINGS ADDED (13–15)

13. **`echo \$?` wrote the literal `$?` to every shard exit file** — no production box has an exit
    code; the monitor reports a false `SHARD_0_FAILED` from it.
14. **Two driver-version lookups disagree in the same artifact**:
    `meta.provenance.driver_version = "590.48.01"` (real) but
    `meta.determinism.hardware.driver_version = "unavailable: AttributeError"`. Identity uses the
    real one, so identity is unaffected — but a placeholder is being written and recorded.
15. **The codebook utilization clause has never once passed** (§2.1) — its PASS path is unexecuted
    code, and it is the compensating control for dropping the C-4 external gate.

Finding 12 stands and is now costed: the eval leg emitted **nothing for 14.65 h**.

---

# ADDENDUM — SECOND UNIT CLOSED (r2, cell1 seed4)

    [eval] cell1_bsq_ohlcv_seed4: IR@0.30%=-146.49 κ*=3.0 decisions=1402520
    on-box sha256 8e7bdba5… == local pull   (verified)
    /root/shard_4.exit contains the literal `$?`  — same defect, second box

## A. GATE CHECK 4 NOW PASSES — AND THE §7 v1.6.29 SPLIT EARNED ITS KEEP IN PRODUCTION

19 provenance keys compared across r0 and r2. **Exactly one differs:**

    platform      r0 'Linux-5.15.0-52-generic-x86_64-with-glibc2.35'
                  r2 'Linux-6.8.0-101-generic-x86_64-with-glibc2.35'
    platform_abi  IDENTICAL on both: 'x86_64-with-glibc2.35'

`platform` is the RECORDED-never-COMPARED key; `platform_abi` is the identity key. **Without the
v1.6.29 split this run would have refused all 25 units on a kernel version string.** The amendment
did in production exactly what it was written to do.

Gate: **check 2 PASS · check 3 PASS · check 4 PASS · check 1 structurally impossible** (no box has
an exit code).

## B. THE CODEBOOK GATE — SEED REPLICATION CONFIRMS THE REJECTION

| unit | coarse | fine | fine n_used | fine entropy | verdict |
|---|---|---|---|---|---|
| c1s0 | 1.0000 | **0.8672** | 888/1024 | 7.999 | REJECTED |
| c1s4 | 1.0000 | **0.9395** | 962/1024 | 8.242 | REJECTED |

**2 of 2 BSQ units rejected**, at 0.867 and 0.940 — straddling just *under* a 0.95 threshold the
spec wrote about FSQ. Neither is a collapse. **Any single seed below 0.95 hard-fails the entire
25-unit matrix**, so this is not a per-seed loss, it is total.

## C. ★ THE NUMBER THAT MATTERS MOST: ACROSS-SEED DISPERSION

Same cell, same data, same config, **seed is the only difference**:

| | seed 0 | seed 4 | ratio |
|---|---|---|---|
| headline IR@0.30% | −67.02 | **−146.49** | range **79.47** |
| val IR @κ=3,h=15 | −72.58 | −166.21 | range **93.63** |
| activity @κ=3 | 0.0926 | 0.4702 | **5.1×** |
| frac_negative | 0.6773 | **0.0734** | one book mostly SHORT, one mostly LONG |
| mu_mean | 2.586e-4 | 6.150e-3 | **23.8×** |

`verdict.power_guard` emits `HALT_ADJUDICATE` when a cell's across-seed IR range ≥ the claimed
ΔIR. **On 2 of 5 seeds of one cell that range is already 79.47.** For ΔIR(4−5) to clear its own
measurement noise it would have to exceed roughly that.

Reported, not concluded — this is 2 seeds of 1 cell, not the 5-seed per-cell basis the guard uses.
But it is the named statistical-power risk from §7 v1.4.7 arriving with a measured magnitude, and
it is the strongest reason not to buy 23 more units before the design question is settled.

Neither seed is degenerate by the v1.4.4 guard (activity ∉ {0,1}); seed 4's frac_negative 0.0734
means **92.7% of its forecasts are positive**, which is a strong directional bias short of a lock.

## D. ECONOMICS, RE-MEASURED ON TWO UNITS

    r0 17.51 h (37.61 ms/dec)   r2 17.91 h (38.38 ms/dec)   mean 17.71 h
    FORWARD REMAINING $193.18 + SPENT ~$41.50 = PROJECTED $234.68
    vs HARD STOP $160 · balance ~$201 · 6.6 DAYS per box

**r0 and r2 are both idle and billing with no work.** r1 is ~1 h from closing.

---

# ADDENDUM 2 — PART 2 ANSWERED: THE EVAL IS EXPENSIVE, NOT BROKEN

## A. A CORRECTION I OWE, STATED PLAINLY

I told the supervisor: *"the probe's 20 ms/dec was 1.88× optimistic on the real workload."*
**That was the wrong comparison and I withdraw it.** It set the probe's MARGINAL rate against the
WHOLE-UNIT rate — precisely the comparison the driver's own manifest says must never be made.

Measured on the production 4090 at n=10,000: **19.851 ms/dec (50.376 dec/s)**, which agrees with
the probe's clean 1024→2048 marginal (20.009 ms/dec) **to 0.8%**. The probe's clean pair was right.

**Finding 10 stands, on its original grounds and now confirmed by direct measurement:**

    shipped headline marginal_seconds_per_decision = 0.009598690  (9.599 ms/dec)
    slope 512 -> 1024   = 0.009598690  <-- EXACTLY the shipped headline
    slope 1024 -> 2048  = 0.020008550  <-- the clean pair, IGNORED
    512-pt wall std 9.172 on mean 15.505 = 59% relative
    measured at n=10,000 on the production card: 19.851 ms/dec
      => the shipped headline is 2.07x OPTIMISTIC

## B. ★ WHERE THE 14.65 HOURS ACTUALLY WENT — AND IT IS A COST-MODEL DEFECT

`n_blocks = 6` (confirmed: the manifest's `grid.n_periods = 35064` matches the primary grid
exactly). The val grid is **7,013 periods = exactly 0.2000× the headline grid**.

| component | hours | share |
|---|---|---|
| headline pass, h=15, 35,064 periods | 7.73 | 52.8% |
| 3 val passes, h=5/15/60, 7,013 each | 4.64 | 31.7% |
| fixed: model load + tokenize 40 symbols | 2.28 | 15.5% |
| **measured total** | **14.65** | |

**The 7.80 h/unit prediction = 1,402,520 / 50.0 dec/s = 7.79 h — the headline pass, exactly, and
nothing else. It omitted 47.2% of the eval leg.**

The eval is **not broken and not mysteriously slow.** The marginal decision rate is exactly what
the probe's clean pair said. **The estimate modeled one of four passes.** The correct forward
model is `eval ≈ 1.6 × headline_pass + 2.3 h fixed`.

## C. B.2 — THE A100 I ACTUALLY RENTED

**$0.7500/h billed** (the supervisor was estimating $0.95), **A100 SXM4 40 GiB** (39.49 GiB
reported by torch), not 80 GiB.

## D. B.3 / B.4 — THE HARDWARE QUESTION, MEASURED ON BOTH CARDS UNDER BOTH CONFIGS

| card | config | ms/dec | reserved | frag | retries | mu_mean / mu_std |
|---|---|---|---|---|---|---|
| RTX 4090 | baseline | **19.851** | 16.605 GiB | 7.700 | 67 | 3.6373e-4 / 3.0115e-3 |
| RTX 4090 | +fix | 19.938 | 11.891 GiB | 2.990 | **0** | 3.6373e-4 / 3.0115e-3 |
| A100 40GB | baseline | 23.788 | 30.859 GiB | 21.955 | 34 | 3.6373e-4 / 3.0115e-3 |
| A100 40GB | +fix | 23.465 | 11.891 GiB | 2.990 | **0** | 3.6373e-4 / 3.0115e-3 |

**`mu_mean` and `mu_std` are identical to all printed digits across both cards and both configs.**
The supervisor's correctness condition holds exactly: the allocator setting moved where tensors
live and changed nothing they contain.

**COST PER DECISION (($/h ÷ 3600) × ms/dec ÷ 1000):**

    RTX 4090 baseline   $0.4622/h   19.851 ms   ->  $2.5486e-06 / decision
    RTX 4090 +fix       $0.4622/h   19.938 ms   ->  $2.5598e-06 / decision
    A100 40GB baseline  $0.7500/h   23.788 ms   ->  $4.9558e-06 / decision
    A100 40GB +fix      $0.7500/h   23.465 ms   ->  $4.8885e-06 / decision

**The A100 costs 1.94× the 4090 per decision.** The 4090 is **1.20× FASTER** in wall-clock and
1.62× cheaper per hour.

**★ I MUST CORRECT MY OWN EARLIER CLAIM: I reported "A100 1.3× faster" from an estimated 4090
rate of ~31 ms/dec that was never measured.** Measured, the 4090 is faster. The operator's
preference for an A100 and my arithmetic were both wrong in the same direction; the 4090 wins on
both axes at once.

## E. THE ALLOCATOR FIX — WHAT IT BUYS AND WHAT IT DOES NOT

`expandable_segments:True` eliminates fragmentation completely on both cards (retries 67→0 and
34→0; reserved 16.6→11.9 and 30.9→11.9 GiB) and **buys no speed on either** (−1.4% A100,
+0.4% 4090 — both inside noise).

**Fragmentation was real and is now measurable, but it was never what cost the time.** The 4090
ran at 19.851 ms/dec while reserving only 16.6 GiB; the A100 ran SLOWER at 23.788 while reserving
30.9 GiB. Memory pressure was not the bottleneck.

**Recommend it anyway** on the narrow ground that it removes an entire failure mode (67 retries
with headroom, and the 22,861 MiB production ceiling) at zero measured cost and provably zero
effect on the numbers. That is a judgement, and it is the supervisor's to take or leave.

## F. WHAT I DID NOT RUN, AND WHY — NO SILENT CAPS

**I did not run 50k or 200k under the fix (~$2.44, ~3.3 h).** The question those points existed to
answer — was the 685× extrapolation safe? — is now answered better by the production unit itself:
1,402,520 real decisions, and the marginal rate at n=10,000 agrees with the probe's clean pair to
0.8%. The baseline curve was already flat 10k→50k (+0.14%) with retries frozen after the 10k phase.
Say the word and I will run them.

---

# ADDENDUM 3 — WHY THE 0.95 THRESHOLD MAY BE INSIDE CLAIM 1

## A. THE THRESHOLD ENCODES AN FSQ CONSTRUCTION PROPERTY THAT BSQ DOES NOT HAVE

    FSQ_LEVELS = (11, 9, 9, 7, 7, 5, 5)        src/trikaal/constants.py:85
    coarse 11*9*9  =  891  ( 9.799 bits)
    fine   7*7*5*5 = 1225  (10.259 bits)
    BSQ    2^10    = 1024  (10.000 bits) per leg

FSQ's code space is a **Cartesian product grid over independently-quantized coordinates**
(`fsq.py:3`: "Each latent coordinate is quantized independently to a small fixed grid"). Near-full
occupancy is a property of that construction — which is exactly what the spec says:

> target ≥ 95% — **FSQ rarely collapses, this confirms it**

That sentence uses 95% as a **confirmation that a known FSQ property holds**, not as a quality bar
a quantizer must clear. **BSQ is binary spherical quantization — 10-bit codes on a sphere, with no
product-grid structure and no equivalent occupancy guarantee.** Applying the threshold to the BSQ
arms tests BSQ for a property only FSQ has by construction.

## B. THE CONSEQUENCE — AND IT LANDS ON THE HEADLINE COMPARISON

Cells 1 and 3 are BSQ; cells 2, 4 and 5 are FSQ. If the threshold is systematically harder for
BSQ, the gate can only ever reject **the baseline arms** — and **claim 1 is "FSQ > BSQ at matched
bits."** A validity gate that is easier for the treatment than for the control sits *inside* the
comparison it is supposed to protect.

**THIS IS A RISK, NOT AN ESTABLISHED FACT, AND I AM NOT ASSERTING IT.** No FSQ production artifact
exists. What is measured is only this:

    c1s0 (BSQ)  fine 888/1024 = 0.8672  REJECTED
    c1s4 (BSQ)  fine 962/1024 = 0.9395  REJECTED
    FSQ         UNMEASURED

If FSQ were also to land below 0.95, the threshold is simply too tight for real data at 26,003
steps and the asymmetry argument does not arise. **Cell 2 seed 0 is one unit — ~17.7 h, ~$8 — and
it is the measurement that separates "threshold too tight" from "threshold asymmetric between the
arms of claim 1."** I am not launching it; it is named here because it is the cheapest thing that
would settle the ruling on evidence rather than on my reading of a spec sentence.

## C. CORRECTION TO MY OWN §6

I wrote that I kept r0 and r2 alive because "they hold trained checkpoints." **Wrong — all three
checkpoints were already banked and verified in Part 1** (18 files, 1.65 GB, predictor/tokenizer/
stage1_state/stage2_state per box). Destroying those boxes would cost only the **provisioned
environment** — toolchain, payload, and the 4.08 GiB lake subset, ~1 h each with a 429 risk on the
lake pull. That is the real and much weaker reason to hold them, and it is worth stating correctly
because it changes the price of the decision.

---

# ADDENDUM 4 — THE RULING IS SETTLEABLE FOR $0.18, AND THE CODE CONTRADICTS ITSELF

## A. ★ THE DIAGNOSTIC'S OWN DOCSTRING FORBIDS THE GATE THAT ENFORCES IT

`src/trikaal/eval/diagnostics.py:75-76`, verbatim, on `cell_codebook_diagnostic`:

> **Non-gating** — reported in every cell eval artifact and the smoke output, **never thresholded**.

`src/trikaal/eval/verdict.py:299` thresholds it at 0.95 and hard-fails the whole 25-unit matrix.

**Two authorities in this repo state opposite things about the same number.** The v1.6.22 fix added
the threshold and never updated the docstring of the function whose output it thresholds — the
class rule, again: the fix corrected the site it was pointed at and left its sibling standing.

Corroborating: the artifact marks `ohlcv_recon.non_gating = true` and
`decode_agreement.non_gating = true`. The `codebook` block carries **no such flag** — yet its
producing function documents it as non-gating.

I am not resolving this. It is the strongest single piece of evidence for the ruling and it is
**internal to the repo**, not my reading of a spec sentence.

## B. THE MEASUREMENT COSTS 23 MINUTES, NOT 17.7 HOURS

`cell_codebook_diagnostic(b_c, b_f, *, v_c, v_f)` takes **token streams and vocab sizes only.**
The AR predictor is not an input. So the number is a **Stage-1 product** — no Stage 2, no eval.

Measured per-stage wall clock from the banked r0 manifest (`throughput` block):

| phase | wall | share of unit |
|---|---|---|
| stage 1 (tokenizer) | 1,168.3 s = 19.5 min | 1.85% |
| tokenize pass (implied) | 215.4 s = 3.6 min | 0.34% |
| stage 2 (AR) | 8,870.2 s = 2.46 h | 14.1% |
| eval leg | 52,745.9 s = 14.65 h | 83.7% |
| **full unit** | **63,037.9 s = 17.51 h** | |

**Stage 1 + tokenize = 1,383.8 s = 23.1 min = $0.178 at $0.4622/h — 46× cheaper than the $8.09
full unit I quoted in Addendum 3.** I withdraw the $8 figure as the price of this question.

`scripts/m6_cell6_stage1_probe.py` is the existing instrument for exactly this shape
(`steps_stage2=0`, "STAGE 1 ONLY — the two numbers are Stage-1 outputs"). Pointing it at
**cell 2 seed 0** yields FSQ coarse/fine utilization directly comparable to cell 1 seed 0 — same
seed, same `ohlcv` arm, **only the quantizer differs.** That is the controlled comparison for
claim 1.

Note on the existing cell-6 receipt: it was produced at `steps_stage1: 3`, `n_symbols_tokenized: 2`
— a smoke run, which is why its utilization read 0.0741/0.1233. **It is not evidence about a
trained FSQ codebook**, and the cell-6 decline should not be re-read as if it were.

## C. NOT LAUNCHED — AND WHY, GIVEN THE MARGINAL COST IS ~$0

r0 is idle and billing $0.4622/h **whether or not this runs**, so the marginal cost of the probe
is effectively zero, and it cannot contaminate anything: Stage 1 only, separate out-dir, no eval
artifact, no verdict clause, HEAD unmoved.

**I have still not launched it.** I have twice told the supervisor that nothing new starts until
the ruling lands, and reversing that unilaterally — even for $0.18 — is precisely the drift these
norms exist to prevent. It is specified, priced, and one word from firing.

## D. ★ CORRECTION TO §A OF THIS ADDENDUM — I OVERSTATED IT

I wrote: *"Two authorities in this repo state opposite things about the same number."* Literally
true, but it implies a symmetry that does not exist, and it could lead a reader to treat the GATE
as the accident. It was not. Dated:

    docstring "never thresholded"   b5a7b5e  2026-07-18  diagnostic introduced, purely diagnostic
    the 0.95 gate                   a974393  2026-08-03  the re-audit fix pass, 16 days LATER

**The gate is the deliberate later decision; the docstring is stale.** `verdict.py:278-281` states
the intent explicitly — *"the codebook diagnostic is REQUIRED per-(cell, seed) and its utilization
is gated at the spec's own >= 95% (design :1859)"*. Nobody gated a number by accident.

**What survives, and it was always the stronger argument:** the gate was introduced **citing
design :1859**, and design :1859 is a sentence about **FSQ** — *"fraction of FSQ codes used …
FSQ rarely collapses, this confirms it"*. The live question is whether that citation supports
applying the threshold to the **BSQ** arms, which the sentence never mentions and whose code
construction carries no equivalent occupancy property.

The stale docstring is **corroborating evidence of documentation lag**, not proof the gate was
unintended. Addendum 4 §A is amended accordingly; §B (the $0.18 measurement) is unaffected.

---

# ADDENDUM 5 — THE THREE RULINGS EXECUTED

## RULING 2 — THE SWEEP. REPORTED, NOTHING PROPOSED.

Instrument: `codebook_scope_sweep.py`, **self-test first** (class rule — a sweep that cannot detect
a BSQ-scoped statement cannot report that there are none):

    SELF_TEST_EXIT=0   6/6 scope cases (FSQ / BSQ / BOTH / UNSCOPED) + negative control

Corpus: the v1 design spec (21 hits), `m6_prereg.md` (15), `m6_design.md` (1).

    TOTAL HITS 37   by scope: {'BOTH': 3, 'FSQ': 6, 'UNSCOPED': 28}
    BSQ-ONLY-scoped statements: 0

### The answer to the question as posed, and it is not "nothing else covers BSQ"

**Something else DOES cover BSQ — and it treats the number as an OUTCOME, not a gate.**

**design :1154** (scope BOTH), verbatim:

> **Key reframing vs BSQ:** because FSQ cannot have *structurally* dead codes (every grid point is
> reachable — §b), the relevant health signal is **effective bpt** (`= log2(global perplexity)`)
> … **The dead-code rate that plagues BSQ is reported for the BSQ ablation arm** and is expected
> to be **0 for FSQ by construction** — itself a result to report (the "no codebook-collapse mode"
> claim, empirically demonstrated).

**design :986** (scope BOTH), verbatim:

> **Why FSQ has NO codebook-collapse mode.** Codebook collapse in **VQ/BSQ** is the failure where
> a learned embedding table has most entries never selected (dead codes) … FSQ has **no learned
> codebook to collapse**: every grid point … is reachable by construction … Usage may still be
> *non-uniform* (some grid cells rarely visited — a data property), but no code is structurally
> unreachable … **We monitor usage as a *health diagnostic* (§f), not as a failure mode to be
> engineered around.**

**design :17** (scope BOTH) — the headline claim, listing "no codebook collapse" among what FSQ
buys over Kronos's BSQ.

**design :1859** (scope FSQ) — the gate's own citation: *"codebook usage = fraction of **FSQ**
codes used over the eval set (target ≥ 95% — **FSQ rarely collapses, this confirms it**)"*.

Remaining FSQ-scoped hits — :859 (unrelated, "collapses them onto one axis"), :1112 (fine-group
underuse), :1149 (effective-bpt table row, target ≈ 20.06), :1920 and :2081 (W&B perplexity vs
V_c=891 / V_f=1225). None states a utilization threshold for BSQ.

### What that means for the two conditions, stated flatly and no further

1. *The sentence at :1859 is about FSQ* — **verified**, and the sweep adds that its neighbours
   :986 and :1154 explain WHY it is FSQ-specific: collapse is described as a **VQ/BSQ** mode that
   FSQ structurally lacks.
2. *No other authority extends it to BSQ* — **NOT what the sweep found.** :1154 covers BSQ
   explicitly. It says the BSQ dead-code rate is **reported for the BSQ ablation arm** and that
   FSQ's rate is **a result to report**. It states a REPORTING obligation. **The sweep found no
   statement, in any of the three documents, setting a MINIMUM utilization for BSQ.**

**PROPOSING NOTHING.** Recording only the shape: the enforced gate is a threshold; the two BSQ-
covering statements in the corpus are reporting obligations; and the corpus contains zero
BSQ-scoped minimums. What follows from that is the writer's and the supervisor's.

## RULING 3 — power_guard READS THE LEVELS. QUOTED.

**LEVELS, not paired differences.** `verdict.py`:

    859-866  irs = [information_ratio(evals[(c, s)]["headline_series"], PRIMARY_H)
                    for s in DSR_SEEDS]                       # per-cell, per-seed IR LEVEL
    876      "ir_range_across_seeds": (max(finite) - min(finite))
    884-887  ranges = [per_cell[c]["ir_range_across_seeds"] for c in cells]
             worst  = max((r for r in ranges if r is not None), default=None)
             trips  = bool(worst is not None and np.isfinite(delta) and worst >= abs(delta))

and the docstring says it in words at 847-849:

> **THE RULE.** For each claimed between-cell ΔIR, take the **WITHIN-cell across-seed IR range** of
> the cells that delta is built from. If that range meets or exceeds |ΔIR| … HALT_ADJUDICATE.

**No paired per-seed difference is computed anywhere in the guard.**

### The paired quantity exists, in the same file, from the same inputs

`verdict.py:942-954` defines `_per_seed_delta(ca, cb)` = `IR(cell_ca, seed) − IR(cell_cb, seed)`
over `DSR_SEEDS`, read from the same `evals[(c, s)]["headline_series"]`. It is already used at
**:956-957** for `pb45` — **the ΔIR(4−5) headline itself** — and `pb42`.

So the guard's basis is a **design choice made in the presence of the paired quantity**, not a
data limitation.

### AND THE PAIRED TRAINING-VARIANCE TERM IS ALREADY IN THE PRIMARY TEST

`paired_bootstrap.py:109-113`:

> ``per_seed_deltas`` (§7 v1.5 item B) is the per-seed contrast {ΔIR_s = IR(a_s) − IR(b_s)}. When
> supplied, **the MDE carries the across-seed TRAINING-VARIANCE term** and uses t-quantiles at the
> Welch–Satterthwaite df — which makes the primary test **HARDER**, by 1.24–1.60× …

So there are **two instruments on the same risk**: the primary test carries across-seed training
variance **on the paired contrast** (the basis your common-mode question is about), and
`power_guard` HALTs on the **within-cell level range** (a different, cruder quantity).

### I WENT LOOKING FOR A CONTRADICTION AND THERE ISN'T ONE

`power_guard`'s rule string asserts *"The tabled MDE contains SCORING noise only … and no
training-variance term"*. **That is TRUE of the tabled value.** `tabled_mde_h15` = 3.518 comes from
`runs_manifest/m6_mde_inputs.json:h15_pooled`, whose inputs are `T=35063`, `T_eff=35002.3`,
`ac1_pooled=0.0009`, `avg_breadth=40.0`, `rho_bar_*` — **all time-series quantities, no training
term**. The v1.5 training-variance term enters at RUN TIME via `pb45.mde_paired`, which the
manifest compares against the tabled value at `verdict.py:1052`
(`mde_paired_exceeds_tabled`). **power_guard's justification is accurate, not stale.**

### WHAT OUR DATA CANNOT SETTLE — CONFIRMED

Your subtlety stands and nothing measured resolves it: **we have two seeds of cell 1 and ZERO
units of cells 4 and 5**, the headline pair. Whether the seed effect is common-mode (paired ΔIR
stable while both levels swing → the guard HALTs on a well-measured effect) or independent
(79.47 genuinely fatal) is **not distinguishable from anything we hold.** Not resolving it by
argument.

## MACHINES

    vastai destroy instance 47479372 -y   ->  "destroying instance 47479372."
    RE-LIST: COUNT 2   r0 running $0.4622/h   r1 running $0.4678/h
             r2 (47479372) PRESENT: False

**r2 destroyed and proven by re-list**, not by the destroy command's exit code — which returned 0
while aborting at a prompt an hour ago.

## RULING 1 — LAUNCHED

Cell 2 (FSQ + ohlcv) seed 0, Stage-1 only, **all 40 pinned symbols** (utilization is
"fraction of codes seen at least once" and rises monotonically with sample size — measuring FSQ on
8 symbols against cell 1's 40-symbol basis would have manufactured the very asymmetry the probe
tests for). `steps_stage2=0`, `money_run=False`, out-dir `/root/m6_cell2_probe`, script at `/root`
so the payload tree is untouched. HEAD unmoved.
