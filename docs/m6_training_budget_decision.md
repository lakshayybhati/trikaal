# The M6 training budget — a decision for Lakshay (drafted 2026-08-03, §7 v1.6.19)

**STATUS: DRAFTED AND HELD. NOT IMPLEMENTED.** Supervisor-found. The builder tried to refute it
and could not — the attempt made it stronger. **This is a BUDGET decision, not a specification
change**, because no prereg constant states the training budget; it therefore does not reopen the
freeze and it is Lakshay's category.

---

## 1. In plain terms

**We are about to spend $33–50 training each model for about 12 minutes and then evaluating it for
about six hours.** The training budget is the step count from a smoke test that checks the model
can overfit one batch. Nobody chose it for the real run.

At that budget each model sees **0.108 epochs** of a 304-million-bar dataset. **A flat ΔIR(4−5)
would not distinguish "microstructure does not help" from "we stopped before it could."**

---

## 2. The finding, and the refutation attempt that failed

**The claim:** `steps_stage1 = steps_stage2 = 2000` is a smoke-test default that reached the money
path unexamined.

**I tried to refute it three ways. All three failed, and the third made it worse.**

| refutation attempted | result |
|---|---|
| "2000 was deliberately sized for M6 somewhere in design/prereg/roadmap" | **NO.** The only occurrence of 2000 as a *step budget* in the design spec is `:1762-1763`, the **G1 overfit-a-single-batch SMOKE GATE** threshold: *"within ≤ 2000 steps"*. |
| "the commit that introduced it shows a sizing argument" | **NO — and it is worse.** `steps_stage1: int = 2000` entered in `a4d242e` (2026-07-04) **in the same dataclass literal as `seeds = (0,1,2)` and `seq_len = 128`** — the two values C-6 later identified as the train/eval split-brain. Those two had pinned counterparts and were corrected. The step budget had none, so it survived; **§7 v1.6.15 C-18 then pinned it, and there is now a mutation KAT defending a rehearsal value.** |
| "the design never specified a budget, so 2000 is as good as anything" | **NO, AND THIS IS THE STRONGEST FORM OF THE FINDING.** The design **does** specify one. |

### The design specifies a budget. The code does not implement it.

From `docs/superpowers/specs/2026-06-18-trikaal-v1-design.md`:

| | designed (spec) | implemented (code) | shortfall |
|---|---|---|---|
| Stage-2 epochs | **1–3 passes over a ≤1B-bar corpus** (`:1912`) | **0.108 epochs** | **9–28×** |
| target global batch | **≈ 0.5M tokens/step** (`:1912`) | 32 × 512 = **16,384 tokens/step** | **30×** |
| stopping rule | **early-stop on val-NLL saturation**, patience 4 (`:1924`) | fixed 2,000 steps | — |
| Stage-1 | ~20 epochs over a ≤150M-bar subsample ≈ **1–2B bar-reconstructions**, 3–6 h (`:1966`) | 32.8M bar-reconstructions | **30–60×** |

**So the supervisor's finding is right, and the honest version is sharper than "never designed":
it WAS designed, and the implementation silently substituted a smoke-test constant.** That is worse
than an unexamined default, because a reader of the spec has every reason to believe the budget
exists.

**One striking corroboration:** the designed Stage-2 budget (1–3 passes over our 304.6M-bar lake =
305M–914M tokens) is **14.3–42.9 tokens per parameter**, which brackets the ~20 tok/param
compute-optimal point. **The design's budget is compute-optimal. The code's is 7.69% of it.**

---

## 3. Why it outranks C-4

The auditor's C-4 concern is a **false-positive** channel: a weak Cell 1 manufactures a positive
FSQ result. This is a **false-NEGATIVE** channel, on the primary claim: an under-trained cell 4
cannot express microstructure information it was never given the compute to learn, and ΔIR(4−5)
goes flat for a reason that has nothing to do with microstructure.

**That is the same failure mode ruled Tier-1 blocking for C-10** — *"our own scientific asset
silently converting itself into a false NULL"* — at the scale of the entire experiment rather than
one gate.

---

## 4. The measured cost split — the part that changes the decision

**Training throughput is MEASURED on a real 4090 at the exact money surface** (`seq_len=512`,
`batch_size=32`, `max_len=576` — `runs_manifest/m6_cuda_probe_cell_manifest.json`):
Stage 1 **16.7859 steps/s**, Stage 2 **3.2967 steps/s** → **0.362906 GPU-seconds per step per unit**.

> **TRAINING ALL 25 UNITS AT THE CURRENT BUDGET COSTS 5.04 GPU-HOURS = $1.31–2.02.**
> **That is 2.6–6.1 % of the approved $33–50. Eval is the other ~94–97 %.**

**We are spending ~95 % of the budget evaluating a model we trained for 3 % of it.** The
supervisor's intuition was right and is now measured rather than reasoned.

**The eval leg cannot be costed without spend, and that is a precise statement, not a hedge:** the
recipe pins `chunk=512`; that **OOM'd on local hardware** (19.69 GiB KV cache > 20.13 GiB limit),
and the only local datum is a `chunk=64` **floor** on `mps` — different hardware, explicitly
labelled *"the operating rate is PENDING the 4090"*
(`runs_manifest/m6_eval_throughput_expectation.json`). Under the standing ~8×-local-variance rule
it may not be quoted.

---

## 5. The budget table — a curve, not a number

All figures from the measured 4090 rates above, 25 units, vast 4090 spot **$0.26–0.40/hr**.

| steps | tokens | tok/param | % compute-optimal | epochs over the lake | GPU-h (25 units) | $ @0.26 | $ @0.40 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **2,000** | 32,768,000 | 1.54 | **7.7 %** | 0.108 | 5.04 | **1.31** | **2.02** |
| 5,000 | 81,920,000 | 3.85 | 19.2 % | 0.269 | 12.60 | 3.28 | 5.04 |
| 10,000 | 163,840,000 | 7.69 | 38.5 % | 0.538 | 25.20 | 6.55 | 10.08 |
| 20,000 | 327,680,000 | 15.38 | 76.9 % | 1.076 | 50.40 | 13.10 | 20.16 |
| **26,003** | 426,033,152 | **20.00** | **100 %** | 1.399 | 65.53 | **17.04** | **26.21** |
| 40,000 | 655,360,000 | 30.77 | 153.8 % | 2.151 | 100.81 | 26.21 | 40.32 |

**The headline number for the decision:**

> **Going from 7.7 % to 100 % of compute-optimal — a 13× training budget — costs +$15.73 to
> +$24.20. Total run: $33–50 → ~$49–74.**

Because eval dominates, **the training budget is cheap to fix.** A ~50 % increase in total spend
buys a 13× increase in training, and lands the run on the budget the design specified.

**26,003 steps also lands inside the design's own "1–3 passes" band** (1.40 epochs) — so this is
not a new choice; it is implementing the one already written down.

---

## 6. The options

| option | what it costs | what it buys | what it risks |
|---|---|---|---|
| **A. Leave at 2,000** | $0 | nothing | a flat ΔIR(4−5) is **uninterpretable** — cannot separate "no effect" from "no training". The paper's null would be unpublishable as a null. |
| **B. 10,000 steps** (38.5 % optimal, 0.54 epochs) | **+$5.2–8.1** | most of the curve's knee, still under one epoch | still below the design's 1–3 passes; a null is still arguable |
| **C. 26,003 steps** (compute-optimal, 1.40 epochs) | **+$15.7–24.2** | the design's own budget; a null becomes interpretable | ~50 % more total spend |
| **D. Early-stop on val NLL, cap at 40,000** | ≤ +$24.9–38.3 | what the spec actually says (`:1924`) | needs a stopping rule wired and pinned — that IS new machinery |

**The builder's view, for what it is worth on a decision that is not his: C.** It is the design's
own number, it makes the pre-committed NULL outcome interpretable, and at +$16–24 against a run
whose eval leg already costs $31–48 it is the cheapest interpretability we can buy. **A is the
only option that risks spending $33–50 on an experiment that cannot answer its own question.**

---

## 7. Both publishable sentences

- **If raised (C):** *"Each cell was trained for 26,003 steps at 32×512 tokens/step — 1.40 passes
  over the 304.6M-bar corpus, ≈20 tokens per parameter, the compute-optimal point for a
  21.3M-parameter model."*
- **If left (A):** *"Each cell was trained for 2,000 steps — 0.108 passes over the corpus, ≈1.5
  tokens per parameter, ≈7.7 % of the compute-optimal budget for a model of this size."*

The second sentence is publishable and honest. **It is also a sentence that invites a referee to
ask whether a null result measures the tokenizer or the budget** — and we would have no answer.

---

## 8. What was done without a ruling, and what was not

**DONE (visibility only, as authorised):** `conformance.PINNED_STEPS_STAGE1/2` now carries a
provenance warning naming the smoke-gate origin, the commit, the designed budget it displaced, and
the resulting 7.69 % figure — for as long as that remains true.

**NOT DONE:** the value is unchanged, the pin is unchanged, the mutation KAT is unchanged. **No
budget was set. That is Lakshay's.**
