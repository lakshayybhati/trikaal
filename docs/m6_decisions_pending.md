# M6 — the four decisions pending, and what they cost together

**FOR LAKSHAY. Drafted 2026-08-03 (§7 v1.6.20). NOTHING HERE IS IMPLEMENTED.**

Four things need your ruling. They are in one document because **they interact**: two of them
change the same cost line, and one of them cannot be resolved the recommended way unless another
is decided first. Deciding them separately, in the cheap order, forecloses the good outcome on two
of the four.

The supervisor's recommendations are marked **[SUPERVISOR]**. Where the builder disagrees it is
marked and argued.

---

## 0. THE SINGLE RECOSTED TOTAL — top up against this

**The $33–50 figure is now known to price training at 2.6–6.1 %.** Everything below is built on
throughput **measured on a real RTX 4090 at the exact money surface** (`seq_len=512`, `batch=32`;
`runs_manifest/m6_cuda_probe_cell_manifest.json` — Stage 1 16.7859 steps/s, Stage 2 3.2967
steps/s) and the **measured** forced-determinism penalty of **13.175 %** (1.152× slower;
`docs/m6_cuda_probe_report.md`).

| scenario | training GPU-h (25 units) | **worst-case total** |
|---|---:|---:|
| A. leave 2,000 · B1 forced · C-4 binding (retrain fires) | 5.81 | **$34–53** |
| B. leave 2,000 · B1 forced · C-4 dropped | 5.81 | **$32–51** |
| **C. raise to 26,003 · B1 forced · C-4 dropped** ← the recommended pair | 75.48 | **$51–79** |
| D. raise to 26,003 · B1 forced · C-4 binding (retrain fires) | 75.48 | **$70–109** |
| E. raise to 26,003 · B1 unforced · C-4 dropped | 65.53 | **$48–75** |

**TOP UP AGAINST $79** if you take the recommended pair; **$109** if C-4 stays binding.

**Two honesty notes on this table.**
- **The eval leg is inferred, not measured** — it is the approved band minus measured training
  ($30.98–48.69). It cannot be measured without spend: the recipe pins `chunk=512`, which OOM'd
  locally (19.69 GiB KV cache > 20.13 GiB), and the only local datum is a `chunk=64` **floor** on
  different hardware, explicitly labelled *"PENDING the 4090"*.
- **The determinism penalty is applied to TRAINING only.** It was measured on the training path.
  Whether forced determinism also slows eval is unmeasured and is not assumed here.

### ★ A number that has been on the board for weeks is wrong, and it is ours

`docs/m6_c4_kronos_gate_requirements.md` banked the C-4 retrain contingency as *"+1× the training
leg, i.e. **roughly doubling worst-case spend**"* → **~$66–100, or ~$86–130 forced**.

**That is only true if training is about half the spend. Measured, it is 2.6–6.1 %.** The retrain
contingency is **+$1.51–2.32** at the current budget and **+$19.62–30.19** at the raised one. **Not
a doubling.** The banked figure has been quoted in every budget discussion since; it is withdrawn
and replaced by the table above.

---

## 1. THE TRAINING BUDGET — and this is implementing the blueprint, not exercising discretion

**Full case: `docs/m6_training_budget_decision.md`. Receipt: `runs_manifest/m6_training_budget.json`.**

### Finding

`steps_stage1 = steps_stage2 = 2000` is the design spec's **G1 overfit-a-single-batch SMOKE GATE**
threshold (`:1762-1763`, *"within ≤ 2000 steps"*). It entered in commit `64f728c` **in the same
dataclass literal as `seeds = (0,1,2)` and `seq_len = 128`** — the two values later caught as the
C-6 train/eval split-brain. Those had pinned counterparts and were corrected; the step budget had
none, so it survived, and the C-18 fix then **pinned** it.

### Mechanism

At 2,000 steps each cell sees **32,768,000 tokens = 1.54 tokens/param = 7.69 % of compute-optimal
= 0.108 epochs** over the 304,625,181-bar lake. **A flat ΔIR(4−5) cannot distinguish "microstructure
does not help" from "we stopped before it could."** That is the false-NULL failure mode already
ruled Tier-1 blocking for C-10, at the scale of the whole experiment.

### The authority question — corrected

**This is not discretion. The design spec states the budget** (`:1912`, `:1924`): Stage-2 *"1–3
passes over a ≤1B-bar corpus"* at ≈0.5M tokens/step with early-stop on val-NLL saturation. docs/ENGINEERING.md
makes the spec a source of truth — *"if anything here conflicts with the spec, the spec wins."*

> **Raising the budget IMPLEMENTS the blueprint. Leaving 2,000 is a standing, undisclosed DEVIATION
> from it — on the one parameter that decides whether a NULL is interpretable.**

Corroboration: the designed 1–3 passes over our lake is **14.3–42.9 tokens/param**, which brackets
the ~20 compute-optimal point. **The design's budget is compute-optimal; the code's is 7.69 % of it.**

**And the schedule confirms the budget was never reconciled:** the spec sets the Stage-2 **eval
interval at every 5,000 steps** against a 2,000-step budget — **the designed schedule's first
evaluation never fires.** The saturation early-stop therefore has nothing to fire on; it is not
merely unimplemented, it would be **inert at this budget even if written**.

### Options

| | steps | % optimal | epochs | Δ cost | consequence |
|---|---:|---:|---:|---:|---|
| A | 2,000 | 7.7 % | 0.108 | $0 | a NULL is uninterpretable; undisclosed spec deviation |
| B | 10,000 | 38.5 % | 0.538 | +$5.2–8.1 | most of the knee; still below the spec's 1–3 passes |
| **C** | **26,003** | **100 %** | **1.399** | **+$15.73–24.20** | the spec's own budget; a NULL becomes interpretable |
| D | early-stop, cap 40,000 | ≤154 % | ≤2.15 | ≤+$24.9–38.3 | what the spec literally says — but needs new machinery |

### Publishable sentences

> **★ PARAMETER-COUNT NOTE (added 2026-08-17).** The candidate sentences below say *"the compute-optimal point for a 21.3M-parameter model"*. **Both halves are wrong against the realized artifact** and §7.10 of the paper retracts them: 21,301,248 is the backbone EXCLUDING the MTP heads, the shipped total is **31,795,200**, and against that total the ratio is **13.40 tokens per parameter**, not 20.00 and not compute-optimal. These sentences are LEFT AS WRITTEN because this document records the options that were on the table at a decision point; editing a recorded option into correctness would destroy the record of what was actually offered.


- **If C:** *"Each cell was trained for 26,003 steps at 32×512 tokens/step — 1.40 passes over the
  304.6M-bar corpus, ≈20 tokens per parameter, the compute-optimal point for a 21.3M-parameter
  model."*
- **If A:** *"Each cell was trained for 2,000 steps — 0.108 passes, ≈1.5 tokens per parameter,
  ≈7.7 % of the compute-optimal budget for a model of this size."* Publishable and honest — and it
  invites a referee to ask whether a null measures the tokenizer or the budget. We would have no
  answer.

**[SUPERVISOR] recommends C.** **The builder agrees**, and adds: option D is what the spec
*literally* says, but it is the only option that requires new machinery — and its trigger cannot
fire below ~25,000 steps anyway (eval interval 5,000 × patience 4), so **C is a precondition for D,
not an alternative to it.**

---

## 2. C-4 / G-§8.C.3 — the external validation gate

**Full case: `docs/m6_c4_kronos_gate_requirements.md`. Receipt: `runs_manifest/m6_c4_resolvability.json`.**

### Finding: unexecutable as specified, on three independent counts

1. **The threshold is ambiguous by 2.4×.** Paper Table 2 gives *two* `Kronos_small` RankICs —
   **0.0254** (price-series) and **0.0622** (return-forecasting) → thresholds 0.02159 vs 0.05287.
   The spread is far larger than the 15 % band the gate is made of, and the prereg does not say
   which.
2. **Both are Shanghai Stock Exchange, 15-minute bars.** Equities, not crypto; 15-minute bars, not
   our 1-minute bars. **The gate's two requirements cannot both hold**: run Kronos on our slice and
   you produce a *new* number, not the published one; use the published number and you are
   comparing crypto-1m against equities-15m.
3. **Running the weights requires Kronos's model code** (the card's own loader is
   `from model import Kronos, KronosTokenizer`), and invariant 8 permits Kronos *weights* in the
   eval harness but Kronos *code* nowhere. *(The supervisor has ruled that a loader written from
   the published paper is self-written code and therefore invariant 8 operating as designed. The
   builder accepts this, with one price recorded in §5 below.)*

**The band itself is fine** — measured N_eff 1,401,637 of 1,402,560 raw (deflation 0.999),
SE(RankIC) 0.00084, band/SE 4.51 and 11.05. Sampling noise is not the obstacle; the builder's
earlier speculation that it might be is withdrawn.

**Licence is not an obstacle:** Kronos is MIT (`Copyright (c) 2025 ShiYu`), the repos are public and
ungated, an unauthenticated pull works, and **no token should be issued.**

### The proposed substitution, and the builder's attack on it

**[SUPERVISOR] proposes:** a baseline *architecturally matched* to Kronos_small (invariant 4,
already true) and *trained to its own measured val-NLL saturation* is defensible without an external
number — saturation is self-certifying against the failure the gate was built to catch, and needs no
Kronos code, no equities data, no invariant-8 amendment.

**IT SURVIVES FOR ONE FAILURE MODE AND FAILS FOR THE OTHER, AND THE OTHER IS THE ONE THAT TOUCHES
OUR SECOND CLAIM.**

- **✅ It does cover "Cell 1 is weak because it was under-trained."** That is the most *likely*
  cause, and saturation is a direct, self-certifying answer to it: more training would not have
  helped. Combined with the G1 overfit-a-single-batch gate (which proves the machinery can learn at
  all), it is a genuine control.
- **❌ It does NOT cover "Cell 1 is weak because OUR BSQ TOKENIZER is worse than the reference BSQ
  tokenizer" — and Kronos-small *is* a BSQ model.** Cell 1 is BSQ+OHLCV. The §5 NULL-fallback
  claims **IR(2) − IR(1)**, which is *FSQ minus BSQ*. **The external gate was, in effect, the only
  control anchoring our BSQ implementation to a reference BSQ implementation — and a
  saturated-but-weak BSQ tokenizer inflates our own FSQ claim by exactly the amount it is weak.**
  Saturation cannot see this: a poorer token stream carries less predictive information, the AR
  converges to a worse val NLL, and saturation certifies "converged" — which is true, and useless.
- **This is not hypothetical in this project.** The v6 canary measured the AR extracting **zero
  nats** from a planted conditional its tokenizer provably encoded — the tokenizer→AR interface
  finding. We have direct evidence that a tokenizer can bottleneck an AR while everything looks
  healthy.
- **Two smaller residuals to name:** the BSQ arms realize **69,632 fewer parameters** (0.327 %,
  C-19) — small, but it sits in the baseline arm specifically; and saturation cannot distinguish
  "converged at capacity" from "flattened because something is broken", so it is only meaningful
  beside the G1 gate and a sane loss level.

### The control stack, corrected — it is five rows, not three

| control | answers | status |
|---|---|---|
| **bpt parity** (G-parity, gated) | is the BSQ arm handicapped on *information capacity*? | **ALREADY CONTROLLED** — 19.5–20.5 band, verified 20.0578 vs 20.0000, \|Δ\| 0.058 bits, tokenizer params 0.25 % apart |
| **recon MAE** (`ohlcv_recon`, REQUIRED per-(cell,seed)) | is the BSQ reconstruction competitive at matched bits? | measured — but the project's own eviction finding says recon fidelity ≠ predictive value |
| **saturation** | was Cell 1 trained enough? | needs ≥25,000 steps to exist at all (decision 1) |
| **codebook health** | did our BSQ **collapse**? | **SPECIFIED, NOT ENFORCED — see below** |
| *(nothing)* | is our BSQ **subtly suboptimal at matched capacity**? | **UNCONTROLLED — this is the disclosure** |

**Attack on the codebook-health control, before agreeing with it.** `diagnostics.codebook_usage`
computes utilization, Shannon entropy of the empirical token distribution, efficiency and
perplexity. **Every one of those is a property of the MARGINAL DISTRIBUTION OF CODE IDS. None
involves the future, the target, or predictive information.** A tokenizer that assigned codes by
hashing noise would score ~100 % utilization and maximal perplexity. **So it discriminates
COLLAPSED from NON-COLLAPSED. It does not discriminate CRIPPLED from COMPETENT** — it is a
degeneracy check, not a quality check, and it is the project's own degeneracy rule inverted (a
maximally-spread input scores perfectly for free, exactly as a collapsed one does on an agreement
statistic). This is also the project's own eviction finding: capacity is allocated by variance, and
utilization measures *that* the bits were spent, never *what on*.

**And it cuts both ways, which is worth stating because it is not obvious.** Our FSQ claim is
explicitly *"no codebook-collapse failure mode"*. So if our BSQ arm is **healthy**, IR(2)−IR(1)
**cannot be attributed to collapse avoidance** — which narrows the FSQ claim rather than validating
the baseline. If our BSQ **collapses**, the comparison is confounded by our own implementation.
**Neither outcome establishes that our BSQ is competitive.**

### Options

| | what it is | cost | what it leaves uncontrolled |
|---|---|---|---|
| A | keep binding as written | **unexecutable** | — |
| B | re-specify: run Kronos on OUR slice, compare Cell 1 to *that* | invariant-8 ruling + implementation | nothing much — but it is real work and D moots the metric cross-check |
| **C** | **drop as binding; substitute saturation; disclose the residual** | $0, removes the retrain contingency (−$20–30 of worst case) | **our BSQ tokenizer vs a reference BSQ tokenizer — which is the confound on IR(2)−IR(1)** |
| D | keep the published figure, name the row, disclose the domain gap | $0 | almost everything — a crypto-vs-equities RankIC ratio is not a baseline check |

### Publishable sentences

- **If C:** *"Cell 1 was trained to measured validation-NLL saturation on the study corpus at
  architecture matched to Kronos-small. We do not externally validate it against published
  Kronos-small figures: the two published RankICs for that model differ by 2.4× and both are
  measured on Shanghai Stock Exchange 15-minute equity bars, a market and frequency outside this
  study. **We therefore cannot exclude that our BSQ baseline is weaker than a reference BSQ
  implementation, which would inflate the FSQ-vs-BSQ comparison reported in §5.**"*
- **If A/B:** *"Cell 1's RankIC reached X, ≥0.85× published Kronos-small on a common slice."* —
  currently not obtainable.

**[SUPERVISOR] recommends C.** **The builder agrees with the choice and disagrees with the word
"substitute":** it is a *narrower, different* control, not a replacement, and the bolded sentence
above must ship with it. Calling it a substitute would claim coverage we do not have — of the one
confound that acts on our second claim.

---

## 3. C-3 — the clause-5 unit fix

**Unchanged. Full case already drafted: `docs/m6_c3_amendment_decision.md`.** One token
(`h` → `PRIMARY_H`), $0, tightens the bar by 9.53 % on the calibration fixture, and the window for
it being a *pre-registered* correction rather than a post-hoc one **closes the moment we spend**.

**[SUPERVISOR] recommends: amend, pre-data, both bases reported.** The builder agrees.

---

## 4. B1 — invariant 7 / forced determinism

**Unchanged. Full case already drafted: `docs/invariant7_amendment_decision.md`.** The measured
penalty is **13.175 %** (1.152×), not the illustrative 1.3× that was double-counted.

**[SUPERVISOR] recommends: pay it.** The builder agrees, and notes the interaction: **the cost of
saying yes to B1 scales with decision 1.** At 2,000 steps it is +0.77 GPU-h (+$0.20–0.31); at
26,003 it is +9.95 GPU-h (**+$2.59–3.98**). Still cheap — but decide 1 first so you are pricing the
real number.

---

## 5. HOW THEY INTERACT — the reason this is one document

1. **Decision 1 is a PRECONDITION for the recommended resolution of decision 2 — and BOTH stages'
   early-stop machinery is inert at the pinned budget, not just one.**

   | stage | eval interval | patience | steps needed before it can fire | at the pinned 2,000 |
   |---|---:|---:|---:|---|
   | Stage 1 (`:1856-1857`) | 2,000 | 5 | **≥12,000** | fires **exactly one** interval |
   | Stage 2 (`:1914`, `:1924`) | 5,000 | 4 | **≥25,000** | fires **none** |

   **Approving the cheap budget silently forecloses the saturation control, and therefore
   forecloses option 2C.** This is the single most important sentence in this document.
2. **Decision 2 changes what you top up against.** Dropping C-4 as binding removes the retrain
   contingency: **$109 → $79** worst case at the raised budget.
3. **Decision 4's price is set by decision 1** — 13× larger once the budget rises, though still
   small in absolute terms.
4. **Decision 3 is logically independent of all three, but time-ordered ahead of all of them.**
   Its entire argument is that it is being made *pre-data*. It is the only one that gets strictly
   worse by waiting, and it costs $0.
5. **The builder's price on the invariant-8 ruling, recorded for completeness:** with *vendored*
   code, a mismatch against published numbers means our metrics are wrong. With *reimplemented*
   code, a mismatch means our metrics **or** our reimplementation is wrong — two unknowns, one
   equation. The reimplementation route is legitimate and it weakens the diagnostic value of the
   very steps it enables. Moot under 2C.

**Suggested order: 3 (free, time-critical) → 1 (unlocks 2) → 2 → 4 (price now known).**

---

## 6. What is NOT in this document

No budget was set, no gate was dropped, no invariant was amended, no weights were pulled. The only
thing that changed in code is a **provenance warning** on `PINNED_STEPS_STAGE1/2` recording that
the value is a smoke-test default rather than a designed budget — visibility, not a decision.


---

## 7. ADDENDUM (§7 v1.6.22) — three more items, all Lakshay's

> **RENUMBERED, AND THE RENUMBERING IS THE RECORD (2026-08-04).** This header read
> **`§7 v1.6.21`**, a tag that **has never existed in the amendment log** — the log runs
> v1.6.20 → v1.6.22 and `v1.6.21` appears nowhere in `docs/m6_prereg.md`. The SUBSTANCE
> below did land: the IR(2)−IR(1) withdrawal is in the **v1.6.22** entry (`m6_prereg.md:606`,
> the withdrawal sentence at **`:646`** — *"covers all three marginals with a BSQ arm
> (IR(2)−IR(1), IR(4)−IR(3), IR(3)−IR(1))"*). So the decision is dated and binding; only
> the pointer was dangling. **Corrected here rather than silently rewritten**, because a
> ruling document that cites a tag which does not exist is the same class as prose that
> describes a gate which does not fire — and the next reader needs the failure mode, not
> just the right number.

### 7.1 ★ IR(2)−IR(1): withdraw it as a CLAIM — and it is worse than stated

**[SUPERVISOR] recommends withdrawing IR(2)−IR(1) as a claim and reporting it descriptively with
all three confounds named** (BSQ quality uncontrolled at matched capacity, the 69,632-param C-19
gap, no external anchor). **The builder agrees, and strengthens the argument in one direction and
corrects its scope in another.**

**STRONGER THAN "THREE CONFOUNDS": the FSQ leg is the ONLY claim in this design with no internal
control, and its sole external control has just been found unexecutable.** Cell 5 exists because
the micro claim needed one — you can destroy information while holding capacity fixed. **You cannot
construct a "placebo BSQ"**: a BSQ tokenizer with our implementation quality but no quantizer
benefit is not a thing, because the quantizer *is* the implementation. And recon MAE cannot serve
as the confound detector, because *reconstructing better at matched bits is the FSQ result itself* —
the treatment and the control would be the same measurement.

**SCOPE CORRECTION — it touches THREE reported marginals, not one.** The supervisor wrote that the
confound lands "entirely on IR(2)−IR(1)". The prereg §5 reports **four** 2×2 marginals, and **three
have a BSQ arm on one side**:

| marginal | arms | affected? |
|---|---|---|
| IR(2) − IR(1) | FSQ vs **BSQ** | **YES** |
| IR(4) − IR(3) | FSQ vs **BSQ** (under micro) | **YES** |
| IR(3) − IR(1) | **BSQ** vs **BSQ** (micro marginal under BSQ) | **YES — both arms carry it** |
| IR(4) − IR(2) | FSQ vs FSQ | no |

**The headline ΔIR(4−5) is safe** — both arms are FSQ, as the supervisor says. But the withdrawal,
if taken, must cover the marginals as a class, not one line of the table.

**The cost, stated plainly:** IR(2)−IR(1) is the consolation prize if the primary returns NULL —
and it is a consolation prize we *know* is biased in our own favour, reached for exactly when the
primary has failed. Same principle as C-3: we do not keep a favourable defect because it was
written down first. **A NULL primary is pre-committed and publishable alone.**

**One clarification the wording needs:** "withdraw as a claim" must not be read as "do not report".
The recommendation is the primary's own Band-B treatment — *report it with CIs, claim nothing* —
not deletion.

### 7.2 THE EARLY-STOP CONFLICT — two sources of truth disagree

An outside reviewer proposed using the spec's early-stop as the actual stopping rule rather than a
fixed step count. **That conflicts with the M6 design, and the conflict is real, not a
misreading.**

- **`m6_design.md:18`:** *"All cells share … the same training draw … the only varied factors are
  {quantizer} × {input arm}."*
- **The v1 spec (`:1912`, `:1924`):** early-stop on val-NLL saturation, patience 4.

**Early stopping is DATA-DEPENDENT**, so it yields 25 different training budgets, and ΔIR(4−5)
would then confound *"micro helps"* with *"cell 4 trained longer"*. **That is the C-12 class —
a capacity/exposure asymmetry between treatment and placebo — reintroduced deliberately into the
PRIMARY.**

| option | what it gives | what it costs |
|---|---|---|
| **Fixed matched budget** for all 25 units, with per-cell val-NLL saturation **measured and reported as a required diagnostic** | the matched design is preserved; you still learn where each cell saturated | if a cell saturates early, the extra steps are wasted compute (cheap — training is 2.6–6.1 % of spend) |
| **Early-stop per cell** | each cell trains exactly as long as it needs | **the stopping point varies with the treatment** — ΔIR(4−5) becomes uninterpretable |

**[SUPERVISOR]'s position, which the builder shares: FIXED matched budget, saturation measured and
reported.** You get the information without letting the stopping point vary with the treatment.
**This is a dated reconciliation between two sources of truth and needs Lakshay's signature — not a
silent winner.**

### 7.3 A FIGURE THAT DOES NOT EXIST, RECORDED SO IT CANNOT PROPAGATE

An outside reviewer has quoted a **"$77 measured"** total for this run. **`grep` across `docs/` and
`runs_manifest/` returns ZERO occurrences.** No such measurement exists in this repository. The
only measured cost component is **training** (5.04 GPU-h unforced, $1.31–2.02); **the eval leg —
94–97 % of the spend — has never been measured at the pinned geometry.** Any total quoted today,
including the $51–79 and $70–109 in §0, is *training measured + eval inferred*. Recorded here
because an unsourced number that sounds measured is exactly how a false anchor enters a budget
discussion — the pipefail lesson, in a spreadsheet.
