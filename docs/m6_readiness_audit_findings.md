# M6 readiness audit — findings register (EXTERNAL, relayed)

**STATUS: PARTIAL. This file is honest about what it does and does not contain.**

The audit was performed by an independent reviewer and relayed to the builder through the
supervisor, in chat. **It was never persisted to the repository**, which is why the builder could
not answer "were S-2..S-4 promoted, refuted, or silently dropped?" two tranches ago. This file
exists to end that, and it is created with a **named gap** rather than a reconstruction.

**THE RULE FOR THIS FILE: nothing here is paraphrased into the auditor's voice.** Each entry is
marked with its provenance. Where only a one-line summary reached the builder, that is what
appears — a fuller-sounding entry would be the builder writing the audit, which is precisely the
failure this file exists to prevent.

| provenance tag | meaning |
|---|---|
| **VERBATIM** | the auditor's words, unedited, as relayed |
| **SUMMARY-ONLY** | the one-line index entry is all that ever reached the builder |
| **PARTIAL** | an opening fragment is verbatim; the body did not reach the builder |
| **MISSING** | the finding is named in the index but no text reached the builder |

**★ OUTSTANDING: the full C-1..C-9 and C-11..C-20 bodies. Please paste them; this file has slots
for them and the triage below references them by ID.**

---

## Tier index, as relayed (SUMMARY-ONLY unless noted)

> C-10 micro-legibility gate passes when all six dims are thin; head-of-stream 150k sampling.
> C-11 determinism hardcoded off with no config knob (B1).
> C-12 placebo destroys micro<->OHLCV contemporaneous dependence, not only temporal alignment;
>      cell 5 carries lambda=3.0 on now-independent noise; no cell matched on both axes.
> C-13 harness.py inside the M6 path via xsection.py while documented as frozen M5-only.
> C-14 eval resume reuses artifacts without binding them to the checkpoints that produced them.
> C-15 causal gate exercised only at non-production feature parameters (target_valid vacuous
>      under the production config).
> C-16 modelled impact term is a hardcoded q_over_adv=1e-3 constant.
> C-17 stale numerics, INCLUDING the clause-5 rule string persisted into the verdict manifest.
> C-18 interpreter unpinned; Stage-1/2 step budget a bare code default; estimator a function
>      default.
> C-19 BSQ cells realize 21,231,616 params vs FSQ 21,301,248 while docs quote one number.
> C-20 headline is one calendar year; embargo_flatness never called; claims-sweep receipt stale.
> S-1..S-4 the four SUSPECTED items — were any promoted, refuted, or silently dropped?

Also relayed verbatim for **C-3**:

> C-3 (var_sr mixes three time-unit Sharpes — verify the auditor's 3.46x span and reproduce the
> 5.8056e-4 hand-calculation before I rule)

And the audit's own process instruction, **VERBATIM**:

> PROCESS: this audit found what two of us missed across weeks. Treat every finding as true until
> you have disproved it in code, not the reverse. Report per finding: CONFIRMED with your own
> reproduction, REFUTED with evidence, or NEEDS-RULING. Do not batch a fix with a disagreement.

---

## C-10 — **PARTIAL** (opening verbatim; body did not reach the builder)

> C-10 · The micro-legibility hard stop passes when it measures nothing
> `src/trikaal/train/gates.py:193-195,202-203`
> **CLAIMED:** …

---

## S-1 — **SUMMARY-ONLY / promoted**

Recorded that the *structure* of the single-bar decode gap is certain and the *magnitude* is not:
a toy an order of magnitude from canonical says little about the effect size on the instrument
that will actually be used. **Promoted into C-1.**

## S-2 — **VERBATIM**

> `power_guard` is nan-fragile in the same way as C-2. With non-finite per-seed IRs,
> `ir_range_across_seeds` is `None`, `worst` is `None`, and `trips` is False; with a NaN Delta it
> reports `seed_spread_exceeds_claim: false`. Reproduced in probe 1, but I could not construct a
> realistic artifact set that reaches it, so I am not claiming it fires in practice.

## S-3 — **VERBATIM**

> `expectation` is a plug-in, not the pre-registered quantity. `decode_latent(E[z])` !=
> `E[decode_latent(z)]` for a nonlinear decoder, and `p_f` is conditioned on the argmax coarse
> rather than marginalized. Both are disclosed in `predict.py:8-22` and were adjudicated in
> v1.4.7 item 4 (retained because mc@32's noise floor exceeds the disagreement). I flag it only
> because C-1 compounds it: the delta-method error is evaluated in the same out-of-distribution
> single-bar regime.

## S-4 — **VERBATIM**

> provenance of two numerical routines. `dsr.norm_ppf` is Acklam's published rational
> approximation (attributed) and `tdist._betacf` is the Numerical Recipes modified-Lentz `betacf`
> in structure. These are standard math, not research surface, and both are cited — but NR code
> carries a restrictive licence, and a released artifact should not transcribe it.
> `attention.apply_rope`/`_rotate_half` is the standard LLaMA/HF RoPE formulation; unavoidable and
> not a violation, but worth one attribution line given invariant 8's strength.

---

## Disposition — the builder's verdicts, with receipts

Findings are closed against the **prereg §7 amendment log**, which carries the full reasoning.
This table is an index, not a substitute.

| ID | verdict | where it is closed |
|---|---|---|
| C-1 | REPORTED — INDETERMINATE at n=3; now a REQUIRED real-cell disclosure | §7 v1.6.7/8/11 |
| C-2 | CONFIRMED, FIXED — blind HALT gate | §7 v1.6.1 |
| C-3 | CONFIRMED, outcome-material — **amendment DRAFTED and HELD** | §7 v1.6.13/14, `m6_c3_amendment_decision.md` |
| C-4 | REPORTED — G-§8.C.3 requirements + retrain contingency costed | §7 v1.6.9, `m6_c4_kronos_gate_requirements.md` |
| C-5 | CONFIRMED, FIXED — the money driver, A1–A9 | §7 v1.6.5/6 |
| C-6 | CONFIRMED, FIXED — train/eval seed+seq_len split-brain | §7 v1.6.4 |
| C-7 | CONFIRMED, FIXED — unguarded verdict thresholds | §7 v1.6.2 |
| C-8 | **REFUTED as stated**; the real finding is OC-1 (S=3 unreachable) | §7 v1.6.12/13 |
| C-9 | CONFIRMED, FIXED — §3/§3a body carried superseded numerics | §7 v1.6.12 |
| C-10 | CONFIRMED, FIXED — both legs; materiality measured on the real lake | §7 v1.6.15, `m6_c10_micro_density.json` |
| C-11 | CONFIRMED — **is B1**, drafted, awaiting Lakshay | `invariant7_amendment_decision.md` |
| C-12 | REPORTED — M1 adopted as a REQUIRED disclosure | §7 v1.6.9, `m6_c12_placebo_mechanism.md` |
| C-13 | CONFIRMED — freeze stands, scope description corrected | §7 v1.6.12 |
| C-14 | CLOSED by C-5/A6 — eight binding fields | §7 v1.6.5 |
| C-15 | CONFIRMED (a) + **CLOSED retrospectively (b)** on real bars | §7 v1.6.16 |
| C-16 | CONFIRMED, bounded — feeds a secondary, `harness.py` frozen | §7 v1.6.14 |
| C-17 | CONFIRMED, FIXED — incl. the shipped manifest rule string | §7 v1.6.12/13 |
| C-18 | 1 leg CLOSED, 2 CONFIRMED and now PINNED | §7 v1.6.15 |
| C-19 | CONFIRMED EXACTLY — now an ASSERTION, prose corrected | §7 v1.6.17 |
| C-20 | 3 legs: `embargo_flatness` KILLED; 1 calendar year CONFIRMED; receipt CONFIRMED stale, fixed | §7 v1.6.16/17 |
| S-1 | PROMOTED into C-1 | §7 v1.6.11 |
| S-2 | CONFIRMED, FIXED — C-2's unswept sibling | §7 v1.6.15 |
| S-3 | ADJUDICATED — non-independence now recorded in the C-1 disclosure | §7 v1.6.17 |
| S-4 | CONFIRMED, FIXED — `_betacf` rewritten, MDE bit-identical | §7 v1.6.17, `m6_s4_betacf_rewrite.json` |
