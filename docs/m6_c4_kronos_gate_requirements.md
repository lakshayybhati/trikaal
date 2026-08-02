# C-4 — What implementing G-§8.C.3 actually requires

**AUDIT TIER-2. REQUIREMENTS ONLY — NOT IMPLEMENTED.** This enumerates what the gate needs and
names what is unresolved. It writes no code and decides nothing.

---

## 1. The finding, confirmed

`grep -rn "from_pretrained\|hf_hub\|huggingface_hub" src/ scripts/` returns **zero hits**.
G-§8.C.3 is a **binding gate with no implementation**. It is named as binding in three places:

- `docs/m6_design.md:51` — *"fails iff RankIC < 0.85 × published Kronos-small on the pinned common
  slice"*, with a pre-committed halt-before-any-Δ / Cell-1-only-fix / full-5-cell-same-seed-retrain
  failure protocol;
- `docs/m6_design.md:94` — *"Verify G-§8.C.3 (Kronos parity) as the **first cloud check**"*;
- `docs/ROADMAP.md:58` — an **entry gate** for M6: *"else the ablation is blocked; we do not claim
  FSQ beats a crippled baseline"*.

It is also invariant 4's external-validation leg, and the **only** place Kronos weights are
permitted to appear anywhere in the project.

## 2. What the gate is, precisely

Two separable pieces, and they were separated deliberately in the ROADMAP:

| | where | what it validates |
|---|---|---|
| **steps 1–2, 4** | M5 | our **metric implementations** (IC / RankIC / MAE / R²) against Kronos's published numbers, by running published weights through our harness on a common BTC slice |
| **step 3** | M6 | our **Cell-1 BSQ baseline**, universe-trained, must reach ≥ **0.85 × published Kronos-small RankIC** on the pinned common slice, *fed Kronos's own input pipeline* |

Step 3 is the entry gate. Steps 1–2/4 are a code cross-check, and the ROADMAP is explicit that
step 3 *cannot* live at M5 because "you cannot validate a baseline that has not been trained yet".

## 3. What implementing it requires

### 3.1 Weights
- Obtain published **Kronos-small** weights. This is a **HuggingFace pull to a rented box**, which
  puts it squarely under the standing credential rule: *never ship a write-capable or
  non-fine-grained credential to rented third-party infrastructure*. A read-only, fine-grained,
  repo-scoped token, or an unauthenticated pull if the repo permits one.
- **Licence and redistribution must be checked before the pull**, not after. Invariant 8 permits
  Kronos weights in exactly one place — the eval harness as an external validation target — and
  they must never enter the model or any checkpoint we publish.
- Content-hash the downloaded weights and record the hash, exactly as every other input is
  hashed. A gate that depends on an unpinned external artifact is not reproducible.

### 3.2 Preprocessing — the largest unresolved item
The gate says Cell 1 is fed **Kronos's own input pipeline**. That is not our pipeline. Concretely
this needs, and none of it exists today:
- Kronos's normalization and feature construction for OHLCV, which differs from our per-bar
  feature spec (invariant: our features are a 16-dim vector with its own causal rules);
- its tokenizer/quantizer front end, if the published inference path requires it;
- its context length and stride conventions;
- **a causal-safety review of the imported path.** Invariant 2 is a hard invariant enforced by a
  CI test on *our* pipeline. An imported preprocessing path is not covered by that test, and it
  must not be able to introduce lookahead into a number we then compare against. This is the
  single most dangerous part of the work.

### 3.3 The published number
- The comparison target is *published Kronos-small RankIC*. Which reported figure, on which
  instrument, over which period, is **not pinned anywhere I can find** — `docs/m6_prereg.md` §6
  pins the slice and the transcription timing, but the specific published value has to be
  transcribed from the paper (arXiv:2508.02739) and content-pinned before any comparison, or the
  gate's threshold is unfalsifiable.
- **Transcription timing is already pinned** in the prereg precisely so the number cannot be
  chosen after seeing ours. That pin must be honoured when the value is entered.

### 3.4 The comparison
- Pinned common slice (already pinned in prereg §6).
- Our RankIC on Cell 1 vs `0.85 × published`.
- **The failure protocol is pre-committed and is not a re-run licence**: halt before any Δ, fix
  Cell 1 only, then a full 5-cell same-seed retrain. The earlier "materially off / fix" wording
  was superseded exactly because it was unlimited.

## 4. Sequencing problem worth naming

The design says G-§8.C.3 is the **first cloud check**, but step 3 requires a **universe-trained
Cell 1** — which only exists *after* the training leg of the money run. So "first cloud check"
cannot mean "before training"; it must mean "first check after Cell 1 exists and before anything
is claimed". If the gate fails, the pre-committed protocol requires a **full 5-cell same-seed
retrain**, i.e. the entire training budget again. **That contingency is not in any cost estimate
I have produced**, and it should be, because it roughly doubles the worst-case spend.

## 4a. THE RETRAIN CONTINGENCY — costed, with its trigger, pre-registered

Supervisor-ordered: *"Name it, cost it, and pre-register it as a contingency with its trigger."*

**TRIGGER (already pre-committed in prereg §6, restated here so it is findable):** Cell 1,
universe-trained, scores **RankIC < 0.85 × published Kronos-small RankIC** on the pinned common
slice. On that trigger the protocol is: **halt before any Δ is computed → fix Cell 1 ONLY → full
5-cell same-seed retrain.** It is not a re-run licence; the earlier "materially off / fix" wording
was superseded precisely because it was unlimited.

**COST.** The contingency is a **second full training leg** — 5 cells × 5 seeds = 25 units. Eval
need not repeat for cells whose checkpoints are unchanged, but under the protocol *all five* cells
retrain at the same seeds, so in practice **both legs repeat**.

| | |
|---|---|
| nominal whole-run estimate on record | **$33–50** (S=5), **$43–65** with forced determinism |
| retrain contingency | **+1× the training leg**, i.e. **roughly doubling worst-case spend** |
| worst case if it fires | **~$66–100**, or **~$86–130** under forced determinism |

**EVERY FIGURE HERE IS ESTIMATED, NOT MEASURED**, and inherits the §7 v1.6 re-labelling of the
$20–30 / $33–50 / $43–65 / "2–3 GPU-days" / 1.3× family — see the claims-audit appendix §2. The
contingency multiplier (×2) is arithmetic on the protocol, not a measurement.

**WHY IT IS NAMED RATHER THAN ABSORBED.** A doubling discovered at the moment it fires is a crisis;
a doubling written down in advance is a budget line. It has appeared in **no** cost estimate the
builder has produced to date, which is the defect this entry closes.

**WHAT WOULD RETIRE IT.** The trigger cannot fire until Cell 1 is universe-trained and compared,
so the contingency is live from the start of the training leg until G-§8.C.3 passes. It retires on
the first pass, not before.

## 5. Cost and risk, unestimated on purpose

I have not costed this. The unknowns that dominate — whether Kronos's inference path can be
reproduced faithfully, and how much work the preprocessing bridge is — are not things I can bound
from here without reading the parent paper's released code, which is itself a decision (invariant
8 permits their weights in the harness; it does not obviously permit vendoring their preprocessing
code, and that distinction needs a ruling).

## 6. Open questions that need a ruling, not a builder decision

1. **Does invariant 8 permit importing Kronos's preprocessing code**, or must it be reimplemented
   from their paper? "No Kronos code is ever part of Trikaal" reads as prohibiting the import; but
   "fed Kronos's own input pipeline" reads as requiring it. These two are in tension and the
   tension is in the spec, not in my reading of it.
2. **Which published RankIC number**, from which table, is the target — and is it transcribed and
   content-pinned yet?
3. **Is the full-retrain contingency budgeted?** If G-§8.C.3 fails, the pre-committed protocol
   costs a second full training run.
4. **Does the gate block the run, or block the claim?** ROADMAP calls it an M6 *entry* gate, but
   step 3 needs a trained Cell 1, so it cannot gate entry in the ordinary sense.

## 7. Not done, deliberately

No weights pulled. No code written. No cost quoted. No sequencing changed. Per the ruling: report
what implementing it requires; do not implement.

---

# UPDATE 2026-08-03 (§7 v1.6.18) — LICENCE, ACCESS, COST, AND THE BLOCKER NOBODY HAD NAMED

**Status change: the gate is now IMPLEMENTED and BLOCKED, where before it was ABSENT.**
`src/trikaal/eval/external_validation.py` + `tests/eval/test_external_validation.py`. It fires
inside `assemble_verdict` **before any between-cell Δ exists**, `money_verdict` defaults to `True`
so the dangerous direction is the one that must be declared (the C-6 rule), and it returns
`BLOCKED` — a HALT — because the published reference has not been obtained. Proven capable of
halting a money verdict and of clearing when a reference is supplied.

## A. The licence — read, not summarized

**HuggingFace model repos are PUBLIC and UNGATED** (`gated: False`, `private: False`,
`license: mit`) for `NeoQuasar/Kronos-small`, `Kronos-base`, and `Kronos-Tokenizer-base`.

The model repos carry **no LICENSE file** (`/raw/main/LICENSE` → HTTP 404); the licence is declared
in the model-card frontmatter (`license: mit`). The full text lives in the code repo,
`https://raw.githubusercontent.com/shiyu-coder/Kronos/master/LICENSE`, quoted verbatim:

> MIT License
>
> Copyright (c) 2025 ShiYu
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND …

**Reading.** MIT permits use, modification and even redistribution, conditioned only on carrying
the copyright and permission notice. Running the weights through our harness for a metric
cross-check is unambiguously permitted. **The licence is therefore NOT the constraint — invariant 8
is**, and it is stricter than the licence: we forbid redistribution of these weights in anything we
publish regardless of MIT allowing it. One honest caveat: the weights repo asserts MIT via card
metadata only, so the *weights'* licence rests on the card, not on a licence file shipped beside
them.

## B. Access — demonstrated, not assumed

An **unauthenticated** request to `https://huggingface.co/api/models/NeoQuasar/Kronos-small`
returns **HTTP 200** with `gated: False, private: False`. A `HEAD` on the weights resolves with
`x-linked-size: 98,980,656` (94.4 MiB) and the tokenizer at `15,842,368` (15.1 MiB).
**NO TOKEN IS REQUIRED and none should be issued.** *(Metadata and text only were fetched; no
weights were transferred — the hard stop on pulling was respected.)*

**So there is nothing for Lakshay to do on credentials.** If a future need arises the scope would
be a fine-grained, read-only, repo-scoped token — but on today's evidence it is unnecessary, and
issuing one would add risk for no capability.

## C. Cost — and why the rental question does not arise

- Pull: **109.5 MiB total**, unauthenticated, over HTTP. **$0, local.**
- Inference: Kronos-small is `d_model 512, n_layers 8, n_heads 8, ff_dim 1024` (its `config.json`)
  — the **same class as our own backbone** — over a BTC slice. Minutes on CPU. **$0, local.**
- **No rental is needed for steps 1–2 and 4.** The standing rule that local timings carry ~8×
  variance and must be re-measured on target hardware does not bite here, because nothing is being
  costed against a rental: there is no rental.

## ★ D. THE BLOCKER: STEPS 1–2 AND 4 CANNOT BE RUN WITHOUT KRONOS CODE

This is not a scheduling failure and it is not fixed by pulling weights.

The published weights are a **bare `state_dict`**. The model card's own loading instructions are:

```python
from model import Kronos, KronosTokenizer, KronosPredictor
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```

**Running the weights requires Kronos's `model.py`.** And CLAUDE.md invariant 8 says:

> No Kronos code or weights are ever part of Trikaal … its public weights appear in exactly one
> place: the eval harness, as an external validation target.

**Weights are permitted in the eval harness. Code is permitted nowhere.** The prereg additionally
requires the comparison be *"fed Kronos's own input pipeline"*, which is more Kronos code. So the
gate as specified is **unexecutable under invariant 8 as written**. The options, none of which is
the builder's to choose:

1. **Vendor Kronos's `model.py` into a quarantined, eval-only, never-published path.** Licence-wise
   fine (MIT + attribution). Requires an invariant-8 amendment — Lakshay's file, like B1.
2. **Reimplement the architecture from the paper + `config.json` to load their `state_dict`.**
   Arguably still "Kronos code", and a subtly-wrong reimplementation produces a *wrong* comparison
   to a published number — worse than no comparison, because it would look authoritative.
3. **Compare against the published NUMBERS without running their model.** Cheapest, but the prereg
   forbids exactly this: published figures are on different data, which is why "on the same bars"
   is in the specification.
4. **Re-specify or drop the gate** — a ROADMAP change, also not ours.

## E. Will Cell 1 miss 0.85 × published Kronos-small RankIC?

**The one hard number, from our own pins:**

| quantity | value |
|---|---|
| Stage-2 budget | 2,000 steps × batch 32 × seq 512 = **32,768,000 tokens** |
| backbone params | 21,301,248 |
| **tokens per parameter** | **1.54** (Chinchilla-optimal ≈ 20) |
| **fraction of compute-optimal** | **7.69 %** |
| epochs over the 304.6M-bar lake | **0.108** |

**Three forces, in opposite directions — which is why a single confident number would be false:**

1. **AGAINST US, and it dominates:** at 7.7 % of compute-optimal we are on the steeply-rising part
   of the loss/skill curve, not the plateau. Reaching 85 % of a well-trained model's RankIC from
   7.7 % of its compute is optimistic.
2. **FOR US, and it is not small:** Kronos-small is a broad multi-market model; Cell 1 is trained
   on 200 crypto symbols at 1-minute and evaluated on a crypto slice. **Domain specialization can
   be worth a large multiple** on in-domain rank correlation, and it is exactly the axis the gate
   does not control for.
3. **NEITHER — the gate may not be RESOLVABLE.** At a published RankIC of 0.02–0.03 (the range M2
   found for *features*; the model figure is not yet identified), the 15 % band is **0.0030–0.0045**.
   `SE(ρ) ≈ 1/√n_eff` needs **n_eff ≈ 49,000–111,000** for the band to equal one standard error.
   The primary region gives **35,064 stride-15 periods per symbol**; the 40-symbol cross-section is
   correlated, so the honest denominator is the autocorrelation-deflated effective N the MDE
   machinery already computes — and it is **not obviously large enough**. A gate whose band sits
   inside its own sampling error is not a gate.

**My answer: I will not give a calibrated probability, and force 3 is why.** If I had to bracket
force 1 alone I would say the budget term makes a miss *more likely than not*; force 2 could
plausibly reverse it; force 3 may make the comparison indecisive regardless of both. **Any single
number I gave you here would be a guess dressed as an estimate.**

**What would make it knowable, in increasing cost:**

- **(i) $0, today** — identify *which* published figure and on which slice. The comparison target
  is still unnamed; without it there is no threshold, which is why the gate returns `BLOCKED`.
- **(ii) $0, today** — compute the deflated effective N on the pinned slice with the existing
  `ic_screen` machinery and check whether a 15 % band is resolvable **at all**. If it is not, the
  probability question is moot and the gate needs re-specification before anything is spent.
- **(iii) a few dollars** — a **scaling probe**: train Cell 1 at 2–3 reduced budgets and read the
  RankIC trend. That converts force 1 from an argument into a measurement, and it is the only
  cheap way to see whether 2,000 steps is anywhere near the knee.

**The consequence the supervisor asked me to state plainly: if the gate is likely to fail, the
retrain contingency is not a contingency — it is the expected path, and its cost (~$66–100, or
~$86–130 forced-deterministic) is then part of the base budget, not a tail risk.** On force 1 alone
that is the reasonable planning assumption. **Lakshay should be told this before he funds the run,
not after it halts** — and (ii) should be run first, because a gate that cannot resolve its own band
changes the question entirely.

---

# UPDATE 2026-08-03b (§7 v1.6.19) — RESOLVABILITY (ii): THE BAND IS FINE. THE GATE IS ILL-POSED.

Ruling 2 ordered resolvability settled before anything touches an invariant. It is settled, and
the answer is not the one I expected — **my own "force 3" worry was wrong, and something worse sits
underneath it.**

## (a) WHICH published figure? THERE ARE TWO, AND THEY DIFFER BY 2.4×

From the paper (arXiv:2508.02739), **Table 2**, `Kronos_small`:

| task | IC | **RankIC** |
|---|---|---|
| Price Series Forecasting | 0.0431 | **0.0254** |
| Return Forecasting | 0.0665 | **0.0622** |
| Volatility Forecasting | MAE 0.0384 | R² 0.2490 |

**The gate says "0.85 × published Kronos-small RankIC". That is 0.0216 or 0.0529 depending on
which row you pick — and the prereg does not say.** The threshold is not under-specified at the
margin; **it is a factor of 2.4 apart**, which is larger than the 15 % band the gate is made of.

## (b) AND BOTH ARE ON A DIFFERENT ASSET CLASS, FREQUENCY AND TASK

Table 2's caption: *"the dataset of **Shanghai Stock Exchange, 15-minute frequency**"*.

- **Asset class:** Chinese A-share equities. Ours is crypto perpetual futures.
- **Bar frequency:** 15-minute bars. Ours is **1-minute** bars (our `h=15` is fifteen 1-minute
  bars, which is not the same object).
- **Task:** "price series" / "return" forecasting as Kronos defines them, not our forward
  log-return at stride h under a cost-aware execution filter.

**This makes the gate as written internally contradictory.** The prereg requires *both*:
(1) the comparator be the **published** RankIC, and (2) the comparison be run **"on the pinned
common slice … fed Kronos's own input pipeline"**. Those cannot both hold: run Kronos on our
crypto slice and you produce a **new** number, not the published one; use the published number and
you are comparing a crypto-1m model against an equities-15m figure.

**AND IT KILLS STEPS 1–2/4 AS A METRIC CROSS-CHECK, INDEPENDENTLY OF EVERYTHING ELSE.** The stated
purpose was to *"validate OUR METRIC IMPLEMENTATIONS against an external reference"* by reproducing
Kronos's published numbers. You cannot reproduce a Shanghai-Stock-Exchange number on a BTC slice.
Doing it properly would need **SSE 15-minute equity bars**, which we do not have and which
CLAUDE.md firewalls out of v1 (*"Equities / cross-asset data"*, out of scope). **The external
metric check, as specified, is not available to us at any price.**

## (c) THE BAND ITSELF IS RESOLVABLE — I WAS WRONG ABOUT THIS

Measured with the project's own `ic_screen.effective_sample_size` on the **pinned 40 symbols** over
the **pinned primary region**, non-overlapping stride-15 forward returns:

| quantity | value |
|---|---|
| raw stride-15 periods (summed over 40 symbols) | 1,402,560 |
| autocorrelation-deflated N_eff (summed) | **1,401,637** (ratio 0.999) |
| SE(RankIC) = 1/√N_eff | **0.00084** |

| published figure | RankIC | 0.85× threshold | 15 % band | band / SE | verdict |
|---|---|---|---|---|---|
| Table 2 price-series | 0.0254 | 0.02159 | 0.00381 | **4.51** | RESOLVABLE |
| Table 2 return-forecast | 0.0622 | 0.05287 | 0.00933 | **11.05** | RESOLVABLE |

**Caveat, stated because it cuts against the conclusion:** summing per-symbol N_eff assumes
cross-sectional independence, which crypto badly violates. This is an **upper bound** on N_eff.
If the 40 symbols behave like ~3–5 independent factors, SE rises ~2.8–3.6× and the band becomes
**1.3–1.6 SE** for the price-series figure (marginal) and **3.1–3.9 SE** for the return figure
(comfortable). Autocorrelation is *not* the binding deflation here — the ratio is 0.999, consistent
with the signed-return ACF of ~0.007 at lag 60 measured in §7 v1.6.16.

**So: sampling noise is not the obstacle I speculated it was.** I flagged it as possibly decisive
and it is not, except marginally at the smaller of the two figures.

## (d) CONCLUSION — SAME DESTINATION AS RULING 2, DIFFERENT ROUTE

Ruling 2 said: *"If it is not resolvable, the gate as specified is not a gate and that conclusion
goes to Lakshay ahead of any implementation work."*

**It IS resolvable, and the gate as specified is still not a gate** — because its threshold is
ambiguous by 2.4×, its reference is measured on an asset class we have firewalled out of v1, and
its two requirements (published number / common slice) cannot both be satisfied.

**No further C-4 implementation work should proceed until the gate is re-specified**, and that is a
prereg/ROADMAP change — Lakshay's, alongside the invariant-8 question. The candidate
re-specifications, for whoever rules:

1. **Run Kronos-small on OUR crypto slice and compare Cell 1 against THAT** (not the published
   figure). Well-posed, needs the invariant-8 ruling, and — usefully — it becomes an
   apples-to-apples baseline instead of a cross-market one.
2. **Keep the published figure but state which row and accept the domain gap as a disclosed
   limitation.** Cheap, and weak: a crypto-vs-equities RankIC ratio is not a baseline check.
3. **Drop it as a binding gate, keep it as a reported diagnostic.** A ROADMAP change.
4. **Replace it with an internal sanity floor** (e.g. Cell 1 must beat a documented naive
   benchmark on our own slice). Loses the external anchor, which was the whole point.
