# V1 LIVE DEMO — SCOPE, BEFORE ANY BUILD

**Status: SCOPE ONLY. Nothing built. Awaiting supervisor ruling.**
Builder, 2026-08-14. $0 spent producing this; the measurements below are local and CPU-only.

The artifact promise is **code + weights + paper + a live demo**. The demo moved onto the critical
path when the economics work was deferred. This document says what a minimal honest demo is, what
it must not imply, how the honesty is *enforced structurally rather than annotated*, and my hours.

---

## 0. THE ONE-SENTENCE RULE THIS WHOLE DOCUMENT SERVES

> **The demo shows a FORECAST. It never shows a P&L, and it must not be possible to mistake it for
> one — including from a cropped screenshot with every caption removed.**

Our own measurement says these models lose money after fees: net IR **−28.47 / −67.02 / −146.49**
across the pre-registered cost band, break-even cost **0.0023–0.0208 %** against a **~0.10 %**
round-trip taker fee (`runs_manifest/m6_horizon_break_even.json`). A demo that implies otherwise is
the context-stripping rule at its worst, and it would be OUR OWN artifact contradicting OUR OWN
headline result.

---

## 1. WHAT WE ACTUALLY HOLD (verified, not assumed)

Three complete cell-1 units in `runs_cloud/rescue/r{0,1,2}/cell1_bsq_ohlcv_seed{0,2,4}/`:

| file | size | needed for inference? |
|---|---|---|
| `tokenizer.pt` | 17 MB | **yes** |
| `predictor.pt` | 121 MB | **yes** |
| `stage1_state.pt` | 52 MB | no — optimizer state |
| `stage2_state.pt` | 363 MB | no — optimizer state |
| `run_manifest.json` | 8 KB | yes (identity + provenance) |

**Shippable inference weight per seed = 138 MB; 414 MB for all three.** The 415 MB of optimizer
state per seed is training-resume material and does not belong in a demo image.

### Two measurements that de-risk the build

1. **NO FROZEN-STATS FILE EXISTS OR IS NEEDED.** `src/trikaal/data/universe_loader.py:17-20`: the
   lake's features are *"causally self-normalized per symbol at build time (streaming EWMA
   z-scores — bars ≤ t only); there is NO train-fitted statistic anywhere in this path."* The demo
   therefore needs no statistics artifact, and it inherits causal safety from the same code path
   the eval used rather than re-implementing it. This was the single largest unknown and it is
   closed favourably.

2. **CPU IS FAST ENOUGH — NO GPU, NO PAID TIER.** Measured locally on 4 torch threads with the real
   `predictor.pt`: **34 ms** per 512-bar forward pass ⇒ **~0.5 s** for a naive 15-step rollout per
   seed, **~1.5 s for all three seeds together**, without a KV cache. A free CPU Space is
   sufficient. (Measured on Apple silicon; a Space's shared vCPU will be slower — budget a
   2–4× margin and it is still interactive.)

### One thing found while measuring, which is NOT a demo issue but is a paper issue

The shipped `predictor.pt` realizes **31,725,568 parameters**. The figure quoted throughout the
repo and docs/ENGINEERING.md — **21,231,616** (BSQ arm) — is exactly `TOTAL − MTP`:

```
blocks       16,785,408   52.91%
mtp          10,493,952   33.08%   <-- excluded from the quoted figure
token_embed   1,572,864    4.96%
cross         1,049,088    3.31%
temporal        775,168    2.44%
w_c / w_f     1,048,576    3.30%
norm_final          512    0.00%
TOTAL        31,725,568
TOTAL - MTP  21,231,616   == the quoted number, to the parameter
```

The quoted count **excludes the MTP heads, which are 33.1 % of the shipped model**. MTP is not an
optional extra — docs/ENGINEERING.md lists it inside the same `trikaal/model/` component whose size that
number labels. **The moment we publish weights, any reader can load the checkpoint and get 31.7 M.**
Whether the quoted figure is redefined, or restated as "backbone excluding MTP heads", is the
writer's and supervisor's call — I am reporting the decomposition, not ruling on it. Flagged here
because the demo and the HF weight release are what make it publicly checkable.

---

## 2. THE MINIMAL HONEST DEMO

**Input.** A symbol and a timestamp from our own lake (a fixed dropdown of the 40 evaluated
symbols and a date slider). Not a live exchange feed, not user-uploaded data.

**Computation.** The existing production path, unchanged: lake bars → per-bar feature vector →
`tokenizer.pt` → `predictor.pt` → `predict_mu(..., estimator="expectation")` → μ̂, the expected
cumulative log-return over the next *h* minutes.

**Output — one chart.**
- The last 512 one-minute bars of realized price (the model's actual context window).
- **Three forecast points at h = 15, one per seed (0, 2, 4), plotted together and never averaged.**
- The realized next-15-minute path, shown *after* the forecast, so the viewer sees what happened.
- An uncertainty band from the available MC trajectory sampling (`estimator="mc_mean"`).

**Why three seeds is the design and not a detail.** Showing one seed's forecast implies a
determinacy we measured and do not have: identical data and configuration produced headline IRs
spanning **118.03**, and activity spanning **24.4×**. The seed spread *is* one of our findings.
Plotting all three makes the demo agree with the paper instead of quietly contradicting it.

---

## 3. WHAT THE DEMO MUST NOT IMPLY — and how each is enforced STRUCTURALLY

The context-stripping rule is explicit that a caveat living *outside* the thing it qualifies is not
protection. Each item below therefore names an enforcement that survives a crop.

| # | Must NOT imply | Structural enforcement (not a caption) |
|---|---|---|
| 1 | **Profit / tradeability** | The app computes no P&L, no equity curve, no cumulative return, no position, no Sharpe. Those quantities are never calculated, so they cannot be rendered or exported. |
| 2 | **That the forecast is usable after costs** | The measured break-even (0.0023–0.0208 %) and the ~0.10 % fee are rendered **inside the plot area**, as axis-anchored text in the same raster as the curve. A crop that removes them removes the chart. |
| 3 | **A single deterministic prediction** | Three seeds always plotted; no "average" line exists in the code. A one-seed view is not reachable through the UI. |
| 4 | **That the 2×2 ablation ran** | The title rendered in-image names the arm: *"cell 1 — BSQ, OHLCV-only"*. Cell 1 is the **baseline**, not the contribution. The microstructure claim was never tested; the legibility gate refused cells 4 and 5 before Stage-2. |
| 5 | **A foundation model** | In-image subtitle: *tokenizer study, 31.7 M-parameter measurement vehicle*. Never "foundation model", never a scale claim. |
| 6 | **Live / real-time capability** | Historical lake bars only. No exchange API is imported, so there is no code path to a live quote. Dates are bounded by the lake. |
| 7 | **Financial advice** | No symbol recommendation, no ranking of symbols by forecast, no "best" anything. |
| 8 | **That we hold micro-arm results** | No cell 2/3/4/5 selector exists. The only weights in the image are cell 1. |

**The strongest enforcement is #1 and it is architectural:** if the P&L is never computed, no
screenshot, no crop, no export, and no future contributor's "small addition" can surface one
without a visible code change.

---

## 4. THE ACCEPTANCE GATE I WOULD HOLD MYSELF TO

A demo that renders a forecast is not evidence the demo is *correct*. Per the execution rule
("has never been executed is not an operational state") and the mock rule:

> **The demo must reproduce a banked μ̂ exactly.** Pick decisions from the money run, run them
> through the demo's own inference path, and assert the μ̂ matches the value the scored artifact was
> built from, bit-for-bit under forced determinism. If it does not match, the demo is showing a
> different model from the one the paper reports, and it ships only when it matches.

This is cheap ($0, local) and it is the difference between "the demo runs" and "the demo shows our
result". I would also mutation-test it: perturb a weight and confirm the gate FAILS.

---

## 5. HOURS

Labor only; **$0 of compute** — everything runs on CPU on hardware we already have, and hosting is
a free CPU Space.

| work | hours |
|---|---|
| Single-symbol inference entry point (load tokenizer + predictor, run `predict_mu`) | 2–3 |
| Lake window selection + feature assembly for an arbitrary timestamp | 1–2 |
| **Reproduce-a-banked-μ̂ acceptance gate + its mutation test** | 2–3 |
| Chart rendering with all disclosures baked into the raster | 3–4 |
| App shell + free CPU Space deployment | 2–3 |
| End-to-end verification pass, cold-start timing, crop test on the rendered image | 1–2 |
| **Total** | **11–17 h** |

Realistically **two to three working sessions**. The two things that could move it: a Space's shared
vCPU being slower than my 34 ms measurement (mitigated by the 2–4× margin, and by pre-tokenizing
the lake windows we serve), and the 414 MB weight payload if we ship all three seeds — which we
should, because three seeds is the honest design.

---

## 6. WHAT I NEED RULED BEFORE I BUILD

1. **Three seeds plotted together — confirm.** It is my recommendation and it makes the demo agree
   with the paper, but it is a presentation decision about how our own instability is shown.
2. **Weights on HuggingFace: all three seeds, inference-only (138 MB each)?** Or one seed? Shipping
   optimizer state (415 MB/seed) has no demo purpose.
3. **The parameter-count decomposition in §1** — the writer's call, but it should be settled before
   weights are public, because publishing the checkpoint is what makes 31.7 M checkable.
4. **Does the demo state the headline result in-image, or only decline to contradict it?** My
   recommendation is to state it: a forecast demo whose own caption says the strategy loses money
   after fees is a stronger honesty signal than silence, and it is the finding we are publishing.
