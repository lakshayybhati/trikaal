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
