# Trikaal — Drift Audit Report

**For:** Supervisor / research lead
**From:** Drift Auditor (failure-mode smoke detector — not a builder, not an architect)
**Date:** 2026-06-22
**Position audited:** M4→M6 boundary. M4b exit gate CLOSED (200/200 symbols, 304.6M bars, Merkle `5dfd667d`). M6 = NEXT, not started.
**Method:** Read the model/tokenizer/eval source directly; ran the full test suite; verified the GitHub push payload on disk; assessed one external ("low-level AI") report against ground truth.

---

## Bottom line

**No drift mode is firing.** The build is sound and the discipline is holding — in several places it is actively *tightening*, not loosening. There are **five open watches** (none blocking, all pre-M6) and **four concrete action items**, all cheap and all to be closed before any GPU-second is spent. Do not re-open any settled decision (see §6).

---

## 1. Verified-good (evidence, not claims)

Each of these I confirmed against source or by execution this session:

| Item | Evidence |
|---|---|
| **Test suite green** | `159 passed in 223s` — ran locally this session (matches the roadmap claim exactly). |
| **FSQ = canonical formulation** | `tokenizer/fsq.py`: `half·tanh(z)` → `round` → STE `z+(round−z).detach()` maps 1:1 onto Mentzer et al. `round(tanh(y)·(K−1)/2)`. Headline contribution implemented correctly; optional per-dim `scale` (residual-decay guard, §e) present. |
| **Ablation parity is in the code** | `predictor.py` hard-locks backbone to Kronos_small (8 layers / d_model 512 / d_ff 1024 / 8 heads). Fairness lever is **bits-per-token** (`FSQ_BPT_BAND = (19.5, 20.5)` around BSQ k=20), vocab is *derived* (`derive_coarse_fine`), not a forced 2²⁰. Invariant 6 respected in source. |
| **Causal-safety is a real gate** | `causal_check.py` is transform-agnostic, no "safe transform" carve-out, covers `target_valid` (survivorship surface), perturbs beyond-horizon with noise/fabricated-revert/sign-flip, samples high-risk strata. M4b promoted the in-line real-bar sweep to a **hard gate** (`FatalIngestError` on RED). |
| **Gate A has teeth and is codified** | `tests/eval/test_gate_a.py`: (1) exhaustive truncation sweep leak-free on a 420-bar Q4 sample at 100% coverage; (2) a **planted lookahead** (`with_leak("global_zscore")`) must be *caught* — explicit anti-vacuity guard. M5 close-out also documents the full harness running end-to-end on the M3 checkpoint, with a 7-agent adversarial review that found+fixed two real lookahead bugs. |
| **Headline metric is the right one** | `strategy.py` = cost-aware **net** log-return, forecast-magnitude execution filter (θ=κ·c), funding subtracted, break-even. `dsr.py` present. `ic_screen.py` = autocorrelation-deflated effective-N + MDE + moving-block bootstrap, all written from scratch. |
| **`TFI`, never `OFI`** | `constants.py` feature layout names the channel TFI throughout. |
| **Honesty in the record** | Param count disclosed as 21.3M realized vs ~27M target. M3 logged as *qualified* PASS (fine token near-noise). 304.6M bars explicitly flagged below band (see §3 OW-1). |
| **Push payload is clean** | 123 tracked files, **zero data** (no parquet/csv/duckdb/pt/ckpt), `.gitignore` root-anchored excludes `/data/ /raw/ /runs/ /wandb/ *.parquet *.pt *.npy`, no `.env`/token/key tracked, no hardcoded `ghp_`/`github_pat_` strings. `origin` already set to `github.com/lakshayybhati/trikaal`. |

---

## 2. Drift audit — all eight failure modes

| # | Failure mode | Status |
|---|---|---|
| 1 | Scope-creep / firewall breach | **Quiet** — 1 watch (OW-3: "Scale-up training" mislabel). |
| 2 | Honesty erosion | **Quiet** — honesty is being *maintained* (placebo not diluted; W1 → pre-registration). |
| 3 | Causal-safety regression | **Quiet** — strengthened at M4b (hard gate), Gate-A test has teeth. |
| 4 | Ablation-validity breach | **Quiet** — bpt parity, Kronos dims locked, placebo retained, 99% micro-availability protects Cell 5. |
| 5 | Kill-switch / gate bypass | **Quiet** — 1 watch (OW-2: §8.C.3 setup gate not yet in the M6 design). |
| 6 | Train/eval corpus confusion | **Quiet** — training-fuel confusion explicitly avoided; eval-power is the live watch (OW-1). |
| 7 | Spending ahead of verification | **Quiet** — recon-only audit, no spend; GH_TOKEN for Phase B only; spend = separate human gate. |
| 8 | Result-before-evidence | **Quiet** — research question framed as open; null is a documented valid outcome. |

---

## 3. Open watches (none blocking; all pre-M6)

- **OW-1 — Eval power (mode 6).** Realized lake is **304.6M bars / 200 symbols / 2021→2025**, thinner per-(symbol×regime) than the original ~500M–1B band. Training is fine (saturation). The risk is **cross-sectional statistical power**: `ic_screen.py`'s MDE is single-symbol/temporal (`mde_two_cell = z·√2·SE`), but the M6 claim is a cross-sectional marginal (Cell 4 − Cell 2 vs placebo Cell 5). **Pre-register the cross-sectional effective-N / MDE / placebo decision threshold before M6.** Input already exists: `_m4b_dq_table.md` (per-symbol coverage + thin-symbol flags). *Status: supervisor already converting this into a pre-registration item — keep it.*
- **OW-2 — §8.C.3 Kronos-parity setup gate (mode 5).** The binding setup gate (Cell-1/BSQ baseline must reach within ~10–15% RankIC of published Kronos-small on a common slice *before* claiming FSQ beats it) is **not named in the builder recon instruction**. Make sure it survives into the M6 design doc — it's easy to drop when unwritten.
- **OW-3 — M7/M8 mislabel (mode 1).** The summary table reads *"M7 / M8 | Scale-up training, …"*. "Scale-up" is a **firewalled word** (base-class ~100M scale is explicit v2/v3). Per the roadmap, M6 *is* the training; M7 = honesty pass + write-up; M8 = release. Confirm "scale-up training" means the M6 cloud run, and relabel so it can't read as base-class creep.
- **OW-4 — Determinism-mode recording (mode 7 / invariant 7).** Per-run recording of attention mode (deterministic fallback vs FlashAttention-2) must exist before CUDA training. Builder audit step 4 already asks for this — keep it, and require it as M6-build output if missing.
- **OW-5 — GitHub push (verification, not a drift mode).** `origin` already exists, so the supervisor's `gh repo create … --source=.` will error. Correct command: **`git push -u origin real-data-slice`**, then verify visibility is **private** (`gh repo view lakshayybhati/trikaal --json visibility`). Keep `GH_TOKEN` out of every tracked file.

---

## 4. Assessment of the external report (how to weight it)

The external report is **competent but context-blind** — it had only the summary table, not the M4b docs, so it re-derives concerns already on record and overstates one.

- **Its bar-count flag is half-wrong, half-right.** *Wrong:* "the bar count dropped and nobody named it / absorbed into a green checkmark." It was flagged explicitly — `milestone4b_universe_ingest.md:47` ("below the nominal 500M–1B band (flagged)"), the exit-gate row *"Realized bar count in band (flag if off)" → "Flagged (realized 304.6M)"* (`:187`), plus the DQ table. The "maybe a filter bug, investigate" is already answered (coverage-bound staggered listings + 41 delisted ≈ 72% fill, expected). *Right:* the eval-power consequence is real — that is OW-1.
- **Its Gate-A flag is right discipline, already satisfied.** "Validated ≠ built+unit-tested" is correct methodology, but the answer exists and is favorable (`test_gate_a.py` teeth + the documented end-to-end M3 run + adversarial review that found real bugs). Its one durable contribution: have the readiness audit **restate Gate A status with evidence**, not just "does M5 import the lake."
- **What it got right outright:** exhaustive 200-hash re-derivation (0 mismatches = adversarial standard, not a sample), clean 30× compaction with backup, recon-only/no-spend discipline, and the design→build-local→only-then-rent sequence.

**Caution for the supervisor:** the report *accuses* a surfacing failure that did not happen. Acting on that framing — treating a documented, reasoned, explicitly-flagged tradeoff as concealed — would be manufacturing doubt about a settled decision. Take its two cheap action items; do not re-open the bar count.

---

## 5. Action items (cheap, pre-M6, ordered)

1. **Amend the readiness-audit instruction with two additions:** (a) explicitly **report Gate A status** (full harness ran on M3 + `test_gate_a.py` teeth), not just M5-imports-lake; (b) **attach `_m4b_dq_table.md` per-(symbol×regime) coverage** so the cross-sectional MDE (OW-1) is computed before M6.
2. **Relabel M7/M8** in the milestone table; confirm "scale-up" ≠ base-class scale (OW-3).
3. **M6 design doc must explicitly carry:** pre-registered cross-sectional eval power + placebo decision threshold (OW-1), the §8.C.3 Kronos-parity setup gate (OW-2), determinism-mode recording (OW-4), seeds ≥ 3, bpt parity. (Supervisor owns this doc.)
4. **Push the repo** with the corrected command (OW-5); verify private; token stays out of the tree. Hold `GH_TOKEN` for Phase B only.

---

## 6. Settled — do NOT re-litigate

These are decided, disclosed, and on the record. Re-opening them is the drift this auditor guards against:

- **304.6M bars** — documented as below-band-but-adequate (eval breadth, not training fuel; 27M saturates ~1–2B). The *only* live item is OW-1's eval-power pre-registration, which needs no re-counting.
- **FSQ formulation** — canonical-correct; not a place to tinker.
- **Backbone dims** — locked to Kronos_small for ablation cleanliness; do not "improve."
- **One headline claim** — the microstructure-aware FSQ tokenizer. No second claim.

---

*Auditor's note: the correct outcome of this audit is "no drift, four cheap closures." The build earned a clean bill; the work now is M6 design rigor, not architecture.*
