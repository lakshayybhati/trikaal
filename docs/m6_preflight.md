# M6 Pre-Flight Gate — Zero First-Time Events Before the 15-GPU-Day Run

**Author:** Supervisor · **Date:** 2026-06-23 (Item 8 added 2026-07-06) · **Status:** OPEN — all **8** items must be GREEN (with the stated evidence committed) before any **full-budget** spend.

## The principle (non-negotiable)
The 15-GPU-day M6 run is **one-shot and expensive**. It must contain **zero first-time events** — everything it will do must have already happened once, **small, on the real hardware, and been watched**. Anything that happens for the first time during the real run is what kills it at hour 200. This gate leaves nothing unrehearsed.

## Two framings that govern this gate
1. **"Everything must work" means the MACHINERY, not the OUTCOME.** The run can be a complete success and the answer can be *"microstructure does not beat the placebo after costs."* That is a **publishable win, not a failed run.** The expense makes us rigorous about the instrument and **genuinely indifferent to which way the result breaks.** One-shot cost pressure must not bleed into needing the answer to be "yes" — that would be the result-before-evidence drift mode (and invariant 3).
2. **Pre-flight de-risks the KNOWABLE; checkpointing contains the UNKNOWABLE.** Even with all 8 green, something can still surprise at scale (a CUDA-specific numeric issue, an untested edge). That residual is handled by **Item 3**: if it breaks on day 6, we resume from day 5, not zero. The toy run kills the known failure modes; checkpoint discipline contains the rest. Together = the maximum certainty available before spend.

## Spend tiers (read this)
- **Phase A prep:** $0 (local).
- **Pre-flight toy run (Items 2–5):** a SHORT real-CUDA rental (~hours, ~$5–20) — this *is* the rehearsal; a small, gated spend, NOT the full budget.
- **The 15-day full run:** the big spend (~$150–450), authorized ONLY when all 8 are GREEN.

## Green-light rule
The 15-day run is authorized **only** when all **8** items are confirmed GREEN with the evidence committed to the repo. Each closes on **testable evidence — an artifact, a hash, a measured number, a passing check — not "done."**

---

## Item 1 — Step 0: the instrument is honest NOW, under a pinned env  *(foundation)*
*(Revised 2026-07-04: the original Item 1 targeted `results_hash == 91ed9054` under `numpy==2.3.3/torch==2.11.0`. Before any lockfile was committed the env drifted AGAIN — the venv and on-disk lockfile now resolve numpy 2.4.6 / torch 2.12.1, and the venv lost the `data` extras entirely (3 test-collection errors). The full 2026-07-04 audit also found instrument defects (m6_design §6 item 10: headline-cost mislabel, flat-period annualization inflation, time-misaligned PBO matrix) that must be fixed before any anchor is meaningful. So Item 1 is now **procedure-anchored, not hash-anchored** — chasing a historical hash of a defective-labeled instrument under a gone env proves nothing; the KATs + causal gates carry logic-integrity across the transition.)*
**Closes on:**
- A **committed** lockfile (tracked — `.gitignore`'s `uv.lock` line removed) + `==` pins in pyproject for numpy/torch, with the **`data` + `dev` extras installed** so the FULL suite collects with zero errors (today `tests/data/test_manifest.py`, `test_reduce_shard.py`, `test_universe.py` — including the multi-symbol causal-sweep gate — cannot even import).
- The **m6_design §6-item-10 instrument fixes landed, each with a KAT**, before the anchor run.
- `scripts/m5_eval_harness.py` on the M3 checkpoint under the committed env, **run ≥2× → bit-identical `results_hash` with zero variation**; that hash becomes the NEW Gate-A anchor (superseding `75caa9ef` and `91ed9054`, both retired as history).
- The anchor recorded in a **committed, durable run-manifest artifact** (results_hash + numpy/torch/python versions + device + seeds + input content-hashes) — never stdout-only again.
- `pytest tests/eval` → the full eval KAT suite green (currently 45 tests + the new item-10 KATs); both Gate-A causal tests green (Q4 sweep leak-free @100% cov + planted-lookahead caught); full suite green + ruff clean.
- The **six M6 governing docs committed** (m6_design, m6_preflight, the audits, the directive) — a pre-registration with no git timestamp is not pre-registered; and `scripts/setup_cloud.sh` changed to **install from the committed lockfile** (today it installs latest torch + floors, reproducing the drift failure mode on the rented box).
- **Artifact:** the lockfile commit + the run-manifest commit showing the same hash twice. *(The env has now drifted TWICE — this item is what makes a third time impossible to miss.)*

## Item 2 — Full 5-cell toy run, end-to-end, on the REAL CUDA box  ★ CENTERPIECE ★
The entire 5-cell arc runs start-to-finish, once, small, on the **actual rented GPU type** — no crash, no config gap. Numbers are garbage by design; the deliverable is "it all ran."
**Closes on — ONE continuous run on the real CUDA GPU (NOT MPS/CPU), toy scale (3–5 symbols incl. ≥1 thin coin from the 43, tiny token budget, a few hundred steps/cell), producing ALL of:**
- `nvidia-smi` in the log proving the real CUDA GPU type; the per-run **determinism mode recorded** (inv. 7).
- **All 5 cells trained without crashing → 5 checkpoints written and reloadable** (5 checkpoint hashes logged).
- **The multi-symbol loader fed all 5 cells incl. ≥1 thin coin** — per-symbol sample counts logged, the thin coin present with its sampling weight applied.
- **The eval harness scored all 5 toy models** — 5 net-IR numbers via the cross-sectional driver over the toy universe.
- **The Cell-5 placebo computed** — ΔIR(Cell4−Cell5) produced; the shuffled-micro arm verifiably trained on *permuted* micro (not real).
- **W&B logged throughout** — one run URL with all 5 cells' loss curves + the eval metrics.
- **The two-arm CANARY (added 2026-07-06 — Item 2 proves the machine RUNS; this proves it MEASURES):**
  (a) **planted-signal arm:** a synthetic toy lake where one micro channel carries a known causal
  relationship to the target → the 5-cell toy must yield ΔIR_info > 0 **and** IR(5) ≈ IR(2)
  (the placebo tracks the OHLCV-only counterfactual when no exploitable info exists in the
  shuffle); (b) **pure-noise arm:** micro = noise → ΔIR_info must NOT fire and |IR(5) − IR(2)|
  must be small vs the toy spread (measured placebo neutrality — the direct check on the
  "placebo is a victim" failure mode). Plus: assert **non-degenerate per-dim reconstruction
  contribution of micro dims 7–12 through Cell 4's tokenizer** (kills the silent-micro-suppression
  false-NULL) and record the **attention-mode decision** (prereg §3a: the one headline mode is
  fixed here, before any real cell).
- **Artifact:** the single continuous run log + 5 checkpoint hashes + the eval verdict + both
  canary verdicts + the W&B URL, content-hashed. If any stage needed a manual fix, that fix lands
  and the toy run is re-run clean — the gate is "ran start-to-finish unattended."

## Item 3 — Checkpoint-and-resume, actually tested (kill it on purpose)
Prove resume by **rehearsing the disaster**, not assuming it.
**Closes on:**
- Train one toy cell **uninterrupted** to step 2N → reference (weights hash + loss at 2N).
- Train the same cell, **SIGKILL at step N**, restart from its last checkpoint, continue to 2N.
- **Assert the resumed run matches the uninterrupted reference at 2N** — same loss + weights hash (bit-identical under the determinism mode, or within the recorded tolerance). A match proves optimizer state, LR-schedule position, RNG state, data-loader position, and step counter were ALL restored (any miss diverges).
- **Artifact:** a committed kill-resume comparison (both trajectories + the 2N match); ideally a committed test.

## Item 4 — G3 throughput on the real hardware with the real lake
The run fits the budget AND the GPU isn't starved.
**Closes on:**
- Measured **end-to-end bars/sec** (compacted lake → loader → batch → GPU) on the real GPU type. *(Distinct from the local 248M/s lake-scan — this is into-the-GPU throughput.)*
- (a) **Arithmetic:** total tokens (5 cells × ≥3 seeds × budget) ÷ measured throughput ≤ ~15 GPU-days — show the math.
- (b) **GPU utilization** during the toy run (nvidia-smi) is high (e.g. >80%) — the loader is not the bottleneck. Low util → fix the feed before spending.
- **Re-plan trigger:** if measured throughput is materially worse than assumed, re-plan (fewer seeds / smaller budget / faster GPU) BEFORE the full run, revised arithmetic committed.

## Item 5 — Disk/storage math, checked once
The box never fills mid-run, and teardown never forces a re-ingest.
**Closes on:**
- **Footprint arithmetic:** 5 cells × ≥3 seeds × (checkpoint + optimizer state + logs + eval outputs) < allocated disk, with headroom — show per-checkpoint size × count.
- **Object-storage snapshot tested:** the lake (15 GB) + a sample checkpoint pushed to object storage AND **restored** successfully (a teardown can never force a re-download of the 52.8h-built lake). Artifact: the snapshot location + a verified round-trip restore.

## Item 6 — Pre-registration locked and DATED before the run
The decision rule committed before any real number exists (anti-p-hack).
**Closes on:**
- The **cross-sectional MDE** committed: **pooled AND per-regime** (2021/2022/2023/2024), **correlation-deflated** (cross-symbol ρ estimated from residuals), α=0.05 / power=0.8 — the actual numbers.
- The **placebo decision threshold** committed: micro survives iff ΔIR(Cell4−Cell5) > 0, CI lower bound > 0, AND ΔIR ≥ MDE_pooled.
- The **κ\* selection rule** committed: κ\* selected over the FULL cross-section; the headline is the **cost-stress curve across costs/κ, NOT a single κ\* point** (per the env-drift incident that flipped κ\* 2.0→3.0 on a knife-edge).
- **Artifact:** a committed `docs/m6_prereg.md` whose **git commit timestamp precedes the first cell's training run** — the timestamp is the proof it was locked before results.

## Item 7 — Garbage-detector tripwire + early-stop rule
An obviously-broken run halts early, not after 15 days of nonsense.
**Closes on:**
- Committed "broken" definitions: (i) a cell's training loss not dropping over the first K steps (flat/rising beyond a threshold); (ii) NaN/Inf in loss, grads, or eval; (iii) net-IR at an absurd magnitude (|IR| beyond a sane bound) or degenerate eval; (iv) GPU-util collapse / a stalled cell.
- A **wired stop-rule:** the orchestrator checks the tripwires each cell/checkpoint and **aborts the run** on a trip.
- **Artifact:** the committed thresholds + a **tested halt** — inject a NaN / a flat-loss cell → confirm the monitor stops the run.

## Item 8 — The analysis that decides the outcome exists, is calibrated, and cannot drift from the lock  *(added 2026-07-06)*
Item 6 locks the prose; Item 8 locks the MATH. The paired bootstrap must exist and be calibrated
BEFORE any real number it could judge; the money-run config must be machine-checked against the
pre-registered surface (a lock is worthless if the computed quantity can drift from the locked one).
**Closes on:**
- The **paired moving-block bootstrap** (prereg §3/§3a recipe: Δr_p resampling, ⌈√T⌉ blocks,
  B=10,000, seed 20260704, percentile CI, SE_boot) implemented in `src/trikaal/eval/` — and
  **calibration-KAT'd on synthetic paired series:** null Δ → rejection rate ≈ α (0.05); planted
  Δ = MDE_paired → power ≈ 0.80 (both within Monte-Carlo tolerance, tolerances stated in the KAT).
- A **conformance script** that asserts the money-run analysis config **equals the pre-registered
  surface**: window + train_frac vs `runs_manifest/m6_mde_inputs.json`, primary region = forward
  blocks 1–5, the hashed 40-symbol list (`60e24f598de96012…`), κ grid {1.0,1.5,2.0,3.0}, seeds
  {0,1,2}, **no `cap_per_symbol` dev bound**, per-symbol spread deciles (not flat "major"), and
  seed-threading of `shuffle_micro` at eval — **fails loudly on ANY diff**. Runs as a hard gate at
  the start of the real eval.
- The §2 MDE table **recomputed on the pinned blocks-1–5 region** (per §3a) and committed.
- **Artifact:** the calibration-KAT results + the conformance script + one committed passing run
  of it against the real run-config.

---

## Sequence
- **Foundation:** Item 1 (Step 0 under the pin) — nothing proceeds until green.
- **Locked-before-rehearsal:** Items 6 (prereg incl. v1.2) + 8 (the analysis math + conformance) + 7 (tripwire) — the instrument and its lock are finished BEFORE the rehearsal exercises them.
- **The real-CUDA rehearsal (centerpiece):** Items 2 (incl. the canary) → 3 → 4 → 5 — all proven in/around ONE short toy run on the actual GPU type. This is where "zero first-time events" is earned.
- **GREEN-LIGHT:** all **8** GREEN with evidence committed → the 15-GPU-day run is authorized. Not before.
