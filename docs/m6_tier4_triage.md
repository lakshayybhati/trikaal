# AUDIT TIER-4 — triage (2026-08-02, §7 v1.6.14). VERIFY ONLY; nothing fixed.

Per finding: **CONFIRMED with my own reproduction**, **REFUTED with evidence**, **CLOSED by earlier
work**, or **NEEDS-RULING**. No fix is proposed in this document and no code was changed by it.

Receipts: `runs_manifest/m6_tier4_vacuous_gates.json` (C-10, C-15),
`runs_manifest/m6_c3_dsr_units.json` (C-3, ruled separately).

---

## The headline of the tier: two findings are the C-2 pattern, not documentation defects

C-2 was ruled **Tier 1, blocking** — *a gate that reports a verdict while examining nothing*.
**C-10 and C-15 are the same defect in different files**, and in both the docstring asserts the
opposite of the behaviour, so neither is visible from reading the fix. Both were confirmed by
construction, each with a control arm.

**I am not re-tiering them — that is the supervisor's call. I am reporting that the tier looks
wrong.**

---

## C-10 — the micro-legibility hard stop passes when it measures nothing · **CONFIRMED**

`src/trikaal/train/gates.py`, `micro_legibility_gate`.

`ok = True` is the initial value. A dim with `< 10_000` unmasked bars hits `continue` **before**
`ok = ok and acc >= min_acc`. So if all six micro dims are thin, the loop skips all six, `ok` is
never touched, no `RuntimeError` is raised, and the function returns `{"pass": True}`.

**Reproduced:** a 12,000-bar window with every micro dim masked to 500 unmasked bars returns
`pass: True`, `raised: False`, with **6 of 6 micro dims carrying `"skipped"`**. The gate that is
described as a *hard stop before any Stage-2 spend* passed having measured nothing.

The docstring says such a dim is *"SKIPPED with a named receipt entry, **never trivially
passed**"*. The receipt entry is real. The second half is not — nothing distinguishes
"all dims skipped" from "all dims passed" in the return value or the control flow.

**Second leg, also confirmed by reading:** `id_legibility_sign_acc` takes `values[:n]`, `b_c[:n]`
with `n = 150_000` — the **head** of a per-symbol concatenation, and the 80/20 split is contiguous
(`[:n_tr]` / `[n_tr:]`). With many symbols concatenated in symbol order, the sample can cover only
the first symbols, and train/val are adjacent time regions of that head rather than a
representative split.

**Not fixed.** `MICRO_LEGIBILITY_MIN` is pinned (`PINNED_MICRO_LEGIBILITY_MIN = 0.9`); the
vacuous-pass behaviour is not a pinned value, so a fix would be in-scope under the freeze — but the
tiering question comes first.

---

## C-15 — the causal gate runs only at non-production parameters · **CONFIRMED, and sharper than stated**

Invariant 2's required CI gate (`tests/data/test_causal_safety.py`, `@pytest.mark.gate`) runs
against `causal_cfg` = `synthetic_test_config(half_life_fast=48, half_life_slow=96, n_warm=12,
w_tau=20, vol_window=24, n_warm_vol=12)` on a 600-bar synthetic stream. Production is
`FeatureConfig()`: `half_life_fast=1440`, `half_life_slow=5760`, `rolling_window=1440`, `w_tau=1440`.

**The audit's "target_valid vacuous under the production config" is confirmed by measurement:**

| config | `effective_n_warm_vol` | `target_valid` True | planted validity leak flipped it |
|---|---|---|---|
| CI gate (reduced) | 48 | **375 / 569** | **14 / 20 bars** |
| production `FeatureConfig()` | **1440** | **0 / 569** | **0 / 20 bars** |

Control arm: the reduced config **must** show the leak firing, or the probe would be measuring a
broken plant rather than a gate. It fires 14/20. The separation is real.

**So the obvious remedy is wrong.** Re-running the CI gate at `FeatureConfig()` would not make it
stronger — it would make it **vacuous**, because the 1440-bar volatility warm-up exceeds the
fixture and no bar has a valid target for a leak to flip. A production-parameter causal gate needs
a fixture longer than the warm-up, which is a **test-cost decision**, not a config swap. Not
proposed here.

**Mitigation, on the record:** `scripts/m4b_universe_ingest.py:292` *does* run
`exhaustive_truncation_sweep(sl, FeatureConfig())` at production parameters on real lake shards,
gated hard on `coverage == 1.0 and n_checks > 0` before any write. **But that guard proves the
sweep ran, not that each output surface was non-degenerate** — `n_checks` counts all outputs, so a
vacuous `target_valid` surface clears it. Same shape as C-10, one level down.

---

## C-16 — the modelled impact term is a hardcoded constant · **CONFIRMED, lands on a secondary**

`src/trikaal/eval/harness.py:83` passes `q_over_adv=1e-3` as a literal into
`costs.total_cost(...)`. It is not pinned, not configurable, and not receipted.

**What limits it:** the headline is the **flat 0.30% netting** (`ir_headline`); the modelled-cost
IR is the named **secondary** (`ir_modeled_cost`). So the constant does not enter the headline, the
clause thresholds, or ΔIR(4−5). It does enter a reported diagnostic, unpinned.

`harness.py` is frozen and load-bearing on the money path via `xsection.py:36` (C-13), so this is
**report-only** regardless of how it is ruled.

---

## C-19 — BSQ and FSQ cells do not realize the same parameter count · **CONFIRMED EXACTLY**

Measured:

| arm | (V_c, V_f) | realized backbone params | vs the quoted 21,301,248 |
|---|---|---|---|
| FSQ (cells 2, 4, 5) | (891, 1225) | **21,301,248** | +0 |
| BSQ (cells 1, 3) | (1024, 1024) | **21,231,616** | **−69,632 (−0.327%)** |

The auditor's 21,231,616 is exact.

**The code already knows.** `cells.build_cell_backbone` documents it and asserts a **±2% band** for
non-FSQ vocabs instead of exact equality, because embedding/head rows differ with vocab size
(2048 vs 2116 rows).

**So the defect is in the prose, and it is not cosmetic.** `CLAUDE.md` and the design doc quote
*"21,301,248 realized params"* as one number for the matched backbone, and invariant 4 requires the
backbone *"held fixed and matched across every ablation arm"*. It is matched to ±0.327%, not
exactly. **Where this bites:** the §5 NULL-fallback claim is `IR(2) − IR(1)` — FSQ vs BSQ — so that
claim is confounded with a 69,632-parameter difference in the same direction as the arm it
compares. The primary ΔIR(4−5) is **unaffected** (both cells are FSQ, identical vocab).

**NEEDS-RULING** on the prose and on whether the fallback claim carries a stated caveat.

---

## C-20 — three legs, mixed · **PARTLY CONFIRMED**

1. **`embargo_flatness` is never called. CONFIRMED.** `src/trikaal/eval/folds.py:88` defines it;
   the only callers anywhere are `tests/eval/test_folds.py`. A purge/embargo leak diagnostic that
   exists, is tested, and never runs on a real fold plan.
2. **Headline is one calendar year.** Not independently verified in this pass — the window is
   assembled per-arm in `run/matrix.py` and bound into every artifact, so it is checkable, but I
   did not measure it and will not report a verdict I have not reproduced.
3. **Claims-sweep receipt stale.** Not verified in this pass.

---

## C-18 — three legs, two open · **PARTLY CONFIRMED, one CLOSED**

1. **Interpreter unpinned — CLOSED.** `utils/provenance.py:68` records
   `platform.python_version()` per unit, added under C-5/A7 precisely because `uv.lock` pins
   packages and not the interpreter (`requires-python ">=3.11"` admitted the 3.14 a rented box
   silently ran).
2. **Stage-1/2 step budget is a bare code default — CONFIRMED.** `OrchestratorConfig.steps_stage1 =
   2000`, `steps_stage2 = 2000` are plain dataclass defaults. The money driver *records* them in
   its manifest, but **recording is not pinning**: no `PINNED_*` key exists and no gate asserts
   them, so a different budget would run and be faithfully reported as the recipe.
3. **The μ estimator is a function default — CONFIRMED, and this is the sharpest of the three.**
   `predict_mu(..., estimator: str = "expectation")` is the v1.4.2 pin, but it exists **only** as a
   Python default. There is no `PINNED_MU_ESTIMATOR`, and `grep` across `conformance.py`,
   `matrix.py` and `orchestrator.py` finds no assertion on it. A caller passing `estimator="argmax"`
   would silently restore the pre-fix biased decode — the exact defect v1.4.2 was written to
   remove — and no gate would notice. **This is the C-7 shape** (a value that decides the verdict,
   under no guard).

---

## C-11 — determinism hardcoded off with no config knob · **CONFIRMED, and already open as B1**

Not a new finding. `orchestrator.py:159` and `m6_canary.py` pass
`set_determinism(..., deterministic_algorithms=False)`; 49/49 records assert `bit_exact_claim: true`
beside it. Fully written up in `docs/invariant7_amendment_decision.md`, **awaiting Lakshay's A/B
ruling and the staged CUDA probe (~$1–2)**. The audit independently rediscovering it is
corroboration, not new work.

---

## C-14 — eval resume does not bind artifacts to their checkpoints · **CLOSED by earlier work**

Closed under C-5/A6. `src/trikaal/run/matrix.py:201-211` compares `meta["checkpoints"]` against the
checkpoint hashes now on disk and refuses the artifact on mismatch, alongside symbols, window, h,
`n_periods` and `start_ms` — eight binding fields, one KAT each. A rerun after retraining can no
longer silently mix training generations.

---

## S-1..S-4 — **I CANNOT ANSWER THIS WITHOUT THE ORIGINAL TEXT**

The audit asked whether the four SUSPECTED items were promoted, refuted, or silently dropped. I can
answer for one:

- **S-1 was PROMOTED**, into C-1 (the single-bar decode gap). It recorded that *the structure is
  certain and the magnitude is not*; the magnitude was then measured under the full M6 pin set,
  came back **INDETERMINATE at n=3** (t = 1.821, Welch df 2.385, crit 2.6226), and the channel is
  now a **REQUIRED non-gating per-(cell, seed) disclosure on the real cells**.

**S-2, S-3 and S-4: I do not have their text.** Rather than guess at findings and report a verdict
on my reconstruction of them, I am asking for the four items as written. Answering "were any
silently dropped?" from a paraphrase is exactly the failure the question is designed to catch.

---

## Summary

| finding | verdict |
|---|---|
| C-3 | CONFIRMED, outcome-material — **amendment drafted and HELD** (`docs/m6_c3_amendment_decision.md`) |
| C-10 | **CONFIRMED** — passes having measured nothing; C-2 pattern; tier looks wrong |
| C-11 | CONFIRMED — is B1, already with Lakshay |
| C-14 | **CLOSED** by C-5/A6 |
| C-15 | **CONFIRMED** — and the obvious remedy makes it vacuous, not stronger |
| C-16 | CONFIRMED — unpinned constant, lands on a secondary, frozen file |
| C-18 | 1 CLOSED, 2 CONFIRMED — the μ-estimator leg is the C-7 shape |
| C-19 | **CONFIRMED EXACTLY** — −69,632 params; confounds the §5 fallback, not the primary |
| C-20 | 1 of 3 CONFIRMED (`embargo_flatness` never called); 2 not verified this pass |
| S-1 | PROMOTED into C-1 |
| S-2..S-4 | **BLOCKED — original text needed** |
